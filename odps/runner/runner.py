# encoding: utf-8
# Copyright 1999-2017 Alibaba Group Holding Ltd.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import gc
import itertools
import logging
import operator
import sys
import time
from copy import deepcopy
from threading import Thread, Lock, Event
from collections import Iterable

from ..compat import reduce, six, futures, raise_exc
from ..config import options
from ..console import in_ipython_frontend, ProgressBar
from ..errors import ODPSError
from ..tempobj import register_temp_table, register_temp_model
from ..ui import ProgressGroupUI, html_notify

from .core import InputOutputNode, RunnerPort
from .enums import PortType
from .engine import create_engine
from .ui import show_ml_retry_button, register_retry_magic
from .user_tb import tb_patched
from .utils import is_temp_table, is_temp_model

logger = logging.getLogger(__name__)

_pre_running_hooks = []
_post_running_hooks = []
_retry_mode = False


def set_retry_mode(value):
    global _retry_mode
    _retry_mode = value


def get_retry_mode():
    global _retry_mode
    return _retry_mode


def register_hook(typ, fun):
    if typ == 'pre_exec':
        _pre_running_hooks.append(fun)
    elif typ == 'post_exec':
        _post_running_hooks.append(fun)
    else:
        raise ValueError


class ProgressVisualizer(object):
    def __init__(self):
        self._bar = None
        self._progress_group = None
        try:
            if in_ipython_frontend():
                self._progress_group = ProgressGroupUI(True)
                self._bar = ProgressBar(1, True)
        except Exception as ex:
            logger.debug(ex)

    def progress(self, val):
        if self._bar:
            self._bar.update(val)

    def status(self, prefix, suffix=''):
        if self._progress_group:
            self._progress_group.prefix = prefix
            self._progress_group.suffix = suffix

    def add_keys(self, keys):
        if self._progress_group:
            self._progress_group.add_keys(keys)

    def remove_keys(self, keys):
        if self._progress_group:
            self._progress_group.remove_keys(keys)

    def update(self):
        if self._progress_group:
            self._progress_group.update()

    @staticmethod
    def show_retry():
        if in_ipython_frontend():
            show_ml_retry_button()

    @staticmethod
    def notify(msg):
        if in_ipython_frontend():
            html_notify(msg)

    def close(self):
        if self._bar:
            self._bar.close()
        if self._progress_group:
            self._progress_group.close()


class Runner(object):
    def __init__(self, odps):
        self._odps = odps
        self._managed_tables = dict()
        self._managed_models = dict()

        self._engines = dict()

        self._visualizer = None

        self._dry_run = options.runner.dry_run

    @tb_patched
    def run(self, target_nodes=None):
        """
        :type target_nodes: BaseDagNode | list[BaseDagNode]
        """
        from .context import RunnerContext
        # call gc before running to eliminate unneeded data sets and models
        gc.collect()

        for hook in _pre_running_hooks:
            hook()

        self._visualizer = ProgressVisualizer()
        self._visualizer.status('Preparing environment...')

        dag = RunnerContext.instance()._dag
        for node in dag.nodes():
            node.scheduled = False

        self._managed_tables = dict()
        self._managed_parts = dict()
        self._managed_models = dict()

        nodes = dag.topological_sort()
        self._fill_table_names(nodes)

        if target_nodes:
            if not isinstance(target_nodes, Iterable):
                target_nodes = set([target_nodes, ])
            restrict = set(dag.ancestors(target_nodes)) | set(target_nodes)
            nodes = [n for n in nodes if n in restrict]

        nodes = self._optimize_nodes(n for n in nodes if not n.executed)
        if not nodes:
            return
        for node in nodes:
            node.scheduled = True

        if not self._dry_run:
            self._delete_managed_objects()

        parallel_num = int(options.runner.parallel_num)

        finish_event = Event()
        actual_exec_nodes = [n for n in nodes if not n.virtual and not n.executed and n.code_name is not None]
        to_be_executed = set(actual_exec_nodes)
        exec_batch = set()
        executed = set()
        exception_sink = []
        state_lock = Lock()

        def exec_thread_proc(node):
            engine = self._engines[node]
            try:
                engine.execute()
            except ODPSError as ex:
                engine.show_error()
                exception_sink.append((type(ex), ex, node.traceback, True))
            except:
                exception_sink.append(sys.exc_info() + (False, ))
            finally:
                state_lock.acquire()
                exec_batch.remove(node)
                executed.add(node)
                state_lock.release()

                finish_event.set()

        node_idx = 0
        threads = []
        while len(executed) < len(to_be_executed):
            new_tasks = []
            while node_idx < len(actual_exec_nodes):
                try:
                    node = actual_exec_nodes[node_idx]
                    state_lock.acquire()
                    exec_batch_size = len(exec_batch)
                    if exec_batch_size >= parallel_num:
                        break

                    input_nodes = (set(edge.from_node for edge_set in six.itervalues(node.input_edges)
                                       for edge in edge_set) - executed) & to_be_executed
                    if not input_nodes:
                        exec_batch.add(node)
                        new_tasks.append(node)
                        node_idx += 1
                    else:
                        break
                finally:
                    state_lock.release()

            last_exec_batch = list(exec_batch)
            # load accurate fields for input data sets
            self._refresh_node_inputs(new_tasks)

            new_engines = dict((n, create_engine(n, self._odps, self)) for n in new_tasks)
            self._engines.update(new_engines)
            self._visualizer.status('Executing:')
            self._visualizer.add_keys([e._progress_group for e in six.itervalues(new_engines)])

            local_threads = [Thread(name='PyODPSExecutionThread%d' % idx, target=exec_thread_proc, args=(node,))
                             for idx, node in enumerate(new_tasks)]
            if local_threads:
                [th.start() for th in local_threads]
                threads.extend(local_threads)

            try:
                counter = 0
                while not finish_event.isSet():
                    time.sleep(0.1)
                    counter += 1
                    if counter >= 50:
                        counter = 0
                        for ex_node in last_exec_batch:
                            self._engines[ex_node].refresh_progress()
                        self._visualizer.update()
                for ex_node in last_exec_batch:
                    self._engines[ex_node].refresh_progress()
            except KeyboardInterrupt:
                for engine in six.itervalues(self._engines):
                    engine.stop()
                self._visualizer.status('Halted by interruption.')
                raise
            finish_event.clear()

            state_lock.acquire()
            self._visualizer.progress(len(executed) * 1.0 / len(to_be_executed))
            self._visualizer.remove_keys([self._engines[n]._progress_group for n in executed])
            state_lock.release()

            if exception_sink:
                for engine in six.itervalues(self._engines):
                    engine.stop()
                break
        [th.join() for th in threads]

        if in_ipython_frontend():
            self._visualizer.show_retry()
            if exception_sink:
                self._visualizer.notify('Execution failed.')
            else:
                self._visualizer.notify('Execution succeeded.')
        self._visualizer.close()

        for hook in _post_running_hooks:
            hook()

        if not exception_sink:
            return True
        else:
            ex_type, ex_value, tb, wrapped = exception_sink[0]
            if not wrapped:
                raise_exc(ex_type, ex_value, tb)
            else:
                ex_value._exc_printed = True
                sys.excepthook(ex_type, ex_value, tb)
                raise ex_value
            return False

    def _delete_managed_objects(self, node=None, with_outputs=False):
        if node is None:
            managed_tables = reduce(operator.add, six.itervalues(self._managed_tables), [])
            managed_models = reduce(operator.add, six.itervalues(self._managed_models), [])
        else:
            managed_tables = self._managed_tables.get(id(node), [])
            managed_models = self._managed_models.get(id(node), [])

        if not with_outputs:
            managed_tables = list(filter(is_temp_table, managed_tables))
            managed_models = list(filter(is_temp_model, managed_models))

        register_temp_table(self._odps, filter(is_temp_table, managed_tables))
        register_temp_model(self._odps, filter(is_temp_model, managed_models))

        executor = futures.ThreadPoolExecutor(100)
        del_table_iter = executor.map(lambda tn: self._odps.delete_table(tn, if_exists=True), managed_tables)
        del_model_iter = executor.map(lambda mn: self._odps.delete_offline_model(mn, if_exists=True), managed_models)

        del_threads = itertools.chain(del_table_iter, del_model_iter)
        list(del_threads)

    def _optimize_nodes(self, nodes):
        nodes = list(nodes)
        new_nodes = []
        for n in nodes:
            optimized, new_output_desc = n.optimize()
            if not optimized:
                new_nodes.append(n)
            elif new_output_desc is not None:
                src_id = new_output_desc.node_id
                self._managed_tables[src_id].extend(six.itervalues(new_output_desc.tables))
                self._managed_models[src_id].extend(six.itervalues(new_output_desc.offline_models))

        return new_nodes

    def _refresh_node_inputs(self, nodes):
        [self._fix_upstream_ports(ep) for node in nodes for ep in six.itervalues(node.inputs)
         if ep.type == PortType.DATA and ep.obj_uuid is not None]

    def _fix_upstream_ports(self, data_obj):
        if isinstance(data_obj, RunnerPort):
            data_obj = data_obj.obj
        if data_obj is None or data_obj._fields_fixed:
            return
        [self._fix_upstream_ports(ul) for ul in data_obj._uplink]
        source_sets = data_obj._uplink
        if not data_obj._operations:
            if source_sets:
                data_obj._fields = deepcopy(source_sets[0]._fields)
        else:
            for op in data_obj._operations:
                op.execute(source_sets, data_obj)
                source_sets = [data_obj, ]
        if not data_obj._bind_node.reload_on_finish:
            data_obj._fields_fixed = True

    def _fill_table_names(self, nodes):
        """
        :type nodes: list[BaseDagNode]
        """
        for node in nodes:
            if id(node) not in self._managed_tables:
                self._managed_tables[id(node)] = []
            if id(node) not in self._managed_models:
                self._managed_models[id(node)] = []

        # first pass: table/model names for sources and dests
        for node in (n for n in nodes if isinstance(n, InputOutputNode)):
            objs = node.propagate_names()
            if not objs:
                continue
            self._managed_tables[id(node)].extend(six.itervalues(objs.tables))
            self._managed_models[id(node)].extend(six.itervalues(objs.offline_models))

        # second pass: assign table/model names for other nodes
        for node in (n for n in nodes if not isinstance(n, InputOutputNode)):
            for output in six.itervalues(node.outputs):
                output_obj = output.obj
                if output_obj is None:
                    continue
                objs = output_obj.gen_temp_names()
                if not objs:
                    continue
                self._managed_tables[id(node)].extend(six.itervalues(objs.tables))
                self._managed_models[id(node)].extend(six.itervalues(objs.offline_models))


if register_retry_magic is not None:
    register_retry_magic()
