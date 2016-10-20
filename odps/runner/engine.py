# encoding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function

import logging
import threading
import warnings
from threading import local

from .. import ODPS, log
from ..compat import six
from ..config import options
from ..errors import ODPSError
from ..ui import create_instance_group, reload_instance_status

from .enums import PortType
from .utils import is_temp_table

logger = logging.getLogger(__name__)
engines = dict()


class ODPSEventMixIn(object):
    @property
    def after_run_sql(self):
        if not hasattr(self, '_local'):
            return None
        if not hasattr(self._local, 'after_run_sql'):
            return None
        return self._local.after_run_sql

    @after_run_sql.setter
    def after_run_sql(self, value):
        if not hasattr(self, '_local'):
            self._local = local()
        self._local.after_run_sql = value

    def run_sql(self, sql, project=None, priority=None, running_cluster=None, **kwargs):
        instance = super(ODPSEventMixIn, self).run_sql(sql, project=project, priority=priority,
                                                       running_cluster=running_cluster, **kwargs)
        if self.after_run_sql:
            self.after_run_sql(instance=instance, sql=sql)
        return instance


def node_engine(engine):
    def _decorator(cls):
        engines[engine] = cls
        return cls

    return _decorator


def create_engine(node, context, runner):
    cls = engines[node.engine]
    return cls(node, context, runner)

_DIRECT_RAISE_CODES = [
    'ODPS-0420061',  # XFlow not exists
    'ODPS-0130211',  # table already exists
]


class BaseNodeEngine(object):
    def __init__(self, node, odps, runner):
        self._node = node
        self._runner = runner
        self._odps = odps
        self._odps.__class__ = type('EventODPS', (ODPSEventMixIn, ODPS), {})
        self._dry_run = options.runner.dry_run
        self._temp_lifecycle = options.temp_lifecycle
        self._global_lifecycle = options.lifecycle
        self._gen_params = None
        self._lifecycle = None
        self._instances = []
        self._progress_group = create_instance_group(node.code_name)
        self._last_cmd = None

        self._output_tables = [p.obj.table for p in six.itervalues(self._node.outputs)
                               if p.obj is not None and p.type == PortType.DATA]
        self._temp_tables = [tt for tt in self._output_tables if is_temp_table(tt)]
        self._persist_tables = [tt for tt in self._output_tables if not is_temp_table(tt)]

        if self.is_lifecycle_needed():
            self._lifecycle = self._temp_lifecycle if not self._persist_tables else self._global_lifecycle
        else:
            self._lifecycle = None

    def is_lifecycle_needed(self):
        if 'withLifecycle' in self._node.metas:
            return bool(self._node.metas['withLifecycle'])
        return any((p.type != PortType.MODEL for p in six.itervalues(self._node.outputs)))

    def execute(self):
        log('Executing {0}...'.format(self._node.code_name))
        self.before_exec()
        if not self._node.executed:
            if self._dry_run:
                self.mock_exec()
            else:
                for trial in range(options.runner.retry_times):
                    try:
                        self.actual_exec()
                        break
                    except Exception as ex:
                        if any(ecode in str(ex) for ecode in _DIRECT_RAISE_CODES) \
                                or trial >= options.runner.retry_times - 1:
                            raise
                        else:
                            warnings.warn('Attempt {0} for {1} failed. Message: {2}'
                                          .format(trial + 1, self._node.code_name, str(ex)))
                            self._runner._delete_managed_objects(self._node, with_outputs=True)
        self.after_exec()

    def before_exec(self):
        from ..runner import RunnerContext
        from .runner import get_retry_mode
        context = RunnerContext.instance()
        self._node_hash = self._node.calc_node_hash(self._odps)
        if get_retry_mode():
            if self._node_hash in context._node_outputs:
                old_outputs = context._node_outputs[self._node_hash]
                for port_name, out_port in six.iteritems(self._node.outputs):
                    if out_port.obj_uuid in context._obj_container:
                        context._obj_container[out_port.obj_uuid].fill(old_outputs[port_name])
                self._node.executed = True

        self._gen_params = self._node.gen_command_params()
        if not self._node.executed and not self._dry_run:
            self._node.before_exec(self._odps, self._gen_params)

    def actual_exec(self):
        raise NotImplementedError

    def mock_exec(self):
        for case in self._node.cases:
            case(self._node, self._gen_params)

    @staticmethod
    def get_output_object(port):
        if port.obj is not None:
            return port.obj.describe()
        else:
            return None

    def after_exec(self):
        from ..runner import RunnerContext
        context = RunnerContext.instance()

        if self._node.reload_on_finish:
            self._reload_output()

        self._node.after_exec(self._odps, True)

        context._node_outputs[self._node_hash] = dict((pn, self.get_output_object(p))
                                                      for pn, p in six.iteritems(self._node.outputs))
        self._set_table_lifecycle(self._temp_tables, self._temp_lifecycle)
        if self._global_lifecycle:
            self._set_table_lifecycle(self._persist_tables, self._global_lifecycle)

    def _set_table_lifecycle(self, table_name, lifecycle, async=True, use_threads=True, wait=False):
        def _setter(tables):
            if isinstance(tables, six.string_types):
                tables = [tables, ]
            for table in tables:
                if not self._odps.exist_table(table_name):
                    return
                instance = self._odps.run_sql('alter table %s set lifecycle %s' % (table, str(lifecycle)))
                if not async:
                    instance.wait_for_success()

        if use_threads:
            th = threading.Thread(target=_setter, args=(table_name, ))
            th.start()
            if wait:
                th.join()
        else:
            _setter(table_name)

    def _reload_output(self):
        if options.runner.dry_run:
            return
        for ep in (ep for ep in six.itervalues(self._node.outputs) if ep.obj_uuid is not None):
            obj = ep.obj
            if obj is not None:
                obj.reload()

    def add_instance(self, inst, show_log_view=True):
        self._instances.append(inst)
        if options.verbose:
            log('Instance ID: ' + inst.id)
            if show_log_view:
                log('  Log view: ' + inst.get_logview_address())

    def refresh_progress(self):
        for inst in self._instances:
            reload_instance_status(self._odps, self._progress_group, inst.id)

    def show_error(self):
        # new-line symbol should not be removed.
        if self._last_cmd:
            logger.error('\nLast executed command: ' + self._last_cmd)

    def stop(self):
        for inst in self._instances:
            try:
                inst.stop()
            except ODPSError:
                logger.warn('Failed to stop instance id=%s.' % inst.id)

    @staticmethod
    def log_instance(inst):
        log('Instance ID: ' + inst.id)
        log('  Log view: ' + inst.get_logview_address())
