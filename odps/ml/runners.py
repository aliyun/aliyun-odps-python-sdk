# encoding: utf-8
# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import time
import logging

from ..config import options  ## don't remove
from ..compat import six, Enum  ## don't remove
from ..utils import write_log
from ..ui import reload_instance_status, fetch_instance_group

logger = logging.getLogger(__name__)
runners = dict()


class RunnerType(Enum):
    MOCK = 'MOCK'
    LAMBDA = 'LAMBDA'
    XFLOW = 'XFLOW'
    XLIB = 'XLIB'
    SQL = 'SQL'
    DF = 'DF'
    PS = 'PS'


def node_runner(engine):
    def _decorator(cls):
        runners[engine] = cls
        return cls

    return _decorator


def create_node_runner(engine, algo_name, params, metas, engine_kw, ui, **kw):
    runner_type = RunnerType(metas.get('engine', 'XFLOW').upper())
    runner = runners[runner_type]
    return runner(engine, algo_name, params, metas, engine_kw, ui, **kw)


class BaseNodeRunner(object):
    def __init__(self, engine, algo_name, params, metas, engine_kw, ui, progress_proportion=1,
                 priority=None, group=None, **kw):
        self._engine = engine
        self._algo_name = algo_name
        self._gen_params = params
        self._instances = []
        self._progress_group = group
        self._metas = metas
        self._engine_kw = engine_kw
        self._ui = ui
        self._progress_portion = progress_proportion
        self._last_cmd = None
        self._cases = kw.get('_cases', [])
        self._output_models_only = kw.get('_output_models_only')

    def execute(self):
        write_log('Executing {0}...'.format(self._algo_name))
        self.before_exec()
        if options.ml.dry_run:
            self.mock_exec()
        else:
            self.actual_exec()

            while not all(inst.is_terminated(retry=True) for inst in self._instances):
                self.refresh_progress()
                time.sleep(1)
            [inst.wait_for_success() for inst in self._instances]
            self._ui.inc(self._progress_portion)
        self.after_exec()

    def before_exec(self):
        pass

    def actual_exec(self):
        raise NotImplementedError

    def mock_exec(self):
        if not hasattr(self, '_cases') or self._cases is None:
            return
        for case in self._cases:
            case(self, self._gen_params)

    def after_exec(self):
        pass

    def add_instance(self, inst, show_log_view=True):
        self._engine._instances.append(inst)
        self._instances.append(inst)
        if options.verbose:
            write_log('Instance ID: ' + inst.id)
            if show_log_view:
                write_log('  Log view: ' + inst.get_logview_address())
        self._ui.status('Executing', 'execution details')

    def refresh_progress(self):
        for inst in self._instances:
            reload_instance_status(self._engine._odps, self._progress_group, inst.id)
        self._ui.update_group()

    @classmethod
    def _format_value(cls, value):
        if value is None:
            return None
        elif isinstance(value, bool):
            if value:
                return 'true'
            else:
                return 'false'
        elif isinstance(value, (list, set)):
            if not value:
                return None
            return ','.join([cls._format_value(v) for v in value])
        else:
            return str(value)


@node_runner(RunnerType.XFLOW)
class XFlowNodeRunner(BaseNodeRunner):
    def __init__(self, *args, **kwargs):
        super(XFlowNodeRunner, self).__init__(*args, **kwargs)
        self._sub_instance_set = set()

    @staticmethod
    def _is_param_valid(val):
        if val is None:
            return False
        if isinstance(val, (list, set, tuple)) and len(val) == 0:
            return False
        return True

    def before_exec(self):
        super(XFlowNodeRunner, self).before_exec()
        if 'core_num' in self._engine_kw:
            self._gen_params['coreNum'] = self._engine_kw['core_num']
        if 'core_mem' in self._engine_kw:
            self._gen_params['memSizePerCore'] = self._engine_kw['core_mem']
        if 'lifecycle' in self._engine_kw and not options.ml.dry_run:
            self._gen_params['lifecycle'] = self._engine_kw['lifecycle']

    def actual_exec(self):
        pai_project = self._metas.get('xflowProjectName', options.ml.xflow_project)
        xflow_name = self._metas.get('xflowName', self._algo_name)

        params = dict([(k, self._format_value(v)) for k, v in six.iteritems(self._gen_params)
                       if self._is_param_valid(v)])

        if self._output_models_only:
            params.pop('lifecycle', None)

        param_args = ' '.join(['-D%s="%s"' % (k, v) for k, v in six.iteritems(params)
                               if self._is_param_valid(v)])
        self._last_cmd = 'PAI -name %s -project %s %s;' % (xflow_name, pai_project, param_args)
        write_log('Command: ' + self._last_cmd)

        inst = self._engine._odps.run_xflow(xflow_name, pai_project, params)
        self.add_instance(inst, show_log_view=False)

    def refresh_progress(self):
        insts = self._instances
        if insts:
            inst = insts[0]
            group_json = fetch_instance_group(self._progress_group)
            group_json.logview = inst.get_logview_address()
        for xflow_inst in insts:
            for inst_name, sub_inst in six.iteritems(self._engine._odps.get_xflow_sub_instances(xflow_inst)):
                if sub_inst.id not in self._sub_instance_set:
                    self._sub_instance_set.add(sub_inst.id)
                    write_log('Sub Instance: {0} ({1})'.format(inst_name, sub_inst.id))
                    write_log('  Log view: ' + sub_inst.get_logview_address())
                reload_instance_status(self._engine._odps, self._progress_group, sub_inst.id)
        self._ui.update_group()


@node_runner(RunnerType.SQL)
class SQLNodeRunner(BaseNodeRunner):
    def actual_exec(self):
        query = self._gen_params['sql']
        self._last_cmd = query
        write_log('Command: ' + self._last_cmd)

        inst = self._engine._odps.run_sql(query)
        self.add_instance(inst)


try:
    from ..internal.ml.runners import *
except ImportError:
    pass
