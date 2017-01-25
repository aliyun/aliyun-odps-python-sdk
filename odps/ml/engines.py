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

import logging

from ..config import options  ## don't remove
from ..compat import six  ## don't remove
from .algorithms.nodes import MetricsMixIn
from ..utils import write_log
from ..runner import node_engine, EngineType, BaseNodeEngine
from ..ui import reload_instance_status, fetch_instance_group
from .. import options

logger = logging.getLogger(__name__)


@node_engine(EngineType.XFLOW)
class XFlowNodeEngine(BaseNodeEngine):
    def __init__(self, node, context, runner):
        super(XFlowNodeEngine, self).__init__(node, context, runner)
        self._sub_instance_set = set()

    @staticmethod
    def _is_param_valid(val):
        if val is None:
            return False
        if isinstance(val, (list, set, tuple)) and len(val) == 0:
            return False
        return True

    def before_exec(self):
        super(XFlowNodeEngine, self).before_exec()
        if 'core_num' in self._node.engine_args:
            self._gen_params['coreNum'] = self._node.engine_args['core_num']
        if 'core_mem' in self._node.engine_args:
            self._gen_params['memSizePerCore'] = self._node.engine_args['core_mem']

    def actual_exec(self):
        if isinstance(self._node, MetricsMixIn):
            self._lifecycle = options.temp_lifecycle
        if self._lifecycle:
            self._gen_params['lifecycle'] = self._lifecycle

        pai_project = self._node.metas.get('xflowProjectName', options.ml.xflow_project)
        xflow_name = self._node.metas.get('xflowName', self._node.code_name)

        params = dict([(k, self._format_value(v)) for k, v in six.iteritems(self._gen_params)
                       if self._is_param_valid(v)])

        param_args = ' '.join(['-D%s="%s"' % (k, v) for k, v in six.iteritems(params)
                               if self._is_param_valid(v)])
        self._last_cmd = 'PAI -name %s -project %s %s;' % (xflow_name, pai_project, param_args)
        write_log('Command: ' + self._last_cmd)

        inst = self._odps.run_xflow(xflow_name, pai_project, params)
        self.add_instance(inst, show_log_view=False)
        inst.wait_for_success()

    def refresh_progress(self):
        if self._instances:
            inst = self._instances[0]
            group_json = fetch_instance_group(self._progress_group)
            group_json.logview = inst.get_logview_address()
        for xflow_inst in self._instances:
            for x_result in filter(lambda xr: xr.node_type != 'Local',
                                   six.itervalues(self._odps.get_xflow_results(xflow_inst))):
                if x_result.node_type == 'Instance' and x_result.instance_id not in self._sub_instance_set:
                    self._sub_instance_set.add(x_result.instance_id)
                    sub_inst = self._odps.get_instance(x_result.instance_id)
                    write_log('Sub Instance: {0} ({1})'.format(x_result.name, x_result.instance_id))
                    write_log('  Log view: ' + sub_inst.get_logview_address())
                reload_instance_status(self._odps, self._progress_group, x_result.instance_id)

    def show_error(self):
        super(XFlowNodeEngine, self).show_error()
        if self._instances:
            self.refresh_progress()
            group_json = fetch_instance_group(self._progress_group)
            if group_json.instances:
                last_inst = list(six.itervalues(group_json.instances))[-1]
                logger.error('Logview: ' + last_inst.logview)


@node_engine(EngineType.SQL)
class SQLNodeEngine(BaseNodeEngine):
    def actual_exec(self):
        scripts = [self._gen_params['script'], ] if isinstance(self._gen_params['script'], six.string_types) \
            else self._gen_params['script']
        self._last_cmd = ';\n'.join(scripts)
        logger.debug('Generated SQL statement:\n' + self._last_cmd)
        for script in (s for s in scripts if s):
            inst = self._odps.run_sql(script)
            self.add_instance(inst)
            inst.wait_for_success()


try:
    from ..internal.ml.engines import *
except ImportError:
    pass
