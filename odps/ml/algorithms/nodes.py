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

import re
import time
from functools import partial

from ... import options, tempobj, utils
from ...compat import six
from ...runner import BaseRunnerNode, RunnerPort, PortType, PortDirection, EngineType
from ..nodes import ReusableMixIn
from ..utils import get_function_args, import_class_member, set_table_lifecycle, TEMP_TABLE_PREFIX
from ..nodes.exporters import get_input_table_name, get_input_partitions, get_input_model_name,  get_output_table_name,\
    get_output_table_partitions, get_original_columns, get_enable_sparse, get_kv_delimiter, get_item_delimiter,\
    get_sparse_predict_feature_columns
from ..nodes import exporters


class PmmlPredictNode(BaseRunnerNode):
    def __init__(self):
        super(PmmlPredictNode, self).__init__("Prediction")
        self.marshal({
            "inputs": [(1, "model", PortType.MODEL), (2, "input", PortType.DATA)],
            "outputs": [(1, "output", PortType.DATA)]
        })
        self.add_exporter("inputTableName", lambda: get_input_table_name(self, "input"))
        self.add_exporter("inputTablePartitions", lambda: get_input_partitions(self, "input"))
        self.add_exporter("modelName", lambda: get_input_model_name(self, "model"))
        self.add_exporter("outputTableName", lambda: get_output_table_name(self, "output"))
        self.add_exporter("outputTablePartition", lambda: get_output_table_partitions(self, "output"))
        self.add_exporter("appendColNames", lambda: get_original_columns(self, "input"))
        self.add_exporter('enableSparse', lambda: get_enable_sparse(self, 'enableSparse', 'input'))
        self.add_exporter('itemDelimiter', lambda: get_item_delimiter(self, 'itemDelimiter', 'input'))
        self.add_exporter('kvDelimiter', lambda: get_kv_delimiter(self, 'kvDelimiter', 'input'))
        self.add_exporter('featureColNames', lambda: get_sparse_predict_feature_columns(self, 'featureColNames', 'input'))


PARAMED_EXPORTER_REGEX = re.compile(r'([^\(]+)\(([^\)]+)\) *')


class ProcessorNode(BaseRunnerNode):
    def __init__(self, name, param_defs, port_defs, meta_defs):
        super(ProcessorNode, self).__init__(name, engine=EngineType(meta_defs.get('engine', 'xflow').upper()))

        self.marshal({
            "parameters": dict((p.name, p.value) for p in six.itervalues(param_defs)),
            "inputs": [RunnerPort(seq=p.sequence, name=p.name, port_type=p.type, required=p.required)
                       for p in six.itervalues(port_defs) if p.io_type == PortDirection.INPUT],
            "outputs": [RunnerPort(seq=p.sequence, name=p.name, port_type=p.type, required=p.required)
                        for p in six.itervalues(port_defs) if p.io_type == PortDirection.OUTPUT],
            "metas": meta_defs,
            "exported": set(p.name for p in six.itervalues(param_defs) if p.exported)
        })

        for param in (p for p in six.itervalues(param_defs) if p.exporter is not None):
            def fetch_exporter_func(func_name):
                match = PARAMED_EXPORTER_REGEX.match(func_name)
                arg_dict = dict()
                if match:
                    func_name, arg_str = match.group(1), match.group(2)
                    arg_str = arg_str.replace(r'\,', '\x01')
                    for arg_desc in arg_str.split(','):
                        arg_desc = arg_desc.replace('\x01', ',')
                        k, v = arg_desc.strip().split('=', 1)
                        arg_dict[k.strip()] = v.strip()
                if '.' not in func_name:
                    func = getattr(exporters, func_name)
                else:
                    func = import_class_member(func_name)
                if not arg_dict:
                    return func
                else:
                    return partial(func, **arg_dict)

            args = dict()
            exporter_func = fetch_exporter_func(param.exporter)
            for arg in get_function_args(exporter_func):
                if arg == 'param_name':
                    args[arg] = param.name
                elif arg == 'param_value':
                    args[arg] = param.value
                elif arg == 'input_name':
                    args[arg] = param.input_name
                elif arg == 'output_name':
                    args[arg] = param.output_name

            def make_exporter_wrapper(exporter_name, args_dict):
                func = fetch_exporter_func(exporter_name)
                return lambda: func(self, **args_dict)

            self.add_exporter(param.name, make_exporter_wrapper(param.exporter, args))


class MetricsMixIn(ReusableMixIn, BaseRunnerNode):
    @property
    def sink(self):
        return self._sink

    def after_exec(self, odps, is_success):
        super(MetricsMixIn, self).after_exec(odps, is_success)
        if is_success:
            self.proc_temp_tables(odps)

    def set_computed(self, odps, value):
        self._sink = value

    def get_computed(self, odps):
        return self._sink

    def compute_result(self, odps):
        self.calc_metrics(odps)

    def calc_metrics(self, odps):
        pass

    def proc_temp_tables(self, odps):
        lifecycle = options.temp_lifecycle
        if isinstance(self.table_names, (list, set, tuple)):
            deal_tables = [tn for tn in self.table_names if tn is not None]
        else:
            deal_tables = [self.table_names, ]
        for table_name in deal_tables:
            tempobj.register_temp_table(odps, table_name)
            set_table_lifecycle(odps, table_name, lifecycle)

    def gen_temp_table_name(self, suffix=None):
        if not options.runner.dry_run:
            ts = int(time.time())
        else:
            ts = 0
        code_name = self.code_name
        if hasattr(self, 'temp_table_code_name'):
            code_name = self.temp_table_code_name
        ret = TEMP_TABLE_PREFIX + '%s_%d_%d_res' % (utils.camel_to_underline(code_name), ts, self.node_id)
        if suffix:
            ret += '_' + suffix
        return ret


class AlgoMetricsNode(MetricsMixIn, ProcessorNode):
    def __init__(self, name, param_defs, port_defs, meta_defs):
        super(AlgoMetricsNode, self).__init__(name, param_defs, port_defs, meta_defs)

        out_table_params = [p for p in six.itervalues(param_defs)
                            if p.output_name is not None and p.name.endswith('TableName')]
        out_seq = [port_defs[p.output_name].sequence for p in out_table_params]
        self.table_names = [None, ] * (1 + max(out_seq)) if len(out_table_params) > 1 else None

        if len(out_table_params) > 1:
            def gen_output_table_name(seq):
                self.table_names[seq] = self.gen_temp_table_name(str(seq))
                return self.table_names[seq]
        else:
            def gen_output_table_name(seq):
                self.table_names = self.gen_temp_table_name()
                return self.table_names

        for p in out_table_params:
            self.add_exporter(p.name, partial(gen_output_table_name, seq=port_defs[p.output_name].sequence))

    def calc_metrics(self, odps):
        if options.runner.dry_run:
            self._sink = None
            return
        metric_func = import_class_member(self.metas['calculator'])
        self._sink = metric_func(odps, self)
