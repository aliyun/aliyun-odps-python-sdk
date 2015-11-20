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

from six import itervalues

from ..core.dag import BaseDagNode, DagEndpointType
from ..nodes.exporters import get_input_table_name, get_input_partitions, get_input_model_name, \
    get_output_model_name, get_output_table_name, get_output_table_partitions, get_original_columns, \
    get_label_column
from ..nodes import exporters


class AlgorithmParameter(object):
    def __init__(self, name, value=None, input_name=None, output_name=None, exporter=None, exported=True):
        self.name = name
        self.value = value
        self.input_name = input_name
        self.output_name = output_name
        self.exporter = exporter
        self.exported = exported


class AlgorithmIO(object):
    def __init__(self, name, seq, direction, port_type='DATA'):
        self.name = name
        self.seq = seq
        self.direction = direction
        self.port_type = DagEndpointType.DATA if port_type == 'DATA' else DagEndpointType.MODEL


class TrainNode(BaseDagNode):
    def __init__(self, algorithm):
        super(TrainNode, self).__init__("train")
        self.marshal({
            "parameters": {
                "algorithm": algorithm,
            },
            "inputs": [(1, "model", DagEndpointType.MODEL), (2, "input", DagEndpointType.DATA)],
            "outputs": [(1, "model", DagEndpointType.MODEL)]
        })
        self.add_exporter("inputTableName", lambda context: get_input_table_name(context, self, "input"))
        self.add_exporter("inputTablePartitions", lambda context: get_input_partitions(context, self, "input"))
        self.add_exporter("labelColName", lambda context: get_label_column(context, self, "input"))
        self.add_exporter("inputModelName", lambda context: get_input_model_name(context, self, "model"))
        self.add_exporter("outputModelName", lambda context: get_output_model_name(context, self, "model"))


class PredictNode(BaseDagNode):
    def __init__(self):
        super(PredictNode, self).__init__("Prediction")
        self.marshal({
            "inputs": [(1, "model", DagEndpointType.MODEL), (2, "input", DagEndpointType.DATA)],
            "outputs": [(1, "output", DagEndpointType.DATA)]
        })
        self.add_exporter("inputTableName", lambda context: get_input_table_name(context, self, "input"))
        self.add_exporter("inputTablePartitions", lambda context: get_input_partitions(context, self, "input"))
        self.add_exporter("modelName", lambda context: get_input_model_name(context, self, "model"))
        self.add_exporter("outputTableName", lambda context: get_output_table_name(context, self, "output"))
        self.add_exporter("outputTablePartitions", lambda context: get_output_table_partitions(context, self, "output"))
        self.add_exporter("appendColNames", lambda context: get_original_columns(context, self, "input"))


class IndependentAlgorithmNode(BaseDagNode):
    def __init__(self, name, parameter_defs):
        super(IndependentAlgorithmNode, self).__init__(name)
        self.virtual = True

        self._id = id
        self.marshal({
            "parameters": {p.name: p.value for p in itervalues(parameter_defs)},
            "outputs": [(1, "model", DagEndpointType.MODEL)],
            "exported": {p.name for p in itervalues(parameter_defs) if p.exported}
        })


class ProcessorNode(BaseDagNode):
    def __init__(self, name, parameter_defs, port_defs, meta_defs):
        super(ProcessorNode, self).__init__(name)

        self.marshal({
            "parameters": {p.name: p.value for p in itervalues(parameter_defs)},
            "inputs": [(p.seq, p.name, p.port_type) for p in itervalues(port_defs) if p.direction == 'INPUT'],
            "outputs": [(p.seq, p.name, p.port_type) for p in itervalues(port_defs) if p.direction == 'OUTPUT'],
            "exported": {p.name for p in itervalues(parameter_defs) if p.exported}
        })

        for param in itervalues(parameter_defs):
            if param.exporter is None:
                continue

            exporter_func = getattr(exporters, param.exporter)
            args = dict()
            for arg in exporter_func.__code__.co_varnames:
                if arg == 'param_name':
                    args[arg] = param.name
                elif arg == 'input_name':
                    args[arg] = param.input_name
                elif arg == 'output_name':
                    args[arg] = param.output_name

            def make_exporter_wrapper(exporter_name, args_dict):
                func = getattr(exporters, exporter_name)
                return lambda context: func(context, self, **args_dict)

            self.add_exporter(param.name, make_exporter_wrapper(param.exporter, args))
