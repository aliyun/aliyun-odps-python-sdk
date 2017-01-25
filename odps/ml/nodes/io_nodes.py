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

from ...runner import InputOutputNode, ObjectDescription, PortType, adapter_from_df
from ..utils import TABLE_MODEL_PREFIX, TABLE_MODEL_SEPARATOR
from ...compat import six


class PmmlModelInputNode(InputOutputNode):
    def __init__(self, model_name, project=None):
        super(PmmlModelInputNode, self).__init__('pmml_input')
        self.virtual = True

        self.marshal({
            'parameters': {
                'modelName': model_name if not project else project + '.' + model_name,
            },
            'outputs': [(1, 'model', PortType.MODEL)]
        })

    def propagate_names(self):
        model_name = self.parameters['modelName']
        for output in six.itervalues(self.outputs):
            model = output.obj
            if not model:
                continue
            model._model_name = model_name
        return None


class PmmlModelOutputNode(InputOutputNode):
    def __init__(self, model_name, project=None):
        super(PmmlModelOutputNode, self).__init__('pmml_output')
        self.virtual = True

        self.marshal({
            'parameters': {
                'modelName': model_name if not project else project + '.' + model_name,
            },
            'inputs': [(1, 'model', PortType.MODEL)]
        })

    def propagate_names(self):
        model_name = self.parameters['modelName']
        executed = False
        for inputs in six.itervalues(self.input_edges):
            for edge in inputs:
                if edge.from_node.executed:
                    executed = True
                model = edge.from_node.outputs[edge.from_arg].obj
                if not model:
                    continue
                model._model_name = model_name
        return ObjectDescription(offline_models=model_name) if not executed else None


class TablesModelInputNode(InputOutputNode):
    def __init__(self, table_prefix, project=None):
        super(TablesModelInputNode, self).__init__('tables_model_input')
        self.virtual = True

        self.marshal({
            'parameters': {
                'inputTablePrefix': table_prefix if not project else project + '.' + table_prefix,
            },
            'outputs': [(1, 'model', PortType.MODEL)]
        })

    def propagate_names(self):
        table_prefix = self.parameters.get('inputTablePrefix')
        for output in six.itervalues(self.outputs):
            model = output.obj
            if not model:
                continue
            for dsname, ds in six.iteritems(model._dfs):
                ds.table = TABLE_MODEL_PREFIX + table_prefix + TABLE_MODEL_SEPARATOR + dsname
        return None


class TablesModelOutputNode(InputOutputNode):
    def __init__(self, table_prefix, project=None):
        super(TablesModelOutputNode, self).__init__('tables_model_output')
        self.virtual = True

        self.marshal({
            'parameters': {
                'outputTablePrefix': table_prefix if not project else project + '.' + table_prefix,
            },
            'inputs': [(1, 'model', PortType.MODEL)]
        })

    def propagate_names(self):
        table_prefix = self.parameters.get('outputTablePrefix')
        tables = []
        executed = False
        for inputs in six.itervalues(self.input_edges):
            for edge in inputs:
                if edge.from_node.executed:
                    executed = True
                model = edge.from_node.outputs[edge.from_arg].obj
                if not model:
                    continue
                for dsname, input_df in six.iteritems(model._dfs):
                    adapter = adapter_from_df(input_df)
                    for adapter in adapter._get_adapter_chain():
                        adapter.table = TABLE_MODEL_PREFIX + table_prefix + TABLE_MODEL_SEPARATOR + dsname
                    tables.append(adapter.table)
        return ObjectDescription(tables=tables) if not executed else None
