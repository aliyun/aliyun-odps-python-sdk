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

from ...utils.odps_utils import FieldRole, FieldContinuity


def get_input_table_name(context, node, input_name):
    data_uuid = node.inputs[input_name].data_set_uuid
    return context._ds_container[data_uuid]._table


def get_input_partitions(context, node, input_name):
    data_uuid = node.inputs[input_name].data_set_uuid
    return context._ds_container[data_uuid]._partition


def get_output_table_name(context, node, output_name):
    data_uuid = node.outputs[output_name].data_set_uuid
    return context._ds_container[data_uuid]._table


def get_output_table_partitions(context, node, output_name):
    data_uuid = node.outputs[output_name].data_set_uuid
    return context._ds_container[data_uuid]._partition


def get_input_model_name(context, node, input_name):
    model_uuid = node.inputs[input_name].model_uuid
    return context._model_container[model_uuid]._model_name


def get_output_model_name(context, node, output_name):
    model_uuid = node.outputs[output_name].model_uuid
    return context._model_container[model_uuid]._model_name


def generate_model_name(context, node, output_name):
    return get_output_model_name(context, node, output_name)


def get_label_column(context, node, input_name):
    return get_input_field_param_names(context, node, input_name, FieldRole.LABEL)


def get_weight_column(context, node, input_name):
    return get_input_field_param_names(context, node, input_name, FieldRole.WEIGHT)


def get_feature_columns(context, node, input_name):
    return get_input_field_param_names(context, node, input_name, FieldRole.FEATURE)


def get_feature_continuous(context, node, input_name):
    return [1 if v == FieldContinuity.CONTINUOUS else 0
            for v in get_input_field_param_continuous(context, node, input_name, FieldRole.FEATURE)]


def get_input_field_param_names(context, node, input_name, frole):
    data_uuid = node.inputs[input_name].data_set_uuid
    fields = context._ds_container[data_uuid]._fields
    return [f.name for f in fields if f.role == frole]


def get_input_field_param_continuous(context, node, input_name, frole):
    data_uuid = node.inputs[input_name].data_set_uuid
    fields = context._ds_container[data_uuid]._fields
    return [f.continuity for f in fields if f.role == frole]


def get_original_columns(context, node, input_name):
    data_uuid = node.inputs[input_name].data_set_uuid
    fields = context._ds_container[data_uuid]._fields
    return [f.name for f in fields if not f.is_append]
