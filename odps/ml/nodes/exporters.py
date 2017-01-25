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

from functools import partial
import json

from ..enums import FieldRole, FieldContinuity
from ...compat import six
from ...runner import adapter_from_df

"""
Common exporters
"""


def repr_exist(obj):
    return repr(obj) if obj else None


def get_input_table_name(node, input_name):
    data_obj = node.inputs[input_name].obj
    if data_obj is None:
        return None
    return data_obj.table


def get_input_partitions(node, input_name):
    data_obj = node.inputs[input_name].obj
    if data_obj is None:
        return None
    return repr_exist(data_obj.partitions)

get_input_table_partitions = get_input_partitions


def get_input_model_table_name(node, input_name, table_name):
    obj = node.inputs[input_name].obj
    if obj is None:
        return None
    return adapter_from_df(getattr(obj, table_name)).table


def get_input_model_param(node, input_name, param_name):
    if node.parameters.get(param_name, None):
        return node.parameters[param_name]
    obj = node.inputs[input_name].obj
    if obj is None:
        return None
    return obj._params[param_name]


def get_input_model_table_partitions(node, input_name, table_name):
    obj = node.inputs[input_name].obj
    if obj is None:
        return None
    return repr_exist(adapter_from_df(getattr(obj, table_name)).partitions)


def get_output_model_table_name(node, output_name, table_name):
    obj = node.outputs[output_name].obj
    if obj is None:
        return None
    if hasattr(obj, table_name):
        return adapter_from_df(getattr(obj, table_name)).table
    else:
        return None


def get_output_model_table_partitions(node, output_name, table_name):
    obj = node.outputs[output_name].obj
    if obj is None:
        return None
    return repr_exist(adapter_from_df(getattr(obj, table_name)).partitions)


def get_output_table_name(node, output_name):
    port = node.outputs[output_name]
    data_obj = port.obj
    if data_obj is None:
        return None
    return data_obj.table


def get_output_table_partitions(node, output_name):
    data_obj = node.outputs[output_name].obj
    if data_obj is None:
        return None
    return repr_exist(data_obj.partitions)


def get_output_table_partition(node, output_name):
    data_obj = node.outputs[output_name].obj
    if data_obj is None:
        return None
    return repr_exist(data_obj.partitions)


def get_input_model_name(node, input_name):
    obj = node.inputs[input_name].obj
    if obj is None:
        return None
    return obj._model_name


def get_output_model_name(node, output_name):
    obj = node.outputs[output_name].obj
    if obj is None:
        return None
    return obj._model_name


def get_input_field_names(node, param_name, input_name, field_role=None, field_func=None):
    if node.parameters.get(param_name, None):
        return node.parameters[param_name]
    data_obj = node.inputs[input_name].obj
    if data_obj is None:
        return None
    fields = data_obj.fields
    if field_role:
        return [f.name for f in fields if field_role in f.role]
    else:
        return [f.name for f in fields if field_func(f)]


def get_input_field_name(node, param_name, input_name, field_role=None, field_func=None):
    v = get_input_field_names(node, param_name, input_name, field_role, field_func)
    return v[0] if v else None


def get_input_field_ids(node, param_name, input_name, field_role=None, field_func=None):
    if node.parameters.get(param_name, None):
        existing_values = node.parameters[param_name]
        if isinstance(existing_values, six.string_types):
            existing_values = set(v.strip() for v in existing_values.split(','))
    else:
        existing_values = set()
    data_obj = node.inputs[input_name].obj
    if data_obj is None:
        return None
    fields = data_obj.fields
    if existing_values:
        return [idx for idx, f in enumerate(fields) if f.name in existing_values]
    if field_role:
        return [idx for idx, f in enumerate(fields) if field_role in f.role]
    else:
        return [idx for idx, f in enumerate(fields) if field_func(f)]


def get_input_field_id(node, param_name, input_name, field_role=None, field_func=None):
    v = get_input_field_ids(node, param_name, input_name, field_role, field_func)
    return v[0] if v else None


generate_model_name = get_output_model_name
get_group_id_column = partial(get_input_field_names, field_role=FieldRole.GROUP_ID)
get_label_column = partial(get_input_field_names, field_role=FieldRole.LABEL)
get_weight_column = partial(get_input_field_names, field_role=FieldRole.WEIGHT)
get_predicted_class_column = partial(get_input_field_names, field_role=FieldRole.PREDICTED_CLASS)
get_predicted_score_column = partial(get_input_field_names, field_role=FieldRole.PREDICTED_SCORE)
get_predicted_detail_column = partial(get_input_field_names, field_role=FieldRole.PREDICTED_DETAIL)
get_feature_columns = partial(get_input_field_names, field_role=FieldRole.FEATURE)
get_non_feature_columns = partial(get_input_field_names, field_func=lambda f: FieldRole.FEATURE not in f.role)


def get_feature_continuous(node, input_name):
    return [1 if v == FieldContinuity.CONTINUOUS else 0
            for v in get_input_field_continuous(node, input_name, FieldRole.FEATURE)]


def get_input_field_continuous(node, input_name, frole):
    data_obj = node.inputs[input_name].obj
    if data_obj is None:
        return None
    fields = data_obj.fields
    return [f.continuity for f in fields if frole in f.role]


def get_original_columns(node, input_name):
    data_obj = node.inputs[input_name].obj
    if data_obj is None:
        return None
    fields = data_obj.fields
    return [f.name for f in fields if not f.is_append]


def get_unique_feature_field_property(node, param_name, input_name, prop_fetcher, default=None):
    if node.parameters.get(param_name, None):
        return node.parameters[param_name]
    data_obj = node.inputs[input_name].obj
    if data_obj is None:
        return None
    feature_fields = [f for f in data_obj.fields if FieldRole.FEATURE in f.role]
    value_set = set(prop_fetcher(f) for f in feature_fields)
    if len(value_set) == 0:
        return default
    if len(value_set) != 1:
        raise ValueError('Property %s must be unique among features.' % param_name)
    ret = next(iter(value_set))
    return ret if ret is not None else default


def convert_json(node, param_name):
    param_val = node.parameters.get(param_name, None)
    if isinstance(param_val, six.string_types):
        return param_val
    elif param_val:
        return json.dumps(node.parameters[param_name], separators=(',', ':'))
    return None


get_kv_delimiter = partial(get_unique_feature_field_property,
                           prop_fetcher=lambda f: f.kv_config.kv_delimiter if f.kv_config else None)
get_item_delimiter = partial(get_unique_feature_field_property,
                             prop_fetcher=lambda f: f.kv_config.item_delimiter if f.kv_config else None)
get_enable_sparse = partial(get_unique_feature_field_property,
                            prop_fetcher=lambda f: True if f.kv_config is not None else None)


def get_sparse_predict_feature_columns(node, param_name, input_name, enable_sparse_param='enableSparse'):
    return get_feature_columns(node, param_name, input_name)\
        if get_enable_sparse(node, enable_sparse_param, input_name) else None
