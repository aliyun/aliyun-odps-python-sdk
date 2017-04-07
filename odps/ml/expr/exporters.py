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

from ..utils import ML_ARG_PREFIX
from ..enums import FieldRole, FieldContinuity
from ...compat import six
from ...df.expr.core import Node
from ...df.expr.collections import FilterPartitionCollectionExpr

"""
Common exporters
"""


def repr_exist(obj):
    return repr(obj) if obj else None


def get_ml_input(expr, input_name):
    input_expr = getattr(expr, ML_ARG_PREFIX + input_name, None)
    if isinstance(input_expr, Node):
        return input_expr
    else:
        return None


def get_input_table_name(expr, input_name):
    input_expr = get_ml_input(expr, input_name)
    if input_expr is None:
        return None
    if isinstance(input_expr, FilterPartitionCollectionExpr):
        try:
            ds = next(input_expr.data_source())
        except StopIteration:
            return None
    else:
        ds = input_expr._source_data
    if ds is None:
        return None
    elif ds.project.name == ds.odps.project:
        return ds.name
    else:
        return '{0}.{1}'.format(ds.project.name, ds.name)


def get_input_partitions(expr, input_name):
    input_expr = get_ml_input(expr, input_name)
    if input_expr is None:
        return None
    if not isinstance(input_expr, FilterPartitionCollectionExpr):
        return None
    return input_expr._predicate_string

get_input_table_partitions = get_input_partitions


def get_input_model_table_name(expr, input_name, table_name):
    obj = get_ml_input(expr, input_name)
    if not getattr(obj, '_source_data', None):
        return None
    tm = obj._source_data
    table_name = tm.tables[table_name].name
    if tm.project.name == tm.odps.project:
        return table_name
    else:
        return '{0}.{1}'.format(tm.project.name, table_name)


def get_input_model_param(expr, input_name, param_name):
    if expr._params.get(param_name, None):
        return expr._params[param_name]
    obj = get_ml_input(expr, input_name)
    if obj is None:
        return None
    return obj._model_params[param_name]


def get_output_model_table_name(expr, output_name, table_name):
    expr = expr.outputs().get(output_name)
    if expr is None or not hasattr(expr, table_name):
        return None
    persist_kw = getattr(expr, 'persist_kw', dict())
    model_name = persist_kw.get('_model')
    table_name = getattr(expr, table_name).table_name(model_name)
    if persist_kw.get('_project') is None:
        return table_name
    else:
        return persist_kw.get('_project') + '.' + table_name


def get_output_table_name(expr, output_name):
    if hasattr(expr, 'tables'):
        return getattr(expr.tables, output_name)

    expr_output = expr.outputs().get(output_name)
    if expr_output is None:
        if 'required_outputs' in expr.shared_kw:
            return expr.shared_kw['required_outputs'].get(output_name)
        else:
            return None
    persist_kw = getattr(expr_output, 'persist_kw', dict())
    if persist_kw.get('_project') is None:
        return persist_kw.get('_table')
    else:
        return persist_kw.get('_project') + '.' + persist_kw.get('_table')


def get_output_table_partition(expr, output_name):
    expr = expr.outputs().get(output_name)
    if expr is None:
        return None
    return repr_exist(getattr(expr, 'persist_kw', dict()).get('_partition'))

get_output_table_partitions = get_output_table_partition


def get_input_model_name(expr, input_name):
    model_expr = get_ml_input(expr, input_name)
    model_obj = getattr(model_expr, '_source_data', None)
    if model_obj is None:
        return None
    elif model_obj.project.name == model_obj.odps.project:
        return model_obj.name
    else:
        return model_obj.project.name + '.' + model_obj.name


def get_output_model_name(expr, output_name):
    expr = expr.outputs().get(output_name)
    if expr is None:
        return None
    persist_kw = getattr(expr, 'persist_kw', dict())
    if persist_kw.get('_project') is None:
        return persist_kw.get('_model')
    else:
        return persist_kw.get('_project') + '.' + persist_kw.get('_model')


def get_input_field_names(expr, param_name, input_name, field_role=None, field_func=None):
    if expr._params.get(param_name, None):
        return expr._params[param_name]
    data_obj = get_ml_input(expr, input_name)
    if data_obj is None:
        return None
    fields = data_obj._ml_fields
    if field_role:
        return [f.name for f in fields if field_role in f.role]
    else:
        return [f.name for f in fields if field_func(f)]


def get_input_field_name(expr, param_name, input_name, field_role=None, field_func=None):
    v = get_input_field_names(expr, param_name, input_name, field_role, field_func)
    return v[0] if v else None


def get_input_field_ids(expr, param_name, input_name, field_role=None, field_func=None):
    if expr._params.get(param_name, None):
        existing_values = expr._params[param_name]
        if isinstance(existing_values, six.string_types):
            existing_values = set(v.strip() for v in existing_values.split(','))
    else:
        existing_values = set()
    data_obj = get_ml_input(expr, input_name)
    if data_obj is None:
        return None
    fields = data_obj._ml_fields
    if existing_values:
        return [idx for idx, f in enumerate(fields) if f.name in existing_values]
    if field_role:
        return [idx for idx, f in enumerate(fields) if field_role in f.role]
    else:
        return [idx for idx, f in enumerate(fields) if field_func(f)]


def get_input_field_id(expr, param_name, input_name, field_role=None, field_func=None):
    v = get_input_field_ids(expr, param_name, input_name, field_role, field_func)
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


def get_feature_continuous(expr, input_name):
    return [1 if v == FieldContinuity.CONTINUOUS else 0
            for v in get_input_field_continuous(expr, input_name, FieldRole.FEATURE)]


def get_input_field_continuous(expr, input_name, field_role):
    data_obj = get_ml_input(expr, input_name)
    if data_obj is None:
        return None
    fields = data_obj._ml_fields
    return [f.continuity for f in fields if field_role in f.role]


def get_original_columns(expr, input_name):
    data_obj = get_ml_input(expr, input_name)
    if data_obj is None:
        return None
    fields = data_obj._ml_fields
    return [f.name for f in fields if not f.is_append]


def get_unique_feature_field_property(expr, param_name, input_name, prop_fetcher, default=None):
    if expr._params.get(param_name, None):
        return expr._params[param_name]
    data_obj = get_ml_input(expr, input_name)
    if data_obj is None:
        return None
    fields = data_obj._ml_fields
    feature_fields = [f for f in fields if FieldRole.FEATURE in f.role]
    value_set = set(prop_fetcher(f) for f in feature_fields)
    if len(value_set) == 0:
        return default
    if len(value_set) != 1:
        raise ValueError('Property %s must be unique among features.' % param_name)
    ret = next(iter(value_set))
    return ret if ret is not None else default


def convert_json(expr, param_name):
    param_val = expr._params.get(param_name, None)
    if isinstance(param_val, six.string_types):
        return param_val
    elif param_val:
        return json.dumps(expr._params[param_name], separators=(',', ':'))
    return None


get_kv_delimiter = partial(get_unique_feature_field_property,
                           prop_fetcher=lambda f: f.kv_config.kv_delimiter if f.kv_config else None)
get_item_delimiter = partial(get_unique_feature_field_property,
                             prop_fetcher=lambda f: f.kv_config.item_delimiter if f.kv_config else None)
get_enable_sparse = partial(get_unique_feature_field_property,
                            prop_fetcher=lambda f: True if f.kv_config is not None else None)


def get_sparse_predict_feature_columns(expr, param_name, input_name, enable_sparse_param='enableSparse'):
    return get_feature_columns(expr, param_name, input_name)\
        if get_enable_sparse(expr, enable_sparse_param, input_name) else None
