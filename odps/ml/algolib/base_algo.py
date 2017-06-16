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

from __future__ import absolute_import

import functools
import re
import time
import uuid
from collections import namedtuple, Iterable

from odps.df.expr.dynamic import DynamicMixin
from .objects import SchemaDef
from ..expr import ModelDataCollectionExpr, ODPSModelExpr
from ..expr.op import ProgrammaticFieldChangeOperation
from ..pipeline.core import PipelineStep
from ..utils import import_class_member, get_function_args, ML_ARG_PREFIX
from ...compat import six, OrderedDict, irange
from ...df.core import DataFrame
from ...models import Schema as TableSchema
from ..enums import PortType, PortDirection
from ...utils import underline_to_capitalized, camel_to_underline, survey


def is_df_object(obj):
    from ...df.core import CollectionExpr
    from ...df.expr.expressions import SequenceExpr
    return isinstance(obj, (CollectionExpr, SequenceExpr))


def is_ml_object(obj):
    return is_df_object(obj) or isinstance(obj, ODPSModelExpr)


class BaseAlgorithm(PipelineStep):
    _entry_method = None

    def __init__(self, name, parameters, ports, metas=None):
        self._name = name
        self._parameters = parameters
        self._ports = ports
        self._metas = metas if metas is not None else {}
        if not hasattr(self, '_reload_ml_fields'):
            self._reload_ml_fields = False

        output_names = [p.name for p in sorted(six.itervalues(self._ports), key=lambda pt: pt.sequence)
                        if p.io_type == PortDirection.OUTPUT]
        param_names = [p for p in six.iterkeys(self._parameters)]
        [self._invoke_setter(p.name, p.value) for p in six.itervalues(self._parameters)]

        super(BaseAlgorithm, self).__init__(name, param_names=param_names, output_names=output_names)

    @staticmethod
    def _get_exec_id():
        return '%d-%s' % (int(time.time()), str(uuid.uuid4()))

    def _invoke_setter(self, param_name, value):
        if not self._parameters[param_name].setter:
            return
        method = import_class_member(self._parameters[param_name].setter)
        method(self, param_name, value)

    def _map_inputs_from_args(self, *args, **kwargs):
        inputs = [p for p in sorted(six.itervalues(self._ports), key=lambda pt: pt.sequence)
                  if p.io_type == PortDirection.INPUT]

        args = [arg for arg in args if is_ml_object(arg)]
        ml_kw = dict((k, v) for k, v in six.iteritems(kwargs) if is_ml_object(v))

        total_input = len(args) + len(ml_kw)
        if total_input == 0:
            raise TypeError('Can only perform transformation on DataFrames')
        if total_input > len(inputs):
            raise ValueError('Input count mismatch.')

        # combine args
        obj_dict = dict(zip((p.name for p in inputs), args))
        obj_dict.update(ml_kw)

        return obj_dict

    def _build_collection(self, input_obj_dict, schema_def, coll_type=None, **expr_kw):
        coll_type = coll_type or self._collection_expr
        if schema_def and schema_def.dynamic:
            coll_type = type(coll_type)('Dynamic' + coll_type.__name__, (DynamicMixin, coll_type), {})

        if schema_def is None:
            schema_def = SchemaDef()
            schema_def.copy_input = six.next(k for k, v in six.iteritems(input_obj_dict) if is_df_object(v))

        kw = dict((ML_ARG_PREFIX + k, v) for k, v in six.iteritems(input_obj_dict))
        kw.update(expr_kw)
        if schema_def.copy_input is not None:
            src_df = coll_type(register_expr=True, _schema=input_obj_dict[schema_def.copy_input]._schema, **kw)
            src_df._ml_uplink = [input_obj_dict[schema_def.copy_input]]
        else:
            src_df = coll_type(register_expr=True, _schema=TableSchema(_columns=[]), **kw)
            src_df._ml_uplink = [o for o in six.itervalues(input_obj_dict) if is_df_object(o)]

        out_schemas = dict((pname, dict((f.name, f.type) for f in df._ml_fields))
                           for pname, df in six.iteritems(input_obj_dict) if is_df_object(df))
        for model_name, model in ((nm, m) for nm, m in six.iteritems(input_obj_dict) if isinstance(m, ODPSModelExpr)):
            if not getattr(model, '_model_collections', None):
                continue
            out_schemas.update(dict(('%s.%s' % (model_name, ds_name), dict((f.name, f.type) for f in df._ml_fields))
                                    for ds_name, df in six.iteritems(model._model_collections)))

        if schema_def.programmatic:
            generator = import_class_member(schema_def.schema)
        else:
            generator = _static_ml_fields_generator
        func_args = get_function_args(generator)

        kw = OrderedDict()
        if 'params' in func_args:
            kw['params'] = src_df.convert_params()
        if 'fields' in func_args:
            kw['fields'] = out_schemas
        if 'algorithm' in func_args:
            kw['algorithm'] = self
        if 'schema' in func_args:
            kw['schema'] = schema_def.schema

        op = ProgrammaticFieldChangeOperation(functools.partial(generator, **kw), schema_def.copy_input is not None)
        src_df._perform_operation(op)
        src_df._rebuild_df_schema(schema_def.dynamic)
        return src_df

    @survey
    def _do_transform(self, *args, **kwargs):
        exec_id = self._get_exec_id()

        out_port_defs = [p for p in sorted(six.itervalues(self._ports), key=lambda pt: pt.sequence)
                         if p.io_type == PortDirection.OUTPUT]
        params = dict((k, v.value) for k, v in six.iteritems(self._parameters))

        obj_dict = self._map_inputs_from_args(*args, **kwargs)
        # pick out engine kwargs such as core number, mem usage, etc.
        engine_kw = dict(p for p in six.iteritems(kwargs) if p[0] not in obj_dict)

        out_objs = dict()
        for port_def in out_port_defs:
            port_param_name = camel_to_underline(port_def.name)

            expr_kw = dict(_params=params, _exec_id=exec_id, _output_name=port_def.name,
                           _engine_kw=engine_kw)
            if port_def.type == PortType.DATA:
                out_objs[port_param_name] = self._build_collection(obj_dict, port_def.schema, **expr_kw)
            elif port_def.type == PortType.MODEL:
                expr_kw.update((ML_ARG_PREFIX + k, v) for k, v in six.iteritems(obj_dict))

                model_type = port_def.model.type if port_def.model else 'PmmlModel'

                if model_type == 'PmmlModel':
                    expr_kw['_is_offline_model'] = True
                else:
                    expr_kw['_is_offline_model'] = False
                    expr_kw['_model_collections'] = None

                out_objs[port_param_name] = model_expr = self._model_expr(register_expr=True, **expr_kw)

                if model_type != 'PmmlModel':
                    dfs = dict()
                    for schema in port_def.model.schemas:
                        if schema.direct_copy:
                            dfs[schema.name] = obj_dict[schema.direct_copy].view()
                        else:
                            ds = self._build_collection(obj_dict, schema, coll_type=ModelDataCollectionExpr,
                                                        _mlattr_model=out_objs[port_param_name], _data_item=schema.name,
                                                        model_inputs=list(six.iterkeys(obj_dict)))
                            dfs[schema.name] = ds
                    model_expr._model_collections = dfs

                    if port_def.model and hasattr(port_def.model, 'copy_params') and port_def.model.copy_params:
                        param_dict = dict((self._parameters[pn].name, self._parameters[pn].value)
                                          for pn in port_def.model.copy_params.split(','))
                        model_expr._model_params.update(param_dict)

            if port_def.exporter:
                out_obj = out_objs[port_param_name]
                method = import_class_member(port_def.exporter)
                out_objs[port_param_name] = method(out_obj)

        result_type = namedtuple(underline_to_capitalized(self._name) + 'Result',
                                 [camel_to_underline(p.name) for p in out_port_defs])
        ret_tuple = result_type(**out_objs)
        if len(ret_tuple) == 1:
            return ret_tuple[0]
        else:
            return ret_tuple


class BaseProcessAlgorithm(BaseAlgorithm):
    _entry_method = 'transform'
    """
    Base processing algorithm class
    """
    def transform(self, *args, **kwargs):
        """
        :type args: list[DataFrame]
        """
        return self._do_transform(*args, **kwargs)


class BaseTrainingAlgorithm(BaseAlgorithm):
    _entry_method = 'train'
    """
    Base class for supervised clustering algorithms
    """
    def train(self, *args, **kwargs):
        """
        Perform training on a DataFrame.
        The label field is specified by the ``label_field`` method.

        :param train_data: DataFrame to be trained. Label field must be specified.
        :type train_data: DataFrame

        :return: Trained model
        :rtype: MLModel
        """
        objs = self._do_transform(*args, **kwargs)
        obj_list = [objs, ] if not isinstance(objs, Iterable) else objs
        for obj in obj_list:
            if not isinstance(obj, ODPSModelExpr):
                continue
            for meta in ['predictor', 'recommender']:
                if meta not in self._metas:
                    continue
                mod = __import__(self.__class__.__module__.__name__, fromlist=[''])\
                    if not hasattr(self, '_env') else self._env
                action_cls_name = underline_to_capitalized(self._metas[meta])
                if not hasattr(mod, action_cls_name):
                    action_cls_name = '_' + action_cls_name
                setattr(obj, '_' + meta, mod + '.' + action_cls_name)

        return objs


class BaseMetricsAlgorithm(BaseAlgorithm):
    def calc(self, *args, **kwargs):
        """
        :type args: list[DataFrame]
        """
        cases = kwargs.pop('_cases', [])
        if not isinstance(cases, Iterable):
            cases = [cases, ]
        result_callback = kwargs.pop('_result_callback', None)
        execute_now = kwargs.pop('execute_now', True)

        exec_id = self._get_exec_id()
        params = dict((k, v.value) for k, v in six.iteritems(self._parameters))

        obj_dict = self._map_inputs_from_args(*args, **kwargs)
        engine_kw = dict(p for p in six.iteritems(kwargs) if p[0] not in obj_dict)

        expr_kw = dict(_params=params, _exec_id=exec_id, _engine_kw=engine_kw)
        if cases:
            expr_kw['_cases'] = cases

        expr_kw.update((ML_ARG_PREFIX + k, v) for k, v in six.iteritems(obj_dict))
        if result_callback:
            expr_kw['_result_callback'] = result_callback
        expr = self._metrics_expr(**expr_kw)

        if execute_now:
            return expr.execute()
        else:
            return expr

_FIELD_PARAM_PLACEHOLDER = re.compile(r'\{([^\}]+)\}')
_REDUNDANT_COMMA = re.compile(r',\s*,')


def _static_ml_fields_generator(params, fields, algorithm, schema):

    def format_col(field_name, field_type, **kw):
        prefix = kw.get('prefix', '')
        role = kw.get('role')
        if role:
            return '%s:%s:%s' % (prefix + field_name, field_type, role)
        else:
            return '%s:%s' % (prefix + field_name, field_type)

    if isinstance(schema, list):
        name_list = []
        for f in schema:
            fr = '{0}:{1}'.format(f.name, f.type)
            if f.role:
                fr += ':' + '#'.join(r.name for r in f.role)
            name_list.append(fr)
        return ','.join(name_list)
    elif schema is None:
        return ''
    else:
        def _sub_func(m):
            cmd = [s.strip() for s in m.group(1).split('|')]
            cmd_name = cmd[0]
            if cmd_name.startswith('#'):
                cmd_name = cmd_name[1:]
                if cmd_name == 'range':
                    for idx, c in enumerate(cmd[1:]):
                        for k, v in six.iteritems(params):
                            c = c.replace(k, str(v))
                        cmd[idx + 1] = c
                    start_expr, end_expr, tmpl = cmd[1:]
                    start_num = eval(start_expr)
                    end_num = eval(end_expr)
                    flist = [tmpl.replace('#n', str(n)) for n in irange(start_num, end_num)]
                    return ','.join(flist)
                elif cmd_name == 'input':
                    format_args = dict(tuple(p.split('=', 1)) for p in cmd[2:])
                    input_ml_fields = fields[cmd[1]]
                    return ','.join(format_col(col, typ, **format_args) for col, typ in six.iteritems(input_ml_fields))
                else:
                    raise ValueError('Unsupported method')
            else:
                format_args = dict(tuple(p.split('=', 1)) for p in cmd[1:])

                input_name = algorithm._parameters[cmd_name].input_name
                if not input_name:
                    return params[cmd_name]
                cols = params[cmd_name] if cmd_name in params else []
                if isinstance(cols, six.string_types):
                    cols = cols.split(',')
                input_ml_fields = fields[input_name]
                cols = cols or []

                return ','.join(format_col(col, input_ml_fields[col], **format_args) for col in cols)
        return _REDUNDANT_COMMA.sub(',', _FIELD_PARAM_PLACEHOLDER.sub(_sub_func, schema))
