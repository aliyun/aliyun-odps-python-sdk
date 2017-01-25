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
from collections import namedtuple, Iterable

from .nodes import ProcessorNode, AlgoMetricsNode
from .objects import SchemaDef
from ..adapter import ProgrammaticFieldChangeOperation
from ..models import MLModel
from ..nodes.transform_nodes import DataCopyNode
from ..pipeline.core import PipelineStep
from ..utils import import_class_member, get_function_args
from ...compat import six, OrderedDict, irange
from ...df.core import DataFrame
from ...ml import models
from ...runner import PortType, PortDirection, RunnerContext, DFAdapter, adapter_from_df
from ...utils import underline_to_capitalized, underline_to_camel, camel_to_underline, survey


def is_df_object(obj):
    from ...df.core import CollectionExpr
    from ...df.expr.expressions import SequenceExpr
    return isinstance(obj, (CollectionExpr, SequenceExpr))


def is_ml_object(obj):
    from ...runner import RunnerObject
    return is_df_object(obj) or isinstance(obj, RunnerObject)


class BaseAlgorithm(PipelineStep):
    _entry_method = None

    def __init__(self, name, parameters, ports, metas=None):
        self._name = name
        self._parameters = parameters
        self._ports = ports
        self._metas = metas if metas is not None else {}
        if not hasattr(self, '_reload_fields'):
            self._reload_fields = False

        output_names = [p.name for p in sorted(six.itervalues(self._ports), key=lambda pt: pt.sequence)
                        if p.io_type == PortDirection.OUTPUT]
        param_names = [p for p in six.iterkeys(self._parameters)]
        [self._invoke_setter(p.name, p.value) for p in six.itervalues(self._parameters)]

        super(BaseAlgorithm, self).__init__(name, param_names=param_names, output_names=output_names)

    def _invoke_setter(self, param_name, value):
        if not self._parameters[param_name].setter:
            return
        method = import_class_member(self._parameters[param_name].setter)
        method(self, param_name, value)

    def _map_inputs_from_args(self, node, *args, **kwargs):
        inputs = [p for p in sorted(six.itervalues(self._ports), key=lambda pt: pt.sequence)
                  if p.io_type == PortDirection.INPUT]

        transform_obj = lambda o: adapter_from_df(o) if is_df_object(o) else o
        args = [transform_obj(arg) for arg in args if is_ml_object(arg)]
        ml_kw = dict((k, transform_obj(v)) for k, v in six.iteritems(kwargs) if is_ml_object(v))

        total_input = len(args) + len(ml_kw)
        if total_input == 0:
            raise TypeError('Can only perform transformation on data sets')
        if total_input > len(inputs):
            raise ValueError('Input count mismatch.')

        # combine args
        obj_dict = dict(zip((p.name for p in inputs), args))
        obj_dict.update(ml_kw)

        # bind object dict
        for input_name, input_obj in six.iteritems(obj_dict):
            input_obj._link_node(node, underline_to_camel(input_name))

        return obj_dict

    def _build_df(self, odps, port_inst, input_obj_dict, schema, exec_node=None):
        if schema is None:
            schema = SchemaDef()
            schema.copy_input = six.next(k for k, v in six.iteritems(input_obj_dict) if isinstance(v, DFAdapter))

        if schema.copy_input is not None:
            src_adapter = input_obj_dict[schema.copy_input]._duplicate_df_adapter(port_inst)
        else:
            ul = [o for o in six.itervalues(input_obj_dict) if isinstance(o, DFAdapter)]
            src_adapter = DFAdapter(odps, port_inst, None, uplink=ul, fields=[])

        out_schemas = dict((pname, dict((f.name, f.type) for f in ds._fields))
                           for pname, ds in six.iteritems(input_obj_dict) if isinstance(ds, DFAdapter))
        for model_name, model in ((nm, m) for nm, m in six.iteritems(input_obj_dict) if isinstance(m, MLModel)):
            if not hasattr(model, '_dfs'):
                continue
            out_schemas.update(dict(('%s.%s' % (model_name, ds_name), dict((f.name, f.type) for f in adapter_from_df(ds)._fields))
                                    for ds_name, ds in six.iteritems(model._dfs)))

        if schema.programmatic:
            generator = import_class_member(schema.schema)
        else:
            generator = _static_fields_generator
        func_args = get_function_args(generator)

        kw = OrderedDict()
        if 'params' in func_args:
            exec_node = exec_node or port_inst.bind_node
            kw['params'] = exec_node.convert_params()
        if 'fields' in func_args:
            kw['fields'] = out_schemas
        if 'algorithm' in func_args:
            kw['algorithm'] = self
        if 'schema' in func_args:
            kw['schema'] = schema.schema

        op = ProgrammaticFieldChangeOperation(functools.partial(generator, **kw), schema.copy_input is not None)
        src_adapter.perform_operation(op)
        return src_adapter.df_from_fields(force_create=True, dynamic=schema.dynamic)

    @survey
    def _do_transform(self, *args, **kwargs):
        cases = kwargs.pop('_cases', [])
        if not isinstance(cases, Iterable):
            cases = [cases, ]

        out_port_defs = [p for p in sorted(six.itervalues(self._ports), key=lambda pt: pt.sequence)
                         if p.io_type == PortDirection.OUTPUT]

        node = ProcessorNode(self._name, self._parameters, self._ports, self._metas)
        node.cases = cases
        node.reload_on_finish = self._reload_fields

        obj_dict = self._map_inputs_from_args(node, *args, **kwargs)
        engine_kw = dict(p for p in six.iteritems(kwargs) if p[0] not in obj_dict)
        node.engine_args.update(engine_kw)
        odps = six.next(six.itervalues(obj_dict))._odps

        out_objs = dict()
        for port_def in out_port_defs:
            port_param_name = camel_to_underline(port_def.name)
            port_inst = node.outputs[port_def.name]

            if port_def.type == PortType.DATA:
                out_objs[port_param_name] = self._build_df(odps, port_inst, obj_dict, port_def.schema)
            elif port_def.type == PortType.MODEL:
                model_type = 'PmmlModel'
                if port_def.model:
                    model_type = port_def.model.type

                if '.' not in model_type:
                    cls = getattr(models, model_type)
                else:
                    cls = import_class_member(model_type)
                obj = cls(odps, port=port_inst)
                if port_def.model and hasattr(port_def.model, 'schemas') and port_def.model.schemas:
                    dfs = dict()
                    obj._set_outputs([s.name for s in port_def.model.schemas])
                    for schema in port_def.model.schemas:
                        if schema.direct_copy:
                            src_adapter = node.inputs[schema.direct_copy].obj
                            if src_adapter.table:  # already stored
                                copy_node = DataCopyNode()
                                src_adapter._link_node(copy_node, 'input1')

                                output_adapter = DFAdapter(odps, copy_node.outputs.get("output1"), None,
                                                           uplink=[src_adapter], fields=src_adapter._fields)
                                dfs[schema.name] = output_adapter.df_from_fields(dynamic=schema.dynamic)
                            else:
                                dfs[schema.name] = src_adapter.df_from_fields(dynamic=schema.dynamic)
                        else:
                            port = obj._get_output_port(schema.name)
                            ds = self._build_df(odps, port, obj_dict, schema, exec_node=node)
                            adapter_from_df(ds)._direct_copy = schema.direct_copy
                            dfs[schema.name] = ds
                    obj._dfs = dfs
                if port_def.model and hasattr(port_def.model, 'copy_params') and port_def.model.copy_params:
                    pdict = dict((self._parameters[pn].name, self._parameters[pn].value)
                                 for pn in port_def.model.copy_params.split(','))
                    obj._params.update(pdict)
                out_objs[port_param_name] = obj

            if port_def.exporter:
                method = import_class_member(port_def.exporter)
                out_objs[port_param_name] = method(out_objs[port_param_name])

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
        Train a data set.
        The label field is specified by the ``label_field`` method.

        :param train_data: Data set to be trained. Label field must be specified.
        :type train_data: DataFrame

        :return: Trained model
        :rtype: MLModel
        """
        objs = self._do_transform(*args, **kwargs)
        obj_list = [objs, ] if not isinstance(objs, Iterable) else objs
        for obj in obj_list:
            if not isinstance(obj, MLModel):
                continue
            for meta in ['predictor', 'recommender']:
                if meta not in self._metas:
                    continue
                mod = __import__(self.__class__.__module__.__name__, fromlist=[''])\
                    if not hasattr(self, '_env') else self._env
                action_cls_name = underline_to_capitalized(self._metas[meta])
                if not hasattr(mod, action_cls_name):
                    action_cls_name = '_' + action_cls_name
                setattr(obj, meta, mod + '.' + action_cls_name)

        return objs


class BaseMetricsAlgorithm(BaseAlgorithm):
    def calc(self, *args, **kwargs):
        """
        :type args: list[DataFrame]
        """
        cases = kwargs.pop('_cases', [])
        if not isinstance(cases, Iterable):
            cases = [cases, ]

        node = AlgoMetricsNode(self._name, self._parameters, self._ports, self._metas)
        if cases:
            node.cases = cases

        obj_dict = self._map_inputs_from_args(node, *args, **kwargs)
        engine_kw = dict(p for p in six.iteritems(kwargs) if p[0] not in obj_dict)
        node.engine_args.update(engine_kw)

        RunnerContext.instance()._run(node)
        return node._sink

_FIELD_PARAM_PLACEHOLDER = re.compile(r'\{([^\}]+)\}')
_REDUNDANT_COMMA = re.compile(r',\s*,')


def _static_fields_generator(params, fields, algorithm, schema):

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
                    input_fields = fields[cmd[1]]
                    return ','.join(format_col(col, typ, **format_args) for col, typ in six.iteritems(input_fields))
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
                input_fields = fields[input_name]
                cols = cols or []

                return ','.join(format_col(col, input_fields[col], **format_args) for col in cols)
        return _REDUNDANT_COMMA.sub(',', _FIELD_PARAM_PLACEHOLDER.sub(_sub_func, schema))
