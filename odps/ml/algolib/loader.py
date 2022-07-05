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

import re
import textwrap
from functools import partial

from .objects import AlgorithmsDef, AlgorithmDef
from ..expr import AlgoCollectionExpr, ODPSModelExpr, MetricsResultExpr
from ..enums import PortType
from ..utils import import_class_member, get_function_args, ML_ARG_PREFIX
from ...compat import OrderedDict, six, Iterable
from ...df.expr.collections import SequenceExpr
from ...utils import camel_to_underline, underline_to_capitalized, load_static_text_file,\
    load_internal_static_text_file, survey

SYSTEM_PARAMS = set("""
isInheritLastType inputTableName inputTablePartitions outputTableName outputTablePartition
""".strip().split())
PREDICTOR_PARAMS = set("""
inputTableName inputPartitions modelTableName outputTableName outputPartition appendColNames
""".strip().split())


def dispatch_args(param_names, *args, **kwargs):
    from .base_algo import is_ml_object
    ml_args = [a for a in args if is_ml_object(a)]
    p_args = [a for a in args if not is_ml_object(a)]
    ml_kw = OrderedDict([(k, v) for k, v in six.iteritems(kwargs)
                         if k not in param_names or k.startswith('_')])
    p_kw = OrderedDict([(k, v) for k, v in six.iteritems(kwargs) if k not in ml_kw and k in param_names])
    return ml_args, p_args, ml_kw, p_kw


def _generate_class_ctor(nested_cls, algo_def):
    for param in algo_def.params:
        if not param.value:
            param.value = None
    metas = OrderedDict([(meta.name, meta.value) for meta in algo_def.metas])
    ports = OrderedDict([(port.name, port) for port in algo_def.ports])

    def _ctor(self, *args, **kwargs):
        def _fill_param(param_obj):
            param_obj = param_obj.copy()
            arg_name = param_obj.friendly_name
            if arg_name in kwargs:
                arg_val = kwargs[arg_name]
                if isinstance(arg_val, SequenceExpr):
                    param_obj.value = arg_val.name
                else:
                    param_obj.value = arg_val
            elif param_obj.sequence is not None and param_obj.sequence <= len(args):
                param_obj.value = args[param_obj.sequence - 1]
            return param_obj

        parameters = OrderedDict([(param.name, _fill_param(param)) for param in algo_def.params])

        sequences = set([p.sequence for p in six.itervalues(parameters)])
        for idx, obj in enumerate(args):
            if (idx + 1) not in sequences:
                raise ValueError('Unexpected sequential argument: {0}.'.format(obj))
        names = set([p.friendly_name for p in six.itervalues(parameters)])
        for k, v in six.iteritems(kwargs):
            if k.startswith('_'):  # predefined args
                continue
            if k not in names:
                raise ValueError('Unexpected keyword argument: {0}={1}.'.format(k, v))

        self._reload_fields = algo_def.reload_fields

        super(nested_cls, self).__init__(algo_def.code_name, parameters, ports, metas)

    if metas.get('survey') == 'true':
        return survey(_ctor)
    return _ctor


def _iter_param_cmp_table(params):
    param_pairs = [(camel_to_underline(p.name), p.name) for p in params if p.name not in SYSTEM_PARAMS]
    method_header_len = len(max(param_pairs, key=lambda p: len(p[0]))[0]) + 1
    param_header_len = len(max(param_pairs, key=lambda p: len(p[1]))[1]) + 1

    sep_line = ' ' * 4 + '=' * method_header_len + ' ' + '=' * param_header_len
    yield sep_line
    yield ' ' * 4 + 'SDK' + ' ' * (method_header_len - len('SDK') + 1) + 'PAI Command'
    yield sep_line

    for method_name, param_name in param_pairs:
        yield ' ' * 4 + method_name + ' ' * (method_header_len - len(method_name) + 1) + param_name
    yield sep_line


def _generate_prop(param_name):
    def _getter(self):
        return self._parameters[param_name].value

    def _setter(self, value):
        if isinstance(value, SequenceExpr):
            value = value.name
        self._parameters[param_name].value = value
        self._invoke_setter(param_name, value)

    return property(_getter, _setter)


def _generate_prop_setter(param_name):
    def _setter(self, value):
        if isinstance(value, SequenceExpr):
            value = value.name
        self._parameters[param_name].value = value
        self._invoke_setter(param_name, value)
        return self

    return _setter


def _generate_algo_func(cls, node_def, func_name):
    param_names = set(p.friendly_name for p in node_def.params)

    def _func(*args, **kwargs):
        ml_args, p_args, ml_kw, p_kw = dispatch_args(param_names, *args, **kwargs)
        algo_obj = cls(*p_args, **p_kw)
        return getattr(algo_obj, func_name)(*ml_args, **ml_kw)

    _func.__name__ = ('' if node_def.public else '_') + camel_to_underline(cls.__name__.lstrip('_'))
    _func.__doc__ = _build_function_docstring(node_def)
    return _func


def _is_param_in_doc(param_def):
    pname = param_def.name
    if pname in SYSTEM_PARAMS:
        return False
    if pname.endswith('TableName') or pname.endswith('Partition') or pname.endswith('Partitions'):
        return False
    if pname.endswith('Name') and 'Table' in pname:
        return False
    if pname == 'modelName':
        return False
    return True


def _get_port_type(port_def, req=False, sphinx_wrap=False):
    if port_def.type == PortType.DATA:
        data_type = 'DataFrame'
    elif port_def.type == PortType.MODEL:
        if not port_def.model or not port_def.model.type:
            data_type = 'PmmlModel'
        else:
            data_type = port_def.model.type
    else:
        return None
    if sphinx_wrap:
        data_type = ':class:`%s`' % data_type
    if req and port_def.required:
        return data_type + ', required'
    else:
        return data_type


def _build_node_def_params(node_def, filter_func=None):
    filter_func = filter_func or (lambda p: True)
    param_docs = []
    for p in node_def.params:
        if not filter_func(p):
            continue
        if not _is_param_in_doc(p):
            continue
        if p.internal:
            continue
        if p.docs:
            param_docs.append(six.text_type('    - **{param_name}** - {docs}').format(
                param_name=p.friendly_name, docs=re.sub(r'\n *', ' ', p.docs))
            )
        else:
            param_docs.append('    - **%s**' % p.friendly_name)
    if param_docs:
        return '\n'.join(param_docs) + '\n'
    else:
        return ''


def _filter_predictor_params(p):
    if p.name in PREDICTOR_PARAMS:
        return False
    if p.exporter and 'get_input_model_param' in p.exporter:
        return False
    return True


def _build_class_docstring(node_def_dict, code_name):
    node_def = node_def_dict[code_name]
    if node_def.public is not None and not node_def.public:
        return ''
    docs = six.text_type(node_def.docs) if node_def.docs else six.text_type('\n%params%\n%predictor_params%\n')
    docs = textwrap.dedent(docs)
    docs = docs.replace('%params%', ':Parameters:\n' + _build_node_def_params(node_def))

    predictor_code_name = node_def.meta_dict.get('predictor')
    if predictor_code_name:
        predictor_node = node_def_dict[predictor_code_name]
        predictor_input_docs = []
        for p in predictor_node.get_input_ports():
            if p.name == 'model':
                continue
            if p.docs:
                predictor_input_docs.append(six.text_type('    - **{param_name}** ({param_type}) - {docs}').format(
                    param_type=_get_port_type(p, sphinx_wrap=True),
                    param_name=camel_to_underline(p.name),
                    docs=re.sub(r'\n *', ' ', p.docs),
                ))
            else:
                predictor_input_docs.append(six.text_type('    - **{param_name}** ({param_type})').format(
                    param_type=_get_port_type(p, sphinx_wrap=True),
                    param_name=camel_to_underline(p.name),
                ))
        predictor_docs = ':Predictor Parameters:\n' + \
                         '\n'.join(predictor_input_docs) + '\n' +\
                         _build_node_def_params(predictor_node, _filter_predictor_params)
        docs = docs.replace('%predictor_params%', predictor_docs)
    else:
        docs = docs.replace('%predictor_params%', '')
    return docs


def _build_entry_method_docs(node_def):
    docs = six.text_type(node_def.entry_docs) if node_def.entry_docs else six.text_type('\n%inputs%\n%outputs%\n')
    if '%inputs%' in docs:
        input_docs = [six.text_type(':param %s %s: %s') % (_get_port_type(p), camel_to_underline(p.name), re.sub(r'\n *', ' ', p.docs))
                      if p.docs else six.text_type(':param %s %s:') % (_get_port_type(p), camel_to_underline(p.name))
                      for p in node_def.get_input_ports()]
        if input_docs:
            docs = docs.replace('%inputs%', six.text_type('.. py:currentmodule:: odps.ml\n\n') + '\n'.join(input_docs) + '\n\n')
        else:
            docs = docs.replace('%inputs%', '')
    if '%outputs%' in docs:
        output_ports = node_def.get_output_ports()
        if len(output_ports) == 1:
            if output_ports[0].docs:
                output_docs = '\n\n:return: %s\n:rtype: %s\n\n' % (output_ports[0].docs, _get_port_type(output_ports[0]))
            else:
                output_docs = '\n\n:rtype: %s\n\n' % _get_port_type(output_ports[0])
            docs = docs.replace('%outputs%', output_docs)
        else:
            output_names = [camel_to_underline(p.name) for p in output_ports]
            output_types = [_get_port_type(p) for p in output_ports]
            output_descs = [u'    * %s – %s' % (camel_to_underline(p.name), p.docs) for p in output_ports if p.docs]

            if output_descs:
                output_docs = six.text_type('\n\n:Returns:\n    * (%s)\n%s\n:rtype: (%s)\n\n') % (', '.join(output_names), '\n'.join(output_descs), ', '.join(output_types))
            else:
                output_docs = six.text_type('\n\n:Returns: (%s)\n:rtype: (%s)\n\n') % (', '.join(output_names), ', '.join(output_types))
            docs = docs.replace('%outputs%', output_docs)
    return docs


def _build_entry_method(class_obj, node_def):
    def _entry_method(self, *args, **kwargs):
        return getattr(super(class_obj, self), class_obj._entry_method)(*args, **kwargs)

    _entry_method.__name__ = class_obj._entry_method
    _entry_method.__doc__ = _build_entry_method_docs(node_def)
    _entry_method.__dict__.update(getattr(class_obj, class_obj._entry_method).__dict__)
    return _entry_method


def _build_function_docstring(node_def):
    if node_def.public is not None and not node_def.public:
        return ""
    docs = six.text_type(node_def.docs) if node_def.docs else six.text_type('\n%params%\n')
    docs = textwrap.dedent(docs)
    if '%params%' in docs:
        arg_str = ''
        input_docs = [six.text_type(':param %s %s: %s') % (_get_port_type(p, False, False), camel_to_underline(p.name), re.sub(r'\n *', ' ', p.docs))
                      if p.docs else six.text_type(':param %s %s: %s') % (_get_port_type(p, False, False), camel_to_underline(p.name), camel_to_underline(p.name))
                      for p in node_def.get_input_ports()]
        arg_str += '\n'.join(input_docs) + '\n'
        param_docs = [six.text_type(':param %s: %s') % (p.friendly_name, re.sub(r'\n *', ' ', p.docs))
                      if p.docs else ':param %s: %s' % (p.friendly_name, p.friendly_name)
                      for p in node_def.params if _is_param_in_doc(p) and not p.internal]
        arg_str += '\n'.join(param_docs)
        if arg_str:
            docs = docs.replace('%params%', six.text_type('.. py:currentmodule:: odps.ml\n\n') + arg_str + '\n\n')
        else:
            docs = docs.replace('%params%', '\n\n')
    return docs

_PARAMED_EXPORTER_REGEX = re.compile(r'([^\(]+)\(([^\)]+)\) *')


def _build_expr(node_def, base_class):
    expr_name = underline_to_capitalized(node_def.code_name)
    input_names = tuple([ML_ARG_PREFIX + p.name for p in node_def.get_input_ports()])
    exported = set(p.name for p in node_def.params if p.exported)
    metas = OrderedDict([(meta.name, meta.value) for meta in node_def.metas])

    from ..expr import exporters
    exporter_dict = dict()
    for param in (p for p in node_def.params if p.exporter is not None):
        def fetch_exporter_func(func_name):
            match = _PARAMED_EXPORTER_REGEX.match(func_name)
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
            exporter_fun_name = exporter_name
            if '.' in exporter_fun_name:
                _, exporter_fun_name = exporter_fun_name.rsplit('.', 1)
            fun = lambda self: func(self, **args_dict)
            fun.__name__ = exporter_fun_name
            return fun

        exporter_dict[param.name] = make_exporter_wrapper(param.exporter, args)

    class_dict = {
        'node_name': node_def.code_name,
        '_algo': node_def.code_name,
        '_args': input_names,
        '_exported': exported,
        '_exporters': exporter_dict,
        'input_ports': node_def.get_input_ports(),
        'output_ports': node_def.get_output_ports(),
        'algo_meta': metas,
    }
    return type(base_class)(expr_name + base_class._suffix, (base_class, ), class_dict)


def _generate_objects(def_list):
    classes = []
    node_def_dict = dict((d.code_name, d) for d in def_list)
    for node_def in (d for d in def_list if d.enabled is None or d.enabled):
        class_name = underline_to_capitalized(node_def.code_name)
        if node_def.public is not None and not node_def.public:
            class_name = '_' + class_name
        base_class = import_class_member('$package_root.algolib.base_algo.' + node_def.base_class)

        if not node_def.export_function:
            doc_str = _build_class_docstring(node_def_dict, node_def.code_name)
        else:
            doc_str = None
        new_cls = type(class_name, (base_class, ), {'__doc__': doc_str})

        setattr(new_cls, '__init__', _generate_class_ctor(new_cls, node_def))
        for param in (p for p in node_def.params if p.name not in SYSTEM_PARAMS):
            user_param_name = param.friendly_name
            setattr(new_cls, user_param_name, _generate_prop(param.name))
            setattr(new_cls, 'set_' + user_param_name, _generate_prop_setter(param.name))

        if new_cls._entry_method:
            setattr(new_cls, new_cls._entry_method, _build_entry_method(new_cls, node_def))

        new_cls._collection_expr = _build_expr(node_def, AlgoCollectionExpr)
        new_cls._model_expr = _build_expr(node_def, ODPSModelExpr)
        if base_class.__name__ == 'BaseMetricsAlgorithm':
            new_cls._metrics_expr = _build_expr(node_def, MetricsResultExpr)

        classes.append(new_cls)

        if node_def.export_function:
            if base_class.__name__ == 'BaseMetricsAlgorithm':
                func_name = 'calc'
            else:
                func_name = 'transform'
            classes.append(_generate_algo_func(new_cls, node_def, func_name))

    return classes


def load_algorithms(json_obj, base_class, env):
    def_list = [def_obj if isinstance(def_obj, AlgorithmDef) else AlgorithmDef.parse(def_obj)
                for def_obj in json_obj]

    for def_obj in def_list:
        if not def_obj.base_class:
            def_obj.base_class = base_class

    objects = sorted(_generate_objects(def_list), key=lambda o: o.__name__.lower())
    for algo_obj in objects:
        algo_obj.__module__ = env.__name__
        setattr(algo_obj, '_env', env.__name__)
        setattr(env, algo_obj.__name__, algo_obj)
    names = [c.__name__ for c in objects]

    # add names into __all__ variable for sphinx docs
    if not hasattr(env, '__all__'):
        setattr(env, '__all__', [])
    existing = set(getattr(env, '__all__')) - set(["Enum", ])
    setattr(env, '__all__', list(existing) + list(filter(lambda v: v not in existing and not v.startswith('_'), names)))


def load_defined_algorithms(env, name):
    algos_all = []
    external_xml = load_static_text_file('algorithms/' + name + '.xml')
    base_class = None
    if external_xml:
        algos = AlgorithmsDef.parse(external_xml)
        base_class = algos.base_class
        for algo in (a for a in algos.algorithms if not a.base_class):
            algo.base_class = algos.base_class
        algos_all.extend(algos.algorithms)

    internal_xml = load_internal_static_text_file('algorithms/' + name + '.xml')
    if internal_xml:
        algos = AlgorithmsDef.parse(internal_xml)
        base_class = algos.base_class
        for algo in (a for a in algos.algorithms if not a.base_class):
            algo.base_class = algos.base_class
        algos_all.extend(algos.algorithms)

    load_algorithms(algos_all, base_class, env)


def load_classifiers(algo_defs, env):
    """
    Load an algorithm into a module. The algorithm is an instance of ``AlgorithmDef`` class.

    :param algo_defs: algorithm definitions
    :type algo_defs: AlgorithmDef | list[AlgorithmDef]
    :param env: environment

    :Example:
    >>> import sys
    >>> from odps.ml.algolib.loader import *
    >>> a = XflowAlgorithmDef('SampleAlgorithm')
    >>> a.add_param(ParamDef('param1', 'val1'))
    >>> a.add_port(PortDef('input'))
    >>> a.add_port(PortDef('output', PortDirection.OUTPUT))
    >>> load_classifiers(a, sys.modules[__name__])
    """
    if not isinstance(algo_defs, Iterable):
        algo_defs = [algo_defs, ]
    load_algorithms(algo_defs, 'BaseTrainingAlgorithm', env)


def load_process_algorithm(algo_defs, env):
    """
    Load an algorithm into a module. The algorithm is an instance of ``AlgorithmDef`` class.

    :param algo_defs: algorithm definitions
    :type algo_defs: AlgorithmDef | list[AlgorithmDef]
    :param env: environment

    :Example:
    >>> import sys
    >>> from odps.ml.algolib.loader import *
    >>> a = XflowAlgorithmDef('SampleAlgorithm')
    >>> a.add_param(ParamDef('param1', 'val1'))
    >>> a.add_port(PortDef('input'))
    >>> a.add_port(PortDef('output', PortDirection.OUTPUT))
    >>> load_process_algorithm(a, sys.modules[__name__])
    """
    if not isinstance(algo_defs, Iterable):
        algo_defs = [algo_defs, ]
    load_algorithms(algo_defs, 'BaseProcessAlgorithm', env)
