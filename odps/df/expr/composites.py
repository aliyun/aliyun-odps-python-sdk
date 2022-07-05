#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from .expressions import Scalar, Expr, Schema, SequenceExpr, ListSequenceExpr, DictSequenceExpr, \
    ListScalar, DictScalar, Column
from .element import AnyOp, ElementWise
from .collections import RowAppliedCollectionExpr
from .. import types as df_types
from ..utils import to_collection
from . import utils


def _scalar(val, tp=None):
    if val is None:
        return
    if isinstance(val, Expr):
        return val
    if isinstance(val, (tuple, list)):
        return type(val)(_scalar(it, tp=tp) for it in val)
    else:
        return Scalar(_value=val, _value_type=tp)


def explode(expr, *args, **kwargs):
    """
    Expand list or dict data into multiple rows

    :param expr: list / dict sequence / scalar
    :return:
    """
    if not isinstance(expr, Column):
        expr = to_collection(expr)[expr.name]

    if isinstance(expr, SequenceExpr):
        dtype = expr.data_type
    else:
        dtype = expr.value_type

    func_name = 'EXPLODE'
    if args and isinstance(args[0], (list, tuple, set)):
        names = list(args[0])
    else:
        names = args
    pos = kwargs.get('pos', False)
    if isinstance(expr, ListSequenceExpr):
        if pos:
            func_name = 'POSEXPLODE'
            typos = [df_types.int64, dtype.value_type]
            if not names:
                names = [expr.name + '_pos', expr.name]
            if len(names) == 1:
                names = [names[0] + '_pos', names[0]]
            if len(names) != 2:
                raise ValueError("The length of parameter 'names' should be exactly 1.")
        else:
            typos = [dtype.value_type]
            if not names:
                names = [expr.name]
            if len(names) != 1:
                raise ValueError("The length of parameter 'names' should be exactly 1.")
    elif isinstance(expr, DictSequenceExpr):
        if pos:
            raise ValueError('Cannot support explosion with pos on dicts.')
        typos = [dtype.key_type, dtype.value_type]
        if not names:
            names = [expr.name + '_key', expr.name + '_value']
        if len(names) != 2:
            raise ValueError("The length of parameter 'names' should be exactly 2.")
    else:
        raise ValueError('Cannot explode expression with type %s' % type(expr).__name__)
    schema = Schema.from_lists(names, typos)
    return RowAppliedCollectionExpr(_input=expr.input, _func=func_name, _schema=schema,
                                    _fields=[expr], _keep_nulls=kwargs.get('keep_nulls', False))


def composite_op(expr, output_expr_cls, output_type=None, **kwargs):
    input_args = kwargs.copy()
    input_args['_input'] = expr

    def check_is_sequence(arg):
        a = input_args.get(arg)
        if isinstance(a, SequenceExpr):
            return True
        elif isinstance(a, (list, tuple)):
            return any(isinstance(la, SequenceExpr) for la in a)
        else:
            return False

    is_sequence = any(check_is_sequence(a) for a in output_expr_cls._args)

    if is_sequence:
        output_type = output_type or expr.data_type
        return output_expr_cls(_data_type=output_type, _input=expr, **kwargs)
    else:
        output_type = output_type or expr.value_type
        return output_expr_cls(_value_type=output_type, _input=expr, **kwargs)


class CompositeOp(ElementWise):
    def accept(self, visitor):
        visitor.visit_composite_op(self)


class CompositeBuilderOp(AnyOp):
    @property
    def node_name(self):
        return self.__class__.__name__

    def accept(self, visitor):
        visitor.visit_composite_op(self)


class ListDictLength(CompositeOp):
    __slots__ = ()


class ListDictGetItem(CompositeOp):
    _args = '_input', '_key', '_negative_handled'


class ListContains(CompositeOp):
    _args = '_input', '_value',


class ListSort(CompositeOp):
    __slots__ = ()


class DictKeys(CompositeOp):
    __slots__ = ()


class DictValues(CompositeOp):
    __slots__ = ()


def _scan_inputs(seq, dtype=None):
    if not seq:
        raise TypeError('Inputs should not be empty')

    seq = [_scalar(a) for a in seq]

    arg_types = set()
    arg_cat = set()
    for a in seq:
        if isinstance(a, SequenceExpr):
            arg_types.add(a.data_type)
            arg_cat.add(SequenceExpr)
        else:
            arg_types.add(a.value_type)
            arg_cat.add(Scalar)

    if dtype is not None:
        if not all(dtype.can_implicit_cast(t) for t in arg_types):
            raise TypeError('Not all given value can be implicitly casted')
    else:
        if len(arg_types) == 1:
            dtype = arg_types.pop()
            if isinstance(dtype, df_types.Integer):
                if dtype != df_types.int64 and df_types.int32.can_implicit_cast(dtype):
                    dtype = df_types.int32
            elif isinstance(dtype, df_types.Float):
                dtype = df_types.float64
        else:
            if all(df_types.int32.can_implicit_cast(t) for t in arg_types):
                dtype = df_types.int32
            elif all(df_types.int64.can_implicit_cast(t) for t in arg_types):
                dtype = df_types.int64
            elif all(df_types.float64.can_implicit_cast(t) for t in arg_types):
                dtype = df_types.float64
            else:
                raise TypeError('Types of inputs should be the same')
    if SequenceExpr in arg_cat:
        return seq, '_data_type', dtype
    else:
        return seq, '_value_type', dtype


class ListBuilder(CompositeBuilderOp):
    _args = '_values',


class DictBuilder(CompositeBuilderOp):
    _args = '_keys', '_values'


def _len(expr):
    """
    Retrieve length of a list or dict sequence / scalar.

    :param expr: list or dict sequence / scalar
    :return:
    """
    return composite_op(expr, ListDictLength, df_types.int64)


def _getitem(expr, key):
    if isinstance(expr, SequenceExpr):
        dtype = expr.data_type.value_type
    else:
        dtype = expr.value_type.value_type
    return composite_op(expr, ListDictGetItem, dtype, _key=_scalar(key))


def _sort(expr):
    """
    Retrieve sorted list

    :param expr: list sequence / scalar
    :return:
    """
    return composite_op(expr, ListSort)


def _contains(expr, value):
    """
    Check whether certain value is in the inspected list

    :param expr: list sequence / scalar
    :param value: value to inspect
    :return:
    """
    return composite_op(expr, ListContains, df_types.boolean, _value=_scalar(value))


def _keys(expr):
    """
    Retrieve keys of a dict

    :param expr: dict sequence / scalar
    :return:
    """
    if isinstance(expr, SequenceExpr):
        dtype = expr.data_type
    else:
        dtype = expr.value_type
    return composite_op(expr, DictKeys, df_types.List(dtype.key_type))


def _values(expr):
    """
    Retrieve values of a dict

    :param expr: dict sequence / scalar
    :return:
    """
    if isinstance(expr, SequenceExpr):
        dtype = expr.data_type
    else:
        dtype = expr.value_type
    return composite_op(expr, DictValues, df_types.List(dtype.value_type))


def make_list(*args, **kwargs):
    dtype = kwargs.get('type')
    if dtype is not None:
        dtype = df_types.validate_data_type(dtype)

    kwargs = dict()
    kwargs['_values'], k, typ = _scan_inputs(args, dtype)
    kwargs[k] = df_types.List(typ)

    return ListBuilder(**kwargs)


def make_dict(*args, **kwargs):
    if len(args) % 2 != 0:
        raise ValueError('Num of inputs to build a dict should be even')

    key_type = kwargs.get('key_type')
    if key_type is not None:
        key_type = df_types.validate_data_type(key_type)

    value_type = kwargs.get('value_type')
    if value_type is not None:
        value_type = df_types.validate_data_type(value_type)

    kwargs = dict()
    keys = list(args[0::2])
    values = list(args[1::2])

    kwargs['_keys'], k1, key_type = _scan_inputs(keys, key_type)
    kwargs['_values'], k2, value_type = _scan_inputs(values, value_type)

    k = '_data_type' if '_data_type' in (k1, k2) else '_value_type'
    kwargs[k] = df_types.Dict(key_type, value_type)

    return DictBuilder(**kwargs)


_list_methods = dict(
    __getitem__=_getitem,
    len=_len,
    sort=_sort,
    contains=_contains,
    explode=explode,
)

_dict_methods = dict(
    __getitem__=_getitem,
    len=_len,
    keys=_keys,
    values=_values,
    explode=explode,
)

utils.add_method(ListSequenceExpr, _list_methods)
utils.add_method(ListScalar, _list_methods)
utils.add_method(DictSequenceExpr, _dict_methods)
utils.add_method(DictScalar, _dict_methods)
