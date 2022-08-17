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

import inspect

from .expressions import TypedExpr, SequenceExpr, Scalar, \
    BooleanSequenceExpr, BooleanScalar, CollectionExpr, Expr, \
    int_number_sequences, int_number_scalars
from .core import NodeMetaclass
from . import utils
from . import errors
from .. import types
from ..utils import FunctionWrapper
from ...compat import six


class AnyOpNodeMetaClass(NodeMetaclass):
    def __new__(mcs, name, bases, kv):
        if '_add_args_slots' not in kv:
            kv['_add_args_slots'] = False

        return super(AnyOpNodeMetaClass, mcs).__new__(mcs, name, bases, kv)


class AnyOp(six.with_metaclass(AnyOpNodeMetaClass, TypedExpr)):
    __slots__ = ()

    @classmethod
    def _get_type(cls, *args, **kwargs):
        if '_data_type' in kwargs:
            return SequenceExpr._get_type(cls, *args, **kwargs)
        else:
            return Scalar._get_type(cls, *args, **kwargs)

    @classmethod
    def _typed_classes(cls, *args, **kwargs):
        if '_data_type' in kwargs:
            return SequenceExpr._typed_classes(cls, *args, **kwargs)
        else:
            return Scalar._typed_classes(cls, *args, **kwargs)

    @classmethod
    def _base_class(cls, *args, **kwargs):
        if '_data_type' in kwargs:
            return SequenceExpr
        else:
            return Scalar

    @classmethod
    def is_seq(cls):
        return issubclass(cls, SequenceExpr)


class ElementWise(AnyOp):
    __slots__ = ()
    _args = '_input',

    @property
    def node_name(self):
        return self.__class__.__name__

    @property
    def name(self):
        return self._name or self._input.name

    @property
    def input(self):
        return self._input

    def iter_args(self):
        for it in zip(['_input'] + [arg.lstrip('_') for arg in self._args[1:]],
                      self.args):
            yield it


class ElementOp(ElementWise):
    __slots__ = ()

    def accept(self, visitor):
        return visitor.visit_element_op(self)


class MappedExpr(ElementWise):
    _slots = '_func', '_func_args', '_func_kwargs', '_resources', \
             '_multiple', '_raw_inputs'
    _args = '_inputs', '_collection_resources'
    node_name = 'Map'

    def _init(self, *args, **kwargs):
        self._init_attr('_multiple', False)
        self._init_attr('_raw_inputs', None)
        super(MappedExpr, self)._init(*args, **kwargs)

    @property
    def inputs(self):
        return self._inputs

    @property
    def name(self):
        if self._name is not None:
            return self._name
        if len(self._inputs) == 1:
            return self._inputs[0].name

    @property
    def source_name(self):
        if self._source_name is not None:
            return self._source_name
        if len(self._inputs) == 1:
            return self._inputs[0].source_name

    @property
    def input_types(self):
        return [it.dtype for it in self._inputs]

    @property
    def raw_input_types(self):
        if self._raw_inputs:
            return [it.dtype for it in self._raw_inputs]
        return self.input_types

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, f):
        self._func = f

    def accept(self, visitor):
        return visitor.visit_function(self)


class IsNull(ElementOp):
    __slots__ = ()


class NotNull(ElementOp):
    __slots__ = ()


class FillNa(ElementOp):
    _args = '_input', '_fill_value'

    def _init(self, *args, **kwargs):
        self._init_attr('_fill_value', None)

        super(FillNa, self)._init(*args, **kwargs)

        if self._fill_value is not None and not isinstance(self._fill_value, Expr):
            tp = types.validate_value_type(self._fill_value)
            if not self.input.dtype.can_implicit_cast(tp):
                raise ValueError('fillna cannot cast value from %s to %s' % (
                    tp, self.input.dtype))
            self._fill_value = Scalar(_value=self._fill_value, _value_type=self.dtype)
        if not self.input.dtype.can_implicit_cast(self._fill_value.dtype):
            raise ValueError('fillna cannot cast value from %s to %s' % (
                self._fill_value.dtype, self.input.dtype))

    @property
    def fill_value(self):
        if self._fill_value is None:
            return
        if isinstance(self._fill_value, Scalar):
            return self._fill_value.value
        return self._fill_value


class IsIn(ElementOp):
    _args = '_input', '_values',

    def _init(self, *args, **kwargs):
        super(IsIn, self)._init(*args, **kwargs)

        self._values = _scalar(self._values, tp=self._input.dtype)
        if isinstance(self._values, list):
            self._values = tuple(self._values)
        if not isinstance(self._values, tuple):
            if not isinstance(self._values, SequenceExpr):
                raise ValueError('isin accept iterable object or sequence')
            self._values = (self._values, )

    @property
    def name(self):
        return self._name


class NotIn(ElementOp):
    _args = '_input', '_values',

    def _init(self, *args, **kwargs):
        super(NotIn, self)._init(*args, **kwargs)

        self._values = _scalar(self._values, tp=self._input.dtype)
        if isinstance(self._values, list):
            self._values = tuple(self._values)
        if not isinstance(self._values, tuple):
            if not isinstance(self._values, SequenceExpr):
                raise ValueError('notin accept iterable object or sequence')
            self._values = (self._values, )

    @property
    def name(self):
        return self._name


class Between(ElementOp):
    _args = '_input', '_left', '_right', '_inclusive'

    def _init(self, *args, **kwargs):
        self._init_attr('_left', None)
        self._init_attr('_right', None)
        self._init_attr('_inclusive', None)

        super(Between, self)._init(*args, **kwargs)

        for attr in self._args[1:]:
            val = getattr(self, attr)
            if val is not None and not isinstance(val, Expr):
                setattr(self, attr, Scalar(_value=val))

    def _get_val(self, attr):
        val = getattr(self, attr)
        if val is None:
            return
        if isinstance(val, Scalar):
            return val.value
        return val

    @property
    def left(self):
        return self._get_val('_left')

    @property
    def right(self):
        return self._get_val('_right')

    @property
    def inclusive(self):
        return self._get_val('_inclusive')


class IfElse(ElementOp):
    _args = '_input', '_then', '_else'

    @property
    def name(self):
        return self._name


class Switch(ElementOp):
    _args = '_input', '_case', '_conditions', '_thens', '_default'

    def iter_args(self):
        def _names():
            if hasattr(self, '_case'):
                yield 'case'
            for _ in self._conditions:
                yield 'when'
                yield 'then'
            yield 'default'

        def _args():
            if hasattr(self, '_case'):
                yield self._case
            for condition, then in zip(self._conditions, self._thens):
                yield condition
                yield then
            yield self._default

        for it in zip(_names(), _args()):
            yield it

    @property
    def name(self):
        return self._name


class Cut(ElementOp):
    _args = '_input', '_bins', '_right', '_labels', '_include_lowest', \
            '_include_under', '_include_over'

    def _init(self, *args, **kwargs):
        super(Cut, self)._init(*args, **kwargs)

        if len(self._bins) == 0:
            raise ValueError('Must be at least one bin edge')
        elif len(self._bins) == 1:
            if not self._include_under or not self._include_over:
                raise ValueError('If one bin edge provided, must have'
                                 ' include_under=True and include_over=True')

        for arg in self._args[1:]:
            obj = getattr(self, arg)
            setattr(self, arg, _scalar(obj))

        under = 1 if self._include_under.value else 0
        over = 1 if self._include_over.value else 0
        size = len(self._bins) - 1 + under + over

        if self._labels is not None and len(self._labels) != size:
            raise ValueError('Labels size must be exactly the size of bins')
        if self._labels is None:
            self._labels = _scalar(list(range(size)))

    @property
    def right(self):
        return self._right.value

    @property
    def include_lowest(self):
        return self._include_lowest.value

    @property
    def include_under(self):
        return self._include_under.value

    @property
    def include_over(self):
        return self._include_over.value

    @property
    def name(self):
        return self._name


class IntToDatetime(ElementOp):
    __slots__ = ()


class Func(AnyOp):
    _slots = '_func_name', '_rtype'
    _args = '_inputs',

    def accept(self, visitor):
        visitor.visit_function(self)


class FuncFactory(object):
    def __getattr__(self, item):
        if not isinstance(item, six.string_types):
            raise TypeError('Function name should be provided, expect str, got %s' % type(item))

        def gen_func(*args, **kwargs):
            func_name = item
            project = kwargs.pop('project', None)
            if project:
                func_name = '%s:%s' % (project, item)

            expr_name = kwargs.pop('name', None)
            rtype = kwargs.pop('rtype', types.string)
            rtype = types.validate_data_type(rtype)
            is_seq = kwargs.pop('seq', None)

            args = tuple(arg if isinstance(arg, Expr) else Scalar(_value=arg) for arg in args)
            is_seq = is_seq if is_seq is not None else any(isinstance(arg, SequenceExpr) for arg in args)
            kw = {'_value_type': rtype} if not is_seq else {'_data_type': rtype}

            if expr_name is None:
                exprs = tuple(arg for arg in args if isinstance(arg, SequenceExpr))
                if len(exprs) == 1:
                    expr_name = exprs[0].name

            if expr_name:
                return Func(_func_name=func_name, _inputs=args, **kw).rename(expr_name)
            else:
                return Func(_func_name=func_name, _inputs=args, **kw)

        return gen_func


def _map(expr, func, rtype=None, resources=None, args=(), **kwargs):
    """
    Call func on each element of this sequence.

    :param func: lambda, function, :class:`odps.models.Function`,
                 or str which is the name of :class:`odps.models.Funtion`
    :param rtype: if not provided, will be the dtype of this sequence
    :return: a new sequence

    :Example:

    >>> df.id.map(lambda x: x + 1)
    """

    name = None
    if isinstance(func, FunctionWrapper):
        if func.output_names:
            if len(func.output_names) > 1:
                raise ValueError('Map column has more than one name')
            name = func.output_names[0]
        if func.output_types:
            rtype = rtype or func.output_types[0]
        func = func._func

    if rtype is None:
        rtype = utils.get_annotation_rtype(func)

    from ...models import Function

    rtype = rtype or expr.dtype
    output_type = types.validate_data_type(rtype)

    if isinstance(func, six.string_types):
        pass
    elif isinstance(func, Function):
        pass
    elif inspect.isclass(func):
        pass
    elif not callable(func):
        raise ValueError('`func` must be a function or a callable class')

    collection_resources = utils.get_collection_resources(resources)

    is_seq = isinstance(expr, SequenceExpr)
    if is_seq:
        return MappedExpr(_data_type=output_type, _func=func, _inputs=[expr, ],
                          _func_args=args, _func_kwargs=kwargs, _name=name,
                          _resources=resources, _collection_resources=collection_resources)
    else:
        return MappedExpr(_value_type=output_type, _func=func, _inputs=[expr, ],
                          _func_args=args, _func_kwargs=kwargs, _name=name,
                          _resources=resources, _collection_resources=collection_resources)


def _hash(expr, func=None):
    """
    Calculate the hash value.

    :param expr:
    :param func: hash function
    :return:
    """
    if func is None:
        func = lambda x: hash(x)

    return _map(expr, func=func, rtype=types.int64)


def _isnull(expr):
    """
    Return a sequence or scalar according to the input indicating if the values are null.

    :param expr: sequence or scalar
    :return: sequence or scalar
    """

    if isinstance(expr, SequenceExpr):
        return IsNull(_input=expr, _data_type=types.boolean)
    elif isinstance(expr, Scalar):
        return IsNull(_input=expr, _value_type=types.boolean)


def _notnull(expr):
    """
    Return a sequence or scalar according to the input indicating if the values are not null.

    :param expr: sequence or scalar
    :return: sequence or scalar
    """

    if isinstance(expr, SequenceExpr):
        return NotNull(_input=expr, _data_type=types.boolean)
    elif isinstance(expr, Scalar):
        return NotNull(_input=expr, _value_type=types.boolean)


def _fillna(expr, value):
    """
    Fill null with value.

    :param expr: sequence or scalar
    :param value: value to fill into
    :return: sequence or scalar
    """

    if isinstance(expr, SequenceExpr):
        return FillNa(_input=expr, _fill_value=value, _data_type=expr.dtype)
    elif isinstance(expr, Scalar):
        return FillNa(_input=expr, _fill_value=value, _value_type=expr.dtype)


def _isin(expr, values):
    """
    Return a boolean sequence or scalar showing whether
    each element is exactly contained in the passed `values`.

    :param expr: sequence or scalar
    :param values: `list` object or sequence
    :return: boolean sequence or scalar
    """

    from .merge import _make_different_sources

    if isinstance(values, SequenceExpr):
        expr, values = _make_different_sources(expr, values)

    if isinstance(expr, SequenceExpr):
        return IsIn(_input=expr, _values=values, _data_type=types.boolean)
    elif isinstance(expr, Scalar):
        return IsIn(_input=expr, _values=values, _value_type=types.boolean)


def _notin(expr, values):
    """
    Return a boolean sequence or scalar showing whether
    each element is not contained in the passed `values`.

    :param expr: sequence or scalar
    :param values: `list` object or sequence
    :return: boolean sequence or scalar
    """

    if isinstance(expr, SequenceExpr):
        return NotIn(_input=expr, _values=values, _data_type=types.boolean)
    elif isinstance(expr, Scalar):
        return NotIn(_input=expr, _values=values, _value_type=types.boolean)


def _between(expr, left, right, inclusive=True):
    """
    Return a boolean sequence or scalar show whether
    each element is between `left` and `right`.

    :param expr: sequence or scalar
    :param left: left value
    :param right: right value
    :param inclusive: if true, will be left <= expr <= right, else will be left < expr < right
    :return: boolean sequence or scalar
    """

    if isinstance(expr, SequenceExpr):
        return Between(_input=expr, _left=left, _right=right,
                       _inclusive=inclusive, _data_type=types.boolean)
    elif isinstance(expr, Scalar):
        return Between(_input=expr, _left=left, _right=right,
                       _inclusive=inclusive, _value_type=types.boolean)


def _scalar(val, tp=None):
    if val is None:
        return
    if isinstance(val, Expr):
        return val
    if isinstance(val, (tuple, list)):
        return type(val)(_scalar(it, tp=tp) for it in val)
    else:
        return Scalar(_value=val, _value_type=tp)


def _ifelse(expr, true_expr, false_expr):
    """
    Given a boolean sequence or scalar, if true will return the left, else return the right one.

    :param expr: sequence or scalar
    :param true_expr:
    :param false_expr:
    :return: sequence or scalar

    :Example:

    >>> (df.id == 3).ifelse(df.id, df.fid.astype('int'))
    >>> df.isMale.ifelse(df.male_count, df.female_count)
    """

    tps = (SequenceExpr, Scalar)
    if not isinstance(true_expr, tps):
        true_expr = Scalar(_value=true_expr)
    if not isinstance(false_expr, tps):
        false_expr = Scalar(_value=false_expr)

    output_type = utils.highest_precedence_data_type(
            *[true_expr.dtype, false_expr.dtype])
    is_sequence = isinstance(expr, SequenceExpr) or \
                  isinstance(true_expr, SequenceExpr) or \
                  isinstance(false_expr, SequenceExpr)

    if is_sequence:
        return IfElse(_input=expr, _then=true_expr, _else=false_expr,
                      _data_type=output_type)
    else:
        return IfElse(_input=expr, _then=true_expr, _else=false_expr,
                      _value_type=output_type)


def _switch(expr, *args, **kw):
    """
    Similar to the case-when in SQL. Refer to the example below

    :param expr:
    :param args:
    :param kw:
    :return: sequence or scalar

    :Example:

    >>> # if df.id == 3 then df.name
    >>> # elif df.id == df.fid.abs() then df.name + 'test'
    >>> # default: 'test'
    >>> df.id.switch(3, df.name, df.fid.abs(), df.name + 'test', default='test')
    """
    default = _scalar(kw.get('default'))

    if len(args) <= 0:
        raise errors.ExpressionError('Switch must accept more than one condition')

    if all(isinstance(arg, tuple) and len(arg) == 2 for arg in args):
        conditions, thens = [list(tp) for tp in zip(*args)]
    else:
        conditions = [arg for i, arg in enumerate(args) if i % 2 == 0]
        thens = [arg for i, arg in enumerate(args) if i % 2 == 1]

    if len(conditions) == len(thens):
        conditions, thens = _scalar(conditions), _scalar(thens)
    else:
        raise errors.ExpressionError('Switch should be called by case and then pairs')

    if isinstance(expr, (Scalar, SequenceExpr)):
        case = expr
    else:
        case = None
        if not all(hasattr(it, 'dtype') and it.dtype == types.boolean for it in conditions):
            raise errors.ExpressionError('Switch must be called by all boolean conditions')

    res = thens if default is None else thens + [default, ]
    output_type = utils.highest_precedence_data_type(*(it.dtype for it in res))

    is_seq = isinstance(expr, SequenceExpr) or \
        any(isinstance(it, SequenceExpr) for it in conditions) or \
        any(isinstance(it, SequenceExpr) for it in res)
    if case is not None:
        is_seq = is_seq or isinstance(case, SequenceExpr)

    kwargs = dict()
    if is_seq:
        kwargs['_data_type'] = output_type
    else:
        kwargs['_value_type'] = output_type
    return Switch(_input=expr, _case=case, _conditions=conditions,
                  _thens=thens, _default=default, **kwargs)


def switch(*args, **kwargs):
    return _switch(None, *args, **kwargs)


def _cut(expr, bins, right=True, labels=None, include_lowest=False,
         include_under=False, include_over=False):
    """
    Return indices of half-open bins to which each value of `expr` belongs.

    :param expr: sequence or scalar
    :param bins: list of scalars
    :param right: indicates whether the bins include the rightmost edge or not. If right == True(the default),
                  then the bins [1, 2, 3, 4] indicate (1, 2], (2, 3], (3, 4]
    :param labels: Usesd as labes for the resulting bins. Must be of the same length as the resulting bins.
    :param include_lowest: Whether the first interval should be left-inclusive or not.
    :param include_under: include the bin below the leftmost edge or not
    :param include_over: include the bin above the rightmost edge or not
    :return: sequence or scalar
    """

    is_seq = isinstance(expr, SequenceExpr)
    dtype = utils.highest_precedence_data_type(
        *(types.validate_value_type(it) for it in labels)) \
        if labels is not None else types.int64
    kw = {}
    if is_seq:
        kw['_data_type'] = dtype
    else:
        kw['_value_type'] = dtype

    return Cut(_input=expr, _bins=bins, _right=right, _labels=labels,
               _include_lowest=include_lowest, _include_under=include_under,
               _include_over=include_over, **kw)


def _int_to_datetime(expr):
    """
    Return a sequence or scalar that is the datetime value of the current numeric sequence or scalar.

    :param expr: sequence or scalar
    :return: sequence or scalar
    """

    if isinstance(expr, SequenceExpr):
        return IntToDatetime(_input=expr, _data_type=types.datetime)
    elif isinstance(expr, Scalar):
        return IntToDatetime(_input=expr, _value_type=types.datetime)


_element_methods = dict(
    map=_map,
    isnull=_isnull,
    notnull=_notnull,
    fillna=_fillna,
    between=_between,
    switch=_switch,
    cut=_cut,
    isin=_isin,
    notin=_notin,
    hash=_hash,
)

utils.add_method(SequenceExpr, _element_methods)
utils.add_method(Scalar, _element_methods)

BooleanSequenceExpr.ifelse = _ifelse
BooleanScalar.ifelse = _ifelse

CollectionExpr.switch = _switch

for int_number_sequence in int_number_sequences + int_number_scalars:
    int_number_sequence.to_datetime = _int_to_datetime
