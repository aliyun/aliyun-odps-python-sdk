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

from .groupby import *
from .reduction import _stats_type
from . import utils, errors
from .. import types
from ...utils import camel_to_underline, object_getattr


class Window(SequenceExpr):
    _args = '_input', '_partition_by', '_order_by', '_preceding', '_following'

    def _init(self, *args, **kwargs):
        self._preceding = None
        self._following = None

        super(Window, self)._init(*args, **kwargs)

        self._preceding, self._following = \
            self._de_scalar(self._preceding), self._de_scalar(self._following)

        if isinstance(self._preceding, list):
            self._preceding = tuple(self._preceding)
        elif isinstance(self._following, list):
            self._following = tuple(self._following)

        if (isinstance(self._preceding, tuple) and self._following is not None) or \
            (isinstance(self._following, tuple) and self._preceding is not None):
            raise ValueError('Cannot specify window of both sides')

        if isinstance(self._preceding, tuple):
            start, end = self._preceding
            if start is None:
                assert end >= 0
            else:
                assert start > end
        elif isinstance(self._following, tuple):
            start, end = self._following
            if end is None:
                assert start >= 0
            else:
                assert start < end
        else:
            if self._preceding is not None and self._preceding < 0:
                raise ValueError('Window offset must be positive')

            if self._following is not None and self._following < 0:
                raise ValueError('Window offset must be positive')
        self._preceding, self._following = \
            self._scalar(self._preceding), self._scalar(self._following)

    def _scalar(self, val):
        if val is None:
            return
        if isinstance(val, Scalar):
            return val
        if isinstance(val, tuple):
            return tuple(self._scalar(it) for it in val)
        else:
            return Scalar(_value=val)

    def _de_scalar(self, val):
        if val is None:
            return
        if isinstance(val, tuple):
            return tuple(self._de_scalar(it) for it in val)
        elif isinstance(val, Scalar):
            return val._value
        else:
            return val

    def iter_args(self):
        for it in zip(['Column', 'PartitionBy', 'OrderBy',
                       'preceding', 'following'], self.args):
            yield it

    @property
    def name(self):
        if self._name:
            return self._name
        source_name = self.source_name
        if source_name:
            name = self.node_name
            if name.startswith('Cum'):
                name = name[3:]
            return '%s_%s' % (source_name, name.lower())

    @property
    def source_name(self):
        input_name = None
        if isinstance(self._input, SequenceExpr):
            input_name = self._input.name
        return self._source_name or input_name

    @property
    def node_name(self):
        return self.__class__.__name__

    @property
    def input(self):
        return self._input

    @property
    def preceding(self):
        return self._de_scalar(self._preceding)

    @property
    def following(self):
        return self._de_scalar(self._following)


class CumulativeOp(Window):
    _args = '_input', '_partition_by', '_distinct', '_order_by', \
            '_preceding', '_following'

    def _init(self, *args, **kwargs):
        self._distinct = False

        super(CumulativeOp, self)._init(*args, **kwargs)

        self._distinct = self._de_scalar(self._distinct)
        self._check_unique()
        self._distinct = self._scalar(self._distinct)

    def _check_unique(self):
        if self._distinct and len(getattr(self, '_order_by', None) or []) > 0:
            raise errors.ExpressionError('Unique and sort cannot exist both')

    def unique(self):
        if self._distinct.value:
            return self

        self._check_unique()

        attr_values = dict((attr, getattr(self, attr, None))
                           for attr in utils.get_attrs(self))
        attr_values['_distinct'] = True
        return type(self)(**attr_values)

    def iter_args(self):
        for it in zip(['Column', 'PartitionBy', 'distinct', 'OrderBy',
                       'preceding', 'following'], self.args):
            yield it

    @property
    def distinct(self):
        return self._distinct.value

    def accept(self, visitor):
        return visitor.visit_cum_window(self)


class CumSum(CumulativeOp):
    __slots__ = ()


class CumMax(CumulativeOp):
    __slots__ = ()


class CumMin(CumulativeOp):
    __slots__ = ()


class CumMean(CumulativeOp):
    __slots__ = ()


class CumMedian(CumulativeOp):
    __slots__ = ()


class CumCount(CumulativeOp):
    __slots__ = ()


class CumStd(CumulativeOp):
    __slots__ = ()


class NthValue(CumulativeOp):
    __slots__ = '_nth', '_skip_nulls'


class RankOp(Window):
    __slots__ = ()

    @property
    def name(self):
        if self._name:
            return self._name
        return camel_to_underline(self.__class__.__name__)

    def accept(self, visitor):
        return visitor.visit_rank_window(self)


class Rank(RankOp):
    __slots__ = ()


class DenseRank(RankOp):
    __slots__ = ()


class PercentRank(RankOp):
    __slots__ = ()


class RowNumber(RankOp):
    __slots__ = ()


class QCut(RankOp):
    __slots__ = '_bins',


class CumeDist(RankOp):
    __slots__ = ()


class ShiftOp(Window):
    _args = '_input', '_partition_by', '_order_by', '_offset', '_default'

    def _init(self, *args, **kwargs):
        self._init_attr('_offset', None)
        self._init_attr('_default', None)
        super(ShiftOp, self)._init(*args, **kwargs)

        if self._offset is not None and not isinstance(self._offset, Expr):
            self._offset = self._scalar(self._offset)
        if self._default is not None and not isinstance(self._default, Expr):
            self._default = self._scalar(self._default)

    def accept(self, visitor):
        return visitor.visit_shift_window(self)


class Lag(ShiftOp):
    __slots__ = ()


class Lead(ShiftOp):
    __slots__ = ()


def _cumulative_op(expr, op_cls, sort=None, ascending=True, unique=False,
                   preceding=None, following=None, data_type=None, **kwargs):
    if isinstance(expr, SequenceGroupBy):
        if sort is not None:
            groupby = expr._input.sort(sort, ascending=ascending)
            expr = groupby[expr._name]

        collection = expr._input._input
        column = collection[expr._name]

        data_type = data_type or expr._data_type

        return op_cls(_input=column, _partition_by=expr._input._by,
                      _order_by=object_getattr(expr._input, '_sorted_fields', None),
                      _preceding=preceding, _following=following,
                      _data_type=data_type, _distinct=unique, **kwargs)


def cumsum(expr, sort=None, ascending=True, unique=False,
           preceding=None, following=None):
    """
    Calculate cumulative summation of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :param unique: whether to eliminate duplicate entries
    :param preceding: the start point of a window
    :param following: the end point of a window
    :return: calculated column
    """
    if expr._data_type == types.boolean:
        output_type = types.int64
    else:
        output_type = expr._data_type
    return _cumulative_op(expr, CumSum, sort=sort, ascending=ascending,
                          unique=unique, preceding=preceding,
                          following=following, data_type=output_type)


def cummax(expr, sort=None, ascending=True, unique=False,
           preceding=None, following=None):
    """
    Calculate cumulative maximum of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :param unique: whether to eliminate duplicate entries
    :param preceding: the start point of a window
    :param following: the end point of a window
    :return: calculated column
    """
    return _cumulative_op(expr, CumMax, sort=sort, ascending=ascending,
                          unique=unique, preceding=preceding,
                          following=following)


def cummin(expr, sort=None, ascending=True, unique=False,
           preceding=None, following=None):
    """
    Calculate cumulative minimum of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :param unique: whether to eliminate duplicate entries
    :param preceding: the start point of a window
    :param following: the end point of a window
    :return: calculated column
    """
    return _cumulative_op(expr, CumMin, sort=sort, ascending=ascending,
                          unique=unique, preceding=preceding,
                          following=following)


def cummean(expr, sort=None, ascending=True, unique=False,
            preceding=None, following=None):
    """
    Calculate cumulative mean of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :param unique: whether to eliminate duplicate entries
    :param preceding: the start point of a window
    :param following: the end point of a window
    :return: calculated column
    """
    data_type = _stats_type(expr)
    return _cumulative_op(expr, CumMean, sort=sort, ascending=ascending,
                          unique=unique, preceding=preceding,
                          following=following, data_type=data_type)


def cummedian(expr, sort=None, ascending=True, unique=False,
              preceding=None, following=None):
    """
    Calculate cumulative median of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :param unique: whether to eliminate duplicate entries
    :param preceding: the start point of a window
    :param following: the end point of a window
    :return: calculated column
    """
    data_type = _stats_type(expr)
    return _cumulative_op(expr, CumMedian, sort=sort, ascending=ascending,
                          unique=unique, preceding=preceding,
                          following=following, data_type=data_type)


def cumcount(expr, sort=None, ascending=True, unique=False,
             preceding=None, following=None):
    """
    Calculate cumulative count of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :param unique: whether to eliminate duplicate entries
    :param preceding: the start point of a window
    :param following: the end point of a window
    :return: calculated column
    """
    data_type = types.int64
    return _cumulative_op(expr, CumCount, sort=sort, ascending=ascending,
                          unique=unique, preceding=preceding,
                          following=following, data_type=data_type)


def cumstd(expr, sort=None, ascending=True, unique=False,
           preceding=None, following=None):
    """
    Calculate cumulative standard deviation of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :param unique: whether to eliminate duplicate entries
    :param preceding: the start point of a window
    :param following: the end point of a window
    :return: calculated column
    """
    data_type = _stats_type(expr)
    return _cumulative_op(expr, CumStd, sort=sort, ascending=ascending,
                          unique=unique, preceding=preceding,
                          following=following, data_type=data_type)


def nth_value(expr, nth, skip_nulls=False, sort=None, ascending=True):
    """
    Get nth value of a grouped and sorted expression.

    :param expr: expression for calculation
    :param nth: integer position
    :param skip_nulls: whether to skip null values, False by default
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :return: calculated column
    """
    return _cumulative_op(expr, NthValue, data_type=expr._data_type, sort=sort,
                          ascending=ascending, _nth=nth, _skip_nulls=skip_nulls)


def _rank_op(expr, op_cls, data_type, sort=None, ascending=True, **kwargs):
    if isinstance(expr, SequenceGroupBy):
        grouped = expr._input
        sort = sort or object_getattr(grouped, '_sorted_fields', None) or expr.name
    elif not isinstance(expr, BaseGroupBy):
        raise NotImplementedError
    else:
        grouped = expr

    if not isinstance(grouped, SortedGroupBy) and sort is not None:
        grouped = grouped.sort(sort, ascending=ascending)
        if isinstance(expr, SequenceGroupBy):
            expr = grouped[expr.name]
    if not isinstance(grouped, SortedGroupBy):
        raise ExpressionError('`sort` arg is required for the rank operation')

    return op_cls(_input=grouped._input, _partition_by=grouped._by,
                  _order_by=object_getattr(grouped, '_sorted_fields', None),
                  _data_type=data_type, **kwargs)


def rank(expr, sort=None, ascending=True):
    """
    Calculate rank of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :return: calculated column
    """
    return _rank_op(expr, Rank, types.int64, sort=sort, ascending=ascending)


def dense_rank(expr, sort=None, ascending=True):
    """
    Calculate dense rank of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :return: calculated column
    """
    return _rank_op(expr, DenseRank, types.int64, sort=sort, ascending=ascending)


def percent_rank(expr, sort=None, ascending=True):
    """
    Calculate percentage rank of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :return: calculated column
    """
    return _rank_op(expr, PercentRank, types.float64, sort=sort, ascending=ascending)


def row_number(expr, sort=None, ascending=True):
    """
    Calculate row number of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :return: calculated column
    """
    return _rank_op(expr, RowNumber, types.int64, sort=sort, ascending=ascending)


def qcut(expr, bins, labels=False, sort=None, ascending=True):
    """
    Get quantile-based bin indices of every element of a grouped and sorted expression.
    The indices of bins start from 0. If cuts are not of equal sizes, extra items will
    be appended into the first group.

    :param expr: expression for calculation
    :param bins: number of bins
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :return: calculated column
    """
    if labels is None or labels:
        raise NotImplementedError('Showing bins or customizing labels not supported')
    return _rank_op(expr, QCut, types.int64, sort=sort, ascending=ascending,
                    _bins=bins)


def cume_dist(expr, sort=None, ascending=True):
    """
    Calculate cumulative ratio of a sequence expression.

    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :return: calculated column
    """
    return _rank_op(expr, CumeDist, types.float64, sort=sort, ascending=ascending)


def _shift_op(expr, op_cls, offset, default=None, sort=None, ascending=True):
    if isinstance(expr, SequenceGroupBy):
        if object_getattr(expr._input, '_sorted_fields', None) is None:
            sort = sort or expr.name
            groupby = expr._input.sort(sort, ascending=ascending)
            expr = groupby[expr._name]

        collection = expr._input._input
        column = collection[expr._name]

        return op_cls(_input=column, _partition_by=expr._input._by,
                      _order_by=object_getattr(expr._input, '_sorted_fields', None),
                      _offset=offset, _default=default,
                      _name=expr._name, _data_type=expr._data_type)


def lag(expr, offset, default=None, sort=None, ascending=True):
    """
    Get value in the row ``offset`` rows prior to the current row.

    :param offset: the offset value
    :param default: default value for the function, when there are no rows satisfying the offset
    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :return: calculated column
    """
    return _shift_op(expr, Lag, offset, default=default,
                     sort=sort, ascending=ascending)


def lead(expr, offset, default=None, sort=None, ascending=True):
    """
    Get value in the row ``offset`` rows after to the current row.

    :param offset: the offset value
    :param default: default value for the function, when there are no rows satisfying the offset
    :param expr: expression for calculation
    :param sort: name of the sort column
    :param ascending: whether to sort in ascending order
    :return: calculated column
    """
    return _shift_op(expr, Lead, offset, default=default,
                     sort=sort, ascending=ascending)


_number_window_methods = dict(
    cumsum=cumsum,
    cummean=cummean,
    cummedian=cummedian,
    cumstd=cumstd,
)

_window_methods = dict(
    cummax=cummax,
    cummin=cummin,
    cumcount=cumcount,
    lag=lag,
    lead=lead,
)

_groupby_methods = dict(
    rank=rank,
    min_rank=rank,
    dense_rank=dense_rank,
    percent_rank=percent_rank,
    row_number=row_number,
    qcut=qcut,
    nth_value=nth_value,
    cume_dist=cume_dist,
)

number_windows = [globals().get(repr(t).capitalize() + SequenceGroupBy.__name__)
                  for t in types.number_types()]

for number_window in number_windows:
    utils.add_method(number_window, _number_window_methods)

utils.add_method(SequenceGroupBy, _window_methods)

StringSequenceGroupBy.cumsum = cumsum  # FIXME: should we support string?
BooleanSequenceGroupBy.cumsum = cumsum

utils.add_method(BaseGroupBy, _groupby_methods)
utils.add_method(SequenceGroupBy, _groupby_methods)
