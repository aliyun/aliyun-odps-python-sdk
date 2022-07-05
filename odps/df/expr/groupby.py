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

import operator
import random

from ...models import Schema
from .expressions import Expr, CollectionExpr, BooleanSequenceExpr, \
    Column, SequenceExpr, Scalar, BooleanScalar, repr_obj
from .collections import SortedExpr, ReshuffledCollectionExpr
from .errors import ExpressionError
from . import utils
from ...compat import reduce, six
from .. import types
from ..utils import is_constant_scalar
from ...utils import object_getattr, camel_to_underline


class BaseGroupBy(Expr):
    __slots__ = '_to_agg', '_by_names'
    _args = '_input', '_by'

    def _init(self, *args, **kwargs):
        self._init_attr('_to_agg', None)
        super(BaseGroupBy, self)._init(*args, **kwargs)
        if isinstance(self._by, list):
            self._by = self._input._get_fields(self._by)
        else:
            self._by = [self._input._get_field(self._by)]
        for idx, by_field in enumerate(self._by):
            if by_field.name is None:
                new_field_name = '_%s_%d' % (camel_to_underline(type(by_field).__name__),
                                             random.randint(10000, 99999))
                self._by[idx] = by_field.rename(new_field_name)
        if self._to_agg is None:
            self._to_agg = self._input.schema

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            if item in self._to_agg:
                return SequenceGroupBy(_input=self, _name=item,
                                       _data_type=self._input._schema[item].type)
            else:
                raise KeyError('Fail to get group by field, unknown field: %s' % repr_obj(item))

        is_field = lambda it: isinstance(it, six.string_types) or isinstance(it, Column)
        if not all(is_field(it) for it in item):
            raise TypeError('Fail to get group by fields, unknown type: %s' % type(item))
        if any(col.is_renamed() for col in item if isinstance(col, Column)):
            raise ValueError('Fail to get group by fields, column cannot be renamed')

        get_name = lambda it: it if isinstance(it, six.string_types) else it.source_name
        _to_agg = type(self._input.schema)(
            columns=self._input.schema[[get_name(field) for field in item
                                        if get_name(field) in self._to_agg]])

        return GroupBy(_input=self._input, _by=self._by,
                       _by_names=getattr(self, '_by_names', None), _to_agg=_to_agg)

    def __getattr__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError as e:
            if attr.startswith('__'):
                raise e

            agg = object.__getattribute__(self, '_to_agg')
            if agg is not None and attr in agg:
                return self[attr]

            raise e

    def sort_values(self, by, ascending=True):
        if hasattr(self, '_having') and self._having is not None:
            raise ExpressionError('Cannot sort GroupBy with `having`')

        if not isinstance(by, list):
            by = [by, ]
        by = [self._defunc(it) for it in by]

        attr_values = dict((attr, object_getattr(self, attr, None))
                           for attr in utils.get_attrs(self))
        attr_values['_sorted_fields'] = by
        attr_values['_ascending'] = ascending
        attr_values.pop('_having', None)

        return SortedGroupBy(**attr_values)

    def sort(self, *args, **kwargs):
        return self.sort_values(*args, **kwargs)

    def mutate(self, *windows, **kw):
        if hasattr(self, '_having') and self._having is not None:
            raise ExpressionError('Cannot mutate GroupBy with `having`')

        if len(windows) == 1 and isinstance(windows[0], list):
            windows = windows[0]
        else:
            windows = list(windows)
        windows = [self._defunc(win) for win in windows]
        if kw:
            windows.extend([self._defunc(win).rename(new_name)
                            for new_name, win in six.iteritems(kw)])

        from .window import Window

        if not windows:
            raise ValueError('Cannot mutate on grouped data')
        if not all(isinstance(win, Window) for win in windows):
            raise TypeError('Only window functions can be provided')

        win_field_names = filter(lambda it: it is not None,
                                 [win.source_name for win in windows])
        if not frozenset(win_field_names).issubset(self._to_agg.names):
            for agg_field_name in win_field_names:
                if agg_field_name not in self._to_agg:
                    raise ValueError('Unknown field to aggregate: %s' % repr_obj(agg_field_name))

        names = [by.name for by in self._by if isinstance(by, Column)] + \
                [win.name for win in windows]
        types = [by._data_type for by in self._by if isinstance(by, Column)] + \
                [win._data_type for win in windows]

        return MutateCollectionExpr(_input=self, _window_fields=windows,
                                    _schema=Schema.from_lists(names, types))

    def apply(self, func, names=None, types=None, resources=None, args=(), **kwargs):
        reshuffled = ReshuffledCollectionExpr(_input=self, _schema=self._input._schema)
        return reshuffled.apply(axis=1, func=func, names=names, types=types,
                                resources=resources, args=args, **kwargs)


class GroupBy(BaseGroupBy):
    __slots__ = '_having',

    def _init(self, *args, **kwargs):
        self._init_attr('_having', None)
        super(GroupBy, self)._init(*args, **kwargs)

    def _same_by(self, other):
        if other._input is not self._input:
            return False
        if len(self._by) != len(other._by):
            return False
        if any(x is not y for x, y in zip(self._by, other._by)):
            return False
        return True

    def _validate_agg(self, agg):
        from .reduction import GroupedSequenceReduction
        from .window import RankOp

        has_reduction = False
        for node in agg.traverse(top_down=True, unique=True,
                                 stop_cond=lambda x: x is self._input):
            if isinstance(node, GroupedSequenceReduction):
                has_reduction = True
                if not self._same_by(node._grouped):
                    raise ExpressionError(
                        'Aggregation has not been applied to the right GroupBy, got: %s' % repr_obj(agg))
            elif isinstance(node, Column):
                if node._input is not self._input:
                    raise ExpressionError(
                        'Aggregation should be applied to the column of %s' % repr_obj(self._input))
            elif isinstance(node, RankOp) and node._input is not self._input:
                raise ExpressionError(
                    'Aggregation should be applied to the column of %s' % repr_obj(self._input))

        if not has_reduction:
            raise ExpressionError('No aggregation found in %s' % repr_obj(agg))

    def _transform(self, reduction_expr):
        if isinstance(reduction_expr, Scalar):
            from .reduction import SequenceReduction
            from .window import RankOp

            dag = reduction_expr.to_dag(copy=False, validate=False)
            for node in dag.traverse(
                    stop_cond=lambda x: isinstance(x, (Column, RankOp)) or x is self._input):
                if isinstance(node, SequenceReduction):
                    to_sub = node.to_grouped_reduction(self)
                    dag.substitute(node, to_sub)
                elif isinstance(node, Scalar) and not is_constant_scalar(node) \
                        and len(node.children()) > 0:
                    to_sub = node.to_sequence()
                    dag.substitute(node, to_sub)
                elif isinstance(node, Column) and node._input is not self._input:
                    field = self._input._get_field(node)
                    if field:
                        dag.substitute(node, field)
                elif isinstance(node, RankOp) and node._input is not self._input:
                    dag.substitute(node._input, self._input, parents=[node])

            return dag.root

        return reduction_expr

    def __repr__(self):
        return object.__repr__(self)

    def __getitem__(self, item):
        item = self._defunc(item)

        if isinstance(item, (BooleanSequenceExpr, BooleanScalar)):
            having = item if isinstance(item, BooleanSequenceExpr) \
                else self._transform(item)
            if self._having is not None:
                having = having & self._having
            return GroupBy(_input=self._input, _by=self._by,
                           _to_agg=self._to_agg,
                           _having=having)
        else:
            return super(GroupBy, self).__getitem__(item)

    def filter(self, *predicates):
        predicates = [self._defunc(it) for it in predicates]

        predicate = reduce(operator.and_, predicates)
        return self[predicate]

    def aggregate(self, *aggregations, **kw):
        sort_by_name = kw.pop('sort_by_name', True)

        if len(aggregations) == 1 and isinstance(aggregations[0], list):
            aggregations = aggregations[0]
        else:
            aggregations = list(aggregations)
        aggregations = [self._defunc(it) for it in aggregations]
        if kw:
            aggregations.extend([self._defunc(agg).rename(new_name)
                                 for new_name, agg in six.iteritems(kw)])

        # keep sequence to ensure that unittests works well
        if sort_by_name:
            aggregations = sorted([self._transform(agg) for agg in aggregations],
                                  key=lambda it: it.name)
        else:
            aggregations = [self._transform(agg) for agg in aggregations]

        if not aggregations:
            raise ValueError('Cannot aggregate on grouped data')
        [self._validate_agg(agg) for agg in aggregations]

        names = [by.name for by in self._by
                 if isinstance(by, (Scalar, SequenceExpr)) and by.name is not None] + \
                [agg.name for agg in aggregations]
        types = [by.dtype for by in self._by
                 if isinstance(by, (Scalar, SequenceExpr)) and by.name is not None] + \
                [agg._data_type for agg in aggregations]

        return GroupByCollectionExpr(_input=self, _aggregations=aggregations,
                                     _schema=Schema.from_lists(names, types))

    def agg(self, *args, **kwargs):
        return self.aggregate(*args, **kwargs)


class SequenceGroupBy(Expr):
    __slots__ = '_name', '_data_type', '_source_data_type'
    _args = '_input',

    def _init(self, *args, **kwargs):
        self._init_attr('_data_type', None)
        self._init_attr('_source_data_type', None)
        super(SequenceGroupBy, self)._init(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        data_type = kwargs.get('_data_type')
        if data_type:
            cls_name = data_type.__class__.__name__ + SequenceGroupBy.__name__
            clazz = globals()[cls_name]

            return super(SequenceGroupBy, clazz).__new__(clazz)
        else:
            return super(SequenceGroupBy, cls).__new__(cls)

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._data_type

    @property
    def input(self):
        return self._input

    def astype(self, data_type):
        data_type = types.validate_data_type(data_type)

        if data_type == self._data_type:
            return self

        attr_dict = dict((k, getattr(self, k, None)) for k in utils.get_attrs(self))
        attr_dict['_data_type'] = data_type
        attr_dict['_source_data_type'] = self._source_data_type

        cls = globals().get(repr(data_type).capitalize() + SequenceGroupBy.__name__)
        new_sequence_groupby = cls(**attr_dict)

        return new_sequence_groupby

    def is_astyped(self):
        if self._source_data_type is None:
            return False
        return self._data_type != self._source_data_type

    def to_column(self):
        collection = self.input.input
        input = collection[self.name]
        if self.is_astyped():
            input = input.astype(self._data_type)
        return input
    

class BooleanSequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(BooleanSequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.boolean


class Int8SequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(Int8SequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.int8


class Int16SequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(Int16SequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.int16


class Int32SequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(Int32SequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.int32


class Int64SequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(Int64SequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.int64


class Float32SequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(Float32SequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.float32


class Float64SequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(Float64SequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.float64


class DecimalSequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(DecimalSequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.decimal


class StringSequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(StringSequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.string


class BinarySequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(BinarySequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.binary


class DatetimeSequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(DatetimeSequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.datetime


class UnknownSequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(UnknownSequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.Unknown()


class SortedGroupBy(BaseGroupBy, SortedExpr):
    __slots__ = '_sorted_fields', '_ascending'


class GroupByCollectionExpr(CollectionExpr):
    _args = '_input', '_by', '_having', '_aggregations', '_fields'
    node_name = 'GroupBy'

    def _init(self, *args, **kwargs):
        self._init_attr('_fields', None)

        super(GroupByCollectionExpr, self)._init(*args, **kwargs)

        if isinstance(self._input, GroupBy):
            self._by = self._input._by
            self._having = self._input._having
            self._input = self._input._input

    def iter_args(self):
        arg_names = ['collection', 'bys', 'having', 'aggregations']
        for it in zip(arg_names, self.args):
            yield it
        if self._fields is not None:
            yield ('selections', self._fields)

    def _name_to_exprs(self):
        if hasattr(self, '_fields') and self._fields is not None:
            exprs = self._fields
        else:
            exprs = self.args[1] + self.args[3]
        return dict((expr.name, expr) for expr in exprs if hasattr(expr, 'name'))

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_groupby(self)

    @property
    def fields(self):
        if self._fields is not None:
            return self._fields

        return self._by + self._aggregations


class MutateCollectionExpr(CollectionExpr):
    _args = '_input', '_by', '_window_fields', '_fields'
    node_name = 'Mutate'

    def _init(self, *args, **kwargs):
        self._init_attr('_fields', None)
        super(MutateCollectionExpr, self)._init(*args, **kwargs)

        if isinstance(self._input, GroupBy):
            self._by = self._input._by
            self._input = self._input._input

    @property
    def _project_fields(self):
        return self._window_fields

    def iter_args(self):
        for it in zip(['collection', 'bys', 'mutates'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    @property
    def fields(self):
        if self._fields is not None:
            return self._fields

        return self._by + self._window_fields

    def accept(self, visitor):
        return visitor.visit_mutate(self)


def groupby(expr, by, *bys):
    """
    Group collection by a series of sequences.

    :param expr: collection
    :param by: columns to group
    :param bys: columns to group
    :return: GroupBy instance
    :rtype: :class:`odps.df.expr.groupby.GroupBy`
    """

    if not isinstance(by, list):
        by = [by, ]
    if len(bys) > 0:
        by = by + list(bys)
    return GroupBy(_input=expr, _by=by)


CollectionExpr.groupby = groupby


class ValueCounts(CollectionExpr):
    _args = '_input', '_by', '_sort', '_ascending', '_dropna'
    node_name = 'ValueCounts'

    def _init(self, *args, **kwargs):
        super(ValueCounts, self)._init(*args, **kwargs)

        if isinstance(self._input, SequenceExpr):
            self._by = self._input
            self._input = next(it for it in self._input.traverse(top_down=True)
                               if isinstance(it, CollectionExpr))
        if isinstance(self._sort, bool):
            self._sort = Scalar(_value=self._sort)
        if isinstance(self._ascending, bool):
            self._ascending = Scalar(_value=self._ascending)
        if isinstance(self._dropna, bool):
            self._dropna = Scalar(_value=self._dropna)

    def iter_args(self):
        for it in zip(['collection', 'by', 'sort', 'ascending', 'dropna'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_value_counts(self)


def value_counts(expr, sort=True, ascending=False, dropna=False):
    """
    Return object containing counts of unique values.

    The resulting object will be in descending order so that the first element is the most frequently-occuring
    element. Exclude NA values by default

    :param expr: sequence
    :param sort: if sort
    :type sort: bool
    :param dropna: Don’t include counts of None, default False
    :return: collection with two columns
    :rtype: :class:`odps.df.expr.expressions.CollectionExpr`
    """

    names = [expr.name, 'count']
    typos = [expr.dtype, types.int64]
    return ValueCounts(_input=expr, _schema=Schema.from_lists(names, typos),
                       _sort=sort, _ascending=ascending, _dropna=dropna)


def topk(expr, k):
    return expr.value_counts().limit(k)


SequenceExpr.value_counts = value_counts
SequenceExpr.topk = topk
