#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import itertools
import operator
import inspect

import six

from ...models import Schema
from .expressions import Expr, CollectionExpr, BooleanSequenceExpr, \
    Column, SequenceExpr, Scalar, repr_obj
from .collections import SortedExpr
from .errors import ExpressionError
from . import utils
from ...compat import reduce
from ..utils import FunctionWrapper
from .. import types
from ..types import string, validate_data_type


class BaseGroupBy(Expr):
    __slots__ = '_by', '_to_agg', '_by_names'
    _args = '_input',

    def _init(self, *args, **kwargs):
        super(BaseGroupBy, self)._init(*args, **kwargs)
        if isinstance(self._by, list):
            self._by = [self._get_group_by_field(field) for field in self._by]
        else:
            self._by = [self._get_group_by_field(self._by)]
        self._to_agg = set(col.name for col in self._input._schema.columns)

    def _get_group_by_field(self, field):
        field = self._defunc(field)

        if isinstance(field, Column):
            return field
        elif isinstance(field, six.string_types):
            return self._input[field]
        else:
            return field

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
            raise TypeError('Fail to get group by fields, column cannot be renamed')

        _to_agg = \
            [field.source_name for field in item if field.source_name in self._to_agg]

        return GroupBy(_input=self._input, _by=self._by,
                       _by_names=self._by_names, _to_agg=_to_agg)

    def __getattr__(self, attr):
        try:
            obj = object.__getattribute__(self, attr)

            return obj
        except AttributeError as e:
            if attr in object.__getattribute__(self, '_to_agg'):
                return self[attr]

            raise e

    def sort_values(self, by, ascending=True):
        if hasattr(self, '_having') and self._having is not None:
            raise ExpressionError('Cannot sort GroupBy with `having`')

        if not isinstance(by, list):
            by = [by, ]
        by = [self._defunc(it) for it in by]

        attr_values = dict((attr, getattr(self, attr, None))
                           for attr in utils.get_attrs(self))
        attr_values['_sorted_fields'] = by
        attr_values['_ascending'] = ascending
        del attr_values['_having']

        return SortedGroupBy(**attr_values)

    def sort(self, *args, **kwargs):
        return self.sort_values(*args, **kwargs)

    def mutate(self, *windows, **kw):
        if hasattr(self, '_having') and self._having is not None:
            raise ExpressionError('Cannot mutate GroupBy with `having`')

        if len(windows) == 0 and isinstance(windows[0], list):
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
        if not frozenset(win_field_names).issubset(self._to_agg):
            for agg_field_name in win_field_names:
                if agg_field_name not in self._to_agg:
                    raise ValueError('Unknown field to aggregate: %s' % repr_obj(agg_field_name))

        names = [by.name for by in self._by if isinstance(by, Column)] + \
                [win.name for win in windows]
        types = [by._data_type for by in self._by if isinstance(by, Column)] + \
                [win._data_type for win in windows]

        return MutateCollectionExpr(_input=self, _window_fields=windows,
                                    _schema=Schema.from_lists(names, types))

    def apply(self, func, names=None, types=None, args=(), **kwargs):
        if isinstance(func, FunctionWrapper):
            names = names or func.output_names
            types = types or func.output_types
            func = func._func

        if names is None:
            raise ValueError('Apply on rows should provide column names')
        tps = (string, ) * len(names) if types is None else tuple(validate_data_type(t) for t in types)
        schema = Schema.from_lists(names, tps)

        if hasattr(self, '_having') and self._having is not None:
            raise ExpressionError('Cannot apply function to GroupBy with `having`')

        return GroupbyAppliedCollectionExpr(_input=self, _func=func, _func_args=args,
                                            _func_kwargs=kwargs, _schema=schema)


class GroupBy(BaseGroupBy):
    __slots__ = '_having',

    def _init(self, *args, **kwargs):
        self._having = None
        super(GroupBy, self)._init(*args, **kwargs)

    def _is_reduction(self, agg):
        from .reduction import GroupedSequenceReduction

        return any(isinstance(node, GroupedSequenceReduction)
                   for node in itertools.chain(*agg.all_path(self._input)))

    def _as_grouped(self, reduction_expr):
        if isinstance(reduction_expr, Scalar):
            from .reduction import as_grouped, SequenceReduction

            root = reduction_expr
            for path in reduction_expr.all_path(self._input):
                for idx, node in enumerate(path):
                    if isinstance(node, SequenceReduction):
                        agg = as_grouped(node)
                        if isinstance(agg.input, Column):
                            agg._input = self[agg.source_name]
                        to_sub = agg
                    elif isinstance(node, Scalar):
                        to_sub = node.to_sequence()
                    else:
                        continue

                    path[idx] = to_sub
                    if idx == 0:
                        root = to_sub
                    else:
                        path[idx - 1].substitute(node, to_sub)

            return root

        return reduction_expr

    def __repr__(self):
        return object.__repr__(self)

    def __getitem__(self, item):
        item = self._defunc(item)

        if isinstance(item, BooleanSequenceExpr):
            having = item
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
        if len(aggregations) == 1 and isinstance(aggregations[0], list):
            aggregations = aggregations[0]
        else:
            aggregations = list(aggregations)
        aggregations = [self._defunc(it) for it in aggregations]
        if kw:
            aggregations.extend([self._defunc(agg).rename(new_name)
                                 for new_name, agg in six.iteritems(kw)])

        # keep sequence to ensure that unittests works well
        aggregations = sorted([self._as_grouped(agg) for agg in aggregations],
                              key=lambda it: it.name)

        if not aggregations:
            raise ValueError('Cannot aggregate on grouped data')
        if not all(self._is_reduction(agg) for agg in aggregations):
            raise TypeError('Only aggregate functions can be provided')

        names = [by.name for by in self._by if isinstance(by, SequenceExpr)] + \
                [agg.name for agg in aggregations]
        types = [by._data_type for by in self._by if isinstance(by, SequenceExpr)] + \
                [agg._data_type for agg in aggregations]

        return GroupByCollectionExpr(_input=self, _aggregations=aggregations,
                                     _schema=Schema.from_lists(names, types))

    def agg(self, *args, **kwargs):
        return self.aggregate(*args, **kwargs)


class SequenceGroupBy(Expr):
    __slots__ = '_name', '_data_type'
    _args = '_input',

    def __new__(cls, *args, **kwargs):
        data_type = kwargs.get('_data_type')
        if data_type:
            cls_name = data_type.__class__.__name__ + SequenceGroupBy.__name__
            clazz = globals()[cls_name]

            return object.__new__(clazz)
        else:
            return object.__new__(cls)

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._data_type

    @property
    def input(self):
        return self._input
    

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


class DatetimeSequenceGroupBy(SequenceGroupBy):
    def _init(self, *args, **kwargs):
        super(DatetimeSequenceGroupBy, self)._init(*args, **kwargs)
        self._data_type = types.datetime


class SortedGroupBy(BaseGroupBy, SortedExpr):
    __slots__ = '_sorted_fields', '_ascending'


class GroupByCollectionExpr(CollectionExpr):
    _args = '_input', '_by', '_having', '_aggregations', '_fields'
    node_name = 'GroupBy'

    def _init(self, *args, **kwargs):
        self._fields = None

        super(GroupByCollectionExpr, self)._init(*args, **kwargs)

        if isinstance(self._input, GroupBy):
            self._by = self._input._by
            self._having = self._input._having
            self._input = self._input._input

    def iter_args(self):
        arg_names = ['collection', 'by', 'having', 'aggregations']
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

    def _project(self, fields):
        # FIXME: move to optimize
        # consider the case:
        # df = input.groupby(**).agg(**)
        # df.cache()
        # df.select(***)
        # the cached will not be executed

        selections = []
        names = []
        tps = []

        select_fields = []
        for field in fields:
            if isinstance(field, CollectionExpr):
                select_fields.extend(field.fields)
            else:
                select_fields.append(field)

        name_to_exprs = self._name_to_exprs()
        for field in select_fields:
            if isinstance(field, six.string_types):
                col = name_to_exprs[field]
                selections.append(col)
                names.append(field)
                tps.append(col.dtype)
            else:
                for path in field.all_path(self):
                    if len(path) >= 2 and isinstance(path[-2], Column):
                        col = path[-2]
                        to_sub = name_to_exprs[col.source_name or col.name]
                        if col.source_name is not None and col.source_name != col.name:
                            to_sub = to_sub.rename(col.name)
                        if len(path) > 2:
                            path[-3].substitute(col, to_sub)
                        else:
                            field = to_sub
                selections.append(field)
                names.append(field.name)
                tps.append(field.dtype)
        return GroupByCollectionExpr(self._input, self._by, self._having,
                                     self._aggregations, selections,
                                     _schema=Schema.from_lists(names, tps))

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
    _args = '_input', '_by', '_window_fields'
    node_name = 'Mutate'

    def _init(self, *args, **kwargs):
        super(MutateCollectionExpr, self)._init(*args, **kwargs)

        self._by = self._input._by
        self._input = self._input._input

    def iter_args(self):
        for it in zip(['collection', 'by', 'mutates'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    @property
    def fields(self):
        return self._by + self._window_fields

    def accept(self, visitor):
        return visitor.visit_mutate(self)


class GroupbyAppliedCollectionExpr(CollectionExpr):
    __slots__ = '_func', '_func_args', '_func_kwargs'
    _args = '_input', '_by', '_sort_fields', '_fields'
    node_name = 'Apply'

    def _init(self, *args, **kwargs):
        self._sort_fields = None

        super(GroupbyAppliedCollectionExpr, self)._init(*args, **kwargs)

        if isinstance(self._input, BaseGroupBy):
            if isinstance(self._input, SortedGroupBy):
                self._sort_fields = self._input._sorted_fields
            self._by = self._input._by
            self._fields = self._input._to_agg
            self._input = self._input._input

        self._fields = sorted(
            [self._input[f] if isinstance(f, six.string_types) else f
             for f in self._fields], key=lambda f: self._input._schema._name_indexes[f.name]
        )

    @property
    def fields(self):
        return self._fields

    @property
    def input(self):
        return self._input

    @property
    def input_types(self):
        return [f.dtype for f in self._fields]

    def iter_args(self):
        arg_names = ['collection', 'by', 'sort', 'fields']
        for it in zip(arg_names, self.args):
            yield it

    def accept(self, visitor):
        return visitor.visit_apply_collection(self)


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
    _args = '_input', '_by', '_sort',
    node_name = 'ValueCounts'

    def _init(self, *args, **kwargs):
        super(ValueCounts, self)._init(*args, **kwargs)

        self._by = self._input
        self._input = next(it for it in self._input.traverse(top_down=True)
                           if isinstance(it, CollectionExpr))
        if isinstance(self._sort, bool):
            self._sort = Scalar(_value=self._sort)

    def iter_args(self):
        for it in zip(['collection', 'by', '_sort'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_value_counts(self)


def value_counts(expr, sort=True):
    """
    Return object containing counts of unique values.

    The resulting object will be in descending order so that the first element is the most frequently-occuring
    element. Exclude NA values by default

    :param expr: sequence
    :return: collection with two columns
    :rtype: :class:`odps.df.expr.expressions.CollectionExpr`
    """

    names = [expr.name, 'count']
    typos = [expr.dtype, types.int64]
    return ValueCounts(_input=expr, _schema=Schema.from_lists(names, typos), _sort=sort)


def topk(expr, k):
    return expr.value_counts().limit(k)


SequenceExpr.value_counts = value_counts
SequenceExpr.topk = topk
