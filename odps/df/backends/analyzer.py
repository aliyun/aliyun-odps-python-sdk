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

from .core import Backend
from ..utils import traverse_until_source
from ..expr.expressions import Scalar, SequenceExpr, CollectionExpr
from ..expr.reduction import GroupedSequenceReduction
from ..expr.element import Switch
from .. import output
from ... import compat
from ...models import Schema
from .utils import refresh_dynamic
from ..types import DynamicSchema
from ...compat import six


class BaseAnalyzer(Backend):
    """
    Analyzer is used before optimzing,
    which analyze some operation that is not supported for this execution backend.
    """

    def __init__(self, expr_dag, traversed=None, on_sub=None):
        self._dag = expr_dag
        self._indexer = itertools.count(0)
        self._traversed = traversed or set()
        self._on_sub = on_sub

    def analyze(self):
        for node in self._iter():
            self._traversed.add(id(node))
            self._visit_node(node)

        return self._dag.root

    def _iter(self):
        for node in traverse_until_source(self._dag, top_down=True,
                                          traversed=self._traversed):
            yield node

        while True:
            all_traversed = True
            for node in traverse_until_source(self._dag, top_down=True):
                if id(node) not in self._traversed:
                    all_traversed = False
                    yield node
            if all_traversed:
                break

    def _visit_node(self, node):
        try:
            node.accept(self)
        except NotImplementedError:
            return

    def _sub(self, expr, sub, parents=None):
        self._dag.substitute(expr, sub, parents=parents)
        if self._on_sub:
            self._on_sub(expr, sub)

    @staticmethod
    def _get_moment_sub_expr(expr, _input, order, center):
        def _group_mean(e):
            m = e.mean()
            if isinstance(expr, GroupedSequenceReduction):
                m = m.to_grouped_reduction(expr._grouped)
            return m

        def _order(e, o):
            if o == 1:
                return e
            else:
                return e ** o

        if not center:
            if order == 0:
                sub = Scalar(1)
            else:
                sub = _group_mean(_input ** order)
        else:
            if order == 0:
                sub = Scalar(1)
            elif order == 1:
                sub = Scalar(0)
            else:
                sub = _group_mean(_input ** order)
                divided = 1
                divisor = 1
                for o in compat.irange(1, order):
                    divided *= order - o + 1
                    divisor *= o
                    part_item = divided // divisor * _group_mean(_order(_input, order - o)) \
                                * (_order(_group_mean(_input), o))
                    if o & 1:
                        sub -= part_item
                    else:
                        sub += part_item
                part_item = _group_mean(_input) ** order
                if order & 1:
                    sub -= part_item
                else:
                    sub += part_item
        return sub

    @classmethod
    def _get_cut_sub_expr(cls, expr):
        is_seq = isinstance(expr, SequenceExpr)
        kw = dict()
        if is_seq:
            kw['_data_type'] = expr.dtype
        else:
            kw['_value_type'] = expr.dtype

        conditions = []
        thens = []

        if expr.include_under:
            bin = expr.bins[0]
            if expr.right and not expr.include_lowest:
                conditions.append(expr.input <= bin)
            else:
                conditions.append(expr.input < bin)
            thens.append(expr.labels[0])
        for i, bin in enumerate(expr.bins[1:]):
            lower_bin = expr.bins[i]
            if not expr.right or (i == 0 and expr.include_lowest):
                condition = lower_bin <= expr.input
            else:
                condition = lower_bin < expr.input

            if expr.right:
                condition = (condition & (expr.input <= bin))
            else:
                condition = (condition & (expr.input < bin))

            conditions.append(condition)
            if expr.include_under:
                thens.append(expr.labels[i + 1])
            else:
                thens.append(expr.labels[i])
        if expr.include_over:
            bin = expr.bins[-1]
            if expr.right:
                conditions.append(bin < expr.input)
            else:
                conditions.append(bin <= expr.input)
            thens.append(expr.labels[-1])

        return Switch(_conditions=conditions, _thens=thens,
                      _default=None, _input=None, **kw)

    @classmethod
    def _get_value_counts_sub_expr(cls, expr):
        collection = expr.input
        by = expr._by
        sort = expr._sort.value
        ascending = expr._ascending.value
        dropna = expr._dropna.value

        sub = collection.groupby(by).agg(count=by.count())
        if sort:
            sub = sub.sort('count', ascending=ascending)
        if dropna:
            sub = sub.filter(sub[by.name].notnull())

        return sub

    def _get_pivot_sub_expr(self, expr):
        columns_expr = expr.input.distinct([c.copy() for c in expr._columns])

        group_names = [g.name for g in expr._group]
        group_types = [g.dtype for g in expr._group]
        exprs = [expr]

        def callback(result, new_expr):
            expr = exprs[0]
            columns = [r[0] for r in result]

            if len(expr._values) > 1:
                names = group_names + \
                        ['{0}_{1}'.format(v.name, c)
                         for v in expr._values for c in columns]
                types = group_types + \
                        list(itertools.chain(*[[n.dtype] * len(columns)
                                               for n in expr._values]))
            else:
                names = group_names + columns
                types = group_types + [expr._values[0].dtype] * len(columns)
            new_expr._schema = Schema.from_lists(names, types)

            column_name = expr._columns[0].name  # column's size can only be 1
            values_names = [v.name for v in expr._values]

            @output(names, types)
            def reducer(keys):
                values = [None] * len(columns) * len(values_names)

                def h(row, done):
                    col = getattr(row, column_name)
                    for val_idx, value_name in enumerate(values_names):
                        val = getattr(row, value_name)
                        idx = len(columns) * val_idx + columns.index(col)
                        if values[idx] is not None:
                            raise ValueError('Row contains duplicate entries')
                        values[idx] = val
                    if done:
                        yield keys + tuple(values)

                return h

            fields = expr._group + expr._columns + expr._values
            pivoted = expr.input.select(fields).map_reduce(reducer=reducer, group=group_names)
            self._sub(new_expr, pivoted)

            # trigger refresh of dynamic operations
            refresh_dynamic(pivoted, self._dag)

        return CollectionExpr(_schema=DynamicSchema.from_lists(group_names, group_types),
                              _deps=[(columns_expr, callback)])

    def _get_pivot_table_sub_expr_without_columns(self, expr):
        def get_agg(field, agg_func, agg_func_name, fill_value):
            if isinstance(agg_func, six.string_types):
                aggregated = getattr(field, agg_func)()
            else:
                aggregated = field.agg(agg_func)
            if fill_value is not None:
                aggregated.fillna(fill_value)
            return aggregated.rename('{0}_{1}'.format(field.name, agg_func_name))

        grouped = expr.input.groupby(expr._group)
        aggs = []
        for agg_func, agg_func_name in zip(expr._agg_func, expr._agg_func_names):
            for value in expr._values:
                agg = get_agg(value, agg_func, agg_func_name, expr.fill_value)
                aggs.append(agg)
        return grouped.aggregate(aggs, sort_by_name=False)

    def _get_pivot_table_sub_expr_with_columns(self, expr):
        columns_expr = expr.input.distinct([c.copy() for c in expr._columns])

        group_names = [g.name for g in expr._group]
        group_types = [g.dtype for g in expr._group]
        exprs = [expr]

        def callback(result, new_expr):
            expr = exprs[0]
            columns = [r[0] for r in result]

            names = list(group_names)
            tps = list(group_types)
            aggs = []
            for agg_func_name, agg_func in zip(expr._agg_func_names, expr._agg_func):
                for value_col in expr._values:
                    for col in columns:
                        base = '{0}_'.format(col) if col is not None else ''
                        name = '{0}{1}_{2}'.format(base, value_col.name, agg_func_name)
                        names.append(name)
                        tps.append(value_col.dtype)

                        field = (expr._columns[0] == col).ifelse(
                            value_col, Scalar(_value_type=value_col.dtype))
                        if isinstance(agg_func, six.string_types):
                            agg = getattr(field, agg_func)()
                        else:
                            func = agg_func()

                            class ActualAgg(object):
                                def buffer(self):
                                    return func.buffer()

                                def __call__(self, buffer, value):
                                    if value is None:
                                        return
                                    func(buffer, value)

                                def merge(self, buffer, pbuffer):
                                    func.merge(buffer, pbuffer)

                                def getvalue(self, buffer):
                                    return func.getvalue(buffer)

                            agg = field.agg(ActualAgg)
                        if expr.fill_value is not None:
                            agg = agg.fillna(expr.fill_value)
                        agg = agg.rename(name)
                        aggs.append(agg)

            new_expr._schema = Schema.from_lists(names, tps)

            pivoted = expr.input.groupby(expr._group).aggregate(aggs, sort_by_name=False)
            self._sub(new_expr, pivoted)

            # trigger refresh of dynamic operations
            refresh_dynamic(pivoted, self._dag)

        return CollectionExpr(_schema=DynamicSchema.from_lists(group_names, group_types),
                              _deps=[(columns_expr, callback)])

    def _get_pivot_table_sub_expr(self, expr):
        if expr._columns is None:
            return self._get_pivot_table_sub_expr_without_columns(expr)
        else:
            return self._get_pivot_table_sub_expr_with_columns(expr)
