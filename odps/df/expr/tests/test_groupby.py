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

import pytest

from ..reduction import *


@pytest.fixture
def src_expr():
    schema = TableSchema.from_lists(types._data_types.keys(), types._data_types.values())
    return CollectionExpr(_source_data=True, _schema=schema)


def test_groupby(src_expr):
    grouped = src_expr.groupby(['int16', 'boolean'])\
        .agg(src_expr.string.sum().rename('sum_string'),
             sum=src_expr.int16.sum())
    assert isinstance(grouped, CollectionExpr)
    assert grouped._schema.names == ['int16', 'boolean', 'sum', 'sum_string']
    assert grouped._schema.types == [types.int16, types.boolean, types.int16, types.string]
    assert isinstance(grouped.sum_string, StringSequenceExpr)

    grouped = src_expr.groupby('datetime').aggregate(
        src_expr.boolean.sum(),
        min_int32=src_expr.int32.min())
    assert grouped._schema.names == ['datetime', 'boolean_sum', 'min_int32']
    assert grouped._schema.types == [types.datetime, types.int64, types.int32]

    grouped = src_expr.groupby(['float32', 'string']).agg(int64std=src_expr.int64.std(ddof=3))
    assert grouped._schema.names == ['float32', 'string', 'int64std']
    assert grouped._schema.types == [types.float32, types.string, types.float64]
    assert grouped._aggregations[0]._ddof == 3

    selected = grouped[[grouped.float32.rename('new_col')]]
    assert selected.schema.names == ['new_col']


def test_groupby_reductions(src_expr):
    expr = src_expr.groupby('string').min()
    assert isinstance(expr, GroupByCollectionExpr)
    assert len(expr.aggregations) > 0
    assert isinstance(expr.aggregations[0], GroupedMin)

    expr = src_expr.groupby('string').max()
    assert isinstance(expr, GroupByCollectionExpr)
    assert len(expr.aggregations) > 0
    assert isinstance(expr.aggregations[0], GroupedMax)

    expr = src_expr.groupby('string').count()
    assert isinstance(expr, SequenceExpr)
    assert isinstance(expr, Int64SequenceExpr)

    expr = src_expr.groupby('string').var()
    assert isinstance(expr, GroupByCollectionExpr)
    assert len(expr.aggregations) > 0
    assert isinstance(expr.aggregations[0], GroupedVar)

    expr = src_expr.groupby('string').sum()
    assert isinstance(expr, GroupByCollectionExpr)
    assert len(expr.aggregations) > 0
    assert isinstance(expr.aggregations[0], GroupedSum)

    expr = src_expr.groupby('string').std()
    assert isinstance(expr, GroupByCollectionExpr)
    assert len(expr.aggregations) > 0
    assert isinstance(expr.aggregations[0], GroupedStd)

    expr = src_expr.groupby('string').mean()
    assert isinstance(expr, GroupByCollectionExpr)
    assert len(expr.aggregations) > 0
    assert isinstance(expr.aggregations[0], GroupedMean)

    expr = src_expr.groupby('string').median()
    assert isinstance(expr, GroupByCollectionExpr)
    assert len(expr.aggregations) > 0
    assert isinstance(expr.aggregations[0], GroupedMedian)

    metric = src_expr.int32.mean() > 10
    field = (metric.ifelse(src_expr.int64.max(), 0) + 1).rename('int64_max')
    expr = src_expr.groupby('string').agg(field)
    assert isinstance(expr, GroupByCollectionExpr)
    assert isinstance(expr.int64_max, Int64SequenceExpr)


def test_groupby_field(src_expr):
    grouped = src_expr.groupby(['int32', 'boolean']).string.sum()
    assert isinstance(grouped, StringSequenceExpr)
    assert isinstance(grouped, GroupedSum)
    assert isinstance(grouped._input, Column)

    grouped = src_expr.groupby('string').int64.count()
    assert isinstance(grouped, Int64SequenceExpr)
    assert isinstance(grouped, GroupedCount)
    assert isinstance(grouped._input, Column)


def test_mutate(src_expr):
    grouped = src_expr.groupby(['int16', src_expr.datetime]).sort(-src_expr.boolean)
    expr = grouped.mutate(grouped.float64.cumsum(), count=grouped.boolean.cumcount())

    assert isinstance(expr, MutateCollectionExpr)
    assert expr._schema.names == ['int16', 'datetime', 'float64_sum', 'count']
    assert expr._schema.types == [types.int16, types.datetime, types.float64, types.int64]


def test_illegal_groupby(src_expr):
    pytest.raises(ExpressionError, lambda: src_expr.groupby('int16').agg(src_expr['string']))
    pytest.raises(ExpressionError,
                      lambda: src_expr.groupby('int16').agg(src_expr['string'] + src_expr['string'].sum()))
    pytest.raises(ExpressionError,
                      lambda: src_expr.groupby('int8').agg(src_expr['boolean', ]['boolean'].sum()))

    grouped = src_expr.groupby('string')
    pytest.raises(ExpressionError, lambda: src_expr.groupby('boolean').agg(grouped.int32.sum()))


def test_backtrack_field(src_expr):
    expr = src_expr[src_expr.int64 < 10].groupby(src_expr.string).agg(s=src_expr.float32.sum())
    assert expr._by[0]._input is expr._input
    assert expr._aggregations[0]._input._input is expr._input
