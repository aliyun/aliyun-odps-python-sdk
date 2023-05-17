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

from collections import OrderedDict

import pytest

from ....models import TableSchema
from ... import output, types
from ..expressions import CollectionExpr, ExpressionError
from ..collections import SampledCollectionExpr, AppendIDCollectionExpr, SplitCollectionExpr


@pytest.fixture
def test_exprs(odps):
    from .core import MockTable

    schema = TableSchema.from_lists(types._data_types.keys(), types._data_types.values())
    expr = CollectionExpr(_source_data=None, _schema=schema)
    sourced_expr = CollectionExpr(_source_data=MockTable(client=odps.rest), _schema=schema)
    return (expr, sourced_expr)


def test_sort(test_exprs):
    test_expr, sourced_expr = test_exprs
    sorted_expr = test_expr.sort(test_expr.int64)
    assert isinstance(sorted_expr, CollectionExpr)
    assert sorted_expr._schema == test_expr._schema
    assert list(sorted_expr._ascending) == [True]

    sorted_expr = test_expr.sort_values([test_expr.float32, 'string'])
    assert isinstance(sorted_expr, CollectionExpr)
    assert sorted_expr._schema == test_expr._schema
    assert list(sorted_expr._ascending) == [True] * 2

    sorted_expr = test_expr.sort([test_expr.decimal, 'boolean', 'string'], ascending=False)
    assert isinstance(sorted_expr, CollectionExpr)
    assert sorted_expr._schema == test_expr._schema
    assert list(sorted_expr._ascending) == [False] * 3

    sorted_expr = test_expr.sort([test_expr.int8, 'datetime', 'float64'],
                                 ascending=[False, True, False])
    assert isinstance(sorted_expr, CollectionExpr)
    assert sorted_expr._schema == test_expr._schema
    assert list(sorted_expr._ascending) == [False, True, False]

    sorted_expr = test_expr.sort([-test_expr.int8, 'datetime', 'float64'])
    assert isinstance(sorted_expr, CollectionExpr)
    assert sorted_expr._schema == test_expr._schema
    assert list(sorted_expr._ascending) == [False, True, True]


def test_distinct(test_exprs):
    test_expr, sourced_expr = test_exprs
    distinct = test_expr.distinct()
    assert isinstance(distinct, CollectionExpr)
    assert distinct._schema == test_expr._schema

    distinct = test_expr.distinct(test_expr.string)
    assert isinstance(distinct, CollectionExpr)
    assert distinct._schema == test_expr[[test_expr.string]]._schema

    distinct = test_expr.distinct([test_expr.boolean, 'decimal'])
    assert isinstance(distinct, CollectionExpr)
    assert distinct._schema == test_expr[['boolean', 'decimal']]._schema

    pytest.raises(ExpressionError, lambda: test_expr['boolean', test_expr.string.unique()])


def test_map_reduce(test_exprs):
    test_expr, sourced_expr = test_exprs

    @output(['id', 'name', 'rating'], ['int', 'string', 'int'])
    def mapper(row):
        yield row.int64, row.string, row.int32

    @output(['name', 'rating'], ['string', 'int'])
    def reducer(_):
        i = [0]
        def h(row):
            if i[0] <= 1:
                yield row.name, row.rating
        return h

    expr = test_expr.map_reduce(mapper._func, mapper_output_types=mapper.output_types)
    assert len(expr.schema.columns) == len(mapper.output_types)
    assert expr.schema.types == [types.validate_data_type(tps) for tps in mapper.output_types]

    expr = test_expr.map_reduce(mapper, reducer, group='name',
                                sort='rating', ascending=False)
    assert expr.schema.names == ['name', 'rating']
    assert len(expr.input._sort_fields) == 2
    assert expr.input._sort_fields[0]._ascending is True
    assert expr.input._sort_fields[1]._ascending is False

    expr = test_expr.map_reduce(mapper, reducer, group='name',
                                sort=['rating', 'id'], ascending=[False, True])

    assert expr.schema.names == ['name', 'rating']
    assert len(expr.input._sort_fields) == 3
    assert expr.input._sort_fields[0]._ascending is True
    assert expr.input._sort_fields[1]._ascending is False
    assert expr.input._sort_fields[2]._ascending is True

    expr = test_expr.map_reduce(mapper, reducer, group='name',
                                sort=['rating', 'id'], ascending=False)

    assert expr.schema.names == ['name', 'rating']
    assert len(expr.input._sort_fields) == 3
    assert expr.input._sort_fields[0]._ascending is True
    assert expr.input._sort_fields[1]._ascending is False
    assert expr.input._sort_fields[2]._ascending is False

    expr = test_expr.map_reduce(mapper, reducer, group='name',
                                sort=['name', 'rating', 'id'],
                                ascending=[False, True, False])

    assert expr.schema.names == ['name', 'rating']
    assert len(expr.input._sort_fields) == 3
    assert expr.input._sort_fields[0]._ascending is False
    assert expr.input._sort_fields[1]._ascending is True
    assert expr.input._sort_fields[2]._ascending is False


def test_sample(test_exprs):
    from ....ml.expr import AlgoCollectionExpr

    test_expr, sourced_expr = test_exprs
    assert isinstance(test_expr.sample(100), SampledCollectionExpr)
    assert isinstance(test_expr.sample(parts=10), SampledCollectionExpr)
    try:
        import pandas
    except ImportError:
        # No pandas: go for XFlow
        assert isinstance(test_expr.sample(frac=0.5), AlgoCollectionExpr)
    else:
        # Otherwise: go for Pandas
        assert isinstance(test_expr.sample(frac=0.5), SampledCollectionExpr)
    assert isinstance(sourced_expr.sample(frac=0.5), AlgoCollectionExpr)

    pytest.raises(ExpressionError, lambda: test_expr.sample())
    pytest.raises(ExpressionError, lambda: test_expr.sample(i=-1))
    pytest.raises(ExpressionError, lambda: test_expr.sample(n=100, frac=0.5))
    pytest.raises(ExpressionError, lambda: test_expr.sample(n=100, parts=10))
    pytest.raises(ExpressionError, lambda: test_expr.sample(frac=0.5, parts=10))
    pytest.raises(ExpressionError, lambda: test_expr.sample(n=100, frac=0.5, parts=10))
    pytest.raises(ExpressionError, lambda: test_expr.sample(frac=-1))
    pytest.raises(ExpressionError, lambda: test_expr.sample(frac=1.5))
    pytest.raises(ExpressionError, lambda: test_expr.sample(parts=10, i=-1))
    pytest.raises(ExpressionError, lambda: test_expr.sample(parts=10, i=10))
    pytest.raises(ExpressionError, lambda: test_expr.sample(parts=10, n=10))
    pytest.raises(ExpressionError, lambda: test_expr.sample(weights='weights', strata='strata'))
    pytest.raises(ExpressionError, lambda: test_expr.sample(frac='Yes:10', strata='strata'))
    pytest.raises(ExpressionError, lambda: test_expr.sample(frac=set(), strata='strata'))
    pytest.raises(ExpressionError, lambda: test_expr.sample(n=set(), strata='strata'))


def test_pivot(test_exprs):
    from ..dynamic import DynamicMixin

    test_expr, sourced_expr = test_exprs
    expr = test_expr.pivot('string', 'int8', 'float32')

    assert 'string' in expr._schema._name_indexes
    assert len(expr._schema._name_indexes) == 1
    assert 'non_exist' in expr._schema
    assert isinstance(expr['non_exist'], DynamicMixin)

    expr = test_expr.pivot(
        ['string', 'int8'], 'int16', ['datetime', 'string'])

    assert 'string' in expr._schema._name_indexes
    assert 'int8' in expr._schema._name_indexes
    assert len(expr._schema._name_indexes) == 2
    assert 'non_exist' in expr._schema
    assert isinstance(expr['non_exist'], DynamicMixin)

    pytest.raises(ValueError, lambda: test_expr.pivot(
        ['string', 'int8'], ['datetime', 'string'], 'int16'))


def test_pivot_table(test_exprs):
    from ..dynamic import DynamicMixin

    test_expr, sourced_expr = test_exprs
    expr = test_expr.pivot_table(values='int8', rows='float32')
    assert not isinstance(expr, DynamicMixin)
    assert expr.schema.names == ['float32', 'int8_mean']

    expr = test_expr.pivot_table(values=('int16', 'int32'), rows=['float32', 'int8'])
    assert expr.schema.names == ['float32', 'int8', 'int16_mean', 'int32_mean']

    expr = test_expr.pivot_table(values=('int16', 'int32'), rows=['string', 'boolean'],
                                 aggfunc=['mean', 'sum'])
    assert expr.schema.names == ['string', 'boolean', 'int16_mean', 'int32_mean',
                                         'int16_sum', 'int32_sum']
    assert expr.schema.types == [types.string, types.boolean, types.float64, types.float64,
                                         types.int16, types.int32]

    @output(['my_mean'], ['float'])
    class Aggregator(object):
        def buffer(self):
            return [0.0, 0]

        def __call__(self, buffer, val):
            buffer[0] += val
            buffer[1] += 1

        def merge(self, buffer, pbuffer):
            buffer[0] += pbuffer[0]
            buffer[1] += pbuffer[1]

        def getvalue(self, buffer):
            if buffer[1] == 0:
                return 0.0
            return buffer[0] / buffer[1]

    expr = test_expr.pivot_table(values='int16', rows='string', aggfunc=Aggregator)
    assert expr.schema.names == ['string', 'int16_my_mean']
    assert expr.schema.types == [types.string, types.float64]

    aggfunc = OrderedDict([('my_agg', Aggregator), ('my_agg2', Aggregator)])

    expr = test_expr.pivot_table(values='int16', rows='string', aggfunc=aggfunc)
    assert expr.schema.names == ['string', 'int16_my_agg', 'int16_my_agg2']
    assert expr.schema.types == [types.string, types.float64, types.float64]

    expr = test_expr.pivot_table(values='int16', columns='boolean', rows='string')
    assert isinstance(expr, DynamicMixin)


def test_scale_value(test_exprs):
    test_expr, sourced_expr = test_exprs
    expr = test_expr.min_max_scale()
    assert isinstance(expr, CollectionExpr)
    assert expr.dtypes.names == test_expr.dtypes.names

    expr = test_expr.min_max_scale(preserve=True)
    assert isinstance(expr, CollectionExpr)
    assert expr.dtypes.names == test_expr.dtypes.names + \
                         [n + '_scaled' for n in test_expr.dtypes.names
                          if n.startswith('int') or n.startswith('float')]

    expr = test_expr.std_scale()
    assert isinstance(expr, CollectionExpr)
    assert expr.dtypes.names == test_expr.dtypes.names

    expr = test_expr.std_scale(preserve=True)
    assert isinstance(expr, CollectionExpr)
    assert expr.dtypes.names == test_expr.dtypes.names + \
                         [n + '_scaled' for n in test_expr.dtypes.names
                          if n.startswith('int') or n.startswith('float')]


def test_apply_map():
    from ..collections import ProjectCollectionExpr, Column
    from ..element import MappedExpr

    schema = TableSchema.from_lists(['idx', 'f1', 'f2', 'f3'], [types.int64] + [types.float64] * 3)
    expr = CollectionExpr(_source_data=None, _schema=schema)

    pytest.raises(ValueError, lambda: expr.applymap(lambda v: v + 1, columns='idx', excludes='f1'))

    mapped = expr.applymap(lambda v: v + 1)
    assert isinstance(mapped, ProjectCollectionExpr)
    for c in mapped._fields:
        assert isinstance(c, MappedExpr)

    mapped = expr.applymap(lambda v: v + 1, columns='f1')
    assert isinstance(mapped, ProjectCollectionExpr)
    for c in mapped._fields:
        assert isinstance(c, MappedExpr if c.name == 'f1' else Column)

    map_cols = set(['f1', 'f2', 'f3'])
    mapped = expr.applymap(lambda v: v + 1, columns=map_cols)
    assert isinstance(mapped, ProjectCollectionExpr)
    for c in mapped._fields:
        assert isinstance(c, MappedExpr if c.name in map_cols else Column)

    mapped = expr.applymap(lambda v: v + 1, excludes='idx')
    assert isinstance(mapped, ProjectCollectionExpr)
    for c in mapped._fields:
        assert isinstance(c, Column if c.name == 'idx' else MappedExpr)

    exc_cols = set(['idx', 'f1'])
    mapped = expr.applymap(lambda v: v + 1, excludes=exc_cols)
    assert isinstance(mapped, ProjectCollectionExpr)
    for c in mapped._fields:
        assert isinstance(c, Column if c.name in exc_cols else MappedExpr)


def test_callable_column():
    from ..collections import ProjectCollectionExpr
    from ..expressions import CallableColumn

    schema = TableSchema.from_lists(['name', 'f1', 'append_id'], [types.string, types.float64, types.int64])
    expr = CollectionExpr(_source_data=None, _schema=schema)
    assert isinstance(expr.append_id, CallableColumn)
    assert not isinstance(expr.f1, CallableColumn)

    projected = expr[expr.name, expr.append_id]
    assert isinstance(projected, ProjectCollectionExpr)
    assert projected.schema.names == ['name', 'append_id']

    projected = expr[expr.name, expr.f1]
    assert not isinstance(projected.append_id, CallableColumn)

    appended = expr.append_id(id_col='id_col')
    assert 'id_col' in appended.schema


def test_append_id(test_exprs):
    from ....ml.expr import AlgoCollectionExpr

    test_expr, sourced_expr = test_exprs
    expr = test_expr.append_id(id_col='id_col')
    try:
        import pandas
    except ImportError:
        # No pandas: go for XFlow
        assert isinstance(expr, AlgoCollectionExpr)
    else:
        # Otherwise: go for Pandas
        assert isinstance(expr, AppendIDCollectionExpr)
    assert 'id_col' in expr.schema

    assert isinstance(sourced_expr.append_id(), AlgoCollectionExpr)


def test_split(test_exprs):
    from ....ml.expr import AlgoCollectionExpr

    test_expr, sourced_expr = test_exprs
    expr1, expr2 = test_expr.split(0.6)
    try:
        import pandas
    except ImportError:
        # No pandas: go for XFlow
        assert isinstance(expr1, AlgoCollectionExpr)
        assert isinstance(expr2, AlgoCollectionExpr)
    else:
        # Otherwise: go for Pandas
        assert isinstance(expr1, SplitCollectionExpr)
        assert isinstance(expr2, SplitCollectionExpr)
        assert (expr1._split_id, expr2._split_id) == (0, 1)
