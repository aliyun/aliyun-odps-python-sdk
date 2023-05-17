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
from collections import namedtuple

import pytest

from ....compat import reduce
from ... import types, Scalar, DataFrame
from ..tests.core import MockTable
from ..merge import *


@pytest.fixture
def exprs(odps):
    schema = TableSchema.from_lists(['name', 'id', 'fid'], [types.string, types.int64, types.float64])
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
    table._client = odps.rest
    expr = CollectionExpr(_source_data=table, _schema=schema)
    table1 = MockTable(name='pyodps_test_expr_table1', table_schema=schema)
    table1._client = odps.rest
    expr1 = CollectionExpr(_source_data=table1, _schema=schema)
    table2 = MockTable(name='pyodps_test_expr_table2', table_schema=schema)
    table2._client = odps.rest
    expr2 = CollectionExpr(_source_data=table2, _schema=schema)

    nt = namedtuple("NT", "expr, expr1, expr2")
    return nt(expr, expr1, expr2)


def test_simple_join():
    schema = TableSchema.from_lists(['name', 'id'], [types.string, types.int64])
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
    expr = CollectionExpr(_source_data=table, _schema=schema)

    schema1 = TableSchema.from_lists(['id', 'value'], [types.int64, types.string])
    table1 = MockTable(name='pyodps_test_expr_table1', table_schema=schema1)
    expr1 = CollectionExpr(_source_data=table1, _schema=schema1)

    schema2 = TableSchema.from_lists(['value', 'num'], [types.string, types.float64])
    table2 = MockTable(name='pyodps_test_expr_table2', table_schema=schema2)
    expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

    df = expr.join(expr1).join(expr2)
    assert df.schema.names == ['name', 'id', 'value', 'num']


def test_join_map_join(exprs):
    e = exprs.expr
    e1 = exprs.expr1
    e2 = exprs.expr2
    pytest.raises(ExpressionError, lambda: e.join(e1, on=[]))
    joined = e.join(e1, mapjoin=True)
    joined = e2.join(joined, mapjoin=True, on=[])
    assert joined._predicate is None

    e.join(e1, mapjoin=True).join(e2, mapjoin=True)


def test_join_skew_join(exprs):
    e = exprs.expr
    e1 = exprs.expr1
    joined = e.join(e1, ['fid'], skewjoin=True)
    assert joined._skewjoin is True
    joined = e.join(e1, ['fid'], skewjoin='fid')
    assert joined._skewjoin == ['fid']
    joined = e.join(e1, ['fid'], skewjoin=['fid'])
    assert joined._skewjoin == ['fid']
    joined = e.join(e1, ['fid'], skewjoin=[{'fid': 0.5}])
    assert joined._skewjoin == ['fid']
    assert joined._skewjoin_values == [[0.5]]

    pytest.raises(TypeError, lambda: e.join(e1, ['fid'], skewjoin=1))
    pytest.raises(ValueError, lambda: e.join(e1, ['fid'], skewjoin={'non-exist': 1}))
    pytest.raises(ValueError, lambda: e.join(e1, ['fid'], skewjoin=['non-exist']))
    pytest.raises(ValueError, lambda: e.join(e1, ['fid'], skewjoin=[{'non-exist': 1}]))
    pytest.raises(ValueError, lambda: e.join(e1, ['fid'], skewjoin=[{'fid': 1}, {'id': 2}])
    )


def test_join(exprs):
    e = exprs.expr
    e1 = exprs.expr1
    e2 = exprs.expr2
    joined = e.join(e1, ['fid'], suffixes=('_tl', '_tr'))
    assert isinstance(joined, JoinCollectionExpr)
    assert isinstance(joined, InnerJoin)
    assert not isinstance(joined, LeftJoin)
    assert isinstance(joined._predicate[0], Equal)
    assert joined._lhs == e
    assert joined._rhs == e1
    assert joined._how == 'INNER'
    assert sorted(joined._schema.names) == sorted(['name_tl', 'id_tl', 'name_tr', 'id_tr', 'fid'])
    assert sorted([t.name for t in joined._schema.types]) == sorted(['string', 'int64', 'string', 'int64', 'float64'])

    joined = e.inner_join(e1, ['fid', 'id'])
    assert isinstance(joined, InnerJoin)
    assert not isinstance(joined, LeftJoin)
    predicate = reduce(operator.and_, joined._predicate)
    pred = predicate.args[0]
    assert isinstance(pred, Equal)
    assert pred._lhs.name == 'fid'
    assert pred._rhs.name == 'fid'
    pred = predicate.args[1]
    assert isinstance(pred, Equal)
    assert pred._lhs.name == 'id'
    assert pred._rhs.name == 'id'
    assert joined._lhs == e
    assert joined._rhs == e1
    assert joined._how == 'INNER'

    joined = e1.left_join(e, e.name == e1.name)
    assert isinstance(joined, LeftJoin)
    assert joined._lhs == e1
    assert joined._rhs == e
    assert joined._how == 'LEFT OUTER'

    joined = e1.left_join(e, e.name == e1.name, merge_columns=True)
    assert isinstance(joined, ProjectCollectionExpr)
    assert isinstance(joined._input, LeftJoin)
    assert joined._input._lhs == e1
    assert joined._input._rhs == e
    assert joined._input._how == 'LEFT OUTER'
    assert 'name' in joined.schema.names
    assert 'id' not in joined.schema.names
    assert 'fid' not in joined.schema.names

    joined = e1.right_join(e, [e.fid == e1.fid, e1.name == e.name])
    assert isinstance(joined, RightJoin)
    assert joined._lhs == e1
    assert joined._rhs == e
    assert joined._how == 'RIGHT OUTER'

    joined = e1.right_join(e, [e.id == e1.id])
    assert isinstance(joined, RightJoin)
    # self.assertEqual(joined._predicates, [(e.id, e1.id)])
    assert joined._lhs == e1
    assert joined._rhs == e
    assert joined._how == 'RIGHT OUTER'

    joined = e1.outer_join(e, [('fid', 'fid'), ('name', 'name')])
    assert isinstance(joined, OuterJoin)
    assert joined._lhs == e1
    assert joined._rhs == e
    assert joined._how == 'FULL OUTER'

    joined = e.join(e1, ['fid', 'name'], 'OuTer')
    assert isinstance(joined, OuterJoin)
    assert not isinstance(joined, InnerJoin)
    # self.assertEqual(joined._predicates, [(e.fid, e1.fid), (e.name, e1.name)])
    assert joined._lhs == e
    assert joined._rhs == e1
    assert joined._how == 'FULL OUTER'

    # join + in projection
    e = e['fid', 'name']
    joined = e.join(e1, ['fid'], 'LEFT')
    assert isinstance(joined, LeftJoin)
    assert not isinstance(joined, InnerJoin)
    assert joined._lhs == e
    assert isinstance(joined._lhs, ProjectCollectionExpr)
    assert joined._rhs == e1
    assert joined._how == 'LEFT OUTER'

    e1 = e1['fid', 'id']
    joined = e.join(e1, [(e.fid, e1.fid)])
    assert isinstance(joined, JoinCollectionExpr)
    assert isinstance(joined, InnerJoin)
    assert joined._lhs == e
    assert joined._rhs == e1
    assert joined._how == 'INNER'

    # projection on join
    e1 = exprs.expr1
    e = exprs.expr
    joined = e.join(e1, ['fid'])
    project = joined[e1, e.name]
    assert isinstance(project, ProjectCollectionExpr)
    assert project._schema.names == ['name_y', 'id_y', 'fid', 'name_x']

    # on is empty, on source is eqaul, on field cannot transformed, other how
    pytest.raises(ValueError, lambda: e.join(e1, ['']))
    pytest.raises(ExpressionError, lambda: e.join(e1, [()]))
    pytest.raises(ExpressionError, lambda: e.join(e1, e.fid == e.fid))
    pytest.raises(TypeError, lambda: e.join(e1, e.name == e1.fid))

    e1 = exprs.expr1.select(name2=exprs.expr1.name, id2=exprs.expr1.id)
    joined = e.join(e1, on=(e.name == Scalar('tt') + e1.name2))
    project = joined[e, e1['name2', ]]
    assert isinstance(joined, JoinCollectionExpr)
    assert isinstance(project, ProjectCollectionExpr)
    assert next(f for f in project._fields if f.name == 'name2').input is joined

    with pytest.raises(ExpressionError):
        exprs.expr.join(exprs.expr1, on=exprs.expr.id == exprs.expr2.id)

    with pytest.raises(ExpressionError):
        expr = exprs.expr.join(exprs.expr1, on='id')
        exprs.expr.join(expr, on=exprs.expr.id == exprs.expr.id)

    # first __setitem__, then join
    e = exprs.expr
    e1 = exprs.expr1['name', exprs.expr1.fid.rename('fid2')]
    e1['fid2'] = e1.fid2.astype('string')
    joined = e.join(e1, on='name')
    assert isinstance(joined, JoinCollectionExpr)
    assert joined.dtypes.names == ['name', 'id', 'fid', 'fid2']
    assert joined.fid2.dtype.name == 'string'


def test_complex_join(exprs):
    df = None
    for i in range(30):
        if df is None:
            df = exprs.expr
        else:
            viewed = exprs.expr.view()[
                'id',
                lambda x: x.name.rename('name%d' % i),
                lambda x: x.fid.rename('fid%d' % i)
            ]
            df = df.outer_join(viewed, on='id', suffixes=('', '_x'))[
                df, viewed.exclude('id')]

    assert any(field.endswith('_x') for field in df.schema.names) is False


def test_join_right_twice(exprs):
    expr = exprs.expr.join(exprs.expr1, on='id')['id', exprs.expr.fid]
    expr = expr.join(exprs.expr1, on='id')[exprs.expr1.name,]

    assert expr._fields[0].input is expr.input


def test_union(exprs):
    df = exprs.expr
    df1 = exprs.expr1
    df2 = exprs.expr2

    expr = df.name.union(df1.join(df2, 'name')[df2['name'], ].name)
    assert isinstance(expr, UnionCollectionExpr)
    assert isinstance(expr._lhs, ProjectCollectionExpr)
    assert isinstance(expr._rhs, ProjectCollectionExpr)

    expr = df.union(df1)
    assert isinstance(expr, UnionCollectionExpr)
    assert isinstance(expr._lhs, CollectionExpr)
    assert isinstance(expr._rhs, CollectionExpr)

    expr = df['name', 'id'].union(df1['name', 'id'])
    assert isinstance(expr, UnionCollectionExpr)
    assert isinstance(expr._lhs, ProjectCollectionExpr)
    assert isinstance(expr._rhs, ProjectCollectionExpr)

    expr = df[df.name.rename('new_name'), 'id'].union(df1[df1.name.rename('new_name'), 'id'])
    assert isinstance(expr, UnionCollectionExpr)
    assert isinstance(expr._lhs, ProjectCollectionExpr)
    assert isinstance(expr._rhs, ProjectCollectionExpr)
    assert 'new_name' in expr._lhs.schema.names
    assert 'new_name' in expr._rhs.schema.names

    expr = df.concat(df1)
    assert isinstance(expr, UnionCollectionExpr)
    assert isinstance(expr._lhs, CollectionExpr)
    assert isinstance(expr._rhs, CollectionExpr)

    expr = df['name', 'id'].concat(df1['name', 'id'])
    assert isinstance(expr, UnionCollectionExpr)
    assert isinstance(expr._lhs, ProjectCollectionExpr)
    assert isinstance(expr._rhs, ProjectCollectionExpr)


def test_concat(odps, exprs):
    from ....ml.expr import AlgoCollectionExpr

    schema = TableSchema.from_lists(['name', 'id'], [types.string, types.int64])
    df = CollectionExpr(_source_data=None, _schema=schema)
    df1 = CollectionExpr(_source_data=None, _schema=schema)
    df2 = CollectionExpr(_source_data=None, _schema=schema)

    schema = TableSchema.from_lists(['fid', 'fid2'], [types.int64, types.float64])
    df3 = CollectionExpr(_source_data=None, _schema=schema)

    schema = TableSchema.from_lists(['fid', 'fid2'], [types.int64, types.float64])
    table = MockTable(name='pyodps_test_expr_table2', table_schema=schema)
    table._client = odps.rest
    df4 = CollectionExpr(_source_data=table, _schema=schema)

    expr = df.concat([df1, df2])
    assert isinstance(expr, UnionCollectionExpr)
    assert isinstance(expr._lhs, CollectionExpr)
    assert isinstance(expr._rhs, CollectionExpr)

    expr = df.concat(df3, axis=1)
    try:
        import pandas as pd
        assert isinstance(expr, ConcatCollectionExpr)
        assert isinstance(expr._lhs, CollectionExpr)
        assert isinstance(expr._rhs, CollectionExpr)
    except ImportError:
        assert isinstance(expr, AlgoCollectionExpr)
    assert 'name' in expr.schema.names
    assert 'id' in expr.schema.names
    assert 'fid' in expr.schema.names
    assert 'fid2' in expr.schema.names

    expr = df.concat(df4, axis=1)
    assert isinstance(expr, AlgoCollectionExpr)
    assert 'name' in expr.schema.names
    assert 'id' in expr.schema.names
    assert 'fid' in expr.schema.names
    assert 'fid2' in expr.schema.names