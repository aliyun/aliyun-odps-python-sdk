#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from odps.tests.core import TestBase
from odps.compat import unittest, reduce
from odps.df.expr.tests.core import MockTable
from odps.df.expr.merge import *
from odps.df import types, Scalar, DataFrame


class Test(TestBase):
    def setup(self):
        schema = Schema.from_lists(['name', 'id', 'fid'], [types.string, types.int64, types.float64])
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        table._client = self.config.odps.rest
        self.expr = CollectionExpr(_source_data=table, _schema=schema)
        table1 = MockTable(name='pyodps_test_expr_table1', schema=schema)
        table1._client = self.config.odps.rest
        self.expr1 = CollectionExpr(_source_data=table1, _schema=schema)
        table2 = MockTable(name='pyodps_test_expr_table2', schema=schema)
        table2._client = self.config.odps.rest
        self.expr2 = CollectionExpr(_source_data=table2, _schema=schema)

    def testSimpleJoin(self):
        schema = Schema.from_lists(['name', 'id'], [types.string, types.int64])
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        expr = CollectionExpr(_source_data=table, _schema=schema)

        schema1 = Schema.from_lists(['id', 'value'], [types.int64, types.string])
        table1 = MockTable(name='pyodps_test_expr_table1', schema=schema1)
        expr1 = CollectionExpr(_source_data=table1, _schema=schema1)

        schema2 = Schema.from_lists(['value', 'num'], [types.string, types.float64])
        table2 = MockTable(name='pyodps_test_expr_table2', schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

        df = expr.join(expr1).join(expr2)
        self.assertEqual(df.schema.names, ['name', 'id', 'value', 'num'])

    def testJoinMapJoin(self):
        e = self.expr
        e1 = self.expr1
        e2 = self.expr2
        self.assertRaises(ExpressionError, lambda: e.join(e1, on=[]))
        joined = e.join(e1, mapjoin=True)
        joined = e2.join(joined, mapjoin=True, on=[])
        self.assertIsNone(joined._predicate)

        e.join(e1, mapjoin=True).join(e2, mapjoin=True)

    def testJoin(self):
        e = self.expr
        e1 = self.expr1
        e2 = self.expr2
        joined = e.join(e1, ['fid'], suffixes=('_tl', '_tr'))
        self.assertIsInstance(joined, JoinCollectionExpr)
        self.assertIsInstance(joined, InnerJoin)
        self.assertNotIsInstance(joined, LeftJoin)
        self.assertIsInstance(joined._predicate[0], Equal)
        self.assertEqual(joined._lhs, e)
        self.assertEqual(joined._rhs, e1)
        self.assertEqual(joined._how, 'INNER')
        self.assertEqual(sorted(joined._schema.names),
                         sorted(['name_tl', 'id_tl', 'name_tr', 'id_tr', 'fid']))
        self.assertEqual(sorted([t.name for t in joined._schema.types]),
                         sorted(['string', 'int64', 'string', 'int64', 'float64']))

        joined = e.inner_join(e1, ['fid', 'id'])
        self.assertIsInstance(joined, InnerJoin)
        self.assertNotIsInstance(joined, LeftJoin)
        predicate = reduce(operator.and_, joined._predicate)
        pred = predicate.args[0]
        self.assertIsInstance(pred, Equal)
        self.assertEqual(pred._lhs.name, 'fid')
        self.assertEqual(pred._rhs.name, 'fid')
        pred = predicate.args[1]
        self.assertIsInstance(pred, Equal)
        self.assertEqual(pred._lhs.name, 'id')
        self.assertEqual(pred._rhs.name, 'id')
        self.assertEqual(joined._lhs, e)
        self.assertEqual(joined._rhs, e1)
        self.assertEqual(joined._how, 'INNER')

        joined = e1.left_join(e, e.name == e1.name)
        self.assertIsInstance(joined, LeftJoin)
        self.assertEqual(joined._lhs, e1)
        self.assertEqual(joined._rhs, e)
        self.assertEqual(joined._how, 'LEFT OUTER')

        joined = e1.left_join(e, e.name == e1.name, merge_columns=True)
        self.assertIsInstance(joined, ProjectCollectionExpr)
        self.assertIsInstance(joined._input, LeftJoin)
        self.assertEqual(joined._input._lhs, e1)
        self.assertEqual(joined._input._rhs, e)
        self.assertEqual(joined._input._how, 'LEFT OUTER')
        self.assertIn('name', joined.schema.names)
        self.assertNotIn('id', joined.schema.names)
        self.assertNotIn('fid', joined.schema.names)

        joined = e1.right_join(e, [e.fid == e1.fid, e1.name == e.name])
        self.assertIsInstance(joined, RightJoin)
        self.assertEqual(joined._lhs, e1)
        self.assertEqual(joined._rhs, e)
        self.assertEqual(joined._how, 'RIGHT OUTER')

        joined = e1.right_join(e, [e.id == e1.id])
        self.assertIsInstance(joined, RightJoin)
        # self.assertEqual(joined._predicates, [(e.id, e1.id)])
        self.assertEqual(joined._lhs, e1)
        self.assertEqual(joined._rhs, e)
        self.assertEqual(joined._how, 'RIGHT OUTER')

        joined = e1.outer_join(e, [('fid', 'fid'), ('name', 'name')])
        self.assertIsInstance(joined, OuterJoin)
        self.assertEqual(joined._lhs, e1)
        self.assertEqual(joined._rhs, e)
        self.assertEqual(joined._how, 'FULL OUTER')

        joined = e.join(e1, ['fid', 'name'], 'OuTer')
        self.assertIsInstance(joined, OuterJoin)
        self.assertNotIsInstance(joined, InnerJoin)
        # self.assertEqual(joined._predicates, [(e.fid, e1.fid), (e.name, e1.name)])
        self.assertEqual(joined._lhs, e)
        self.assertEqual(joined._rhs, e1)
        self.assertEqual(joined._how, 'FULL OUTER')

        # join + in projection
        e = e['fid', 'name']
        joined = e.join(e1, ['fid'], 'LEFT')
        self.assertIsInstance(joined, LeftJoin)
        self.assertNotIsInstance(joined, InnerJoin)
        self.assertEqual(joined._lhs, e)
        self.assertIsInstance(joined._lhs, ProjectCollectionExpr)
        self.assertEqual(joined._rhs, e1)
        self.assertEqual(joined._how, 'LEFT OUTER')

        e1 = e1['fid', 'id']
        joined = e.join(e1, [(e.fid, e1.fid)])
        self.assertIsInstance(joined, JoinCollectionExpr)
        self.assertIsInstance(joined, InnerJoin)
        self.assertEqual(joined._lhs, e)
        self.assertEqual(joined._rhs, e1)
        self.assertEqual(joined._how, 'INNER')

        # projection on join
        e1 = self.expr1
        e = self.expr
        joined = e.join(e1, ['fid'])
        project = joined[e1, e.name]
        self.assertIsInstance(project, ProjectCollectionExpr)
        self.assertSequenceEqual(project._schema.names, ['name_y', 'id_y', 'fid', 'name_x'])

        # on is empty, on source is eqaul, on field cannot transformed, other how
        self.assertRaises(ValueError, lambda: e.join(e1, ['']))
        self.assertRaises(ExpressionError, lambda: e.join(e1, [()]))
        self.assertRaises(ExpressionError, lambda: e.join(e1, e.fid == e.fid))
        self.assertRaises(TypeError, lambda: e.join(e1, e.name == e1.fid))

        e1 = self.expr1.select(name2=self.expr1.name, id2=self.expr1.id)
        joined = e.join(e1, on=(e.name == Scalar('tt') + e1.name2))
        project = joined[e, e1['name2', ]]
        self.assertIsInstance(joined, JoinCollectionExpr)
        self.assertIsInstance(project, ProjectCollectionExpr)
        self.assertIs(next(f for f in project._fields if f.name == 'name2').input, joined)

        with self.assertRaises(ExpressionError):
            self.expr.join(self.expr1, on=self.expr.id == self.expr2.id)

        with self.assertRaises(ExpressionError):
            expr = self.expr.join(self.expr1, on='id')
            self.expr.join(expr, on=self.expr.id == self.expr.id)

        # first __setitem__, then join
        e = self.expr
        e1 = self.expr1['name', self.expr1.fid.rename('fid2')]
        e1['fid2'] = e1.fid2.astype('string')
        joined = e.join(e1, on='name')
        self.assertIsInstance(joined, JoinCollectionExpr)
        self.assertListEqual(joined.dtypes.names, ['name', 'id', 'fid', 'fid2'])
        self.assertEqual(joined.fid2.dtype.name, 'string')

    def testComplexJoin(self):
        df = None
        for i in range(30):
            if df is None:
                df = self.expr
            else:
                viewed = self.expr.view()[
                    'id',
                    lambda x: x.name.rename('name%d' % i),
                    lambda x: x.fid.rename('fid%d' % i)
                ]
                df = df.outer_join(viewed, on='id', suffixes=('', '_x'))[
                    df, viewed.exclude('id')]

        self.assertFalse(any(field.endswith('_x') for field in df.schema.names))

    def testUnion(self):
        df = self.expr
        df1 = self.expr1
        df2 = self.expr2

        expr = df.name.union(df1.join(df2, 'name')[df2['name'], ].name)
        self.assertIsInstance(expr, UnionCollectionExpr)
        self.assertIsInstance(expr._lhs, ProjectCollectionExpr)
        self.assertIsInstance(expr._rhs, ProjectCollectionExpr)

        expr = df.union(df1)
        self.assertIsInstance(expr, UnionCollectionExpr)
        self.assertIsInstance(expr._lhs, CollectionExpr)
        self.assertIsInstance(expr._rhs, CollectionExpr)

        expr = df['name', 'id'].union(df1['name', 'id'])
        self.assertIsInstance(expr, UnionCollectionExpr)
        self.assertIsInstance(expr._lhs, ProjectCollectionExpr)
        self.assertIsInstance(expr._rhs, ProjectCollectionExpr)

        expr = df[df.name.rename('new_name'), 'id'].union(df1[df1.name.rename('new_name'), 'id'])
        self.assertIsInstance(expr, UnionCollectionExpr)
        self.assertIsInstance(expr._lhs, ProjectCollectionExpr)
        self.assertIsInstance(expr._rhs, ProjectCollectionExpr)
        self.assertIn('new_name', expr._lhs.schema.names)
        self.assertIn('new_name', expr._rhs.schema.names)

        expr = df.concat(df1)
        self.assertIsInstance(expr, UnionCollectionExpr)
        self.assertIsInstance(expr._lhs, CollectionExpr)
        self.assertIsInstance(expr._rhs, CollectionExpr)

        expr = df['name', 'id'].concat(df1['name', 'id'])
        self.assertIsInstance(expr, UnionCollectionExpr)
        self.assertIsInstance(expr._lhs, ProjectCollectionExpr)
        self.assertIsInstance(expr._rhs, ProjectCollectionExpr)

    def testConcat(self):
        from odps.ml.expr import AlgoCollectionExpr

        schema = Schema.from_lists(['name', 'id'], [types.string, types.int64])
        df = CollectionExpr(_source_data=None, _schema=schema)
        df1 = CollectionExpr(_source_data=None, _schema=schema)
        df2 = CollectionExpr(_source_data=None, _schema=schema)

        schema = Schema.from_lists(['fid', 'fid2'], [types.int64, types.float64])
        df3 = CollectionExpr(_source_data=None, _schema=schema)

        schema = Schema.from_lists(['fid', 'fid2'], [types.int64, types.float64])
        table = MockTable(name='pyodps_test_expr_table2', schema=schema)
        table._client = self.config.odps.rest
        df4 = CollectionExpr(_source_data=table, _schema=schema)

        expr = df.concat([df1, df2])
        self.assertIsInstance(expr, UnionCollectionExpr)
        self.assertIsInstance(expr._lhs, CollectionExpr)
        self.assertIsInstance(expr._rhs, CollectionExpr)

        expr = df.concat(df3, axis=1)
        try:
            import pandas as pd
            self.assertIsInstance(expr, ConcatCollectionExpr)
            self.assertIsInstance(expr._lhs, CollectionExpr)
            self.assertIsInstance(expr._rhs, CollectionExpr)
        except ImportError:
            self.assertIsInstance(expr, AlgoCollectionExpr)
        self.assertIn('name', expr.schema.names)
        self.assertIn('id', expr.schema.names)
        self.assertIn('fid', expr.schema.names)
        self.assertIn('fid2', expr.schema.names)

        expr = df.concat(df4, axis=1)
        self.assertIsInstance(expr, AlgoCollectionExpr)
        self.assertIn('name', expr.schema.names)
        self.assertIn('id', expr.schema.names)
        self.assertIn('fid', expr.schema.names)
        self.assertIn('fid2', expr.schema.names)

if __name__ == '__main__':
    unittest.main()