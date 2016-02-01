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


from odps.tests.core import TestBase
from odps.config import option_context
from odps.compat import unittest
from odps.df.expr.expressions import *
from odps.df.expr import errors
from odps.df.expr.tests.core import MockTable
from odps.df.expr.arithmetic import Add, GreaterEqual
from odps.df import types
from odps.df.expr.merge import *
from odps.df.expr.arithmetic import Equal


class Test(TestBase):
    def setup(self):
        schema = Schema.from_lists(['name', 'id', 'fid'], [types.string, types.int64, types.float64])
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        self.expr = CollectionExpr(_source_data=table, _schema=schema)
        table1 = MockTable(name='pyodps_test_expr_table1', schema=schema)
        self.expr1 = CollectionExpr(_source_data=table1, _schema=schema)
        table2 = MockTable(name='pyodps_test_expr_table2', schema=schema)
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

    def testJoin(self):
        e = self.expr
        e1 = self.expr1
        e2 = self.expr2
        joined = e.join(e1, ['fid'], suffix=('_tl', '_tr'))
        self.assertIsInstance(joined, JoinCollectionExpr)
        self.assertIsInstance(joined, InnerJoin)
        self.assertNotIsInstance(joined, LeftJoin)
        self.assertIsInstance(joined._predicate, Equal)
        self.assertEqual(joined._lhs, e)
        self.assertEqual(joined._rhs, e1)
        self.assertEqual(joined._how, 'INNER')
        self.assertEqual(sorted(joined._schema.names), sorted(['name_tl', 'id_tl', 'fid_tl', 'name_tr', 'id_tr', 'fid_tr']))
        self.assertEqual(sorted([t.name for t in joined._schema.types]), sorted(['string', 'int64', 'float64', 'string', 'int64', 'float64']))

        joined = e.inner_join(e1, ['fid', 'id'])
        self.assertIsInstance(joined, InnerJoin)
        self.assertNotIsInstance(joined, LeftJoin)
        pred = joined._predicate.args[0]
        self.assertIsInstance(pred, Equal)
        self.assertEqual(pred._lhs.name, 'fid')
        self.assertEqual(pred._rhs.name, 'fid')
        pred = joined._predicate.args[1]
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

        #projection on join
        e1 = self.expr1
        e = self.expr
        joined = e.join(e1, ['fid'])
        project = joined[e1, e.name]
        self.assertIsInstance(project, ProjectCollectionExpr)
        self.assertSequenceEqual(project._schema.names, ['name_y', 'id_y', 'fid_y', 'name_x'])

        # on is empty, on source is eqaul, on field cannot transformed, other how
        self.assertRaises(ValueError, lambda: e.join(e1, ['']))
        self.assertRaises(ExpressionError, lambda: e.join(e1, [()]))
        self.assertRaises(ExpressionError, lambda: e.join(e1, e.fid == e.fid))
        self.assertRaises(TypeError, lambda: e.join(e1, e.name == e1.fid))

    def testProjection(self):
        projected = self.expr['name', self.expr.id.rename('new_id')]

        self.assertIsInstance(projected, CollectionExpr)
        self.assertEqual(projected._schema,
                         Schema.from_lists(['name', 'new_id'], [types.string, types.int64]))

        projected = self.expr[[self.expr.name, self.expr.id.astype('string')]]

        self.assertIsInstance(projected, ProjectCollectionExpr)
        self.assertEqual(projected._schema,
                         Schema.from_lists(['name', 'id'], [types.string, types.string]))

        self.assertRaises(ExpressionError, lambda: self.expr[[self.expr.id + self.expr.fid]])

        with option_context() as options:
            options.interactive = True

            self.assertRaises(ExpressionError, lambda: self.expr['name', 'id'][[self.expr.name, ]])

    def testFilter(self):
        filtered = self.expr[(self.expr.id < 10) & (self.expr.name == 'test')]

        self.assertIsInstance(filtered, FilterCollectionExpr)

    def testSlice(self):
        sliced = self.expr[:100]

        self.assertIsInstance(sliced, SliceCollectionExpr)
        self.assertEqual(sliced._schema, self.expr._schema)
        self.assertIsInstance(sliced._indexes, tuple)

        not_sliced = self.expr[:]

        self.assertNotIsInstance(not_sliced, SliceCollectionExpr)
        self.assertIsInstance(not_sliced, CollectionExpr)

    def testAsType(self):
        fid = self.expr.id.astype('float')

        self.assertIsInstance(fid._source_data_type, types.Int64)
        self.assertIsInstance(fid._data_type, types.Float64)
        self.assertIsInstance(fid, Float64SequenceExpr)
        self.assertNotIsInstance(fid, Int64SequenceExpr)

        int_fid = fid.astype('int')

        self.assertIsInstance(int_fid._source_data_type, types.Int64)
        self.assertIsInstance(int_fid._data_type, types.Int64)
        self.assertIsInstance(int_fid, Int64SequenceExpr)
        self.assertNotIsInstance(int_fid, Float64SequenceExpr)

        float_fid = (fid + 1).astype('float32')

        self.assertIsInstance(float_fid, Float32SequenceExpr)
        self.assertNotIsInstance(float_fid, Int32SequenceExpr)
        self.assertIsInstance(float_fid, AsTypedSequenceExpr)

    def testRename(self):
        new_id = self.expr.id.rename('new_id')

        self.assertIsInstance(new_id, SequenceExpr)
        self.assertEqual(new_id._source_name, 'id')
        self.assertEqual(new_id._name, 'new_id')

        double_new_id = new_id.rename('2new_id')

        self.assertIsInstance(double_new_id, SequenceExpr)
        self.assertEqual(double_new_id._source_name, 'id')
        self.assertEqual(double_new_id._name, '2new_id')

        self.assertIsNot(double_new_id, new_id)

        add_id = (self.expr.id + self.expr.fid).rename('add_id')
        self.assertIsInstance(add_id, Float64SequenceExpr)
        self.assertNotIsInstance(add_id, Int64SequenceExpr)
        self.assertIsNone(add_id._source_name)
        self.assertIsInstance(add_id, Add)
        self.assertEqual(add_id.name, 'add_id')
        self.assertIsInstance(add_id._lhs, Int64SequenceExpr)
        self.assertIsInstance(add_id._rhs, Float64SequenceExpr)
        self.assertEqual(add_id._lhs._source_name, 'id')
        self.assertEqual(add_id._rhs._source_name, 'fid')

        add_scalar_id = (self.expr.id + 5).rename('add_s_id')
        self.assertNotIsInstance(add_scalar_id, Float64SequenceExpr)
        self.assertIsInstance(add_scalar_id, Int64SequenceExpr)
        self.assertIsInstance(add_scalar_id, Add)
        self.assertEqual(add_scalar_id.name, 'add_s_id')
        self.assertEqual(add_scalar_id._lhs._source_name, 'id')

    def testNewSequence(self):
        column = Column(_data_type='int32')

        self.assertIn(Int32SequenceExpr, type(column).mro())
        self.assertIsInstance(column, Int32SequenceExpr)

        column = type(column)._new(_data_type='string')
        self.assertNotIn(Int32SequenceExpr, type(column).mro())
        self.assertIn(StringSequenceExpr, type(column).mro())
        self.assertIsInstance(column, StringSequenceExpr)
        self.assertNotIsInstance(column, Int32SequenceExpr)
        self.assertIsInstance(column, Column)

        seq = SequenceExpr(_data_type='int64')
        self.assertIsInstance(seq, Int64SequenceExpr)

        seq = BooleanSequenceExpr(_data_type='boolean')
        self.assertIsInstance(seq, BooleanSequenceExpr)

        seq = DatetimeSequenceExpr(_data_type='float32')
        self.assertIsInstance(seq, Float32SequenceExpr)

        class Int64Column(Column):
            __slots__ = 'test',

        column = Int64Column(_data_type='float64', test='value')

        self.assertIsInstance(column, Float64SequenceExpr)
        self.assertNotIsInstance(column, Int64SequenceExpr)

        column = type(column)._new(_data_type='int8', test=column.test)
        self.assertEqual(column.test, 'value')
        self.assertIsInstance(column, Int8SequenceExpr)
        self.assertNotIsInstance(column, Float64SequenceExpr)
        self.assertNotIsInstance(column, Int64SequenceExpr)
        self.assertIsInstance(column, Int64Column)

        class Int64Column(Int64SequenceExpr):
            pass

        column = Int64Column(_data_type='float64')

        self.assertIsInstance(column, Float64SequenceExpr)
        self.assertNotIsInstance(column, Int64SequenceExpr)

        column = type(column)._new(_data_type='int8')
        self.assertIsInstance(column, Int8SequenceExpr)
        self.assertNotIsInstance(column, Float64SequenceExpr)
        self.assertNotIsInstance(column, Int64SequenceExpr)
        self.assertNotIsInstance(column, Int64Column)

    def testExprFieldValidation(self):
        df = self.expr
        self.assertRaises(errors.ExpressionError, lambda: df[df[:10].id])

        df2 = self.expr[['id']]
        self.assertRaises(errors.ExpressionError, lambda: df[df2.id])

    def testUnion(self):
        df = self.expr
        df1 = self.expr1
        df2 = self.expr2

        expr = df.name.rename('name_x').union(df1.join(df2, 'name')['name_x'])
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

        expr = df.concat(df1)
        self.assertIsInstance(expr, UnionCollectionExpr)
        self.assertIsInstance(expr._lhs, CollectionExpr)
        self.assertIsInstance(expr._rhs, CollectionExpr)

        expr = df['name', 'id'].concat(df1['name', 'id'])
        self.assertIsInstance(expr, UnionCollectionExpr)
        self.assertIsInstance(expr._lhs, ProjectCollectionExpr)
        self.assertIsInstance(expr._rhs, ProjectCollectionExpr)

if __name__ == '__main__':
    unittest.main()
