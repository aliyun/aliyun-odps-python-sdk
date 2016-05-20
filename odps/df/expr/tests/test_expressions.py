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
from odps.models import Schema
from odps.df.expr.expressions import *
from odps.df.expr import errors
from odps.df.expr.tests.core import MockTable
from odps.df.expr.arithmetic import Add


class Test(TestBase):
    def setup(self):
        schema = Schema.from_lists(['name', 'id', 'fid'], [types.string, types.int64, types.float64])
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        table._client = self.config.odps.rest
        self.expr = CollectionExpr(_source_data=table, _schema=schema)

    def testProjection(self):
        projected = self.expr['name', self.expr.id.rename('new_id')]

        self.assertIsInstance(projected, CollectionExpr)
        self.assertEqual(projected._schema,
                         Schema.from_lists(['name', 'new_id'], [types.string, types.int64]))

        projected = self.expr[[self.expr.name, self.expr.id.astype('string')]]

        self.assertIsInstance(projected, ProjectCollectionExpr)
        self.assertEqual(projected._schema,
                         Schema.from_lists(['name', 'id'], [types.string, types.string]))

        projected = self.expr.select(self.expr.name, Scalar('abc').rename('word'), size=5)

        self.assertIsInstance(projected, ProjectCollectionExpr)
        self.assertEqual(projected._schema,
                         Schema.from_lists(['name', 'word', 'size'],
                                           [types.string, types.string, types.int8]))
        self.assertIsInstance(projected._fields[1], StringScalar)
        self.assertEqual(projected._fields[1].value, 'abc')
        self.assertIsInstance(projected._fields[2], Int8Scalar)
        self.assertEqual(projected._fields[2].value, 5)

        expr = self.expr[lambda x: x.exclude('id')]
        self.assertEqual(expr.schema.names, [n for n in expr.schema.names if n != 'id'])

        self.assertRaises(ExpressionError, lambda: self.expr[self.expr.distinct('id', 'fid'), 'name'])
        self.assertRaises(ExpressionError, lambda: self.expr[[self.expr.id + self.expr.fid]])

        with option_context() as options:
            options.interactive = True

            self.assertRaises(ExpressionError, lambda: self.expr['name', 'id'][[self.expr.name, ]])

        self.assertRaises(ExpressionError, lambda: self.expr[self.expr.name])
        self.assertRaises(ExpressionError, lambda: self.expr['name', self.expr.groupby('name').id.sum()])

    def testFilter(self):
        filtered = self.expr[(self.expr.id < 10) & (self.expr.name == 'test')]

        self.assertIsInstance(filtered, FilterCollectionExpr)

        filtered = self.expr.filter(self.expr.id < 10, self.expr.name == 'test')

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

    def testSequenceCache(self):
        df = self.expr.name
        self.assertRaises(ExpressionError, lambda: df.cache())

    def testExprFieldValidation(self):
        df = self.expr
        self.assertRaises(errors.ExpressionError, lambda: df[df[:10].id])

        df2 = self.expr[['id']]
        self.assertRaises(errors.ExpressionError, lambda: df[df2.id])

if __name__ == '__main__':
    unittest.main()
