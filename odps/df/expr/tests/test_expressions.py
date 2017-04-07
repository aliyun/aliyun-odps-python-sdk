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


from odps.tests.core import TestBase
from odps.config import option_context
from odps.compat import unittest
from odps.models import Schema
from odps.df.expr.expressions import *
from odps.df.expr.core import ExprDictionary
from odps.df.expr import errors
from odps.df.expr.tests.core import MockTable
from odps.df.expr.arithmetic import Add


class Test(TestBase):
    def setup(self):
        schema = Schema.from_lists(['name', 'id', 'fid'], [types.string, types.int64, types.float64])
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        table._client = self.config.odps.rest
        self.expr = CollectionExpr(_source_data=table, _schema=schema)

        schema2 = Schema.from_lists(['name', 'id', 'fid'], [types.string, types.int64, types.float64],
                                    ['part1', 'part2'], [types.string, types.int64])
        table2 = MockTable(name='pyodps_test_expr_table2', schema=schema2)
        table2._client = self.config.odps.rest
        self.expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

    def testDir(self):
        expr_dir = dir(self.expr)
        self.assertIn('id', expr_dir)
        self.assertIn('fid', expr_dir)

        new_df = self.expr[self.expr.id, self.expr.fid, self.expr.name.rename('if')]
        self.assertNotIn('if', dir(new_df))

        self.assertEqual(self.expr._id, self.expr.copy()._id)

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

            self.expr['name', 'id'][[self.expr.name, ]]

        self.assertRaises(ExpressionError, lambda: self.expr[self.expr.name])
        self.assertRaises(ExpressionError, lambda: self.expr['name', self.expr.groupby('name').id.sum()])

        expr = self.expr.filter(self.expr.id < 0)
        expr[self.expr.name, self.expr.id]

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

    def testFilterPartition(self):
        self.assertRaises(ExpressionError, lambda: self.expr.filter_partition(None))
        self.assertRaises(ExpressionError, lambda: self.expr.filter_partition('part1=a/part2=1,part1=b/part2=2'))
        self.assertRaises(ExpressionError, lambda: self.expr2.filter_partition('part1/part2=1,part1=b/part2=2'))

        filtered1 = self.expr2.filter_partition('part1=a/part2=1,part1=b/part2=2')
        self.assertIsInstance(filtered1, FilterPartitionCollectionExpr)
        self.assertEqual(filtered1.schema, self.expr.schema)
        self.assertEqual(filtered1.predicate_string, 'part1=a/part2=1,part1=b/part2=2')

        filtered2 = self.expr2.filter_partition('part1=a/part2=1,part1=b/part2=2', exclude=False)
        self.assertIsInstance(filtered2, FilterCollectionExpr)

        try:
            import pandas as pd
            from odps.df import DataFrame
            pd_df = pd.DataFrame([['Col1', 1], ['Col2', 2]], columns=['Field1', 'Field2'])
            df = DataFrame(pd_df)
            self.assertRaises(ExpressionError, lambda: df.filter_partition('Fieldd2=2'))
        except ImportError:
            pass

    def testDepExpr(self):
        expr1 = Scalar('1')
        expr2 = self.expr['id']
        expr2.add_deps(expr1)

        self.assertIn(expr1, expr2.deps)

    def testBacktrackField(self):
        expr = self.expr.filter(self.expr.id < 3)[self.expr.id + 1, self.expr.name.rename('name2')]

        self.assertIs(expr._fields[0].lhs.input, expr.input)
        self.assertIs(expr._fields[1].input, expr.input)
        self.assertEqual(expr._fields[1].name, 'name2')

        with self.assertRaises(ExpressionError):
            self.expr[self.expr.id + 1, self.expr.name][self.expr.name, self.expr.id]

        expr = self.expr[self.expr.id + 1, 'name'].filter(self.expr.name == 'a')

        self.assertIs(expr._predicate.lhs.input, expr.input)

        with self.assertRaises(ExpressionError):
            self.expr[self.expr.id + 1, self.expr.name][self.expr2.name,]

    def testSetitemField(self):
        from odps.df.expr.groupby import GroupByCollectionExpr

        expr = self.expr.copy()

        expr['new_id'] = expr.id + 1

        self.assertIn('new_id', expr.schema.names)
        self.assertIs(expr._fields[-1].lhs.input, expr.input)

        self.assertEqual(expr.schema.names, ['name', 'id', 'fid', 'new_id'])

        expr['new_id2'] = expr.id + 2

        self.assertIn('new_id2', expr.schema.names)
        self.assertIs(expr._fields[-1].lhs.input, expr.input)

        self.assertEqual(expr.schema.names, ['name', 'id', 'fid', 'new_id', 'new_id2'])
        self.assertIsNone(expr._input._proxy)

        expr['new_id2'] = expr.new_id

        expr['new_id3'] = expr.id + expr.new_id2
        self.assertIs(expr._fields[-1].lhs.input, expr.input)
        self.assertIs(expr._fields[-1].rhs.lhs.input, expr.input)

        self.assertIsInstance(expr, ProjectCollectionExpr)
        self.assert_(isinstance(expr, ProjectCollectionExpr))

        expr2 = expr.groupby('name').agg(expr.id.sum())
        expr2['new_id2'] = expr2.id_sum + 1
        self.assertIsInstance(expr2, ProjectCollectionExpr)
        self.assertNotIsInstance(expr2, GroupByCollectionExpr)
        self.assertNotIsInstance(expr2, FilterCollectionExpr)

    def testDelitemField(self):
        from odps.df.expr.groupby import GroupByCollectionExpr
        from odps.df.expr.collections import DistinctCollectionExpr

        expr = self.expr.copy()

        del expr['fid']

        self.assertNotIn('fid', expr.schema)
        self.assertEqual(expr.schema.names, ['name', 'id'])
        self.assertIsInstance(expr, ProjectCollectionExpr)

        expr['id2'] = self.expr.id + 1
        del expr['id2']

        self.assertNotIn('id2', expr.schema)
        self.assertEqual(expr.schema.names, ['name', 'id'])

        expr['id3'] = expr.id
        del expr['id']

        self.assertNotIn('id', expr.schema)
        self.assertIn('id3', expr.schema)
        self.assertEqual(expr.schema.names, ['name', 'id3'])

        expr2 = expr.groupby('name').agg(expr.id3.sum().rename('id'))
        del expr2.name

        self.assertNotIn('name', expr2.schema)
        self.assertIn('id', expr2.schema)
        self.assertEqual(expr2.schema.names, ['id'])
        self.assertIsInstance(expr2, ProjectCollectionExpr)
        self.assertNotIsInstance(expr2, GroupByCollectionExpr)

        expr3 = expr2.distinct()
        expr3['new_id'] = expr3.id + 1
        expr3['new_id2'] = expr3.new_id * 2
        del expr3['new_id']

        self.assertNotIn('new_id', expr3.schema)
        self.assertIn('new_id2', expr3.schema)
        self.assertEqual(expr3.schema.names, ['id', 'new_id2'])
        self.assertIsInstance(expr3, ProjectCollectionExpr)
        self.assertNotIsInstance(expr3, DistinctCollectionExpr)

if __name__ == '__main__':
    unittest.main()
