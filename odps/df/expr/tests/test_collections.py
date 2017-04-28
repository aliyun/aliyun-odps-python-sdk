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
from odps.compat import unittest, OrderedDict
from odps.models import Schema
from odps.df.expr.expressions import CollectionExpr, ExpressionError
from odps.df.expr.collections import SampledCollectionExpr, AppendIDCollectionExpr, SplitCollectionExpr
from odps.df import DataFrame, output, types


class Test(TestBase):
    def setup(self):
        from odps.df.expr.tests.core import MockTable
        schema = Schema.from_lists(types._data_types.keys(), types._data_types.values())
        self.expr = CollectionExpr(_source_data=None, _schema=schema)
        self.sourced_expr = CollectionExpr(_source_data=MockTable(client=self.odps.rest), _schema=schema)

    def testSort(self):
        sorted_expr = self.expr.sort(self.expr.int64)
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [True])

        sorted_expr = self.expr.sort_values([self.expr.float32, 'string'])
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [True] * 2)

        sorted_expr = self.expr.sort([self.expr.decimal, 'boolean', 'string'], ascending=False)
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [False] * 3)

        sorted_expr = self.expr.sort([self.expr.int8, 'datetime', 'float64'],
                                     ascending=[False, True, False])
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [False, True, False])

        sorted_expr = self.expr.sort([-self.expr.int8, 'datetime', 'float64'])
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [False, True, True])

    def testDistinct(self):
        distinct = self.expr.distinct()
        self.assertIsInstance(distinct, CollectionExpr)
        self.assertEqual(distinct._schema, self.expr._schema)

        distinct = self.expr.distinct(self.expr.string)
        self.assertIsInstance(distinct, CollectionExpr)
        self.assertEqual(distinct._schema, self.expr[[self.expr.string]]._schema)

        distinct = self.expr.distinct([self.expr.boolean, 'decimal'])
        self.assertIsInstance(distinct, CollectionExpr)
        self.assertEqual(distinct._schema, self.expr[['boolean', 'decimal']]._schema)

        self.assertRaises(ExpressionError, lambda: self.expr['boolean', self.expr.string.unique()])

    def testMapReduce(self):
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

        expr = self.expr.map_reduce(mapper, reducer, group='name',
                                    sort='rating', ascending=False)
        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr.input._sort_fields), 2)
        self.assertTrue(expr.input._sort_fields[0]._ascending)
        self.assertFalse(expr.input._sort_fields[1]._ascending)

        expr = self.expr.map_reduce(mapper, reducer, group='name',
                                    sort=['rating', 'id'], ascending=[False, True])

        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr.input._sort_fields), 3)
        self.assertTrue(expr.input._sort_fields[0]._ascending)
        self.assertFalse(expr.input._sort_fields[1]._ascending)
        self.assertTrue(expr.input._sort_fields[2]._ascending)

        expr = self.expr.map_reduce(mapper, reducer, group='name',
                                    sort=['rating', 'id'], ascending=False)

        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr.input._sort_fields), 3)
        self.assertTrue(expr.input._sort_fields[0]._ascending)
        self.assertFalse(expr.input._sort_fields[1]._ascending)
        self.assertFalse(expr.input._sort_fields[2]._ascending)

        expr = self.expr.map_reduce(mapper, reducer, group='name',
                                    sort=['name', 'rating', 'id'],
                                    ascending=[False, True, False])

        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr.input._sort_fields), 3)
        self.assertFalse(expr.input._sort_fields[0]._ascending)
        self.assertTrue(expr.input._sort_fields[1]._ascending)
        self.assertFalse(expr.input._sort_fields[2]._ascending)

    def testSample(self):
        from odps.ml.expr import AlgoCollectionExpr

        self.assertIsInstance(self.expr.sample(100), SampledCollectionExpr)
        self.assertIsInstance(self.expr.sample(parts=10), SampledCollectionExpr)
        try:
            import pandas
        except ImportError:
            # No pandas: go for XFlow
            self.assertIsInstance(self.expr.sample(frac=0.5), AlgoCollectionExpr)
        else:
            # Otherwise: go for Pandas
            self.assertIsInstance(self.expr.sample(frac=0.5), SampledCollectionExpr)
        self.assertIsInstance(self.sourced_expr.sample(frac=0.5), AlgoCollectionExpr)

        self.assertRaises(ExpressionError, lambda: self.expr.sample())
        self.assertRaises(ExpressionError, lambda: self.expr.sample(i=-1))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(n=100, frac=0.5))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(n=100, parts=10))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(frac=0.5, parts=10))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(n=100, frac=0.5, parts=10))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(frac=-1))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(frac=1.5))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(parts=10, i=-1))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(parts=10, i=10))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(parts=10, n=10))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(weights='weights', strata='strata'))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(frac='Yes:10', strata='strata'))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(frac=set(), strata='strata'))
        self.assertRaises(ExpressionError, lambda: self.expr.sample(n=set(), strata='strata'))

    def testPivot(self):
        from odps.df.expr.dynamic import DynamicMixin

        expr = self.expr.pivot('string', 'int8', 'float32')

        self.assertIn('string', expr._schema._name_indexes)
        self.assertEqual(len(expr._schema._name_indexes), 1)
        self.assertIn('non_exist', expr._schema)
        self.assertIsInstance(expr['non_exist'], DynamicMixin)

        expr = self.expr.pivot(
            ['string', 'int8'], 'int16', ['datetime', 'string'])

        self.assertIn('string', expr._schema._name_indexes)
        self.assertIn('int8', expr._schema._name_indexes)
        self.assertEqual(len(expr._schema._name_indexes), 2)
        self.assertIn('non_exist', expr._schema)
        self.assertIsInstance(expr['non_exist'], DynamicMixin)

        self.assertRaises(ValueError, lambda: self.expr.pivot(
            ['string', 'int8'], ['datetime', 'string'], 'int16'))

    def testPivotTable(self):
        from odps.df.expr.dynamic import DynamicMixin

        expr = self.expr.pivot_table(values='int8', rows='float32')
        self.assertNotIsInstance(expr, DynamicMixin)
        self.assertEqual(expr.schema.names, ['float32', 'int8_mean'])

        expr = self.expr.pivot_table(values=('int16', 'int32'), rows=['float32', 'int8'])
        self.assertEqual(expr.schema.names, ['float32', 'int8', 'int16_mean', 'int32_mean'])

        expr = self.expr.pivot_table(values=('int16', 'int32'), rows=['string', 'boolean'],
                                     aggfunc=['mean', 'sum'])
        self.assertEqual(expr.schema.names, ['string', 'boolean', 'int16_mean', 'int32_mean',
                                             'int16_sum', 'int32_sum'])
        self.assertEqual(expr.schema.types, [types.string, types.boolean, types.float64, types.float64,
                                             types.int16, types.int32])

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

        expr = self.expr.pivot_table(values='int16', rows='string', aggfunc=Aggregator)
        self.assertEqual(expr.schema.names, ['string', 'int16_my_mean'])
        self.assertEqual(expr.schema.types, [types.string, types.float64])

        aggfunc = OrderedDict([('my_agg', Aggregator), ('my_agg2', Aggregator)])

        expr = self.expr.pivot_table(values='int16', rows='string', aggfunc=aggfunc)
        self.assertEqual(expr.schema.names, ['string', 'int16_my_agg', 'int16_my_agg2'])
        self.assertEqual(expr.schema.types, [types.string, types.float64, types.float64])

        expr = self.expr.pivot_table(values='int16', columns='boolean', rows='string')
        self.assertIsInstance(expr, DynamicMixin)

    def testScaleValue(self):
        expr = self.expr.min_max_scale()
        self.assertIsInstance(expr, CollectionExpr)
        self.assertListEqual(expr.dtypes.names, self.expr.dtypes.names)

        expr = self.expr.min_max_scale(preserve=True)
        self.assertIsInstance(expr, CollectionExpr)
        self.assertListEqual(expr.dtypes.names, self.expr.dtypes.names +
                             [n + '_scaled' for n in self.expr.dtypes.names
                              if n.startswith('int') or n.startswith('float')])

        expr = self.expr.std_scale()
        self.assertIsInstance(expr, CollectionExpr)
        self.assertListEqual(expr.dtypes.names, self.expr.dtypes.names)

        expr = self.expr.std_scale(preserve=True)
        self.assertIsInstance(expr, CollectionExpr)
        self.assertListEqual(expr.dtypes.names, self.expr.dtypes.names +
                             [n + '_scaled' for n in self.expr.dtypes.names
                              if n.startswith('int') or n.startswith('float')])

    def testApplyMap(self):
        from odps.df.expr.collections import ProjectCollectionExpr, Column
        from odps.df.expr.element import MappedExpr

        schema = Schema.from_lists(['idx', 'f1', 'f2', 'f3'], [types.int64] + [types.float64] * 3)
        expr = CollectionExpr(_source_data=None, _schema=schema)

        self.assertRaises(ValueError, lambda: expr.applymap(lambda v: v + 1, columns='idx', excludes='f1'))

        mapped = expr.applymap(lambda v: v + 1)
        self.assertIsInstance(mapped, ProjectCollectionExpr)
        for c in mapped._fields:
            self.assertIsInstance(c, MappedExpr)

        mapped = expr.applymap(lambda v: v + 1, columns='f1')
        self.assertIsInstance(mapped, ProjectCollectionExpr)
        for c in mapped._fields:
            self.assertIsInstance(c, MappedExpr if c.name == 'f1' else Column)

        map_cols = set(['f1', 'f2', 'f3'])
        mapped = expr.applymap(lambda v: v + 1, columns=map_cols)
        self.assertIsInstance(mapped, ProjectCollectionExpr)
        for c in mapped._fields:
            self.assertIsInstance(c, MappedExpr if c.name in map_cols else Column)

        mapped = expr.applymap(lambda v: v + 1, excludes='idx')
        self.assertIsInstance(mapped, ProjectCollectionExpr)
        for c in mapped._fields:
            self.assertIsInstance(c, Column if c.name == 'idx' else MappedExpr)

        exc_cols = set(['idx', 'f1'])
        mapped = expr.applymap(lambda v: v + 1, excludes=exc_cols)
        self.assertIsInstance(mapped, ProjectCollectionExpr)
        for c in mapped._fields:
            self.assertIsInstance(c, Column if c.name in exc_cols else MappedExpr)

    def testCallableColumn(self):
        from odps.df.expr.expressions import CallableColumn
        from odps.df.expr.collections import ProjectCollectionExpr

        schema = Schema.from_lists(['name', 'f1', 'append_id'], [types.string, types.float64, types.int64])
        expr = CollectionExpr(_source_data=None, _schema=schema)
        self.assertIsInstance(expr.append_id, CallableColumn)
        self.assertNotIsInstance(expr.f1, CallableColumn)

        projected = expr[expr.name, expr.append_id]
        self.assertIsInstance(projected, ProjectCollectionExpr)
        self.assertListEqual(projected.schema.names, ['name', 'append_id'])

        projected = expr[expr.name, expr.f1]
        self.assertNotIsInstance(projected.append_id, CallableColumn)

        appended = expr.append_id(id_col='id_col')
        self.assertIn('id_col', appended.schema)

    def testAppendId(self):
        from odps.ml.expr import AlgoCollectionExpr

        expr = self.expr.append_id(id_col='id_col')
        try:
            import pandas
        except ImportError:
            # No pandas: go for XFlow
            self.assertIsInstance(expr, AlgoCollectionExpr)
        else:
            # Otherwise: go for Pandas
            self.assertIsInstance(expr, AppendIDCollectionExpr)
        self.assertIn('id_col', expr.schema)

        self.assertIsInstance(self.sourced_expr.append_id(), AlgoCollectionExpr)

    def testSplit(self):
        from odps.ml.expr import AlgoCollectionExpr

        expr1, expr2 = self.expr.split(0.6)
        try:
            import pandas
        except ImportError:
            # No pandas: go for XFlow
            self.assertIsInstance(expr1, AlgoCollectionExpr)
            self.assertIsInstance(expr2, AlgoCollectionExpr)
        else:
            # Otherwise: go for Pandas
            self.assertIsInstance(expr1, SplitCollectionExpr)
            self.assertIsInstance(expr2, SplitCollectionExpr)
            self.assertTupleEqual((expr1._split_id, expr2._split_id), (0, 1))


if __name__ == '__main__':
    unittest.main()
