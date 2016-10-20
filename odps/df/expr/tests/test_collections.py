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
from odps.compat import unittest, OrderedDict
from odps.models import Schema
from odps.df.expr.expressions import CollectionExpr, ExpressionError
from odps.df.expr.collections import SampledCollectionExpr
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
        self.assertEqual(len(expr._sort_fields), 2)
        self.assertTrue(expr._sort_fields[0]._ascending)
        self.assertFalse(expr._sort_fields[1]._ascending)

        expr = self.expr.map_reduce(mapper, reducer, group='name',
                                    sort=['rating', 'id'], ascending=[False, True])

        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr._sort_fields), 3)
        self.assertTrue(expr._sort_fields[0]._ascending)
        self.assertFalse(expr._sort_fields[1]._ascending)
        self.assertTrue(expr._sort_fields[2]._ascending)

        expr = self.expr.map_reduce(mapper, reducer, group='name',
                                    sort=['rating', 'id'], ascending=False)

        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr._sort_fields), 3)
        self.assertTrue(expr._sort_fields[0]._ascending)
        self.assertFalse(expr._sort_fields[1]._ascending)
        self.assertFalse(expr._sort_fields[2]._ascending)

    def testSample(self):
        self.assertIsInstance(self.expr.sample(100), SampledCollectionExpr)
        self.assertIsInstance(self.expr.sample(parts=10), SampledCollectionExpr)
        try:
            import pandas
        except ImportError:
            # No pandas: go for XFlow
            self.assertIsInstance(self.expr.sample(frac=0.5), DataFrame)
        else:
            # Otherwise: go for Pandas
            self.assertIsInstance(self.expr.sample(frac=0.5), SampledCollectionExpr)
        self.assertIsInstance(self.sourced_expr.sample(frac=0.5), DataFrame)

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

if __name__ == '__main__':
    unittest.main()
