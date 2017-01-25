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
from odps.compat import unittest
from odps.df.expr.reduction import *


class Test(TestBase):
    def setup(self):
        schema = Schema.from_lists(types._data_types.keys(), types._data_types.values())
        self.expr = CollectionExpr(_source_data=True, _schema=schema)

    def testGroupby(self):
        grouped = self.expr.groupby(['int16', 'boolean'])\
            .agg(self.expr.string.sum().rename('sum_string'),
                 sum=self.expr.int16.sum())
        self.assertIsInstance(grouped, CollectionExpr)
        self.assertSequenceEqual(grouped._schema.names, ['int16', 'boolean', 'sum', 'sum_string'])
        self.assertSequenceEqual(grouped._schema.types, [types.int16, types.boolean, types.int16, types.string])
        self.assertIsInstance(grouped.sum_string, StringSequenceExpr)

        grouped = self.expr.groupby('datetime').aggregate(
            self.expr.boolean.sum(),
            min_int32=self.expr.int32.min())
        self.assertSequenceEqual(grouped._schema.names, ['datetime', 'boolean_sum', 'min_int32'])
        self.assertSequenceEqual(grouped._schema.types, [types.datetime, types.int64, types.int32])

        grouped = self.expr.groupby(['float32', 'string']).agg(int64std=self.expr.int64.std(ddof=3))
        self.assertSequenceEqual(grouped._schema.names, ['float32', 'string', 'int64std'])
        self.assertSequenceEqual(grouped._schema.types, [types.float32, types.string, types.float64])
        self.assertEqual(grouped._aggregations[0]._ddof, 3)

        selected = grouped[[grouped.float32.rename('new_col')]]
        self.assertEqual(selected.schema.names, ['new_col'])

    def testGroupbyReductions(self):
        expr = self.expr.groupby('string').min()
        self.assertIsInstance(expr, GroupByCollectionExpr)
        self.assertGreater(len(expr.aggregations), 0)
        self.assertIsInstance(expr.aggregations[0], GroupedMin)

        expr = self.expr.groupby('string').max()
        self.assertIsInstance(expr, GroupByCollectionExpr)
        self.assertGreater(len(expr.aggregations), 0)
        self.assertIsInstance(expr.aggregations[0], GroupedMax)

        expr = self.expr.groupby('string').count()
        self.assertIsInstance(expr, SequenceExpr)
        self.assertIsInstance(expr, Int64SequenceExpr)

        expr = self.expr.groupby('string').var()
        self.assertIsInstance(expr, GroupByCollectionExpr)
        self.assertGreater(len(expr.aggregations), 0)
        self.assertIsInstance(expr.aggregations[0], GroupedVar)

        expr = self.expr.groupby('string').sum()
        self.assertIsInstance(expr, GroupByCollectionExpr)
        self.assertGreater(len(expr.aggregations), 0)
        self.assertIsInstance(expr.aggregations[0], GroupedSum)

        expr = self.expr.groupby('string').std()
        self.assertIsInstance(expr, GroupByCollectionExpr)
        self.assertGreater(len(expr.aggregations), 0)
        self.assertIsInstance(expr.aggregations[0], GroupedStd)

        expr = self.expr.groupby('string').mean()
        self.assertIsInstance(expr, GroupByCollectionExpr)
        self.assertGreater(len(expr.aggregations), 0)
        self.assertIsInstance(expr.aggregations[0], GroupedMean)

        expr = self.expr.groupby('string').median()
        self.assertIsInstance(expr, GroupByCollectionExpr)
        self.assertGreater(len(expr.aggregations), 0)
        self.assertIsInstance(expr.aggregations[0], GroupedMedian)

        metric = self.expr.int32.mean() > 10
        field = (metric.ifelse(self.expr.int64.max(), 0) + 1).rename('int64_max')
        expr = self.expr.groupby('string').agg(field)
        self.assertIsInstance(expr, GroupByCollectionExpr)
        self.assertIsInstance(expr.int64_max, Int64SequenceExpr)

    def testGroupbyField(self):
        grouped = self.expr.groupby(['int32', 'boolean']).string.sum()
        self.assertIsInstance(grouped, StringSequenceExpr)
        self.assertIsInstance(grouped, GroupedSum)
        self.assertIsInstance(grouped._input, Column)

    def testMutate(self):
        grouped = self.expr.groupby(['int16', self.expr.datetime]).sort(-self.expr.boolean)
        expr = grouped.mutate(grouped.float64.cumsum(), count=grouped.boolean.cumcount())

        self.assertIsInstance(expr, MutateCollectionExpr)
        self.assertSequenceEqual(expr._schema.names,
                                 ['int16', 'datetime', 'float64_sum', 'count'])
        self.assertSequenceEqual(expr._schema.types,
                                 [types.int16, types.datetime, types.float64, types.int64])

    def testIllegalGroupby(self):
        self.assertRaises(ExpressionError, lambda: self.expr.groupby('int16').agg(self.expr['string']))
        self.assertRaises(ExpressionError,
                          lambda: self.expr.groupby('int16').agg(self.expr['string'] + self.expr['string'].sum()))
        self.assertRaises(ExpressionError,
                          lambda: self.expr.groupby('int8').agg(self.expr['boolean', ]['boolean'].sum()))

        grouped = self.expr.groupby('string')
        self.assertRaises(ExpressionError, lambda: self.expr.groupby('boolean').agg(grouped.int32.sum()))

        expr = self.expr[self.expr, Scalar(1).rename('id')]
        self.assertRaises(ExpressionError, lambda: expr.groupby('id').agg(self.expr.int32.sum()))

if __name__ == '__main__':
    unittest.main()
