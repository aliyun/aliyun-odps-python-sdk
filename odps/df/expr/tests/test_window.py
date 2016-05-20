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
from odps.compat import unittest
from odps.df.expr.arithmetic import *
from odps.df.expr.window import *
from odps.df.expr.errors import ExpressionError
from odps.df import types


class Test(TestBase):
    def setup(self):
        schema = Schema.from_lists(types._data_types.keys(), types._data_types.values())
        self.expr = CollectionExpr(_source_data=None, _schema=schema)

    # def testCacheNode(self):
    #     self.assertIs(self.expr.groupby('boolean').int16.cumsum(),
    #                   self.expr.groupby('boolean').int16.cumsum())
    #     self.assertIs(self.expr.groupby('boolean').sort('int8').int16.cumsum(),
    #                   self.expr.groupby('boolean').sort('int8').int16.cumsum())
    #     self.assertIs(self.expr.groupby('string').decimal.lag(-5, sort=['int8']),
    #                   self.expr.groupby('string').decimal.lag(-5, sort=['int8']))

    def testCumSum(self):
        grouped = self.expr.groupby('boolean')

        self.assertRaises(AttributeError, lambda: grouped.datetime.cumsum())

        cumsum = grouped.sort('int8').int16.cumsum()

        self.assertIsInstance(cumsum, CumSum)
        self.assertIsInstance(cumsum, Int16SequenceExpr)
        self.assertSequenceEqual([by.name for by in cumsum._partition_by], ['boolean', ])
        self.assertSequenceEqual([by.name for by in cumsum._order_by], ['int8', ])

    def testCumMax(self):
        cummax = self.expr.groupby(['int16', 'float64']).boolean.cummax()
        self.assertIsInstance(cummax, CumMax)
        self.assertIsInstance(cummax, BooleanSequenceExpr)
        self.assertSequenceEqual([by.name for by in cummax._partition_by], ['int16', 'float64'])

    def testCumMin(self):
        cummin = self.expr.groupby('datetime').int16.cummin()
        self.assertIsInstance(cummin, CumMin)
        self.assertIsInstance(cummin, Int16SequenceExpr)
        self.assertSequenceEqual([by.name for by in cummin._partition_by], ['datetime', ])

    def testCumMean(self):
        cummean = self.expr.groupby(self.expr.int16 * 2).sort(self.expr.float32 + 1).int64.cummean()
        self.assertIsInstance(cummean, CumMean)
        self.assertIsInstance(cummean, Float64SequenceExpr)
        self.assertSequenceEqual([by.name for by in cummean._partition_by], ['int16', ])
        self.assertEqual(len(cummean._partition_by), 1)
        self.assertIsInstance(cummean._partition_by[0], Multiply)
        self.assertSequenceEqual([by.name for by in cummean._order_by], ['float32', ])
        self.assertEqual(len(cummean._order_by), 1)
        self.assertIsInstance(cummean._order_by[0].input, Add)

    def testCumMedian(self):
        cummedian = self.expr.groupby(['float64', self.expr.string]).int32.cummedian(
                preceding=3, following=10)
        self.assertIsInstance(cummedian, CumMedian)
        self.assertIsInstance(cummedian, Float64SequenceExpr)
        self.assertEqual(cummedian._preceding, 3)
        self.assertEqual(cummedian._following, 10)

    def testCumCount(self):
        cumcount = self.expr.groupby('decimal').string.cumcount().unique()
        self.assertIsInstance(cumcount, CumCount)
        self.assertIsInstance(cumcount, Int64SequenceExpr)
        self.assertEqual(cumcount._distinct, True)

        s = self.expr.groupby('decimal').sort('string').string
        self.assertRaises(ExpressionError, lambda: s.cumcount(unique=True))
        s = self.expr.groupby('decimal').string.cumcount(unique=True)
        self.assertSequenceEqual([by.name for by in s._partition_by], ['decimal', ])

    def testCumStd(self):
        grouped = self.expr.groupby('datetime')

        self.assertRaises(AttributeError, lambda: grouped.string.cumstd())

        cumstd = grouped.decimal.cumstd(preceding=(10, 3))
        self.assertIsInstance(cumstd, CumStd)
        self.assertIsInstance(cumstd, DecimalSequenceExpr)
        self.assertSequenceEqual(cumstd.preceding, [10, 3])

        cumstd = grouped.int16.cumstd(following=(4, 8))
        self.assertIsInstance(cumstd, Float64SequenceExpr)
        self.assertSequenceEqual(cumstd.following, [4, 8])

        self.assertRaises(AssertionError, lambda: grouped.int32.cumstd(preceding=(4, 5)))
        self.assertRaises(AssertionError, lambda: grouped.int32.cumstd(following=(5, 4)))

        self.assertRaises(ValueError,
                          lambda: grouped.int64.cumstd(preceding=(10, 3), following=20))
        self.assertRaises(ValueError,
                          lambda: grouped.int64.cumstd(preceding=20, following=(3, 10)))
        self.assertRaises(ValueError,
                          lambda: grouped.int64.cumstd(preceding=(10, 3), following=(3, 10)))

    def testRank(self):
        rank = self.expr.groupby('boolean').sort(lambda x: x['float64']).rank()

        self.assertIsInstance(rank, Rank)
        self.assertIsInstance(rank, Int64SequenceExpr)
        self.assertSequenceEqual([by.name for by in rank._partition_by], ['boolean', ])
        self.assertSequenceEqual([by.name for by in rank._order_by], ['float64', ])

    def testDenseRank(self):
        grouped = self.expr.groupby('datetime')
        denserank = grouped.boolean.dense_rank()

        self.assertIs(denserank._input, grouped.dense_rank(sort='boolean')._input)
        self.assertIsInstance(denserank, DenseRank)
        self.assertIsInstance(denserank, Int64SequenceExpr)

    def testPercentRank(self):
        percentrank = self.expr.groupby('boolean').percent_rank(sort='int16')

        self.assertIsInstance(percentrank, PercentRank)
        self.assertIsInstance(percentrank, Float64SequenceExpr)

    def testRowNumber(self):
        rownumber = self.expr.groupby('string').row_number(sort='int8').rename('rank')

        self.assertEqual(rownumber.name, 'rank')
        self.assertIsInstance(rownumber, RowNumber)
        self.assertIsInstance(rownumber, Int64SequenceExpr)

    def testLead(self):
        grouped = self.expr.groupby(self.expr.string)

        self.assertRaises(AttributeError, lambda: grouped.lead())

        lead = grouped.int8.lead(5, default=10)
        self.assertIsInstance(lead, Lead)
        self.assertIsInstance(lead, Int8SequenceExpr)
        self.assertEqual(lead._offset, 5)
        self.assertEqual(lead._default, 10)

    def testLag(self):
        lag = self.expr.groupby('string').decimal.lag(-5, sort=['int8'])

        self.assertIsInstance(lag, Lag)
        self.assertIsInstance(lag, DecimalSequenceExpr)
        self.assertEqual(lag._offset, -5)
        self.assertSequenceEqual([by.name for by in lag._partition_by], ['string', ])
        self.assertSequenceEqual([by.name for by in lag._order_by], ['int8', ])


if __name__ == '__main__':
    unittest.main()