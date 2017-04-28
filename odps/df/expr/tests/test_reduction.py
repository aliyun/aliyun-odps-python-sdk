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

import textwrap

from odps.tests.core import TestBase
from odps.compat import unittest, six, LESS_PY35
from odps.df.expr.reduction import *
from odps.df import types


class Test(TestBase):
    def setup(self):
        schema = Schema.from_lists(types._data_types.keys(), types._data_types.values())
        self.expr = CollectionExpr(_source_data=None, _schema=schema)

    def testMin(self):
        min_ = self.expr.string.min()
        self.assertIsInstance(min_, StringScalar)

        min_ = self.expr.boolean.min()
        self.assertIsInstance(min_, BooleanScalar)

        min_ = self.expr.int16.min()
        self.assertIsInstance(min_, Int16Scalar)

        expr = self.expr.min()
        self.assertIsInstance(expr, Summary)
        self.assertEqual(len(expr.fields), len(types._data_types))
        self.assertTrue(all(isinstance(node, Min) for node in expr.fields))

    def testMax(self):
        max_ = self.expr.string.max()
        self.assertIsInstance(max_, StringScalar)

        max_ = self.expr.boolean.max()
        self.assertIsInstance(max_, BooleanScalar)

        max_ = self.expr.int32.max()
        self.assertIsInstance(max_, Int32Scalar)

        expr = self.expr.max()
        self.assertIsInstance(expr, Summary)
        self.assertLessEqual(len(expr.fields), len(types._data_types))
        self.assertTrue(all(isinstance(node, Max) for node in expr.fields))

    def testCount(self):
        count = self.expr.string.count()
        self.assertIsInstance(count, Int64Scalar)

        expr = self.expr.count()
        self.assertIsInstance(expr, Int64Scalar)

    def testSum(self):
        sum = self.expr.string.sum()
        self.assertIsInstance(sum, StringScalar)

        sum = self.expr.boolean.sum()
        self.assertIsInstance(sum, Int64Scalar)

        sum = self.expr.int32.sum()
        self.assertIsInstance(sum, Int32Scalar)

        sum = self.expr.decimal.sum()
        self.assertIsInstance(sum, DecimalScalar)

        expr = self.expr.sum()
        self.assertIsInstance(expr, Summary)
        self.assertLessEqual(len(expr.fields), len(types._data_types))
        self.assertTrue(all(isinstance(node, Sum) for node in expr.fields))

    def testVar(self):
        self.assertRaises(AttributeError, lambda: self.expr.string.var())

        var = self.expr.int8.var()
        self.assertIsInstance(var, Float64Scalar)

        var = self.expr.decimal.var()
        self.assertIsInstance(var, DecimalScalar)

        expr = self.expr.var()
        self.assertIsInstance(expr, Summary)
        self.assertLessEqual(len(expr.fields), len(types._data_types))
        self.assertTrue(all(isinstance(node, Var) for node in expr.fields))

    def testStd(self):
        self.assertRaises(AttributeError, lambda: self.expr.boolean.var())

        std = self.expr.int64.std()
        self.assertIsInstance(std, Float64Scalar)

        std = self.expr.decimal.std()
        self.assertIsInstance(std, DecimalScalar)

        expr = self.expr.std()
        self.assertIsInstance(expr, Summary)
        self.assertLessEqual(len(expr.fields), len(types._data_types))
        self.assertTrue(all(isinstance(node, Std) for node in expr.fields))

    def testMean(self):
        self.assertRaises(AttributeError, lambda: self.expr.datetime.mean())

        mean = self.expr.float32.mean()
        self.assertIsInstance(mean, Float64Scalar)

        mean = self.expr.decimal.std()
        self.assertIsInstance(mean, DecimalScalar)

        expr = self.expr.mean()
        self.assertIsInstance(expr, Summary)
        self.assertLessEqual(len(expr.fields), len(types._data_types))
        self.assertTrue(all(isinstance(node, Mean) for node in expr.fields))

    def testMedian(self):
        self.assertRaises(AttributeError, lambda: self.expr.string.median())

        median = self.expr.float64.median()
        self.assertIsInstance(median, Float64Scalar)

        median = self.expr.decimal.median()
        self.assertIsInstance(median, DecimalScalar)

        expr = self.expr.median()
        self.assertIsInstance(expr, Summary)
        self.assertLessEqual(len(expr.fields), len(types._data_types))
        self.assertTrue(all(isinstance(node, Median) for node in expr.fields))

    def testAny(self):
        self.assertRaises(AttributeError, lambda: self.expr.string.any())
        self.assertRaises(AttributeError, lambda: self.expr.int64.any())

        any_ = self.expr.boolean.any()
        self.assertIsInstance(any_, BooleanScalar)

        expr = self.expr.any()
        self.assertIsInstance(expr, Summary)
        self.assertLessEqual(len(expr.fields), len(types._data_types))
        self.assertTrue(all(isinstance(node, Any) for node in expr.fields))

    def testAll(self):
        self.assertRaises(AttributeError, lambda: self.expr.string.all())
        self.assertRaises(AttributeError, lambda: self.expr.int64.all())

        any_ = self.expr.boolean.all()
        self.assertIsInstance(any_, BooleanScalar)

        expr = self.expr.all()
        self.assertIsInstance(expr, Summary)
        self.assertLessEqual(len(expr.fields), len(types._data_types))
        self.assertTrue(all(isinstance(node, All) for node in expr.fields))

    def testCat(self):
        self.assertRaises(AttributeError, lambda: self.expr.int64.cat(sep=','))
        self.assertRaises(AttributeError, lambda: self.expr.float.cat(sep=','))
        self.assertRaises(AttributeError, lambda: self.expr.cat(sep=','))

        cat = self.expr.string.cat(sep=',')
        self.assertIsInstance(cat, StringScalar)

    def testAgg(self):
        class Agg(object):
            def buffer(self):
                return [0]

            def __call__(self, buffer, val):
                buffer[0] += val

            def merge(self, buffer, pbuffer):
                buffer[0] += pbuffer[0]

            def getvalue(self, buffer):
                return buffer[0]

        expr = self.expr.int64.agg(Agg)

        self.assertIsInstance(expr, Aggregation)
        self.assertEqual(expr.dtype, types.int64)

        if not LESS_PY35:
            l = locals().copy()
            six.exec_(textwrap.dedent("""
            class Agg(object):
                def buffer(self):
                    return [0]
    
                def __call__(self, buffer, val):
                    buffer[0] += val
    
                def merge(self, buffer, pbuffer):
                    buffer[0] += pbuffer[0]
    
                def getvalue(self, buffer) -> float:
                    return buffer[0]
    
            expr = self.expr.int64.agg(Agg)
            """), globals(), l)
            expr = l['expr']
            self.assertIsInstance(expr, Aggregation)
            self.assertIsInstance(expr.dtype, types.Float)

if __name__ == '__main__':
    unittest.main()
