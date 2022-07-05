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


from odps.tests.core import TestBase
from odps.compat import unittest
from odps.models import Schema
from odps.df.types import validate_data_type
from odps.df.expr.tests.core import MockTable
from odps.df.expr.math import *


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        table = MockTable(name='pyodps_test_expr_table', schema=schema)

        self.expr = CollectionExpr(_source_data=table, _schema=schema)

    # def testCacheNode(self):
    #     self.assertIs(self.expr.fid.abs(), self.expr.fid.abs())
    #     self.assertIs(self.expr.id.log(2), self.expr.id.log(2))

    def testMath(self):
        self.assertRaises(AttributeError, lambda: self.expr.name.sin())
        self.assertRaises(AttributeError, lambda: self.expr.isMale.cos())
        self.assertRaises(AttributeError, lambda: self.expr.birth.tan())

        self.assertIsInstance(self.expr.fid.abs(), Abs)
        self.assertIsInstance(self.expr.fid.abs(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.fid.sum().abs(), Float64Scalar)

        self.assertIsInstance(self.expr.id.sqrt(), Sqrt)
        self.assertIsInstance(self.expr.id.sqrt(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.sum().sqrt(), DecimalScalar)

        self.assertIsInstance(self.expr.id.sqrt(), Sqrt)
        self.assertIsInstance(self.expr.id.sqrt(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.sum().sqrt(), DecimalScalar)

        self.assertIsInstance(self.expr.id.sin(), Sin)
        self.assertIsInstance(self.expr.id.sin(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().sin(), DecimalScalar)

        self.assertIsInstance(self.expr.id.sinh(), Sinh)
        self.assertIsInstance(self.expr.id.sinh(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().sinh(), DecimalScalar)

        self.assertIsInstance(self.expr.id.cos(), Cos)
        self.assertIsInstance(self.expr.id.cos(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().cos(), DecimalScalar)

        self.assertIsInstance(self.expr.id.cosh(), Cosh)
        self.assertIsInstance(self.expr.id.cosh(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().cosh(), DecimalScalar)

        self.assertIsInstance(self.expr.id.tan(), Tan)
        self.assertIsInstance(self.expr.id.tan(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().tan(), DecimalScalar)

        self.assertIsInstance(self.expr.id.tanh(), Tanh)
        self.assertIsInstance(self.expr.id.tanh(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().tanh(), DecimalScalar)

        self.assertIsInstance(self.expr.id.exp(), Exp)
        self.assertIsInstance(self.expr.id.exp(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().exp(), DecimalScalar)

        self.assertIsInstance(self.expr.id.expm1(), Expm1)
        self.assertIsInstance(self.expr.id.expm1(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().expm1(), DecimalScalar)

        self.assertIsInstance(self.expr.id.log(), Log)
        self.assertIsInstance(self.expr.id.log(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().log(), DecimalScalar)

        self.assertIsInstance(self.expr.id.log(2), Log)
        self.assertIsInstance(self.expr.id.log(2), Float64SequenceExpr)
        self.assertEqual(self.expr.id.log(2)._base, 2)
        self.assertIsInstance(self.expr.scale.max().log(2), DecimalScalar)

        self.assertIsInstance(self.expr.id.log10(), Log10)
        self.assertIsInstance(self.expr.id.log10(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().log10(), DecimalScalar)

        self.assertIsInstance(self.expr.id.log1p(), Log1p)
        self.assertIsInstance(self.expr.id.log1p(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().log1p(), DecimalScalar)

        self.assertIsInstance(self.expr.id.arccos(), Arccos)
        self.assertIsInstance(self.expr.id.arccos(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().arccos(), DecimalScalar)

        self.assertIsInstance(self.expr.id.arccosh(), Arccosh)
        self.assertIsInstance(self.expr.id.arccosh(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().arccosh(), DecimalScalar)

        self.assertIsInstance(self.expr.id.arcsin(), Arcsin)
        self.assertIsInstance(self.expr.id.arcsin(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().arcsin(), DecimalScalar)

        self.assertIsInstance(self.expr.id.arcsinh(), Arcsinh)
        self.assertIsInstance(self.expr.id.arcsin(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().arcsin(), DecimalScalar)

        self.assertIsInstance(self.expr.id.arctan(), Arctan)
        self.assertIsInstance(self.expr.id.arctan(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().arctan(), DecimalScalar)

        self.assertIsInstance(self.expr.id.arctanh(), Arctanh)
        self.assertIsInstance(self.expr.id.sin(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().sin(), DecimalScalar)

        self.assertIsInstance(self.expr.id.radians(), Radians)
        self.assertIsInstance(self.expr.id.radians(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().radians(), DecimalScalar)

        self.assertIsInstance(self.expr.id.degrees(), Degrees)
        self.assertIsInstance(self.expr.id.degrees(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().degrees(), DecimalScalar)

        self.assertIsInstance(self.expr.fid.ceil(), Ceil)
        self.assertIsInstance(self.expr.fid.ceil(), Int64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().ceil(), Int64Scalar)

        self.assertIsInstance(self.expr.fid.floor(), Floor)
        self.assertIsInstance(self.expr.fid.floor(), Int64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().floor(), Int64Scalar)

        self.assertIsInstance(self.expr.id.trunc(), Trunc)
        self.assertIsInstance(self.expr.id.trunc(), Float64SequenceExpr)
        self.assertIsInstance(self.expr.scale.max().trunc(), DecimalScalar)

        self.assertIsInstance(self.expr.id.trunc(2), Trunc)
        self.assertIsInstance(self.expr.id.trunc(2), Float64SequenceExpr)
        self.assertEqual(self.expr.id.trunc(2)._decimals, 2)
        self.assertIsInstance(self.expr.scale.max().trunc(2), DecimalScalar)


if __name__ == '__main__':
    unittest.main()