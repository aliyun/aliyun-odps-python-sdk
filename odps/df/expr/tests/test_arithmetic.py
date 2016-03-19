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
from odps.models import Schema
from odps.df.expr.expressions import CollectionExpr, SequenceExpr, Scalar, Float64SequenceExpr
from odps.df.types import validate_data_type
from odps.df.expr.tests.core import MockTable
import odps.df.expr.arithmetic as arithmetic
import decimal
import datetime


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        table = MockTable(name='pyodps_test_expr_table', schema=schema)

        self.expr = CollectionExpr(_source_data=table, _schema=schema)

    def test_binary_operate(self):
        # test string
        expr = self.expr
        self.assertIsInstance("hello" + expr['name'] + "test", arithmetic.Add)
        self.assertIsInstance(expr['name'] + expr['name'], arithmetic.Add)
        self.assertIsInstance(expr['name'] != "test", arithmetic.NotEqual)
        self.assertIsInstance(expr['name'] == "test", arithmetic.Equal)

        # test number
        d = decimal.Decimal('3.145926')
        self.assertIsInstance(20.0 + expr['id'] + 10, arithmetic.Add)
        self.assertIsInstance(float(10 / 3) + expr['id'] + float(4.5556), arithmetic.Add)
        self.assertIsInstance(expr['id'] + expr['id'], arithmetic.Add)

        self.assertIsInstance(20.0 + expr['fid'] + 10, arithmetic.Add)
        self.assertIsInstance(float(10 / 3) + expr['fid'] + float(4.5556), arithmetic.Add)
        self.assertIsInstance(expr['fid'] + expr['fid'], arithmetic.Add)

        self.assertIsInstance(expr['id'] + expr['fid'], arithmetic.Add)

        self.assertIsInstance(expr['scale'] + d, arithmetic.Add)
        self.assertIsInstance(expr['scale'] + expr['scale'], arithmetic.Add)
        self.assertIsInstance(d + expr['scale'], arithmetic.Add)

        self.assertIsInstance(20 + expr['scale'], arithmetic.Add)
        self.assertIsInstance(expr['id'] + expr['scale'], arithmetic.Add)
        self.assertIsInstance(expr['scale'] + 20, arithmetic.Add)

        self.assertIsInstance(20.0 - expr['id'] - 10, arithmetic.Substract)
        self.assertIsInstance(float(10 / 3) - expr['id'] - float(4.555667), arithmetic.Substract)
        self.assertIsInstance(expr['id'] - expr['id'], arithmetic.Substract)

        self.assertIsInstance(20.0 - expr['fid'] - 10, arithmetic.Substract)
        self.assertIsInstance(float(10 / 3) - expr['fid'] - float(4.555667), arithmetic.Substract)
        self.assertIsInstance(expr['fid'] - expr['fid'], arithmetic.Substract)

        self.assertIsInstance(expr['id'] - expr['fid'], arithmetic.Substract)

        self.assertIsInstance(expr['scale'] - d, arithmetic.Substract)
        self.assertIsInstance(expr['scale'] - expr['scale'], arithmetic.Substract)
        self.assertIsInstance(d - expr['scale'], arithmetic.Substract)

        self.assertIsInstance(expr['scale'] - 20, arithmetic.Substract)
        self.assertIsInstance(20 - expr['scale'], arithmetic.Substract)
        self.assertIsInstance(expr['id'] - expr['scale'], arithmetic.Substract)

        # multiply
        self.assertIsInstance(20 * expr['id'] * 10, arithmetic.Multiply)
        self.assertIsInstance(float(4.5) * expr['id'] * float(4.556), arithmetic.Multiply)
        self.assertIsInstance(expr['id'] * expr['id'], arithmetic.Multiply)

        self.assertIsInstance(20 * expr['fid'] * 10, arithmetic.Multiply)
        self.assertIsInstance(float(4.5) * expr['fid'] * float(4.556), arithmetic.Multiply)
        self.assertIsInstance(expr['fid'] * expr['fid'], arithmetic.Multiply)

        self.assertIsInstance(expr['id'] * expr['fid'], arithmetic.Multiply)

        self.assertIsInstance(expr['scale'] * expr['scale'], arithmetic.Multiply)
        self.assertIsInstance(expr['scale'] * d, arithmetic.Multiply)
        self.assertIsInstance(expr['scale'] * expr['scale'], arithmetic.Multiply)
        self.assertIsInstance(d * expr['scale'], arithmetic.Multiply)

        # divide
        self.assertIsInstance(20 / expr['id'] / 3, arithmetic.Divide)
        self.assertIsInstance(float(34.6) / expr['id'] / float(4.5), arithmetic.Divide)
        self.assertIsInstance(expr['id'] / expr['id'], arithmetic.Divide)

        self.assertIsInstance(20 / expr['fid'] / 3, arithmetic.Divide)
        self.assertIsInstance(float(34.6) / expr['fid'] / float(4.5), arithmetic.Divide)
        self.assertIsInstance(expr['fid'] / expr['fid'], arithmetic.Divide)

        self.assertIsInstance(expr['id'] / expr['fid'], arithmetic.Divide)

        self.assertIsInstance(d / expr['scale'], arithmetic.Divide)
        self.assertIsInstance(expr['scale'] / d, arithmetic.Divide)
        self.assertIsInstance(expr['scale'] / expr['scale'], arithmetic.Divide)

        self.assertIsInstance(expr.id / 20, Float64SequenceExpr)

        # power
        self.assertIsInstance(pow(20, expr['id']), arithmetic.Power)
        self.assertIsInstance(pow(float(34.6), expr['id']), arithmetic.Power)
        self.assertIsInstance(pow(expr['id'], 20), arithmetic.Power)
        self.assertIsInstance(pow(expr['id'], float(34.6)), arithmetic.Power)
        self.assertIsInstance(pow(expr['id'], expr['id']), arithmetic.Power)

        self.assertIsInstance(pow(20, expr['fid']), arithmetic.Power)
        self.assertIsInstance(pow(float(34.6), expr['fid']), arithmetic.Power)
        self.assertIsInstance(pow(expr['fid'], 20), arithmetic.Power)
        self.assertIsInstance(pow(expr['fid'], float(34.6)), arithmetic.Power)
        self.assertIsInstance(pow(expr['fid'], expr['fid']), arithmetic.Power)

        self.assertIsInstance(pow(expr['id'], expr['fid']), arithmetic.Power)
        self.assertIsInstance(pow(expr['fid'], expr['id']), arithmetic.Power)

        self.assertIsInstance(pow(d, expr['scale']), arithmetic.Power)
        self.assertIsInstance(pow(expr['scale'], 20), arithmetic.Power)
        self.assertIsInstance(pow(expr['scale'], expr['scale']), arithmetic.Power)

        # floor divide
        self.assertIsInstance(20 // expr['id'] // 3, arithmetic.FloorDivide)
        self.assertIsInstance(float(34.6) // expr['id'] // float(4.5), arithmetic.FloorDivide)
        self.assertIsInstance(expr['id'] // expr['id'], arithmetic.FloorDivide)

        self.assertIsInstance(20 // expr['fid'] // 3, arithmetic.FloorDivide)
        self.assertIsInstance(float(34.6) // expr['fid'] // float(4.5), arithmetic.FloorDivide)
        self.assertIsInstance(expr['fid'] // expr['fid'], arithmetic.FloorDivide)

        self.assertIsInstance(expr['id'] // expr['fid'], arithmetic.FloorDivide)

        self.assertIsInstance(d // expr['scale'], arithmetic.FloorDivide)
        self.assertIsInstance(expr['scale'] // d, arithmetic.FloorDivide)
        self.assertIsInstance(expr['scale'] // expr['scale'], arithmetic.FloorDivide)

        # comparision
        self.assertIsInstance(expr['id'] == expr['fid'], arithmetic.Equal)
        self.assertIsInstance(expr['id'] == 20, arithmetic.Equal)
        self.assertIsInstance(expr['fid'] == 20, arithmetic.Equal)
        self.assertIsInstance(expr['scale'] == d, arithmetic.Equal)

        self.assertIsInstance(expr['id'] != expr['fid'], arithmetic.NotEqual)
        self.assertIsInstance(expr['id'] != 20, arithmetic.NotEqual)
        self.assertIsInstance(expr['fid'] != 20, arithmetic.NotEqual)
        self.assertIsInstance(expr['scale'] != d, arithmetic.NotEqual)

        self.assertIsInstance(expr['id'] <= expr['id'], arithmetic.LessEqual)
        self.assertIsInstance(expr['id'] <= expr['fid'], arithmetic.LessEqual)
        self.assertIsInstance(expr['id'] <= 20, arithmetic.LessEqual)
        self.assertIsInstance(expr['id'] <= 20.123, arithmetic.LessEqual)
        self.assertIsInstance(expr['id'] <= float(10 / 3), arithmetic.LessEqual)

        self.assertIsInstance(expr['fid'] <= expr['fid'], arithmetic.LessEqual)
        self.assertIsInstance(expr['fid'] <= expr['id'], arithmetic.LessEqual)
        self.assertIsInstance(expr['fid'] <= 20, arithmetic.LessEqual)
        self.assertIsInstance(expr['fid'] <= 20.123, arithmetic.LessEqual)
        self.assertIsInstance(expr['fid'] <= float(10 / 3), arithmetic.LessEqual)

        compareExpr3 = 20 <= expr['id']
        self.assertIsInstance(compareExpr3, arithmetic.GreaterEqual)
        self.assertIsInstance(compareExpr3._lhs, SequenceExpr)
        self.assertIsInstance(compareExpr3._rhs, Scalar)

        self.assertIsInstance(20.123 <= expr['id'], arithmetic.GreaterEqual)
        self.assertIsInstance(float(10 / 3) <= expr['id'], arithmetic.GreaterEqual)

        self.assertIsInstance(20 <= expr['fid'], arithmetic.GreaterEqual)
        self.assertIsInstance(20.123 <= expr['fid'], arithmetic.GreaterEqual)
        self.assertIsInstance(float(10 / 3) <= expr['fid'], arithmetic.GreaterEqual)

        self.assertIsInstance(expr['scale'] <= d, arithmetic.LessEqual)
        self.assertIsInstance(expr['scale'] <= expr['scale'], arithmetic.LessEqual)
        self.assertIsInstance(d <= expr['scale'], arithmetic.GreaterEqual)

        self.assertIsInstance(expr['id'] < expr['id'], arithmetic.Less)
        self.assertIsInstance(expr['id'] < expr['fid'], arithmetic.Less)
        self.assertIsInstance(expr['id'] < 20, arithmetic.Less)
        self.assertIsInstance(expr['id'] < 20.123, arithmetic.Less)
        self.assertIsInstance(expr['id'] < float(10 / 3), arithmetic.Less)

        self.assertIsInstance(expr['fid'] < expr['fid'], arithmetic.Less)
        self.assertIsInstance(expr['fid'] < expr['id'], arithmetic.Less)
        self.assertIsInstance(expr['fid'] < 20, arithmetic.Less)
        self.assertIsInstance(expr['fid'] < 20.123, arithmetic.Less)
        self.assertIsInstance(expr['fid'] < float(10 / 3), arithmetic.Less)

        compareExpr = 20 < expr['id']
        self.assertIsInstance(compareExpr, arithmetic.Greater)
        self.assertIsInstance(compareExpr._lhs, SequenceExpr)
        self.assertIsInstance(compareExpr._rhs, Scalar)

        self.assertIsInstance(20.123 < expr['id'], arithmetic.Greater)
        self.assertIsInstance(float(10 / 3) < expr['id'], arithmetic.Greater)

        self.assertIsInstance(20 < expr['fid'], arithmetic.Greater)
        self.assertIsInstance(20.123 < expr['fid'], arithmetic.Greater)
        self.assertIsInstance(float(10 / 3) < expr['fid'], arithmetic.Greater)

        self.assertIsInstance(d < expr['scale'], arithmetic.Greater)
        self.assertIsInstance(expr['scale'] < d, arithmetic.Less)
        self.assertIsInstance(expr['scale'] < expr['scale'], arithmetic.Less)

        # bool
        self.assertIsInstance(expr['isMale'] == False, arithmetic.Equal)
        self.assertIsInstance(expr['isMale'] != False, arithmetic.NotEqual)

        self.assertIsInstance((expr['isMale'] & True), arithmetic.And)
        self.assertIsInstance(True & expr['isMale'], arithmetic.And)
        self.assertIsInstance((expr['isMale'] | False), arithmetic.Or)
        self.assertIsInstance(True | expr['isMale'], arithmetic.Or)

        # date
        date = datetime.datetime(2015, 12, 2)

        self.assertIsInstance(expr['birth'] == date, arithmetic.Equal)
        self.assertIsInstance(expr['birth'] != date, arithmetic.NotEqual)
        self.assertIsInstance(date - expr['birth'] - date, arithmetic.Substract)
        self.assertIsInstance(expr['birth'] >= date, arithmetic.GreaterEqual)
        self.assertIsInstance(expr['birth'] <= date, arithmetic.LessEqual)

        compareExpr2 = expr['birth'] > date
        self.assertIsInstance(compareExpr2, arithmetic.Greater)
        self.assertIsInstance(compareExpr2._lhs, SequenceExpr)
        self.assertIsInstance(compareExpr2._rhs, Scalar)
        self.assertIsInstance(expr['birth'] < date, arithmetic.Less)
        self.assertIsInstance(expr['birth'] < expr['birth'], arithmetic.Less)

    def test_unary_operate(self):
        expr = self.expr
        self.assertIsInstance(-expr['id'], arithmetic.Negate)
        self.assertIsInstance(-(-expr['id']), SequenceExpr)
        self.assertIsInstance(-(-(-expr['id'])), arithmetic.Negate)
        self.assertIsInstance(-(-(-(-expr['id']))), SequenceExpr)

        self.assertIsInstance(-expr['scale'], arithmetic.Negate)
        self.assertIsInstance(-(-expr['scale']), SequenceExpr)
        self.assertIsInstance(-(-(-expr['scale'])), arithmetic.Negate)
        self.assertIsInstance(-(-(-(-expr['scale']))), SequenceExpr)

        self.assertIsInstance(-expr['fid'], arithmetic.Negate)
        self.assertIsInstance(-(-expr['fid']), SequenceExpr)
        self.assertIsInstance(-(-(-expr['fid'])), arithmetic.Negate)
        self.assertIsInstance(-(-(-(-expr['fid']))), SequenceExpr)

        self.assertIsInstance(~expr['id'], arithmetic.Invert)
        self.assertIsInstance(~(~expr['id']), SequenceExpr)
        self.assertIsInstance(~(~(~expr['id'])), arithmetic.Invert)
        self.assertIsInstance(~(~(~(~expr['id']))), SequenceExpr)

        self.assertIsInstance(~expr['isMale'], arithmetic.Invert)
        self.assertIsInstance(~(~expr['isMale']), SequenceExpr)
        self.assertIsInstance(~(~(~expr['isMale'])), arithmetic.Invert)
        self.assertIsInstance(~(~(~(~expr['isMale']))), SequenceExpr)

        self.assertIsInstance(abs(expr['id']), arithmetic.Abs)
        self.assertIsInstance(abs(abs(expr['id'])), arithmetic.Abs)

        self.assertIsInstance(abs(expr['fid']), arithmetic.Abs)
        self.assertIsInstance(abs(abs(expr['fid'])), arithmetic.Abs)

    def test_negative_operate(self):
        expr = self.expr
        self.assertRaises(AttributeError, lambda: expr['name'] - 10)
        self.assertRaises(AttributeError, lambda: ~expr['fid'])

        self.assertRaises(TypeError, lambda: expr['name'] + expr['id'])
        self.assertRaises(TypeError, lambda: 'hello' + expr['id'])

    def test_complex_airith(self):
        expr = self.expr
        self.assertIsInstance(20 - expr['id'] / expr['id'] + 10, arithmetic.Add)
        self.assertIsInstance(
            -(expr['id']) + 20.34 - expr['fid'] + float(20) * expr['id'] - expr['fid'] / 4.9 + 40 // 2 + expr[
                'fid'] // 1.2, arithmetic.Add)

if __name__ == '__main__':
    unittest.main()
