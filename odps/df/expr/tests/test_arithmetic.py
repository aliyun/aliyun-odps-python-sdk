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

import decimal
import datetime

import pytest

from ....models import TableSchema
from ...types import validate_data_type
from .. import arithmetic
from ..expressions import CollectionExpr, SequenceExpr, Scalar, Float64SequenceExpr
from ..tests.core import MockTable


@pytest.fixture
def test_expr():
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                    datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)

    return CollectionExpr(_source_data=table, _schema=schema)


def test_binary_operate(test_expr):
    expr = test_expr
    # test string
    assert isinstance("hello" + expr['name'] + "test", arithmetic.Add)
    assert isinstance(expr['name'] + expr['name'], arithmetic.Add)
    assert isinstance(expr['name'] != "test", arithmetic.NotEqual)
    assert isinstance(expr['name'] == "test", arithmetic.Equal)

    # test number
    d = decimal.Decimal('3.145926')
    assert isinstance(20.0 + expr['id'] + 10, arithmetic.Add)
    assert isinstance(float(10 / 3) + expr['id'] + float(4.5556), arithmetic.Add)
    assert isinstance(expr['id'] + expr['id'], arithmetic.Add)

    assert isinstance(20.0 + expr['fid'] + 10, arithmetic.Add)
    assert isinstance(float(10 / 3) + expr['fid'] + float(4.5556), arithmetic.Add)
    assert isinstance(expr['fid'] + expr['fid'], arithmetic.Add)

    assert isinstance(expr['id'] + expr['fid'], arithmetic.Add)

    assert isinstance(expr['scale'] + d, arithmetic.Add)
    assert isinstance(expr['scale'] + expr['scale'], arithmetic.Add)
    assert isinstance(d + expr['scale'], arithmetic.Add)

    assert isinstance(20 + expr['scale'], arithmetic.Add)
    assert isinstance(expr['id'] + expr['scale'], arithmetic.Add)
    assert isinstance(expr['scale'] + 20, arithmetic.Add)

    assert isinstance(20.0 - expr['id'] - 10, arithmetic.Substract)
    assert isinstance(float(10 / 3) - expr['id'] - float(4.555667), arithmetic.Substract)
    assert isinstance(expr['id'] - expr['id'], arithmetic.Substract)

    assert isinstance(20.0 - expr['fid'] - 10, arithmetic.Substract)
    assert isinstance(float(10 / 3) - expr['fid'] - float(4.555667), arithmetic.Substract)
    assert isinstance(expr['fid'] - expr['fid'], arithmetic.Substract)

    assert isinstance(expr['id'] - expr['fid'], arithmetic.Substract)

    assert isinstance(expr['scale'] - d, arithmetic.Substract)
    assert isinstance(expr['scale'] - expr['scale'], arithmetic.Substract)
    assert isinstance(d - expr['scale'], arithmetic.Substract)

    assert isinstance(expr['scale'] - 20, arithmetic.Substract)
    assert isinstance(20 - expr['scale'], arithmetic.Substract)
    assert isinstance(expr['id'] - expr['scale'], arithmetic.Substract)

    # multiply
    assert isinstance(20 * expr['id'] * 10, arithmetic.Multiply)
    assert isinstance(float(4.5) * expr['id'] * float(4.556), arithmetic.Multiply)
    assert isinstance(expr['id'] * expr['id'], arithmetic.Multiply)

    assert isinstance(20 * expr['fid'] * 10, arithmetic.Multiply)
    assert isinstance(float(4.5) * expr['fid'] * float(4.556), arithmetic.Multiply)
    assert isinstance(expr['fid'] * expr['fid'], arithmetic.Multiply)

    assert isinstance(expr['id'] * expr['fid'], arithmetic.Multiply)

    assert isinstance(expr['scale'] * expr['scale'], arithmetic.Multiply)
    assert isinstance(expr['scale'] * d, arithmetic.Multiply)
    assert isinstance(expr['scale'] * expr['scale'], arithmetic.Multiply)
    assert isinstance(d * expr['scale'], arithmetic.Multiply)

    # divide
    assert isinstance(20 / expr['id'] / 3, arithmetic.Divide)
    assert isinstance(float(34.6) / expr['id'] / float(4.5), arithmetic.Divide)
    assert isinstance(expr['id'] / expr['id'], arithmetic.Divide)

    assert isinstance(20 / expr['fid'] / 3, arithmetic.Divide)
    assert isinstance(float(34.6) / expr['fid'] / float(4.5), arithmetic.Divide)
    assert isinstance(expr['fid'] / expr['fid'], arithmetic.Divide)

    assert isinstance(expr['id'] / expr['fid'], arithmetic.Divide)

    assert isinstance(d / expr['scale'], arithmetic.Divide)
    assert isinstance(expr['scale'] / d, arithmetic.Divide)
    assert isinstance(expr['scale'] / expr['scale'], arithmetic.Divide)

    assert isinstance(expr.id / 20, Float64SequenceExpr)

    # power
    assert isinstance(pow(20, expr['id']), arithmetic.Power)
    assert isinstance(pow(float(34.6), expr['id']), arithmetic.Power)
    assert isinstance(pow(expr['id'], 20), arithmetic.Power)
    assert isinstance(pow(expr['id'], float(34.6)), arithmetic.Power)
    assert isinstance(pow(expr['id'], expr['id']), arithmetic.Power)

    assert isinstance(pow(20, expr['fid']), arithmetic.Power)
    assert isinstance(pow(float(34.6), expr['fid']), arithmetic.Power)
    assert isinstance(pow(expr['fid'], 20), arithmetic.Power)
    assert isinstance(pow(expr['fid'], float(34.6)), arithmetic.Power)
    assert isinstance(pow(expr['fid'], expr['fid']), arithmetic.Power)

    assert isinstance(pow(expr['id'], expr['fid']), arithmetic.Power)
    assert isinstance(pow(expr['fid'], expr['id']), arithmetic.Power)

    assert isinstance(pow(d, expr['scale']), arithmetic.Power)
    assert isinstance(pow(expr['scale'], 20), arithmetic.Power)
    assert isinstance(pow(expr['scale'], expr['scale']), arithmetic.Power)

    # floor divide
    assert isinstance(20 // expr['id'] // 3, arithmetic.FloorDivide)
    assert isinstance(float(34.6) // expr['id'] // float(4.5), arithmetic.FloorDivide)
    assert isinstance(expr['id'] // expr['id'], arithmetic.FloorDivide)

    assert isinstance(20 // expr['fid'] // 3, arithmetic.FloorDivide)
    assert isinstance(float(34.6) // expr['fid'] // float(4.5), arithmetic.FloorDivide)
    assert isinstance(expr['fid'] // expr['fid'], arithmetic.FloorDivide)

    assert isinstance(expr['id'] // expr['fid'], arithmetic.FloorDivide)

    assert isinstance(d // expr['scale'], arithmetic.FloorDivide)
    assert isinstance(expr['scale'] // d, arithmetic.FloorDivide)
    assert isinstance(expr['scale'] // expr['scale'], arithmetic.FloorDivide)

    # comparison
    assert isinstance(expr['id'] == expr['fid'], arithmetic.Equal)
    assert isinstance(expr['id'] == 20, arithmetic.Equal)
    assert isinstance(expr['fid'] == 20, arithmetic.Equal)
    assert isinstance(expr['scale'] == d, arithmetic.Equal)

    assert isinstance(expr['id'] != expr['fid'], arithmetic.NotEqual)
    assert isinstance(expr['id'] != 20, arithmetic.NotEqual)
    assert isinstance(expr['fid'] != 20, arithmetic.NotEqual)
    assert isinstance(expr['scale'] != d, arithmetic.NotEqual)

    assert isinstance(expr['id'] <= expr['id'], arithmetic.LessEqual)
    assert isinstance(expr['id'] <= expr['fid'], arithmetic.LessEqual)
    assert isinstance(expr['id'] <= 20, arithmetic.LessEqual)
    assert isinstance(expr['id'] <= 20.123, arithmetic.LessEqual)
    assert isinstance(expr['id'] <= float(10 / 3), arithmetic.LessEqual)

    assert isinstance(expr['fid'] <= expr['fid'], arithmetic.LessEqual)
    assert isinstance(expr['fid'] <= expr['id'], arithmetic.LessEqual)
    assert isinstance(expr['fid'] <= 20, arithmetic.LessEqual)
    assert isinstance(expr['fid'] <= 20.123, arithmetic.LessEqual)
    assert isinstance(expr['fid'] <= float(10 / 3), arithmetic.LessEqual)

    compareExpr3 = 20 <= expr['id']
    assert isinstance(compareExpr3, arithmetic.GreaterEqual)
    assert isinstance(compareExpr3._lhs, SequenceExpr)
    assert isinstance(compareExpr3._rhs, Scalar)

    assert isinstance(20.123 <= expr['id'], arithmetic.GreaterEqual)
    assert isinstance(float(10 / 3) <= expr['id'], arithmetic.GreaterEqual)

    assert isinstance(20 <= expr['fid'], arithmetic.GreaterEqual)
    assert isinstance(20.123 <= expr['fid'], arithmetic.GreaterEqual)
    assert isinstance(float(10 / 3) <= expr['fid'], arithmetic.GreaterEqual)

    assert isinstance(expr['scale'] <= d, arithmetic.LessEqual)
    assert isinstance(expr['scale'] <= expr['scale'], arithmetic.LessEqual)
    assert isinstance(d <= expr['scale'], arithmetic.GreaterEqual)

    assert isinstance(expr['id'] < expr['id'], arithmetic.Less)
    assert isinstance(expr['id'] < expr['fid'], arithmetic.Less)
    assert isinstance(expr['id'] < 20, arithmetic.Less)
    assert isinstance(expr['id'] < 20.123, arithmetic.Less)
    assert isinstance(expr['id'] < float(10 / 3), arithmetic.Less)

    assert isinstance(expr['fid'] < expr['fid'], arithmetic.Less)
    assert isinstance(expr['fid'] < expr['id'], arithmetic.Less)
    assert isinstance(expr['fid'] < 20, arithmetic.Less)
    assert isinstance(expr['fid'] < 20.123, arithmetic.Less)
    assert isinstance(expr['fid'] < float(10 / 3), arithmetic.Less)

    compareExpr = 20 < expr['id']
    assert isinstance(compareExpr, arithmetic.Greater)
    assert isinstance(compareExpr._lhs, SequenceExpr)
    assert isinstance(compareExpr._rhs, Scalar)

    assert isinstance(20.123 < expr['id'], arithmetic.Greater)
    assert isinstance(float(10 / 3) < expr['id'], arithmetic.Greater)

    assert isinstance(20 < expr['fid'], arithmetic.Greater)
    assert isinstance(20.123 < expr['fid'], arithmetic.Greater)
    assert isinstance(float(10 / 3) < expr['fid'], arithmetic.Greater)

    assert isinstance(d < expr['scale'], arithmetic.Greater)
    assert isinstance(expr['scale'] < d, arithmetic.Less)
    assert isinstance(expr['scale'] < expr['scale'], arithmetic.Less)

    # bool
    assert isinstance(expr['isMale'] == False, arithmetic.Equal)
    assert isinstance(expr['isMale'] != False, arithmetic.NotEqual)

    assert isinstance((expr['isMale'] & True), arithmetic.And)
    assert isinstance(True & expr['isMale'], arithmetic.And)
    assert isinstance((expr['isMale'] | False), arithmetic.Or)
    assert isinstance(True | expr['isMale'], arithmetic.Or)

    # date
    date = datetime.datetime(2015, 12, 2)

    assert isinstance(expr['birth'] == date, arithmetic.Equal)
    assert isinstance(expr['birth'] != date, arithmetic.NotEqual)
    assert isinstance(date - expr['birth'] - date, arithmetic.Substract)
    assert isinstance(expr['birth'] >= date, arithmetic.GreaterEqual)
    assert isinstance(expr['birth'] <= date, arithmetic.LessEqual)

    compareExpr2 = expr['birth'] > date
    assert isinstance(compareExpr2, arithmetic.Greater)
    assert isinstance(compareExpr2._lhs, SequenceExpr)
    assert isinstance(compareExpr2._rhs, Scalar)
    assert isinstance(expr['birth'] < date, arithmetic.Less)
    assert isinstance(expr['birth'] < expr['birth'], arithmetic.Less)


def test_unary_operate(test_expr):
    expr = test_expr
    assert isinstance(-expr['id'], arithmetic.Negate)
    assert isinstance(-(-expr['id']), SequenceExpr)
    assert isinstance(-(-(-expr['id'])), arithmetic.Negate)
    assert isinstance(-(-(-(-expr['id']))), SequenceExpr)

    assert isinstance(-expr['scale'], arithmetic.Negate)
    assert isinstance(-(-expr['scale']), SequenceExpr)
    assert isinstance(-(-(-expr['scale'])), arithmetic.Negate)
    assert isinstance(-(-(-(-expr['scale']))), SequenceExpr)

    assert isinstance(-expr['fid'], arithmetic.Negate)
    assert isinstance(-(-expr['fid']), SequenceExpr)
    assert isinstance(-(-(-expr['fid'])), arithmetic.Negate)
    assert isinstance(-(-(-(-expr['fid']))), SequenceExpr)

    assert isinstance(~expr['id'], arithmetic.Invert)
    assert isinstance(~(~expr['id']), SequenceExpr)
    assert isinstance(~(~(~expr['id'])), arithmetic.Invert)
    assert isinstance(~(~(~(~expr['id']))), SequenceExpr)

    assert isinstance(~expr['isMale'], arithmetic.Invert)
    assert isinstance(~(~expr['isMale']), SequenceExpr)
    assert isinstance(~(~(~expr['isMale'])), arithmetic.Invert)
    assert isinstance(~(~(~(~expr['isMale']))), SequenceExpr)

    assert isinstance(abs(expr['id']), arithmetic.Abs)
    assert isinstance(abs(abs(expr['id'])), arithmetic.Abs)

    assert isinstance(abs(expr['fid']), arithmetic.Abs)
    assert isinstance(abs(abs(expr['fid'])), arithmetic.Abs)


def test_negative_operate(test_expr):
    expr = test_expr
    pytest.raises(AttributeError, lambda: expr['name'] - 10)
    pytest.raises(AttributeError, lambda: ~expr['fid'])

    pytest.raises(TypeError, lambda: expr['name'] + expr['id'])
    pytest.raises(TypeError, lambda: 'hello' + expr['id'])


def test_complex_airith(test_expr):
    expr = test_expr
    assert isinstance(20 - expr['id'] / expr['id'] + 10, arithmetic.Add)
    assert isinstance(-(expr['id']) + 20.34 - expr['fid'] + float(20) * expr['id'] - expr['fid'] / 4.9 + 40 // 2 + expr[
            'fid'] // 1.2, arithmetic.Add)
