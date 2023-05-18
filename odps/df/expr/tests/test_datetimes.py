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

from datetime import datetime

import pytest

from ....models import TableSchema
from ...types import validate_data_type
from ..tests.core import MockTable
from ..datetimes import *


@pytest.fixture
def src_expr():
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
    )
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)

    return CollectionExpr(_source_data=table, _schema=schema)


def test_datetimes(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.id.hour)
    pytest.raises(AttributeError, lambda: src_expr.fid.minute)
    pytest.raises(AttributeError, lambda: src_expr.isMale.week)
    pytest.raises(AttributeError, lambda: src_expr.scale.year)
    pytest.raises(AttributeError, lambda: src_expr.name.strftime('%Y'))

    assert isinstance(src_expr.birth.date, Date)
    assert isinstance(src_expr.birth.date, DatetimeSequenceExpr)
    assert isinstance(src_expr.birth.max().date, DatetimeScalar)

    assert isinstance(src_expr.birth.time, Time)
    assert isinstance(src_expr.birth.time, DatetimeSequenceExpr)
    assert isinstance(src_expr.birth.max().time, DatetimeScalar)

    assert isinstance(src_expr.birth.year, Year)
    assert isinstance(src_expr.birth.year, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().year, Int64Scalar)

    assert isinstance(src_expr.birth.month, Month)
    assert isinstance(src_expr.birth.month, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().month, Int64Scalar)

    assert isinstance(src_expr.birth.day, Day)
    assert isinstance(src_expr.birth.day, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().day, Int64Scalar)

    assert isinstance(src_expr.birth.hour, Hour)
    assert isinstance(src_expr.birth.hour, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().hour, Int64Scalar)

    assert isinstance(src_expr.birth.minute, Minute)
    assert isinstance(src_expr.birth.minute, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().minute, Int64Scalar)

    assert isinstance(src_expr.birth.second, Second)
    assert isinstance(src_expr.birth.second, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().second, Int64Scalar)

    assert isinstance(src_expr.birth.microsecond, MicroSecond)
    assert isinstance(src_expr.birth.microsecond, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().microsecond, Int64Scalar)

    assert isinstance(src_expr.birth.week, Week)
    assert isinstance(src_expr.birth.week, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().week, Int64Scalar)

    assert isinstance(src_expr.birth.weekofyear, WeekOfYear)
    assert isinstance(src_expr.birth.weekofyear, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().weekofyear, Int64Scalar)

    assert isinstance(src_expr.birth.dayofweek, WeekDay)
    assert isinstance(src_expr.birth.dayofweek, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().dayofweek, Int64Scalar)

    assert isinstance(src_expr.birth.weekday, WeekDay)
    assert isinstance(src_expr.birth.weekday, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().weekday, Int64Scalar)

    assert isinstance(src_expr.birth.dayofyear, DayOfYear)
    assert isinstance(src_expr.birth.dayofyear, Int64SequenceExpr)
    assert isinstance(src_expr.birth.max().dayofyear, Int64Scalar)

    assert isinstance(src_expr.birth.is_month_start, IsMonthStart)
    assert isinstance(src_expr.birth.is_month_start, BooleanSequenceExpr)
    assert isinstance(src_expr.birth.max().is_month_start, BooleanScalar)

    assert isinstance(src_expr.birth.is_month_end, IsMonthEnd)
    assert isinstance(src_expr.birth.is_month_end, BooleanSequenceExpr)
    assert isinstance(src_expr.birth.max().is_month_end, BooleanScalar)

    assert isinstance(src_expr.birth.is_year_start, IsYearStart)
    assert isinstance(src_expr.birth.is_year_start, BooleanSequenceExpr)
    assert isinstance(src_expr.birth.max().is_year_start, BooleanScalar)

    assert isinstance(src_expr.birth.is_year_end, IsYearEnd)
    assert isinstance(src_expr.birth.is_year_end, BooleanSequenceExpr)
    assert isinstance(src_expr.birth.max().is_year_end, BooleanScalar)

    assert isinstance(src_expr.birth.strftime('%Y'), Strftime)
    assert isinstance(src_expr.birth.strftime('%Y'), StringSequenceExpr)
    assert isinstance(src_expr.birth.max().strftime('%Y'), StringScalar)

    expr = src_expr.birth + hour(10)
    assert isinstance(expr, DatetimeSequenceExpr)

    expr = src_expr.birth - microsecond(100)
    assert isinstance(expr, DatetimeSequenceExpr)

    expr = src_expr.birth - datetime.now()
    assert isinstance(expr, Int64SequenceExpr)

    pytest.raises(ExpressionError, lambda: src_expr.birth + datetime.now())
