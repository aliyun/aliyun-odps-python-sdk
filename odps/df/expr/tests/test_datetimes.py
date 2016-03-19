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

from datetime import datetime

from odps.tests.core import TestBase
from odps.compat import unittest
from odps.models import Schema
from odps.df.types import validate_data_type
from odps.df.expr.tests.core import MockTable
from odps.df.expr.datetimes import *


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        table = MockTable(name='pyodps_test_expr_table', schema=schema)

        self.expr = CollectionExpr(_source_data=table, _schema=schema)

    def testDatetimes(self):
        self.assertRaises(AttributeError, lambda: self.expr.id.hour)
        self.assertRaises(AttributeError, lambda: self.expr.fid.minute)
        self.assertRaises(AttributeError, lambda: self.expr.isMale.week)
        self.assertRaises(AttributeError, lambda: self.expr.scale.year)
        self.assertRaises(AttributeError, lambda: self.expr.name.strftime('%Y'))

        self.assertIsInstance(self.expr.birth.date, Date)
        self.assertIsInstance(self.expr.birth.date, DatetimeSequenceExpr)
        self.assertIsInstance(self.expr.birth.max().date, DatetimeScalar)

        self.assertIsInstance(self.expr.birth.time, Time)
        self.assertIsInstance(self.expr.birth.time, DatetimeSequenceExpr)
        self.assertIsInstance(self.expr.birth.max().time, DatetimeScalar)

        self.assertIsInstance(self.expr.birth.year, Year)
        self.assertIsInstance(self.expr.birth.year, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().year, Int64Scalar)

        self.assertIsInstance(self.expr.birth.month, Month)
        self.assertIsInstance(self.expr.birth.month, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().month, Int64Scalar)

        self.assertIsInstance(self.expr.birth.day, Day)
        self.assertIsInstance(self.expr.birth.day, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().day, Int64Scalar)

        self.assertIsInstance(self.expr.birth.hour, Hour)
        self.assertIsInstance(self.expr.birth.hour, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().hour, Int64Scalar)

        self.assertIsInstance(self.expr.birth.minute, Minute)
        self.assertIsInstance(self.expr.birth.minute, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().minute, Int64Scalar)

        self.assertIsInstance(self.expr.birth.second, Second)
        self.assertIsInstance(self.expr.birth.second, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().second, Int64Scalar)

        self.assertIsInstance(self.expr.birth.microsecond, MicroSecond)
        self.assertIsInstance(self.expr.birth.microsecond, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().microsecond, Int64Scalar)

        self.assertIsInstance(self.expr.birth.week, Week)
        self.assertIsInstance(self.expr.birth.week, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().week, Int64Scalar)

        self.assertIsInstance(self.expr.birth.weekofyear, WeekOfYear)
        self.assertIsInstance(self.expr.birth.weekofyear, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().weekofyear, Int64Scalar)

        self.assertIsInstance(self.expr.birth.dayofweek, WeekDay)
        self.assertIsInstance(self.expr.birth.dayofweek, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().dayofweek, Int64Scalar)

        self.assertIsInstance(self.expr.birth.weekday, WeekDay)
        self.assertIsInstance(self.expr.birth.weekday, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().weekday, Int64Scalar)

        self.assertIsInstance(self.expr.birth.dayofyear, DayOfYear)
        self.assertIsInstance(self.expr.birth.dayofyear, Int64SequenceExpr)
        self.assertIsInstance(self.expr.birth.max().dayofyear, Int64Scalar)

        self.assertIsInstance(self.expr.birth.is_month_start, IsMonthStart)
        self.assertIsInstance(self.expr.birth.is_month_start, BooleanSequenceExpr)
        self.assertIsInstance(self.expr.birth.max().is_month_start, BooleanScalar)

        self.assertIsInstance(self.expr.birth.is_month_end, IsMonthEnd)
        self.assertIsInstance(self.expr.birth.is_month_end, BooleanSequenceExpr)
        self.assertIsInstance(self.expr.birth.max().is_month_end, BooleanScalar)

        self.assertIsInstance(self.expr.birth.is_year_start, IsYearStart)
        self.assertIsInstance(self.expr.birth.is_year_start, BooleanSequenceExpr)
        self.assertIsInstance(self.expr.birth.max().is_year_start, BooleanScalar)

        self.assertIsInstance(self.expr.birth.is_year_end, IsYearEnd)
        self.assertIsInstance(self.expr.birth.is_year_end, BooleanSequenceExpr)
        self.assertIsInstance(self.expr.birth.max().is_year_end, BooleanScalar)

        self.assertIsInstance(self.expr.birth.strftime('%Y'), Strftime)
        self.assertIsInstance(self.expr.birth.strftime('%Y'), StringSequenceExpr)
        self.assertIsInstance(self.expr.birth.max().strftime('%Y'), StringScalar)

        expr = self.expr.birth + hour(10)
        self.assertIsInstance(expr, DatetimeSequenceExpr)

        expr = self.expr.birth - microsecond(100)
        self.assertIsInstance(expr, DatetimeSequenceExpr)

        expr = self.expr.birth - datetime.now()
        self.assertIsInstance(expr, Int64SequenceExpr)

        self.assertRaises(ExpressionError, lambda: self.expr.birth + datetime.now())


if __name__ == '__main__':
    unittest.main()
