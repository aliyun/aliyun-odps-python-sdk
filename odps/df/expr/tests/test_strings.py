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
from odps.df.types import validate_data_type
from odps.df.expr.tests.core import MockTable
from odps.df.expr.expressions import *
from odps.df.expr.strings import *


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        table = MockTable(name='pyodps_test_expr_table', schema=schema)

        self.expr = CollectionExpr(_source_data=table, _schema=schema)

    def testStrings(self):
        self.assertRaises(AttributeError, lambda: self.expr.id.strip())
        self.assertRaises(AttributeError, lambda: self.expr.fid.upper())
        self.assertRaises(AttributeError, lambda: self.expr.isMale.lower())
        self.assertRaises(AttributeError, lambda: self.expr.scale.repeat(3))
        self.assertRaises(AttributeError, lambda: self.expr.birth.len())

        self.assertIsInstance(self.expr.name.capitalize(), Capitalize)
        self.assertIsInstance(self.expr.name.capitalize(), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().capitalize(), StringScalar)

        self.assertIsInstance(self.expr.name.contains('test'), Contains)
        self.assertIsInstance(self.expr.name.contains('test'), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().contains('test'), BooleanScalar)

        self.assertIsInstance(self.expr.name.count('test'), Count)
        self.assertIsInstance(self.expr.name.count('test'), Int64SequenceExpr)
        self.assertIsInstance(self.expr.name.sum().count('test'), Int64Scalar)

        self.assertIsInstance(self.expr.name.endswith('test'), Endswith)
        self.assertIsInstance(self.expr.name.endswith('test'), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().endswith('test'), BooleanScalar)

        self.assertIsInstance(self.expr.name.startswith('test'), Startswith)
        self.assertIsInstance(self.expr.name.startswith('test'), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().startswith('test'), BooleanScalar)

        self.assertIsInstance(self.expr.name.extract('[ab](\d)'), Extract)
        self.assertIsInstance(self.expr.name.extract('[ab](\d)'), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().extract('[ab](\d)'), StringScalar)

        self.assertIsInstance(self.expr.name.find('test'), Find)
        self.assertIsInstance(self.expr.name.find('test'), Int64SequenceExpr)
        self.assertIsInstance(self.expr.name.sum().find('test'), Int64Scalar)

        self.assertIsInstance(self.expr.name.rfind('test'), RFind)
        self.assertIsInstance(self.expr.name.rfind('test'), Int64SequenceExpr)
        self.assertIsInstance(self.expr.name.sum().rfind('test'), Int64Scalar)

        self.assertIsInstance(self.expr.name.replace('test', 'test2'), Replace)
        self.assertIsInstance(self.expr.name.replace('test', 'test2'), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().replace('test', 'test2'), StringScalar)

        self.assertIsInstance(self.expr.name.get(1), Get)
        self.assertIsInstance(self.expr.name.get(1), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().get(1), StringScalar)

        self.assertIsInstance(self.expr.name.len(), Len)
        self.assertIsInstance(self.expr.name.len(), Int64SequenceExpr)
        self.assertIsInstance(self.expr.name.sum().len(), Int64Scalar)

        self.assertIsInstance(self.expr.name.ljust(3), Ljust)
        self.assertIsInstance(self.expr.name.ljust(3), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().ljust(3), StringScalar)

        self.assertIsInstance(self.expr.name.rjust(3), Rjust)
        self.assertIsInstance(self.expr.name.rjust(3), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().rjust(3), StringScalar)

        self.assertIsInstance(self.expr.name.lower(), Lower)
        self.assertIsInstance(self.expr.name.lower(), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().lower(), StringScalar)

        self.assertIsInstance(self.expr.name.upper(), Upper)
        self.assertIsInstance(self.expr.name.upper(), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().upper(), StringScalar)

        self.assertIsInstance(self.expr.name.lstrip(), Lstrip)
        self.assertIsInstance(self.expr.name.lstrip(), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().lstrip(), StringScalar)

        self.assertIsInstance(self.expr.name.strip(), Strip)
        self.assertIsInstance(self.expr.name.strip(), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().strip(), StringScalar)

        self.assertIsInstance(self.expr.name.pad(4), Pad)
        self.assertIsInstance(self.expr.name.pad(4), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().pad(4), StringScalar)

        self.assertIsInstance(self.expr.name.repeat(4), Repeat)
        self.assertIsInstance(self.expr.name.repeat(4), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().repeat(4), StringScalar)

        self.assertIsInstance(self.expr.name.slice(0, 4), Slice)
        self.assertIsInstance(self.expr.name.slice(0, 4), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().slice(0, 4), StringScalar)

        self.assertIsInstance(self.expr.name.swapcase(), Swapcase)
        self.assertIsInstance(self.expr.name.swapcase(), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().swapcase(), StringScalar)

        self.assertIsInstance(self.expr.name.title(), Title)
        self.assertIsInstance(self.expr.name.title(), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().title(), StringScalar)

        self.assertIsInstance(self.expr.name.zfill(5), Zfill)
        self.assertIsInstance(self.expr.name.zfill(5), StringSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().zfill(5), StringScalar)

        self.assertIsInstance(self.expr.name.isalnum(), Isalnum)
        self.assertIsInstance(self.expr.name.isalnum(), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().isalnum(), BooleanScalar)

        self.assertIsInstance(self.expr.name.isalpha(), Isalpha)
        self.assertIsInstance(self.expr.name.isalpha(), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().isalpha(), BooleanScalar)

        self.assertIsInstance(self.expr.name.isdigit(), Isdigit)
        self.assertIsInstance(self.expr.name.isdigit(), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().isdigit(), BooleanScalar)

        self.assertIsInstance(self.expr.name.isspace(), Isspace)
        self.assertIsInstance(self.expr.name.isspace(), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().isspace(), BooleanScalar)

        self.assertIsInstance(self.expr.name.islower(), Islower)
        self.assertIsInstance(self.expr.name.islower(), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().islower(), BooleanScalar)

        self.assertIsInstance(self.expr.name.isupper(), Isupper)
        self.assertIsInstance(self.expr.name.isupper(), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().isupper(), BooleanScalar)

        self.assertIsInstance(self.expr.name.istitle(), Istitle)
        self.assertIsInstance(self.expr.name.istitle(), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().istitle(), BooleanScalar)

        self.assertIsInstance(self.expr.name.isnumeric(), Isnumeric)
        self.assertIsInstance(self.expr.name.isnumeric(), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().isnumeric(), BooleanScalar)

        self.assertIsInstance(self.expr.name.isdecimal(), Isdecimal)
        self.assertIsInstance(self.expr.name.isdecimal(), BooleanSequenceExpr)
        self.assertIsInstance(self.expr.name.sum().isdecimal(), BooleanScalar)


if __name__ == '__main__':
    unittest.main()
