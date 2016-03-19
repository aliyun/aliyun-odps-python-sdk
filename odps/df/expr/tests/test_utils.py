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
from odps.df.expr.utils import *
from odps.models import Schema
from odps.df.expr.expressions import *
from odps.df import types
from odps.df.expr.tests.core import MockTable


class Test(TestBase):

    def testGetAttrs(self):
        schema = Schema.from_lists(['name', 'id'], [types.string, types.int64])
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        expr = CollectionExpr(_source_data=table, _schema=schema)

        expected = ('_lhs', '_rhs', '_data_type', '_source_data_type', '_name',
                    '_source_name', '_engine', '_cache_data', '_need_cache', '_cached_args')
        self.assertSequenceEqual(expected, get_attrs(expr.id + 1))

if __name__ == '__main__':
    unittest.main()