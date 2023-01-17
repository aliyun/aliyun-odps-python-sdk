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
from odps.df.expr.utils import *
from odps.models import TableSchema
from odps.df.expr.expressions import *
from odps.df import types
from odps.df.expr.tests.core import MockTable


class Test(TestBase):

    def testGetAttrs(self):
        schema = TableSchema.from_lists(['name', 'id'], [types.string, types.int64])
        table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
        expr = CollectionExpr(_source_data=table, _schema=schema)

        expected = ('_lhs', '_rhs', '_data_type', '_source_data_type',
                    '_ml_fields_cache', '_ml_uplink', '_ml_operations',
                    '_name', '_source_name', '_deps', '_ban_optimize', '_engine',
                    '_need_cache', '_mem_cache', '_id', '_args_indexes', )
        self.assertSequenceEqual(expected, get_attrs(expr.id + 1))

    def testIsChanged(self):
        schema = TableSchema.from_lists(['name', 'id'], [types.string, types.int64])
        table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
        expr = CollectionExpr(_source_data=table, _schema=schema)
        expr2 = CollectionExpr(_source_data=table, _schema=schema)

        self.assertFalse(is_changed(expr[expr.id < 3], expr.id))
        self.assertTrue(is_changed(expr[expr.id + 2,], expr.id))
        self.assertIsNone(is_changed(expr[expr.id < 3], expr2.id))
        self.assertTrue(is_changed(expr.groupby('name').agg(id=expr.id.sum()), expr.id))
        self.assertFalse(is_changed(expr.groupby('name').agg(id=expr.id.sum()), expr.name))

if __name__ == '__main__':
    unittest.main()