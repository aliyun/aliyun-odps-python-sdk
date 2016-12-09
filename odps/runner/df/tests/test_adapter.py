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
from odps.runner import adapter_from_df
from odps.df.expr.tests.core import MockTable
from odps.df.expr.merge import *
from odps.df import DataFrame, types


class Test(TestBase):
    def setup(self):
        schema = Schema.from_lists(['name', 'id', 'fid'], [types.string, types.int64, types.float64])
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        table._client = self.config.odps.rest
        self.expr = CollectionExpr(_source_data=table, _schema=schema)
        table1 = MockTable(name='pyodps_test_expr_table1', schema=schema)
        table1._client = self.config.odps.rest
        self.expr1 = CollectionExpr(_source_data=table1, _schema=schema)
        table2 = MockTable(name='pyodps_test_expr_table2', schema=schema)
        table2._client = self.config.odps.rest
        self.expr2 = CollectionExpr(_source_data=table2, _schema=schema)

    def testSimpleJoin(self):
        schema = Schema.from_lists(['name', 'id'], [types.string, types.int64])
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        expr = CollectionExpr(_source_data=table, _schema=schema)

        schema1 = Schema.from_lists(['id', 'value'], [types.int64, types.string])
        table1 = MockTable(name='pyodps_test_expr_table1', schema=schema1)
        expr1 = CollectionExpr(_source_data=table1, _schema=schema1)

        schema2 = Schema.from_lists(['value', 'num'], [types.string, types.float64])
        table2 = MockTable(name='pyodps_test_expr_table2', schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

        df = expr.join(expr1).join(expr2)
        adapter = adapter_from_df(df)
        self.assertEqual(len(adapter._bind_node.inputs), 0)
        self.assertEqual(len(adapter._bind_node.outputs), 1)
