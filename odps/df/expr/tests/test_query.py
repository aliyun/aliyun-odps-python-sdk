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
from odps.df.expr.expressions import *
from odps.df.expr.tests.core import MockTable


class Test(TestBase):
    def setup(self):
        schema = TableSchema.from_lists(['name', 'id', 'fid'], [types.string, types.int64, types.float64])
        table = MockTable(name='pyodps_test_query_table', table_schema=schema)
        table._client = self.config.odps.rest
        self.expr = CollectionExpr(_source_data=table, _schema=schema)

    def testBaseQuery(self):
        expr = self.expr

        result = expr.query('@expr.id > 0')
        self.assertIsInstance(result, FilterCollectionExpr)

        result = expr.query('name == "test"')
        self.assertIsInstance(result, FilterCollectionExpr)

        result = expr.query('id + fid > id * fid')
        self.assertIsInstance(result, FilterCollectionExpr)

        result = expr.query('id ** fid <= id / fid - 1')
        self.assertIsInstance(result, FilterCollectionExpr)

    def testChainedCmp(self):
        expr = self.expr

        result = expr.query('id > 0 & fid < 10 and (name in ["test1", "test2"])')
        self.assertIsInstance(result, FilterCollectionExpr)

        result = expr.query('id >= 0 | fid in id or name != "test"')
        self.assertIsInstance(result, FilterCollectionExpr)

    def testLocalVariable(self):
        expr = self.expr
        id = 1
        name = ['test1', 'test2']

        result = expr.query('id + 1 > @id & name in @name')
        self.assertIsInstance(result, FilterCollectionExpr)

        self.assertRaises(KeyError, lambda: expr.query('id == @fid'))




