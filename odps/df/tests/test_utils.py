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


from odps.tests.core import TestBase, tn
from odps.models import Schema
from odps.df import DataFrame
from odps.df.backends.utils import fetch_data_source_size


class Test(TestBase):

    def setup(self):
        test_table_name = tn('pyodps_test_dataframe')
        schema = Schema.from_lists(['id', 'name'], ['bigint', 'string'],
                                   ['ds', 'mm', 'hh'], ['string'] * 3)

        self.odps.delete_table(test_table_name, if_exists=True)
        self.table = self.odps.create_table(test_table_name, schema)

        self.pt = 'ds=today,mm=now,hh=curr'
        self.table.create_partition(self.pt)
        with self.table.open_writer(self.pt) as w:
            w.write([[1, 'name1'], [2, 'name2'], [3, 'name3']])

    def teardown(self):
        self.table.drop()

    def testFetchTableSize(self):
        df = DataFrame(self.table)

        expr = df.filter_parts(self.pt)
        dag = expr.to_dag(copy=False)
        self.assertGreater(fetch_data_source_size(dag, df, self.table), 0)

        expr = df.filter_parts('ds=today,hh=curr,mm=now')
        dag = expr.to_dag(copy=False)
        self.assertGreater(fetch_data_source_size(dag, df, self.table), 0)

        expr = df.filter_parts('ds=today,hh=curr,mm=now2')
        dag = expr.to_dag(copy=False)
        self.assertIsNone(fetch_data_source_size(dag, df, self.table))

        expr = df.filter_parts('ds=today,hh=curr')
        dag = expr.to_dag(copy=False)
        self.assertIsNone(fetch_data_source_size(dag, df, self.table))

        expr = df.filter_parts('ds=today,mm=now')
        dag = expr.to_dag(copy=False)
        self.assertGreater(fetch_data_source_size(dag, df, self.table), 0)

        expr = df.filter(df.ds == 'today', df.mm == 'now', df.hh == 'curr')
        dag = expr.to_dag(copy=False)
        self.assertGreater(fetch_data_source_size(dag, df, self.table), 0)

        expr = df.filter(df.ds == 'today', df.hh == 'curr', df.mm == 'now')
        dag = expr.to_dag(copy=False)
        self.assertGreater(fetch_data_source_size(dag, df, self.table), 0)

        expr = df.filter(df.ds == 'today', df.hh == 'curr', df.mm == 'now2')
        dag = expr.to_dag(copy=False)
        self.assertIsNone(fetch_data_source_size(dag, df, self.table))

        expr = df.filter(df.ds == 'today', df.hh == 'curr')
        dag = expr.to_dag(copy=False)
        self.assertIsNone(fetch_data_source_size(dag, df, self.table))

        expr = df.filter(df.ds == 'today', df.mm == 'now')
        dag = expr.to_dag(copy=False)
        self.assertGreater(fetch_data_source_size(dag, df, self.table), 0)
