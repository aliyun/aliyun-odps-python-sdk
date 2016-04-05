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
from odps.errors import ODPSError
from odps.df.backends.engine import MixedEngine
from odps.df.backends.pd.engine import PandasEngine
from odps.df import DataFrame


class Test(TestBase):
    def setup(self):
        import pandas as pd

        odps_data = [
            ['name1', 1],
            ['name2', 2],
            ['name1', 3],
        ]

        pd_data = [
            ['name1', 5],
            ['name2', 6]
        ]

        names = ['name', 'id']
        types = ['string', 'bigint']

        table = 'pyodps_df_mixed'
        self.odps.delete_table(table, if_exists=True)
        self.t = self.odps.create_table(table, Schema.from_lists(names, types))
        with self.t.open_writer() as w:
            w.write([self.t.new_record(r) for r in odps_data])

        self.odps_df = DataFrame(self.t)
        self.pd_df = DataFrame(pd.DataFrame(pd_data, columns=names))

        self.engine = MixedEngine(self.odps)
        self.pd_engine = PandasEngine(self.odps)

    def teardown(self):
        self.t.drop()

    def testJoin(self):
        expr = self.odps_df.join(self.pd_df, 'name').sort('id_x')
        result = self.engine.execute(expr).values

        df = DataFrame(self.odps_df.to_pandas())
        expected = self.pd_engine.execute(df.join(self.pd_df, 'name').sort('id_x')).values
        self.assertTrue(result.equals(expected))

    def testUnion(self):
        expr = self.odps_df.union(self.pd_df).sort(['id', 'name'])
        result = self.engine.execute(expr).values

        df = DataFrame(self.odps_df.to_pandas())
        expected = self.pd_engine.execute(df.union(self.pd_df).sort(['id', 'name'])).values
        self.assertTrue(result.equals(expected))

    def testIsIn(self):
        expr = self.odps_df['name'].isin(self.pd_df['name']).rename('isin')
        result = self.engine.execute(expr).values

        df = DataFrame(self.odps_df.to_pandas())
        expected = self.pd_engine.execute(df['name'].isin(self.pd_df['name']).rename('isin')).values
        self.assertTrue(result.equals(expected))

    def testMixed(self):
        expr = self.odps_df.union(
            self.odps_df.join(self.pd_df, 'name')[
                lambda x: x.name_x.rename('name'),
                lambda x: x.id_x.rename('id')
            ]).sort(['name', 'id'])
        expr = expr[expr['name'].isin(self.pd_df['name'])]
        result = self.engine.execute(expr).values

        df = DataFrame(self.odps_df.to_pandas())
        test_expr = df.union(
            df.join(self.pd_df, 'name')[
                lambda x: x.name_x.rename('name'),
                lambda x: x.id_x.rename('id')
            ]).sort(['name', 'id'])
        test_expr = test_expr[test_expr['name'].isin(self.pd_df['name'])]
        expected = self.pd_engine.execute(test_expr).values

        self.assertTrue(result.equals(expected))

    def testPandasPersist(self):
        import pandas as pd, numpy as np

        self.odps.to_global()

        tmp_table_name = 'pyodps_test_mixed_persist'
        self.odps.delete_table(tmp_table_name, if_exists=True)

        pd_df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc'))
        df = DataFrame(pd_df).persist(tmp_table_name)

        self.assertTrue(df.to_pandas().equals(pd_df))

        self.odps.delete_table(tmp_table_name)

    def testExecuteCacheTable(self):
        df = self.odps_df[self.odps_df.name == 'name1']
        result = df.execute().values
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(df._cache_data)

        dag = df.compile()
        expr, _ = dag.nodes()[0]

        self.assertIsNotNone(expr._source_data)

        df2 = df[:5]
        result = df2.execute()
        self.assertEqual(len(result), 2)
        self.assertIsNone(expr._cache_data)

    def testCacheTable(self):
        df = self.odps_df.join(self.pd_df, 'name').cache()
        df2 = df.sort('id_x')

        dag = df2.compile()
        self.assertEqual(len(dag.nodes()), 3)

        result = self.engine.execute(df2).values

        df3 = DataFrame(self.odps_df.to_pandas())
        expected = self.pd_engine.execute(df3.join(self.pd_df, 'name').sort('id_x')).values
        self.assertTrue(result.equals(expected))

        self.assertEqual(len(self.engine._generated_table_names), 2)

        table = df._cache_data
        self.assertEqual(len(df.execute()), len(expected))

        self.assertIs(df._cache_data, table)

        df4 = df[df.id_x < 3].count()
        result = self.engine.execute(df4)
        self.assertEqual(result, 2)

        self.assertEqual(df4._cache_data, 2)

    def testUseCache(self):
        df = self.odps_df[self.odps_df['name'] == 'name1']
        self.assertEqual(len(df.head(10)), 2)

        df._cache_data.drop()

        self.assertRaises(ODPSError, lambda: self.engine.execute(df['name', 'id']))

        def plot(**_):
            pass
        self.assertRaises(ODPSError, lambda: df.plot(x='id', plot_func=plot))

    def testHeadAndTail(self):
        res = self.odps_df.head(2)
        self.assertEqual(len(res), 2)

        df = self.odps_df[self.odps_df['name'] == 'name1']
        res = df.head(1)
        self.assertEqual(len(res), 1)
        self.assertIsNotNone(df._cache_data)

        res = self.odps_df.tail(2)
        self.assertEqual(len(res), 2)
        self.assertTrue(all(it > 1 for it in res.values['id']))

        self.assertEqual(len(self.odps_df.name.head(2)), 2)
        self.assertEqual(len(self.odps_df.name.tail(2)), 2)

        res = self.pd_df.head(1)
        self.assertEqual(len(res), 1)

        df = self.pd_df[self.pd_df['name'] == 'name1']
        res = df.head(1)
        self.assertEqual(len(res), 1)
        self.assertIsNotNone(df._cache_data)

        res = self.pd_df.tail(1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.values['id'][0], 6)

        self.assertEqual(len(self.pd_df.name.head(1)), 1)
        self.assertEqual(len(self.pd_df.name.tail(1)), 1)

if __name__ == '__main__':
    unittest.main()