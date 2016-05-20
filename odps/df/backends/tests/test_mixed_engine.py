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

from odps.tests.core import TestBase, tn, pandas_case
from odps.compat import unittest
from odps.models import Schema
from odps.errors import ODPSError
from odps.df.backends.engine import MixedEngine
from odps.df.backends.pd.engine import PandasEngine
from odps.df import DataFrame, output


@pandas_case
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

        table = tn('pyodps_df_mixed')
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

    def assertPandasEqual(self, df1, df2):
        from odps.compat import six
        from odps import types as o_types
        from pandas.util.testing import assert_frame_equal

        # compare column types
        def get_odps_type(p_type):
            for data_type, builtin_type in six.iteritems(o_types._odps_primitive_to_builtin_types):
                if issubclass(p_type.type, builtin_type):
                    return data_type

        types1 = [get_odps_type(dt) for dt in df1.dtypes]
        types2 = [get_odps_type(dt) for dt in df2.dtypes]
        self.assertSequenceEqual(types1, types2)
        assert_frame_equal(df1, df2, check_dtype=False)

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
                lambda x: x.name,
                lambda x: x.id_x.rename('id')
            ]).sort(['name', 'id'])
        expr = expr[expr['name'].isin(self.pd_df['name'])]
        result = self.engine.execute(expr).values

        df = DataFrame(self.odps_df.to_pandas())
        test_expr = df.union(
            df.join(self.pd_df, 'name')[
                lambda x: x.name,
                lambda x: x.id_x.rename('id')
            ]).sort(['name', 'id'])
        test_expr = test_expr[test_expr['name'].isin(self.pd_df['name'])]
        expected = self.pd_engine.execute(test_expr).values

        self.assertTrue(result.equals(expected))

    def testPandasPersist(self):
        import pandas as pd, numpy as np

        self.odps.to_global()

        tmp_table_name = tn('pyodps_test_mixed_persist')
        self.odps.delete_table(tmp_table_name, if_exists=True)

        pd_df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc'))
        df = DataFrame(pd_df).persist(tmp_table_name)

        self.assertPandasEqual(df.to_pandas(), pd_df)

        self.odps.delete_table(tmp_table_name)

    def testExecuteCacheTable(self):
        df = self.odps_df[self.odps_df.name == 'name1']
        result = df.execute().values
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(df._cache_data)

        new_df = self.engine._pre_process(df)
        _, new_df, cbs = self.engine._compile(new_df)
        try:
            self.assertIsNotNone(new_df._source_data)
        finally:
            [cb() for cb in cbs]

        df2 = df[:5]
        result = df2.execute()
        self.assertEqual(len(result), 2)

    def testHandleCache(self):
        df = self.pd_df['name', self.pd_df.id + 1]
        df.execute()
        self.assertIsNotNone(df._cache_data)

        df2 = df[df.id < 10]
        new_df2 = self.engine._pre_process(df2)
        _, new_df2, cbs = self.engine._compile(new_df2)
        try:
            self.assertIsNotNone(new_df2.input._source_data)
        finally:
            [cb() for cb in cbs]

    def testCacheTable(self):
        df = self.odps_df.join(self.pd_df, 'name').cache()
        df2 = df.sort('id_x')

        dag = self.engine._compile_dag(df2)
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

    def testMapReduceWithResource(self):
        pd_df2 = self.odps_df.to_pandas(wrap=True)

        @output(['name', 'id'], ['string', 'int'])
        def reducer(resources):
            d = dict()
            for r in resources[0]:
                if r.name in d:
                    d[r.name] += r.id
                else:
                    d[r.name] = r.id

            def inner(keys):

                def h(row, done):
                    if row.name in d:
                        d[row.name] += row.id
                    else:
                        d[row.name] = row.id

                    if done:
                        yield row.name, d[row.name]
                return h
            return inner

        expr = pd_df2.map_reduce(reducer=reducer, reducer_resources=[self.pd_df], group='name')
        result = expr.execute()
        self.assertEqual(result.values['id'].sum(), 17)

        odps_df2 = self.pd_df.persist('pyodps_df_mixed2', odps=self.odps)
        try:
            expr = self.odps_df.map_reduce(reducer=reducer, reducer_resources=[odps_df2], group='name')
            result = expr.execute()
            self.assertEqual(result.values['id'].sum(), 17)

            expr = self.odps_df.map_reduce(reducer=reducer, reducer_resources=[self.pd_df], group='name')
            result = expr.execute()
            self.assertEqual(result.values['id'].sum(), 17)

            expr = pd_df2.map_reduce(reducer=reducer, reducer_resources=[odps_df2], group='name')
            result = expr.execute()
            self.assertEqual(result.values['id'].sum(), 17)
        finally:
            next(odps_df2.data_source()).drop()

    def testBloomFilter(self):
        import numpy as np

        data2 = [
            ['name1'],
            ['name3']
        ]

        table_name = tn('pyodps_test_mixed_engine_bf_table2')
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name,
                                        schema=Schema.from_lists(['name'], ['string']))
        expr2 = DataFrame(table2)

        self.odps.write_table(table2, 0, [table2.new_record(values=d) for d in data2])

        try:
            expr = self.odps_df.bloom_filter('name', expr2[:1].name, capacity=10)

            res = self.engine.execute(expr)

            self.assertTrue(np.all(res['name'] != 'name2'))
        finally:
            table2.drop()

if __name__ == '__main__':
    unittest.main()