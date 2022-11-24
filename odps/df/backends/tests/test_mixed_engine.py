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

import os
import time
import uuid

from odps import types as odps_types
from odps.tests.core import tn, pandas_case, odps2_typed_case
from odps.df.backends.tests.core import TestBase
from odps.config import options
from odps.compat import unittest, OrderedDict
from odps.models import Schema, Instance
from odps.df.backends.engine import MixedEngine
from odps.df.backends.odpssql.engine import ODPSSQLEngine
from odps.df.backends.pd.engine import PandasEngine
from odps.df.backends.seahawks.models import SeahawksTable
from odps.df.backends.context import context
from odps.df.utils import is_source_collection
from odps.df import DataFrame, output, func


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

        table = tn('pyodps_df_mixed_%d' % os.getpid())
        if self.odps.exist_table(table):
            self.t = self.odps.get_table(table)
        else:
            self.t = self.odps.create_table(table, Schema.from_lists(names, types), lifecycle=1)
            with self.t.open_writer() as w:
                w.write([self.t.new_record(r) for r in odps_data])

        self.odps_df = DataFrame(self.t)
        self.pd_df = DataFrame(pd.DataFrame(pd_data, columns=names))

        self.engine = MixedEngine(self.odps)
        self.pd_engine = PandasEngine(self.odps)

    def teardown(self):
        self.engine._selecter.force_odps = False

    def testGroupReduction(self):
        expr = self.odps_df.select(self.odps_df, id2=self.odps_df.id.map(lambda x: x + 1))
        expr = expr.groupby('name').id2.sum()

        expected = [
            ['name1', 6],
            ['name2', 3]
        ]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(sorted([[r[1]] for r in expected]), sorted(result))

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

        schema = Schema.from_lists([c.name for c in self.t.schema.columns if c.name != 'name'],
                                   [c.type for c in self.t.schema.columns if c.name != 'name'],
                                   ['name'], ['string'])
        t = self.odps.create_table('tmp_pyodps_%s' % str(uuid.uuid4()).replace('-', '_'), schema)
        try:
            expr = self.odps_df.union(self.pd_df)
            expr.persist(t.name, create_table=False, partitions=['name'])

            self.assertEqual(self.engine.execute(DataFrame(t).count()), 5)

            self.engine._selecter.force_odps = False
            df = DataFrame(t)
            self.assertGreaterEqual(
                len(self.engine.execute(df.filter(df.name > 'a', df.name < 'b'))), 0)
        finally:
            t.drop()

    def testIsIn(self):
        expr = self.odps_df['name'].isin(self.pd_df['name']).rename('isin')
        result = self.engine.execute(expr).values

        df = DataFrame(self.odps_df.to_pandas())
        expected = self.pd_engine.execute(df['name'].isin(self.pd_df['name']).rename('isin')).values
        self.assertTrue(result.equals(expected))

        expr = (self.odps_df.id + 2).isin(self.pd_df['id']).rename('isin')
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [[False], [False], [True]]
        self.assertEqual(result, expected)

    def testMixed(self):
        expr = self.odps_df.union(
            self.odps_df.join(self.pd_df, 'name')[
                lambda x: x.name,
                lambda x: x.id_x.rename('id')
            ]).sort(['name', 'id'])
        expr = expr[expr['name'].isin(self.pd_df['name'])]
        expr = expr[expr, func.rand(rtype='float').rename('rand')]
        result = self.engine.execute(expr).values[['name', 'id']]

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

        tmp_table_name = tn('pyodps_test_mixed_persist')
        self.odps.delete_table(tmp_table_name, if_exists=True)
        t = self.odps.create_table(tmp_table_name, ('a bigint, b bigint, c bigint', 'ds string'))
        t.create_partition('ds=today')
        try:
            pd_df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc'))
            df = DataFrame(pd_df).persist(tmp_table_name, partition='ds=today', odps=self.odps)

            self.assertPandasEqual(df[list('abc')].to_pandas(), pd_df)
        finally:
            self.odps.delete_table(tmp_table_name)

        self.odps.to_global()

        tmp_table_name = tn('pyodps_test_mixed_persist2')
        self.odps.delete_table(tmp_table_name, if_exists=True)

        try:
            pd_df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc'))
            df = DataFrame(pd_df).persist(tmp_table_name)

            self.assertPandasEqual(df.to_pandas(), pd_df)
        finally:
            self.odps.delete_table(tmp_table_name)

    @odps2_typed_case
    def testPandasPersistODPS2(self):
        import pandas as pd
        import numpy as np

        data_int8 = np.random.randint(0, 10, (1,), dtype=np.int8)
        data_int16 = np.random.randint(0, 10, (1,), dtype=np.int16)
        data_int32 = np.random.randint(0, 10, (1,), dtype=np.int32)
        data_int64 = np.random.randint(0, 10, (1,), dtype=np.int64)
        data_float32 = np.random.random((1,)).astype(np.float32)
        data_float64 = np.random.random((1,)).astype(np.float64)

        df = DataFrame(pd.DataFrame(OrderedDict([
            ('data_int8', data_int8), ('data_int16', data_int16),
            ('data_int32', data_int32), ('data_int64', data_int64),
            ('data_float32', data_float32), ('data_float64', data_float64)
        ])))
        tmp_table_name = tn('pyodps_test_mixed_persist_odps2_types')

        self.odps.delete_table(tmp_table_name, if_exists=True)
        df.persist(tmp_table_name, lifecycle=1, drop_table=True, odps=self.odps)

        t = self.odps.get_table(tmp_table_name)
        expected_types = [odps_types.tinyint, odps_types.smallint, odps_types.int_,
                          odps_types.bigint, odps_types.float_, odps_types.double]
        self.assertEqual(expected_types, t.schema.types)

    def testExecuteCacheTable(self):
        df = self.odps_df[self.odps_df.name == 'name1']
        result = df.execute().values
        self.assertEqual(len(result), 2)
        self.assertTrue(context.is_cached(df))

        dag = self.engine.compile(df)
        calls = dag.topological_sort()
        self.assertEqual(len(calls), 1)
        self.assertTrue(is_source_collection(calls[0].expr))

        df2 = df[:5]
        result = df2.execute()
        self.assertEqual(len(result), 2)

    def testHandleCache(self):
        df = self.pd_df['name', self.pd_df.id + 1]
        df.execute()
        self.assertTrue(context.is_cached(df))

        df2 = df[df.id < 10]
        dag = self.engine.compile(df2)
        self.assertEqual(len(dag.nodes()), 1)
        self.assertTrue(is_source_collection(dag.nodes()[0].expr.input))

        df3 = self.pd_df[self.pd_df.id < 10].count()
        i = df3.execute()
        self.assertTrue(context.is_cached(df3))

        df4 = df3 + 1
        dag = self.engine.compile(df4)
        self.assertEqual(len(dag.nodes()), 1)
        self.assertIsNotNone(dag.nodes()[0].expr._fields[0].lhs.value)
        self.assertEqual(df4.execute(), i + 1)

    def testCacheTable(self):
        self.engine._selecter.force_odps = True

        df = self.odps_df.join(self.pd_df, 'name').cache()
        df2 = df.sort('id_x')

        dag = self.engine.compile(df2)
        self.assertEqual(len(dag.nodes()), 3)

        result = self.engine.execute(df2).values

        df3 = DataFrame(self.odps_df.to_pandas())
        expected = self.pd_engine.execute(df3.join(self.pd_df, 'name').sort('id_x')).values
        self.assertTrue(result.equals(expected))

        self.assertEqual(len(self.engine._generated_table_names), 2)

        table = context.get_cached(df)
        self.assertEqual(len(self.engine.execute(df)), len(expected))

        self.assertIs(context.get_cached(df), table)
        if not isinstance(table, SeahawksTable):
            self.assertEqual(context.get_cached(df).lifecycle, 1)

        df4 = df[df.id_x < 3].count()
        result = self.engine.execute(df4)
        self.assertEqual(result, 2)

        self.assertEqual(context.get_cached(df4), 2)

    def testUseCache(self):
        self.engine._selecter.force_odps = True

        df_cache = self.odps_df[self.odps_df['name'] == 'name1'].cache()
        df = df_cache[df_cache.id * 2, df_cache.exclude('id')]
        self.assertEqual(len(self.engine.execute(df, head=10)), 2)

        context.get_cached(df_cache).drop()

        self.assertEqual(len(self.engine.execute(df_cache['name', df_cache.id * 2], head=10)), 2)
        self.assertTrue(context.is_cached(df_cache))
        self.assertTrue(self.odps.exist_table(context.get_cached(df_cache).name))

    def testHeadAndTail(self):
        res = self.odps_df.head(2)
        self.assertEqual(len(res), 2)

        df = self.odps_df[self.odps_df['name'] == 'name1']
        res = df.head(1)
        self.assertEqual(len(res), 1)
        self.assertTrue(context.is_cached(df))

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
        self.assertTrue(context.is_cached(df))

        res = self.pd_df.tail(1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.values['id'][0], 6)

        self.assertEqual(len(self.pd_df.name.head(1)), 1)
        self.assertEqual(len(self.pd_df.name.tail(1)), 1)

        class TunnelOnlyODPSEngine(ODPSSQLEngine):
            def _do_execute(self, *args, **kwargs):
                kwargs['_force_tunnel'] = True
                return super(TunnelOnlyODPSEngine, self)._do_execute(*args, **kwargs)

        engine = MixedEngine(self.odps)
        engine._odpssql_engine = TunnelOnlyODPSEngine(self.odps)

        res = engine.execute(self.odps_df['id'], head=3)
        self.assertIsNotNone(res)
        self.assertEqual(sum(res.values['id']), 6)

        table_name = tn('pyodps_df_mixed2')
        self.odps.delete_table(table_name, if_exists=True)
        table = next(self.odps_df.data_source())
        table2 = self.odps.create_table(table_name, table.schema)
        try:
            res = DataFrame(table2).head(10)
            self.assertEqual(len(res), 0)
        finally:
            table2.drop()

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

        odps_df2 = self.pd_df.persist(tn('pyodps_df_mixed2'), odps=self.odps)
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

        self.odps.write_table(table2, 0, data2)

        try:
            expr = self.odps_df.bloom_filter('name', expr2[:1].name, capacity=10)

            res = self.engine.execute(expr)

            self.assertTrue(np.all(res['name'] != 'name2'))
        finally:
            table2.drop()

    def testCachePersist(self):
        expr = self.odps_df

        data2 = [
            ['name1', 3.2],
            ['name3', 2.4]
        ]

        table_name = tn('pyodps_test_mixed_engine_cp_table2')
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name,
                                        schema=Schema.from_lists(['name', 'fid'], ['string', 'double']))
        expr2 = DataFrame(table2)
        self.odps.write_table(table2, 0, data2)

        @output(expr.schema.names, expr.schema.types)
        def h(row):
            yield row

        l = expr.filter(expr.id > 0).apply(h, axis=1).cache()
        r = expr2.filter(expr2.fid > 0)
        joined = l.join(r, on=['name', r.fid < 4])['id', 'fid'].cache()

        output_table = tn('pyodps_test_mixed_engine_cp_output_table')
        self.odps.delete_table(output_table, if_exists=True)
        schema = Schema.from_lists(['id', 'fid'], ['bigint', 'double'], ['ds'], ['string'])
        output_t = self.odps.create_table(output_table, schema, if_not_exists=True)

        t = joined.persist(output_table, partition='ds=today', create_partition=True)
        self.assertEqual(len(t.execute()), 2)

        # test seahawks fallback
        self.assertEqual(t.input.count().execute(), 2)

        output_t.drop()

    def testBigintPartitionedCache(self):
        table = tn('pyodps_test_bigint_partitioned_cache')
        self.odps.delete_table(table, if_exists=True)
        expr = self.odps_df.persist(table, partitions=['id'])

        @output(['id', 'name'], ['int', 'string'])
        def handle(row):
            return row.id + 1, row.name

        expr = expr['tt' + expr.name, expr.id].cache()
        new_expr = expr.map_reduce(mapper=handle)

        res = self.engine.execute(new_expr)
        self.assertEqual(len(res), 3)

    def testAsync(self):
        expr = self.odps_df[self.odps_df.name == 'name1']
        future = self.engine.execute(expr, async_=True)
        self.assertFalse(future.done())
        res = future.result()
        self.assertEqual(len(res), 2)

    def testBatch(self):
        odps_expr = self.odps_df[self.odps_df.id < 4].cache()
        expr = odps_expr.join(self.pd_df, 'name').sort('id_x')

        dag = self.engine.compile(expr)
        self.assertEqual(len(dag.nodes()), 3)

        f = self.engine.execute(expr, async_=True, n_parallel=2)

        result = f.result().values

        df = DataFrame(self.odps_df.to_pandas())
        expected = self.pd_engine.execute(df.join(self.pd_df, 'name').sort('id_x')).values
        self.assertTrue(result.equals(expected))

    def testBatchStop(self):
        self.engine._selecter.force_odps = True

        expr1 = self.odps_df[self.odps_df.id < 3].cache()
        expr2 = self.odps_df[self.odps_df.id > 3].cache()
        expr3 = expr1.union(expr2)

        self.engine.execute([expr1, expr2, expr3], n_parallel=2, async_=True)
        time.sleep(2)

        instance_ids = self.engine._odpssql_engine._instances
        self.assertEqual(len(instance_ids), 2)

        self.engine.stop()
        instances = [self.odps.get_instance(i) for i in instance_ids]
        [i.wait_for_completion() for i in instances]
        self.assertEqual(list(instances[0].get_task_statuses().values())[0].status,
                         Instance.Task.TaskStatus.CANCELLED)
        self.assertEqual(list(instances[1].get_task_statuses().values())[0].status,
                         Instance.Task.TaskStatus.CANCELLED)

    def testFailure(self):
        from odps.df.backends.errors import DagDependencyError

        def err_maker(x):
            raise ValueError(x)

        expr1 = self.odps_df[self.odps_df.id.map(err_maker), ].cache()
        expr2 = expr1.count()

        fs = self.engine.execute(expr2, async_=True)
        self.assertRaises(DagDependencyError, fs.result)

    def testAppendIDCache(self):
        options.ml.dry_run = False

        @output(['id1'] + self.odps_df.schema.names, ['int'] + self.odps_df.schema.types)
        def h(row):
            yield row

        expr1 = self.odps_df.append_id(id_col='id1').apply(h, axis=1)
        expr2 = self.odps_df.append_id(id_col='id2')
        expr3 = expr1.join(expr2, on='id')['id1', 'id2']
        self.assertEqual(len(expr3.execute()), 3)

    def testAppendId(self):
        options.ml.dry_run = False

        expr = self.odps_df['name', ].distinct()
        expr = expr.append_id(id_col='id2')
        expr = expr.join(self.odps_df, on=['name'])
        tablename = tn('pyodps_test_append_id_persist')
        self.odps.delete_table(tablename, if_exists=True)
        expr.persist(tablename, partitions=['name'], lifecycle=1)

    def testHorzConcat(self):
        options.ml.dry_run = False

        table_name = tn('test_horz_concat_table2_xxx_yyy')
        self.odps.delete_table(table_name, if_exists=True)

        result_table_name = tn('test_horz_concat_result')
        self.odps.delete_table(result_table_name, if_exists=True)

        self.odps_df[self.odps_df.name, (self.odps_df.id * 2).rename('ren_id')].persist(table_name)
        df2 = self.odps.get_table(table_name).to_df()
        df2 = df2[:3]
        expr = self.odps_df.concat(df2.ren_id, axis=1)
        expr.persist(result_table_name, lifecycle=1)

    def testAsTypeMapReduce(self):
        expr = self.odps_df[self.odps_df.exclude('id'), self.odps_df.id.astype('float')]
        expr = expr.filter(expr.id < 10)['id', 'name']

        @output(['id', 'name'], ['float', 'string'])
        def h(group):
            def inn(row, done):
                yield row

            return inn

        expr = expr.map_reduce(reducer=h)
        expr.execute()

        expr = self.odps_df[self.odps_df.exclude('id'), self.odps_df.id.astype('float')]
        expr = expr.filter(expr.id < 10).distinct('id', 'name')

        @output(['id', 'name'], ['float', 'string'])
        def h(group):
            def inn(row, done):
                yield row

            return inn

        expr = expr.map_reduce(reducer=h)
        expr.execute()

if __name__ == '__main__':
    unittest.main()