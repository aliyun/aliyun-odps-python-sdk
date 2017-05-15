#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

import math
import itertools
from datetime import datetime, timedelta
from functools import partial
from random import randint
import uuid
import os
import zipfile
import tarfile
import re
import functools
import time

from odps.df.backends.tests.core import TestBase, to_str, tn
from odps.compat import unittest, irange as xrange, six, OrderedDict, futures
from odps.models import Schema
from odps import types
from odps.df.types import validate_data_type, DynamicSchema
from odps.df.backends.odpssql.types import df_schema_to_odps_schema, \
    odps_schema_to_df_schema
from odps.df.backends.context import context
from odps.df.expr.expressions import CollectionExpr
from odps.df.backends.odpssql.engine import ODPSSQLEngine
from odps.df import Scalar, output_names, output_types, output, day, millisecond, agg
from odps import options


class ODPSEngine(ODPSSQLEngine):

    def _pre_process(self, expr):
        src_expr = expr
        expr = self._convert_table(expr)
        return self._analyze(expr.to_dag(), src_expr)


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        self.schema = df_schema_to_odps_schema(schema)
        table_name = tn('pyodps_test_engine_table_%s' % str(uuid.uuid4()).replace('-', '_'))
        self.odps.delete_table(table_name, if_exists=True)
        self.table = self.odps.create_table(
                name=table_name, schema=self.schema)
        self.expr = CollectionExpr(_source_data=self.table, _schema=schema)

        self.engine = ODPSEngine(self.odps)

        class FakeBar(object):
            def update(self, *args, **kwargs):
                pass

            def inc(self, *args, **kwargs):
                pass

            def status(self, *args, **kwargs):
                pass
        self.faked_bar = FakeBar()

    def _gen_data(self, rows=None, data=None, nullable_field=None, value_range=None):
        if data is None:
            data = []
            for _ in range(rows):
                record = []
                for t in self.schema.types:
                    method = getattr(self, '_gen_random_%s' % t.name)
                    if t.name == 'bigint':
                        record.append(method(value_range=value_range))
                    else:
                        record.append(method())
                data.append(record)

            if nullable_field is not None:
                j = self.schema._name_indexes[nullable_field]
                for i, l in enumerate(data):
                    if i % 2 == 0:
                        data[i][j] = None

        self.odps.write_table(self.table, 0, data)
        return data

    def testTunnelCases(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr.count()
        res = self.engine._handle_cases(expr, self.faked_bar)
        result = self._get_result(res)
        self.assertEqual(10, result)

        expr = self.expr.name.count()
        res = self.engine._handle_cases(expr, self.faked_bar)
        result = self._get_result(res)
        self.assertEqual(10, result)

        res = self.engine._handle_cases(self.expr, self.faked_bar)
        result = self._get_result(res)
        self.assertEqual(data, result)
        self.assertIsNotNone(next(res)['name'])

        expr = self.expr['name', self.expr.id.rename('new_id')]
        res = self.engine._handle_cases(expr, self.faked_bar)
        result = self._get_result(res)
        self.assertEqual([it[:2] for it in data], result)

        expr = self.expr['name']
        res = self.engine._handle_cases(
            self.engine._pre_process(expr), self.faked_bar)
        result = self._get_result(res)
        self.assertEqual([it[:1] for it in data], result)

        table_name = tn('pyodps_test_engine_partitioned')
        self.odps.delete_table(table_name, if_exists=True)

        df = self.engine.persist(self.expr, table_name, partitions=['name'])

        try:
            expr = df.count()
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertIsNone(res)

            expr = df[df.name == data[0][0]]['fid', 'id'].count()
            expr = self.engine._pre_process(expr)
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(res, 0)

            expr = df[df.name == data[0][0]]['fid', 'id']
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(len(res), 0)

            expr = df[df.name == data[0][0]]
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertTrue(all(r is not None for r in res[:, -1]))

            expr = df[df.name == data[1][0]]
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertTrue(all(r is not None for r in res[:, -1]))

            expr = df.filter_partition('name={0}'.format(data[0][0])).count()
            expr = self.engine._pre_process(expr)
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(res, 0)

            expr = df.filter_partition('name={0}'.format(data[0][0]))
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(len(res), 0)

            expr = df.filter_partition('name={0}'.format(data[0][0]))['fid', 'id'].count()
            expr = self.engine._pre_process(expr)
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(res, 0)

            expr = df.filter_partition('name={0}'.format(data[0][0]))['fid', 'id']
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(len(res), 0)
        finally:
            self.odps.delete_table(table_name, if_exists=True)

        df = self.engine.persist(self.expr, table_name, partitions=['name', 'id'])

        try:
            expr = df.filter(df.ismale == data[0][3],
                             df.name == data[0][0], df.id == data[0][1],
                             df.scale == data[0][4]
                             )
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertIsNone(res)

            expr = df.count()
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertIsNone(res)

            expr = df[(df.name == data[0][0]) & (df.id == data[0][1])]['fid', 'ismale'].count()
            expr = self.engine._pre_process(expr)
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(res, 0)

            expr = df[(df.name == data[0][0]) & (df.id == data[0][1])]['fid', 'ismale']
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(len(res), 0)
        finally:
            self.odps.delete_table(table_name, if_exists=True)

        table =self.odps.create_table(
            table_name, Schema.from_lists(['val'], ['bigint'], ['name', 'id'], ['string', 'bigint']))
        table.create_partition('name=a,id=1')
        with table.open_writer('name=a,id=1') as writer:
            writer.write([[0], [1], [2]])
        table.create_partition('name=a,id=2')
        with table.open_writer('name=a,id=2') as writer:
            writer.write([[3], [4], [5]])
        table.create_partition('name=b,id=1')
        with table.open_writer('name=b,id=1') as writer:
            writer.write([[6], [7], [8]])

        df = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(table.schema))

        try:
            expr = df.count()
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertIsNone(res)

            expr = df[df.name == 'a'].count()
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertIsNone(res)

            expr = df[df.id == 1].count()
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertIsNone(res)

            expr = df.filter(df.name == 'a', df.id == 1).count()
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertEqual(res, 3)

            expr = df
            res = self.engine._handle_cases(expr, self.faked_bar, update_progress_count=1)
            self.assertEqual(len(res), 9)
            self.assertIsNotNone(res[0][-1])
            self.assertIsNotNone(res[0][-2])

            expr = df[df.name == 'a']
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertEqual(len(res), 6)

            expr = df[df.id == 1]
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertIsNone(res)

            expr = df[df.name == 'a'][:4]
            res = self.engine._handle_cases(expr, self.faked_bar, head=5)
            result = self._get_result(res)
            self.assertEqual(sum(r[0] for r in result), 6)

            expr = df[df.name == 'a'][:5]
            res = self.engine._handle_cases(expr, self.faked_bar, head=4)
            result = self._get_result(res)
            self.assertEqual(sum(r[0] for r in result), 6)

            expr = df[df.name == 'a']
            res = self.engine._handle_cases(expr, self.faked_bar, head=4)
            result = self._get_result(res)
            self.assertEqual(sum(r[0] for r in result), 6)

            expr = df[df.name == 'a'][:5]
            res = self.engine._handle_cases(expr, self.faked_bar, tail=4)
            self.assertIsNone(res)

            expr = df.filter(df.name == 'a', df.id == 1)[:2]
            res = self.engine._handle_cases(expr, self.faked_bar, tail=1)
            result = self._get_result(res)
            self.assertEqual(sum(r[0] for r in result), 1)
        finally:
            self.odps.delete_table(table_name, if_exists=True)

    def testAsync(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr.id.sum()

        future = self.engine.execute(expr, async=True, priority=4, ret_instance=True)
        self.assertFalse(future.done())
        inst, res = future.result()

        self.assertEqual(sum(it[1] for it in data), res)
        self.assertEqual(inst.priority, 4)

    def testNoPermission(self):
        class NoPermissionEngine(ODPSEngine):
            def _handle_cases(self, *args, **kwargs):
                from odps.errors import NoPermission
                raise NoPermission('No permission to use tunnel')

            def _open_reader(self, t, **kwargs):
                raise RuntimeError('Cannot use tunnel')

        data = self._gen_data(10, value_range=(-1000, 1000))

        res = NoPermissionEngine(self.odps).execute(self.expr)
        result = self._get_result(res)
        self.assertEqual(data, result)

    def testNoTunnelCase(self):
        class NoTunnelCaseEngine(ODPSEngine):
            def _handle_cases(self, *args, **kwargs):
                raise RuntimeError('Not allow to use tunnel')

        data = self._gen_data(10, value_range=(-1000, 1000))

        options.df.optimizes.tunnel = False

        try:
            res = NoTunnelCaseEngine(self.odps).execute(self.expr)
            result = self._get_result(res)
            self.assertEqual(data, result)
        finally:
            options.df.optimizes.tunnel = True

    def testCache(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr[self.expr.id < 10].cache()
        cnt = expr.count()

        dag = self.engine.compile(expr)
        self.assertEqual(len(dag.nodes()), 2)

        res = self.engine.execute(cnt)
        self.assertEqual(len([it for it in data if it[1] < 10]), res)
        self.assertTrue(context.is_cached(expr))

        table = context.get_cached(expr)
        table.reload()
        self.assertEqual(table.lifecycle, 1)

        expr2 = expr['id']
        res = self.engine.execute(expr2, _force_tunnel=True)
        result = self._get_result(res)
        self.assertEqual([[it[1]] for it in data if it[1] < 10], result)

        expr3 = self.expr.sample(parts=10)
        expr3.cache()
        self.assertIsNotNone(self.engine.execute(expr3.id.sum()))

        expr4 = self.expr['id', 'name', 'fid'].cache()
        expr5 = expr4[expr4.id < 0]['id', 'name']
        expr6 = expr4[expr4.id >= 0]['name', 'id']
        expr7 = expr5.union(expr6).union(expr4['name', 'id'])
        self.assertGreater(self.engine.execute(expr7.count()), 0)

    def testBatch(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr[self.expr.id < 10].cache()
        expr1 = expr.id.sum()
        expr2 = expr.id.mean()

        dag = self.engine.compile([expr1, expr2])
        self.assertEqual(len(dag.nodes()), 3)
        self.assertEqual(sum(len(v) for v in dag._graph.values()), 2)

        expect1 = sum(d[1] for d in data if d[1] < 10)
        length = len([d[1] for d in data if d[1] < 10])
        expect2 = (expect1 / float(length)) if length > 0 else 0.0

        res = self.engine.execute([expr1, expr2], n_parallel=2)
        self.assertEqual(res[0], expect1)
        self.assertAlmostEqual(res[1], expect2)
        self.assertTrue(context.is_cached(expr))

        # test async and timeout
        expr = self.expr[self.expr.id < 10].cache()
        expr1 = expr.id.sum()
        expr2 = expr.id.mean()

        fs = self.engine.execute([expr, expr1, expr2], n_parallel=2, async=True, timeout=1)
        self.assertEqual(len(fs), 3)

        futures.wait((fs[0], ), timeout=120)
        time.sleep(.5)
        self.assertTrue(fs[1].running() and fs[2].running())
        time.sleep(.5)
        self.assertTrue(fs[2].running() or fs[2].done())

        self.assertEqual(fs[1].result(), expect1)
        self.assertAlmostEqual(fs[2].result(), expect2)
        self.assertTrue(context.is_cached(expr))

    def testBase(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr[self.expr.id < 10]['name', lambda x: x.id, ]
        result = self._get_result(self.engine.execute(expr).values)
        self.assertEqual(len([it for it in data if it[1] < 10]), len(result))
        if len(result) > 0:
            self.assertEqual(2, len(result[0]))

        expr = self.expr[Scalar(3).rename('const'), self.expr.id, (self.expr.id + 1).rename('id2')]
        res = self.engine.execute(expr)
        result = self._get_result(res.values)
        self.assertEqual([c.name for c in res.columns], ['const', 'id', 'id2'])
        self.assertTrue(all(it[0] == 3 for it in result))
        self.assertEqual(len(data), len(result))
        self.assertEqual([it[1]+1 for it in data], [it[2] for it in result])

        expr = self.expr.sort('id')[:5]
        res = self.engine.execute(expr)
        result = self._get_result(res.values)
        self.assertEqual(sorted(data, key=lambda it: it[1])[:5], result)

        expr = self.expr.sort('id')[:5]
        # test do not use tunnel
        res = self.engine.execute(expr, use_tunnel=False)
        result = self._get_result(res.values)
        self.assertEqual(sorted(data, key=lambda it: it[1])[:5], result)

        exp = self.expr[
            self.expr.scale.map(lambda x: x + 1).rename('scale1'),
            self.expr.scale.map(functools.partial(lambda v, x: x + v, 10)).rename('scale2'),
        ]
        res = self.engine.execute(exp)
        result = self._get_result(res.values)
        self.assertEqual([[r[4] + 1, r[4] + 10] for r in data], result)

        if six.PY2:  # Skip in Python 3, as hash() behaves randomly.
            expr = self.expr.name.hash()
            res = self.engine.execute(expr)
            result = self._get_result(res.values)
            self.assertEqual([[hash(r[0])] for r in data], result),

        expr = self.expr.sample(parts=10)
        res = self.engine.execute(expr)
        self.assertGreaterEqual(len(res), 1)

        expr = self.expr.sample(parts=10, columns=self.expr.id)
        res = self.engine.execute(expr)
        self.assertGreaterEqual(len(res), 0)

        expr = self.expr[:1].filter(lambda x: x.name == data[1][0])
        res = self.engine.execute(expr)
        self.assertEqual(len(res), 0)

        expr = self.expr.exclude('scale').describe()
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertSequenceEqual(result[0][1:], [10, 10])

    def testChinese(self):
        data = [
            ['中文', 4, 5.3, None, None, None],
            ['\'中文2', 2, 3.5, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr.filter(self.expr.name == '中文')
        res = self.engine.execute(expr)
        self.assertEqual(len(res), 1)

        expr = self.expr.filter(self.expr.name == '\'中文2')
        res = self.engine.execute(expr)
        self.assertEqual(len(res), 1)

        expr = self.expr.filter(self.expr.name == u'中文')
        res = self.engine.execute(expr)
        self.assertEqual(len(res), 1)

    def testFunctions(self):
        data = self._gen_data(5, value_range=(-1000, 1000))

        f = lambda x: x + 1

        expr = self.expr[self.expr.id.map(f).rename('id1'),
                         self.expr.fid.map(f).rename('id2'),
                         (self.expr.id + 1).map(f).rename('id3')]
        expected = [[r[1] + 1, r[2] + 1, r[1] + 2] for r in data]
        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(expected, result)
        self.assertEqual(len(self.engine._ctx._registered_funcs.values()), 2)

        self.engine._ctx._registered_funcs.clear()

        def h(row):
            yield row

        expr1 = self.expr['name', 'id'].apply(h, axis=1, names=['name', 'id'], types=['string', 'int'])
        expr2 = self.expr['id', 'name'].apply(h, axis=1, names=['id', 'name'], types=['int', 'string'])
        expr3 = expr1.join(expr2)
        self.assertEqual(len(self.engine.execute(expr3)), 5)

        self.assertEqual(len(self.engine._ctx._registered_funcs.values()), 2)

    def testElement(self):
        data = self._gen_data(5, nullable_field='name')

        fields = [
            self.expr.name.isnull().rename('name1'),
            self.expr.name.notnull().rename('name2'),
            self.expr.name.fillna('test').rename('name3'),
            self.expr.id.isin([1, 2, 3]).rename('id1'),
            self.expr.id.isin(self.expr.fid.astype('int')).rename('id2'),
            self.expr.id.notin([1, 2, 3]).rename('id3'),
            self.expr.id.notin(self.expr.fid.astype('int')).rename('id4'),
            self.expr.id.between(self.expr.fid, 3).rename('id5'),
            self.expr.name.fillna('test').switch('test', 'test' + self.expr.name.fillna('test'),
                                                 'test2', 'test2' + self.expr.name.fillna('test'),
                                                 default=self.expr.name).rename('name4'),
            self.expr.name.fillna('test').switch('test', 1, 'test2', 2).rename('name5'),
            self.expr.id.cut([100, 200, 300],
                             labels=['xsmall', 'small', 'large', 'xlarge'],
                             include_under=True, include_over=True).rename('id6'),
            self.expr.birth.unix_timestamp.to_datetime().rename('birth1'),
        ]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(len(data), len(result))

        self.assertEqual(len([it for it in data if it[0] is None]),
                         len([it[0] for it in result if it[0]]))

        self.assertEqual(len([it[0] for it in data if it[0] is not None]),
                         len([it[1] for it in result if it[1]]))

        self.assertEqual([(it[0] if it[0] is not None else 'test') for it in data],
                         [it[2] for it in result])

        self.assertEqual([(it[1] in (1, 2, 3)) for it in data],
                         [it[3] for it in result])

        fids = [int(it[2]) for it in data]
        self.assertEqual([(it[1] in fids) for it in data],
                         [it[4] for it in result])

        self.assertEqual([(it[1] not in (1, 2, 3)) for it in data],
                         [it[5] for it in result])

        self.assertEqual([(it[1] not in fids) for it in data],
                         [it[6] for it in result])

        self.assertEqual([(it[2] <= it[1] <= 3) for it in data],
                         [it[7] for it in result])

        self.assertEqual([to_str('testtest' if it[0] is None else it[0]) for it in data],
                         [to_str(it[8]) for it in result])

        self.assertEqual([to_str(1 if it[0] is None else None) for it in data],
                         [to_str(it[9]) for it in result])

        self.assertListEqual([it[5] for it in data],
                             [it[11] for it in result])

        def get_val(val):
            if val <= 100:
                return 'xsmall'
            elif 100 < val <= 200:
                return 'small'
            elif 200 < val <= 300:
                return 'large'
            else:
                return 'xlarge'
        self.assertEqual([to_str(get_val(it[1])) for it in data], [to_str(it[10]) for it in result])

    def testArithmetic(self):
        data = self._gen_data(5, value_range=(-1000, 1000))

        fields = [
            (self.expr.id + 1).rename('id1'),
            (self.expr.fid - 1).rename('fid1'),
            (self.expr.scale * 2).rename('scale1'),
            (self.expr.scale + self.expr.id).rename('scale2'),
            (self.expr.id / 2).rename('id2'),
            (self.expr.id ** 2).rename('id3'),
            abs(self.expr.id).rename('id4'),
            (~self.expr.id).rename('id5'),
            (-self.expr.fid).rename('fid2'),
            (~self.expr.isMale).rename('isMale1'),
            (-self.expr.isMale).rename('isMale2'),
            (self.expr.id // 2).rename('id6'),
            (self.expr.birth + day(1).rename('birth1')),
            (self.expr.birth - (self.expr.birth - millisecond(10))).rename('birth2'),
            (self.expr.id % 2).rename('id7'),
        ]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(len(data), len(result))

        self.assertEqual([it[1] + 1 for it in data],
                         [it[0] for it in result])

        self.assertAlmostEqual([it[2] - 1 for it in data],
                               [it[1] for it in result])

        self.assertEqual([it[4] * 2 for it in data],
                         [it[2] for it in result])

        self.assertEqual([it[4] + it[1] for it in data],
                         [it[3] for it in result])

        self.assertAlmostEqual([float(it[1]) / 2 for it in data],
                               [it[4] for it in result])

        self.assertEqual([int(it[1] ** 2) for it in data],
                         [it[5] for it in result])

        self.assertEqual([abs(it[1]) for it in data],
                         [it[6] for it in result])

        self.assertEqual([~it[1] for it in data],
                         [it[7] for it in result])

        self.assertAlmostEqual([-it[2] for it in data],
                               [it[8] for it in result])

        self.assertEqual([not it[3] for it in data],
                         [it[9] for it in result])

        self.assertEqual([it[1] // 2 for it in data],
                         [it[11] for it in result])

        self.assertEqual([it[5] + timedelta(days=1) for it in data],
                         [it[12] for it in result])

        self.assertEqual([10] * len(data), [it[13] for it in result])

        self.assertEqual([it[1] % 2 for it in data],
                         [it[14] for it in result])

    def testGenFunc(self):
        data = self._gen_data(5, value_range=(1, 1))

        def func(x, val):
            return x + val

        expr = self.expr[self.expr.id.map(func, args=(1,)),
                         self.expr.id.map(func, args=(2,)).rename('id2')]
        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertTrue(all(r[0] == 2 for r in result))
        self.assertTrue(all(r[1] == 3 for r in result))

        def func(x, val):
            return x + val['id']

        expr = self.expr[self.expr.id.map(func, args=({'id': 1},)),
                         self.expr.id.map(func, args=({'id': 2},)).rename('id2')]
        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertTrue(all(r[0] == 2 for r in result))
        self.assertTrue(all(r[1] == 3 for r in result))

    def testMath(self):
        data = self._gen_data(5, value_range=(1, 90))

        if hasattr(math, 'expm1'):
            expm1 = math.expm1
        else:
            expm1 = lambda x: 2 * math.exp(x / 2.0) * math.sinh(x / 2.0)

        methods_to_fields = [
            (math.sin, self.expr.id.sin()),
            (math.cos, self.expr.id.cos()),
            (math.tan, self.expr.id.tan()),
            (math.sinh, self.expr.id.sinh()),
            (math.cosh, self.expr.id.cosh()),
            (math.tanh, self.expr.id.tanh()),
            (math.log, self.expr.id.log()),
            (lambda v: math.log(v, 2), self.expr.id.log2()),
            (math.log10, self.expr.id.log10()),
            (math.log1p, self.expr.id.log1p()),
            (math.exp, self.expr.id.exp()),
            (expm1, self.expr.id.expm1()),
            (math.acosh, self.expr.id.arccosh()),
            (math.asinh, self.expr.id.arcsinh()),
            (math.atanh, self.expr.id.arctanh()),
            (math.atan, self.expr.id.arctan()),
            (math.sqrt, self.expr.id.sqrt()),
            (abs, self.expr.id.abs()),
            (math.ceil, self.expr.id.ceil()),
            (math.floor, self.expr.id.floor()),
            (math.trunc, self.expr.id.trunc()),
        ]

        fields = [it[1].rename('id'+str(i)) for i, it in enumerate(methods_to_fields)]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        for i, it in enumerate(methods_to_fields):
            mt = it[0]

            def method(v):
                try:
                    return mt(v)
                except ValueError:
                    return float('nan')

            first = [method(it[1]) for it in data]
            second = [it[i] for it in result]
            self.assertEqual(len(first), len(second))
            for it1, it2 in zip(first, second):
                not_valid = lambda x: \
                    x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
                if not_valid(it1) and not_valid(it2):
                    continue
                if isinstance(it1, float) and it1 > 1.0e15:
                    scale = 0.1 ** (int(math.log10(it1)) - 15)
                    self.assertAlmostEqual(it1 * scale, it2 * scale, delta=2)
                else:
                    self.assertAlmostEqual(it1, it2, delta=2)

    def testString(self):
        data = self._gen_data(5)

        def extract(x, pat, group):
            regex = re.compile(pat)
            m = regex.match(x)
            if m:
                return m.group(group)

        methods_to_fields = [
            (lambda s: s.capitalize(), self.expr.name.capitalize()),
            (lambda s: data[0][0] in s, self.expr.name.contains(data[0][0], regex=False)),
            (lambda s: s[0] + '|' + str(s[1]), self.expr.name.cat(self.expr.id.astype('string'), sep='|')),
            (lambda s: s.count(data[0][0]), self.expr.name.count(data[0][0])),
            (lambda s: s.endswith(data[0][0]), self.expr.name.endswith(data[0][0])),
            (lambda s: s.startswith(data[0][0]), self.expr.name.startswith(data[0][0])),
            (lambda s: extract('123'+s, r'[^a-z]*(\w+)', group=1), ('123'+self.expr.name).extract(r'[^a-z]*(\w+)', group=1)),
            (lambda s: s.find(data[0][0]), self.expr.name.find(data[0][0])),
            (lambda s: s.rfind(data[0][0]), self.expr.name.rfind(data[0][0])),
            (lambda s: s.replace(data[0][0], 'test'), self.expr.name.replace(data[0][0], 'test')),
            (lambda s: s[0], self.expr.name.get(0)),
            (lambda s: len(s), self.expr.name.len()),
            (lambda s: s.ljust(10), self.expr.name.ljust(10)),
            (lambda s: s.ljust(20, '*'), self.expr.name.ljust(20, fillchar='*')),
            (lambda s: s.rjust(10), self.expr.name.rjust(10)),
            (lambda s: s.rjust(20, '*'), self.expr.name.rjust(20, fillchar='*')),
            (lambda s: s * 4, self.expr.name.repeat(4)),
            (lambda s: s[1:], self.expr.name.slice(1)),
            (lambda s: s[1: 6], self.expr.name.slice(1, 6)),
            (lambda s: s[2: 10: 2], self.expr.name.slice(2, 10, 2)),
            (lambda s: s[1: s.find('a')], self.expr.name[1: self.expr.name.find('a')]),
            (lambda s: s[-5: -1], self.expr.name.slice(-5, -1)),
            (lambda s: s.title(), self.expr.name.title()),
            (lambda s: s.rjust(20, '0'), self.expr.name.zfill(20)),
            (lambda s: s.isalnum(), self.expr.name.isalnum()),
            (lambda s: s.isalpha(), self.expr.name.isalpha()),
            (lambda s: s.isdigit(), self.expr.name.isdigit()),
            (lambda s: s.isspace(), self.expr.name.isspace()),
            (lambda s: s.isupper(), self.expr.name.isupper()),
            (lambda s: s.istitle(), self.expr.name.istitle()),
            (lambda s: to_str(s).isnumeric(), self.expr.name.isnumeric()),
            (lambda s: to_str(s).isdecimal(), self.expr.name.isdecimal()),
        ]

        fields = [it[1].rename('id'+str(i)) for i, it in enumerate(methods_to_fields)]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        for i, it in enumerate(methods_to_fields):
            method = it[0]

            if i != 2:
                first = [method(it[0]) for it in data]
            else:
                # cat
                first = [method(it) for it in data]
            second = [it[i] for it in result]
            self.assertEqual(first, second)

    def testFunctionResources(self):
        data = self._gen_data(5)

        class my_func(object):
            def __init__(self, resources):
                self.file_resource = resources[0]
                self.table_resource = resources[1]

                self.valid_ids = [int(l) for l in self.file_resource]
                self.valid_ids.extend([int(l[0]) for l in self.table_resource])

            def __call__(self, arg):
                if isinstance(arg, tuple):
                    if arg[1] in self.valid_ids:
                        return arg
                else:
                    if arg in self.valid_ids:
                        return arg

        def my_func2(resources):
            file_resource = resources[0]
            table_resource = resources[1]

            valid_ids = [int(l) for l in file_resource]
            valid_ids.extend([int(l[0]) for l in table_resource])

            def h(arg):
                if isinstance(arg, tuple):
                    if arg[1] in valid_ids:
                        return arg
                else:
                    if arg in valid_ids:
                        return arg
            return h

        file_resource_name = tn('pyodps_tmp_file_resource')
        table_resource_name = tn('pyodps_tmp_table_resource')
        table_name = tn('pyodps_tmp_function_resource_table')
        try:
            self.odps.delete_resource(file_resource_name)
        except:
            pass
        file_resource = self.odps.create_resource(file_resource_name, 'file',
                                                  file_obj='\n'.join(str(r[1]) for r in data[:3]))
        self.odps.delete_table(table_name, if_exists=True)
        t = self.odps.create_table(table_name, Schema.from_lists(['id'], ['bigint']))
        with t.open_writer() as writer:
            writer.write([r[1: 2] for r in data[3: 4]])
        try:
            self.odps.delete_resource(table_resource_name)
        except:
            pass
        table_resource = self.odps.create_resource(table_resource_name, 'table',
                                                   table_name=t.name)

        try:
            expr = self.expr.id.map(my_func, resources=[file_resource, table_resource])

            res = self.engine.execute(expr)
            result = self._get_result(res)
            result = [r for r in result if r[0] is not None]

            self.assertEqual(sorted([[r[1]] for r in data[:4]]), sorted(result))

            expr = self.expr['name', 'id', 'fid']
            expr = expr.apply(my_func, axis=1, resources=[file_resource, table_resource],
                              names=expr.schema.names, types=expr.schema.types)

            res = self.engine.execute(expr)
            result = self._get_result(res)

            self.assertEqual(sorted([r[:3] for r in data[:4]]), sorted(result))

            expr = self.expr['name', 'id', 'fid']
            expr = expr.apply(my_func2, axis=1, resources=[file_resource, table_resource],
                              names=expr.schema.names, types=expr.schema.types)

            res = self.engine.execute(expr)
            result = self._get_result(res)

            self.assertEqual(sorted([r[:3] for r in data[:4]]), sorted(result))
        finally:
            try:
                file_resource.drop()
            except:
                pass
            try:
                t.drop()
            except:
                pass
            try:
                table_resource.drop()
            except:
                pass

    def testThirdPartyLibraries(self):
        import requests
        from odps.compat import BytesIO

        data = [
            ['2016-08-18', 4, 5.3, None, None, None],
            ['2015-08-18', 2, 3.5, None, None, None],
            ['2014-08-18', 4, 4.2, None, None, None],
            ['2013-08-18', 3, 2.2, None, None, None],
            ['2012-08-18', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        dateutil_urls = [
            'http://mirrors.aliyun.com/pypi/packages/33/68/'
            '9eadc96f9899caebd98f55f942d6a8f3fb2b8f8e69ba81a0f771269897e9/'
            'python_dateutil-2.5.3-py2.py3-none-any.whl#md5=dbcd46b171e01d4518db96e3571810db',
            'http://mirrors.aliyun.com/pypi/packages/3e/f5/'
            'aad82824b369332a676a90a8c0d1e608b17e740bbb6aeeebca726f17b902/'
            'python-dateutil-2.5.3.tar.gz#md5=05ffc6d2cc85a7fd93bb245807f715ef',
            'http://mirrors.aliyun.com/pypi/packages/b7/9f/'
            'ba2b6aaf27e74df59f31b77d1927d5b037cc79a89cda604071f93d289eaf/'
            'python-dateutil-2.5.3.zip#md5=52b3f339f41986c25c3a2247e722db17'
        ]
        dateutil_resources = []
        for dateutil_url, name in zip(dateutil_urls, ['dateutil.whl', 'dateutil.tar.gz', 'dateutil.zip']):
            obj = BytesIO(requests.get(dateutil_url).content)
            res_name = '%s_%s.%s' % (
                name.split('.', 1)[0], str(uuid.uuid4()).replace('-', '_'), name.split('.', 1)[1])
            res = self.odps.create_resource(res_name, 'file', file_obj=obj)
            dateutil_resources.append(res)

        obj = BytesIO(requests.get(dateutil_urls[0]).content)
        res_name = 'dateutil_archive_%s.zip' % str(uuid.uuid4()).replace('-', '_')
        res = self.odps.create_resource(res_name, 'archive', file_obj=obj)
        dateutil_resources.append(res)

        resources = []
        six_path = os.path.join(os.path.dirname(os.path.abspath(six.__file__)), 'six.py')

        zip_io = BytesIO()
        zip_f = zipfile.ZipFile(zip_io, 'w')
        zip_f.write(six_path, arcname='mylib/six.py')
        zip_f.close()
        zip_io.seek(0)

        rn = 'six_%s.zip' % str(uuid.uuid4())
        resource = self.odps.create_resource(rn, 'file', file_obj=zip_io)
        resources.append(resource)

        tar_io = BytesIO()
        tar_f = tarfile.open(fileobj=tar_io, mode='w:gz')
        tar_f.add(six_path, arcname='mylib/six.py')
        tar_f.close()
        tar_io.seek(0)

        rn = 'six_%s.tar.gz' % str(uuid.uuid4())
        resource = self.odps.create_resource(rn, 'file', file_obj=tar_io)
        resources.append(resource)

        try:
            for resource in resources:
                for dateutil_resource in dateutil_resources:
                    def f(x):
                        from dateutil.parser import parse
                        return int(parse(x).strftime('%Y'))

                    expr = self.expr.name.map(f, rtype='int')

                    res = self.engine.execute(expr, libraries=[resource.name, dateutil_resource])
                    result = self._get_result(res)

                    self.assertEqual(result, [[int(r[0].split('-')[0])] for r in data])

                    def f(row):
                        from dateutil.parser import parse
                        return int(parse(row.name).strftime('%Y')),

                    expr = self.expr.apply(f, axis=1, names=['name', ], types=['int', ])

                    res = self.engine.execute(expr, libraries=[resource, dateutil_resource])
                    result = self._get_result(res)

                    self.assertEqual(result, [[int(r[0].split('-')[0])] for r in data])

                    class Agg(object):
                        def buffer(self):
                            return [0]

                        def __call__(self, buffer, val):
                            from dateutil.parser import parse
                            buffer[0] += int(parse(val).strftime('%Y'))

                        def merge(self, buffer, pbuffer):
                            buffer[0] += pbuffer[0]

                        def getvalue(self, buffer):
                            return buffer[0]

                    expr = self.expr.name.agg(Agg, rtype='int')

                    options.df.libraries = [resource.name, dateutil_resource]
                    try:
                        res = self.engine.execute(expr)
                    finally:
                        options.df.libraries = None

                    self.assertEqual(res, sum([int(r[0].split('-')[0]) for r in data]))
        finally:
            [res.drop() for res in resources + dateutil_resources]

    def testApply(self):
        data = self._gen_data(5)

        def my_func(row):
            return row.name, row.scale + 1, row.birth

        expr = self.expr['name', 'id', 'scale', 'birth'].apply(my_func, axis=1, names=['name', 'scale', 'birth'],
                                                               types=['string', 'decimal', 'datetime'])

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual([[r[0], r[1]] for r in result], [[r[0], r[4] + 1] for r in data])

        def my_func2(row):
            yield len(row.name)
            yield row.id

        expr = self.expr['name', 'id'].apply(my_func2, axis=1, names='cnt', types='int')
        expr = expr.filter(expr.cnt > 1)

        res = self.engine.execute(expr)
        result = self._get_result(res)

        def gen_expected(data):
            for r in data:
                yield len(r[0])
                yield r[1]

        self.assertEqual([r[0] for r in result], [r for r in gen_expected(data) if r > 1])

    def testDatetime(self):
        data = self._gen_data(5)

        def date_value(sel):
            if isinstance(sel, six.string_types):
                fun = lambda v: getattr(v, sel)
            else:
                fun = sel
            col_id = [idx for idx, col in enumerate(self.schema.names) if col == 'birth'][0]
            return [fun(row[col_id]) for row in data]

        methods_to_fields = [
            (partial(date_value, 'year'), self.expr.birth.year),
            (partial(date_value, 'month'), self.expr.birth.month),
            (partial(date_value, 'day'), self.expr.birth.day),
            (partial(date_value, 'hour'), self.expr.birth.hour),
            (partial(date_value, 'minute'), self.expr.birth.minute),
            (partial(date_value, 'second'), self.expr.birth.second),
            (partial(date_value, lambda d: d.isocalendar()[1]), self.expr.birth.weekofyear),
            (partial(date_value, lambda d: d.weekday()), self.expr.birth.dayofweek),
            (partial(date_value, lambda d: d.weekday()), self.expr.birth.weekday),
            (partial(date_value, lambda d: time.mktime(d.timetuple())), self.expr.birth.unix_timestamp),
            (partial(date_value, lambda d: datetime.combine(d.date(), datetime.min.time())), self.expr.birth.date),
            (partial(date_value, lambda d: d.strftime('%Y%d')), self.expr.birth.strftime('%Y%d')),
            (partial(date_value, lambda d: datetime.strptime(d.strftime('%Y%d'), '%Y%d')), self.expr.birth.strftime('%Y%d').strptime('%Y%d')),
        ]

        fields = [it[1].rename('birth'+str(i)) for i, it in enumerate(methods_to_fields)]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        for i, it in enumerate(methods_to_fields):
            method = it[0]

            first = method()

            try:
                import pandas as pd

                def conv(v):
                    if isinstance(v, pd.Timestamp):
                        return v.to_datetime()
                    else:
                        return v
            except ImportError:
                conv = lambda v: v

            second = [conv(it[i]) for it in result]
            self.assertEqual(first, second)

    def testSortDistinct(self):
        data = [
            ['name1', 4, None, None, None, None],
            ['name2', 2, None, None, None, None],
            ['name1', 4, None, None, None, None],
            ['name1', 3, None, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr.sort(['name', -self.expr.id]).distinct(['name', lambda x: x.id + 1])[:50]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(len(result), 3)

        expected = [
            ['name1', 5],
            ['name1', 4],
            ['name2', 3]
        ]
        self.assertEqual(sorted(expected), sorted(result))

    def testPivot(self):
        data = [
            ['name1', 1, 1.0, True, None, None],
            ['name1', 2, 2.0, True, None, None],
            ['name2', 1, 3.0, False, None, None],
            ['name2', 3, 4.0, False, None, None]
        ]
        self._gen_data(data=data)

        expr = self.expr

        expr1 = expr.pivot(rows='id', columns='name', values='fid').distinct()
        res = self.engine.execute(expr1)
        result = self._get_result(res)

        expected = [
            [1, 1.0, 3.0],
            [2, 2.0, None],
            [3, None, 4.0]
        ]
        self.assertEqual(sorted(result), sorted(expected))

        expr2 = expr.pivot(rows='id', columns='name', values=['fid', 'isMale'])
        res = self.engine.execute(expr2)
        result = self._get_result(res)

        expected = [
            [1, 1.0, 3.0, True, False],
            [2, 2.0, None, True, None],
            [3, None, 4.0, None, False]
        ]
        self.assertEqual(sorted(result), sorted(expected))

        expr3 = expr.pivot(rows='id', columns='name', values='fid')['name3']
        with self.assertRaises(ValueError) as cm:
            self.engine.execute(expr3)
        self.assertIn('name3', str(cm.exception))

        expr4 = expr.pivot(rows='id', columns='name', values='fid')['id', 'name1']
        res = self.engine.execute(expr4)
        result = self._get_result(res)

        expected = [
            [1, 1.0],
            [2, 2.0],
            [3, None]
        ]
        self.assertEqual(sorted(result), sorted(expected))

        expr5 = expr.pivot(rows='id', columns='name', values='fid')
        expr5 = expr5[expr5, (expr5['name1'].astype('int') + 1).rename('new_name')]
        res = self.engine.execute(expr5)
        result = self._get_result(res)

        expected = [
            [1, 1.0, 3.0, 2.0],
            [2, 2.0, None, 3.0],
            [3, None, 4.0, None]
        ]
        self.assertEqual(sorted(result), sorted(expected))

        odps_data = [
            ['name1', 1],
            ['name2', 2],
            ['name1', 3],
        ]

        names = ['name', 'id']
        types = ['string', 'bigint']

        table = tn('pyodps_df_pivot_to_join')
        self.odps.delete_table(table, if_exists=True)
        t = self.odps.create_table(table, Schema.from_lists(names, types))
        with t.open_writer() as w:
            w.write([t.new_record(r) for r in odps_data])

        from odps.df import DataFrame
        self.odps_df = DataFrame(t)

        try:
            expr6 = expr.pivot(rows='id', columns='name', values='fid')
            expr6 = expr6.join(self.odps_df, on='id')[expr6, 'name']
            res = self.engine.execute(expr6)
            result = self._get_result(res)

            expected = [
                [1, 1.0, 3.0, 'name1'],
                [2, 2.0, None, 'name2'],
                [3, None, 4.0, 'name1']
            ]
            self.assertEqual(sorted(result), sorted(expected))
        finally:
            t.drop()

    def testPivotTable(self):
        data = [
            ['name1', 1, 1.0, True, None, None],
            ['name1', 1, 5.0, True, None, None],
            ['name1', 2, 2.0, True, None, None],
            ['name2', 1, 3.0, False, None, None],
            ['name2', 3, 4.0, False, None, None]
        ]

        self._gen_data(data=data)

        expr = self.expr

        expr1 = expr.pivot_table(rows='name', values='fid')
        res = self.engine.execute(expr1)
        result = self._get_result(res)

        expected = [
            ['name1', 8.0 / 3],
            ['name2', 3.5],
        ]
        self.assertEqual(sorted(result), sorted(expected))

        expr2 = expr.pivot_table(rows='name', values='fid', aggfunc=['mean', 'sum'])
        res = self.engine.execute(expr2)
        result = self._get_result(res)

        expected = [
            ['name1', 8.0 / 3, 8.0],
            ['name2', 3.5, 7.0],
        ]
        self.assertEqual(res.schema.names, ['name', 'fid_mean', 'fid_sum'])
        self.assertEqual(sorted(result), sorted(expected))

        expr5 = expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum'])
        expr6 = expr5['name1_fid_mean',
                      expr5.groupby(Scalar(1)).sort('name1_fid_mean').name1_fid_mean.astype('float').cumsum()]

        k = lambda x: list(0 if it is None else it for it in x)

        # TODO: fix this situation, act different compared to pandas
        expected = [
            [2, 2], [3, 5], [None, None]
        ]
        res = self.engine.execute(expr6)
        result = self._get_result(res)
        self.assertEqual(sorted(result, key=k), sorted(expected, key=k))

        expr3 = expr.pivot_table(rows='id', values='fid', columns='name', fill_value=0).distinct()
        res = self.engine.execute(expr3)
        result = self._get_result(res)

        expected = [
            [1, 3.0, 3.0],
            [2, 2.0, 0],
            [3, 0, 4.0]
        ]

        self.assertEqual(res.schema.names, ['id', 'name1_fid_mean', 'name2_fid_mean'])
        self.assertEqual(result, expected)

        class Agg(object):
            def buffer(self):
                return [0]

            def __call__(self, buffer, val):
                buffer[0] += val

            def merge(self, buffer, pbuffer):
                buffer[0] += pbuffer[0]

            def getvalue(self, buffer):
                return buffer[0]

        aggfuncs = OrderedDict([('my_sum', Agg), ('mean', 'mean')])
        expr4 = expr.pivot_table(rows='id', values='fid', columns='name', fill_value=0,
                                 aggfunc=aggfuncs)
        res = self.engine.execute(expr4)
        result = self._get_result(res)

        expected = [
            [1, 6.0, 3.0, 3.0, 3.0],
            [2, 2.0, 0, 2.0, 0],
            [3, 0, 4.0, 0, 4.0]
        ]

        self.assertEqual(res.schema.names, ['id', 'name1_fid_my_sum', 'name2_fid_my_sum',
                                            'name1_fid_mean', 'name2_fid_mean'])
        self.assertEqual(result, expected)

        expr7 = expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum']).cache()
        self.assertEqual(len(self.engine.execute(expr7)), 3)

        expr5 = self.expr.pivot_table(rows='id', values='fid', columns='name').cache()
        expr6 = expr5[expr5['name1_fid_mean'].rename('tname1'), expr5['name2_fid_mean'].rename('tname2')]

        @output(['tname1', 'tname2'], ['float', 'float'])
        def h(row):
            yield row.tname1, row.tname2

        expr6 = expr6.map_reduce(mapper=h)
        self.assertEqual(len(self.engine.execute(expr6)), 3)

        expr8 = self.expr.pivot_table(rows='id', values='fid', columns='name')
        self.assertEqual(len(self.engine.execute(expr8)), 3)
        self.assertNotIsInstance(expr8.schema, DynamicSchema)
        expr9 =(expr8['name1_fid_mean'] - expr8['name2_fid_mean']).rename('substract')
        self.assertEqual(len(self.engine.execute(expr9)), 3)
        expr10 = expr8.distinct()
        self.assertEqual(len(self.engine.execute(expr10)), 3)

    def testExtractKV(self):
        from odps.df import DataFrame

        data = [
            ['name1', 'k1=1,k2=3,k5=10', '1=5,3=7,2=1'],
            ['name1', '', '3=1,4=2'],
            ['name1', 'k1=7.1,k7=8.2', '1=1,5=6'],
            ['name2', 'k2=1.2,k3=1.5', None],
            ['name2', 'k9=1.1,k2=1', '4=2']
        ]

        table_name = tn('pyodps_test_mixed_engine_extract_kv')
        self.odps.delete_table(table_name, if_exists=True)
        table = self.odps.create_table(
            name=table_name,
            schema=Schema.from_lists(['name', 'kv', 'kv2'],
                                     ['string', 'string', 'string']))
        expr = DataFrame(table)
        try:
            self.odps.write_table(table, 0, data)

            expr1 = expr.extract_kv(columns=['kv', 'kv2'], kv_delim='=')
            res = self.engine.execute(expr1)
            result = self._get_result(res)

            expected_cols = [
                'name',
                'kv_k1', 'kv_k2', 'kv_k3', 'kv_k5', 'kv_k7', 'kv_k9',
                'kv2_1', 'kv2_2', 'kv2_3', 'kv2_4', 'kv2_5'
            ]
            expected = [
                ['name1', 1.0, 3.0, None, 10.0, None, None, 5.0, 1.0, 7.0, None, None],
                ['name1', None, None, None, None, None, None, None, None, 1.0, 2.0, None],
                ['name1', 7.1, None, None, None, 8.2, None, 1.0, None, None, None, 6.0],
                ['name2', None, 1.2, 1.5, None, None, None, None, None, None, None, None],
                ['name2', None, 1.0, None, None, None, 1.1, None, None, None, 2.0, None]
            ]

            self.assertListEqual([c.name for c in res.columns], expected_cols)
            self.assertEqual(result, expected)
        finally:
            table.drop()

    def testGroupbyAggregation(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        class Agg(object):
            def buffer(self):
                return [0]

            def __call__(self, buffer, val):
                buffer[0] += val

            def merge(self, buffer, pbuffer):
                buffer[0] += pbuffer[0]

            def getvalue(self, buffer):
                return buffer[0]

        expr = self.expr.groupby(['name', 'id'])[lambda x: x.fid.min() * 2 < 8] \
            .agg(self.expr.fid.max() + 1, new_id=self.expr.id.sum(),
                 new_id2=self.expr.id.agg(Agg))

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 3, 5.1, 6, 6],
            ['name2', 2, 4.5, 2, 2]
        ]

        result = sorted(result, key=lambda k: k[0])

        self.assertEqual(expected, result)

        field = self.expr.groupby('name').sort(['id', -self.expr.fid]).row_number()
        expr = self.expr['name', 'id', 'fid', field]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 3, 4.1, 1],
            ['name1', 3, 2.2, 2],
            ['name1', 4, 5.3, 3],
            ['name1', 4, 4.2, 4],
            ['name2', 2, 3.5, 1],
        ]

        result = sorted(result, key=lambda k: (k[0], k[1], -k[2]))

        self.assertEqual(expected, result)

        expr = self.expr.name.value_counts(dropna=True)[:25]

        expected = [
            ['name1', 4],
            ['name2', 1]
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(expected, result)

        expr = self.expr.name.topk(25)

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(expected, result)

        expr = self.expr.groupby('name').count()

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual([it[1:] for it in expected], result)

        expected = [
            ['name1', 2],
            ['name2', 1]
        ]

        expr = self.expr.groupby('name').id.nunique()

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual([it[1:] for it in expected], result)

        expr = self.expr[self.expr['id'] > 2].name.value_counts()[:25]

        expected = [
            ['name1', 4]
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(expected, result)

        expr = self.expr.groupby('name', Scalar(1).rename('constant')) \
            .agg(id=self.expr.id.sum())

        expected = [
            ['name1', 1, 14],
            ['name2', 1, 2]
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(expected, result)

        expr = self.expr[:1]
        expr = expr.groupby('name').agg(expr.id.sum())

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 4]
        ]

        self.assertEqual(expected, result)

    def testProjectionGroupbyFilter(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        df = self.expr.copy()
        df['id'] = df.id + 1
        df2 = df.groupby('name').agg(id=df.id.sum())[lambda x: x.name == 'name2']

        expected = [['name2', 3]]
        res = self.engine.execute(df2)
        result = self._get_result(res)
        self.assertEqual(expected, result)

    def testJoinGroupby(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        schema2 = Schema.from_lists(['name', 'id2', 'id3'],
                                    [types.string, types.bigint, types.bigint])

        table_name = tn('pyodps_test_engine_table2')
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name, schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

        self._gen_data(data=data)

        data2 = [
            ['name1', 4, -1],
            ['name2', 1, -2]
        ]

        self.odps.write_table(table2, 0, data2)

        expr = self.expr.join(expr2, on='name')[self.expr]
        expr = expr.groupby('id').agg(expr.fid.sum())

        res = self.engine.execute(expr)
        result = self._get_result(res)

        id_idx = [idx for idx, col in enumerate(self.expr.schema.names) if col == 'id'][0]
        fid_idx = [idx for idx, col in enumerate(self.expr.schema.names) if col == 'fid'][0]
        expected = [[k, sum(v[fid_idx] for v in row)]
                    for k, row in itertools.groupby(sorted(data, key=lambda r: r[id_idx]), lambda r: r[id_idx])]
        for it in zip(sorted(expected, key=lambda it: it[0]), sorted(result, key=lambda it: it[0])):
            self.assertAlmostEqual(it[0][0], it[1][0])
            self.assertAlmostEqual(it[0][1], it[1][1])

    def testFilterGroupby(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr.groupby(['name']).agg(id=self.expr.id.max())[lambda x: x.id > 3]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(len(result), 1)

        expected = [
            ['name1', 4]
        ]

        self.assertEqual(expected, result)

    def testWindowFunction(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 6.1, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr.groupby('name').id.cumsum()

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [[14]] * 4 + [[2]]
        self.assertEqual(sorted(expected), sorted(result))

        expr = self.expr.groupby('name').sort('fid').id.cummax()

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [[3], [4], [4], [4], [2]]
        self.assertEqual(sorted(expected), sorted(result))

        expr = self.expr[
            self.expr.groupby('name', 'id').sort('fid').id.cummean(),
            self.expr.groupby('name').id.cummedian()
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [3, 3.5], [3, 3.5], [4, 3.5], [4, 3.5], [2, 2]
        ]
        self.assertEqual(sorted(expected), sorted(result))

        expr = self.expr.groupby('name').mutate(id2=lambda x: x.id.cumcount(unique=True),
                                                fid2=lambda x: x.fid.cummin(sort='id'))

        res = self.engine.execute(expr['name', 'id2', 'fid2'])
        result = self._get_result(res)

        expected = [
            ['name1', 2, 2.2],
            ['name1', 2, 2.2],
            ['name1', 2, 2.2],
            ['name1', 2, 2.2],
            ['name2', 1, 3.5],
        ]
        self.assertEqual(sorted(expected), sorted(result))

        expr = self.expr[
            self.expr.id,
            self.expr.groupby('name').rank('id'),
            self.expr.groupby('name').dense_rank('fid', ascending=False),
            self.expr.groupby('name').row_number(sort=['id', 'fid'], ascending=[True, False]),
            self.expr.groupby('name').percent_rank('id'),
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [4, 3, 2, 3, float(2) / 3],
            [2, 1, 1, 1, 0.0],
            [4, 3, 3, 4, float(2) / 3],
            [3, 1, 4, 2, float(0) / 3],
            [3, 1, 1, 1, float(0) / 3]
        ]
        self.assertEqual(sorted(expected), sorted(result))

        expr = self.expr[
            self.expr.id,
            self.expr.groupby('name').id.lag(offset=3, default=0, sort=['id', 'fid']).rename('id2'),
            self.expr.groupby('name').id.lead(offset=1, default=-1,
                                              sort=['id', 'fid'], ascending=[False, False]).rename('id3'),
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [4, 3, 4],
            [2, 0, -1],
            [4, 0, 3],
            [3, 0, -1],
            [3, 0, 3]
        ]
        self.assertEqual(sorted(expected), sorted(result))

    def testWindowRewrite(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr[self.expr.id - self.expr.id.mean() < 10][
            [lambda x: x.id - x.id.max()]][[lambda x: x.id - x.id.min()]][lambda x: x.id - x.id.std() > 0]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        id_idx = [idx for idx, col in enumerate(self.expr.schema.names) if col == 'id'][0]
        expected = [r[id_idx] for r in data]
        maxv = max(expected)
        expected = [v - maxv for v in expected]
        minv = min(expected)
        expected = [v - minv for v in expected]

        meanv = sum(expected) * 1.0 / len(expected)
        meanv2 = sum([v ** 2 for v in expected]) * 1.0 / len(expected)
        std = math.sqrt(meanv2 - meanv ** 2)
        expected = [v for v in expected if v > std]

        self.assertEqual(expected, [it[0] for it in result])

    def testReduction(self):
        data = self._gen_data(rows=5, value_range=(-100, 100))

        def stats(col, func):
            col_idx = [idx for idx, cn in enumerate(self.expr.schema.names) if cn == col][0]
            return func([r[col_idx] for r in data])

        def median(vct):
            sorted_lst = sorted(vct)
            lst_len = len(vct)
            index = (lst_len - 1) // 2
            if lst_len % 2:
                return sorted_lst[index]
            else:
                return (sorted_lst[index] + sorted_lst[index + 1]) / 2.0

        def var(vct, ddof=0):
            meanv = mean(vct)
            meanv2 = mean([v ** 2 for v in vct])
            return (meanv2 - meanv ** 2) * len(vct) / (len(vct) - ddof)

        def moment(vct, order, central=False, absolute=False):
            abs_fun = abs if absolute else lambda x: x
            if central:
                m = mean(vct)
                return mean([abs_fun(v - m) ** order for v in vct])
            else:
                return mean([abs_fun(v) ** order for v in vct])

        def skew(vct):
            n = len(vct)
            return moment(vct, 3, central=True) / (std(vct, 1) ** 3) * (n ** 2) / (n - 1) / (n - 2)

        def kurtosis(vct):
            n = len(vct)
            m4 = moment(vct, 4, central=True)
            m2 = var(vct, 0)
            return 1.0 / (n - 2) / (n - 3) * ((n * n - 1.0) * m4 / m2 ** 2 - 3 * (n - 1) ** 2)

        mean = lambda v: sum(v) * 1.0 / len(v)
        std = lambda v, ddof=0: math.sqrt(var(v, ddof))
        nunique = lambda v: len(set(v))
        cat = lambda v: len([it for it in v if it is not None])

        class Agg(object):
            def buffer(self):
                return [0.0, 0]

            def __call__(self, buffer, val):
                buffer[0] += val
                # meaningless condition, just test if rewriting JUMP instructions works under Python 3
                if val > 1000:
                    buffer[1] += 2
                else:
                    buffer[1] += 1

            def merge(self, buffer, pbuffer):
                buffer[0] += pbuffer[0]
                buffer[1] += pbuffer[1]

            def getvalue(self, buffer):
                if buffer[1] == 0:
                    return 0.0
                return buffer[0] / buffer[1]

        methods_to_fields = [
            (partial(stats, 'id', mean), self.expr.id.mean()),
            (partial(len, data), self.expr.count()),
            (partial(stats, 'id', var), self.expr.id.var(ddof=0)),
            (partial(stats, 'id', lambda x: var(x, 1)), self.expr.id.var(ddof=1)),
            (partial(stats, 'id', std), self.expr.id.std(ddof=0)),
            (partial(stats, 'id', lambda x: moment(x, 3, central=True)), self.expr.id.moment(3, central=True)),
            (partial(stats, 'id', skew), self.expr.id.skew()),
            (partial(stats, 'id', kurtosis), self.expr.id.kurtosis()),
            (partial(stats, 'id', median), self.expr.id.median()),
            (partial(stats, 'id', sum), self.expr.id.sum()),
            (partial(stats, 'id', min), self.expr.id.min()),
            (partial(stats, 'id', max), self.expr.id.max()),
            (partial(stats, 'isMale', min), self.expr.isMale.min()),
            (partial(stats, 'name', max), self.expr.name.max()),
            (partial(stats, 'birth', max), self.expr.birth.max()),
            (partial(stats, 'isMale', sum), self.expr.isMale.sum()),
            (partial(stats, 'isMale', any), self.expr.isMale.any()),
            (partial(stats, 'isMale', all), self.expr.isMale.all()),
            (partial(stats, 'name', nunique), self.expr.name.nunique()),
            (partial(stats, 'name', cat), self.expr.name.cat(sep='|').map(lambda x: len(x.split('|')), rtype='int')),
            (partial(stats, 'id', mean), self.expr.id.agg(Agg, rtype='float')),
            (partial(stats, 'id', lambda x: len(x)), self.expr.id.count()),
        ]

        fields = [it[1].rename('f'+str(i)) for i, it in enumerate(methods_to_fields)]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        for i, it in enumerate(methods_to_fields):
            method = it[0]

            first = method()
            second = [it[i] for it in result][0]
            if isinstance(first, float):
                self.assertAlmostEqual(first, second)
            else:
                if first != second:
                    pass
                self.assertEqual(first, second)

        expr = self.expr['id', 'fid'].apply(Agg, types=['float'] * 2)

        expected = [[mean([l[1] for l in data])], [mean([l[2] for l in data])]]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        for first, second in zip(expected, result):
            first = first[0]
            second = second[0]

            if isinstance(first, float):
                self.assertAlmostEqual(first, second)
            else:
                self.assertEqual(first, second)

    def testUserDefinedAggregators(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        @output_types('float')
        class Aggregator(object):
            def buffer(self):
                return [0.0, 0]

            def __call__(self, buffer, val):
                buffer[0] += val
                buffer[1] += 1

            def merge(self, buffer, pbuffer):
                buffer[0] += pbuffer[0]
                buffer[1] += pbuffer[1]

            def getvalue(self, buffer):
                if buffer[1] == 0:
                    return 0.0
                return buffer[0] / buffer[1]

        expr = self.expr.id.agg(Aggregator)

        expected = float(16) / 5

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertAlmostEqual(expected, result)

        expr = self.expr.groupby(Scalar('const').rename('s')).id.agg(Aggregator)

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertAlmostEqual(expected, result[0][0])

        expr = self.expr.groupby('name').agg(self.expr.id.agg(Aggregator))

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', float(14) / 4],
            ['name2', 2]
        ]
        for expect_r, actual_r in zip(expected, result):
            self.assertEqual(expect_r[0], actual_r[0])
            self.assertAlmostEqual(expect_r[1], actual_r[1])

        expr = self.expr[
            (self.expr['name'] + ',' + self.expr['id'].astype('string')).rename('name'),
            self.expr.id
        ]
        expr = expr.groupby('name').agg(expr.id.agg(Aggregator).rename('id'))

        expected = [
            ['name1,4', 4],
            ['name1,3', 3],
            ['name2,2', 2],
        ]
        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(sorted(result), sorted(expected))

        expr = self.expr[self.expr.name, Scalar(1).rename('id')]
        expr = expr.groupby('name').agg(expr.id.sum())

        expected = [
            ['name1', 4],
            ['name2', 1]
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(expected, result)

        @output_types('float')
        class Aggregator(object):
            def buffer(self):
                return [0.0, 0]

            def __call__(self, buffer, val0, val1):
                buffer[0] += val0
                buffer[1] += val1

            def merge(self, buffer, pbuffer):
                buffer[0] += pbuffer[0]
                buffer[1] += pbuffer[1]

            def getvalue(self, buffer):
                if buffer[1] == 0:
                    return 0.0
                return buffer[0] / buffer[1]

        expr = agg([self.expr['fid'], self.expr['id']], Aggregator).rename('agg')

        expected = sum(r[2] for r in data) / sum(r[1] for r in data)
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertAlmostEqual(expected, result)

    def testMapReduceByApplyDistributeSort(self):
        data = [
            ['name key', 4, 5.3, None, None, None],
            ['name', 2, 3.5, None, None, None],
            ['key', 4, 4.2, None, None, None],
            ['name', 3, 2.2, None, None, None],
            ['key name', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        def mapper(row):
            for word in row[0].split():
                yield word, 1

        class reducer(object):
            def __init__(self):
                self._curr = None
                self._cnt = 0

            def __call__(self, row):
                if self._curr is None:
                    self._curr = row.word
                elif self._curr != row.word:
                    yield (self._curr, self._cnt)
                    self._curr = row.word
                    self._cnt = 0
                self._cnt += row.count

            def close(self):
                if self._curr is not None:
                    yield (self._curr, self._cnt)

        expr = self.expr['name', ].apply(
            mapper, axis=1, names=['word', 'count'], types=['string', 'int'])
        expr = expr.groupby('word').sort('word').apply(
            reducer, names=['word', 'count'], types=['string', 'int'])

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['key', 3], ['name', 4]]
        self.assertEqual(sorted(result), sorted(expected))

    def testMapReduce(self):
        data = [
            ['name key', 4, 5.3, None, None, None],
            ['name', 2, 3.5, None, None, None],
            ['key', 4, 4.2, None, None, None],
            ['name', 3, 2.2, None, None, None],
            ['key name', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        @output(['word', 'cnt'], ['string', 'int'])
        def mapper(row, mul=1):
            for word in row[0].split():
                yield word, mul

        @output(['word', 'cnt'], ['string', 'int'])
        def reducer(keys):
            cnt = [0, ]

            def h(row, done):
                cnt[0] += row[1]
                if done:
                    yield keys[0], cnt[0]

            return h

        expr = self.expr['name', ].map_reduce(functools.partial(mapper, mul=2), reducer, group='word')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['key', 6], ['name', 8]]
        self.assertEqual(sorted(result), sorted(expected))

        @output(['word', 'cnt'], ['string', 'int'])
        class reducer2(object):
            def __init__(self, keys):
                self.cnt = 0

            def __call__(self, row, done):
                self.cnt += row.cnt
                if done:
                    yield row.word, self.cnt

        expr = self.expr['name', ].map_reduce(mapper, reducer2, group='word')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['key', 3], ['name', 4]]
        self.assertEqual(sorted(result), sorted(expected))

        # test both combiner and reducer
        expr = self.expr['name',].map_reduce(mapper, reducer, combiner=reducer2, group='word')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(sorted(result), sorted(expected))

    def testReduceOnly(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        data = self._gen_data(data=data)

        df = self.expr[(self.expr.id < 10) & (self.expr.name.startswith('n'))]
        df = df['name', 'id']

        @output(['name', 'id'], ['string', 'int'])
        def reducer(keys):
            def h(row, done):
                yield row
            return h

        df2 = df.map_reduce(reducer=reducer, group='name')
        res = self.engine.execute(df2)
        result = self._get_result(res)
        self.assertEqual(sorted([r[:2] for r in data]), sorted(result))

    def testJoinMapReduce(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        schema2 = Schema.from_lists(['name2', 'id2', 'id3'],
                                    [types.string, types.bigint, types.bigint])

        table_name = tn('pyodps_test_engine_table2')
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name, schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

        self._gen_data(data=data)

        data2 = [
            ['name1', 4, -1],
        ]

        self.odps.write_table(table2, 0, data2)

        @output(['id'], ['int'])
        def reducer(keys):
            sums = [0]

            def h(row, done):
                sums[0] += row.id
                if done:
                    yield sums[0]

            return h

        expr = self.expr.join(expr2, on=('name', 'name2'))
        expr = expr.map_reduce(reducer=reducer, group='name')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 14)

    def testScaleMapReduce(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        scaled = self.expr.min_max_scale()

        @output(scaled.schema.names, scaled.schema.types)
        def h(row):
            yield row

        df = scaled.map_reduce(mapper=h)

        self.assertEqual(len(self.engine.execute(df)), 5)

    def testDistributeSort(self):
        data = [
            ['name', 4, 5.3, None, None, None],
            ['name', 2, 3.5, None, None, None],
            ['key', 4, 4.2, None, None, None],
            ['name', 3, 2.2, None, None, None],
            ['key', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        @output_names('name', 'id')
        @output_types('string', 'int')
        class reducer(object):
            def __init__(self):
                self._curr = None
                self._cnt = 0

            def __call__(self, row):
                if self._curr is None:
                    self._curr = row.name
                elif self._curr != row.name:
                    yield (self._curr, self._cnt)
                    self._curr = row.name
                    self._cnt = 0
                self._cnt += 1

            def close(self):
                if self._curr is not None:
                    yield (self._curr, self._cnt)

        expr = self.expr['name', ].groupby('name').sort('name').apply(reducer)

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['key', 2], ['name', 3]]
        self.assertEqual(sorted(expected), sorted(result))

    def testJoin(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        schema2 = Schema.from_lists(['name', 'id2', 'id3'],
                                    [types.string, types.bigint, types.bigint])
        table_name = tn('pyodps_test_engine_table2')
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name, schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

        self._gen_data(data=data)

        data2 = [
            ['name1', 4, -1],
            ['name2', 1, -2]
        ]

        self.odps.write_table(table2, 0, data2)

        try:
            expr = self.expr.join(expr2)['name', 'id2']

            res = self.engine.execute(expr)
            result = self._get_result(res)

            self.assertEqual(len(result), 5)
            expected = [
                [to_str('name1'), 4],
                [to_str('name2'), 1]
            ]
            self.assertTrue(all(it in expected for it in result))

            expr = self.expr.join(expr2, on=['name', ('id', 'id2')])[self.expr.name, expr2.id2]
            res = self.engine.execute(expr)
            result = self._get_result(res)
            self.assertEqual(len(result), 2)
            expected = [to_str('name1'), 4]
            self.assertTrue(all(it == expected for it in result))

            expr = self.expr.left_join(expr2, on=['name', ('id', 'id2')])[self.expr.name, expr2.id2]
            res = self.engine.execute(expr)
            result = self._get_result(res)
            expected = [
                ['name1', 4],
                ['name2', None],
                ['name1', 4],
                ['name1', None],
                ['name1', None]
            ]
            self.assertEqual(len(result), 5)
            self.assertTrue(all(it in expected for it in result))

            expr = self.expr.right_join(expr2, on=['name', ('id', 'id2')])[self.expr.name, expr2.id2]
            res = self.engine.execute(expr)
            result = self._get_result(res)
            expected = [
                ['name1', 4],
                ['name1', 4],
                [None, 1],
            ]
            self.assertEqual(len(result), 3)
            self.assertTrue(all(it in expected for it in result))

            expr = self.expr.outer_join(expr2, on=['name', ('id', 'id2')])[self.expr.name, expr2.id2]
            res = self.engine.execute(expr)
            result = self._get_result(res)
            expected = [
                ['name1', 4],
                ['name1', 4],
                ['name2', None],
                ['name1', None],
                ['name1', None],
                [None, 1],
            ]
            self.assertEqual(len(result), 6)
            self.assertTrue(all(it in expected for it in result))

            grouped = self.expr.groupby('name').agg(new_id=self.expr.id.sum()).cache()
            self.engine.execute(self.expr.join(grouped, on='name'))

            expr = self.expr.join(expr2, on=['name', ('id', 'id2')])[
                lambda x: x.groupby(Scalar(1)).sort('name').row_number(), ]
            self.engine.execute(expr)
        finally:
            table2.drop()

    def testUnion(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        schema2 = Schema.from_lists(['name', 'id2', 'id3'],
                                    [types.string, types.bigint, types.bigint])
        table_name = tn('pyodps_test_engine_table2')
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name, schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

        self._gen_data(data=data)

        data2 = [
            ['name3', 5, -1],
            ['name4', 6, -2]
        ]

        self.odps.write_table(table2, 0, data2)

        try:
            expr = self.expr['name', 'id'].distinct().union(expr2[expr2.id2.rename('id'), 'name'])

            res = self.engine.execute(expr)
            result = self._get_result(res)

            expected = [
                ['name1', 4],
                ['name1', 3],
                ['name2', 2],
                ['name3', 5],
                ['name4', 6]
            ]

            result = sorted(result)
            expected = sorted(expected)

            self.assertEqual(len(result), len(expected))
            for e, r in zip(result, expected):
                self.assertEqual([to_str(t) for t in e],
                                 [to_str(t) for t in r])

        finally:
            table2.drop()

    def testScaleValue(self):
        data = [
            ['name1', 4, 5.3],
            ['name2', 2, 3.5],
            ['name2', 3, 1.5],
            ['name1', 4, 4.2],
            ['name1', 3, 2.2],
            ['name1', 3, 4.1],
        ]
        schema = Schema.from_lists(['name', 'id', 'fid'],
                                   [types.string, types.bigint, types.double])
        table_name = tn('pyodps_test_engine_scale_table')
        self.odps.delete_table(table_name, if_exists=True)
        table = self.odps.create_table(name=table_name, schema=schema)
        self.odps.write_table(table_name, 0, data)
        expr_input = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))

        # test simple min_max_scale
        expr = expr_input.min_max_scale(columns=['fid'])

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 4, 1.0],
            ['name2', 2, 0.5263157894736842],
            ['name2', 3, 0.0],
            ['name1', 4, 0.7105263157894738],
            ['name1', 3, 0.18421052631578952],
            ['name1', 3, 0.6842105263157894]
        ]

        result = sorted(result)
        expected = sorted(expected)

        for first, second in zip(result, expected):
            self.assertEqual(len(first), len(second))
            for it1, it2 in zip(first, second):
                self.assertAlmostEqual(it1, it2)

        # test grouped min_max_scale
        expr = expr_input.min_max_scale(columns=['fid'], group=['name'])

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 4, 1.0],
            ['name1', 4, 0.6451612903225807],
            ['name1', 3, 0.0],
            ['name1', 3, 0.6129032258064515],
            ['name2', 2, 1.0],
            ['name2', 3, 0.0]
        ]

        result = sorted(result)
        expected = sorted(expected)

        for first, second in zip(result, expected):
            self.assertEqual(len(first), len(second))
            for it1, it2 in zip(first, second):
                self.assertAlmostEqual(it1, it2)

        # test simple std_scale
        expr = expr_input.std_scale(columns=['fid'])

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 4, 1.436467168552818],
            ['name2', 2, 0.026117584882778763],
            ['name2', 3, -1.5409375080839316],
            ['name1', 4, 0.5745868674211275],
            ['name1', 3, -0.9924682255455829],
            ['name1', 3, 0.4962341127727916]
        ]

        result = sorted(result)
        expected = sorted(expected)

        for first, second in zip(result, expected):
            self.assertEqual(len(first), len(second))
            for it1, it2 in zip(first, second):
                self.assertAlmostEqual(it1, it2)

        # test grouped std_scale
        expr = expr_input.std_scale(columns=['fid'], group=['name'])

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 4, 1.211115520843893],
            ['name1', 4, 0.22428065200812874],
            ['name1', 3, -1.569964564056898],
            ['name1', 3, 0.13456839120487693],
            ['name2', 2, 1.0],
            ['name2', 3, -1.0]
        ]

        result = sorted(result)
        expected = sorted(expected)

        for first, second in zip(result, expected):
            self.assertEqual(len(first), len(second))
            for it1, it2 in zip(first, second):
                self.assertAlmostEqual(it1, it2)

    def testHllc(self):
        names = [randint(0, 100000) for _ in xrange(100000)]
        data = [[n] + [None] * 5 for n in names]

        self._gen_data(data=data)

        expr = self.expr.name.hll_count()

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expect = len(set(names))
        self.assertAlmostEqual(expect, result, delta=result * 0.1)

    def testBloomFilter(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        data2 = [
            ['name1'],
            ['name3']
        ]

        self._gen_data(data=data)

        schema2 = Schema.from_lists(['name', ], [types.string])

        table_name = tn('pyodps_test_engine_bf_table2')
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name, schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

        self.odps.write_table(table2, 0, data2)

        try:
            expr = self.expr.bloom_filter('name', expr2.name, capacity=10)

            res = self.engine.execute(expr)
            result = self._get_result(res)

            self.assertTrue(all(r[0] != 'name2' for r in result))
        finally:
            table2.drop()

    def testPersist(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        table_name = tn('pyodps_test_engine_persist_table')
        self.odps.delete_table(table_name, if_exists=True)

        # simple persist
        try:
            df = self.engine.persist(self.expr, table_name)

            res = self.engine.execute(df)
            result = self._get_result(res)
            self.assertEqual(len(result), 5)
            self.assertEqual(data, result)
        finally:
            self.odps.delete_table(table_name, if_exists=True)

        # persist over existing table
        try:
            self.odps.create_table(table_name,
                                   'name string, fid double, id bigint, isMale boolean, scale decimal, birth datetime',
                                   lifecycle=1)
            df = self.engine.persist(self.expr, table_name)

            res = self.engine.execute(df)
            result = self._get_result(res)
            self.assertEqual(len(result), 5)
            self.assertEqual(data, [[r[0], r[2], r[1], None, None, None] for r in result])
        finally:
            self.odps.delete_table(table_name, if_exists=True)

        try:
            schema = Schema.from_lists(self.schema.names, self.schema.types, ['ds'], ['string'])
            self.odps.create_table(table_name, schema)
            df = self.engine.persist(self.expr, table_name, partition='ds=today', create_partition=True)

            res = self.engine.execute(df)
            result = self._get_result(res)
            self.assertEqual(len(result), 5)
            self.assertEqual(data, [d[:-1] for d in result])

            df2 = self.engine.persist(self.expr[self.expr.id.astype('float'), 'name'], table_name,
                                      partition='ds=today2', create_partition=True, cast=True)

            res = self.engine.execute(df2)
            result = self._get_result(res)
            self.assertEqual(len(result), 5)
            self.assertEqual([d[:2] + [None] * (len(d) - 2) for d in data], [d[:-1] for d in result])
        finally:
            self.odps.delete_table(table_name, if_exists=True)

        try:
            self.engine.persist(self.expr, table_name, partitions=['name'])

            t = self.odps.get_table(table_name)
            self.assertEqual(2, len(list(t.partitions)))
            with t.open_reader(partition='name=name1', reopen=True) as r:
                self.assertEqual(4, r.count)
            with t.open_reader(partition='name=name2', reopen=True) as r:
                self.assertEqual(1, r.count)
        finally:
            self.odps.delete_table(table_name, if_exists=True)

    def testMakeKV(self):
        from odps import types as odps_types
        data = [
            ['name1', 1.0, 3.0, None, 10.0, None, None],
            ['name1', None, 3.0, 5.1, None, None, None],
            ['name1', 7.1, None, None, None, 8.2, None],
            ['name2', None, 1.2, 1.5, None, None, None],
            ['name2', None, 1.0, None, None, None, 1.1],
        ]
        kv_cols = ['k1', 'k2', 'k3', 'k5', 'k7', 'k9']
        schema = Schema.from_lists(['name'] + kv_cols,
                                   [odps_types.string] + [odps_types.double] * 6)
        table_name = tn('pyodps_test_engine_make_kv')
        self.odps.delete_table(table_name, if_exists=True)
        table = self.odps.create_table(name=table_name, schema=schema)
        expr = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))
        try:
            self.odps.write_table(table, 0, data)
            expr1 = expr.to_kv(columns=kv_cols, kv_delim='=')

            res = self.engine.execute(expr1)
            result = self._get_result(res)

            expected = [
                ['name1', 'k1=1.0,k2=3.0,k5=10.0'],
                ['name1', 'k2=3.0,k3=5.1'],
                ['name1', 'k1=7.1,k7=8.2'],
                ['name2', 'k2=1.2,k3=1.5'],
                ['name2', 'k2=1.0,k9=1.1'],
            ]

            self.assertListEqual(result, expected)
        finally:
            table.drop()

    def testMelt(self):
        data = [
            ['name1', 1.0, 3.0, 10.0],
            ['name1', None, 3.0, 5.1],
            ['name1', 7.1, None, 8.2],
            ['name2', None, 1.2, 1.5],
            ['name2', None, 1.0, 1.1],
        ]
        schema = Schema.from_lists(['name', 'k1', 'k2', 'k3'],
                                   [types.string, types.double, types.double, types.double])
        table_name = tn('pyodps_test_engine_melt')
        self.odps.delete_table(table_name, if_exists=True)
        table = self.odps.create_table(name=table_name, schema=schema)
        expr = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))
        try:
            self.odps.write_table(table, 0, data)
            expr1 = expr.melt('name')

            res = self.engine.execute(expr1)
            result = self._get_result(res)

            expected = [
                ['name1', 'k1', 1.0], ['name1', 'k2', 3.0], ['name1', 'k3', 10.0],
                ['name1', 'k1', None], ['name1', 'k2', 3.0], ['name1', 'k3', 5.1],
                ['name1', 'k1', 7.1], ['name1', 'k2', None], ['name1', 'k3', 8.2],
                ['name2', 'k1', None], ['name2', 'k2', 1.2], ['name2', 'k3', 1.5],
                ['name2', 'k1', None], ['name2', 'k2', 1.0], ['name2', 'k3', 1.1]
            ]

            self.assertListEqual(result, expected)
        finally:
            table.drop()

    def testCollectionNa(self):
        from odps.compat import reduce

        data = [
            [0, 'name1', 1.0, None, 3.0, 4.0],
            [1, 'name1', 2.0, None, None, 1.0],
            [2, 'name1', 3.0, 4.0, 1.0, None],
            [3, 'name1', None, 1.0, 2.0, 3.0],
            [4, 'name1', 1.0, None, 3.0, 4.0],
            [5, 'name1', 1.0, 2.0, 3.0, 4.0],
            [6, 'name1', None, None, None, None],
        ]

        schema = Schema.from_lists(['rid', 'name', 'f1', 'f2', 'f3', 'f4'],
                                   [types.bigint, types.string] + [types.double] * 4)

        table_name = tn('pyodps_test_engine_collection_na')
        self.odps.delete_table(table_name, if_exists=True)
        table = self.odps.create_table(name=table_name, schema=schema)
        expr = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))

        try:
            self.odps.write_table(table, 0, data)

            exprs = [
                expr[Scalar(0).rename('gidx'), expr].fillna(100, subset=['f1', 'f2', 'f3', 'f4']),
                expr[Scalar(1).rename('gidx'), expr].fillna(expr.f3, subset=['f1', 'f2', 'f3', 'f4']),
                expr[Scalar(2).rename('gidx'), expr].fillna(method='ffill', subset=['f1', 'f2', 'f3', 'f4']),
                expr[Scalar(3).rename('gidx'), expr].fillna(method='bfill', subset=['f1', 'f2', 'f3', 'f4']),
                expr[Scalar(4).rename('gidx'), expr].dropna(thresh=3, subset=['f1', 'f2', 'f3', 'f4']),
                expr[Scalar(5).rename('gidx'), expr].dropna(how='any', subset=['f1', 'f2', 'f3', 'f4']),
                expr[Scalar(6).rename('gidx'), expr].dropna(how='all', subset=['f1', 'f2', 'f3', 'f4']),
            ]

            uexpr = reduce(lambda a, b: a.union(b), exprs).sort(['gidx', 'rid'])

            ures = self.engine.execute(uexpr)
            uresult = self._get_result(ures)

            expected = [
                [0, 0, 'name1', 1.0, 100.0, 3.0, 4.0],
                [0, 1, 'name1', 2.0, 100.0, 100.0, 1.0],
                [0, 2, 'name1', 3.0, 4.0, 1.0, 100.0],
                [0, 3, 'name1', 100.0, 1.0, 2.0, 3.0],
                [0, 4, 'name1', 1.0, 100.0, 3.0, 4.0],
                [0, 5, 'name1', 1.0, 2.0, 3.0, 4.0],
                [0, 6, 'name1', 100.0, 100.0, 100.0, 100.0],

                [1, 0, 'name1', 1.0, 3.0, 3.0, 4.0],
                [1, 1, 'name1', 2.0, None, None, 1.0],
                [1, 2, 'name1', 3.0, 4.0, 1.0, 1.0],
                [1, 3, 'name1', 2.0, 1.0, 2.0, 3.0],
                [1, 4, 'name1', 1.0, 3.0, 3.0, 4.0],
                [1, 5, 'name1', 1.0, 2.0, 3.0, 4.0],
                [1, 6, 'name1', None, None, None, None],

                [2, 0, 'name1', 1.0, 1.0, 3.0, 4.0],
                [2, 1, 'name1', 2.0, 2.0, 2.0, 1.0],
                [2, 2, 'name1', 3.0, 4.0, 1.0, 1.0],
                [2, 3, 'name1', None, 1.0, 2.0, 3.0],
                [2, 4, 'name1', 1.0, 1.0, 3.0, 4.0],
                [2, 5, 'name1', 1.0, 2.0, 3.0, 4.0],
                [2, 6, 'name1', None, None, None, None],

                [3, 0, 'name1', 1.0, 3.0, 3.0, 4.0],
                [3, 1, 'name1', 2.0, 1.0, 1.0, 1.0],
                [3, 2, 'name1', 3.0, 4.0, 1.0, None],
                [3, 3, 'name1', 1.0, 1.0, 2.0, 3.0],
                [3, 4, 'name1', 1.0, 3.0, 3.0, 4.0],
                [3, 5, 'name1', 1.0, 2.0, 3.0, 4.0],
                [3, 6, 'name1', None, None, None, None],

                [4, 0, 'name1', 1.0, None, 3.0, 4.0],
                [4, 2, 'name1', 3.0, 4.0, 1.0, None],
                [4, 3, 'name1', None, 1.0, 2.0, 3.0],
                [4, 4, 'name1', 1.0, None, 3.0, 4.0],
                [4, 5, 'name1', 1.0, 2.0, 3.0, 4.0],

                [5, 5, 'name1', 1.0, 2.0, 3.0, 4.0],

                [6, 0, 'name1', 1.0, None, 3.0, 4.0],
                [6, 1, 'name1', 2.0, None, None, 1.0],
                [6, 2, 'name1', 3.0, 4.0, 1.0, None],
                [6, 3, 'name1', None, 1.0, 2.0, 3.0],
                [6, 4, 'name1', 1.0, None, 3.0, 4.0],
                [6, 5, 'name1', 1.0, 2.0, 3.0, 4.0],
            ]
            self.assertListEqual(uresult, expected)
        finally:
            table.drop()

    def testDrop(self):
        data1 = [
            ['name1', 1, 3.0], ['name1', 2, 3.0], ['name1', 2, 2.5],
            ['name2', 1, 1.2], ['name2', 3, 1.0],
            ['name3', 1, 1.2], ['name3', 3, 1.2],
        ]
        schema1 = Schema.from_lists(['name', 'id', 'fid'],
                                    [types.string, types.bigint, types.double])
        table_name1 = tn('pyodps_test_engine_drop1')
        self.odps.delete_table(table_name1, if_exists=True)
        table1 = self.odps.create_table(name=table_name1, schema=schema1)
        expr1 = CollectionExpr(_source_data=table1, _schema=odps_schema_to_df_schema(schema1))

        data2 = [
            ['name1', 1], ['name1', 2],
            ['name2', 1], ['name2', 2],
        ]
        schema2 = Schema.from_lists(['name', 'id'],
                                    [types.string, types.bigint])
        table_name2 = tn('pyodps_test_engine_drop2')
        self.odps.delete_table(table_name2, if_exists=True)
        table2 = self.odps.create_table(name=table_name2, schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))
        try:
            self.odps.write_table(table1, 0, data1)
            self.odps.write_table(table2, 0, data2)

            expr_result = expr1.drop(expr2)
            res = self.engine.execute(expr_result)
            result = self._get_result(res)

            expected = [['name2', 3, 1.0], ['name3', 1, 1.2], ['name3', 3, 1.2]]
            self.assertListEqual(sorted(result), sorted(expected))

            expr_result = expr1.drop(expr2, columns='name')
            res = self.engine.execute(expr_result)
            result = self._get_result(res)

            expected = [['name3', 1, 1.2], ['name3', 3, 1.2]]
            self.assertListEqual(sorted(result), sorted(expected))

            expr_result = expr1.drop(['id'], axis=1)
            res = self.engine.execute(expr_result)
            result = self._get_result(res)

            expected = [
                ['name1', 3.0], ['name1', 3.0], ['name1', 2.5],
                ['name2', 1.2], ['name2', 1.0],
                ['name3', 1.2], ['name3', 1.2],
            ]
            self.assertListEqual(sorted(result), sorted(expected))

            expr_result = expr1.drop(expr2[['id']], axis=1)
            res = self.engine.execute(expr_result)
            result = self._get_result(res)

            expected = [
                ['name1', 3.0], ['name1', 3.0], ['name1', 2.5],
                ['name2', 1.2], ['name2', 1.0],
                ['name3', 1.2], ['name3', 1.2],
            ]
            self.assertListEqual(sorted(result), sorted(expected))
        finally:
            table1.drop()
            table2.drop()

    def testExceptIntersect(self):
        data1 = [
            ['name1', 1], ['name1', 2], ['name1', 2], ['name1', 2],
            ['name2', 1], ['name2', 3],
            ['name3', 1], ['name3', 3],
        ]
        schema1 = Schema.from_lists(['name', 'id'], [types.string, types.bigint])
        table_name1 = tn('pyodps_test_engine_drop1')
        self.odps.delete_table(table_name1, if_exists=True)
        table1 = self.odps.create_table(name=table_name1, schema=schema1)
        expr1 = CollectionExpr(_source_data=table1, _schema=odps_schema_to_df_schema(schema1))

        data2 = [
            ['name1', 1], ['name1', 2], ['name1', 2],
            ['name2', 1], ['name2', 2],
        ]
        schema2 = Schema.from_lists(['name', 'id'], [types.string, types.bigint])
        table_name2 = tn('pyodps_test_engine_drop2')
        self.odps.delete_table(table_name2, if_exists=True)
        table2 = self.odps.create_table(name=table_name2, schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))
        try:
            self.odps.write_table(table1, 0, data1)
            self.odps.write_table(table2, 0, data2)

            expr_result = expr1.setdiff(expr2)
            res = self.engine.execute(expr_result)
            result = self._get_result(res)

            expected = [['name1', 2], ['name2', 3], ['name3', 1], ['name3', 3]]
            self.assertListEqual(sorted(result), sorted(expected))

            expr_result = expr1.setdiff(expr2, distinct=True)
            res = self.engine.execute(expr_result)
            result = self._get_result(res)

            expected = [['name2', 3], ['name3', 1], ['name3', 3]]
            self.assertListEqual(sorted(result), sorted(expected))

            expr_result = expr1.intersect(expr2)
            res = self.engine.execute(expr_result)
            result = self._get_result(res)

            expected = [['name1', 1], ['name1', 2], ['name1', 2], ['name2', 1]]
            self.assertListEqual(sorted(result), sorted(expected))

            expr_result = expr1.intersect(expr2, distinct=True)
            res = self.engine.execute(expr_result)
            result = self._get_result(res)

            expected = [['name1', 1], ['name1', 2], ['name2', 1]]
            self.assertListEqual(sorted(result), sorted(expected))
        finally:
            table1.drop()
            table2.drop()

    def testFilterOrder(self):
        table_name = tn('pyodps_test_division_error')
        self.odps.delete_table(table_name, if_exists=True)
        table = self.odps.create_table(table_name, 'divided bigint, divisor bigint', lifecycle=1)

        try:
            self.odps.write_table(table_name, [[2, 0], [1, 1], [1, 2], [5, 1], [5, 0]])
            df = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(table.schema))
            fdf = df[df.divisor > 0]
            ddf = fdf[(fdf.divided / fdf.divisor).rename('result'),]
            expr = ddf[ddf.result > 1]

            res = self.engine.execute(expr)
            result = self._get_result(res)
            self.assertEqual(result, [[5, ]])
        finally:
            table.drop()

    def teardown(self):
        self.table.drop()


if __name__ == '__main__':
    unittest.main()