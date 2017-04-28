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

from datetime import datetime
import math
import time
import itertools
from functools import partial
import uuid

from odps.df.backends.tests.core import TestBase, to_str, tn
from odps.models import Schema
from odps import types
from odps.compat import six
from odps.df.types import validate_data_type, DynamicSchema
from odps.df.backends.odpssql.types import df_schema_to_odps_schema, \
    odps_schema_to_df_schema
from odps.df.expr.expressions import CollectionExpr
from odps.df.backends.seahawks.engine import SeahawksEngine
from odps.df.backends.seahawks.models import SeahawksTable
from odps.df.backends.context import context
from odps.df import Scalar
from odps.tests.core import sqlalchemy_case


@sqlalchemy_case
class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'birth', 'scale'][:5],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'datetime', 'decimal')[:5])
        self.schema = df_schema_to_odps_schema(schema)
        table_name = tn('pyodps_test_%s' % str(uuid.uuid4()).replace('-', '_'))
        self.odps.delete_table(table_name, if_exists=True)
        self.table = self.odps.create_table(
                name=table_name, schema=self.schema)
        self.expr = CollectionExpr(_source_data=self.table, _schema=schema)

        self.engine = SeahawksEngine(self.odps)

        class FakeBar(object):
            def update(self, *args, **kwargs):
                pass

            def inc(self, *args, **kwargs):
                pass

            def status(self, *args, **kwargs):
                pass
        self.faked_bar = FakeBar()

    def teardown(self):
        self.table.drop()

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

    def testAsync(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr.id.sum()

        future = self.engine.execute(expr, async=True)
        self.assertFalse(future.done())
        res = future.result()

        self.assertEqual(sum(it[1] for it in data), res)

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
        self.assertIsInstance(table, SeahawksTable)

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
        expr = self.expr[self.expr.id < 10]
        expr1 = expr.id.sum()
        expr2 = expr.id.mean()

        fs = self.engine.execute([expr, expr1, expr2], n_parallel=2, async=True, timeout=1)
        self.assertEqual(len(fs), 3)

        self.assertEqual(fs[1].result(), expect1)
        self.assertAlmostEqual(fs[2].result(), expect2)
        self.assertTrue(context.is_cached(expr))

    def testBase(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr[self.expr.id < 10]['name', lambda x: x.id]
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
        self.assertEqual([it[1] + 1 for it in data], [it[2] for it in result])

        expr = self.expr.sort('id')[:5]
        res = self.engine.execute(expr)
        result = self._get_result(res.values)
        self.assertListAlmostEqual(sorted(data, key=lambda it: it[1])[:5],
                                   [r[:-1] + [r[-1].replace(tzinfo=None)] for r in result],
                                   only_float=False, delta=.001)

        expr = self.expr[:1].filter(lambda x: x.name == data[1][0])
        res = self.engine.execute(expr)
        self.assertEqual(len(res), 0)

    def testChinese(self):
        data = [
            ['中文', 4, 5.3, None, None],
            ['\'中文2', 2, 3.5, None, None],
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
            self.expr.id.between(self.expr.fid, 3, inclusive=False).rename('id7'),
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

        self.assertEqual([(it[2] < it[1] < 3) for it in data],
                         [it[11] for it in result])

    def testArithmetic(self):
        data = self._gen_data(5, value_range=(-1000, 1000))

        fields = [
            (self.expr.id + 1).rename('id1'),
            (self.expr.fid - 1).rename('fid1'),
            (self.expr.id / 2).rename('id2'),
            (self.expr.id ** 2).rename('id3'),
            abs(self.expr.id).rename('id4'),
            (~self.expr.id).rename('id5'),
            (-self.expr.fid).rename('fid2'),
            (~self.expr.isMale).rename('isMale1'),
            (-self.expr.isMale).rename('isMale2'),
            (self.expr.id // 2).rename('id6'),
            (self.expr.id % 2).rename('id7'),
        ]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(len(data), len(result))

        self.assertEqual([it[1] + 1 for it in data],
                         [it[0] for it in result])

        self.assertListAlmostEqual([it[2] - 1 for it in data],
                                   [it[1] for it in result], delta=.001)


        self.assertListAlmostEqual([float(it[1]) / 2 for it in data],
                                   [it[2] for it in result], delta=.001)

        self.assertEqual([int(it[1] ** 2) for it in data],
                         [it[3] for it in result])

        self.assertEqual([abs(it[1]) for it in data],
                         [it[4] for it in result])

        self.assertEqual([~it[1] for it in data],
                         [it[5] for it in result])

        self.assertListAlmostEqual([-it[2] for it in data],
                                   [it[6] for it in result], delta=.001)

        self.assertEqual([not it[3] for it in data],
                         [it[7] for it in result])

        self.assertEqual([it[1] // 2 for it in data],
                         [it[9] for it in result])

        self.assertEqual([it[1] % 2 for it in data],
                         [it[10] for it in result])

    def testMath(self):
        # TODO: test sinh, cosh..., and acosh, asinh...
        data = self._gen_data(5, value_range=(1, 90))

        if hasattr(math, 'expm1'):
            expm1 = math.expm1
        else:
            expm1 = lambda x: 2 * math.exp(x / 2.0) * math.sinh(x / 2.0)

        methods_to_fields = [
            (math.sin, self.expr.id.sin()),
            (math.cos, self.expr.id.cos()),
            (math.tan, self.expr.id.tan()),
            (math.log, self.expr.id.log()),
            (lambda v: math.log(v, 2), self.expr.id.log2()),
            (math.log10, self.expr.id.log10()),
            (math.log1p, self.expr.id.log1p()),
            (math.exp, self.expr.id.exp()),
            (expm1, self.expr.id.expm1()),
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
                    self.assertAlmostEqual(it1 * scale, it2 * scale, delta=8)
                else:
                    self.assertAlmostEqual(it1, it2, delta=2)

    def testString(self):
        data = self._gen_data(5)

        methods_to_fields = [
            (lambda s: s.capitalize(), self.expr.name.capitalize()),
            (lambda s: data[0][0] in s, self.expr.name.contains(data[0][0], regex=False)),
            (lambda s: s[0] + '|' + str(s[1]), self.expr.name.cat(self.expr.id.astype('string'), sep='|')),
            (lambda s: s.endswith(data[0][0]), self.expr.name.endswith(data[0][0])),
            (lambda s: s.startswith(data[0][0]), self.expr.name.startswith(data[0][0])),
            (lambda s: s.replace(data[0][0], 'test'), self.expr.name.replace(data[0][0], 'test', regex=False)),
            (lambda s: s[0], self.expr.name.get(0)),
            (lambda s: len(s), self.expr.name.len()),
            (lambda s: s.ljust(10), self.expr.name.ljust(10)),
            (lambda s: s.ljust(20, '*'), self.expr.name.ljust(20, fillchar='*')),
            (lambda s: s.rjust(10), self.expr.name.rjust(10)),
            (lambda s: s.rjust(20, '*'), self.expr.name.rjust(20, fillchar='*')),
            (lambda s: s * 4, self.expr.name.repeat(4)),
            (lambda s: s[1:], self.expr.name.slice(1)),
            (lambda s: s[1: 6], self.expr.name.slice(1, 6)),
            (lambda s: s.title(), self.expr.name.title()),
            (lambda s: s.rjust(20, '0'), self.expr.name.zfill(20)),
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
                        v = v.to_datetime()
                    if isinstance(v, datetime):
                        return v.replace(tzinfo=None)
                    return v
            except ImportError:
                conv = lambda v: v

            second = [conv(it[i]) for it in result]
            self.assertEqual(first, second)

    def testSortDistinct(self):
        data = [
            ['name1', 4, None, None, None],
            ['name2', 2, None, None, None],
            ['name1', 4, None, None, None],
            ['name1', 3, None, None, None],
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

    def testPivotTable(self):
        data = [
            ['name1', 1, 1.0, True, None],
            ['name1', 1, 5.0, True, None],
            ['name1', 2, 2.0, True, None],
            ['name2', 1, 3.0, False, None],
            ['name2', 3, 4.0, False, None]
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
        self.assertListAlmostEqual(sorted(result), sorted(expected), only_float=False)

        expr2 = expr.pivot_table(rows='name', values='fid', aggfunc=['mean', 'sum'])
        res = self.engine.execute(expr2)
        result = self._get_result(res)

        expected = [
            ['name1', 8.0 / 3, 8.0],
            ['name2', 3.5, 7.0],
        ]
        self.assertEqual(res.schema.names, ['name', 'fid_mean', 'fid_sum'])
        self.assertListAlmostEqual(sorted(result), sorted(expected), only_float=False)

        expr5 = expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum'])
        expr6 = expr5['name1_fid_mean',
                      expr5.groupby(Scalar(1)).sort('name1_fid_mean').name1_fid_mean.astype('float').cumsum()]

        k = lambda x: list(0 if it is None else it for it in x)

        expected = [
            [2, 2], [3, 5], [None, 5]
        ]
        res = self.engine.execute(expr6)
        result = self._get_result(res)
        self.assertEqual(sorted(result, key=k), sorted(expected, key=k))

        expr3 = expr.pivot_table(rows='id', values='fid', columns='name', fill_value=0).distinct()
        res = self.engine.execute(expr3)
        result = self._get_result(res)

        expected = [
            [2, 0, 2.0],
            [3, 4.0, 0],
            [1, 3.0, 3.0],
        ]

        self.assertEqual(res.schema.names, ['id', 'name2_fid_mean', 'name1_fid_mean'])
        self.assertEqual(result, expected)

        expr7 = expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum']).cache()
        self.assertEqual(len(self.engine.execute(expr7)), 3)

        expr8 = self.expr.pivot_table(rows='id', values='fid', columns='name')
        self.assertEqual(len(self.engine.execute(expr8)), 3)
        self.assertNotIsInstance(expr8.schema, DynamicSchema)
        expr9 =(expr8['name1_fid_mean'] - expr8['name2_fid_mean']).rename('substract')
        self.assertEqual(len(self.engine.execute(expr9)), 3)
        expr10 = expr8.distinct()
        self.assertEqual(len(self.engine.execute(expr10)), 3)

    def testGroupbyAggregation(self):
        data = [
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 4.1, None, None],
        ]
        self._gen_data(data=data)

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

        self.assertEqual(sorted([it[1:] for it in expected]), sorted(result))

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
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 4.1, None, None],
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
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 4.1, None, None],
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
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 4.1, None, None],
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
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 6.1, None, None],
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
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [3], [3], [4], [4], [2]
        ]
        self.assertEqual(sorted(expected), sorted(result))

        expr = self.expr.groupby('name').mutate(id2=lambda x: x.id.cumcount(),
                                                fid2=lambda x: x.fid.cummin(sort='id'))

        res = self.engine.execute(expr['name', 'id2', 'fid2'])
        result = self._get_result(res)

        expected = [
            ['name1', 4, 2.2],
            ['name1', 4, 2.2],
            ['name1', 4, 2.2],
            ['name1', 4, 2.2],
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
        [self.assertListAlmostEqual(l, r) for l, r in zip(sorted(expected), sorted(result))]

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
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 4.1, None, None],
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

        methods_to_fields = [
            (partial(stats, 'id', mean), self.expr.id.mean()),
            (partial(len, data), self.expr.count()),
            (partial(stats, 'id', var), self.expr.id.var(ddof=0)),
            (partial(stats, 'id', lambda x: var(x, 1)), self.expr.id.var(ddof=1)),
            (partial(stats, 'id', std), self.expr.id.std(ddof=0)),
            (partial(stats, 'id', lambda x: moment(x, 3, central=True)), self.expr.id.moment(3, central=True)),
            (partial(stats, 'id', skew), self.expr.id.skew()),
            (partial(stats, 'id', kurtosis), self.expr.id.kurtosis()),
            (partial(stats, 'id', sum), self.expr.id.sum()),
            (partial(stats, 'id', min), self.expr.id.min()),
            (partial(stats, 'id', max), self.expr.id.max()),
            (partial(stats, 'isMale', min), self.expr.isMale.min()),
            (partial(stats, 'isMale', sum), self.expr.isMale.sum()),
            (partial(stats, 'isMale', any), self.expr.isMale.any()),
            (partial(stats, 'isMale', all), self.expr.isMale.all()),
            (partial(stats, 'name', nunique), self.expr.name.nunique()),
            (partial(stats, 'name', cat), self.expr.name.cat(sep='|')),
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
            if i == len(methods_to_fields) - 2:  # cat
                second = len(second.split('|'))
            if isinstance(first, float):
                self.assertAlmostEqual(first, second)
            else:
                if first != second:
                    pass
                self.assertEqual(first, second)

    def testJoin(self):
        data = [
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 4.1, None, None],
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
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 4.1, None, None],
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

        expr = expr_input.min_max_scale(columns=['fid'])

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 4, 1.0],
            ['name2', 2, 0.41935483870967744],
            ['name1', 4, 0.6451612903225807],
            ['name1', 3, 0.0],
            ['name1', 3, 0.6129032258064515]
        ]

        result = sorted(result)
        expected = sorted(expected)

        for first, second in zip(result, expected):
            self.assertEqual(len(first), len(second))
            for it1, it2 in zip(first, second):
                self.assertAlmostEqual(it1, it2)

        expr = expr_input.std_scale(columns=['fid'])

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 4, 1.4213602653434203],
            ['name2', 2, -0.3553400663358544],
            ['name1', 4, 0.3355989515394193],
            ['name1', 3, -1.6385125281042194],
            ['name1', 3, 0.23689337755723686]
        ]

        result = sorted(result)
        expected = sorted(expected)

        for first, second in zip(result, expected):
            self.assertEqual(len(first), len(second))
            for it1, it2 in zip(first, second):
                self.assertAlmostEqual(it1, it2)

    def testPersist(self):
        data = [
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 4.1, None, None],
        ]
        self._gen_data(data=data)

        table_name = tn('pyodps_test_engine_persist_seahawks_table')

        try:
            df = self.engine.persist(self.expr, table_name)

            res = self.engine.execute(df)
            result = self._get_result(res)
            self.assertEqual(len(result), 5)
            self.assertEqual(data, result)
        finally:
            self.odps.delete_table(table_name, if_exists=True)

        try:
            schema = Schema.from_lists(self.schema.names, self.schema.types, ['ds'], ['string'])
            self.odps.create_table(table_name, schema)
            df = self.engine.persist(self.expr, table_name, partition='ds=today', create_partition=True)

            res = self.engine.execute(df)
            result = self._get_result(res)
            self.assertEqual(len(result), 5)
            self.assertEqual(data, [r[:-1] for r in result])
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
                ['name1', 'k1=1,k2=3,k5=10'],
                ['name1', 'k2=3,k3=5.1'],
                ['name1', 'k1=7.1,k7=8.2'],
                ['name2', 'k2=1.2,k3=1.5'],
                ['name2', 'k2=1,k9=1.1'],
            ]

            self.assertListEqual(result, expected)
        finally:
            table.drop()

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

    def testAXFException(self):
        import sqlalchemy

        data = [
            ['name1', 4, 5.3, None, None],
            ['name2', 2, 3.5, None, None],
            ['name1', 4, 4.2, None, None],
            ['name1', 3, 2.2, None, None],
            ['name1', 3, 4.1, None, None],
        ]
        self._gen_data(data=data)

        table_name = tn('pyodps_test_engine_axf_seahawks_table')

        try:
            schema = Schema.from_lists(self.schema.names, self.schema.types, ['ds'], ['string'])
            self.odps.create_table(table_name, schema)
            df = self.engine.persist(self.expr, table_name, partition='ds=today', create_partition=True)

            with self.assertRaises(sqlalchemy.exc.DatabaseError):
                self.engine.execute(df.input)
        finally:
            self.odps.delete_table(table_name, if_exists=True)