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

from datetime import datetime, timedelta
import math
import itertools
from functools import partial
import time

from odps.tests.core import ci_skip_case
from odps.compat import unittest, six, futures
from odps.models import Schema
from odps.df.backends.tests.core import TestBase, to_str, tn
from odps.df.backends.context import context
from odps.df.types import validate_data_type
from odps.df.backends.odpssql.types import df_schema_to_odps_schema
from odps.df.expr.expressions import CollectionExpr
from odps.df.backends.sqlalchemy.engine import SQLAlchemyEngine
from odps.df.backends.sqlalchemy.types import df_schema_to_sqlalchemy_columns
from odps.df.backends.sqlalchemy.engine import _engine_to_connections
from odps.df import Scalar, output_names, output_types, output, day, millisecond, agg


@ci_skip_case
class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        self.df_schema = schema
        self.schema = df_schema_to_odps_schema(schema)
        self.df = None
        self.expr = None

        self.engine = SQLAlchemyEngine()

        import sqlalchemy
        from sqlalchemy import create_engine

        self.sql_engine = engine = create_engine('postgres://localhost/pyodps')
        # self.sql_engine = engine = create_engine('mysql://localhost/pyodps')
        # self.sql_engine = engine = create_engine('sqlite://')
        self.conn = engine.connect()

        self.metadata = metadata = sqlalchemy.MetaData(bind=engine)
        columns = df_schema_to_sqlalchemy_columns(self.df_schema, engine=self.sql_engine)
        t = sqlalchemy.Table('pyodps_test_data', metadata, *columns)

        metadata.create_all()

        self.table = t
        self.expr = CollectionExpr(_source_data=self.table, _schema=self.df_schema)

        class FakeBar(object):
            def update(self, *args, **kwargs):
                pass
        self.faked_bar = FakeBar()

    def teardown(self):
        [conn.close() for conn in _engine_to_connections.values()]
        self.table.drop()
        self.conn.close()

    def _create_table_and_insert_data(self, table_name, df_schema, data, drop_first=True):
        import sqlalchemy

        columns = df_schema_to_sqlalchemy_columns(df_schema, engine=self.sql_engine)
        t = sqlalchemy.Table(table_name, self.metadata, *columns)

        self.conn.execute('DROP TABLE IF EXISTS %s' % table_name)
        t.create()

        self.conn.execute(t.insert(), [
            dict((n, v) for n, v in zip(df_schema.names, d)) for d in data
        ])

        return t

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

        self.conn.execute(self.table.insert(), [
            dict((n, v) for n, v in zip(self.schema.names, d)) for d in data])
        return data

    def testAsync(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr.id.sum()

        future = self.engine.execute(expr, async=True, priority=4)
        self.assertFalse(future.done())
        res = future.result()

        self.assertEqual(sum(it[1] for it in data), res)

    def testCache(self):
        import sqlalchemy

        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr[self.expr.id < 10].cache()
        cnt = expr.count()

        dag = self.engine.compile(expr)
        self.assertEqual(len(dag.nodes()), 2)

        res = self.engine.execute(cnt)
        self.assertEqual(len([it for it in data if it[1] < 10]), res)
        self.assertTrue(context.is_cached(expr))

        table = context.get_cached(expr)
        self.assertIsInstance(table, sqlalchemy.Table)

    def testBatch(self):
        if self.sql_engine.name == 'mysql':
            # TODO: mysqldb is not thread-safe, skip first
            return

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
        self.assertAlmostEqual(res[1], expect2, delta=0.001)
        self.assertTrue(context.is_cached(expr))

        # test async and timeout
        expr = self.expr[self.expr.id < 10]
        expr1 = expr.id.sum()
        expr2 = expr.id.mean()

        fs = self.engine.execute([expr, expr1, expr2], n_parallel=2, async=True, timeout=1)
        self.assertEqual(len(fs), 3)

        self.assertEqual(fs[1].result(), expect1)
        self.assertAlmostEqual(fs[2].result(), expect2, delta=0.001)
        self.assertTrue(context.is_cached(expr))

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
                             include_under=True, include_over=True).rename('id6')
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

        self.assertListAlmostEqual([it[2] - 1 for it in data],
                                   [it[1] for it in result], delta=.001)

        self.assertEqual([it[4] * 2 for it in data],
                         [it[2] for it in result])

        self.assertEqual([it[4] + it[1] for it in data],
                         [it[3] for it in result])

        self.assertListAlmostEqual([float(it[1]) / 2 for it in data],
                                   [it[4] for it in result], delta=.001)

        self.assertEqual([int(it[1] ** 2) for it in data],
                         [it[5] for it in result])

        self.assertEqual([abs(it[1]) for it in data],
                         [it[6] for it in result])

        self.assertEqual([~it[1] for it in data],
                         [it[7] for it in result])

        self.assertListAlmostEqual([-it[2] for it in data],
                                   [it[8] for it in result], delta=.001)

        self.assertEqual([not it[3] for it in data],
                         [it[9] for it in result])

        self.assertEqual([it[1] // 2 for it in data],
                         [it[11] for it in result])

        self.assertEqual([it[5] + timedelta(days=1) for it in data],
                         [it[12].replace(tzinfo=None) for it in result])

        self.assertEqual([10] * len(data), [it[13] for it in result])

        self.assertEqual([it[1] % 2 for it in data],
                         [it[14] for it in result])

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
                    self.assertAlmostEqual(it1 * scale, it2 * scale, delta=5)
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

        if self.sql_engine.name == 'mysql':
            methods_to_fields = methods_to_fields[:-2] + methods_to_fields[-1:]

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

    def testGroupbyAggregation(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr.groupby(['name', 'id'])[lambda x: x.fid.min() * 2 < 8] \
            .agg(self.expr.fid.max() + 1, new_id=self.expr.id.sum())

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 3, 5.1, 6],
            ['name2', 2, 4.5, 2]
        ]

        result = sorted(result, key=lambda k: k[0])

        self.assertListAlmostEqual(expected, result, only_float=False, delta=.001)

        expr = self.expr.name.value_counts()[:25]

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

        expr = self.expr.groupby('id').name.cat(sep=',')
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['name2'], ['name1,name1'], ['name1,name1']]
        self.assertEqual(sorted(result), sorted(expected))

    def testJoinGroupby(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        data2 = [
            ['name1', 4, -1],
            ['name2', 1, -2]
        ]

        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema2 = Schema.from_lists(['name', 'id2', 'id3'],
                                    datatypes('string', 'int64', 'int64'))
        table_name = tn('pyodps_test_engine_table2')
        table2 = self._create_table_and_insert_data(table_name, schema2, data2)
        expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

        self._gen_data(data=data)

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
        if self.sql_engine.name == 'mysql':
            # mysql doesn't support window function
            return

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

    def testJoin(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        data2 = [
            ['name1', 4, -1],
            ['name2', 1, -2]
        ]

        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema2 = Schema.from_lists(['name', 'id2', 'id3'],
                                    datatypes('string', 'int64', 'int64'))
        table_name = tn('pyodps_test_engine_table2')
        table2 = self._create_table_and_insert_data(table_name, schema2, data2)
        expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

        self._gen_data(data=data)

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

            if self.sql_engine.name != 'mysql':
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

            if self.sql_engine.name != 'mysql':
                expr = self.expr.join(expr2, on=['name', ('id', 'id2')])[
                    lambda x: x.groupby(Scalar(1)).sort('name').row_number(), ]
                self.engine.execute(expr)
        finally:
            [conn.close() for conn in _engine_to_connections.values()]
            table2.drop()

    def testUnion(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        data2 = [
            ['name3', 5, -1],
            ['name4', 6, -2]
        ]

        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema2 = Schema.from_lists(['name', 'id2', 'id3'],
                                    datatypes('string', 'int64', 'int64'))
        table_name = tn('pyodps_test_engine_table2')
        table2 = self._create_table_and_insert_data(table_name, schema2, data2)
        expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

        self._gen_data(data=data)

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
            [conn.close() for conn in _engine_to_connections.values()]
            table2.drop()