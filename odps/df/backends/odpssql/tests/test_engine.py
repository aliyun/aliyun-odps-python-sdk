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

from odps.df.backends.tests.core import TestBase, to_str
from odps.compat import unittest
from odps.models import Schema, Instance
from odps import types
from odps.df.types import validate_data_type
from odps.df.backends.odpssql.types import df_schema_to_odps_schema, \
    odps_schema_to_df_schema
from odps.df.expr.expressions import CollectionExpr
from odps.df.backends.odpssql.engine import ODPSEngine
from odps.df import Scalar, output_names, output_types, output, day, millisecond


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        self.schema = df_schema_to_odps_schema(schema)
        table_name = 'pyodps_test_engine_table'
        self.odps.delete_table(table_name, if_exists=True)
        self.table = self.odps.create_table(
                name='pyodps_test_engine_table', schema=self.schema)
        self.expr = CollectionExpr(_source_data=self.table, _schema=schema)

        self.engine = ODPSEngine(self.odps)

        class FakeBar(object):
            def update(self, *args, **kwargs):
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

        self.odps.write_table(self.table, 0, [self.table.new_record(values=d) for d in data])
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

        expr = self.expr['name', self.expr.id.rename('new_id')]
        res = self.engine._handle_cases(expr, self.faked_bar)
        result = self._get_result(res)
        self.assertEqual([it[:2] for it in data], result)

        table_name = 'pyodps_test_engine_partitioned'
        self.odps.delete_table(table_name, if_exists=True)

        df = self.engine.persist(self.expr, table_name, partitions=['name'])

        try:
            expr = df.count()
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertIsNone(res)

            expr = df[df.name == data[0][0]]['fid', 'id'].count()
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(res, 0)

            expr = df[df.name == data[0][0]]['fid', 'id']
            res = self.engine._handle_cases(expr, self.faked_bar)
            self.assertGreater(len(res), 0)
        finally:
            self.odps.delete_table(table_name, if_exists=True)

    def testAsync(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr.id.sum()

        res = self.engine.execute(expr, async=True)
        self.assertNotEqual(res.instance.status, Instance.Status.TERMINATED)
        res.wait()

        self.assertEqual(sum(it[1] for it in data), res.fetch())

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

        def get_val(val):
            if val <= 100:
                return 'xsmall'
            elif 100 < val <= 200:
                return 'small'
            elif 200 < val <= 300:
                return 'large'
            else:
                return 'xlarge'
        self.assertEqual([to_str(get_val(it[1])) for it in data], [to_str(it[9]) for it in result])

    def testArithmetic(self):
        data = self._gen_data(5, value_range=(-1000, 1000))

        fields = [
            (self.expr.id + 1).rename('id1'),
            (self.expr.fid - 1).rename('fid1'),
            (self.expr.scale * 2).rename('scale1'),
            (self.expr.scale + self.expr.id).rename('scale2'),
            (self.expr.id / 2).rename('id2'),
            (self.expr.id ** -2).rename('id3'),
            abs(self.expr.id).rename('id4'),
            (~self.expr.id).rename('id5'),
            (-self.expr.fid).rename('fid2'),
            (~self.expr.isMale).rename('isMale1'),
            (-self.expr.isMale).rename('isMale2'),
            (self.expr.id // 2).rename('id6'),
            (self.expr.birth + day(1).rename('birth1')),
            (self.expr.birth - (self.expr.birth - millisecond(10))).rename('birth2'),
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

        self.assertEqual([int(it[1] ** -2) for it in data],
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

    def testMath(self):
        data = self._gen_data(5, value_range=(1, 90))

        import numpy as np

        methods_to_fields = [
            (np.sin, self.expr.id.sin()),
            (np.cos, self.expr.id.cos()),
            (np.tan, self.expr.id.tan()),
            (np.sinh, self.expr.id.sinh()),
            (np.cosh, self.expr.id.cosh()),
            (np.tanh, self.expr.id.tanh()),
            (np.log, self.expr.id.log()),
            (np.log2, self.expr.id.log2()),
            (np.log10, self.expr.id.log10()),
            (np.log1p, self.expr.id.log1p()),
            (np.exp, self.expr.id.exp()),
            (np.expm1, self.expr.id.expm1()),
            (np.arccosh, self.expr.id.arccosh()),
            (np.arcsinh, self.expr.id.arcsinh()),
            (np.arctanh, self.expr.id.arctanh()),
            (np.arctan, self.expr.id.arctan()),
            (np.sqrt, self.expr.id.sqrt()),
            (np.abs, self.expr.id.abs()),
            (np.ceil, self.expr.id.ceil()),
            (np.floor, self.expr.id.floor()),
            (np.trunc, self.expr.id.trunc()),
        ]

        fields = [it[1].rename('id'+str(i)) for i, it in enumerate(methods_to_fields)]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        for i, it in enumerate(methods_to_fields):
            method = it[0]

            first = [method(it[1]) for it in data]
            second = [it[i] for it in result]
            self.assertEqual(len(first), len(second))
            for it1, it2 in zip(first, second):
                if np.isnan(it1) and np.isnan(it2):
                    continue
                self.assertAlmostEqual(it1, it2)

    def testString(self):
        data = self._gen_data(5)

        methods_to_fields = [
            (lambda s: s.capitalize(), self.expr.name.capitalize()),
            (lambda s: data[0][0] in s, self.expr.name.contains(data[0][0], regex=False)),
            (lambda s: s.count(data[0][0]), self.expr.name.count(data[0][0])),
            (lambda s: s.endswith(data[0][0]), self.expr.name.endswith(data[0][0])),
            (lambda s: s.startswith(data[0][0]), self.expr.name.startswith(data[0][0])),
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
            (lambda s: s[2: 10: 2], self.expr.name.slice(2, 10, 2)),
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

            first = [method(it[0]) for it in data]
            second = [it[i] for it in result]
            self.assertEqual(first, second)

    def testApply(self):
        data = self._gen_data(5)

        def my_func(row):
            return row.name,

        expr = self.expr['name', 'id'].apply(my_func, axis=1, names='name')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual([r[0] for r in result], [r[0] for r in data])

        def my_func2(row):
            yield len(row.name)
            yield row.id

        expr = self.expr['name', 'id'].apply(my_func2, axis=1, names='cnt', types='int')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        def gen_expected(data):
            for r in data:
                yield len(r[0])
                yield r[1]

        self.assertEqual([r[0] for r in result], [r for r in gen_expected(data)])

    def testDatetime(self):
        data = self._gen_data(5)

        import pandas as pd

        methods_to_fields = [
            (lambda s: list(s.birth.dt.year.values), self.expr.birth.year),
            (lambda s: list(s.birth.dt.month.values), self.expr.birth.month),
            (lambda s: list(s.birth.dt.day.values), self.expr.birth.day),
            (lambda s: list(s.birth.dt.hour.values), self.expr.birth.hour),
            (lambda s: list(s.birth.dt.minute.values), self.expr.birth.minute),
            (lambda s: list(s.birth.dt.second.values), self.expr.birth.second),
            (lambda s: list(s.birth.dt.weekofyear.values), self.expr.birth.weekofyear),
            (lambda s: list(s.birth.dt.dayofweek.values), self.expr.birth.dayofweek),
            (lambda s: list(s.birth.dt.weekday.values), self.expr.birth.weekday),
            (lambda s: list(s.birth.dt.date.values), self.expr.birth.date),
            (lambda s: list(s.birth.dt.strftime('%Y%d')), self.expr.birth.strftime('%Y%d'))
        ]

        fields = [it[1].rename('birth'+str(i)) for i, it in enumerate(methods_to_fields)]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        df = pd.DataFrame(data, columns=self.schema.names)

        for i, it in enumerate(methods_to_fields):
            method = it[0]

            first = method(df)

            def conv(v):
                if isinstance(v, pd.Timestamp):
                    return v.to_datetime().date()
                else:
                    return v

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
        self.assertEqual(expected, result)

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

        import pandas as pd
        df = pd.DataFrame(data, columns=self.schema.names)
        expected = df.id - df.id.max()
        expected = expected - expected.min()
        expected = list(expected[expected - expected.std() > 0])

        self.assertEqual(expected, [it[0] for it in result])

    def testReduction(self):
        data = self._gen_data(rows=5, value_range=(-100, 100))

        import pandas as pd
        df = pd.DataFrame(data, columns=self.schema.names)

        methods_to_fields = [
            (lambda s: df.id.mean(), self.expr.id.mean()),
            (lambda s: len(df), self.expr.count()),
            (lambda s: df.id.var(ddof=0), self.expr.id.var(ddof=0)),
            (lambda s: df.id.std(ddof=0), self.expr.id.std(ddof=0)),
            (lambda s: df.id.median(), self.expr.id.median()),
            (lambda s: df.id.sum(), self.expr.id.sum()),
            (lambda s: df.id.min(), self.expr.id.min()),
            (lambda s: df.id.max(), self.expr.id.max()),
            (lambda s: df.isMale.min(), self.expr.isMale.min()),
            (lambda s: df.name.max(), self.expr.name.max()),
            (lambda s: df.birth.max(), self.expr.birth.max()),
            (lambda s: df.isMale.sum(), self.expr.isMale.sum()),
            (lambda s: df.isMale.any(), self.expr.isMale.any()),
            (lambda s: df.isMale.all(), self.expr.isMale.all()),
            (lambda s: df.name.nunique(), self.expr.name.nunique()),
        ]

        fields = [it[1].rename('f'+str(i)) for i, it in enumerate(methods_to_fields)]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        df = pd.DataFrame(data, columns=self.schema.names)

        for i, it in enumerate(methods_to_fields):
            method = it[0]

            first = method(df)
            second = [it[i] for it in result][0]
            if isinstance(first, float):
                self.assertAlmostEqual(first, second)
            else:
                self.assertEqual(first, second)

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
        def mapper(row):
            for word in row[0].split():
                yield word, 1

        @output(['word', 'cnt'], ['string', 'int'])
        def reducer(keys):
            cnt = [0, ]

            def h(row, done):
                cnt[0] += row[1]
                if done:
                    yield keys[0], cnt[0]

            return h

        expr = self.expr['name', ].map_reduce(mapper, reducer, group='word')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['key', 3], ['name', 4]]
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
        table_name = 'pyodps_test_engine_table2'
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name, schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

        self._gen_data(data=data)

        data2 = [
            ['name1', 4, -1],
            ['name2', 1, -2]
        ]

        self.odps.write_table(table2, 0, [table2.new_record(values=d) for d in data2])

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
        table_name = 'pyodps_test_engine_table2'
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name, schema=schema2)
        expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

        self._gen_data(data=data)

        data2 = [
            ['name3', 5, -1],
            ['name4', 6, -2]
        ]

        self.odps.write_table(table2, 0, [table2.new_record(values=d) for d in data2])

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

    def testPersist(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        table_name = 'pyodps_test_engine_persist_table'

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
            self.assertEqual(data, [d[:-1] for d in result])
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

    def teardown(self):
        self.table.drop()


if __name__ == '__main__':
    unittest.main()