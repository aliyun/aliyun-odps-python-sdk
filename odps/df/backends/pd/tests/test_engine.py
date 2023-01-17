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

import functools
import os
import re
import time
import uuid
from datetime import timedelta, datetime
from random import randint
from decimal import Decimal

from odps.df.backends.tests.core import TestBase, to_str, tn, pandas_case
from odps.compat import unittest, irange as xrange, OrderedDict
from odps.errors import ODPSError
from odps.df.types import validate_data_type, DynamicSchema
from odps.df.expr.expressions import *
from odps.df.backends.pd.engine import PandasEngine
from odps.df.backends.odpssql.engine import ODPSSQLEngine
from odps.df.backends.odpssql.types import df_schema_to_odps_schema
from odps.df.backends.context import context
from odps.df.backends.errors import CompileError
from odps.df import output_types, output_names, output, day, millisecond, agg, make_list, make_dict

TEMP_FILE_RESOURCE = tn('pyodps_tmp_file_resource')
TEMP_TABLE = tn('pyodps_temp_table')
TEMP_TABLE_RESOURCE = tn('pyodps_temp_table_resource')


@pandas_case
class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = TableSchema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        self.schema = df_schema_to_odps_schema(schema)

        import pandas as pd
        self.df = pd.DataFrame(None, columns=schema.names)
        self.expr = CollectionExpr(_source_data=self.df, _schema=schema)

        self.engine = PandasEngine(self.odps)
        self.odps_engine = ODPSSQLEngine(self.odps)

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

        import pandas as pd
        self.expr._source_data = pd.DataFrame(data, columns=self.schema.names)
        return data

    def testCache(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr[self.expr.id < 10].cache()
        cnt = expr.count()

        dag = self.engine.compile(expr)
        self.assertEqual(len(dag.nodes()), 2)

        res = self.engine.execute(cnt)
        self.assertEqual(len([it for it in data if it[1] < 10]), res)
        self.assertTrue(context.is_cached(expr))

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

        expr8 = self.expr.fid.max().cache()
        expr9 = self.expr.fid / expr8
        res = self.engine.execute(expr9)
        result = self._get_result(res)
        actual_max = max([it[2] for it in data])
        self.assertListAlmostEqual([it[2] / actual_max for it in data],
                                   [it[0] for it in result])

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

    def testBase(self):
        data = self._gen_data(10, value_range=(-1000, 1000))

        expr = self.expr[::2]
        result = self._get_result(self.engine.execute(expr).values)
        self.assertEqual(data[::2], result)

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

        expr = self.expr.sort('id')[1:5:2]
        res = self.engine.execute(expr)
        result = self._get_result(res.values)
        self.assertEqual(sorted(data, key=lambda it: it[1])[1:5:2], result)

        res = self.expr.head(10)
        result = self._get_result(res.values)
        self.assertEqual(data[:10], result)

        expr = self.expr.name.hash()
        res = self.engine.execute(expr)
        result = self._get_result(res.values)
        self.assertEqual([[hash(r[0])] for r in data], result)

        expr = (self.expr.name == data[0][0]).ifelse('eq', 'ne').rename('name')
        res = self.engine.execute(expr)
        result = self._get_result(res.values)
        self.assertEqual([['eq' if d[0] == data[0][0] else 'ne'] for d in data], result)

        expr = self.expr.sample(parts=10)
        res = self.engine.execute(expr)
        self.assertEqual(len(res), 1)

        expr = self.expr.sample(parts=10, i=(2, 3))
        self.assertRaises(NotImplementedError, lambda: self.engine.execute(expr))

        expr = self.expr.sample(strata='isMale', n={'True': 1, 'False': 1})
        res = self.engine.execute(expr)
        self.assertGreaterEqual(len(res), 1)

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
            (Scalar(10) * 2).rename('const1'),
            (RandomScalar() * 10).rename('rand1'),
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

        self.assertListEqual([it[5] for it in data], [it[11] for it in result])

        self.assertEqual([20] * len(data), [it[12] for it in result])

        for it in result:
            self.assertTrue(0 <= it[13] <= 10)

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

        self.assertEqual([it[1] % 2 for it in data],
                         [it[14] for it in result])

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
            (lambda v: np.trunc(v * 10.0) / 10.0, self.expr.id.trunc(1)),
            (round, self.expr.id.round()),
            (lambda x: round(x, 2), self.expr.id.round(2)),
        ]

        fields = [it[1].rename('id' + str(i)) for i, it in enumerate(methods_to_fields)]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        for i, it in enumerate(methods_to_fields):
            method = it[0]

            first = [method(it[1]) for it in data]
            second = [it[i] for it in result]
            self.assertEqual(len(first), len(second))
            for it1, it2 in zip(first, second):
                if isinstance(it1, float) and np.isnan(it1) and it2 is None:
                    continue
                self.assertAlmostEqual(it1, it2)

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
            (lambda s: extract('123' + s, '[^a-z]*(\w+)', group=1),
             ('123' + self.expr.name).extract('[^a-z]*(\w+)', group=1)),
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

        fields = [it[1].rename('id' + str(i)) for i, it in enumerate(methods_to_fields)]

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
            (lambda s: list(s.birth.map(lambda x: time.mktime(x.timetuple()))),
             self.expr.birth.unix_timestamp),
            (lambda s: list(s.birth.dt.strftime('%Y%d')), self.expr.birth.strftime('%Y%d')),
            (lambda s: list(s.birth.dt.strftime('%Y%d').map(lambda x: datetime.strptime(x, '%Y%d'))),
             self.expr.birth.strftime('%Y%d').strptime('%Y%d')),
        ]

        fields = [it[1].rename('birth' + str(i)) for i, it in enumerate(methods_to_fields)]

        expr = self.expr[fields]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        df = pd.DataFrame(data, columns=self.schema.names)

        for i, it in enumerate(methods_to_fields):
            method = it[0]

            first = method(df)

            second = [it[i] for it in result]
            self.assertEqual(first, second)

    def testFuncion(self):
        data = [
            ['name1', 4, None, None, None, None],
            ['name2', 2, None, None, None, None],
            ['name1', 4, None, None, None, None],
            ['name1', 3, None, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr['id'].map(lambda x: x + 1)

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(result, [[r[1] + 1] for r in data])

        expr = self.expr['id'].map(functools.partial(lambda v, x: x + v, 10))

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(result, [[r[1] + 10] for r in data])

        expr = self.expr['id'].mean().map(lambda x: x + 1)

        res = self.engine.execute(expr)
        ids = [r[1] for r in data]
        self.assertEqual(res, sum(ids) / float(len(ids)) + 1)

        expr = self.expr.apply(lambda row: row.name + str(row.id), axis=1, reduce=True).rename('name')

        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, [[r[0] + str(r[1])] for r in data])

        @output(['id', 'id2'], ['int', 'int'])
        def h(row):
            yield row.id + row.id2, row.id - row.id2

        expr = self.expr['id', Scalar(2).rename('id2')].apply(h, axis=1)

        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual([[d[1] + 2, d[1] - 2] for d in data], result)

        def h(row):
            yield row.id + row.id2

        expr = self.expr['id', Scalar(2).rename('id2')].apply(h, axis=1, reduce=True).rename('addid')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual([[d[1] + 2] for d in data], result)

        def h(row):
            return row.id + row.id2

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual([[d[1] + 2] for d in data], result)

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

        try:
            self.odps.delete_resource(TEMP_FILE_RESOURCE)
        except:
            pass
        file_resource = self.odps.create_resource(TEMP_FILE_RESOURCE, 'file',
                                                  file_obj='\n'.join(str(r[1]) for r in data[:3]))
        self.odps.delete_table(TEMP_TABLE, if_exists=True)
        t = self.odps.create_table(TEMP_TABLE, TableSchema.from_lists(['id'], ['bigint']))
        with t.open_writer() as writer:
            writer.write([r[1: 2] for r in data[3: 4]])
        try:
            self.odps.delete_resource(TEMP_TABLE_RESOURCE)
        except:
            pass
        table_resource = self.odps.create_resource(TEMP_TABLE_RESOURCE, 'table',
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
        import zipfile
        import tarfile
        import requests
        from odps.compat import BytesIO

        data = [
            ['2016', 4, 5.3, None, None, None],
            ['2015', 2, 3.5, None, None, None],
            ['2014', 4, 4.2, None, None, None],
            ['2013', 3, 2.2, None, None, None],
            ['2012', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        utils_urls = [
            'http://mirrors.aliyun.com/pypi/packages/39/7b/'
            '1cb2391517d9cb30001140c6662e00d7443752e5a1713e317fb93267da3f/'
            'python_utils-2.1.0-py2.py3-none-any.whl#md5=9dabec0d4f224ba90fd4c53064e7c016',
            'http://mirrors.aliyun.com/pypi/packages/70/7e/'
            'a2fcd97ec348e63be034027d4475986063c6d869f7e9f1b7802a8b17304e/'
            'python-utils-2.1.0.tar.gz#md5=9891e757c629fc43ccd2c896852f8266',
        ]
        utils_resources = []
        for utils_url, name in zip(utils_urls, ['python_utils.whl', 'python_utils.tar.gz']):
            obj = BytesIO(requests.get(utils_url).content)
            res_name = '%s_%s.%s' % (
                name.split('.', 1)[0], str(uuid.uuid4()).replace('-', '_'), name.split('.', 1)[1])
            res = self.odps.create_resource(res_name, 'file', file_obj=obj)
            utils_resources.append(res)

        obj = BytesIO(requests.get(utils_urls[0]).content)
        res_name = 'python_utils_%s.zip' % str(uuid.uuid4()).replace('-', '_')
        res = self.odps.create_resource(res_name, 'archive', file_obj=obj)
        utils_resources.append(res)

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
                for utils_resource in utils_resources:
                    def f(x):
                        try:
                            from python_utils import converters
                        except ImportError:
                            raise
                        return converters.to_int(x)

                    expr = self.expr.name.map(f, rtype='int')

                    res = self.engine.execute(expr, libraries=[resource.name, utils_resource])
                    result = self._get_result(res)

                    self.assertEqual(result, [[int(r[0].split('-')[0])] for r in data])

                    def f(row):
                        try:
                            from python_utils import converters
                        except ImportError:
                            raise
                        return converters.to_int(row.name),

                    expr = self.expr.apply(f, axis=1, names=['name', ], types=['int', ])

                    res = self.engine.execute(expr, libraries=[resource, utils_resource])
                    result = self._get_result(res)

                    self.assertEqual(result, [[int(r[0].split('-')[0])] for r in data])

                    class Agg(object):
                        def buffer(self):
                            return [0]

                        def __call__(self, buffer, val):
                            try:
                                from python_utils import converters
                            except ImportError:
                                raise
                            buffer[0] += converters.to_int(val)

                        def merge(self, buffer, pbuffer):
                            buffer[0] += pbuffer[0]

                        def getvalue(self, buffer):
                            return buffer[0]

                    expr = self.expr.name.agg(Agg, rtype='int')

                    options.df.libraries = [resource.name, utils_resource]
                    try:
                        res = self.engine.execute(expr)
                    finally:
                        options.df.libraries = None

                    self.assertEqual(res, sum([int(r[0].split('-')[0]) for r in data]))
        finally:
            [res.drop() for res in resources + utils_resources]

    def testCustomLibraries(self):
        data = self._gen_data(5)

        import textwrap
        user_script = textwrap.dedent("""
        def user_fun(a):
            return a + 1
        """)
        rn = 'test_res_%s' % str(uuid.uuid4()).replace('-', '_')
        res = self.odps.create_resource(rn + '.py', 'file', file_obj=user_script)

        def get_fun(code, fun_name):
            g, loc = globals(), dict()
            six.exec_(code, g, loc)
            return loc[fun_name]

        f = get_fun(textwrap.dedent("""
        def f(v):
            from %s import user_fun
            return user_fun(v)
        """ % rn), 'f')

        try:
            expr = self.expr.id.map(f)
            r = self.engine.execute(expr, libraries=[rn + '.py'])
            result = self._get_result(r)
            expect = [[v[1] + 1] for v in data]
            self.assertListEqual(result, expect)
        finally:
            res.drop()

        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            package_dir = os.path.join(temp_dir, 'test_package')
            os.makedirs(package_dir)
            with open(os.path.join(package_dir, '__init__.py'), 'w'):
                pass
            with open(os.path.join(package_dir, 'adder.py'), 'w') as fo:
                fo.write(user_script)

            single_dir = os.path.join(temp_dir, 'test_single')
            os.makedirs(single_dir)
            with open(os.path.join(single_dir, 'user_script.py'), 'w') as fo:
                fo.write(user_script)

            f1 = get_fun(textwrap.dedent("""
            def f1(v):
                from user_script import user_fun
                return user_fun(v)
            """), 'f1')

            expr = self.expr.id.map(f1)
            r = self.engine.execute(expr, libraries=[os.path.join(single_dir, 'user_script.py')])
            result = self._get_result(r)
            expect = [[v[1] + 1] for v in data]
            self.assertListEqual(result, expect)

            f2 = get_fun(textwrap.dedent("""
            def f2(v):
                from test_package.adder import user_fun
                return user_fun(v)
            """), 'f2')

            expr = self.expr.id.map(f2)
            r = self.engine.execute(expr, libraries=[package_dir])
            result = self._get_result(r)
            expect = [[v[1] + 1] for v in data]
            self.assertListEqual(result, expect)
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def testApply(self):
        data = [
            ['name1', 4, None, None, None, None],
            ['name2', 2, None, None, None, None],
            ['name1', 4, None, None, None, None],
            ['name1', 3, None, None, None, None],
        ]
        data = self._gen_data(data=data)

        def my_func(row):
            return row.name

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

        self.assertEqual(sorted([r[0] for r in result]), sorted([r for r in gen_expected(data)]))

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

        expr = self.expr['name',].apply(
            mapper, axis=1, names=['word', 'count'], types=['string', 'int'])
        expr = expr.groupby('word').sort('word').apply(
            reducer, names=['word', 'count'], types=['string', 'int'])

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['key', 3], ['name', 4]]
        self.assertEqual(sorted(result), sorted(expected))

        class reducer(object):
            def __init__(self):
                self._curr = None
                self._cnt = 0

            def __call__(self, row):
                if self._curr is None:
                    self._curr = row.word
                elif self._curr != row.word:
                    assert self._curr > row.word
                    yield (self._curr, self._cnt)
                    self._curr = row.word
                    self._cnt = 0
                self._cnt += row.count

            def close(self):
                if self._curr is not None:
                    yield (self._curr, self._cnt)

        expr = self.expr['name',].apply(
            mapper, axis=1, names=['word', 'count'], types=['string', 'int'])
        expr = expr.groupby('word').sort('word', ascending=False).apply(
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

        expr = self.expr['name',].map_reduce(mapper, reducer, group='word')

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

        expr = self.expr['name',].map_reduce(mapper, reducer2, group='word')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['key', 3], ['name', 4]]
        self.assertEqual(sorted(result), sorted(expected))

        # test no reducer with just combiner
        expr = self.expr['name',].map_reduce(mapper, combiner=reducer2, group='word')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(sorted(result), sorted(expected))

        expr = self.expr['name',].map_reduce(mapper, combiner=reducer, group='word')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(sorted(result), sorted(expected))

        # test both combiner and reducer
        expr = self.expr['name',].map_reduce(mapper, reducer, combiner=reducer2, group='word')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(sorted(result), sorted(expected))

        # test both combiner and reducer and combiner with small buffer
        expr = self.expr['name',].map_reduce(mapper, reducer,
                                             combiner=reducer2, combiner_buffer_size=2, group='word')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(sorted(result), sorted(expected))

    def testMapReduceTypeCheck(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        data = self._gen_data(data=data)

        df = self.expr[self.expr.id.astype('string'),]

        @output(['id'], ['int'])
        def reducer(keys):
            def h(row, done):
                yield row.id

            return h

        df = df.map_reduce(reducer=reducer)
        with self.assertRaises(TypeError):
            self.engine.execute(df)

        df = self.expr[self.expr.id.astype('int8'),]

        @output(['id'], ['int'])
        def reducer(keys):
            def h(row, done):
                yield row.id

            return h

        df = df.map_reduce(reducer=reducer)
        self.engine.execute(df)

        @output(['id'], ['int8'])
        def reducer(keys):
            def h(row, done):
                yield row.id

            return h

        df = self.expr['id',].map_reduce(reducer=reducer)
        with self.assertRaises(TypeError):
            self.engine.execute(df)

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

        schema2 = TableSchema.from_lists(['name2', 'id2', 'id3'],
                                    [types.string, types.int64, types.int64])

        self._gen_data(data=data)

        data2 = [
            ['name1', 4, -1],
        ]

        import pandas as pd
        expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                               _schema=schema2)

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

        expr = self.expr['name',].groupby('name').sort('name').apply(reducer)

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['key', 2], ['name', 3]]
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
        self.assertEqual(expected, result)

    def testPivot(self):
        data = [
            ['name1', 1, 1.0, True, None, None],
            ['name1', 2, 2.0, True, None, None],
            ['name2', 1, 3.0, False, None, None],
            ['name2', 3, 4.0, False, None, None]
        ]
        self._gen_data(data=data)

        expr = self.expr.pivot(rows='id', columns='name', values='fid').distinct()
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [1, 1.0, 3.0],
            [2, 2.0, None],
            [3, None, 4.0]
        ]
        self.assertEqual(sorted(result), sorted(expected))

        expr = self.expr.pivot(rows='id', columns='name', values=['fid', 'isMale'])
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [1, 1.0, 3.0, True, False],
            [2, 2.0, None, True, None],
            [3, None, 4.0, None, False]
        ]
        self.assertEqual(res.schema.names,
                         ['id', 'name1_fid', 'name2_fid', 'name1_isMale', 'name2_isMale'])
        self.assertEqual(sorted(result), sorted(expected))

    def testPivotTable(self):
        data = [
            ['name1', 1, 1.0, True, None, None],
            ['name1', 1, 5.0, True, None, None],
            ['name1', 2, 2.0, True, None, None],
            ['name2', 1, 3.0, False, None, None],
            ['name2', 3, 4.0, False, None, None]
        ]
        self._gen_data(data=data)

        expr = self.expr.pivot_table(rows='name', values='fid')
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 8.0 / 3],
            ['name2', 3.5],
        ]
        self.assertEqual(sorted(result), sorted(expected))

        expr = self.expr.pivot_table(rows='name', values='fid', aggfunc=['mean', 'sum', 'quantile(0.2)'])
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 8.0 / 3, 8.0, 1.4],
            ['name2', 3.5, 7.0, 3.2],
        ]
        self.assertEqual(res.schema.names, ['name', 'fid_mean', 'fid_sum', 'fid_quantile_0_2'])
        self.assertEqual(sorted(result), sorted(expected))

        expr = self.expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum'])
        expr = expr['name1_fid_mean',
                    expr.groupby(Scalar(1)).sort('name1_fid_mean').name1_fid_mean.astype('float').cumsum()]

        k = lambda x: list(0 if it is None else it for it in x)

        expected = [
            [2, 2], [3, 5], [None, 5]
        ]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(sorted(result, key=k), sorted(expected, key=k))

        expr = self.expr.pivot_table(rows='id', values='fid', columns='name', fill_value=0).distinct()
        res = self.engine.execute(expr)
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
        expr = self.expr.pivot_table(rows='id', values='fid', columns='name', fill_value=0,
                                     aggfunc=aggfuncs)
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [1, 6.0, 3.0, 3.0, 3.0],
            [2, 2.0, 0, 2.0, 0],
            [3, 0, 4.0, 0, 4.0]
        ]

        self.assertEqual(res.schema.names, ['id', 'name1_fid_my_sum', 'name2_fid_my_sum',
                                            'name1_fid_mean', 'name2_fid_mean'])
        self.assertEqual(result, expected)

        expr7 = self.expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum']).cache()
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
        expr9 = (expr8['name1_fid_mean'] - expr8['name2_fid_mean']).rename('substract')
        self.assertEqual(len(self.engine.execute(expr9)), 3)
        expr10 = expr8.distinct()
        self.assertEqual(len(self.engine.execute(expr10)), 3)

        expr11 = self.expr.pivot_table(rows='name', columns='id', values='fid', aggfunc='nunique')
        self.assertEqual(len(self.engine.execute(expr11)), 2)

    def testMelt(self):
        import pandas as pd
        data = [
            ['name1', 1.0, 3.0, 10.0],
            ['name1', None, 3.0, 5.1],
            ['name1', 7.1, None, 8.2],
            ['name2', None, 1.2, 1.5],
            ['name2', None, 1.0, 1.1],
        ]
        schema = TableSchema.from_lists(['name', 'k1', 'k2', 'k3'],
                                   [types.string, types.float64, types.float64, types.float64])
        expr_input = CollectionExpr(_source_data=pd.DataFrame(data, columns=schema.names),
                                    _schema=schema)
        expr = expr_input.melt('name')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 'k1', 1.0], ['name1', 'k2', 3.0], ['name1', 'k3', 10.0],
            ['name1', 'k1', None], ['name1', 'k2', 3.0], ['name1', 'k3', 5.1],
            ['name1', 'k1', 7.1], ['name1', 'k2', None], ['name1', 'k3', 8.2],
            ['name2', 'k1', None], ['name2', 'k2', 1.2], ['name2', 'k3', 1.5],
            ['name2', 'k1', None], ['name2', 'k2', 1.0], ['name2', 'k3', 1.1]
        ]

        self.assertListEqual(result, expected)

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

        expr = self.expr.groupby(Scalar(1).rename('s')).count()

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual([5], result[0])

        expr = self.expr.groupby(Scalar('const').rename('s')).id.sum()
        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual([16], result[0])

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

        expr = self.expr.groupby('id').name.cat(sep=',')
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [['name2'], ['name1,name1'], ['name1,name1']]
        self.assertEqual(sorted(result), sorted(expected))

        # only for pandas backend
        self.expr._source_data.loc[5] = ['name2', 4, 4.2, None, None, None]
        expr = self.expr[self.expr.id.isin([2, 4])]
        expr = expr.groupby('id').agg(n=expr.name.nunique())
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [4, 2],
            [2, 1]
        ]
        self.assertEqual(sorted(result), sorted(expected))

    def testMultiNunique(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 2, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr.groupby('name').agg(val=self.expr['name', 'id'].nunique())

        expected = [
            ['name1', 3],
            ['name2', 1]
        ]
        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(sorted(result), sorted(expected))

        expr = self.expr['name', 'id'].nunique()
        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(result, 4)

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

        schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                    [types.string, types.int64, types.int64])

        self._gen_data(data=data)

        data2 = [
            ['name1', 4, -1],
            ['name2', 1, -2]
        ]

        import pandas as pd
        expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                               _schema=schema2)

        expr = self.expr.join(expr2, on='name')[self.expr]
        expr = expr.groupby('id').agg(expr.fid.sum())

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = pd.DataFrame(data, columns=self.expr.schema.names).groupby('id').agg({'fid': 'sum'})
        self.assertEqual(expected.reset_index().values.tolist(), result)

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

        data = [
            [None, 2001, 1, None, None, None],
            [None, 2002, 2, None, None, None],
            [None, 2003, 3, None, None, None]
        ]
        self._gen_data(data=data)

        expr = self.expr.groupby('id').agg(self.expr.fid.sum())
        expr = expr[expr.id == 2003]

        expected = [
            [2003, 3]
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual(expected, result)

    def testGroupbyProjection(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr.groupby('name').agg(id=self.expr.id.max())[
            lambda x: 't' + x.name, lambda x: x.id + 1]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['tname1', 5],
            ['tname2', 3]
        ]

        self.assertEqual(expected, result)

    def testDistinctScalar(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr.distinct('name', 'id')
        expr['scalar'] = 3

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 4, 3],
            ['name2', 2, 3],
            ['name1', 3, 3],
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
            self.expr.groupby('name', 'id').sort('fid').id.nth_value(1).fillna(-1),
            self.expr.groupby('name').id.cummedian(),
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [3, -1, 3.5], [3, 3, 3.5], [4, -1, 3.5], [4, 4, 3.5], [2, -1, 2]
        ]
        self.assertEqual(sorted(expected), sorted(result))

        expr = self.expr.groupby('name').mutate(id2=lambda x: x.id.cumcount(unique=True),
                                                fid=lambda x: x.fid.cummin(sort='id'))

        res = self.engine.execute(expr['name', 'id2', 'fid'])
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
            self.expr.groupby(Scalar(1)).id.rank().rename('rank2'),
            self.expr.groupby('name').sort('fid').qcut(2),
            self.expr.groupby('name').sort('fid').cume_dist(),
        ]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [4, 3, 2, 3, float(2) / 3, 4, 1, 0.75],
            [2, 1, 1, 1, 0.0, 1, 0, 1.0],
            [4, 3, 3, 4, float(2) / 3, 4, 0, 0.5],
            [3, 1, 4, 2, float(0) / 3, 2, 0, 0.25],
            [3, 1, 1, 1, float(0) / 3, 2, 1, 1.0],
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

        import pandas as pd
        df = pd.DataFrame(data, columns=self.schema.names)
        expected = df.id - df.id.max()
        expected = expected - expected.min()
        expected = list(expected[expected - expected.std() > 0])

        self.assertEqual(expected, [it[0] for it in result])

    def testReduction(self):
        data = self._gen_data(rows=5, value_range=(-100, 100), nullable_field='name')

        import pandas as pd
        df = pd.DataFrame(data, columns=self.schema.names)

        class Agg(object):
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

        def quantile(col, prob):
            if isinstance(prob, (list, set)):
                return [quantile(col, p) for p in prob]
            else:
                return col.quantile(prob)

        def filter_none(col):
            import numpy as np
            if hasattr(col, 'dropna'):
                col = col.dropna()
            else:
                try:
                    col = col[~np.isnan(col)]
                except TypeError:
                    col = col[np.fromiter((v is not None for v in col), np.bool_)]
            return col

        methods_to_fields = [
            (lambda s: df.id.mean(), self.expr.id.mean()),
            (lambda s: len(df), self.expr.count()),
            (lambda s: df.id.var(ddof=0), self.expr.id.var(ddof=0)),
            (lambda s: df.id.std(ddof=0), self.expr.id.std(ddof=0)),
            (lambda s: df.id.median(), self.expr.id.median()),
            (lambda s: quantile(df.id, 0.3), self.expr.id.quantile(0.3)),
            (lambda s: quantile(df.id, [0.3, 0.6]), self.expr.id.quantile([0.3, 0.6])),
            (lambda s: df.id.sum(), self.expr.id.sum()),
            (lambda s: df.id.unique().sum(), self.expr.id.unique().sum()),
            (lambda s: (df.id ** 3).mean(), self.expr.id.moment(3)),
            (lambda s: df.id.var(ddof=0), self.expr.id.moment(2, central=True)),
            (lambda s: df.id.skew(), self.expr.id.skew()),
            (lambda s: df.id.kurtosis(), self.expr.id.kurtosis()),
            (lambda s: df.id.min(), self.expr.id.min()),
            (lambda s: df.id.max(), self.expr.id.max()),
            (lambda s: df.isMale.min(), self.expr.isMale.min()),
            (lambda s: len(filter_none(df.name)), self.expr.name.count()),
            (lambda s: filter_none(df.name).max(), self.expr.name.max()),
            (lambda s: df.birth.max(), self.expr.birth.max()),
            (lambda s: filter_none(df.name).sum(), self.expr.name.sum()),
            (lambda s: df.isMale.sum(), self.expr.isMale.sum()),
            (lambda s: df.isMale.any(), self.expr.isMale.any()),
            (lambda s: df.isMale.all(), self.expr.isMale.all()),
            (lambda s: filter_none(df.name).nunique(), self.expr.name.nunique()),
            (lambda s: len(filter_none(df.name).str.cat(sep='|').split('|')),
             self.expr.name.cat(sep='|').map(lambda x: len(x.split('|')), rtype='int')),
            (lambda s: df.id.mean(), self.expr.id.agg(Agg, rtype='float')),
            (lambda s: df.id.count(), self.expr.id.count()),
        ]

        fields = [it[1].rename('f' + str(i)) for i, it in enumerate(methods_to_fields)]

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
            elif isinstance(first, list):
                self.assertListAlmostEqual(first, second)
            else:
                self.assertEqual(first, second)

        self.assertEqual(self.engine.execute(self.expr.id.sum() + 1),
                         sum(it[1] for it in data) + 1)

        expr = self.expr['id', 'fid'].apply(Agg, types=['float'] * 2)

        expected = [[df.id.mean()], [df.fid.mean()]]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        for first, second in zip(expected, result):
            first = first[0]
            second = second[0]

            if isinstance(first, float):
                self.assertAlmostEqual(first, second)
            elif isinstance(first, list):
                self.assertListAlmostEqual(first, second)
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

        expr = self.expr.id.unique().agg(Aggregator)
        expected = float(9) / 3

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertAlmostEqual(expected, result)

        expr = self.expr.groupby(Scalar('const').rename('s')).id.agg(Aggregator)
        expected = float(16) / 5

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

    def testJoin(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                    [types.string, types.int64, types.int64])

        self._gen_data(data=data)

        data2 = [
            ['name1', 4, -1],
            ['name2', 1, -2]
        ]

        import pandas as pd
        expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                               _schema=schema2)

        expr = self.expr.join(expr2).join(expr2)['name', 'id2']

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

        expr = self.expr.join(expr2, on=['name', expr2.id2 == self.expr.id])[self.expr.name, expr2.id2]
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

    def testJoinAggregation(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        expr = self.expr.join(self.expr.view(), on=['name', 'id'])[
            lambda x: x.count(), self.expr.id.sum()]

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertEqual([[9, 30]], result)

    def testUnion(self):
        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                    [types.string, types.int64, types.int64])

        self._gen_data(data=data)

        data2 = [
            ['name3', 5, -1],
            ['name4', 6, -2]
        ]

        import pandas as pd
        expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                               _schema=schema2)

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

    def testConcat(self):
        import pandas as pd
        data = [
            ['name1', 4, 5.3],
            ['name2', 2, 3.5],
            ['name1', 4, 4.2],
            ['name1', 3, 2.2],
            ['name1', 3, 4.1],
        ]
        schema = TableSchema.from_lists(['name', 'id', 'fid'],
                                   [types.string, types.int64, types.float64])
        expr1 = CollectionExpr(_source_data=pd.DataFrame(data, columns=schema.names),
                               _schema=schema)
        data2 = [
            [7.4, 1.5],
            [2.3, 1.7],
            [9.8, 5.4],
            [1.9, 2.2],
            [7.1, 6.2],
        ]
        schema2 = TableSchema.from_lists(['fid2', 'fid3'],
                                    [types.float64, types.float64])
        expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                               _schema=schema2)

        expr = expr1.concat(expr2, axis=1)
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 4, 5.3, 7.4, 1.5],
            ['name2', 2, 3.5, 2.3, 1.7],
            ['name1', 4, 4.2, 9.8, 5.4],
            ['name1', 3, 2.2, 1.9, 2.2],
            ['name1', 3, 4.1, 7.1, 6.2]
        ]

        result = sorted(result)
        expected = sorted(expected)

        self.assertEqual(len(result), len(expected))
        for e, r in zip(result, expected):
            self.assertEqual([to_str(t) for t in e],
                             [to_str(t) for t in r])

    def testScaleValue(self):
        import pandas as pd
        data = [
            ['name1', 4, 5.3],
            ['name2', 2, 3.5],
            ['name2', 3, 1.5],
            ['name1', 4, 4.2],
            ['name1', 3, 2.2],
            ['name1', 3, 4.1],
        ]
        schema = TableSchema.from_lists(['name', 'id', 'fid'],
                                   [types.string, types.int64, types.float64])
        expr_input = CollectionExpr(_source_data=pd.DataFrame(data, columns=schema.names),
                                    _schema=schema)

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

    def testExtractKV(self):
        import pandas as pd
        data = [
            ['name1', 'k1=1,k2=3,k5=10'],
            ['name1', 'k2=3,k3=5.1'],
            ['name1', 'k1=7.1,k7=8.2'],
            ['name2', 'k2=1.2,k3=1.5'],
            ['name2', 'k9=1.1,k2=1'],
        ]
        schema = TableSchema.from_lists(['name', 'kv'],
                                   [types.string, types.string])
        expr_input = CollectionExpr(_source_data=pd.DataFrame(data, columns=schema.names),
                                    _schema=schema)
        expr = expr_input.extract_kv(columns=['kv'], kv_delim='=', fill_value=0)

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected_cols = ['name', 'kv_k1', 'kv_k2', 'kv_k3', 'kv_k5', 'kv_k7', 'kv_k9']
        expected = [
            ['name1', 1.0, 3.0, 0, 10.0, 0, 0],
            ['name1', 0, 3.0, 5.1, 0, 0, 0],
            ['name1', 7.1, 0, 0, 0, 8.2, 0],
            ['name2', 0, 1.2, 1.5, 0, 0, 0],
            ['name2', 0, 1.0, 0, 0, 0, 1.1],
        ]

        self.assertListEqual([c.name for c in res.columns], expected_cols)
        self.assertListEqual(result, expected)

    def testMakeKV(self):
        import pandas as pd
        data = [
            ['name1', 1.0, 3.0, None, 10.0, None, None],
            ['name1', None, 3.0, 5.1, None, None, None],
            ['name1', 7.1, None, None, None, 8.2, None],
            ['name2', None, 1.2, 1.5, None, None, None],
            ['name2', None, 1.0, None, None, None, 1.1],
        ]
        kv_cols = ['k1', 'k2', 'k3', 'k5', 'k7', 'k9']
        schema = TableSchema.from_lists(['name'] + kv_cols,
                                   [types.string] + [types.float64] * 6)
        expr_input = CollectionExpr(_source_data=pd.DataFrame(data, columns=schema.names),
                                    _schema=schema)
        expr = expr_input.to_kv(columns=kv_cols, kv_delim='=')

        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            ['name1', 'k1=1.0,k2=3.0,k5=10.0'],
            ['name1', 'k2=3.0,k3=5.1'],
            ['name1', 'k1=7.1,k7=8.2'],
            ['name2', 'k2=1.2,k3=1.5'],
            ['name2', 'k2=1.0,k9=1.1'],
        ]

        self.assertListEqual(result, expected)

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
            [None, 3, 4.1, None, None, None],
        ]

        data2 = [
            ['name1'],
            ['name3']
        ]

        self._gen_data(data=data)

        schema2 = TableSchema.from_lists(['name', ], [types.string])

        import pandas as pd
        expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                               _schema=schema2)

        expr = self.expr.bloom_filter('name', expr2[:1].name, capacity=10)

        res = self.engine.execute(expr)
        result = self._get_result(res)

        self.assertTrue(all(r[0] != 'name2' for r in result))

    def testPersist(self):
        data = [
            ['name1', 4, 5.3, True, Decimal('3.14'), datetime(1999, 5, 25, 3, 10)],
            ['name2', 2, 3.5, False, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]
        self._gen_data(data=data)

        def simple_persist_test(table_name):
            self.odps.delete_table(table_name, if_exists=True)
            try:
                with self.assertRaises(ODPSError):
                    self.engine.persist(self.expr, table_name, create_table=False)

                df = self.engine.persist(self.expr, table_name)

                res = df.to_pandas()
                result = self._get_result(res)
                self.assertEqual(len(result), 5)
                self.assertEqual(data, result)

                with self.assertRaises(ValueError):
                    self.engine.persist(self.expr, table_name, create_partition=True)
                with self.assertRaises(ValueError):
                    self.engine.persist(self.expr, table_name, drop_partition=True)
            finally:
                self.odps.delete_table(table_name, if_exists=True)

        def persist_existing_test(table_name):
            self.odps.delete_table(table_name, if_exists=True)
            try:
                self.odps.create_table(
                    table_name,
                    'name string, fid double, id bigint, isMale boolean, scale decimal, birth datetime',
                    lifecycle=1,
                )

                expr = self.expr[self.expr, Scalar(1).rename('name2')]
                with self.assertRaises(CompileError):
                    self.engine.persist(expr, table_name)

                expr = self.expr['name', 'fid', self.expr.id.astype('int32'), 'isMale', 'scale', 'birth']
                df = self.engine.persist(expr, table_name)

                res = df.to_pandas()
                result = self._get_result(res)
                self.assertEqual(len(result), 5)
                self.assertEqual(data, [[r[0], r[2], r[1], r[3], r[4], r[5]] for r in result])
            finally:
                self.odps.delete_table(table_name, if_exists=True)

        def persist_with_partition_test(table_name):
            self.odps.delete_table(table_name, if_exists=True)
            try:
                df = self.engine.persist(self.expr, table_name, partition={'ds': 'today'})

                res = self.odps_engine.execute(df)
                result = self._get_result(res)
                self.assertEqual(len(result), 5)
            finally:
                self.odps.delete_table(table_name, if_exists=True)

        def persist_with_create_partition_test(table_name):
            self.odps.delete_table(table_name, if_exists=True)
            try:
                schema = TableSchema.from_lists(self.schema.names, self.schema.types, ['ds'], ['string'])
                self.odps.create_table(table_name, schema)
                df = self.engine.persist(self.expr, table_name, partition='ds=today', create_partition=True)

                res = self.odps_engine.execute(df)
                result = self._get_result(res)
                self.assertEqual(len(result), 5)
                self.assertEqual(data, [d[:-1] for d in result])

                df2 = self.engine.persist(self.expr[self.expr.id.astype('float'), 'name'], table_name,
                                          partition='ds=today2', create_partition=True, cast=True)

                res = self.odps_engine.execute(df2)
                result = self._get_result(res)
                self.assertEqual(len(result), 5)
                self.assertEqual([d[:2] + [None] * (len(d) - 2) for d in data], [d[:-1] for d in result])
            finally:
                self.odps.delete_table(table_name, if_exists=True)

        def persist_with_create_multi_part_test(table_name):
            self.odps.delete_table(table_name, if_exists=True)
            try:
                schema = TableSchema.from_lists(self.schema.names, self.schema.types, ['ds', 'hh'], ['string', 'string'])
                table = self.odps.create_table(table_name, schema)

                with self.assertRaises(ValueError):
                    self.engine.persist(self.expr, table_name, partition='ds=today', create_partition=True)

                self.engine.persist(self.expr, table, partition=OrderedDict([('hh', 'now'), ('ds', 'today')]))
                self.assertTrue(table.exist_partition('ds=today,hh=now'))
            finally:
                self.odps.delete_table(table_name, if_exists=True)

        def persist_with_dyna_part_test(table_name):
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

        sub_tests = [
            simple_persist_test,
            persist_existing_test,
            persist_with_partition_test,
            persist_with_create_partition_test,
            persist_with_create_multi_part_test,
            persist_with_dyna_part_test,
        ]
        base_table_name = tn('pyodps_test_pd_engine_persist_table')
        self.run_sub_tests_in_parallel(
            10,
            [
                functools.partial(sub_test, base_table_name + "_%d" % idx)
                for idx, sub_test in enumerate(sub_tests)
            ]
        )

    def testAppendID(self):
        import pandas as pd
        data = [
            ['name1', 4, 5.3],
            ['name2', 2, 3.5],
            ['name1', 4, 4.2],
            ['name1', 3, 2.2],
            ['name1', 3, 4.1],
        ]
        schema = TableSchema.from_lists(['name', 'id', 'fid'],
                                   [types.string, types.int64, types.float64])
        expr1 = CollectionExpr(_source_data=pd.DataFrame(data, columns=schema.names),
                               _schema=schema)

        expr = expr1.append_id()
        res = self.engine.execute(expr)
        result = self._get_result(res)

        expected = [
            [0, 'name1', 4, 5.3],
            [1, 'name2', 2, 3.5],
            [2, 'name1', 4, 4.2],
            [3, 'name1', 3, 2.2],
            [4, 'name1', 3, 4.1],
        ]
        self.assertListEqual(result, expected)

    def testSplit(self):
        import pandas as pd
        data = [
            [0, 'name1', 4, 5.3],
            [1, 'name2', 2, 3.5],
            [2, 'name1', 4, 4.2],
            [3, 'name1', 3, 2.2],
            [4, 'name1', 3, 4.1],
        ]
        schema = TableSchema.from_lists(['rid', 'name', 'id', 'fid'],
                                   [types.int64, types.string, types.int64, types.float64])
        expr1 = CollectionExpr(_source_data=pd.DataFrame(data, columns=schema.names),
                               _schema=schema)

        expr1, expr2 = expr1.split(0.6)
        res1 = self.engine.execute(expr1)
        result1 = self._get_result(res1)
        res2 = self.engine.execute(expr2)
        result2 = self._get_result(res2)

        merged = sorted(result1 + result2, key=lambda r: r[0])
        self.assertListEqual(data, merged)

    def testCollectionNa(self):
        import pandas as pd
        import numpy as np

        from odps.compat import reduce

        data = [
            [0, 'name1', 1.0, None, 3.0, 4.0],
            [1, 'name1', 2.0, None, None, 1.0],
            [2, 'name1', 3.0, 4.0, 1.0, None],
            [3, 'name1', None, 1.0, 2.0, 3.0],
            [4, 'name1', 1.0, np.nan, 3.0, 4.0],
            [5, 'name1', 1.0, 2.0, 3.0, 4.0],
            [6, 'name1', None, None, np.nan, None],
        ]

        schema = TableSchema.from_lists(['rid', 'name', 'f1', 'f2', 'f3', 'f4'],
                                   [types.int64, types.string] + [types.float64] * 4)
        expr = CollectionExpr(_source_data=pd.DataFrame(data, columns=schema.names),
                              _schema=schema)

        exprs = [
            expr.fillna(100, subset=['f1', 'f2', 'f3', 'f4']),
            expr.fillna(expr.f3, subset=['f1', 'f2', 'f3', 'f4']),
            expr.fillna(method='ffill', subset=['f1', 'f2', 'f3', 'f4']),
            expr.fillna(method='bfill', subset=['f1', 'f2', 'f3', 'f4']),
            expr.dropna(thresh=3, subset=['f1', 'f2', 'f3', 'f4']),
            expr.dropna(how='any', subset=['f1', 'f2', 'f3', 'f4']),
            expr.dropna(how='all', subset=['f1', 'f2', 'f3', 'f4']),
        ]

        uexpr = reduce(lambda a, b: a.union(b), exprs)

        ures = self.engine.execute(uexpr)
        uresult = self._get_result(ures)

        expected = [
            [0, 'name1', 1.0, 100.0, 3.0, 4.0],
            [1, 'name1', 2.0, 100.0, 100.0, 1.0],
            [2, 'name1', 3.0, 4.0, 1.0, 100.0],
            [3, 'name1', 100.0, 1.0, 2.0, 3.0],
            [4, 'name1', 1.0, 100.0, 3.0, 4.0],
            [5, 'name1', 1.0, 2.0, 3.0, 4.0],
            [6, 'name1', 100.0, 100.0, 100.0, 100.0],

            [0, 'name1', 1.0, 3.0, 3.0, 4.0],
            [1, 'name1', 2.0, None, None, 1.0],
            [2, 'name1', 3.0, 4.0, 1.0, 1.0],
            [3, 'name1', 2.0, 1.0, 2.0, 3.0],
            [4, 'name1', 1.0, 3.0, 3.0, 4.0],
            [5, 'name1', 1.0, 2.0, 3.0, 4.0],
            [6, 'name1', None, None, None, None],

            [0, 'name1', 1.0, 1.0, 3.0, 4.0],
            [1, 'name1', 2.0, 2.0, 2.0, 1.0],
            [2, 'name1', 3.0, 4.0, 1.0, 1.0],
            [3, 'name1', None, 1.0, 2.0, 3.0],
            [4, 'name1', 1.0, 1.0, 3.0, 4.0],
            [5, 'name1', 1.0, 2.0, 3.0, 4.0],
            [6, 'name1', None, None, None, None],

            [0, 'name1', 1.0, 3.0, 3.0, 4.0],
            [1, 'name1', 2.0, 1.0, 1.0, 1.0],
            [2, 'name1', 3.0, 4.0, 1.0, None],
            [3, 'name1', 1.0, 1.0, 2.0, 3.0],
            [4, 'name1', 1.0, 3.0, 3.0, 4.0],
            [5, 'name1', 1.0, 2.0, 3.0, 4.0],
            [6, 'name1', None, None, None, None],

            [0, 'name1', 1.0, None, 3.0, 4.0],
            [2, 'name1', 3.0, 4.0, 1.0, None],
            [3, 'name1', None, 1.0, 2.0, 3.0],
            [4, 'name1', 1.0, None, 3.0, 4.0],
            [5, 'name1', 1.0, 2.0, 3.0, 4.0],

            [5, 'name1', 1.0, 2.0, 3.0, 4.0],

            [0, 'name1', 1.0, None, 3.0, 4.0],
            [1, 'name1', 2.0, None, None, 1.0],
            [2, 'name1', 3.0, 4.0, 1.0, None],
            [3, 'name1', None, 1.0, 2.0, 3.0],
            [4, 'name1', 1.0, None, 3.0, 4.0],
            [5, 'name1', 1.0, 2.0, 3.0, 4.0],
        ]
        self.assertListEqual(uresult, expected)

    def testDrop(self):
        import pandas as pd

        data1 = [
            ['name1', 1, 3.0], ['name1', 2, 3.0], ['name1', 2, 2.5],
            ['name2', 1, 1.2], ['name2', 3, 1.0],
            ['name3', 1, 1.2], ['name3', 3, 1.2],
        ]
        schema1 = TableSchema.from_lists(['name', 'id', 'fid'],
                                    [types.string, types.int64, types.float64])
        expr1 = CollectionExpr(_source_data=pd.DataFrame(data1, columns=schema1.names),
                               _schema=schema1)

        data2 = [
            ['name1', 1], ['name1', 2],
            ['name2', 1], ['name2', 2],
        ]
        schema2 = TableSchema.from_lists(['name', 'id'],
                                    [types.string, types.int64])
        expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                               _schema=schema2)

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

    def testExceptIntersect(self):
        import pandas as pd
        data1 = [
            ['name1', 1], ['name1', 2], ['name1', 2], ['name1', 2],
            ['name2', 1], ['name2', 3],
            ['name3', 1], ['name3', 3],
        ]
        schema1 = TableSchema.from_lists(['name', 'id'], [types.string, types.int64])
        expr1 = CollectionExpr(_source_data=pd.DataFrame(data1, columns=schema1.names),
                               _schema=schema1)

        data2 = [
            ['name1', 1], ['name1', 2], ['name1', 2],
            ['name2', 1], ['name2', 2],
        ]
        schema2 = TableSchema.from_lists(['name', 'id'], [types.string, types.int64])
        expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                               _schema=schema2)

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

    def testFilterOrder(self):
        import pandas as pd

        schema = TableSchema.from_lists(['divided', 'divisor'], [types.int64, types.int64])
        pd_df = pd.DataFrame([[2, 0], [1, 1], [1, 2], [5, 1], [5, 0]], columns=schema.names)
        df = CollectionExpr(_source_data=pd_df, _schema=schema)
        fdf = df[df.divisor > 0]
        ddf = fdf[(fdf.divided / fdf.divisor).rename('result'), ]
        expr = ddf[ddf.result > 1]

        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, [[5, ]])

    def testLateralView(self):
        import pandas as pd

        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = TableSchema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        pd_df = pd.DataFrame(data, columns=schema.names)
        expr_in = CollectionExpr(_source_data=pd_df, _schema=schema)

        @output(['name', 'id'], ['string', 'int64'])
        def mapper(row):
            for idx in range(row.id):
                yield '%s_%d' % (row.name, idx), row.id * idx

        expected = [
            [5.3, 'name1_0', 0], [5.3, 'name1_1', 4], [5.3, 'name1_2', 8], [5.3, 'name1_3', 12],
            [3.5, 'name2_0', 0], [3.5, 'name2_1', 2],
            [4.2, 'name1_0', 0], [4.2, 'name1_1', 4], [4.2, 'name1_2', 8], [4.2, 'name1_3', 12],
            [2.2, 'name1_0', 0], [2.2, 'name1_1', 3], [2.2, 'name1_2', 6],
            [4.1, 'name1_0', 0], [4.1, 'name1_1', 3], [4.1, 'name1_2', 6],
        ]

        expr = expr_in[expr_in.fid, expr_in['name', 'id'].apply(mapper, axis=1)]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            [5.3, 'name1_0', 0],
            [3.5, 'name2_0', 0],
            [4.2, 'name1_0', 0],
            [2.2, 'name1_0', 0], [2.2, 'name1_1', 2],
            [4.1, 'name1_0', 0], [4.1, 'name1_1', 2],
        ]

        expr = expr_in[expr_in.fid, expr_in['name', expr_in.id % 2 + 1].apply(mapper, axis=1)]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            [5, 3.5, 'name2_0', 0, '0'], [5, 3.5, 'name2_1', 2, '0'],
            [5, 2.2, 'name1_0', 0, '0'], [5, 2.2, 'name1_0', 0, '1'], [5, 2.2, 'name1_1', 3, '0'],
            [5, 2.2, 'name1_1', 3, '1'], [5, 2.2, 'name1_2', 6, '0'], [5, 2.2, 'name1_2', 6, '1'],
            [5, 4.1, 'name1_0', 0, '0'], [5, 4.1, 'name1_0', 0, '1'], [5, 4.1, 'name1_1', 3, '0'],
            [5, 4.1, 'name1_1', 3, '1'], [5, 4.1, 'name1_2', 6, '0'], [5, 4.1, 'name1_2', 6, '1'],
        ]

        @output(['bin_id'], ['string'])
        def mapper2(row):
            for idx in range(row.id % 2 + 1):
                yield str(idx)

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in['id', ].apply(mapper2, axis=1)]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        @output(['bin_id'], ['string'])
        def mapper3(row):
            for idx in range(row.ren_id % 2 + 1):
                yield str(idx)

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in[expr_in.id.rename('ren_id'), ].apply(mapper3, axis=1)]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            ['name1_0', 12.6], ['name1_1', 12.6], ['name1_2', 12.6],
            ['name2_0', 3.5], ['name2_1', 3.5]
        ]

        @output(['name', 'id'], ['string', 'float'])
        def reducer(keys):
            sums = [0.0]

            def h(row, done):
                sums[0] += row.fid
                if done:
                    yield tuple(keys) + (sums[0], )

            return h

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in['id', ].apply(mapper2, axis=1)]
        expr = expr.map_reduce(reducer=reducer, group='name')
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(sorted(result), sorted(expected))

        expected = [
            [5, 2.2, 'name1_0', 0, '0'], [5, 2.2, 'name1_1', 3, '0'],
            [5, 2.2, 'name1_2', 6, '0'], [5, 4.1, 'name1_0', 0, '0'],
            [5, 4.1, 'name1_1', 3, '0'], [5, 4.1, 'name1_2', 6, '0'],
        ]

        @output(['bin_id'], ['string'])
        def mapper3(row):
            for idx in range(row.id % 2):
                yield str(idx)

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in['id', ].apply(mapper3, axis=1)]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            [5, 3.5, 'name2_0', 0, None], [5, 3.5, 'name2_1', 2, None],
            [5, 2.2, 'name1_0', 0, '0'], [5, 2.2, 'name1_1', 3, '0'],
            [5, 2.2, 'name1_2', 6, '0'], [5, 4.1, 'name1_0', 0, '0'],
            [5, 4.1, 'name1_1', 3, '0'], [5, 4.1, 'name1_2', 6, '0']
        ]

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in['id', ].apply(mapper3, axis=1, keep_nulls=True)]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

    def testComposites(self):
        import pandas as pd

        data = [
            ['name1', 4, 5.3, OrderedDict([('a', 123.2), ('b', 567.1)]), ['YTY', 'HKG', 'SHA', 'PEK']],
            ['name2', 2, 3.5, OrderedDict([('c', 512.1), ('b', 711.2)]), None],
            ['name1', 4, 4.2, None, ['Hawaii', 'Texas']],
            ['name1', 3, 2.2, OrderedDict([('u', 115.4), ('v', 312.1)]), ['Washington', 'London', 'Paris', 'Frankfort']],
            ['name1', 3, 4.1, OrderedDict([('w', 923.2), ('x', 456.1)]), ['Moscow', 'Warsaw', 'Prague', 'Belgrade']],
        ]

        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = TableSchema.from_lists(['name', 'grade', 'score', 'detail', 'locations'],
                                   datatypes('string', 'int64', 'float64',
                                             'dict<string, float64>', 'list<string>'))
        pd_df = pd.DataFrame(data, columns=schema.names)
        expr_in = CollectionExpr(_source_data=pd_df, _schema=schema)

        expected = [
            ['name1', 3, 'Washington', 'u', 115.4], ['name1', 3, 'Washington', 'v', 312.1],
            ['name1', 3, 'London', 'u', 115.4], ['name1', 3, 'London', 'v', 312.1],
            ['name1', 3, 'Paris', 'u', 115.4], ['name1', 3, 'Paris', 'v', 312.1],
            ['name1', 3, 'Frankfort', 'u', 115.4], ['name1', 3, 'Frankfort', 'v', 312.1]
        ]

        expr = expr_in[expr_in.score < 4][expr_in.name, expr_in.grade, expr_in.locations.explode(),
                                          expr_in.detail.explode()]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            ['name1', 123.2], ['name1', 567.1],
            ['name2', 512.1], ['name2', 711.2],
            ['name1', 115.4], ['name1', 312.1],
            ['name1', 923.2], ['name1', 456.1],
        ]

        expr = expr_in[expr_in.name, expr_in.detail.values().explode()]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            ['name1', 3, 0, 'Washington', 'u', 115.4, [1, 2], [1, 3], OrderedDict([('a', 1), ('b', 2)]),
             OrderedDict([('a', 3), ('b', 6)])],
            ['name1', 3, 0, 'Washington', 'v', 312.1, [1, 2], [1, 3], OrderedDict([('a', 1), ('b', 2)]),
             OrderedDict([('a', 3), ('b', 6)])],
            ['name1', 3, 1, 'London', 'u', 115.4, [1, 2], [1, 3], OrderedDict([('a', 1), ('b', 2)]),
             OrderedDict([('a', 3), ('b', 6)])],
            ['name1', 3, 1, 'London', 'v', 312.1, [1, 2], [1, 3], OrderedDict([('a', 1), ('b', 2)]),
             OrderedDict([('a', 3), ('b', 6)])],
            ['name1', 3, 2, 'Paris', 'u', 115.4, [1, 2], [1, 3], OrderedDict([('a', 1), ('b', 2)]),
             OrderedDict([('a', 3), ('b', 6)])],
            ['name1', 3, 2, 'Paris', 'v', 312.1, [1, 2], [1, 3], OrderedDict([('a', 1), ('b', 2)]),
             OrderedDict([('a', 3), ('b', 6)])],
            ['name1', 3, 3, 'Frankfort', 'u', 115.4, [1, 2], [1, 3], OrderedDict([('a', 1), ('b', 2)]),
             OrderedDict([('a', 3), ('b', 6)])],
            ['name1', 3, 3, 'Frankfort', 'v', 312.1, [1, 2], [1, 3], OrderedDict([('a', 1), ('b', 2)]),
             OrderedDict([('a', 3), ('b', 6)])],
        ]

        expr = expr_in[expr_in.score < 4][expr_in.name, expr_in.grade, expr_in.locations.explode(pos=True),
                                          expr_in.detail.explode(), make_list(1, 2).rename('arr1'),
                                          make_list(1, expr_in.grade).rename('arr2'),
                                          make_dict('a', 1, 'b', 2).rename('dict1'),
                                          make_dict('a', expr_in.grade, 'b', expr_in.grade * 2).rename('dict2')]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            ['name1', 4.0, 2.0, False, False, ['HKG', 'PEK', 'SHA', 'YTY'],
             ['a', 'b'], [123.2, 567.1], None, 'YTY', 'PEK', 'PEK'],
            ['name2', None, 2.0, None, None, None,
             ['b', 'c'], [512.1, 711.2], None, None, None, None],
            ['name1', 2.0, None, False, False, ['Hawaii', 'Texas'],
             None, None, None, 'Hawaii', 'Texas', None],
            ['name1', 4.0, 2.0, False, False, ['Frankfort', 'London', 'Paris', 'Washington'],
             ['u', 'v'], [115.4, 312.1], 115.4, 'Washington', 'Frankfort', 'Paris'],
            ['name1', 4.0, 2.0, True, False, ['Belgrade', 'Moscow', 'Prague', 'Warsaw'],
             ['w', 'x'], [456.1, 923.2], None, 'Moscow', 'Belgrade', 'Prague']
        ]

        expr = expr_in[expr_in.name, expr_in.locations.len().rename('loc_len'),
                       expr_in.detail.len().rename('detail_len'),
                       expr_in.locations.contains('Moscow').rename('has_mos'),
                       expr_in.locations.contains(expr_in.name).rename('has_name'),
                       expr_in.locations.sort().rename('loc_sort'),
                       expr_in.detail.keys().sort().rename('detail_keys'),
                       expr_in.detail.values().sort().rename('detail_values'),
                       expr_in.detail['u'].rename('detail_u'),
                       expr_in.locations[0].rename('loc_0'),
                       expr_in.locations[-1].rename('loc_m1'),
                       expr_in.locations[expr_in.grade - 1].rename('loc_id')]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            ['name1', [4, 4, 3, 3]],
            ['name2', [2]]
        ]

        expr = expr_in.groupby(expr_in.name).agg(agg_grades=expr_in.grade.tolist())
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            ['name1', [3, 4]],
            ['name2', [2]]
        ]

        expr = expr_in.groupby(expr_in.name).agg(agg_grades=expr_in.grade.tolist(unique=True).sort())
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [[['name1', 'name2', 'name1', 'name1', 'name1'], [4, 2, 4, 3, 3]]]

        expr = expr_in['name', 'grade'].tolist()
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [[['name1', 'name2'], [2, 3, 4]]]

        expr = expr_in['name', 'grade'].tolist(unique=True)
        expr = expr[tuple(f.sort() for f in expr.columns)]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

    def testStringSplits(self):
        import pandas as pd

        data = [
            ['name1:a,name3:5', 4],
            ['name2:4,name7:1', 2],
            ['name1:1', 4],
            ['name1:4,name5:6,name4:1', 3],
            ['name1:2,name10:1', 3],
        ]

        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = TableSchema.from_lists(['name', 'id'],
                                   datatypes('string', 'int64'))
        pd_df = pd.DataFrame(data, columns=schema.names)
        expr_in = CollectionExpr(_source_data=pd_df, _schema=schema)

        expected = [
            [4, ['name1:a', 'name3:5']],
            [2, ['name2:4', 'name7:1']],
            [4, ['name1:1']],
            [3, ['name1:4', 'name5:6', 'name4:1']],
            [3, ['name1:2', 'name10:1']],
        ]

        expr = expr_in[expr_in.id, expr_in.name.split(',')]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)

        expected = [
            [4, {'name1': 'a', 'name3': '5'}],
            [2, {'name2': '4', 'name7': '1'}],
            [4, {'name1': '1'}],
            [3, {'name1': '4', 'name5': '6', 'name4': '1'}],
            [3, {'name1': '2', 'name10': '1'}],
        ]

        expr = expr_in[expr_in.id, expr_in.name.todict(kv_delim=':')]
        res = self.engine.execute(expr)
        result = self._get_result(res)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
