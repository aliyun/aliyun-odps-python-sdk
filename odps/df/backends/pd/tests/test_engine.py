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
import re
import time
import uuid
import warnings
from collections import namedtuple
from datetime import timedelta, datetime
from decimal import Decimal
from random import randint

import pytest
try:
    import pandas as pd
except ImportError:
    pd = None
    pytestmark = pytest.mark.skip

from .....compat import irange as xrange, BytesIO
from .....errors import ODPSError
from .....tests.core import get_result, approx_list, run_sub_tests_in_parallel, tn
from .....utils import to_text
from .... import output_types, output_names, output, day, millisecond, agg, make_list, make_dict
from ....expr.expressions import *
from ....types import validate_data_type, DynamicSchema
from ...tests.core import NumGenerators
from ...odpssql.engine import ODPSSQLEngine
from ...odpssql.types import df_schema_to_odps_schema
from ...context import context
from ...errors import CompileError
from ..engine import PandasEngine

TEMP_FILE_RESOURCE = tn('pyodps_tmp_file_resource')
TEMP_TABLE = tn('pyodps_temp_table')
TEMP_TABLE_RESOURCE = tn('pyodps_temp_table_resource')


@pytest.fixture
def setup(odps):
    def gen_data(rows=None, data=None, nullable_field=None, value_range=None):
        if data is None:
            data = []
            for _ in range(rows):
                record = []
                for t in schema.types:
                    method = getattr(NumGenerators, 'gen_random_%s' % t.name)
                    if t.name == 'bigint':
                        record.append(method(value_range=value_range))
                    else:
                        record.append(method())
                data.append(record)

            if nullable_field is not None:
                j = schema._name_indexes[nullable_field]
                for i, l in enumerate(data):
                    if i % 2 == 0:
                        data[i][j] = None

        expr._source_data = pd.DataFrame(data, columns=schema.names)
        return data

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    pd_schema = TableSchema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                               datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
    schema = df_schema_to_odps_schema(pd_schema)

    df = pd.DataFrame(None, columns=pd_schema.names)
    expr = CollectionExpr(_source_data=df, _schema=pd_schema)

    engine = PandasEngine(odps)
    odps_engine = ODPSSQLEngine(odps)

    class FakeBar(object):
        def update(self, *args, **kwargs):
            pass

    faked_bar = FakeBar()

    nt = namedtuple("NT", "schema df expr engine odps_engine faked_bar gen_data")
    return nt(schema, df, expr, engine, odps_engine, faked_bar, gen_data)


def test_cache(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr[setup.expr.id < 10].cache()
    cnt = expr.count()

    dag = setup.engine.compile(expr)
    assert len(dag.nodes()) == 2

    res = setup.engine.execute(cnt)
    assert len([it for it in data if it[1] < 10]) == res
    assert context.is_cached(expr) is True

    expr2 = expr['id']
    res = setup.engine.execute(expr2, _force_tunnel=True)
    result = get_result(res)
    assert [[it[1]] for it in data if it[1] < 10] == result

    expr3 = setup.expr.sample(parts=10)
    expr3.cache()
    assert setup.engine.execute(expr3.id.sum()) is not None

    expr4 = setup.expr['id', 'name', 'fid'].cache()
    expr5 = expr4[expr4.id < 0]['id', 'name']
    expr6 = expr4[expr4.id >= 0]['name', 'id']
    expr7 = expr5.union(expr6).union(expr4['name', 'id'])
    assert setup.engine.execute(expr7.count()) > 0

    expr8 = setup.expr.fid.max().cache()
    expr9 = setup.expr.fid / expr8
    res = setup.engine.execute(expr9)
    result = get_result(res)
    actual_max = max([it[2] for it in data])
    assert approx_list([it[2] / actual_max for it in data]) == [it[0] for it in result]


def test_batch(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr[setup.expr.id < 10].cache()
    expr1 = expr.id.sum()
    expr2 = expr.id.mean()

    dag = setup.engine.compile([expr1, expr2])
    assert len(dag.nodes()) == 3
    assert sum(len(v) for v in dag._graph.values()) == 2

    expect1 = sum(d[1] for d in data if d[1] < 10)
    length = len([d[1] for d in data if d[1] < 10])
    expect2 = (expect1 / float(length)) if length > 0 else 0.0

    res = setup.engine.execute([expr1, expr2], n_parallel=2)
    assert res[0] == expect1
    assert pytest.approx(res[1]) == expect2
    assert context.is_cached(expr) is True


def test_base(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr[::2]
    result = get_result(setup.engine.execute(expr).values)
    assert data[::2] == result

    expr = setup.expr[setup.expr.id < 10]['name', lambda x: x.id]
    result = get_result(setup.engine.execute(expr).values)
    assert len([it for it in data if it[1] < 10]) == len(result)
    if len(result) > 0:
        assert 2 == len(result[0])

    expr = setup.expr[Scalar(3).rename('const'), setup.expr.id, (setup.expr.id + 1).rename('id2')]
    res = setup.engine.execute(expr)
    result = get_result(res.values)
    assert [c.name for c in res.columns] == ['const', 'id', 'id2']
    assert all(it[0] == 3 for it in result) is True
    assert len(data) == len(result)
    assert [it[1] + 1 for it in data] == [it[2] for it in result]

    expr = setup.expr.sort('id')[1:5:2]
    res = setup.engine.execute(expr)
    result = get_result(res.values)
    assert sorted(data, key=lambda it: it[1])[1:5:2] == result

    res = setup.expr.head(10)
    result = get_result(res.values)
    assert data[:10] == result

    expr = setup.expr.name.hash()
    res = setup.engine.execute(expr)
    result = get_result(res.values)
    assert [[hash(r[0])] for r in data] == result

    expr = (setup.expr.name == data[0][0]).ifelse('eq', 'ne').rename('name')
    res = setup.engine.execute(expr)
    result = get_result(res.values)
    assert [['eq' if d[0] == data[0][0] else 'ne'] for d in data] == result

    expr = setup.expr.sample(parts=10)
    res = setup.engine.execute(expr)
    assert len(res) == 1

    expr = setup.expr.sample(parts=10, i=(2, 3))
    pytest.raises(NotImplementedError, lambda: setup.engine.execute(expr))

    expr = setup.expr.sample(strata='isMale', n={'True': 1, 'False': 1})
    res = setup.engine.execute(expr)
    assert len(res) >= 1

    expr = setup.expr[:1].filter(lambda x: x.name == data[1][0])
    res = setup.engine.execute(expr)
    assert len(res) == 0

    expr = setup.expr.exclude('scale').describe()
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert list(result[0][1:]) == [10, 10]


def test_chinese(odps, setup):
    data = [
        ['中文', 4, 5.3, None, None, None],
        ['\'中文2', 2, 3.5, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.filter(setup.expr.name == '中文')
    res = setup.engine.execute(expr)
    assert len(res) == 1

    expr = setup.expr.filter(setup.expr.name == '\'中文2')
    res = setup.engine.execute(expr)
    assert len(res) == 1


def test_element(odps, setup):
    data = setup.gen_data(5, nullable_field='name')

    fields = [
        setup.expr.name.isnull().rename('name1'),
        setup.expr.name.notnull().rename('name2'),
        setup.expr.name.isna().rename('name3'),
        setup.expr.name.notna().rename('name4'),
        setup.expr.name.fillna('test').rename('name5'),
        setup.expr.id.isin([1, 2, 3]).rename('id1'),
        setup.expr.id.isin(setup.expr.fid.astype('int')).rename('id2'),
        setup.expr.id.notin([1, 2, 3]).rename('id3'),
        setup.expr.id.notin(setup.expr.fid.astype('int')).rename('id4'),
        setup.expr.id.between(setup.expr.fid, 3).rename('id5'),
        setup.expr.name.fillna('test').switch('test', 'test' + setup.expr.name.fillna('test'),
                                             'test2', 'test2' + setup.expr.name.fillna('test'),
                                             default=setup.expr.name).rename('name6'),
        setup.expr.name.fillna('test').switch('test', 1, 'test2', 2).rename('name7'),
        setup.expr.id.cut([100, 200, 300],
                         labels=['xsmall', 'small', 'large', 'xlarge'],
                         include_under=True, include_over=True).rename('id6'),
        setup.expr.birth.unix_timestamp.to_datetime().rename('birth1'),
        (Scalar(10) * 2).rename('const1'),
        (RandomScalar() * 10).rename('rand1'),
    ]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert len(data) == len(result)

    assert len([it for it in data if it[0] is None]) == len([it[0] for it in result if it[0]])

    assert len([it[0] for it in data if it[0] is not None]) == len([it[1] for it in result if it[1]])

    assert len([it for it in data if it[0] is None]) == len([it[2] for it in result if it[0]])

    assert len([it[0] for it in data if it[0] is not None]) == len([it[3] for it in result if it[1]])

    assert [(it[0] if it[0] is not None else 'test') for it in data] == [it[4] for it in result]

    assert [(it[1] in (1, 2, 3)) for it in data] == [it[5] for it in result]

    fids = [int(it[2]) for it in data]
    assert [(it[1] in fids) for it in data] == [it[6] for it in result]

    assert [(it[1] not in (1, 2, 3)) for it in data] == [it[7] for it in result]

    assert [(it[1] not in fids) for it in data] == [it[8] for it in result]

    assert [(it[2] <= it[1] <= 3) for it in data] == [it[9] for it in result]

    assert [to_text('testtest' if it[0] is None else it[0]) for it in data] == [to_text(it[10]) for it in result]

    assert [1 if it[0] is None else None for it in data] == [it[11] for it in result]

    def get_val(val):
        if val <= 100:
            return 'xsmall'
        elif 100 < val <= 200:
            return 'small'
        elif 200 < val <= 300:
            return 'large'
        else:
            return 'xlarge'

    assert [to_text(get_val(it[1])) for it in data] == [to_text(it[12]) for it in result]

    assert [it[5] for it in data] == [it[13] for it in result]

    assert [20] * len(data) == [it[14] for it in result]

    for it in result:
        assert 0 <= it[15] <= 10


def test_arithmetic(odps, setup):
    data = setup.gen_data(5, value_range=(-1000, 1000))

    fields = [
        (setup.expr.id + 1).rename('id1'),
        (setup.expr.fid - 1).rename('fid1'),
        (setup.expr.scale * 2).rename('scale1'),
        (setup.expr.scale + setup.expr.id).rename('scale2'),
        (setup.expr.id / 2).rename('id2'),
        (setup.expr.id ** -2).rename('id3'),
        abs(setup.expr.id).rename('id4'),
        (~setup.expr.id).rename('id5'),
        (-setup.expr.fid).rename('fid2'),
        (~setup.expr.isMale).rename('isMale1'),
        (-setup.expr.isMale).rename('isMale2'),
        (setup.expr.id // 2).rename('id6'),
        (setup.expr.birth + day(1).rename('birth1')),
        (setup.expr.birth - (setup.expr.birth - millisecond(10))).rename('birth2'),
        (setup.expr.id % 2).rename('id7'),
    ]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert len(data) == len(result)

    assert [it[1] + 1 for it in data] == [it[0] for it in result]

    assert pytest.approx([it[2] - 1 for it in data]) == [it[1] for it in result]

    assert [it[4] * 2 for it in data] == [it[2] for it in result]

    assert [it[4] + it[1] for it in data] == [it[3] for it in result]

    assert pytest.approx([float(it[1]) / 2 for it in data]) == [it[4] for it in result]

    assert [int(it[1] ** -2) for it in data] == [it[5] for it in result]

    assert [abs(it[1]) for it in data] == [it[6] for it in result]

    assert [~it[1] for it in data] == [it[7] for it in result]

    assert pytest.approx([-it[2] for it in data]) == [it[8] for it in result]

    assert [not it[3] for it in data] == [it[9] for it in result]

    assert [it[1] // 2 for it in data] == [it[11] for it in result]

    assert [it[5] + timedelta(days=1) for it in data] == [it[12] for it in result]

    assert [10] * len(data) == [it[13] for it in result]

    assert [it[1] % 2 for it in data] == [it[14] for it in result]


def test_math(odps, setup):
    data = setup.gen_data(5, value_range=(1, 90))

    import numpy as np

    methods_to_fields = [
        (np.sin, setup.expr.id.sin()),
        (np.cos, setup.expr.id.cos()),
        (np.tan, setup.expr.id.tan()),
        (np.sinh, setup.expr.id.sinh()),
        (np.cosh, setup.expr.id.cosh()),
        (np.tanh, setup.expr.id.tanh()),
        (np.log, setup.expr.id.log()),
        (np.log2, setup.expr.id.log2()),
        (np.log10, setup.expr.id.log10()),
        (np.log1p, setup.expr.id.log1p()),
        (np.exp, setup.expr.id.exp()),
        (np.expm1, setup.expr.id.expm1()),
        (np.arccosh, setup.expr.id.arccosh()),
        (np.arcsinh, setup.expr.id.arcsinh()),
        (np.arctanh, setup.expr.id.arctanh()),
        (np.arctan, setup.expr.id.arctan()),
        (np.sqrt, setup.expr.id.sqrt()),
        (np.abs, setup.expr.id.abs()),
        (np.ceil, setup.expr.id.ceil()),
        (np.floor, setup.expr.id.floor()),
        (lambda v: np.trunc(v * 10.0) / 10.0, setup.expr.id.trunc(1)),
        (round, setup.expr.id.round()),
        (lambda x: round(x, 2), setup.expr.id.round(2)),
    ]

    fields = [it[1].rename('id' + str(i)) for i, it in enumerate(methods_to_fields)]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    for i, it in enumerate(methods_to_fields):
        method = it[0]

        first = [method(it[1]) for it in data]
        second = [it[i] for it in result]
        assert len(first) == len(second)
        for it1, it2 in zip(first, second):
            if isinstance(it1, float) and np.isnan(it1) and it2 is None:
                continue
            assert pytest.approx(it1) == it2


def test_string(odps, setup):
    data = setup.gen_data(5)

    def extract(x, pat, group):
        regex = re.compile(pat)
        m = regex.match(x)
        if m:
            return m.group(group)

    methods_to_fields = [
        (lambda s: s.capitalize(), setup.expr.name.capitalize()),
        (lambda s: data[0][0] in s, setup.expr.name.contains(data[0][0], regex=False)),
        (lambda s: s[0] + '|' + str(s[1]), setup.expr.name.cat(setup.expr.id.astype('string'), sep='|')),
        (lambda s: s.count(data[0][0]), setup.expr.name.count(data[0][0])),
        (lambda s: s.endswith(data[0][0]), setup.expr.name.endswith(data[0][0])),
        (lambda s: s.startswith(data[0][0]), setup.expr.name.startswith(data[0][0])),
        (lambda s: extract('123' + s, r'[^a-z]*(\w+)', group=1),
         ('123' + setup.expr.name).extract(r'[^a-z]*(\w+)', group=1)),
        (lambda s: s.find(data[0][0]), setup.expr.name.find(data[0][0])),
        (lambda s: s.rfind(data[0][0]), setup.expr.name.rfind(data[0][0])),
        (lambda s: s.replace(data[0][0], 'test'), setup.expr.name.replace(data[0][0], 'test')),
        (lambda s: s[0], setup.expr.name.get(0)),
        (lambda s: len(s), setup.expr.name.len()),
        (lambda s: s.ljust(10), setup.expr.name.ljust(10)),
        (lambda s: s.ljust(20, '*'), setup.expr.name.ljust(20, fillchar='*')),
        (lambda s: s.rjust(10), setup.expr.name.rjust(10)),
        (lambda s: s.rjust(20, '*'), setup.expr.name.rjust(20, fillchar='*')),
        (lambda s: s * 4, setup.expr.name.repeat(4)),
        (lambda s: s[1:], setup.expr.name.slice(1)),
        (lambda s: s[1: 6], setup.expr.name.slice(1, 6)),
        (lambda s: s[2: 10: 2], setup.expr.name.slice(2, 10, 2)),
        (lambda s: s[-5: -1], setup.expr.name.slice(-5, -1)),
        (lambda s: s.title(), setup.expr.name.title()),
        (lambda s: s.rjust(20, '0'), setup.expr.name.zfill(20)),
        (lambda s: s.isalnum(), setup.expr.name.isalnum()),
        (lambda s: s.isalpha(), setup.expr.name.isalpha()),
        (lambda s: s.isdigit(), setup.expr.name.isdigit()),
        (lambda s: s.isspace(), setup.expr.name.isspace()),
        (lambda s: s.isupper(), setup.expr.name.isupper()),
        (lambda s: s.istitle(), setup.expr.name.istitle()),
        (lambda s: to_text(s).isnumeric(), setup.expr.name.isnumeric()),
        (lambda s: to_text(s).isdecimal(), setup.expr.name.isdecimal()),
    ]

    fields = [it[1].rename('id' + str(i)) for i, it in enumerate(methods_to_fields)]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    for i, it in enumerate(methods_to_fields):
        method = it[0]

        if i != 2:
            first = [method(it[0]) for it in data]
        else:
            # cat
            first = [method(it) for it in data]
        second = [it[i] for it in result]
        assert first == second


def test_datetime(odps, setup):
    data = setup.gen_data(5)

    import pandas as pd

    methods_to_fields = [
        (lambda s: list(s.birth.dt.year.values), setup.expr.birth.year),
        (lambda s: list(s.birth.dt.month.values), setup.expr.birth.month),
        (lambda s: list(s.birth.dt.day.values), setup.expr.birth.day),
        (lambda s: list(s.birth.dt.hour.values), setup.expr.birth.hour),
        (lambda s: list(s.birth.dt.minute.values), setup.expr.birth.minute),
        (lambda s: list(s.birth.dt.second.values), setup.expr.birth.second),
        (lambda s: list(s.birth.dt.weekofyear.values), setup.expr.birth.weekofyear),
        (lambda s: list(s.birth.dt.dayofweek.values), setup.expr.birth.dayofweek),
        (lambda s: list(s.birth.dt.weekday.values), setup.expr.birth.weekday),
        (lambda s: list(s.birth.dt.date.values), setup.expr.birth.date),
        (lambda s: list(s.birth.map(lambda x: time.mktime(x.timetuple()))),
         setup.expr.birth.unix_timestamp),
        (lambda s: list(s.birth.dt.strftime('%Y%d')), setup.expr.birth.strftime('%Y%d')),
        (lambda s: list(s.birth.dt.strftime('%Y%d').map(lambda x: datetime.strptime(x, '%Y%d'))),
         setup.expr.birth.strftime('%Y%d').strptime('%Y%d')),
    ]

    fields = [it[1].rename('birth' + str(i)) for i, it in enumerate(methods_to_fields)]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    df = pd.DataFrame(data, columns=setup.schema.names)

    for i, it in enumerate(methods_to_fields):
        method = it[0]

        first = method(df)

        second = [it[i] for it in result]
        assert first == second


def test_function(odps, setup):
    data = [
        ['name1', 4, None, None, None, None],
        ['name2', 2, None, None, None, None],
        ['name1', 4, None, None, None, None],
        ['name1', 3, None, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr['id'].map(lambda x: x + 1)

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert result == [[r[1] + 1] for r in data]

    expr = setup.expr['id'].map(functools.partial(lambda v, x: x + v, 10))

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert result == [[r[1] + 10] for r in data]

    expr = setup.expr['id'].mean().map(lambda x: x + 1)

    res = setup.engine.execute(expr)
    ids = [r[1] for r in data]
    assert res == sum(ids) / float(len(ids)) + 1

    expr = setup.expr.apply(lambda row: row.name + str(row.id), axis=1, reduce=True).rename('name')

    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == [[r[0] + str(r[1])] for r in data]

    @output(['id', 'id2'], ['int', 'int'])
    def h(row):
        yield row.id + row.id2, row.id - row.id2

    expr = setup.expr['id', Scalar(2).rename('id2')].apply(h, axis=1)

    res = setup.engine.execute(expr)
    result = get_result(res)
    assert [[d[1] + 2, d[1] - 2] for d in data] == result

    def h(row):
        yield row.id + row.id2

    expr = setup.expr['id', Scalar(2).rename('id2')].apply(h, axis=1, reduce=True).rename('addid')

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [[d[1] + 2] for d in data] == result

    def h(row):
        return row.id + row.id2

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [[d[1] + 2] for d in data] == result


def test_function_resources(odps, setup):
    data = setup.gen_data(5)

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
        odps.delete_resource(TEMP_FILE_RESOURCE)
    except:
        pass
    file_resource = odps.create_resource(TEMP_FILE_RESOURCE, 'file',
                                              file_obj='\n'.join(str(r[1]) for r in data[:3]))
    odps.delete_table(TEMP_TABLE, if_exists=True)
    t = odps.create_table(TEMP_TABLE, TableSchema.from_lists(['id'], ['bigint']))
    with t.open_writer() as writer:
        writer.write([r[1: 2] for r in data[3: 4]])
    try:
        odps.delete_resource(TEMP_TABLE_RESOURCE)
    except:
        pass
    table_resource = odps.create_resource(TEMP_TABLE_RESOURCE, 'table',
                                               table_name=t.name)

    try:
        expr = setup.expr.id.map(my_func, resources=[file_resource, table_resource])

        res = setup.engine.execute(expr)
        result = get_result(res)
        result = [r for r in result if r[0] is not None]

        assert sorted([[r[1]] for r in data[:4]]) == sorted(result)

        expr = setup.expr['name', 'id', 'fid']
        expr = expr.apply(my_func, axis=1, resources=[file_resource, table_resource],
                          names=expr.schema.names, types=expr.schema.types)

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert sorted([r[:3] for r in data[:4]]) == sorted(result)

        expr = setup.expr['name', 'id', 'fid']
        expr = expr.apply(my_func2, axis=1, resources=[file_resource, table_resource],
                          names=expr.schema.names, types=expr.schema.types)

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert sorted([r[:3] for r in data[:4]]) == sorted(result)
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


def test_third_party_libraries(odps, setup):
    import zipfile
    import tarfile
    import requests
    from .....compat import reduce

    data = [
        ['2016', 4, 5.3, None, None, None],
        ['2015', 2, 3.5, None, None, None],
        ['2014', 4, 4.2, None, None, None],
        ['2013', 3, 2.2, None, None, None],
        ['2012', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

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
        res = odps.create_resource(res_name, 'file', file_obj=obj)
        utils_resources.append(res)

    obj = BytesIO(requests.get(utils_urls[0]).content)
    res_name = 'python_utils_%s.zip' % str(uuid.uuid4()).replace('-', '_')
    res = odps.create_resource(res_name, 'archive', file_obj=obj)
    utils_resources.append(res)

    resources = []
    six_path = os.path.join(os.path.dirname(os.path.abspath(six.__file__)), 'six.py')

    zip_io = BytesIO()
    zip_f = zipfile.ZipFile(zip_io, 'w')
    zip_f.write(six_path, arcname='mylib/six.py')
    zip_f.close()
    zip_io.seek(0)

    rn = 'six_%s.zip' % str(uuid.uuid4())
    resource = odps.create_resource(rn, 'file', file_obj=zip_io)
    resources.append(resource)

    tar_io = BytesIO()
    tar_f = tarfile.open(fileobj=tar_io, mode='w:gz')
    tar_f.add(six_path, arcname='mylib/six.py')
    tar_f.close()
    tar_io.seek(0)

    rn = 'six_%s.tar.gz' % str(uuid.uuid4())
    resource = odps.create_resource(rn, 'file', file_obj=tar_io)
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

                expr = setup.expr.name.map(f, rtype='int')

                res = setup.engine.execute(expr, libraries=[resource.name, utils_resource])
                result = get_result(res)

                assert result == [[int(r[0].split('-')[0])] for r in data]

                def f(row):
                    try:
                        from python_utils import converters
                    except ImportError:
                        raise
                    return converters.to_int(row.name),

                expr = setup.expr.apply(f, axis=1, names=['name', ], types=['int', ])

                res = setup.engine.execute(expr, libraries=[resource, utils_resource])
                result = get_result(res)

                assert result == [[int(r[0].split('-')[0])] for r in data]

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

                expr = setup.expr.name.agg(Agg, rtype='int')

                options.df.libraries = [resource.name, utils_resource]
                try:
                    res = setup.engine.execute(expr)
                finally:
                    options.df.libraries = None

                assert res == sum([int(r[0].split('-')[0]) for r in data])
    finally:
        [res.drop() for res in resources + utils_resources]


def test_custom_libraries(odps, setup):
    data = setup.gen_data(5)

    import textwrap
    user_script = textwrap.dedent("""
    def user_fun(a):
        return a + 1
    """)
    rn = 'test_res_%s' % str(uuid.uuid4()).replace('-', '_')
    res = odps.create_resource(rn + '.py', 'file', file_obj=user_script)

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
        expr = setup.expr.id.map(f)
        r = setup.engine.execute(expr, libraries=[rn + '.py'])
        result = get_result(r)
        expect = [[v[1] + 1] for v in data]
        assert result == expect
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

        expr = setup.expr.id.map(f1)
        r = setup.engine.execute(expr, libraries=[os.path.join(single_dir, 'user_script.py')])
        result = get_result(r)
        expect = [[v[1] + 1] for v in data]
        assert result == expect

        f2 = get_fun(textwrap.dedent("""
        def f2(v):
            from test_package.adder import user_fun
            return user_fun(v)
        """), 'f2')

        expr = setup.expr.id.map(f2)
        r = setup.engine.execute(expr, libraries=[package_dir])
        result = get_result(r)
        expect = [[v[1] + 1] for v in data]
        assert result == expect
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_apply(odps, setup):
    data = [
        ['name1', 4, None, None, None, None],
        ['name2', 2, None, None, None, None],
        ['name1', 4, None, None, None, None],
        ['name1', 3, None, None, None, None],
    ]
    data = setup.gen_data(data=data)

    def my_func(row):
        return row.name

    expr = setup.expr['name', 'id'].apply(my_func, axis=1, names='name')

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [r[0] for r in result] == [r[0] for r in data]

    def my_func2(row):
        yield len(row.name)
        yield row.id

    expr = setup.expr['name', 'id'].apply(my_func2, axis=1, names='cnt', types='int')

    res = setup.engine.execute(expr)
    result = get_result(res)

    def gen_expected(data):
        for r in data:
            yield len(r[0])
            yield r[1]

    assert sorted([r[0] for r in result]) == sorted([r for r in gen_expected(data)])


def test_map_reduce_by_apply_distribute_sort(odps, setup):
    data = [
        ['name key', 4, 5.3, None, None, None],
        ['name', 2, 3.5, None, None, None],
        ['key', 4, 4.2, None, None, None],
        ['name', 3, 2.2, None, None, None],
        ['key name', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

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

    expr = setup.expr['name',].apply(
        mapper, axis=1, names=['word', 'count'], types=['string', 'int'])
    expr = expr.groupby('word').sort('word').apply(
        reducer, names=['word', 'count'], types=['string', 'int'])

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [['key', 3], ['name', 4]]
    assert sorted(result) == sorted(expected)

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

    expr = setup.expr['name',].apply(
        mapper, axis=1, names=['word', 'count'], types=['string', 'int'])
    expr = expr.groupby('word').sort('word', ascending=False).apply(
        reducer, names=['word', 'count'], types=['string', 'int'])

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [['key', 3], ['name', 4]]
    assert sorted(result) == sorted(expected)


def test_map_reduce(odps, setup):
    data = [
        ['name key', 4, 5.3, None, None, None],
        ['name', 2, 3.5, None, None, None],
        ['key', 4, 4.2, None, None, None],
        ['name', 3, 2.2, None, None, None],
        ['key name', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

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

    expr = setup.expr['name',].map_reduce(mapper, reducer, group='word')

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [['key', 3], ['name', 4]]
    assert sorted(result) == sorted(expected)

    @output(['word', 'cnt'], ['string', 'int'])
    class reducer2(object):
        def __init__(self, keys):
            self.cnt = 0

        def __call__(self, row, done):
            self.cnt += row.cnt
            if done:
                yield row.word, self.cnt

    expr = setup.expr['name',].map_reduce(mapper, reducer2, group='word')

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [['key', 3], ['name', 4]]
    assert sorted(result) == sorted(expected)

    # test no reducer with just combiner
    expr = setup.expr['name',].map_reduce(mapper, combiner=reducer2, group='word')

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert sorted(result) == sorted(expected)

    expr = setup.expr['name',].map_reduce(mapper, combiner=reducer, group='word')

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert sorted(result) == sorted(expected)

    # test both combiner and reducer
    expr = setup.expr['name',].map_reduce(mapper, reducer, combiner=reducer2, group='word')

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert sorted(result) == sorted(expected)

    # test both combiner and reducer and combiner with small buffer
    expr = setup.expr['name',].map_reduce(mapper, reducer,
                                         combiner=reducer2, combiner_buffer_size=2, group='word')

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert sorted(result) == sorted(expected)


def test_map_reduce_type_check(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    data = setup.gen_data(data=data)

    df = setup.expr[setup.expr.id.astype('string'),]

    @output(['id'], ['int'])
    def reducer(keys):
        def h(row, done):
            yield row.id

        return h

    df = df.map_reduce(reducer=reducer)
    with pytest.raises(TypeError):
        setup.engine.execute(df)

    df = setup.expr[setup.expr.id.astype('int8'),]

    @output(['id'], ['int'])
    def reducer(keys):
        def h(row, done):
            yield row.id

        return h

    df = df.map_reduce(reducer=reducer)
    setup.engine.execute(df)

    @output(['id'], ['int8'])
    def reducer(keys):
        def h(row, done):
            yield row.id

        return h

    df = setup.expr['id',].map_reduce(reducer=reducer)
    with pytest.raises(TypeError):
        setup.engine.execute(df)


def test_reduce_only(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]

    data = setup.gen_data(data=data)

    df = setup.expr[(setup.expr.id < 10) & (setup.expr.name.startswith('n'))]
    df = df['name', 'id']

    @output(['name', 'id'], ['string', 'int'])
    def reducer(keys):
        def h(row, done):
            yield row

        return h

    df2 = df.map_reduce(reducer=reducer, group='name')
    res = setup.engine.execute(df2)
    result = get_result(res)
    assert sorted([r[:2] for r in data]) == sorted(result)


def test_join_map_reduce(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]

    schema2 = TableSchema.from_lists(['name2', 'id2', 'id3'],
                                [types.string, types.int64, types.int64])

    setup.gen_data(data=data)

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

    expr = setup.expr.join(expr2, on=('name', 'name2'))
    expr = expr.map_reduce(reducer=reducer, group='name')

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert len(result) == 1
    assert result[0][0] == 14


def test_distribute_sort(odps, setup):
    data = [
        ['name', 4, 5.3, None, None, None],
        ['name', 2, 3.5, None, None, None],
        ['key', 4, 4.2, None, None, None],
        ['name', 3, 2.2, None, None, None],
        ['key', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

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

    expr = setup.expr['name',].groupby('name').sort('name').apply(reducer)

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [['key', 2], ['name', 3]]
    assert sorted(expected) == sorted(result)


def test_sort_distinct(odps, setup):
    data = [
        ['name1', 4, None, None, None, None],
        ['name2', 2, None, None, None, None],
        ['name1', 4, None, None, None, None],
        ['name1', 3, None, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.sort(['name', -setup.expr.id]).distinct(['name', lambda x: x.id + 1])[:50]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert len(result) == 3

    expected = [
        ['name1', 5],
        ['name1', 4],
        ['name2', 3]
    ]
    assert expected == result


def test_pivot(odps, setup):
    data = [
        ['name1', 1, 1.0, True, None, None],
        ['name1', 2, 2.0, True, None, None],
        ['name2', 1, 3.0, False, None, None],
        ['name2', 3, 4.0, False, None, None]
    ]
    setup.gen_data(data=data)

    expr = setup.expr.pivot(rows='id', columns='name', values='fid').distinct()
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [1, 1.0, 3.0],
        [2, 2.0, None],
        [3, None, 4.0]
    ]
    assert sorted(result) == sorted(expected)

    expr = setup.expr.pivot(rows='id', columns='name', values=['fid', 'isMale'])
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [1, 1.0, 3.0, True, False],
        [2, 2.0, None, True, None],
        [3, None, 4.0, None, False]
    ]
    assert res.schema.names == ['id', 'name1_fid', 'name2_fid', 'name1_isMale', 'name2_isMale']
    assert sorted(result) == sorted(expected)


def test_pivot_table(odps, setup):
    data = [
        ['name1', 1, 1.0, True, None, None],
        ['name1', 1, 5.0, True, None, None],
        ['name1', 2, 2.0, True, None, None],
        ['name2', 1, 3.0, False, None, None],
        ['name2', 3, 4.0, False, None, None]
    ]
    setup.gen_data(data=data)

    expr = setup.expr.pivot_table(rows='name', values='fid')
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 8.0 / 3],
        ['name2', 3.5],
    ]
    assert sorted(result) == sorted(expected)

    expr = setup.expr.pivot_table(rows='name', values='fid', aggfunc=['mean', 'sum', 'quantile(0.2)'])
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 8.0 / 3, 8.0, 1.4],
        ['name2', 3.5, 7.0, 3.2],
    ]
    assert res.schema.names == ['name', 'fid_mean', 'fid_sum', 'fid_quantile_0_2']
    assert sorted(result) == sorted(expected)

    expr = setup.expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum'])
    expr = expr['name1_fid_mean',
                expr.groupby(Scalar(1)).sort('name1_fid_mean').name1_fid_mean.astype('float').cumsum()]

    k = lambda x: list(0 if it is None else it for it in x)

    expected = [
        [2, 2], [3, 5], [None, 5]
    ]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert sorted(result, key=k) == sorted(expected, key=k)

    expr = setup.expr.pivot_table(rows='id', values='fid', columns='name', fill_value=0).distinct()
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [1, 3.0, 3.0],
        [2, 2.0, 0],
        [3, 0, 4.0]
    ]

    assert res.schema.names == ['id', 'name1_fid_mean', 'name2_fid_mean']
    assert result == expected

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
    expr = setup.expr.pivot_table(rows='id', values='fid', columns='name', fill_value=0,
                                 aggfunc=aggfuncs)
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [1, 6.0, 3.0, 3.0, 3.0],
        [2, 2.0, 0, 2.0, 0],
        [3, 0, 4.0, 0, 4.0]
    ]

    assert res.schema.names == ['id', 'name1_fid_my_sum', 'name2_fid_my_sum',
                                        'name1_fid_mean', 'name2_fid_mean']
    assert result == expected

    expr7 = setup.expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum']).cache()
    assert len(setup.engine.execute(expr7)) == 3

    expr5 = setup.expr.pivot_table(rows='id', values='fid', columns='name').cache()
    expr6 = expr5[expr5['name1_fid_mean'].rename('tname1'), expr5['name2_fid_mean'].rename('tname2')]

    @output(['tname1', 'tname2'], ['float', 'float'])
    def h(row):
        yield row.tname1, row.tname2

    expr6 = expr6.map_reduce(mapper=h)
    assert len(setup.engine.execute(expr6)) == 3

    expr8 = setup.expr.pivot_table(rows='id', values='fid', columns='name')
    assert len(setup.engine.execute(expr8)) == 3
    assert not isinstance(expr8.schema, DynamicSchema)
    expr9 = (expr8['name1_fid_mean'] - expr8['name2_fid_mean']).rename('substract')
    assert len(setup.engine.execute(expr9)) == 3
    expr10 = expr8.distinct()
    assert len(setup.engine.execute(expr10)) == 3

    expr11 = setup.expr.pivot_table(rows='name', columns='id', values='fid', aggfunc='nunique')
    assert len(setup.engine.execute(expr11)) == 2


def test_melt(odps, setup):
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

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 'k1', 1.0], ['name1', 'k2', 3.0], ['name1', 'k3', 10.0],
        ['name1', 'k1', None], ['name1', 'k2', 3.0], ['name1', 'k3', 5.1],
        ['name1', 'k1', 7.1], ['name1', 'k2', None], ['name1', 'k3', 8.2],
        ['name2', 'k1', None], ['name2', 'k2', 1.2], ['name2', 'k3', 1.5],
        ['name2', 'k1', None], ['name2', 'k2', 1.0], ['name2', 'k3', 1.1]
    ]

    assert result == expected


def test_groupby_aggregation(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    class Agg(object):
        def buffer(self):
            return [0]

        def __call__(self, buffer, val):
            buffer[0] += val

        def merge(self, buffer, pbuffer):
            buffer[0] += pbuffer[0]

        def getvalue(self, buffer):
            return buffer[0]

    expr = setup.expr.groupby(['name', 'id'])[lambda x: x.fid.min() * 2 < 8] \
        .agg(setup.expr.fid.max() + 1, new_id=setup.expr.id.sum(),
             new_id2=setup.expr.id.agg(Agg))

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 3, 5.1, 6, 6],
        ['name2', 2, 4.5, 2, 2]
    ]

    result = sorted(result, key=lambda k: k[0])

    assert expected == result

    expr = setup.expr.groupby(Scalar(1).rename('s')).count()

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [5] == result[0]

    expr = setup.expr.groupby(Scalar('const').rename('s')).id.sum()
    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [16] == result[0]

    field = setup.expr.groupby('name').sort(['id', -setup.expr.fid]).row_number()
    expr = setup.expr['name', 'id', 'fid', field]

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 3, 4.1, 1],
        ['name1', 3, 2.2, 2],
        ['name1', 4, 5.3, 3],
        ['name1', 4, 4.2, 4],
        ['name2', 2, 3.5, 1],
    ]

    result = sorted(result, key=lambda k: (k[0], k[1], -k[2]))

    assert expected == result

    expr = setup.expr.name.value_counts(dropna=True)[:25]

    expected = [
        ['name1', 4],
        ['name2', 1]
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert expected == result

    expr = setup.expr.name.topk(25)

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert expected == result

    expr = setup.expr.groupby('name').count()

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [it[1:] for it in expected] == result

    expected = [
        ['name1', 2],
        ['name2', 1]
    ]

    expr = setup.expr.groupby('name').id.nunique()

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [it[1:] for it in expected] == result

    expr = setup.expr[setup.expr['id'] > 2].name.value_counts()[:25]

    expected = [
        ['name1', 4]
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert expected == result

    expr = setup.expr.groupby('name', Scalar(1).rename('constant')) \
        .agg(id=setup.expr.id.sum())

    expected = [
        ['name1', 1, 14],
        ['name2', 1, 2]
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert expected == result

    expr = setup.expr[:1]
    expr = expr.groupby('name').agg(expr.id.sum())

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 4]
    ]

    assert expected == result

    expr = setup.expr.groupby('id').name.cat(sep=',')
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [['name2'], ['name1,name1'], ['name1,name1']]
    assert sorted(result) == sorted(expected)

    # only for pandas backend
    setup.expr._source_data.loc[5] = ['name2', 4, 4.2, None, None, None]
    expr = setup.expr[setup.expr.id.isin([2, 4])]
    expr = expr.groupby('id').agg(n=expr.name.nunique())
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [4, 2],
        [2, 1]
    ]
    assert sorted(result) == sorted(expected)


def test_multi_nunique(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 2, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.groupby('name').agg(val=setup.expr['name', 'id'].nunique())

    expected = [
        ['name1', 3],
        ['name2', 1]
    ]
    res = setup.engine.execute(expr)
    result = get_result(res)

    assert sorted(result) == sorted(expected)

    expr = setup.expr['name', 'id'].nunique()
    res = setup.engine.execute(expr)
    result = get_result(res)

    assert result == 4


def test_projection_groupby_filter(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    df = setup.expr.copy()
    df['id'] = df.id + 1
    df2 = df.groupby('name').agg(id=df.id.sum())[lambda x: x.name == 'name2']

    expected = [['name2', 3]]
    res = setup.engine.execute(df2)
    result = get_result(res)
    assert expected == result


def test_join_groupby(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]

    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                [types.string, types.int64, types.int64])

    setup.gen_data(data=data)

    data2 = [
        ['name1', 4, -1],
        ['name2', 1, -2]
    ]

    import pandas as pd
    expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                           _schema=schema2)

    expr = setup.expr.join(expr2, on='name')[setup.expr]
    expr = expr.groupby('id').agg(expr.fid.sum())

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = pd.DataFrame(data, columns=setup.expr.schema.names).groupby('id').agg({'fid': 'sum'})
    assert expected.reset_index().values.tolist() == result


def test_filter_groupby(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.groupby(['name']).agg(id=setup.expr.id.max())[lambda x: x.id > 3]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert len(result) == 1

    expected = [
        ['name1', 4]
    ]

    assert expected == result

    data = [
        [None, 2001, 1, None, None, None],
        [None, 2002, 2, None, None, None],
        [None, 2003, 3, None, None, None]
    ]
    setup.gen_data(data=data)

    expr = setup.expr.groupby('id').agg(setup.expr.fid.sum())
    expr = expr[expr.id == 2003]

    expected = [
        [2003, 3]
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert expected == result


def test_groupby_projection(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.groupby('name').agg(id=setup.expr.id.max())[
        lambda x: 't' + x.name, lambda x: x.id + 1]

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['tname1', 5],
        ['tname2', 3]
    ]

    assert expected == result


def test_distinct_scalar(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.distinct('name', 'id')
    expr['scalar'] = 3

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 4, 3],
        ['name2', 2, 3],
        ['name1', 3, 3],
    ]

    assert expected == result


def test_window_function(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 6.1, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.groupby('name').id.cumsum()

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [[14]] * 4 + [[2]]
    assert sorted(expected) == sorted(result)

    expr = setup.expr.groupby('name').sort('fid').id.cummax()

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [[3], [4], [4], [4], [2]]
    assert sorted(expected) == sorted(result)

    expr = setup.expr[
        setup.expr.groupby('name', 'id').sort('fid').id.cummean(),
        setup.expr.groupby('name', 'id').sort('fid').id.nth_value(1).fillna(-1),
        setup.expr.groupby('name').id.cummedian(),
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [3, -1, 3.5], [3, 3, 3.5], [4, -1, 3.5], [4, 4, 3.5], [2, -1, 2]
    ]
    assert sorted(expected) == sorted(result)

    expr = setup.expr.groupby('name').mutate(id2=lambda x: x.id.cumcount(unique=True),
                                            fid=lambda x: x.fid.cummin(sort='id'))

    res = setup.engine.execute(expr['name', 'id2', 'fid'])
    result = get_result(res)

    expected = [
        ['name1', 2, 2.2],
        ['name1', 2, 2.2],
        ['name1', 2, 2.2],
        ['name1', 2, 2.2],
        ['name2', 1, 3.5],
    ]
    assert sorted(expected) == sorted(result)

    expr = setup.expr[
        setup.expr.id,
        setup.expr.groupby('name').rank('id'),
        setup.expr.groupby('name').dense_rank('fid', ascending=False),
        setup.expr.groupby('name').row_number(sort=['id', 'fid'], ascending=[True, False]),
        setup.expr.groupby('name').percent_rank('id'),
        setup.expr.groupby(Scalar(1)).id.rank().rename('rank2'),
        setup.expr.groupby('name').sort('fid').qcut(2),
        setup.expr.groupby('name').sort('fid').cume_dist(),
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [4, 3, 2, 3, float(2) / 3, 4, 1, 0.75],
        [2, 1, 1, 1, 0.0, 1, 0, 1.0],
        [4, 3, 3, 4, float(2) / 3, 4, 0, 0.5],
        [3, 1, 4, 2, float(0) / 3, 2, 0, 0.25],
        [3, 1, 1, 1, float(0) / 3, 2, 1, 1.0],
    ]
    assert sorted(expected) == sorted(result)

    expr = setup.expr[
        setup.expr.id,
        setup.expr.groupby('name').id.lag(offset=3, default=0, sort=['id', 'fid']).rename('id2'),
        setup.expr.groupby('name').id.lead(offset=1, default=-1,
                                          sort=['id', 'fid'], ascending=[False, False]).rename('id3'),
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [4, 3, 4],
        [2, 0, -1],
        [4, 0, 3],
        [3, 0, -1],
        [3, 0, 3]
    ]
    assert sorted(expected) == sorted(result)


def test_window_rewrite(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr[setup.expr.id - setup.expr.id.mean() < 10][
        [lambda x: x.id - x.id.max()]][[lambda x: x.id - x.id.min()]][lambda x: x.id - x.id.std() > 0]

    res = setup.engine.execute(expr)
    result = get_result(res)

    import pandas as pd
    df = pd.DataFrame(data, columns=setup.schema.names)
    expected = df.id - df.id.max()
    expected = expected - expected.min()
    expected = list(expected[expected - expected.std() > 0])

    assert expected == [it[0] for it in result]


def test_reduction(odps, setup):
    data = setup.gen_data(rows=5, value_range=(-100, 100), nullable_field='name')

    import pandas as pd
    df = pd.DataFrame(data, columns=setup.schema.names)

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
        (lambda s: df.id.mean(), setup.expr.id.mean()),
        (lambda s: len(df), setup.expr.count()),
        (lambda s: df.id.var(ddof=0), setup.expr.id.var(ddof=0)),
        (lambda s: df.id.std(ddof=0), setup.expr.id.std(ddof=0)),
        (lambda s: df.id.median(), setup.expr.id.median()),
        (lambda s: quantile(df.id, 0.3), setup.expr.id.quantile(0.3)),
        (lambda s: quantile(df.id, [0.3, 0.6]), setup.expr.id.quantile([0.3, 0.6])),
        (lambda s: df.id.sum(), setup.expr.id.sum()),
        (lambda s: df.id.unique().sum(), setup.expr.id.unique().sum()),
        (lambda s: (df.id ** 3).mean(), setup.expr.id.moment(3)),
        (lambda s: df.id.var(ddof=0), setup.expr.id.moment(2, central=True)),
        (lambda s: df.id.skew(), setup.expr.id.skew()),
        (lambda s: df.id.kurtosis(), setup.expr.id.kurtosis()),
        (lambda s: df.id.min(), setup.expr.id.min()),
        (lambda s: df.id.max(), setup.expr.id.max()),
        (lambda s: df.isMale.min(), setup.expr.isMale.min()),
        (lambda s: len(filter_none(df.name)), setup.expr.name.count()),
        (lambda s: filter_none(df.name).max(), setup.expr.name.max()),
        (lambda s: df.birth.max(), setup.expr.birth.max()),
        (lambda s: filter_none(df.name).sum(), setup.expr.name.sum()),
        (lambda s: df.isMale.sum(), setup.expr.isMale.sum()),
        (lambda s: df.isMale.any(), setup.expr.isMale.any()),
        (lambda s: df.isMale.all(), setup.expr.isMale.all()),
        (lambda s: filter_none(df.name).nunique(), setup.expr.name.nunique()),
        (lambda s: len(filter_none(df.name).str.cat(sep='|').split('|')),
         setup.expr.name.cat(sep='|').map(lambda x: len(x.split('|')), rtype='int')),
        (lambda s: df.id.mean(), setup.expr.id.agg(Agg, rtype='float')),
        (lambda s: df.id.count(), setup.expr.id.count()),
    ]

    fields = [it[1].rename('f' + str(i)) for i, it in enumerate(methods_to_fields)]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    df = pd.DataFrame(data, columns=setup.schema.names)

    for i, it in enumerate(methods_to_fields):
        method = it[0]

        first = method(df)
        second = [it[i] for it in result][0]
        if isinstance(first, float):
            assert pytest.approx(first) == second
        elif isinstance(first, list):
            assert approx_list(first) == second
        else:
            assert first == second

    assert setup.engine.execute(setup.expr.id.sum() + 1) == sum(it[1] for it in data) + 1

    expr = setup.expr['id', 'fid'].apply(Agg, types=['float'] * 2)

    expected = [[df.id.mean()], [df.fid.mean()]]

    res = setup.engine.execute(expr)
    result = get_result(res)

    for first, second in zip(expected, result):
        first = first[0]
        second = second[0]

        if isinstance(first, float):
            assert pytest.approx(first) == second
        elif isinstance(first, list):
            assert approx_list(first) == second
        else:
            assert first == second


def test_user_defined_aggregators(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

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

    expr = setup.expr.id.agg(Aggregator)
    expected = float(16) / 5

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert pytest.approx(expected) == result

    expr = setup.expr.id.unique().agg(Aggregator)
    expected = float(9) / 3

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert pytest.approx(expected) == result

    expr = setup.expr.groupby(Scalar('const').rename('s')).id.agg(Aggregator)
    expected = float(16) / 5

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert pytest.approx(expected) == result[0][0]

    expr = setup.expr.groupby('name').agg(setup.expr.id.agg(Aggregator))

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', float(14) / 4],
        ['name2', 2]
    ]
    for expect_r, actual_r in zip(expected, result):
        assert expect_r[0] == actual_r[0]
        assert pytest.approx(expect_r[1]) == actual_r[1]

    expr = setup.expr[
        (setup.expr['name'] + ',' + setup.expr['id'].astype('string')).rename('name'),
        setup.expr.id
    ]
    expr = expr.groupby('name').agg(expr.id.agg(Aggregator).rename('id'))

    expected = [
        ['name1,4', 4],
        ['name1,3', 3],
        ['name2,2', 2],
    ]
    res = setup.engine.execute(expr)
    result = get_result(res)

    assert sorted(result) == sorted(expected)

    expr = setup.expr[setup.expr.name, Scalar(1).rename('id')]
    expr = expr.groupby('name').agg(expr.id.sum())

    expected = [
        ['name1', 4],
        ['name2', 1]
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert expected == result

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

    expr = agg([setup.expr['fid'], setup.expr['id']], Aggregator).rename('agg')

    expected = sum(r[2] for r in data) / sum(r[1] for r in data)
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert pytest.approx(expected) == result


def test_join(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]

    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                [types.string, types.int64, types.int64])

    setup.gen_data(data=data)

    data2 = [
        ['name1', 4, -1],
        ['name2', 1, -2]
    ]

    import pandas as pd
    expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                           _schema=schema2)

    expr = setup.expr.join(expr2).join(expr2)['name', 'id2']

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert len(result) == 5
    expected = [
        [to_text('name1'), 4],
        [to_text('name2'), 1]
    ]
    assert all(it in expected for it in result) is True

    expr = setup.expr.join(expr2, on=['name', ('id', 'id2')])[setup.expr.name, expr2.id2]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert len(result) == 2
    expected = [to_text('name1'), 4]
    assert all(it == expected for it in result) is True

    expr = setup.expr.join(expr2, on=['name', expr2.id2 == setup.expr.id])[setup.expr.name, expr2.id2]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert len(result) == 2
    expected = [to_text('name1'), 4]
    assert all(it == expected for it in result) is True

    expr = setup.expr.left_join(expr2, on=['name', ('id', 'id2')])[setup.expr.name, expr2.id2]
    res = setup.engine.execute(expr)
    result = get_result(res)
    expected = [
        ['name1', 4],
        ['name2', None],
        ['name1', 4],
        ['name1', None],
        ['name1', None]
    ]
    assert len(result) == 5
    assert all(it in expected for it in result) is True

    expr = setup.expr.right_join(expr2, on=['name', ('id', 'id2')])[setup.expr.name, expr2.id2]
    res = setup.engine.execute(expr)
    result = get_result(res)
    expected = [
        ['name1', 4],
        ['name1', 4],
        [None, 1],
    ]
    assert len(result) == 3
    assert all(it in expected for it in result) is True

    expr = setup.expr.outer_join(expr2, on=['name', ('id', 'id2')])[setup.expr.name, expr2.id2]
    res = setup.engine.execute(expr)
    result = get_result(res)
    expected = [
        ['name1', 4],
        ['name1', 4],
        ['name2', None],
        ['name1', None],
        ['name1', None],
        [None, 1],
    ]
    assert len(result) == 6
    assert all(it in expected for it in result) is True


def test_join_aggregation(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.join(setup.expr.view(), on=['name', 'id'])[
        lambda x: x.count(), setup.expr.id.sum()]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [[9, 30]] == result


def test_union(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]

    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                [types.string, types.int64, types.int64])

    setup.gen_data(data=data)

    data2 = [
        ['name3', 5, -1],
        ['name4', 6, -2]
    ]

    import pandas as pd
    expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                           _schema=schema2)

    expr = setup.expr['name', 'id'].distinct().union(expr2[expr2.id2.rename('id'), 'name'])

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 4],
        ['name1', 3],
        ['name2', 2],
        ['name3', 5],
        ['name4', 6]
    ]

    result = sorted(result)
    expected = sorted(expected)

    assert len(result) == len(expected)
    for e, r in zip(result, expected):
        assert [to_text(t) for t in e] == [to_text(t) for t in r]


def test_concat(odps, setup):
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
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 4, 5.3, 7.4, 1.5],
        ['name2', 2, 3.5, 2.3, 1.7],
        ['name1', 4, 4.2, 9.8, 5.4],
        ['name1', 3, 2.2, 1.9, 2.2],
        ['name1', 3, 4.1, 7.1, 6.2]
    ]

    result = sorted(result)
    expected = sorted(expected)

    assert len(result) == len(expected)
    for e, r in zip(result, expected):
        assert [to_text(t) for t in e] == [to_text(t) for t in r]


def test_scale_value(odps, setup):
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

    res = setup.engine.execute(expr)
    result = get_result(res)

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
        assert len(first) == len(second)
        for it1, it2 in zip(first, second):
            assert pytest.approx(it1) == it2

    # test grouped min_max_scale
    expr = expr_input.min_max_scale(columns=['fid'], group=['name'])

    res = setup.engine.execute(expr)
    result = get_result(res)

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
        assert len(first) == len(second)
        for it1, it2 in zip(first, second):
            assert pytest.approx(it1) == it2

    # test simple std_scale
    expr = expr_input.std_scale(columns=['fid'])

    res = setup.engine.execute(expr)
    result = get_result(res)

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
        assert len(first) == len(second)
        for it1, it2 in zip(first, second):
            assert pytest.approx(it1) == it2

    # test grouped std_scale
    expr = expr_input.std_scale(columns=['fid'], group=['name'])

    res = setup.engine.execute(expr)
    result = get_result(res)

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
        assert len(first) == len(second)
        for it1, it2 in zip(first, second):
            assert pytest.approx(it1) == it2


def test_extract_kv(odps, setup):
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

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected_cols = ['name', 'kv_k1', 'kv_k2', 'kv_k3', 'kv_k5', 'kv_k7', 'kv_k9']
    expected = [
        ['name1', 1.0, 3.0, 0, 10.0, 0, 0],
        ['name1', 0, 3.0, 5.1, 0, 0, 0],
        ['name1', 7.1, 0, 0, 0, 8.2, 0],
        ['name2', 0, 1.2, 1.5, 0, 0, 0],
        ['name2', 0, 1.0, 0, 0, 0, 1.1],
    ]

    assert [c.name for c in res.columns] == expected_cols
    assert result == expected


def test_make_kv(odps, setup):
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

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 'k1=1.0,k2=3.0,k5=10.0'],
        ['name1', 'k2=3.0,k3=5.1'],
        ['name1', 'k1=7.1,k7=8.2'],
        ['name2', 'k2=1.2,k3=1.5'],
        ['name2', 'k2=1.0,k9=1.1'],
    ]

    assert result == expected


def test_hllc(odps, setup):
    names = [randint(0, 100000) for _ in xrange(100000)]
    data = [[n] + [None] * 5 for n in names]

    setup.gen_data(data=data)

    expr = setup.expr.name.hll_count()

    res = setup.engine.execute(expr)
    result = get_result(res)

    expect = len(set(names))
    assert pytest.approx(expect, abs=result * 0.1) == result


def test_bloom_filter(odps, setup):
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

    setup.gen_data(data=data)

    schema2 = TableSchema.from_lists(['name', ], [types.string])

    import pandas as pd
    expr2 = CollectionExpr(_source_data=pd.DataFrame(data2, columns=schema2.names),
                           _schema=schema2)

    expr = setup.expr.bloom_filter('name', expr2[:1].name, capacity=10)

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert all(r[0] != 'name2' for r in result) is True


def test_persist(odps, setup):
    data = [
        ['name1', 4, 5.3, True, Decimal('3.14'), datetime(1999, 5, 25, 3, 10)],
        ['name2', 2, 3.5, False, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)
    odps_engine = setup.odps_engine

    def simple_persist_test(table_name):
        odps.delete_table(table_name, if_exists=True)
        try:
            with pytest.raises(ODPSError):
                setup.engine.persist(setup.expr, table_name, create_table=False)

            df = setup.engine.persist(setup.expr, table_name)

            res = df.to_pandas()
            result = get_result(res)
            assert len(result) == 5
            assert data == result

            with pytest.raises(ValueError):
                setup.engine.persist(setup.expr, table_name, create_partition=True)
            with pytest.raises(ValueError):
                setup.engine.persist(setup.expr, table_name, drop_partition=True)
        finally:
            odps.delete_table(table_name, if_exists=True)

    def persist_existing_test(table_name):
        odps.delete_table(table_name, if_exists=True)
        try:
            odps.create_table(
                table_name,
                'name string, fid double, id bigint, isMale boolean, scale decimal, birth datetime',
                lifecycle=1,
            )

            expr = setup.expr[setup.expr, Scalar(1).rename('name2')]
            with pytest.raises(CompileError):
                setup.engine.persist(expr, table_name)

            expr = setup.expr['name', 'fid', setup.expr.id.astype('int32'), 'isMale', 'scale', 'birth']
            df = setup.engine.persist(expr, table_name)

            res = df.to_pandas()
            result = get_result(res)
            assert len(result) == 5
            assert data == [[r[0], r[2], r[1], r[3], r[4], r[5]] for r in result]
        finally:
            odps.delete_table(table_name, if_exists=True)

    def persist_with_partition_test(table_name):
        odps.delete_table(table_name, if_exists=True)
        try:
            df = setup.engine.persist(setup.expr, table_name, partition={'ds': 'today'})

            res = odps_engine.execute(df)
            result = get_result(res)
            assert len(result) == 5

            df = setup.engine.persist(setup.expr, table_name, partition={'ds': 'today'}, overwrite=True)

            res = odps_engine.execute(df)
            result = get_result(res)
            assert len(result) == 5
        finally:
            odps.delete_table(table_name, if_exists=True)

    def persist_with_create_partition_test(table_name):
        odps.delete_table(table_name, if_exists=True)
        try:
            schema = TableSchema.from_lists(setup.schema.names, setup.schema.types, ['ds'], ['string'])
            odps.create_table(table_name, schema)
            df = setup.engine.persist(setup.expr, table_name, partition='ds=today', create_partition=True)

            res = odps_engine.execute(df)
            result = get_result(res)
            assert len(result) == 5
            assert data == [d[:-1] for d in result]

            df2 = setup.engine.persist(
                setup.expr[setup.expr.id.astype('float'), 'name'], table_name,
                partition='ds=today2', create_partition=True, cast=True
            )

            res = odps_engine.execute(df2)
            result = get_result(res)
            assert len(result) == 5
            assert [d[:2] + [None] * (len(d) - 2) for d in data] == [d[:-1] for d in result]
        finally:
            odps.delete_table(table_name, if_exists=True)

    def persist_with_create_multi_part_test(table_name):
        odps.delete_table(table_name, if_exists=True)
        try:
            schema = TableSchema.from_lists(setup.schema.names, setup.schema.types, ['ds', 'hh'], ['string', 'string'])
            table = odps.create_table(table_name, schema)

            with pytest.raises(ValueError):
                setup.engine.persist(setup.expr, table_name, partition='ds=today', create_partition=True)

            setup.engine.persist(setup.expr, table, partition=OrderedDict([('hh', 'now'), ('ds', 'today')]))
            assert table.exist_partition('ds=today,hh=now') is True
        finally:
            odps.delete_table(table_name, if_exists=True)

    def persist_with_dyna_part_test(table_name):
        for overwrite in [False, True]:
            odps.delete_table(table_name, if_exists=True)
            try:
                setup.engine.persist(
                    setup.expr, table_name, partitions=['name'], overwrite=overwrite
                )

                t = odps.get_table(table_name)
                assert 2 == len(list(t.partitions))
                with t.open_reader(partition='name=name1', reopen=True) as r:
                    assert 4 == r.count
                with t.open_reader(partition='name=name2', reopen=True) as r:
                    assert 1 == r.count
            finally:
                odps.delete_table(table_name, if_exists=True)

    sub_tests = [
        simple_persist_test,
        persist_existing_test,
        persist_with_partition_test,
        persist_with_create_partition_test,
        persist_with_create_multi_part_test,
        persist_with_dyna_part_test,
    ]
    base_table_name = tn('pyodps_test_pd_engine_persist_table')
    try:
        options.tunnel.write_row_batch_size = 1
        run_sub_tests_in_parallel(
            10,
            [
                functools.partial(sub_test, base_table_name + "_%d" % idx)
                for idx, sub_test in enumerate(sub_tests)
            ]
        )
    finally:
        options.tunnel.write_row_batch_size = 1024


def test_append_id(odps, setup):
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
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [0, 'name1', 4, 5.3],
        [1, 'name2', 2, 3.5],
        [2, 'name1', 4, 4.2],
        [3, 'name1', 3, 2.2],
        [4, 'name1', 3, 4.1],
    ]
    assert result == expected


def test_split(odps, setup):
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
    res1 = setup.engine.execute(expr1)
    result1 = get_result(res1)
    res2 = setup.engine.execute(expr2)
    result2 = get_result(res2)

    merged = sorted(result1 + result2, key=lambda r: r[0])
    assert data == merged


def test_collection_na(odps, setup):
    import pandas as pd
    import numpy as np

    from .....compat import reduce

    data = [
        [0, 'name1', 1.0, None, 3.0, 4.0],
        [1, 'name1', 2.0, None, None, 1.0],
        [2, 'name1', 3.0, 4.0, 1.0, None],
        [3, 'name1', float('nan'), 1.0, 2.0, 3.0],
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

    ures = setup.engine.execute(uexpr)
    uresult = get_result(ures)

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
    assert uresult == expected


def test_drop(odps, setup):
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
    res = setup.engine.execute(expr_result)
    result = get_result(res)

    expected = [['name2', 3, 1.0], ['name3', 1, 1.2], ['name3', 3, 1.2]]
    assert sorted(result) == sorted(expected)

    expr_result = expr1.drop(expr2, columns='name')
    res = setup.engine.execute(expr_result)
    result = get_result(res)

    expected = [['name3', 1, 1.2], ['name3', 3, 1.2]]
    assert sorted(result) == sorted(expected)

    expr_result = expr1.drop(['id'], axis=1)
    res = setup.engine.execute(expr_result)
    result = get_result(res)

    expected = [
        ['name1', 3.0], ['name1', 3.0], ['name1', 2.5],
        ['name2', 1.2], ['name2', 1.0],
        ['name3', 1.2], ['name3', 1.2],
    ]
    assert sorted(result) == sorted(expected)

    expr_result = expr1.drop(expr2[['id']], axis=1)
    res = setup.engine.execute(expr_result)
    result = get_result(res)

    expected = [
        ['name1', 3.0], ['name1', 3.0], ['name1', 2.5],
        ['name2', 1.2], ['name2', 1.0],
        ['name3', 1.2], ['name3', 1.2],
    ]
    assert sorted(result) == sorted(expected)


def test_except_intersect(odps, setup):
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
    res = setup.engine.execute(expr_result)
    result = get_result(res)

    expected = [['name1', 2], ['name2', 3], ['name3', 1], ['name3', 3]]
    assert sorted(result) == sorted(expected)

    expr_result = expr1.setdiff(expr2, distinct=True)
    res = setup.engine.execute(expr_result)
    result = get_result(res)

    expected = [['name2', 3], ['name3', 1], ['name3', 3]]
    assert sorted(result) == sorted(expected)

    expr_result = expr1.intersect(expr2)
    res = setup.engine.execute(expr_result)
    result = get_result(res)

    expected = [['name1', 1], ['name1', 2], ['name1', 2], ['name2', 1]]
    assert sorted(result) == sorted(expected)

    expr_result = expr1.intersect(expr2, distinct=True)
    res = setup.engine.execute(expr_result)
    result = get_result(res)

    expected = [['name1', 1], ['name1', 2], ['name2', 1]]
    assert sorted(result) == sorted(expected)


def test_filter_order(odps, setup):
    import pandas as pd

    schema = TableSchema.from_lists(['divided', 'divisor'], [types.int64, types.int64])
    pd_df = pd.DataFrame([[2, 0], [1, 1], [1, 2], [5, 1], [5, 0]], columns=schema.names)
    df = CollectionExpr(_source_data=pd_df, _schema=schema)
    fdf = df[df.divisor > 0]
    ddf = fdf[(fdf.divided / fdf.divisor).rename('result'), ]
    expr = ddf[ddf.result > 1]

    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == [[5, ]]


def test_lateral_view(odps, setup):
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
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

    expected = [
        [5.3, 'name1_0', 0],
        [3.5, 'name2_0', 0],
        [4.2, 'name1_0', 0],
        [2.2, 'name1_0', 0], [2.2, 'name1_1', 2],
        [4.1, 'name1_0', 0], [4.1, 'name1_1', 2],
    ]

    expr = expr_in[expr_in.fid, expr_in['name', expr_in.id % 2 + 1].apply(mapper, axis=1)]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

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
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

    @output(['bin_id'], ['string'])
    def mapper3(row):
        for idx in range(row.ren_id % 2 + 1):
            yield str(idx)

    expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                   expr_in['name', 'id'].apply(mapper, axis=1),
                                   expr_in[expr_in.id.rename('ren_id'), ].apply(mapper3, axis=1)]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

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
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert sorted(result) == sorted(expected)

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
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

    expected = [
        [5, 3.5, 'name2_0', 0, None], [5, 3.5, 'name2_1', 2, None],
        [5, 2.2, 'name1_0', 0, '0'], [5, 2.2, 'name1_1', 3, '0'],
        [5, 2.2, 'name1_2', 6, '0'], [5, 4.1, 'name1_0', 0, '0'],
        [5, 4.1, 'name1_1', 3, '0'], [5, 4.1, 'name1_2', 6, '0']
    ]

    expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                   expr_in['name', 'id'].apply(mapper, axis=1),
                                   expr_in['id', ].apply(mapper3, axis=1, keep_nulls=True)]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected


def test_composites(odps, setup):
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
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

    expected = [
        ['name1', 123.2], ['name1', 567.1],
        ['name2', 512.1], ['name2', 711.2],
        ['name1', 115.4], ['name1', 312.1],
        ['name1', 923.2], ['name1', 456.1],
    ]

    expr = expr_in[expr_in.name, expr_in.detail.values().explode()]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

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
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

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
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

    expected = [
        ['name1', [4, 4, 3, 3]],
        ['name2', [2]]
    ]

    expr = expr_in.groupby(expr_in.name).agg(agg_grades=expr_in.grade.tolist())
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

    expected = [
        ['name1', [3, 4]],
        ['name2', [2]]
    ]

    expr = expr_in.groupby(expr_in.name).agg(agg_grades=expr_in.grade.tolist(unique=True).sort())
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

    expected = [[['name1', 'name2', 'name1', 'name1', 'name1'], [4, 2, 4, 3, 3]]]

    expr = expr_in['name', 'grade'].tolist()
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

    expected = [[['name1', 'name2'], [2, 3, 4]]]

    expr = expr_in['name', 'grade'].tolist(unique=True)
    expr = expr[tuple(f.sort() for f in expr.columns)]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected


def test_string_splits(odps, setup):
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
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected

    expected = [
        [4, {'name1': 'a', 'name3': '5'}],
        [2, {'name2': '4', 'name7': '1'}],
        [4, {'name1': '1'}],
        [3, {'name1': '4', 'name5': '6', 'name4': '1'}],
        [3, {'name1': '2', 'name10': '1'}],
    ]

    expr = expr_in[expr_in.id, expr_in.name.todict(kv_delim=':')]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == expected


def test_df_reference_warnings(setup):
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

    def df_ref_func(x):
        return x + type(expr_in).__name__

    expr = expr_in[expr_in.name.map(df_ref_func), expr_in.id]
    with pytest.warns(RuntimeWarning) as warn_info:
        setup.engine.execute(expr)
    assert any(info for info in warn_info if "df_ref_func" in str(info.message))

    class NestedCls(object):
        def nested_fun(self, x):
            return x + ".abcd"

    def df_nested_fun(x):
        return NestedCls().nested_fun(x)

    # make sure the class is in anotheer module
    NestedCls.__module__ = "odps.df"

    expr = expr_in[expr_in.name.map(df_nested_fun), expr_in.id]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        setup.engine.execute(expr)
    assert 0 == len([
        info for info in record
        if issubclass(info.category, RuntimeWarning) and "remotely" in str(info.message)
    ])
