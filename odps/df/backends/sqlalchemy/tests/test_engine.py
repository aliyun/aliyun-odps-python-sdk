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

import math
import itertools
import time
from collections import namedtuple
from datetime import datetime, timedelta
from functools import partial

import pytest

from .....compat import six, futures
from .....models import TableSchema
from .....tests.core import get_result, approx_list
from .....utils import to_text
from ....expr.expressions import CollectionExpr
from ....types import validate_data_type
from .... import Scalar, output_names, output_types, output, day, millisecond, agg
from ...tests.core import tn, NumGenerators
from ...context import context
from ...odpssql.types import df_schema_to_odps_schema
from ..engine import SQLAlchemyEngine, _engine_to_connections
from ..types import df_schema_to_sqlalchemy_columns

pytestmark = pytest.mark.skip


@pytest.fixture
def setup(odps):
    def create_table_and_insert_data(table_name, df_schema, data, drop_first=True):
        import sqlalchemy

        columns = df_schema_to_sqlalchemy_columns(df_schema, engine=sql_engine)
        t = sqlalchemy.Table(table_name, metadata, *columns)

        conn.execute('DROP TABLE IF EXISTS %s' % table_name)
        t.create()

        conn.execute(t.insert(), [
            dict((n, v) for n, v in zip(df_schema.names, d)) for d in data
        ])

        return t

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
                j = schema._name_indexes[nullable_field.lower()]
                for i, l in enumerate(data):
                    if i % 2 == 0:
                        data[i][j] = None

        conn.execute(table.insert(), [
            dict((n, v) for n, v in zip(schema.names, d)) for d in data])
        return data

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    pd_schema = TableSchema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                               datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
    df_schema = pd_schema
    schema = df_schema_to_odps_schema(pd_schema)
    df = None
    expr = None

    engine = SQLAlchemyEngine()

    import sqlalchemy
    from sqlalchemy import create_engine

    sql_engine = sa_engine = create_engine('postgres://localhost/pyodps')
    # setup.sql_engine = engine = create_engine('mysql://localhost/pyodps')
    # setup.sql_engine = engine = create_engine('sqlite://')
    conn = sa_engine.connect()

    metadata = metadata = sqlalchemy.MetaData(bind=sa_engine)
    columns = df_schema_to_sqlalchemy_columns(df_schema, engine=sql_engine)
    t = sqlalchemy.Table('pyodps_test_data', metadata, *columns)

    metadata.create_all()

    table = t
    expr = CollectionExpr(_source_data=table, _schema=df_schema)

    class FakeBar(object):
        def update(self, *args, **kwargs):
            pass
    faked_bar = FakeBar()

    nt = namedtuple("NT", "df_schema, schema, df, expr, engine, sql_engine, conn, "
                          "metadata, table, faked_bar, gen_data, create_table_and_insert_data")
    try:
        yield nt(df_schema, schema, df, expr, engine, sql_engine,
                 conn, metadata, table, faked_bar, gen_data,
                 create_table_and_insert_data)
    finally:
        [conn.close() for conn in _engine_to_connections.values()]
        table.drop()
        conn.close()


def test_async(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr.id.sum()

    future = setup.engine.execute(expr, async_=True, priority=4)
    assert future.done() is False
    res = future.result()

    assert sum(it[1] for it in data) == res


def test_cache(odps, setup):
    import sqlalchemy

    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr[setup.expr.id < 10].cache()
    cnt = expr.count()

    dag = setup.engine.compile(expr)
    assert len(dag.nodes()) == 2

    res = setup.engine.execute(cnt)
    assert len([it for it in data if it[1] < 10]) == res
    assert context.is_cached(expr) is True

    table = context.get_cached(expr)
    assert isinstance(table, sqlalchemy.Table)


def test_batch(odps, setup):
    if setup.sql_engine.name == 'mysql':
        # TODO: mysqldb is not thread-safe, skip first
        return

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

    # test async and timeout
    expr = setup.expr[setup.expr.id < 10]
    expr1 = expr.id.sum()
    expr2 = expr.id.mean()

    fs = setup.engine.execute([expr, expr1, expr2], n_parallel=2, async_=True, timeout=1)
    assert len(fs) == 3

    assert fs[1].result() == expect1
    assert pytest.approx(fs[2].result()) == expect2
    assert context.is_cached(expr) is True


def test_element(odps, setup):
    data = setup.gen_data(5, nullable_field='name')

    fields = [
        setup.expr.name.isnull().rename('name1'),
        setup.expr.name.notnull().rename('name2'),
        setup.expr.name.fillna('test').rename('name3'),
        setup.expr.id.isin([1, 2, 3]).rename('id1'),
        setup.expr.id.isin(setup.expr.fid.astype('int')).rename('id2'),
        setup.expr.id.notin([1, 2, 3]).rename('id3'),
        setup.expr.id.notin(setup.expr.fid.astype('int')).rename('id4'),
        setup.expr.id.between(setup.expr.fid, 3).rename('id5'),
        setup.expr.name.fillna('test').switch('test', 'test' + setup.expr.name.fillna('test'),
                                             'test2', 'test2' + setup.expr.name.fillna('test'),
                                             default=setup.expr.name).rename('name4'),
        setup.expr.name.fillna('test').switch('test', 1, 'test2', 2).rename('name5'),
        setup.expr.id.cut(
            [100, 200, 300],
            labels=['xsmall', 'small', 'large', 'xlarge'],
            include_under=True,
            include_over=True,
        ).rename('id6')
    ]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert len(data) == len(result)

    assert len([it for it in data if it[0] is None]) == len([it[0] for it in result if it[0]])

    assert len([it[0] for it in data if it[0] is not None]) == len([it[1] for it in result if it[1]])

    assert [(it[0] if it[0] is not None else 'test') for it in data] == [it[2] for it in result]

    assert [(it[1] in (1, 2, 3)) for it in data] == [it[3] for it in result]

    fids = [int(it[2]) for it in data]
    assert [(it[1] in fids) for it in data] == [it[4] for it in result]

    assert [(it[1] not in (1, 2, 3)) for it in data] == [it[5] for it in result]

    assert [(it[1] not in fids) for it in data] == [it[6] for it in result]

    assert [(it[2] <= it[1] <= 3) for it in data] == [it[7] for it in result]

    assert [to_text('testtest' if it[0] is None else it[0]) for it in data] == [to_text(it[8]) for it in result]

    assert [to_text(1 if it[0] is None else None) for it in data] == [to_text(it[9]) for it in result]

    def get_val(val):
        if val <= 100:
            return 'xsmall'
        elif 100 < val <= 200:
            return 'small'
        elif 200 < val <= 300:
            return 'large'
        else:
            return 'xlarge'
    assert [to_text(get_val(it[1])) for it in data] == [to_text(it[10]) for it in result]


def test_arithmetic(odps, setup):
    data = setup.gen_data(5, value_range=(-1000, 1000))

    fields = [
        (setup.expr.id + 1).rename('id1'),
        (setup.expr.fid - 1).rename('fid1'),
        (setup.expr.scale * 2).rename('scale1'),
        (setup.expr.scale + setup.expr.id).rename('scale2'),
        (setup.expr.id / 2).rename('id2'),
        (setup.expr.id ** 2).rename('id3'),
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

    assert approx_list([it[2] - 1 for it in data], abs=.001) \
        == [it[1] for it in result]

    assert [it[4] * 2 for it in data] == [it[2] for it in result]

    assert [it[4] + it[1] for it in data] == [it[3] for it in result]

    assert approx_list([float(it[1]) / 2 for it in data], abs=.001) \
        == [it[4] for it in result]

    assert [int(it[1] ** 2) for it in data] == [it[5] for it in result]

    assert [abs(it[1]) for it in data] == [it[6] for it in result]

    assert [~it[1] for it in data] == [it[7] for it in result]

    assert approx_list([-it[2] for it in data], abs=.001) \
        == [it[8] for it in result]

    assert [not it[3] for it in data] == [it[9] for it in result]

    assert [it[1] // 2 for it in data] == [it[11] for it in result]

    assert [it[5] + timedelta(days=1) for it in data] == [it[12].replace(tzinfo=None) for it in result]

    assert [10] * len(data) == [it[13] for it in result]

    assert [it[1] % 2 for it in data] == [it[14] for it in result]


def test_math(odps, setup):
    # TODO: test sinh, cosh..., and acosh, asinh...
    data = setup.gen_data(5, value_range=(1, 90))

    if hasattr(math, 'expm1'):
        expm1 = math.expm1
    else:
        expm1 = lambda x: 2 * math.exp(x / 2.0) * math.sinh(x / 2.0)

    methods_to_fields = [
        (math.sin, setup.expr.id.sin()),
        (math.cos, setup.expr.id.cos()),
        (math.tan, setup.expr.id.tan()),
        (math.log, setup.expr.id.log()),
        (lambda v: math.log(v, 2), setup.expr.id.log2()),
        (math.log10, setup.expr.id.log10()),
        (math.log1p, setup.expr.id.log1p()),
        (math.exp, setup.expr.id.exp()),
        (expm1, setup.expr.id.expm1()),
        (math.atan, setup.expr.id.arctan()),
        (math.sqrt, setup.expr.id.sqrt()),
        (abs, setup.expr.id.abs()),
        (math.ceil, setup.expr.id.ceil()),
        (math.floor, setup.expr.id.floor()),
        (math.trunc, setup.expr.id.trunc()),
        (round, setup.expr.id.round()),
        (lambda x: round(x, 2), setup.expr.id.round(2)),
    ]

    fields = [it[1].rename('id'+str(i)) for i, it in enumerate(methods_to_fields)]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    for i, it in enumerate(methods_to_fields):
        mt = it[0]

        def method(v):
            try:
                return mt(v)
            except ValueError:
                return float('nan')

        first = [method(it[1]) for it in data]
        second = [it[i] for it in result]
        assert len(first) == len(second)
        for it1, it2 in zip(first, second):
            not_valid = lambda x: \
                x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
            if not_valid(it1) and not_valid(it2):
                continue
            if isinstance(it1, float) and it1 > 1.0e15:
                scale = 0.1 ** (int(math.log10(it1)) - 15)
                assert pytest.approx(it1 * scale) == it2 * scale
            else:
                assert pytest.approx(it1) == it2


def test_string(odps, setup):
    data = setup.gen_data(5)

    methods_to_fields = [
        (lambda s: s.capitalize(), setup.expr.name.capitalize()),
        (lambda s: data[0][0] in s, setup.expr.name.contains(data[0][0], regex=False)),
        (lambda s: s[0] + '|' + str(s[1]), setup.expr.name.cat(setup.expr.id.astype('string'), sep='|')),
        (lambda s: s.endswith(data[0][0]), setup.expr.name.endswith(data[0][0])),
        (lambda s: s.startswith(data[0][0]), setup.expr.name.startswith(data[0][0])),
        (lambda s: s.replace(data[0][0], 'test'), setup.expr.name.replace(data[0][0], 'test', regex=False)),
        (lambda s: s[0], setup.expr.name.get(0)),
        (lambda s: len(s), setup.expr.name.len()),
        (lambda s: s.ljust(10), setup.expr.name.ljust(10)),
        (lambda s: s.ljust(20, '*'), setup.expr.name.ljust(20, fillchar='*')),
        (lambda s: s.rjust(10), setup.expr.name.rjust(10)),
        (lambda s: s.rjust(20, '*'), setup.expr.name.rjust(20, fillchar='*')),
        (lambda s: s * 4, setup.expr.name.repeat(4)),
        (lambda s: s[1:], setup.expr.name.slice(1)),
        (lambda s: s[1: 6], setup.expr.name.slice(1, 6)),
        (lambda s: s.title(), setup.expr.name.title()),
        (lambda s: s.rjust(20, '0'), setup.expr.name.zfill(20)),
    ]

    if setup.sql_engine.name == 'mysql':
        methods_to_fields = methods_to_fields[:-2] + methods_to_fields[-1:]

    fields = [it[1].rename('id'+str(i)) for i, it in enumerate(methods_to_fields)]

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

    def date_value(sel):
        if isinstance(sel, six.string_types):
            fun = lambda v: getattr(v, sel)
        else:
            fun = sel
        col_id = [idx for idx, col in enumerate(setup.schema.names) if col == 'birth'][0]
        return [fun(row[col_id]) for row in data]

    methods_to_fields = [
        (partial(date_value, 'year'), setup.expr.birth.year),
        (partial(date_value, 'month'), setup.expr.birth.month),
        (partial(date_value, 'day'), setup.expr.birth.day),
        (partial(date_value, 'hour'), setup.expr.birth.hour),
        (partial(date_value, 'minute'), setup.expr.birth.minute),
        (partial(date_value, 'second'), setup.expr.birth.second),
        (partial(date_value, lambda d: d.isocalendar()[1]), setup.expr.birth.weekofyear),
        (partial(date_value, lambda d: d.weekday()), setup.expr.birth.dayofweek),
        (partial(date_value, lambda d: d.weekday()), setup.expr.birth.weekday),
        (partial(date_value, lambda d: time.mktime(d.timetuple())), setup.expr.birth.unix_timestamp),
        (partial(date_value, lambda d: datetime.combine(d.date(), datetime.min.time())), setup.expr.birth.date),
    ]

    fields = [it[1].rename('birth'+str(i)) for i, it in enumerate(methods_to_fields)]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

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
        assert first == second


def test_groupby_aggregation(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.groupby(['name', 'id'])[lambda x: x.fid.min() * 2 < 8] \
        .agg(setup.expr.fid.max() + 1, new_id=setup.expr.id.sum())

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        ['name1', 3, 5.1, 6],
        ['name2', 2, 4.5, 2]
    ]

    result = sorted(result, key=lambda k: k[0])

    assert approx_list(expected, abs=.001) == result

    expr = setup.expr.name.value_counts()[:25]

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


def test_join_groupby(odps, setup):
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
    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                datatypes('string', 'int64', 'int64'))
    table_name = tn('pyodps_test_engine_table2')
    table2 = setup.create_table_and_insert_data(table_name, schema2, data2)
    expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

    setup.gen_data(data=data)

    expr = setup.expr.join(expr2, on='name')[setup.expr]
    expr = expr.groupby('id').agg(expr.fid.sum())

    res = setup.engine.execute(expr)
    result = get_result(res)

    id_idx = [idx for idx, col in enumerate(setup.expr.schema.names) if col == 'id'][0]
    fid_idx = [idx for idx, col in enumerate(setup.expr.schema.names) if col == 'fid'][0]
    expected = [[k, sum(v[fid_idx] for v in row)]
                for k, row in itertools.groupby(sorted(data, key=lambda r: r[id_idx]), lambda r: r[id_idx])]
    for it in zip(sorted(expected, key=lambda it: it[0]), sorted(result, key=lambda it: it[0])):
        assert pytest.approx(it[0][0]) == it[1][0]
        assert pytest.approx(it[0][1]) == it[1][1]


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


def test_window_function(odps, setup):
    if setup.sql_engine.name == 'mysql':
        # mysql doesn't support window function
        return

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
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [3], [3], [4], [4], [2]
    ]
    assert sorted(expected) == sorted(result)

    expr = setup.expr.groupby('name').mutate(id2=lambda x: x.id.cumcount(),
                                            fid2=lambda x: x.fid.cummin(sort='id'))

    res = setup.engine.execute(expr['name', 'id2', 'fid2'])
    result = get_result(res)

    expected = [
        ['name1', 4, 2.2],
        ['name1', 4, 2.2],
        ['name1', 4, 2.2],
        ['name1', 4, 2.2],
        ['name2', 1, 3.5],
    ]
    assert sorted(expected) == sorted(result)

    expr = setup.expr[
        setup.expr.id,
        setup.expr.groupby('name').rank('id'),
        setup.expr.groupby('name').dense_rank('fid', ascending=False),
        setup.expr.groupby('name').row_number(sort=['id', 'fid'], ascending=[True, False]),
        setup.expr.groupby('name').percent_rank('id'),
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [4, 3, 2, 3, float(2) / 3],
        [2, 1, 1, 1, 0.0],
        [4, 3, 3, 4, float(2) / 3],
        [3, 1, 4, 2, float(0) / 3],
        [3, 1, 1, 1, float(0) / 3]
    ]
    for l, r in zip(sorted(expected), sorted(result)):
        assert approx_list(l) == r

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
    assert sorted(expected) == sorted(result)


def test_join(odps, setup):
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
    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                datatypes('string', 'int64', 'int64'))
    table_name = tn('pyodps_test_engine_table2')
    table2 = setup.create_table_and_insert_data(table_name, schema2, data2)
    expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

    setup.gen_data(data=data)

    try:
        expr = setup.expr.join(expr2)['name', 'id2']

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

        if setup.sql_engine.name != 'mysql':
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

        grouped = setup.expr.groupby('name').agg(new_id=setup.expr.id.sum()).cache()
        setup.engine.execute(setup.expr.join(grouped, on='name'))

        if setup.sql_engine.name != 'mysql':
            expr = setup.expr.join(expr2, on=['name', ('id', 'id2')])[
                lambda x: x.groupby(Scalar(1)).sort('name').row_number(), ]
            setup.engine.execute(expr)
    finally:
        [conn.close() for conn in _engine_to_connections.values()]
        table2.drop()


def test_union(odps, setup):
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
    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                datatypes('string', 'int64', 'int64'))
    table_name = tn('pyodps_test_engine_table2')
    table2 = setup.create_table_and_insert_data(table_name, schema2, data2)
    expr2 = CollectionExpr(_source_data=table2, _schema=schema2)

    setup.gen_data(data=data)

    try:
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

    finally:
        [conn.close() for conn in _engine_to_connections.values()]
        table2.drop()
