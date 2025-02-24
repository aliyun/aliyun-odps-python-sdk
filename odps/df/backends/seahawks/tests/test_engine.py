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

import itertools
import math
import time
import uuid
from collections import namedtuple, OrderedDict
from datetime import datetime
from functools import partial

import pytest

from .....df.backends.tests.core import tn, NumGenerators
from .....models import TableSchema
from ..... import types
from .....compat import six
from .....errors import ODPSError
from .....tests.core import get_result, approx_list
from .....utils import to_text
from .... import Scalar
from ....types import validate_data_type, DynamicSchema
from ....expr.expressions import CollectionExpr
from ...odpssql.types import df_schema_to_odps_schema, odps_schema_to_df_schema
from ...context import context
from ...errors import CompileError
from ..engine import SeahawksEngine
from ..models import SeahawksTable

pytestmark = pytest.mark.skip


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
                j = schema._name_indexes[nullable_field.lower()]
                for i, l in enumerate(data):
                    if i % 2 == 0:
                        data[i][j] = None

        odps.write_table(table, 0, data)
        return data

    try:
        import sqlalchemy
    except ImportError:
        pytest.skip("Must install sqlalchemy to run the test")

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    pd_schema = TableSchema.from_lists(['name', 'id', 'fid', 'isMale', 'birth', 'scale'][:5],
                               datatypes('string', 'int64', 'float64', 'boolean', 'datetime', 'decimal')[:5])
    schema = df_schema_to_odps_schema(pd_schema)
    table_name = tn('pyodps_test_%s' % str(uuid.uuid4()).replace('-', '_'))
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(
        name=table_name, table_schema=schema
    )
    expr = CollectionExpr(_source_data=table, _schema=pd_schema)

    engine = SeahawksEngine(odps)

    class FakeBar(object):
        def update(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def status(self, *args, **kwargs):
            pass
    faked_bar = FakeBar()

    nt = namedtuple("NT", "schema, table, expr, engine, faked_bar, gen_data")
    try:
        yield nt(schema, table, expr, engine, faked_bar, gen_data)
    finally:
        table.drop()


def test_async(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr.id.sum()

    future = setup.engine.execute(expr, async_=True)
    assert future.done() is False
    res = future.result()

    assert sum(it[1] for it in data) == res


def test_cache(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr[setup.expr.id < 10].cache()
    cnt = expr.count()

    dag = setup.engine.compile(expr)
    assert len(dag.nodes()) == 2

    res = setup.engine.execute(cnt)
    assert len([it for it in data if it[1] < 10]) == res
    assert context.is_cached(expr) is True

    table = context.get_cached(expr)
    assert isinstance(table, SeahawksTable)


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

    # test async and timeout
    expr = setup.expr[setup.expr.id < 10]
    expr1 = expr.id.sum()
    expr2 = expr.id.mean()

    fs = setup.engine.execute([expr, expr1, expr2], n_parallel=2, async_=True, timeout=1)
    assert len(fs) == 3

    assert fs[1].result() == expect1
    assert pytest.approx(fs[2].result()) == expect2
    assert context.is_cached(expr) is True


def test_base(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

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

    expr = setup.expr.sort('id')[:5]
    res = setup.engine.execute(expr)
    result = get_result(res.values)
    assert approx_list(sorted(data, key=lambda it: it[1])[:5], abs=.001) == \
        [r[:-1] + [r[-1].replace(tzinfo=None)] for r in result]

    expr = setup.expr[:1].filter(lambda x: x.name == data[1][0])
    res = setup.engine.execute(expr)
    assert len(res) == 0


def test_chinese(odps, setup):
    data = [
        ['中文', 4, 5.3, None, None],
        ['\'中文2', 2, 3.5, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.filter(setup.expr.name == '中文')
    res = setup.engine.execute(expr)
    assert len(res) == 1

    expr = setup.expr.filter(setup.expr.name == '\'中文2')
    res = setup.engine.execute(expr)
    assert len(res) == 1

    expr = setup.expr.filter(setup.expr.name == u'中文')
    res = setup.engine.execute(expr)
    assert len(res) == 1


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
        setup.expr.id.cut([100, 200, 300],
                         labels=['xsmall', 'small', 'large', 'xlarge'],
                         include_under=True, include_over=True).rename('id6'),
        setup.expr.id.between(setup.expr.fid, 3, inclusive=False).rename('id7'),
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

    assert [(it[2] < it[1] < 3) for it in data] == [it[11] for it in result]


def test_arithmetic(odps, setup):
    data = setup.gen_data(5, value_range=(-1000, 1000))

    fields = [
        (setup.expr.id + 1).rename('id1'),
        (setup.expr.fid - 1).rename('fid1'),
        (setup.expr.id / 2).rename('id2'),
        (setup.expr.id ** 2).rename('id3'),
        abs(setup.expr.id).rename('id4'),
        (~setup.expr.id).rename('id5'),
        (-setup.expr.fid).rename('fid2'),
        (~setup.expr.isMale).rename('isMale1'),
        (-setup.expr.isMale).rename('isMale2'),
        (setup.expr.id // 2).rename('id6'),
        (setup.expr.id % 2).rename('id7'),
    ]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert len(data) == len(result)

    assert [it[1] + 1 for it in data] == [it[0] for it in result]

    assert approx_list([it[2] - 1 for it in data], abs=.001) \
        == [it[1] for it in result]

    assert approx_list([float(it[1]) / 2 for it in data], abs=.001) \
        == [it[2] for it in result]

    assert [int(it[1] ** 2) for it in data] == [it[3] for it in result]

    assert [abs(it[1]) for it in data] == [it[4] for it in result]

    assert [~it[1] for it in data] == [it[5] for it in result]

    assert approx_list([-it[2] for it in data], abs=.001) \
        == [it[6] for it in result]

    assert [not it[3] for it in data] == [it[7] for it in result]

    assert [it[1] // 2 for it in data] == [it[9] for it in result]

    assert [it[1] % 2 for it in data] == [it[10] for it in result]


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


def test_sort_distinct(odps, setup):
    data = [
        ['name1', 4, None, None, None],
        ['name2', 2, None, None, None],
        ['name1', 4, None, None, None],
        ['name1', 3, None, None, None],
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


def test_pivot_table(odps, setup):
    data = [
        ['name1', 1, 1.0, True, None],
        ['name1', 1, 5.0, True, None],
        ['name1', 2, 2.0, True, None],
        ['name2', 1, 3.0, False, None],
        ['name2', 3, 4.0, False, None]
    ]

    setup.gen_data(data=data)

    expr = setup.expr

    expr1 = expr.pivot_table(rows='name', values='fid')
    res = setup.engine.execute(expr1)
    result = get_result(res)

    expected = [
        ['name1', 8.0 / 3],
        ['name2', 3.5],
    ]
    assert approx_list(sorted(result)) == sorted(expected)

    expr2 = expr.pivot_table(rows='name', values='fid', aggfunc=['mean', 'sum'])
    res = setup.engine.execute(expr2)
    result = get_result(res)

    expected = [
        ['name1', 8.0 / 3, 8.0],
        ['name2', 3.5, 7.0],
    ]
    assert res.schema.names == ['name', 'fid_mean', 'fid_sum']
    assert approx_list(sorted(result)) == sorted(expected)

    expr5 = expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum'])
    expr6 = expr5['name1_fid_mean',
                  expr5.groupby(Scalar(1)).sort('name1_fid_mean').name1_fid_mean.astype('float').cumsum()]

    k = lambda x: list(0 if it is None else it for it in x)

    expected = [
        [2, 2], [3, 5], [None, 5]
    ]
    res = setup.engine.execute(expr6)
    result = get_result(res)
    assert sorted(result, key=k) == sorted(expected, key=k)

    expr3 = expr.pivot_table(rows='id', values='fid', columns='name', fill_value=0).distinct()
    res = setup.engine.execute(expr3)
    result = get_result(res)

    expected = [
        [2, 0, 2.0],
        [3, 4.0, 0],
        [1, 3.0, 3.0],
    ]

    assert res.schema.names == ['id', 'name2_fid_mean', 'name1_fid_mean']
    assert result == expected

    expr7 = expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum']).cache()
    assert len(setup.engine.execute(expr7)) == 3

    expr8 = setup.expr.pivot_table(rows='id', values='fid', columns='name')
    assert len(setup.engine.execute(expr8)) == 3
    assert not isinstance(expr8.schema, DynamicSchema)
    expr9 =(expr8['name1_fid_mean'] - expr8['name2_fid_mean']).rename('substract')
    assert len(setup.engine.execute(expr9)) == 3
    expr10 = expr8.distinct()
    assert len(setup.engine.execute(expr10)) == 3


def test_groupby_aggregation(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
    ]
    setup.gen_data(data=data)

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

    assert sorted([it[1:] for it in expected]) == sorted(result)

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


def test_projection_groupby_filter(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
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
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
    ]

    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                [types.string, types.bigint, types.bigint])

    table_name = tn('pyodps_test_engine_table2')
    odps.delete_table(table_name, if_exists=True)
    table2 = odps.create_table(name=table_name, table_schema=schema2)
    expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

    setup.gen_data(data=data)

    data2 = [
        ['name1', 4, -1],
        ['name2', 1, -2]
    ]

    odps.write_table(table2, 0, data2)

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
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
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
    data = [
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 6.1, None, None],
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


def test_window_rewrite(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr[setup.expr.id - setup.expr.id.mean() < 10][
        [lambda x: x.id - x.id.max()]][[lambda x: x.id - x.id.min()]][lambda x: x.id - x.id.std() > 0]

    res = setup.engine.execute(expr)
    result = get_result(res)

    id_idx = [idx for idx, col in enumerate(setup.expr.schema.names) if col == 'id'][0]
    expected = [r[id_idx] for r in data]
    maxv = max(expected)
    expected = [v - maxv for v in expected]
    minv = min(expected)
    expected = [v - minv for v in expected]

    meanv = sum(expected) * 1.0 / len(expected)
    meanv2 = sum([v ** 2 for v in expected]) * 1.0 / len(expected)
    std = math.sqrt(meanv2 - meanv ** 2)
    expected = [v for v in expected if v > std]

    assert expected == [it[0] for it in result]


def test_reduction(odps, setup):
    data = setup.gen_data(rows=5, value_range=(-100, 100))

    def stats(col, func):
        col_idx = [idx for idx, cn in enumerate(setup.expr.schema.names) if cn == col][0]
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
        (partial(stats, 'id', mean), setup.expr.id.mean()),
        (partial(len, data), setup.expr.count()),
        (partial(stats, 'id', var), setup.expr.id.var(ddof=0)),
        (partial(stats, 'id', lambda x: var(x, 1)), setup.expr.id.var(ddof=1)),
        (partial(stats, 'id', std), setup.expr.id.std(ddof=0)),
        (partial(stats, 'id', lambda x: moment(x, 3, central=True)), setup.expr.id.moment(3, central=True)),
        (partial(stats, 'id', skew), setup.expr.id.skew()),
        (partial(stats, 'id', kurtosis), setup.expr.id.kurtosis()),
        (partial(stats, 'id', sum), setup.expr.id.sum()),
        (partial(stats, 'id', min), setup.expr.id.min()),
        (partial(stats, 'id', max), setup.expr.id.max()),
        (partial(stats, 'isMale', min), setup.expr.isMale.min()),
        (partial(stats, 'isMale', sum), setup.expr.isMale.sum()),
        (partial(stats, 'isMale', any), setup.expr.isMale.any()),
        (partial(stats, 'isMale', all), setup.expr.isMale.all()),
        (partial(stats, 'name', nunique), setup.expr.name.nunique()),
        (partial(stats, 'name', cat), setup.expr.name.cat(sep='|')),
        (partial(stats, 'id', lambda x: len(x)), setup.expr.id.count()),
    ]

    fields = [it[1].rename('f'+str(i)) for i, it in enumerate(methods_to_fields)]

    expr = setup.expr[fields]

    res = setup.engine.execute(expr)
    result = get_result(res)

    for i, it in enumerate(methods_to_fields):
        method = it[0]

        first = method()
        second = [it[i] for it in result][0]
        if i == len(methods_to_fields) - 2:  # cat
            second = len(second.split('|'))
        if isinstance(first, float):
            assert pytest.approx(first) == second
        else:
            if first != second:
                pass
            assert first == second


def test_join(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
    ]

    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                [types.string, types.bigint, types.bigint])
    table_name = tn('pyodps_test_engine_table2')
    odps.delete_table(table_name, if_exists=True)
    table2 = odps.create_table(name=table_name, table_schema=schema2)
    expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

    setup.gen_data(data=data)

    data2 = [
        ['name1', 4, -1],
        ['name2', 1, -2]
    ]

    odps.write_table(table2, 0, data2)

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

        expr = setup.expr.join(expr2, on=['name', ('id', 'id2')])[
            lambda x: x.groupby(Scalar(1)).sort('name').row_number(), ]
        setup.engine.execute(expr)
    finally:
        table2.drop()


def test_join_aggregation(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
    ]
    setup.gen_data(data=data)

    expr = setup.expr.join(setup.expr.view(), on=['name', 'id'])[
        lambda x: x.count(), setup.expr.id.sum()]

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [[9, 30]] == result


def test_union(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
    ]
    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                [types.string, types.bigint, types.bigint])
    table_name = tn('pyodps_test_engine_table2')
    odps.delete_table(table_name, if_exists=True)
    table2 = odps.create_table(name=table_name, table_schema=schema2)
    expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

    setup.gen_data(data=data)

    data2 = [
        ['name3', 5, -1],
        ['name4', 6, -2]
    ]

    odps.write_table(table2, 0, data2)

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
        table2.drop()


def test_scale_value(odps, setup):
    data = [
        ['name1', 4, 5.3],
        ['name2', 2, 3.5],
        ['name1', 4, 4.2],
        ['name1', 3, 2.2],
        ['name1', 3, 4.1],
    ]
    schema = TableSchema.from_lists(['name', 'id', 'fid'],
                               [types.string, types.bigint, types.double])
    table_name = tn('pyodps_test_engine_scale_table')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(name=table_name, table_schema=schema)
    odps.write_table(table_name, 0, data)
    expr_input = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))

    expr = expr_input.min_max_scale(columns=['fid'])

    res = setup.engine.execute(expr)
    result = get_result(res)

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
        assert len(first) == len(second)
        for it1, it2 in zip(first, second):
            assert pytest.approx(it1) == it2

    expr = expr_input.std_scale(columns=['fid'])

    res = setup.engine.execute(expr)
    result = get_result(res)

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
        assert len(first) == len(second)
        for it1, it2 in zip(first, second):
            assert pytest.approx(it1) == it2


def test_persist(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
    ]
    setup.gen_data(data=data)

    table_name = tn('pyodps_test_engine_persist_seahawks_table')

    # simple persist
    try:
        with pytest.raises(ODPSError):
            setup.engine.persist(setup.expr, table_name, create_table=False)

        df = setup.engine.persist(setup.expr, table_name)

        res = setup.engine.execute(df)
        result = get_result(res)
        assert len(result) == 5
        assert data == result

        with pytest.raises(ValueError):
            setup.engine.persist(setup.expr, table_name, create_partition=True)
        with pytest.raises(ValueError):
            setup.engine.persist(setup.expr, table_name, drop_partition=True)
    finally:
        odps.delete_table(table_name, if_exists=True)

    # persist over existing table
    try:
        odps.create_table(table_name,
                               'name string, fid double, id bigint, isMale boolean, birth datetime',
                               lifecycle=1)

        expr = setup.expr[setup.expr, Scalar(1).rename('name2')]
        with pytest.raises(CompileError):
            setup.engine.persist(expr, table_name)

        expr = setup.expr['name', 'fid', setup.expr.id.astype('int32'), 'isMale', 'birth']
        df = setup.engine.persist(expr, table_name)

        res = setup.engine.execute(df)
        result = get_result(res)
        assert len(result) == 5
        assert data == [[r[0], r[2], r[1], None, None] for r in result]
    finally:
        odps.delete_table(table_name, if_exists=True)

    try:
        df = setup.engine.persist(setup.expr, table_name, partition={'ds': 'today'})

        res = setup.engine.execute(df)
        result = get_result(res)
        assert len(result) == 5
    finally:
        odps.delete_table(table_name, if_exists=True)

    try:
        schema = TableSchema.from_lists(setup.schema.names, setup.schema.types, ['ds'], ['string'])
        odps.create_table(table_name, schema)
        df = setup.engine.persist(setup.expr, table_name, partition='ds=today', create_partition=True)

        res = setup.engine.execute(df)
        result = get_result(res)
        assert len(result) == 5
        assert data == [d[:-1] for d in result]

        df2 = setup.engine.persist(setup.expr[setup.expr.id.astype('float'), 'name'], table_name,
                                  partition='ds=today2', create_partition=True, cast=True)

        res = setup.engine.execute(df2)
        result = get_result(res)
        assert len(result) == 5
        assert [d[:2] + [None] * (len(d) - 2) for d in data] == [d[:-1] for d in result]
    finally:
        odps.delete_table(table_name, if_exists=True)

    try:
        schema = TableSchema.from_lists(setup.schema.names, setup.schema.types, ['ds', 'hh'], ['string', 'string'])
        odps.create_table(table_name, schema)

        with pytest.raises(ValueError):
            setup.engine.persist(setup.expr, table_name, partition='ds=today', create_partition=True)

        setup.engine.persist(setup.expr, table_name, partition=OrderedDict([('hh', 'now'), ('ds', 'today')]))
        assert odps.get_table(table_name).exist_partition('ds=today,hh=now') is True
    finally:
        odps.delete_table(table_name, if_exists=True)

    try:
        setup.engine.persist(setup.expr, table_name, partitions=['name'])

        t = odps.get_table(table_name)
        assert 2 == len(list(t.partitions))
        with t.open_reader(partition='name=name1', reopen=True) as r:
            assert 4 == r.count
        with t.open_reader(partition='name=name2', reopen=True) as r:
            assert 1 == r.count
    finally:
        odps.delete_table(table_name, if_exists=True)


def test_make_kv(odps, setup):
    from odps import types as odps_types
    data = [
        ['name1', 1.0, 3.0, None, 10.0, None, None],
        ['name1', None, 3.0, 5.1, None, None, None],
        ['name1', 7.1, None, None, None, 8.2, None],
        ['name2', None, 1.2, 1.5, None, None, None],
        ['name2', None, 1.0, None, None, None, 1.1],
    ]
    kv_cols = ['k1', 'k2', 'k3', 'k5', 'k7', 'k9']
    schema = TableSchema.from_lists(['name'] + kv_cols,
                               [odps_types.string] + [odps_types.double] * 6)
    table_name = tn('pyodps_test_engine_make_kv')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(name=table_name, table_schema=schema)
    expr = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))
    try:
        odps.write_table(table, 0, data)
        expr1 = expr.to_kv(columns=kv_cols, kv_delim='=')

        res = setup.engine.execute(expr1)
        result = get_result(res)

        expected = [
            ['name1', 'k1=1,k2=3,k5=10'],
            ['name1', 'k2=3,k3=5.1'],
            ['name1', 'k1=7.1,k7=8.2'],
            ['name2', 'k2=1.2,k3=1.5'],
            ['name2', 'k2=1,k9=1.1'],
        ]

        assert result == expected
    finally:
        table.drop()


def test_filter_order(odps, setup):
    table_name = tn('pyodps_test_division_error')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(table_name, 'divided bigint, divisor bigint', lifecycle=1)

    try:
        odps.write_table(table_name, [[2, 0], [1, 1], [1, 2], [5, 1], [5, 0]])
        df = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(table.table_schema))
        fdf = df[df.divisor > 0]
        ddf = fdf[(fdf.divided / fdf.divisor).rename('result'),]
        expr = ddf[ddf.result > 1]

        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == [[5, ]]
    finally:
        table.drop()


def test_axf_exception(odps, setup):
    import sqlalchemy

    data = [
        ['name1', 4, 5.3, None, None],
        ['name2', 2, 3.5, None, None],
        ['name1', 4, 4.2, None, None],
        ['name1', 3, 2.2, None, None],
        ['name1', 3, 4.1, None, None],
    ]
    setup.gen_data(data=data)

    table_name = tn('pyodps_test_engine_axf_seahawks_table')

    try:
        schema = TableSchema.from_lists(setup.schema.names, setup.schema.types, ['ds'], ['string'])
        odps.create_table(table_name, schema)
        df = setup.engine.persist(setup.expr, table_name, partition='ds=today', create_partition=True)

        with pytest.raises(sqlalchemy.exc.DatabaseError):
            setup.engine.execute(df.input)
    finally:
        odps.delete_table(table_name, if_exists=True)