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
import sys
import uuid
import os
import zipfile
import tarfile
import re
import functools
import time
from collections import namedtuple, OrderedDict
from datetime import datetime, timedelta
from functools import partial
from random import randint

import pytest

from ..... import options, types
from .....compat import PY27, irange as xrange, six, futures, BytesIO
from .....errors import ODPSError
from .....models import TableSchema
from .....tests.core import get_result, approx_list, py_and_c, run_sub_tests_in_parallel
from .....utils import to_text
from ....expr.expressions import CollectionExpr
from ....types import validate_data_type, DynamicSchema
from .... import Scalar, output_names, output_types, output, day, millisecond, agg, func
from ...context import context
from ...odpssql.engine import ODPSSQLEngine
from ...errors import CompileError
from ...tests.core import NumGenerators, tn
from ..types import df_schema_to_odps_schema, odps_schema_to_df_schema


def _reloader():
    from .....conftest import get_config
    from .....tunnel import TableTunnel

    cfg = get_config()
    cfg.tunnel = TableTunnel(cfg.odps, endpoint=cfg.odps._tunnel_endpoint)


py_and_c_deco = py_and_c([
    "odps.models.record", "odps.models", "odps.tunnel.io.reader",
    "odps.tunnel.io.writer", "odps.tunnel.tabletunnel",
    "odps.tunnel.instancetunnel",
], _reloader)


class ODPSEngine(ODPSSQLEngine):

    def _pre_process(self, expr):
        src_expr = expr
        expr = self._convert_table(expr)
        return self._analyze(expr.to_dag(), src_expr)


class FakeBar(object):
    def update(self, *args, **kwargs):
        pass

    def inc(self, *args, **kwargs):
        pass

    def status(self, *args, **kwargs):
        pass


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

        odps.write_table(table, 0, data)
        return data

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    pd_schema = TableSchema.from_lists(
        ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
    )
    schema = df_schema_to_odps_schema(pd_schema)
    table_name = tn('pyodps_test_engine_table_%s' % str(uuid.uuid4()).replace('-', '_'))
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(table_name, schema, lifecycle=1)
    expr = CollectionExpr(_source_data=table, _schema=pd_schema)

    engine = ODPSEngine(odps)
    faked_bar = FakeBar()

    try:
        nt = namedtuple("NT", "table schema expr engine faked_bar gen_data")
        yield nt(table, schema, expr, engine, faked_bar, gen_data)
    finally:
        table.drop()


def test_tunnel_cases(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr.count()
    res = setup.engine._handle_cases(expr, setup.faked_bar)
    result = get_result(res)
    assert 10 == result

    expr = setup.expr.name.count()
    res = setup.engine._handle_cases(expr, setup.faked_bar)
    result = get_result(res)
    assert 10 == result

    res = setup.engine._handle_cases(setup.expr, setup.faked_bar)
    result = get_result(res)
    assert data == result
    assert list(res)[0]['name'] is not None

    expr = setup.expr['name', setup.expr.id.rename('new_id')]
    res = setup.engine._handle_cases(expr, setup.faked_bar)
    result = get_result(res)
    assert [it[:2] for it in data] == result

    expr = setup.expr['name']
    res = setup.engine._handle_cases(
        setup.engine._pre_process(expr), setup.faked_bar)
    result = get_result(res)
    assert [it[:1] for it in data] == result

    table_name = tn('pyodps_test_engine_partitioned')
    odps.delete_table(table_name, if_exists=True)

    df = setup.engine.persist(setup.expr, table_name, partitions=['name'])

    try:
        expr = df[df.name == func.max_pt('{0}.{1}'.format(odps.project, table_name))]
        res = setup.engine._handle_cases(expr, setup.faked_bar, verify=True)
        assert res is False

        expr = df.count()
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res is None

        expr = df[df.name == data[0][0]]['fid', 'id'].count()
        expr = setup.engine._pre_process(expr)
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res > 0

        expr = df[df.name == data[0][0]]['fid', 'id']
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert len(res) > 0

        expr = df[df.name == data[0][0]]
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert all(r is not None for r in res[:, -1])

        expr = df[df.name == data[1][0]]
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert all(r is not None for r in res[:, -1])

        expr = df.filter_parts('name={0}'.format(data[0][0])).count()
        expr = setup.engine._pre_process(expr)
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res > 0

        expr = df.filter_parts('name={0}'.format(data[0][0]))
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert len(res) > 0

        expr = df.filter_parts('name={0}'.format(data[0][0]))['fid', 'id'].count()
        expr = setup.engine._pre_process(expr)
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res > 0

        expr = df.filter_parts('name={0}'.format(data[0][0]))['fid', 'id']
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert len(res) > 0
    finally:
        odps.delete_table(table_name, if_exists=True)

    df = setup.engine.persist(setup.expr, table_name, partitions=['name', 'id'])

    try:
        expr = df.filter(df.ismale == data[0][3],
                         df.name == data[0][0], df.id == data[0][1],
                         df.scale == data[0][4]
                         )
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res is None

        expr = df.count()
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res is None

        expr = df.filter_parts('name={0}/id={1}'.format(data[0][0], data[0][1]))
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res is None

        expr = df[(df.name == data[0][0]) & (df.id == data[0][1])]['fid', 'ismale'].count()
        expr = setup.engine._pre_process(expr)
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res > 0

        expr = df[(df.name == data[0][0]) & (df.id == data[0][1])]['fid', 'ismale']
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert len(res) > 0
    finally:
        odps.delete_table(table_name, if_exists=True)

    table = odps.create_table(
        table_name, TableSchema.from_lists(['val'], ['bigint'], ['name', 'id'], ['string', 'bigint']))
    table.create_partition('name=a,id=1')
    with table.open_writer('name=a,id=1') as writer:
        writer.write([[0], [1], [2]])
    table.create_partition('name=a,id=2')
    with table.open_writer('name=a,id=2') as writer:
        writer.write([[3], [4], [5]])
    table.create_partition('name=b,id=1')
    with table.open_writer('name=b,id=1') as writer:
        writer.write([[6], [7], [8]])

    df = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(table.table_schema))

    try:
        expr = df.count()
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res is None

        expr = df[df.name == 'a'].count()
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res is None

        expr = df[df.id == 1].count()
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res is None

        expr = df.filter(df.name == 'a', df.id == 1).count()
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert res == 3

        expr = df
        res = setup.engine._handle_cases(expr, setup.faked_bar, update_progress_count=1)
        assert len(res) == 9
        assert res[0][-1] is not None
        assert res[0][-2] is not None

        expr = df[df.name == 'a']
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert len(res) == 6

        expr = df[df.id == 1]
        res = setup.engine._handle_cases(expr, setup.faked_bar)
        assert len(res) == 6

        expr = df[df.name == 'a'][:4]
        res = setup.engine._handle_cases(expr, setup.faked_bar, head=5)
        result = get_result(res)
        assert sum(r[0] for r in result) == 6

        expr = df[df.name == 'a'][:5]
        res = setup.engine._handle_cases(expr, setup.faked_bar, head=4)
        result = get_result(res)
        assert sum(r[0] for r in result) == 6

        expr = df[df.name == 'a']
        res = setup.engine._handle_cases(expr, setup.faked_bar, head=4)
        result = get_result(res)
        assert sum(r[0] for r in result) == 6

        expr = df[df.name == 'a'][:5]
        res = setup.engine._handle_cases(expr, setup.faked_bar, tail=4)
        assert res is None

        expr = df.filter(df.name == 'a', df.id == 1)[:2]
        res = setup.engine._handle_cases(expr, setup.faked_bar, tail=1)
        result = get_result(res)
        assert sum(r[0] for r in result) == 1
    finally:
        odps.delete_table(table_name, if_exists=True)


def test_async(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr.id.sum()

    future = setup.engine.execute(expr, async_=True, priority=4, ret_instance=True)
    assert future.done() is False
    inst, res = future.result()

    assert sum(it[1] for it in data) == res
    assert inst.priority == 4


def test_dataframe_progress_log(odps, setup):
    setup.gen_data(10, value_range=(-1000, 1000))

    logs = []
    try:
        options.verbose = True
        options.verbose_log = logs.append
        options.progress_time_interval = 0.1

        def func(x):
            time.sleep(0.1)
            return x

        expr = setup.expr.id.map(func).sum()
        setup.engine.execute(expr)

        assert any("select" in log.lower() for log in logs)
        assert any("instance" in log.lower() for log in logs)
        assert any("_job_" in log.lower() for log in logs)
    finally:
        options.verbose = False
        options.verbose_log = None
        options.progress_time_interval = 5 * 60


def test_no_permission(odps, setup):
    class NoPermissionEngine(ODPSEngine):
        def _handle_cases(self, *args, **kwargs):
            from .....errors import NoPermission
            raise NoPermission('No permission to use tunnel')

        def _open_reader(self, t, **kwargs):
            raise RuntimeError('Cannot use tunnel')

    data = setup.gen_data(10, value_range=(-1000, 1000))

    res = NoPermissionEngine(odps).execute(setup.expr)
    result = get_result(res)
    assert data == result


def test_no_tunnel_case(odps, setup):
    class NoTunnelCaseEngine(ODPSEngine):
        def _handle_cases(self, *args, **kwargs):
            raise RuntimeError('Not allow to use tunnel')

    data = setup.gen_data(10, value_range=(-1000, 1000))

    options.df.optimizes.tunnel = False

    try:
        res = NoTunnelCaseEngine(odps).execute(setup.expr)
        result = get_result(res)
        assert data == result
    finally:
        options.df.optimizes.tunnel = True


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
    table.reload()
    assert table.lifecycle == 1

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

    persist_table_name = 'pyodps_test_persist_del' + str(uuid.uuid4()).replace('-', '_')
    expr10 = setup.expr[setup.expr.id * 2, 'name', 'fid']
    setup.engine.persist(expr10, persist_table_name, lifecycle=1)
    odps.delete_table(persist_table_name)
    res = setup.engine.execute(expr10[expr10.id * 2, expr10.exclude('id')])
    result = get_result(res)
    assert len(result[0]) == 3


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
    expr = setup.expr[setup.expr.id < 10].cache()
    expr1 = expr.id.sum()
    expr2 = expr.id.mean()

    fs = setup.engine.execute([expr, expr1, expr2], n_parallel=2, async_=True, timeout=1)
    assert len(fs) == 3

    futures.wait((fs[0], ), timeout=120)
    time.sleep(.5)
    assert fs[1].running() and fs[2].running() is True
    time.sleep(.5)
    assert fs[2].running() or fs[2].done() is True

    assert fs[1].result() == expect1
    assert pytest.approx(fs[2].result()) == expect2
    assert context.is_cached(expr) is True


def test_base(odps, setup):
    data = setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr[setup.expr.id < 10]['name', lambda x: x.id, ]
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
    assert [it[1]+1 for it in data] == [it[2] for it in result]

    expr = setup.expr.sort('id')[:5]
    res = setup.engine.execute(expr)
    result = get_result(res.values)
    assert sorted(data, key=lambda it: it[1])[:5] == result

    expr = setup.expr.sort('id')[:5]
    # test do not use tunnel
    res = setup.engine.execute(expr, use_tunnel=False)
    result = get_result(res.values)
    assert sorted(data, key=lambda it: it[1])[:5] == result

    exp = setup.expr[
        setup.expr.scale.map(lambda x: x + 1).rename('scale1'),
        setup.expr.scale.map(functools.partial(lambda v, x: x + v, 10)).rename('scale2'),
    ]
    res = setup.engine.execute(exp)
    result = get_result(res.values)
    assert [[r[4] + 1, r[4] + 10] for r in data] == result

    if six.PY2:  # Skip in Python 3, as hash() behaves randomly.
        expr = setup.expr.name.hash()
        res = setup.engine.execute(expr)
        result = get_result(res.values)
        assert [[hash(r[0])] for r in data] == result

    expr = setup.expr.sample(parts=10)
    res = setup.engine.execute(expr)
    assert len(res) >= 1

    expr = setup.expr.sample(parts=10, columns=setup.expr.id)
    res = setup.engine.execute(expr)
    assert len(res) >= 0

    expr = setup.expr.sample(frac=0.5)
    res = setup.engine.execute(expr)
    assert len(res) >= 1

    expr = setup.expr.sample(n=5)
    res = setup.engine.execute(expr)
    assert len(res) >= 1

    expr = setup.expr[:1].filter(lambda x: x.name == data[1][0])
    res = setup.engine.execute(expr)
    assert len(res) == 0


def test_describe(odps, setup):
    setup.gen_data(10, value_range=(-1000, 1000))

    expr = setup.expr.exclude('scale').describe()
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert list(result[0][1:]) == [10, 10]

    expr = setup.expr['name', 'birth'].describe()
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

    expr = setup.expr.filter(setup.expr.name == u'中文')
    res = setup.engine.execute(expr)
    assert len(res) == 1


@py_and_c_deco
@pytest.mark.skipif(not PY27, reason="known bug for binary results in py3")
def test_non_utf8(odps, setup):
    data = [
        ['中文', 4, 5.3, None, None, None],
    ]
    setup.gen_data(data=data)

    const = to_text(data[0]).encode('gbk')
    expr = setup.expr[:1].map_reduce(lambda row: const,
                                    mapper_output_names=['s'],
                                    mapper_output_types=['string' if PY27 else 'binary'])
    options.tunnel.string_as_binary = True
    if not PY27:
        options.sql.use_odps2_extension = True
    try:
        res = get_result(setup.engine.execute(expr))
        assert res[0][0] == const
    finally:
        options.tunnel.string_as_binary = False
        options.sql.use_odps2_extension = None


def test_functions(odps, setup):
    data = setup.gen_data(5, value_range=(-1000, 1000))

    f = lambda x: x + 1

    expr = setup.expr[setup.expr.id.map(f).rename('id1'),
                     setup.expr.fid.map(f).rename('id2'),
                     (setup.expr.id + 1).map(f).rename('id3')]
    expected = [[r[1] + 1, r[2] + 1, r[1] + 2] for r in data]
    res = setup.engine.execute(expr)
    result = get_result(res)

    assert expected == result
    assert len(setup.engine._ctx._registered_funcs.values()) == 2

    setup.engine._ctx._registered_funcs.clear()

    def h(row):
        yield row

    expr1 = setup.expr['name', 'id'].apply(h, axis=1, names=['name', 'id'], types=['string', 'int'])
    expr2 = setup.expr['id', 'name'].apply(h, axis=1, names=['id', 'name'], types=['int', 'string'])
    expr3 = expr1.join(expr2)
    assert len(setup.engine.execute(expr3)) == 5

    assert len(setup.engine._ctx._registered_funcs.values()) == 2

    expr = setup.expr.apply(lambda row: row.name + str(row.id), axis=1, reduce=True).rename('name')

    res = setup.engine.execute(expr)
    result = get_result(res)
    assert result == [[r[0] + str(r[1])] for r in data]


def test_element(odps, setup):
    data = setup.gen_data(5, nullable_field='name')
    data = sorted(data, key=lambda v: v[1])

    fields = [
        setup.expr.id.rename('_sort_key'),
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
    ]

    expr = setup.expr[fields].sort(['_sort_key']).exclude(['_sort_key'])

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

    assert [it[5] for it in data] == [it[13] for it in result]

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

    assert pytest.approx([it[2] - 1 for it in data]) == [it[1] for it in result]

    assert [it[4] * 2 for it in data] == [it[2] for it in result]

    assert [it[4] + it[1] for it in data] == [it[3] for it in result]

    assert pytest.approx([float(it[1]) / 2 for it in data]) == [it[4] for it in result]

    assert [int(it[1] ** 2) for it in data] == [it[5] for it in result]

    assert [abs(it[1]) for it in data] == [it[6] for it in result]

    assert [~it[1] for it in data] == [it[7] for it in result]

    assert pytest.approx([-it[2] for it in data]) == [it[8] for it in result]

    assert [not it[3] for it in data] == [it[9] for it in result]

    assert [it[1] // 2 for it in data] == [it[11] for it in result]

    assert [it[5] + timedelta(days=1) for it in data] == [it[12] for it in result]

    assert [10] * len(data) == [it[13] for it in result]

    assert [it[1] % 2 for it in data] == [it[14] for it in result]


def test_gen_func(odps, setup):
    data = setup.gen_data(5, value_range=(1, 1))

    def func(x, val):
        return x + val

    expr = setup.expr[setup.expr.id.map(func, args=(1,)),
                     setup.expr.id.map(func, args=(2,)).rename('id2')]
    res = setup.engine.execute(expr)
    result = get_result(res)

    assert all(r[0] == 2 for r in result) is True
    assert all(r[1] == 3 for r in result) is True

    def func(x, val):
        return x + val['id']

    expr = setup.expr[setup.expr.id.map(func, args=({'id': 1},)),
                     setup.expr.id.map(func, args=({'id': 2},)).rename('id2')]
    res = setup.engine.execute(expr)
    result = get_result(res)

    assert all(r[0] == 2 for r in result) is True
    assert all(r[1] == 3 for r in result) is True


def test_math(odps, setup):
    data = setup.gen_data(5, value_range=(1, 90))

    if hasattr(math, 'expm1'):
        expm1 = math.expm1
    else:
        expm1 = lambda x: 2 * math.exp(x / 2.0) * math.sinh(x / 2.0)

    methods_to_fields = [
        (math.sin, setup.expr.id.sin()),
        (math.cos, setup.expr.id.cos()),
        (math.tan, setup.expr.id.tan()),
        (math.sinh, setup.expr.id.sinh()),
        (math.cosh, setup.expr.id.cosh()),
        (math.tanh, setup.expr.id.tanh()),
        (math.log, setup.expr.id.log()),
        (lambda v: math.log(v, 2), setup.expr.id.log2()),
        (math.log10, setup.expr.id.log10()),
        (math.log1p, setup.expr.id.log1p()),
        (math.exp, setup.expr.id.exp()),
        (expm1, setup.expr.id.expm1()),
        (math.acosh, setup.expr.id.arccosh()),
        (math.asinh, setup.expr.id.arcsinh()),
        (math.atanh, setup.expr.id.arctanh()),
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
        (lambda s: extract('123'+s, r'[^a-z]*(\w+)', group=1), ('123'+setup.expr.name).extract(r'[^a-z]*(\w+)', group=1)),
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
        (lambda s: s[1: s.find('a')], setup.expr.name[1: setup.expr.name.find('a')]),
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
        (lambda s: None, setup.expr.name.map(lambda x: None).contains('abc')),
        (lambda s: None, setup.expr.name.map(lambda x: None).replace(data[0][0], 'test')),
    ]

    fields = [it[1].rename('id'+str(i)) for i, it in enumerate(methods_to_fields)]

    try:
        for use_odps2 in [False, True]:
            options.sql.use_odps2_extension = use_odps2

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
    finally:
        options.sql.use_odps2_extension = None


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

    class my_func3(object):
        def __init__(self, resources):
            self.file_resource = resources[0]
            self.table_resource = resources[1]
            self.table_resource2 = list(resources[2])

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
    table_name2 = tn('pyodps_tmp_function_resource_table2')
    try:
        odps.delete_resource(file_resource_name)
    except:
        pass
    file_resource = odps.create_resource(file_resource_name, 'file',
                                         file_obj='\n'.join(str(r[1]) for r in data[:3]))
    odps.delete_table(table_name, if_exists=True)
    t = odps.create_table(table_name, TableSchema.from_lists(['id'], ['bigint']))
    with t.open_writer() as writer:
        writer.write([r[1:2] for r in data[3:4]])
    odps.delete_table(table_name2, if_exists=True)
    t2 = odps.create_table(table_name2, TableSchema.from_lists(['name', 'id'], ['string', 'bigint']))
    with t2.open_writer() as writer:
        writer.write([r[:2] for r in data[3:4]])
    try:
        odps.delete_resource(table_resource_name)
    except:
        pass
    table_resource = odps.create_resource(table_resource_name, 'table',
                                          table_name=t.name)

    try:
        expr = setup.expr.id.map(my_func, resources=[file_resource, table_resource])

        res = setup.engine.execute(expr)
        result = get_result(res)
        result = [r for r in result if r[0] is not None]

        assert sorted([[r[1]] for r in data[:4]]) == sorted(result)

        expr = setup.expr['name', 'id', 'fid']
        expr = expr.apply(my_func3, axis=1, resources=[file_resource, table_resource, t2.to_df()],
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

        expr = setup.expr['name', 'id', 'fid']
        expr2 = expr.distinct('id')

        def my_func3(resources):
            def h(row):
                return row
            return h

        expr4 = expr2.apply(my_func3, axis=1, names=expr2.dtypes.names, types=expr2.dtypes.types,
                            resources=[expr])

        res = setup.engine.execute(expr4)
        result = get_result(res)

        assert len(result) == 5
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


def test_function_resources_with_partition(odps, setup):
    data = setup.gen_data(5)

    table_name = tn('pyodps_tmp_function_resource_part_table')
    odps.delete_table(table_name, if_exists=True)
    t = odps.create_table(table_name, TableSchema.from_lists(
        ['id', 'id2'], ['bigint', 'bigint'], ['ds'], ['string']
    ))
    with t.open_writer(partition='ds=ds1', create_partition=True) as w:
        w.write([1, 2])
    with t.open_writer(partition='ds=ds2', create_partition=True) as w:
        w.write([2, 3])

    table_resource_name = tn('pyodps_tmp_part_table_resource')
    try:
        odps.delete_resource(table_resource_name)
    except:
        pass
    table_resource = odps.create_resource(table_resource_name, 'table',
                                               table_name=table_name, partition='ds=ds1')
    df = odps.get_table(table_name).get_partition('ds=ds2').to_df()
    try:
        @output(['n'], ['int'])
        def func(resources):
            n1 = next(sum(r) for r in resources[0])
            n2 = next(sum(r) for r in resources[1])
            n = [n1 - n2]

            def h(row):
                yield row[0] + n[0]

            return h

        expr = setup.expr['id',].apply(func, axis=1, resources=[table_resource, df])
        res = setup.engine.execute(expr)
        result = get_result(res)

        expected = [[r[1] - 2] for r in data]
        assert sorted(expected) == sorted(result)

    finally:
        table_resource.drop()


def test_third_party_libraries(odps, setup):
    import requests
    from .....compat import BytesIO

    data = [
        ['2016-08-18', 4, 5.3, None, None, None],
        ['2015-08-18', 2, 3.5, None, None, None],
        ['2014-08-18', 4, 4.2, None, None, None],
        ['2013-08-18', 3, 2.2, None, None, None],
        ['2012-08-18', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    dateutil_urls = [
        'http://mirrors.aliyun.com/pypi/packages/d4/70/'
        'd60450c3dd48ef87586924207ae8907090de0b306af2bce5d134d78615cb/'
        'python_dateutil-2.8.1-py2.py3-none-any.whl',
        'https://mirrors.aliyun.com/pypi/packages/be/ed/'
        '5bbc91f03fa4c839c4c7360375da77f9659af5f7086b7a7bdda65771c8e0/'
        'python-dateutil-2.8.1.tar.gz',
        'http://mirrors.aliyun.com/pypi/packages/b7/9f/'
        'ba2b6aaf27e74df59f31b77d1927d5b037cc79a89cda604071f93d289eaf/'
        'python-dateutil-2.5.3.zip#md5=52b3f339f41986c25c3a2247e722db17',
    ]
    if sys.version_info[:2] >= (3, 11):
        # since Python 3.11, collections.Callable is removed and no zip archives available
        dateutil_urls = dateutil_urls[:2]

    dateutil_resources = []
    for dateutil_url, name in zip(dateutil_urls, ['dateutil.whl', 'dateutil.tar.gz', 'dateutil.zip']):
        obj = BytesIO(requests.get(dateutil_url).content)
        res_name = '%s_%s.%s' % (
            name.split('.', 1)[0], str(uuid.uuid4()).replace('-', '_'), name.split('.', 1)[1])
        res = odps.create_resource(res_name, 'file', file_obj=obj)
        dateutil_resources.append(res)

    obj = BytesIO(requests.get(dateutil_urls[0]).content)
    res_name = 'dateutil_archive_%s.zip' % str(uuid.uuid4()).replace('-', '_')
    res = odps.create_resource(res_name, 'archive', file_obj=obj)
    dateutil_resources.append(res)

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

    def subtest1(resource, dateutil_resource):
        def f(x):
            from dateutil.parser import parse
            return int(parse(x).strftime('%Y'))

        expr = setup.expr.name.map(f, rtype='int')

        res = setup.engine.execute(expr, libraries=[resource.name, dateutil_resource])
        result = get_result(res)

        assert result == [[int(r[0].split('-')[0])] for r in data]

    def subtest2(resource, dateutil_resource):
        def f(row):
            from dateutil.parser import parse
            return int(parse(row.name).strftime('%Y')),

        expr = setup.expr.apply(f, axis=1, names=['name', ], types=['int', ])

        res = setup.engine.execute(expr, libraries=[resource, dateutil_resource])
        result = get_result(res)

        assert result == [[int(r[0].split('-')[0])] for r in data]

    def subtest3(resource, dateutil_resource):
        def f(row):
            from dateutil.parser import parse
            return int(parse(row.name).strftime('%Y')),

        expr = setup.expr.apply(f, axis=1, names=['name', ], types=['int', ])
        table_name = 'test_pyodps_table_' + str(uuid.uuid4()).replace('-', '_')
        try:
            new_expr = setup.engine.persist(expr, table_name, libraries=[resource, dateutil_resource])
            res = setup.engine.execute(new_expr)
            result = get_result(res)

            assert result == [[int(r[0].split('-')[0])] for r in data]
        finally:
            odps.delete_table(table_name, if_exists=True)

    def subtest4(resource, dateutil_resource):
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

        expr = setup.expr.name.agg(Agg, rtype='int')
        res = setup.engine.execute(expr, libraries=[resource, dateutil_resource])

        assert res == sum([int(r[0].split('-')[0]) for r in data])

    try:
        run_sub_tests_in_parallel(
            30,
            [
                functools.partial(st, res, dateutil_res)
                for res, dateutil_res in itertools.product(resources, dateutil_resources)
                for st in [subtest1, subtest2, subtest3, subtest4]
            ]
        )
    finally:
        [res.drop() for res in resources + dateutil_resources]


def test_third_party_wheel(odps, setup):
    import requests
    from .....compat import BytesIO

    data = [
        ['2016-08-18', 4, 5.3, None, None, None],
        ['2015-08-18', 2, 3.5, None, None, None],
        ['2014-08-18', 4, 4.2, None, None, None],
        ['2013-08-18', 3, 2.2, None, None, None],
        ['2012-08-18', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    dateutil_url = (
        'http://mirrors.aliyun.com/pypi/packages/d4/70/'
        'd60450c3dd48ef87586924207ae8907090de0b306af2bce5d134d78615cb/'
        'python_dateutil-2.8.1-py2.py3-none-any.whl'
    )
    obj = BytesIO(requests.get(dateutil_url).content)
    wheel_res = odps.create_resource(
        'dateutil_%s.whl' % str(uuid.uuid4()), 'archive', file_obj=obj)
    six_path = os.path.join(os.path.dirname(os.path.abspath(six.__file__)), 'six.py')

    try:
        def f(x):
            from dateutil.parser import parse
            return int(parse(x).strftime('%Y'))

        expr = setup.expr.name.map(f, rtype='int')

        try:
            options.df.libraries = [wheel_res, six_path]
            res = setup.engine.execute(expr)
        finally:
            options.df.libraries = None
        result = get_result(res)

        assert result == [[int(r[0].split('-')[0])] for r in data]
    finally:
        wheel_res.drop()


def test_supersede_numpy_library(odps, setup):
    data = [
        ['2016-08-18', 4, 5.3, None, None, None],
    ]
    setup.gen_data(data=data)

    tar_io = BytesIO()
    pseudo_np_tar = tarfile.open(fileobj=tar_io, mode='w:gz')
    init_content = b"__version__ = '100.100.100'\n"
    info = tarfile.TarInfo(name='numpy/__init__.py')
    info.size = len(init_content)
    pseudo_np_tar.addfile(info, fileobj=BytesIO(init_content))
    pseudo_np_tar.addfile(tarfile.TarInfo('numpy/pseudo.so'), fileobj=BytesIO())
    pseudo_np_tar.close()

    res_name = 'pseudo_numpy_%s.tar.gz' % str(uuid.uuid4())
    res = None
    try:
        res = odps.create_resource(res_name, 'archive', fileobj=tar_io.getvalue())

        def fun(_):
            try:
                import numpy
                return numpy.__version__
            except ImportError:
                return 'not installed'

        options.sql.settings = {'odps.isolation.session.enable': True}

        expr = setup.expr.name.map(fun)
        r = setup.engine.execute(expr)
        original = get_result(r)[0]
        if to_text(original[0]) == 'not installed':
            return

        options.df.supersede_libraries = True
        expr = setup.expr.name.map(fun)
        r = setup.engine.execute(expr, libraries=[res_name])
        result = get_result(r)[0]
        assert result == ['100.100.100']

        options.df.supersede_libraries = False
        expr = setup.expr.name.map(fun)
        r = setup.engine.execute(expr, libraries=[res_name])
        result = get_result(r)[0]
        assert result == original
    finally:
        options.df.libraries = []
        options.df.supersede_libraries = False
        options.sql.settings = {}
        if res:
            res.drop()


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
    from odps import types as odps_types

    data = setup.gen_data(5)

    def my_func(row):
        return row.name, row.scale + 1, row.birth

    expr = setup.expr['name', 'id', 'scale', 'birth'].apply(
        my_func,
        axis=1,
        names=['name', 'scale', 'birth'],
        types=['string', 'decimal', 'datetime'],
    )

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [[r[0], r[1]] for r in result] == [[r[0], r[4] + 1] for r in data]

    def my_func2(row):
        yield len(row["name"])
        yield row.id

    expr = setup.expr['name', 'id'].apply(my_func2, axis=1, names='cnt', rtype='int')
    expr = expr.filter(expr.cnt > 1)

    res = setup.engine.execute(expr)
    result = get_result(res)

    def gen_expected(data):
        for r in data:
            yield len(r[0])
            yield r[1]

    assert [r[0] for r in result] == [r for r in gen_expected(data) if r > 1]

    # test custom apply with wrapper
    @output(['name', 'scale', 'birth'], [odps_types.string, 'decimal', 'datetime'])
    def my_func3(row):
        return row.name, row.scale + 1, row.birth

    expr = setup.expr['name', 'id', 'scale', 'birth'].apply(my_func3, axis=1)

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [[r[0], r[1]] for r in result] == [[r[0], r[4] + 1] for r in data]

    # test custom apply with invocation of other custom apply functions
    @output(['name', 'scale', 'birth'], ['string', 'decimal', 'datetime'])
    def my_func4(row):
        return my_func3(row)

    expr = setup.expr['name', 'id', 'scale', 'birth'].apply(my_func4, axis=1)

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert [[r[0], r[1]] for r in result] == [[r[0], r[4] + 1] for r in data]


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
        (partial(date_value, lambda d: d.strftime('%Y%d')), setup.expr.birth.strftime('%Y%d')),
        (partial(date_value, lambda d: datetime.strptime(d.strftime('%Y%d'), '%Y%d')),
         setup.expr.birth.strftime('%Y%d').strptime('%Y%d')),
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
                    return v.to_datetime()
                else:
                    return v
        except ImportError:
            conv = lambda v: v

        second = [conv(it[i]) for it in result]
        assert first == second


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


def test_pivot(odps, setup):
    data = [
        ['name1', 1, 1.0, True, None, None],
        ['name1', 2, 2.0, True, None, None],
        ['name2', 1, 3.0, False, None, None],
        ['name2', 3, 4.0, False, None, None]
    ]
    setup.gen_data(data=data)

    expr = setup.expr

    def subtest1():
        expr1 = expr.pivot(rows='id', columns='name', values='fid').distinct()
        res = setup.engine.execute(expr1)
        result = get_result(res)

        expected = [
            [1, 1.0, 3.0],
            [2, 2.0, None],
            [3, None, 4.0]
        ]
        assert sorted(result) == sorted(expected)

    def subtest2():
        expr2 = expr.pivot(rows='id', columns='name', values=['fid', 'isMale'])
        res = setup.engine.execute(expr2)
        result = get_result(res)

        expected = [
            [1, 1.0, 3.0, True, False],
            [2, 2.0, None, True, None],
            [3, None, 4.0, None, False]
        ]
        assert sorted(result) == sorted(expected)

    def subtest3():
        expr3 = expr.pivot(rows='id', columns='name', values='fid')['name3']
        with pytest.raises(ValueError) as cm:
            setup.engine.execute(expr3)
        assert 'name3' in str(cm.value)

    def subtest4():
        expr4 = expr.pivot(rows='id', columns='name', values='fid')['id', 'name1']
        res = setup.engine.execute(expr4)
        result = get_result(res)

        expected = [
            [1, 1.0],
            [2, 2.0],
            [3, None]
        ]
        assert sorted(result) == sorted(expected)

    def subtest5():
        expr5 = expr.pivot(rows='id', columns='name', values='fid')
        expr5 = expr5[expr5, (expr5['name1'].astype('int') + 1).rename('new_name')]
        res = setup.engine.execute(expr5)
        result = get_result(res)

        expected = [
            [1, 1.0, 3.0, 2.0],
            [2, 2.0, None, 3.0],
            [3, None, 4.0, None]
        ]
        assert sorted(result) == sorted(expected)

    def subtest6():
        odps_data = [
            ['name1', 1],
            ['name2', 2],
            ['name1', 3],
        ]

        names = ['name', 'id']
        types = ['string', 'bigint']

        table = tn('pyodps_df_pivot_to_join')
        odps.delete_table(table, if_exists=True)
        t = odps.create_table(table, TableSchema.from_lists(names, types))
        with t.open_writer() as w:
            w.write([t.new_record(r) for r in odps_data])

        from .... import DataFrame
        odps_df = DataFrame(t)

        try:
            expr6 = expr.pivot(rows='id', columns='name', values='fid')
            expr6 = expr6.join(odps_df, on='id')[expr6, 'name']
            res = setup.engine.execute(expr6)
            result = get_result(res)

            expected = [
                [1, 1.0, 3.0, 'name1'],
                [2, 2.0, None, 'name2'],
                [3, None, 4.0, 'name1']
            ]
            assert sorted(result) == sorted(expected)
        finally:
            t.drop()

    run_sub_tests_in_parallel(10, [
        subtest1, subtest2, subtest3, subtest4, subtest5, subtest6
    ])


def test_pivot_table(odps, setup):
    data = [
        ['name1', 1, 1.0, True, None, None],
        ['name1', 1, 5.0, True, None, None],
        ['name1', 2, 2.0, True, None, None],
        ['name2', 1, 3.0, False, None, None],
        ['name2', 3, 4.0, False, None, None]
    ]

    setup.gen_data(data=data)

    expr = setup.expr

    def subtest1():
        expr1 = expr.pivot_table(rows='name', values='fid')
        res = setup.engine.execute(expr1)
        result = get_result(res)

        expected = [
            ['name1', 8.0 / 3],
            ['name2', 3.5],
        ]
        assert sorted(result) == sorted(expected)

    def subtest2():
        expr2 = expr.pivot_table(rows='name', values='fid', aggfunc=['mean', 'sum', 'quantile(0.2)'])
        res = setup.engine.execute(expr2)
        result = get_result(res)

        expected = [
            ['name1', 8.0 / 3, 8.0, 1.4],
            ['name2', 3.5, 7.0, 3.2],
        ]
        assert res.schema.names == ['name', 'fid_mean', 'fid_sum', 'fid_quantile_0_2']
        assert sorted(result) == sorted(expected)

    def subtest3():
        expr5 = expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum'])
        expr6 = expr5['name1_fid_mean',
                      expr5.groupby(Scalar(1)).sort('name1_fid_mean').name1_fid_mean.astype('float').cumsum()]

        k = lambda x: list(0 if it is None else it for it in x)

        # TODO: fix this situation, act different compared to pandas
        expected = [
            [2, 2], [3, 5], [None, None]
        ]
        res = setup.engine.execute(expr6)
        result = get_result(res)
        assert sorted(result, key=k) == sorted(expected, key=k)

    def subtest4():
        expr3 = expr.pivot_table(rows='id', values='fid', columns='name', fill_value=0).distinct()
        res = setup.engine.execute(expr3)
        result = get_result(res)

        expected = [
            [1, 3.0, 3.0],
            [2, 2.0, 0],
            [3, 0, 4.0]
        ]

        assert res.schema.names == ['id', 'name1_fid_mean', 'name2_fid_mean']
        assert result == expected

    def subtest5():
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
        res = setup.engine.execute(expr4)
        result = get_result(res)

        expected = [
            [1, 6.0, 3.0, 3.0, 3.0],
            [2, 2.0, 0, 2.0, 0],
            [3, 0, 4.0, 0, 4.0]
        ]

        assert res.schema.names == ['id', 'name1_fid_my_sum', 'name2_fid_my_sum',
                                            'name1_fid_mean', 'name2_fid_mean']
        assert result == expected

    def subtest6():
        expr7 = expr.pivot_table(rows='id', values='fid', columns='name', aggfunc=['mean', 'sum']).cache()
        assert len(setup.engine.execute(expr7)) == 3

    def subtest7():
        expr5 = setup.expr.pivot_table(rows='id', values='fid', columns='name').cache()
        expr6 = expr5[expr5['name1_fid_mean'].rename('tname1'), expr5['name2_fid_mean'].rename('tname2')]

        @output(['tname1', 'tname2'], ['float', 'float'])
        def h(row):
            yield row.tname1, row.tname2

        expr6 = expr6.map_reduce(mapper=h)
        assert len(setup.engine.execute(expr6)) == 3

    def subtest8():
        expr8 = setup.expr.pivot_table(rows='id', values='fid', columns='name')
        assert len(setup.engine.execute(expr8)) == 3
        assert not isinstance(expr8.schema, DynamicSchema)
        expr9 = (expr8['name1_fid_mean'] - expr8['name2_fid_mean']).rename('subtract')
        assert len(setup.engine.execute(expr9)) == 3
        expr10 = expr8.distinct()
        assert len(setup.engine.execute(expr10)) == 3

    def subtest9():
        expr11 = expr.pivot_table(rows='name', columns='id', values='fid', aggfunc='nunique')
        assert len(setup.engine.execute(expr11)) == 2

    run_sub_tests_in_parallel(10, [
        subtest1, subtest2, subtest3, subtest4, subtest5, subtest6, subtest7, subtest8, subtest9
    ])


def test_extract_kv(odps, setup):
    from .... import DataFrame

    data = [
        ['name1', 'k1=1,k2=3,k5=10', '1=5,3=7,2=1'],
        ['name1', '', '3=1,4=2'],
        ['name1', 'k1=7.1,k7=8.2', '1=1,5=6'],
        ['name2', 'k2=1.2,k3=1.5', None],
        ['name2', 'k9=1.1,k2=1', '4=2']
    ]

    table_name = tn('pyodps_test_mixed_engine_extract_kv')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(
        name=table_name,
        table_schema=TableSchema.from_lists(
            ['name', 'kv', 'kv2'],
            ['string', 'string', 'string']
        )
    )
    expr = DataFrame(table)
    try:
        odps.write_table(table, 0, data)

        expr1 = expr.extract_kv(columns=['kv', 'kv2'], kv_delim='=')
        res = setup.engine.execute(expr1)
        result = get_result(res)

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

        assert [c.name for c in res.columns] == expected_cols
        assert result == expected
    finally:
        table.drop()


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

    def subtest1():
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

    def subtest2():
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

    def subtest3():
        expr = setup.expr.name.value_counts(dropna=True)[:25]

        expected = [
            ['name1', 4],
            ['name2', 1]
        ]

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert expected == result

    def subtest4():
        expected = [
            ['name1', 4],
            ['name2', 1]
        ]
        expr = setup.expr.name.topk(25)

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert expected == result

    def subtest5():
        expected = [
            ['name1', 4],
            ['name2', 1]
        ]

        expr = setup.expr.groupby('name').count()

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert [it[1:] for it in expected] == result

    def subtest6():
        expected = [
            ['name1', 2],
            ['name2', 1]
        ]

        expr = setup.expr.groupby('name').id.nunique()

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert [it[1:] for it in expected] == result

    def subtest7():
        expr = setup.expr[setup.expr['id'] > 2].name.value_counts()[:25]

        expected = [
            ['name1', 4]
        ]

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert expected == result

    def subtest8():
        expr = setup.expr.groupby('name', Scalar(1).rename('constant')) \
            .agg(id=setup.expr.id.sum())

        expected = [
            ['name1', 1, 14],
            ['name2', 1, 2]
        ]

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert expected == result

    def subtest9():
        expr = setup.expr[:1]
        expr = expr.groupby('name').agg(expr.id.sum())

        res = setup.engine.execute(expr)
        result = get_result(res)

        expected = [
            ['name1', 4]
        ]

        assert expected == result

    def subtest10():
        expr = setup.expr.groupby('id').name.cat(sep=',')
        res = setup.engine.execute(expr)
        result = get_result(res)

        expected = [['name2'], ['name1,name1'], ['name1,name1']]
        assert sorted(result) == sorted(expected)

    run_sub_tests_in_parallel(
        10, [subtest1, subtest2, subtest3, subtest4, subtest5, subtest6, subtest7, subtest8, subtest9, subtest10]
    )


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

    schema2 = TableSchema.from_lists(
        ['name', 'id2', 'id3'], [types.string, types.bigint, types.bigint]
    )

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
                                            fid2=lambda x: x.fid.cummin(sort='id'))

    res = setup.engine.execute(expr['name', 'id2', 'fid2'])
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
        setup.expr.groupby('name').sort('fid').qcut(2),
        setup.expr.groupby('name').sort('fid').cume_dist(),
    ]

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [
        [4, 3, 2, 3, float(2) / 3, 1, 0.75],
        [2, 1, 1, 1, 0.0, 0, 1.0],
        [4, 3, 3, 4, float(2) / 3, 0, 0.5],
        [3, 1, 4, 2, float(0) / 3, 0, 0.25],
        [3, 1, 1, 1, float(0) / 3, 1, 1.0]
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
    data = setup.gen_data(rows=5, value_range=(-100, 100), nullable_field='name')

    def stats(col, func):
        col_idx = [idx for idx, cn in enumerate(setup.expr.schema.names) if cn == col][0]
        return func([r[col_idx] for r in data if r[col_idx] is not None])

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

    def quantile(vct, percent):
        if not vct:
            return None
        if isinstance(percent, (list, set)):
            return [quantile(vct, p) for p in percent]
        vct = sorted(vct)
        k = (len(vct) - 1) * percent
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vct[int(k)]
        d0 = vct[int(f)] * (c - k)
        d1 = vct[int(c)] * (k - f)
        return d0 + d1

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
        (partial(stats, 'id', mean), setup.expr.id.mean()),
        (partial(len, data), setup.expr.count()),
        (partial(stats, 'id', var), setup.expr.id.var(ddof=0)),
        (partial(stats, 'id', lambda x: var(x, 1)), setup.expr.id.var(ddof=1)),
        (partial(stats, 'id', std), setup.expr.id.std(ddof=0)),
        (partial(stats, 'id', lambda x: moment(x, 3, central=True)), setup.expr.id.moment(3, central=True)),
        (partial(stats, 'id', skew), setup.expr.id.skew()),
        (partial(stats, 'id', kurtosis), setup.expr.id.kurtosis()),
        (partial(stats, 'id', median), setup.expr.id.median()),
        (partial(stats, 'id', lambda x: quantile(x, 0.3)), setup.expr.id.quantile(0.3)),
        (partial(stats, 'id', lambda x: quantile(x, [0.3, 0.6])), setup.expr.id.quantile([0.3, 0.6])),
        (partial(stats, 'id', sum), setup.expr.id.sum()),
        (partial(stats, 'id', min), setup.expr.id.min()),
        (partial(stats, 'id', max), setup.expr.id.max()),
        (partial(stats, 'isMale', min), setup.expr.isMale.min()),
        (partial(stats, 'name', len), setup.expr.name.count()),
        (partial(stats, 'name', max), setup.expr.name.max()),
        (partial(stats, 'birth', max), setup.expr.birth.max()),
        (partial(stats, 'isMale', sum), setup.expr.isMale.sum()),
        (partial(stats, 'isMale', any), setup.expr.isMale.any()),
        (partial(stats, 'isMale', all), setup.expr.isMale.all()),
        (partial(stats, 'name', nunique), setup.expr.name.nunique()),
        (partial(stats, 'name', cat), setup.expr.name.cat(sep='|').map(lambda x: len(x.split('|')), rtype='int')),
        (partial(stats, 'id', mean), setup.expr.id.agg(Agg, rtype='float')),
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
        if isinstance(first, float):
            assert pytest.approx(first) == second
        elif isinstance(first, list):
            assert approx_list(first) == second
        else:
            assert first == second

    expr = setup.expr['id', 'fid'].apply(Agg, types=['float'] * 2)

    expected = [[mean([l[1] for l in data])], [mean([l[2] for l in data])]]

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

    @output_types('float')
    class Aggregator2(object):
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

    def subtest1():
        expr = setup.expr.id.agg(Aggregator)
        expected = float(16) / 5

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert pytest.approx(expected) == result

    def subtest2():
        expr = setup.expr.id.unique().agg(Aggregator)
        expected = float(9) / 3

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert pytest.approx(expected) == result

    def subtest3():
        expr = setup.expr.groupby(Scalar('const').rename('s')).id.agg(Aggregator)
        expected = float(16) / 5

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert pytest.approx(expected) == result[0][0]

    def subtest4():
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

    def subtest5():
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

    def subtest6():
        expr = setup.expr[setup.expr.name, Scalar(1).rename('id')]
        expr = expr.groupby('name').agg(expr.id.sum())

        expected = [
            ['name1', 4],
            ['name2', 1]
        ]

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert expected == result

    def subtest7():
        expr = agg([setup.expr['fid'], setup.expr['id']], Aggregator2).rename('agg')

        expected = sum(r[2] for r in data) / sum(r[1] for r in data)
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert pytest.approx(expected) == result

    run_sub_tests_in_parallel(
        10, [subtest1, subtest2, subtest3, subtest4, subtest5, subtest6, subtest7]
    )


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

    expr = setup.expr['name', ].apply(
        mapper, axis=1, names=['word', 'count'], types=['string', 'int'])
    expr = expr.groupby('word').sort('word').apply(
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

    expr = setup.expr['name', ].map_reduce(functools.partial(mapper, mul=2), reducer, group='word')

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [['key', 6], ['name', 8]]
    assert sorted(result) == sorted(expected)

    @output(['word', 'cnt'], ['string', 'int'])
    class reducer2(object):
        def __init__(self, keys):
            self.cnt = 0

        def __call__(self, row, done):
            self.cnt += row.cnt
            if done:
                yield row.word, self.cnt

    expr = setup.expr['name', ].map_reduce(mapper, reducer2, group='word')

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [['key', 3], ['name', 4]]
    assert sorted(result) == sorted(expected)

    # test both combiner and reducer
    expr = setup.expr['name',].map_reduce(mapper, reducer, combiner=reducer2, group='word')

    res = setup.engine.execute(expr)
    result = get_result(res)

    assert sorted(result) == sorted(expected)


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

    schema2 = TableSchema.from_lists(
        ['name2', 'id2', 'id3'], [types.string, types.bigint, types.bigint]
    )

    table_name = tn('pyodps_test_engine_table2')
    odps.delete_table(table_name, if_exists=True)
    table2 = odps.create_table(name=table_name, table_schema=schema2)
    expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

    setup.gen_data(data=data)

    data2 = [
        ['name1', 4, -1],
    ]

    odps.write_table(table2, 0, data2)

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


def test_scale_map_reduce(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    scaled = setup.expr.min_max_scale()

    @output(scaled.schema.names, scaled.schema.types)
    def h(row):
        yield row

    df = scaled.map_reduce(mapper=h)

    assert len(setup.engine.execute(df)) == 5


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

    expr = setup.expr['name', ].groupby('name').sort('name').apply(reducer)

    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [['key', 2], ['name', 3]]
    assert sorted(expected) == sorted(result)


def test_join(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]

    schema2 = TableSchema.from_lists(
        ['name', 'id2', 'id3'], [types.string, types.bigint, types.bigint]
    )
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

    def subtest1():
        expr = setup.expr.join(expr2).join(expr2)['name', 'id2']

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert len(result) == 5
        expected = [
            [to_text('name1'), 4],
            [to_text('name2'), 1]
        ]
        assert all(it in expected for it in result) is True

    def subtest2():
        expr = setup.expr.join(expr2, on=['name', ('id', 'id2')])[setup.expr.name, expr2.id2]
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert len(result) == 2
        expected = [to_text('name1'), 4]
        assert all(it == expected for it in result) is True

    def subtest3():
        expr = setup.expr.join(expr2, on=['name', expr2.id2 == setup.expr.id])[setup.expr.name, expr2.id2]
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert len(result) == 2
        expected = [to_text('name1'), 4]
        assert all(it == expected for it in result) is True

    def subtest4():
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

    def subtest5():
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

    def subtest6():
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

    def subtest7():
        grouped = setup.expr.groupby('name').agg(new_id=setup.expr.id.sum()).cache()
        setup.engine.execute(setup.expr.join(grouped, on='name'))

        expr = setup.expr.join(expr2, on=['name', ('id', 'id2')])[
            lambda x: x.groupby(Scalar(1)).sort('name').row_number(), ]
        setup.engine.execute(expr)

    def subtest8():
        expr = setup.expr.id.join(expr2.id2)
        res = setup.engine.execute(expr)
        result = get_result(res)
        expected = [[4, 4], [4, 4]]
        assert len(result) == 2
        assert all(it in expected for it in result) is True

    try:
        run_sub_tests_in_parallel(
            10, [subtest1, subtest2, subtest3, subtest4, subtest5, subtest6, subtest7, subtest8]
        )
    finally:
        table2.drop()


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

    schema2 = TableSchema.from_lists(
        ['name', 'id2', 'id3'], [types.string, types.bigint, types.bigint]
    )
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
        ['name2', 3, 1.5],
        ['name1', 4, 4.2],
        ['name1', 3, 2.2],
        ['name1', 3, 4.1],
    ]
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid'], [types.string, types.bigint, types.double]
    )
    table_name = tn('pyodps_test_engine_scale_table')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(name=table_name, table_schema=schema)
    odps.write_table(table_name, 0, data)
    expr_input = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))

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

    table_name = tn('pyodps_test_engine_bf_table2')
    odps.delete_table(table_name, if_exists=True)
    table2 = odps.create_table(name=table_name, table_schema=schema2)
    expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

    odps.write_table(table2, 0, data2)

    try:
        expr = setup.expr.bloom_filter('name', expr2.name, capacity=10)

        res = setup.engine.execute(expr)
        result = get_result(res)

        assert all(r[0] != 'name2' for r in result) is True
    finally:
        table2.drop()


def test_persist(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    base_table_name = tn('pyodps_test_engine_persist_table')

    def simple_persist_test(table_name):
        odps.delete_table(table_name, if_exists=True)
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
    def persist_existing_table_test(table_name):
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

            res = setup.engine.execute(df)
            result = get_result(res)
            assert len(result) == 5
            assert data == [[r[0], r[2], r[1], None, None, None] for r in result]
        finally:
            odps.delete_table(table_name, if_exists=True)

    def persist_with_part_test(table_name):
        odps.delete_table(table_name, if_exists=True)
        try:
            df = setup.engine.persist(setup.expr, table_name, partition={'ds': 'today'})

            res = setup.engine.execute(df)
            result = get_result(res)
            assert len(result) == 5
        finally:
            odps.delete_table(table_name, if_exists=True)

    def persist_with_create_part_test(table_name):
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

    def persist_with_create_part_test2(table_name):
        odps.delete_table(table_name, if_exists=True)
        try:
            schema = TableSchema.from_lists(setup.schema.names, setup.schema.types, ['dsi'], ['bigint'])
            table = odps.create_table(table_name, schema)
            table.create_partition("dsi='00'")
            df = setup.engine.persist(setup.expr, table_name, partition="dsi='00'", create_partition=True)

            res = setup.engine.execute(df)
            result = get_result(res)
            assert len(result) == 5
            assert data == [d[:-1] for d in result]

            df2 = setup.engine.persist(setup.expr[setup.expr.id.astype('float'), 'name'], table_name,
                                       partition="dsi='01'", create_partition=True, cast=True)

            res = setup.engine.execute(df2)
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

    def persist_with_dynamic_parts_test(table_name):
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

    sub_tests = [
        simple_persist_test,
        persist_existing_table_test,
        persist_with_part_test,
        persist_with_create_part_test,
        persist_with_create_part_test2,
        persist_with_create_multi_part_test,
        persist_with_dynamic_parts_test,
    ]
    base_table_name = tn('pyodps_test_pd_schema_persist_table')
    run_sub_tests_in_parallel(
        10,
        [
            functools.partial(sub_test, base_table_name + "_%d" % idx)
            for idx, sub_test in enumerate(sub_tests)
        ]
    )


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
    schema = TableSchema.from_lists(
        ['name'] + kv_cols, [odps_types.string] + [odps_types.double] * 6
    )
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
            ['name1', 'k1=1.0,k2=3.0,k5=10.0'],
            ['name1', 'k2=3.0,k3=5.1'],
            ['name1', 'k1=7.1,k7=8.2'],
            ['name2', 'k2=1.2,k3=1.5'],
            ['name2', 'k2=1.0,k9=1.1'],
        ]

        assert result == expected
    finally:
        table.drop()


def test_melt(odps, setup):
    data = [
        ['name1', 1.0, 3.0, 10.0],
        ['name1', None, 3.0, 5.1],
        ['name1', 7.1, None, 8.2],
        ['name2', None, 1.2, 1.5],
        ['name2', None, 1.0, 1.1],
    ]
    schema = TableSchema.from_lists(
        ['name', 'k1', 'k2', 'k3'],
        [types.string, types.double, types.double, types.double],
    )
    table_name = tn('pyodps_test_engine_melt')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(name=table_name, table_schema=schema)
    expr = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))
    try:
        odps.write_table(table, 0, data)
        expr1 = expr.melt('name')

        res = setup.engine.execute(expr1)
        result = get_result(res)

        expected = [
            ['name1', 'k1', 1.0], ['name1', 'k2', 3.0], ['name1', 'k3', 10.0],
            ['name1', 'k1', None], ['name1', 'k2', 3.0], ['name1', 'k3', 5.1],
            ['name1', 'k1', 7.1], ['name1', 'k2', None], ['name1', 'k3', 8.2],
            ['name2', 'k1', None], ['name2', 'k2', 1.2], ['name2', 'k3', 1.5],
            ['name2', 'k1', None], ['name2', 'k2', 1.0], ['name2', 'k3', 1.1]
        ]

        assert result == expected
    finally:
        table.drop()


def test_collection_na(odps, setup):
    from .....compat import reduce

    data = [
        [0, 'name1', 1.0, None, 3.0, 4.0],
        [1, 'name1', 2.0, None, None, 1.0],
        [2, 'name1', 3.0, 4.0, 1.0, None],
        [3, 'name1', float('nan'), 1.0, 2.0, 3.0],
        [4, 'name1', 1.0, None, 3.0, 4.0],
        [5, 'name1', 1.0, 2.0, 3.0, 4.0],
        [6, 'name1', None, None, None, None],
    ]

    schema = TableSchema.from_lists(
        ['rid', 'name', 'f1', 'f2', 'f3', 'f4'],
        [types.bigint, types.string] + [types.double] * 4,
    )

    table_name = tn('pyodps_test_engine_collection_na')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(name=table_name, table_schema=schema)
    expr = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))

    try:
        odps.write_table(table, 0, data)

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

        ures = setup.engine.execute(uexpr)
        uresult = get_result(ures)

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
        assert uresult == expected
    finally:
        table.drop()


def test_drop(odps, setup):
    data1 = [
        ['name1', 1, 3.0], ['name1', 2, 3.0], ['name1', 2, 2.5],
        ['name2', 1, 1.2], ['name2', 3, 1.0],
        ['name3', 1, 1.2], ['name3', 3, 1.2],
    ]
    schema1 = TableSchema.from_lists(
        ['name', 'id', 'fid'], [types.string, types.bigint, types.double]
    )
    table_name1 = tn('pyodps_test_engine_drop1')
    odps.delete_table(table_name1, if_exists=True)
    table1 = odps.create_table(name=table_name1, table_schema=schema1)
    expr1 = CollectionExpr(_source_data=table1, _schema=odps_schema_to_df_schema(schema1))

    data2 = [
        ['name1', 1], ['name1', 2],
        ['name2', 1], ['name2', 2],
    ]
    schema2 = TableSchema.from_lists(['name', 'id'], [types.string, types.bigint])
    table_name2 = tn('pyodps_test_engine_drop2')
    odps.delete_table(table_name2, if_exists=True)
    table2 = odps.create_table(name=table_name2, table_schema=schema2)
    expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))
    try:
        odps.write_table(table1, 0, data1)
        odps.write_table(table2, 0, data2)

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
    finally:
        table1.drop()
        table2.drop()


def test_except_intersect(odps, setup):
    data1 = [
        ['name1', 1], ['name1', 2], ['name1', 2], ['name1', 2],
        ['name2', 1], ['name2', 3],
        ['name3', 1], ['name3', 3],
    ]
    schema1 = TableSchema.from_lists(['name', 'id'], [types.string, types.bigint])
    table_name1 = tn('pyodps_test_engine_drop1')
    odps.delete_table(table_name1, if_exists=True)
    table1 = odps.create_table(name=table_name1, table_schema=schema1)
    expr1 = CollectionExpr(_source_data=table1, _schema=odps_schema_to_df_schema(schema1))

    data2 = [
        ['name1', 1], ['name1', 2], ['name1', 2],
        ['name2', 1], ['name2', 2],
    ]
    schema2 = TableSchema.from_lists(['name', 'id'], [types.string, types.bigint])
    table_name2 = tn('pyodps_test_engine_drop2')
    odps.delete_table(table_name2, if_exists=True)
    table2 = odps.create_table(name=table_name2, table_schema=schema2)
    expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))
    try:
        odps.write_table(table1, 0, data1)
        odps.write_table(table2, 0, data2)

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
    finally:
        table1.drop()
        table2.drop()


def test_filter_order(odps, setup):
    table_name = tn('pyodps_test_sql_engine_filter_order')
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


def test_lateral_view(odps, setup):
    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
    )
    odps_schema = df_schema_to_odps_schema(schema)
    table_name = tn('pyodps_test_engine_lateral_view1')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(name=table_name, table_schema=odps_schema)
    expr_in = CollectionExpr(_source_data=table, _schema=schema)

    @output(['name', 'id'], ['string', 'int64'])
    def mapper(row):
        for idx in range(row.id):
            yield '%s_%d' % (row.name, idx), row.id * idx

    @output(['bin_id'], ['string'])
    def mapper2(row):
        for idx in range(row.id % 2 + 1):
            yield str(idx)

    @output(['bin_id'], ['string'])
    def mapper3(row):
        for idx in range(row.ren_id % 2 + 1):
            yield str(idx)

    @output(['bin_id'], ['string'])
    def mapper4(row):
        for idx in range(row.id % 2):
            yield str(idx)

    @output(['name', 'id'], ['string', 'float'])
    def reducer(keys):
        sums = [0.0]

        def h(row, done):
            sums[0] += row.fid
            if done:
                yield tuple(keys) + (sums[0], )

        return h

    def subtest1():
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

    def subtest2():
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

    def subtest3():
        expected = [
            [5, 3.5, 'name2_0', 0, '0'], [5, 3.5, 'name2_1', 2, '0'],
            [5, 2.2, 'name1_0', 0, '0'], [5, 2.2, 'name1_0', 0, '1'], [5, 2.2, 'name1_1', 3, '0'],
            [5, 2.2, 'name1_1', 3, '1'], [5, 2.2, 'name1_2', 6, '0'], [5, 2.2, 'name1_2', 6, '1'],
            [5, 4.1, 'name1_0', 0, '0'], [5, 4.1, 'name1_0', 0, '1'], [5, 4.1, 'name1_1', 3, '0'],
            [5, 4.1, 'name1_1', 3, '1'], [5, 4.1, 'name1_2', 6, '0'], [5, 4.1, 'name1_2', 6, '1'],
        ]

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in['id', ].apply(mapper2, axis=1)]
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == expected

    def subtest4():
        expected = [
            [5, 3.5, 'name2_0', 0, '0'], [5, 3.5, 'name2_1', 2, '0'],
            [5, 2.2, 'name1_0', 0, '0'], [5, 2.2, 'name1_0', 0, '1'], [5, 2.2, 'name1_1', 3, '0'],
            [5, 2.2, 'name1_1', 3, '1'], [5, 2.2, 'name1_2', 6, '0'], [5, 2.2, 'name1_2', 6, '1'],
            [5, 4.1, 'name1_0', 0, '0'], [5, 4.1, 'name1_0', 0, '1'], [5, 4.1, 'name1_1', 3, '0'],
            [5, 4.1, 'name1_1', 3, '1'], [5, 4.1, 'name1_2', 6, '0'], [5, 4.1, 'name1_2', 6, '1'],
        ]

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in[expr_in.id.rename('ren_id'), ].apply(mapper3, axis=1)]
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == expected

    def subtest5():
        expected = [
            ['name1_0', 12.6], ['name1_1', 12.6], ['name1_2', 12.6],
            ['name2_0', 3.5], ['name2_1', 3.5]
        ]

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in['id', ].apply(mapper2, axis=1)]
        expr = expr.map_reduce(reducer=reducer, group='name')
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert approx_list(sorted(result)) == sorted(expected)

    def subtest6():
        expected = [
            [5, 2.2, 'name1_0', 0, '0'], [5, 2.2, 'name1_1', 3, '0'],
            [5, 2.2, 'name1_2', 6, '0'], [5, 4.1, 'name1_0', 0, '0'],
            [5, 4.1, 'name1_1', 3, '0'], [5, 4.1, 'name1_2', 6, '0'],
        ]

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in['id', ].apply(mapper4, axis=1)]
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == expected

    def subtest7():
        expected = [
            [5, 3.5, 'name2_0', 0, None], [5, 3.5, 'name2_1', 2, None],
            [5, 2.2, 'name1_0', 0, '0'], [5, 2.2, 'name1_1', 3, '0'],
            [5, 2.2, 'name1_2', 6, '0'], [5, 4.1, 'name1_0', 0, '0'],
            [5, 4.1, 'name1_1', 3, '0'], [5, 4.1, 'name1_2', 6, '0']
        ]

        expr = expr_in[expr_in.id < 4][Scalar(5).rename('five'), expr_in.fid,
                                       expr_in['name', 'id'].apply(mapper, axis=1),
                                       expr_in['id', ].apply(mapper4, axis=1, keep_nulls=True)]
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == expected

    try:
        odps.write_table(table_name, data)
        run_sub_tests_in_parallel(
            10, [subtest1, subtest2, subtest3, subtest4, subtest5, subtest6, subtest7]
        )
    finally:
        table.drop()


def test_composites(odps, setup):
    data = [
        ['name1', 4, 5.3, {'a': 123.2, 'b': 567.1}, ['YTY', 'HKG', 'SHA', 'PEK']],
        ['name2', 2, 3.5, {'c': 512.1, 'b': 711.2}, None],
        ['name1', 4, 4.2, None, ['Hawaii', 'Texas']],
        ['name1', 3, 2.2, {'u': 115.4, 'v': 312.1}, ['Washington', 'London', 'Paris', 'Frankfort']],
        ['name1', 3, 4.1, {'w': 923.2, 'x': 456.1}, ['Moscow', 'Warsaw', 'Prague', 'Belgrade']],
    ]

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['name', 'grade', 'score', 'detail', 'locations'],
        datatypes('string', 'int64', 'float64', 'dict<string, float64>', 'list<string>')
    )
    odps_schema = df_schema_to_odps_schema(schema)
    table_name = tn('pyodps_test_engine_composites1')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(name=table_name, table_schema=odps_schema)
    expr_in = CollectionExpr(_source_data=table, _schema=schema)

    def subtest1():
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

    def subtest2():
        expected = [
            ['name1', 3, 0, 'Washington', 'u', 115.4],
            ['name1', 3, 0, 'Washington', 'v', 312.1],
            ['name1', 3, 1, 'London', 'u', 115.4],
            ['name1', 3, 1, 'London', 'v', 312.1],
            ['name1', 3, 2, 'Paris', 'u', 115.4],
            ['name1', 3, 2, 'Paris', 'v', 312.1],
            ['name1', 3, 3, 'Frankfort', 'u', 115.4],
            ['name1', 3, 3, 'Frankfort', 'v', 312.1]
        ]

        expr = expr_in[expr_in.score < 4][expr_in.name, expr_in.grade, expr_in.locations.explode(pos=True),
                                          expr_in.detail.explode()]
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == expected

    def subtest3():
        expected = [
            ['name1', 123.2], ['name1', 567.1],
            ['name2', 711.2], ['name2', 512.1],
            ['name1', 115.4], ['name1', 312.1],
            ['name1', 923.2], ['name1', 456.1]
        ]

        expr = expr_in[expr_in.name, expr_in.detail.values().explode()]
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert sorted(result) == sorted(expected)

    def subtest4():
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

    def subtest5():
        expected = [
            ['name1', [4, 4, 3, 3]],
            ['name2', [2]]
        ]

        expr = expr_in.groupby(expr_in.name).agg(agg_grades=expr_in.grade.tolist())
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == expected

    def subtest6():
        expected = [
            ['name1', [3, 4]],
            ['name2', [2]]
        ]

        expr = expr_in.groupby(expr_in.name).agg(agg_grades=expr_in.grade.tolist(unique=True).sort())
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == expected

    def subtest7():
        expected = [[['name1', 'name2', 'name1', 'name1', 'name1'], [4, 2, 4, 3, 3]]]

        expr = expr_in['name', 'grade'].tolist()
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == expected

    def subtest8():
        expected = [[['name1', 'name2'], [2, 3, 4]]]

        expr = expr_in['name', 'grade'].tolist(unique=True)
        expr = expr[tuple(f.sort() for f in expr.columns)]
        res = setup.engine.execute(expr)
        result = get_result(res)
        assert result == expected

    try:
        odps.write_table(table_name, data)
        run_sub_tests_in_parallel(
            10, [subtest1, subtest2, subtest3, subtest4, subtest5, subtest6, subtest7, subtest8]
        )
    finally:
        table.drop()


def test_string_splits(odps, setup):
    data = [
        ['name1:a,name3:5', 4],
        ['name2:4,name7:1', 2],
        ['name1:1', 4],
        ['name1:4,name5:6,name4:1', 3],
        ['name1:2,name10:1', 3],
    ]

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(['name', 'id'], datatypes('string', 'int64'))
    odps_schema = df_schema_to_odps_schema(schema)
    table_name = tn('pyodps_test_engine_composites1')
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(name=table_name, table_schema=odps_schema)
    expr_in = CollectionExpr(_source_data=table, _schema=schema)

    try:
        odps.write_table(table_name, data)

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
    finally:
        table.drop()


def test_map_complex_functions(odps, setup):
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    table_name = tn("test_map_complex_functions_table")
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(
        table_name,
        "col1 array<string>, col2 map<string, string>, col3 struct<attr: string, value: bigint>",
    )
    schema = TableSchema.from_lists(
        ['col1', 'col2', 'col3'],
        datatypes('list<string>', 'dict<string, string>', 'struct<attr: string, value: int64>'),
    )

    data = [
        [["abcd", "efgh"], {"k1": "val1", "k2": "val2"}, ("uvw", 123)],
        [["uvwx", "yz"], {"k3": "pt", "k4": "xyz"}, ("zyx", 456)],
        [["mprz", "uw"], {"k5": "dz", "k6": "mvw"}, ("wez", 789)],
        [[";lol", "te"], {"k7": "utz", "k8": "ore"}, ("exz", 191)],
    ]
    for row in data:
        row[-1] = schema.types[-1].namedtuple_type(*row[-1])
    odps.write_table(table, data)

    def f1(value):
        return ",".join(value)

    def f2(value):
        return ",".join(k + ":" + v for k, v in value.items())

    def f3(value):
        return "attr:" + value.attr + ",value:" + str(value.value)

    try:
        expr_in = CollectionExpr(_source_data=table, _schema=schema)
        expr = expr_in[
            expr_in.col1.map(f1, rtype="string"),
            expr_in.col2.map(f2, rtype="string"),
            expr_in.col3.map(f3, rtype="string"),
        ]
        res = setup.engine.execute(expr)
        result = get_result(res)
        expected = [[f1(v1), f2(v2), f3(v3)] for v1, v2, v3 in data]
        assert result == expected
    finally:
        table.drop()
