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
from collections import namedtuple, OrderedDict

import pytest

from .... import types as odps_types
from ....config import options
from ....models import TableSchema, Instance
from ....tests.core import tn, pandas_case, odps2_typed_case, get_result
from ... import DataFrame, output, func
from ...utils import is_source_collection
from ..context import context
from ..engine import MixedEngine
from ..odpssql.engine import ODPSSQLEngine
from ..pd.engine import PandasEngine
from ..seahawks.models import SeahawksTable


def assertPandasEqual(df1, df2):
    from .... import types as o_types
    from ....compat import six

    from pandas.testing import assert_frame_equal

    # compare column types
    def get_odps_type(p_type):
        for data_type, builtin_type in six.iteritems(o_types._odps_primitive_to_builtin_types):
            if issubclass(p_type.type, builtin_type):
                return data_type

    types1 = [get_odps_type(dt) for dt in df1.dtypes]
    types2 = [get_odps_type(dt) for dt in df2.dtypes]
    assert list(types1) == list(types2)
    assert_frame_equal(df1, df2, check_dtype=False)


@pytest.fixture
@pandas_case
def setup(odps):
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
    if odps.exist_table(table):
        t = odps.get_table(table)
    else:
        t = odps.create_table(table, TableSchema.from_lists(names, types), lifecycle=1)
        with t.open_writer() as w:
            w.write([t.new_record(r) for r in odps_data])

    odps_df = DataFrame(t)
    pd_df = DataFrame(pd.DataFrame(pd_data, columns=names))

    engine = MixedEngine(odps)
    pd_engine = PandasEngine(odps)

    nt = namedtuple("NT", "engine t pd_df odps_df pd_engine")
    try:
        yield nt(engine, t, pd_df, odps_df, pd_engine)
    finally:
        engine._selecter.force_odps = False


def test_group_reduction(odps, setup):
    expr = setup.odps_df.select(setup.odps_df, id2=setup.odps_df.id.map(lambda x: x + 1))
    expr = expr.groupby('name').id2.sum()

    expected = [
        ['name1', 6],
        ['name2', 3]
    ]
    res = setup.engine.execute(expr)
    result = get_result(res)
    assert sorted([[r[1]] for r in expected]) == sorted(result)


def test_join(odps, setup):
    expr = setup.odps_df.join(setup.pd_df, 'name').sort('id_x')
    result = setup.engine.execute(expr).values

    df = DataFrame(setup.odps_df.to_pandas())
    expected = setup.pd_engine.execute(df.join(setup.pd_df, 'name').sort('id_x')).values
    assert result.equals(expected) is True


def test_union(odps, setup):
    expr = setup.odps_df.union(setup.pd_df).sort(['id', 'name'])
    result = setup.engine.execute(expr).values

    df = DataFrame(setup.odps_df.to_pandas())
    expected = setup.pd_engine.execute(df.union(setup.pd_df).sort(['id', 'name'])).values
    assert result.equals(expected) is True

    schema = TableSchema.from_lists(
        [c.name for c in setup.t.table_schema.columns if c.name != 'name'],
        [c.type for c in setup.t.table_schema.columns if c.name != 'name'],
        ['name'],
        ['string'],
    )
    t = odps.create_table('tmp_pyodps_%s' % str(uuid.uuid4()).replace('-', '_'), schema)
    try:
        expr = setup.odps_df.union(setup.pd_df)
        expr.persist(t.name, create_table=False, partitions=['name'])

        assert setup.engine.execute(DataFrame(t).count()) == 5

        setup.engine._selecter.force_odps = False
        df = DataFrame(t)
        assert len(setup.engine.execute(df.filter(df.name > 'a', df.name < 'b'))) >= 0
    finally:
        t.drop()


def test_is_in(odps, setup):
    expr = setup.odps_df['name'].isin(setup.pd_df['name']).rename('isin')
    result = setup.engine.execute(expr).values

    df = DataFrame(setup.odps_df.to_pandas())
    expected = setup.pd_engine.execute(df['name'].isin(setup.pd_df['name']).rename('isin')).values
    assert result.equals(expected) is True

    expr = (setup.odps_df.id + 2).isin(setup.pd_df['id']).rename('isin')
    res = setup.engine.execute(expr)
    result = get_result(res)

    expected = [[False], [False], [True]]
    assert result == expected


def test_mixed(odps, setup):
    expr = setup.odps_df.union(
        setup.odps_df.join(setup.pd_df, 'name')[
            lambda x: x.name,
            lambda x: x.id_x.rename('id')
        ]).sort(['name', 'id'])
    expr = expr[expr['name'].isin(setup.pd_df['name'])]
    expr = expr[expr, func.rand(rtype='float').rename('rand')]
    result = setup.engine.execute(expr).values[['name', 'id']]

    df = DataFrame(setup.odps_df.to_pandas())
    test_expr = df.union(
        df.join(setup.pd_df, 'name')[
            lambda x: x.name,
            lambda x: x.id_x.rename('id')
        ]).sort(['name', 'id'])
    test_expr = test_expr[test_expr['name'].isin(setup.pd_df['name'])]
    expected = setup.pd_engine.execute(test_expr).values

    assert result.equals(expected) is True


def test_pandas_persist(odps, setup):
    import pandas as pd, numpy as np

    tmp_table_name = tn('pyodps_test_mixed_persist')
    odps.delete_table(tmp_table_name, if_exists=True)
    t = odps.create_table(tmp_table_name, ('a bigint, b bigint, c bigint', 'ds string'))
    t.create_partition('ds=today')
    try:
        pd_df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc'))
        df = DataFrame(pd_df).persist(tmp_table_name, partition='ds=today', odps=odps)

        assertPandasEqual(df[list('abc')].to_pandas(), pd_df)
    finally:
        odps.delete_table(tmp_table_name)

    odps.to_global()

    tmp_table_name = tn('pyodps_test_mixed_persist2')
    odps.delete_table(tmp_table_name, if_exists=True)

    try:
        pd_df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc'))
        df = DataFrame(pd_df).persist(tmp_table_name)

        assertPandasEqual(df.to_pandas(), pd_df)
    finally:
        odps.delete_table(tmp_table_name)


@odps2_typed_case
def test_pandas_persist_odps2(odps, setup):
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

    odps.delete_table(tmp_table_name, if_exists=True)
    df.persist(tmp_table_name, lifecycle=1, drop_table=True, odps=odps)

    t = odps.get_table(tmp_table_name)
    expected_types = [odps_types.tinyint, odps_types.smallint, odps_types.int_,
                      odps_types.bigint, odps_types.float_, odps_types.double]
    assert expected_types == t.table_schema.types


def test_execute_cache_table(odps, setup):
    df = setup.odps_df[setup.odps_df.name == 'name1']
    result = df.execute().values
    assert len(result) == 2
    assert context.is_cached(df) is True

    dag = setup.engine.compile(df)
    calls = dag.topological_sort()
    assert len(calls) == 1
    assert is_source_collection(calls[0].expr) is True

    df2 = df[:5]
    result = df2.execute()
    assert len(result) == 2


def test_handle_cache(odps, setup):
    df = setup.pd_df['name', setup.pd_df.id + 1]
    df.execute()
    assert context.is_cached(df) is True

    df2 = df[df.id < 10]
    dag = setup.engine.compile(df2)
    assert len(dag.nodes()) == 1
    assert is_source_collection(dag.nodes()[0].expr.input) is True

    df3 = setup.pd_df[setup.pd_df.id < 10].count()
    i = df3.execute()
    assert context.is_cached(df3) is True

    df4 = df3 + 1
    dag = setup.engine.compile(df4)
    assert len(dag.nodes()) == 1
    assert dag.nodes()[0].expr._fields[0].lhs.value is not None
    assert df4.execute() == i + 1


def test_cache_table(odps, setup):
    setup.engine._selecter.force_odps = True

    df = setup.odps_df.join(setup.pd_df, 'name').cache()
    df2 = df.sort('id_x')

    dag = setup.engine.compile(df2)
    assert len(dag.nodes()) == 3

    result = setup.engine.execute(df2).values

    df3 = DataFrame(setup.odps_df.to_pandas())
    expected = setup.pd_engine.execute(df3.join(setup.pd_df, 'name').sort('id_x')).values
    assert result.equals(expected) is True

    assert len(setup.engine._generated_table_names) == 2

    table = context.get_cached(df)
    assert len(setup.engine.execute(df)) == len(expected)

    assert context.get_cached(df) is table
    if not isinstance(table, SeahawksTable):
        assert context.get_cached(df).lifecycle == 1

    df4 = df[df.id_x < 3].count()
    result = setup.engine.execute(df4)
    assert result == 2

    assert context.get_cached(df4) == 2


def test_use_cache(odps, setup):
    setup.engine._selecter.force_odps = True

    df_cache = setup.odps_df[setup.odps_df['name'] == 'name1'].cache()
    df = df_cache[df_cache.id * 2, df_cache.exclude('id')]
    assert len(setup.engine.execute(df, head=10)) == 2

    context.get_cached(df_cache).drop()

    assert len(setup.engine.execute(df_cache['name', df_cache.id * 2], head=10)) == 2
    assert context.is_cached(df_cache) is True
    assert odps.exist_table(context.get_cached(df_cache).name) is True


def test_head_and_tail(odps, setup):
    res = setup.odps_df.head(2)
    assert len(res) == 2

    df = setup.odps_df[setup.odps_df['name'] == 'name1']
    res = df.head(1)
    assert len(res) == 1
    assert context.is_cached(df) is True

    res = setup.odps_df.tail(2)
    assert len(res) == 2
    assert all(it > 1 for it in res.values['id']) is True

    assert len(setup.odps_df.name.head(2)) == 2
    assert len(setup.odps_df.name.tail(2)) == 2

    res = setup.pd_df.head(1)
    assert len(res) == 1

    df = setup.pd_df[setup.pd_df['name'] == 'name1']
    res = df.head(1)
    assert len(res) == 1
    assert context.is_cached(df) is True

    res = setup.pd_df.tail(1)
    assert len(res) == 1
    assert res.values['id'][0] == 6

    assert len(setup.pd_df.name.head(1)) == 1
    assert len(setup.pd_df.name.tail(1)) == 1

    class TunnelOnlyODPSEngine(ODPSSQLEngine):
        def _do_execute(self, *args, **kwargs):
            kwargs['_force_tunnel'] = True
            return super(TunnelOnlyODPSEngine, self)._do_execute(*args, **kwargs)

    engine = MixedEngine(odps)
    engine._odpssql_engine = TunnelOnlyODPSEngine(odps)

    res = engine.execute(setup.odps_df['id'], head=3)
    assert res is not None
    assert sum(res.values['id']) == 6

    table_name = tn('pyodps_df_mixed2')
    odps.delete_table(table_name, if_exists=True)
    table = next(setup.odps_df.data_source())
    table2 = odps.create_table(table_name, table.table_schema)
    try:
        res = DataFrame(table2).head(10)
        assert len(res) == 0
    finally:
        table2.drop()


def test_map_reduce_with_resource(odps, setup):
    pd_df2 = setup.odps_df.to_pandas(wrap=True)

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

    expr = pd_df2.map_reduce(reducer=reducer, reducer_resources=[setup.pd_df], group='name')
    result = expr.execute()
    assert result.values['id'].sum() == 17

    odps_df2 = setup.pd_df.persist(tn('pyodps_df_mixed2'), odps=odps)
    try:
        expr = setup.odps_df.map_reduce(reducer=reducer, reducer_resources=[odps_df2], group='name')
        result = expr.execute()
        assert result.values['id'].sum() == 17

        expr = setup.odps_df.map_reduce(reducer=reducer, reducer_resources=[setup.pd_df], group='name')
        result = expr.execute()
        assert result.values['id'].sum() == 17

        expr = pd_df2.map_reduce(reducer=reducer, reducer_resources=[odps_df2], group='name')
        result = expr.execute()
        assert result.values['id'].sum() == 17
    finally:
        next(odps_df2.data_source()).drop()


def test_bloom_filter(odps, setup):
    import numpy as np

    data2 = [
        ['name1'],
        ['name3']
    ]

    table_name = tn('pyodps_test_mixed_engine_bf_table2')
    odps.delete_table(table_name, if_exists=True)
    table2 = odps.create_table(
        name=table_name, table_schema=TableSchema.from_lists(['name'], ['string'])
    )
    expr2 = DataFrame(table2)

    odps.write_table(table2, 0, data2)

    try:
        expr = setup.odps_df.bloom_filter('name', expr2[:1].name, capacity=10)

        res = setup.engine.execute(expr)

        assert np.all(res['name'] != 'name2')
    finally:
        table2.drop()


def test_cache_persist(odps, setup):
    expr = setup.odps_df

    data2 = [
        ['name1', 3.2],
        ['name3', 2.4]
    ]

    table_name = tn('pyodps_test_mixed_engine_cp_table2')
    odps.delete_table(table_name, if_exists=True)
    table2 = odps.create_table(
        name=table_name,
        table_schema=TableSchema.from_lists(['name', 'fid'], ['string', 'double']),
    )
    expr2 = DataFrame(table2)
    odps.write_table(table2, 0, data2)

    @output(expr.schema.names, expr.schema.types)
    def h(row):
        yield row

    l = expr.filter(expr.id > 0).apply(h, axis=1).cache()
    r = expr2.filter(expr2.fid > 0)
    joined = l.join(r, on=['name', r.fid < 4])['id', 'fid'].cache()

    output_table = tn('pyodps_test_mixed_engine_cp_output_table')
    odps.delete_table(output_table, if_exists=True)
    schema = TableSchema.from_lists(['id', 'fid'], ['bigint', 'double'], ['ds'], ['string'])
    output_t = odps.create_table(output_table, schema, if_not_exists=True)

    t = joined.persist(output_table, partition='ds=today', create_partition=True)
    assert len(t.execute()) == 2

    # test seahawks fallback
    assert t.input.count().execute() == 2

    output_t.drop()


def test_bigint_partitioned_cache(odps, setup):
    table = tn('pyodps_test_bigint_partitioned_cache')
    odps.delete_table(table, if_exists=True)
    expr = setup.odps_df.persist(table, partitions=['id'])

    @output(['id', 'name'], ['int', 'string'])
    def handle(row):
        return row.id + 1, row.name

    expr = expr['tt' + expr.name, expr.id].cache()
    new_expr = expr.map_reduce(mapper=handle)

    res = setup.engine.execute(new_expr)
    assert len(res) == 3


def test_async(odps, setup):
    expr = setup.odps_df[setup.odps_df.name == 'name1']
    future = setup.engine.execute(expr, async_=True)
    assert future.done() is False
    res = future.result()
    assert len(res) == 2


def test_batch(odps, setup):
    odps_expr = setup.odps_df[setup.odps_df.id < 4].cache()
    expr = odps_expr.join(setup.pd_df, 'name').sort('id_x')

    dag = setup.engine.compile(expr)
    assert len(dag.nodes()) == 3

    f = setup.engine.execute(expr, async_=True, n_parallel=2)

    result = f.result().values

    df = DataFrame(setup.odps_df.to_pandas())
    expected = setup.pd_engine.execute(df.join(setup.pd_df, 'name').sort('id_x')).values
    assert result.equals(expected) is True


def test_batch_stop(odps, setup):
    setup.engine._selecter.force_odps = True

    expr1 = setup.odps_df[setup.odps_df.id < 3].cache()
    expr2 = setup.odps_df[setup.odps_df.id > 3].cache()
    expr3 = expr1.union(expr2)

    setup.engine.execute([expr1, expr2, expr3], n_parallel=2, async_=True)
    time.sleep(2)

    instance_ids = setup.engine._odpssql_engine._instances
    assert len(instance_ids) == 2

    setup.engine.stop()
    instances = [odps.get_instance(i) for i in instance_ids]
    [i.wait_for_completion() for i in instances]
    assert list(instances[0].get_task_statuses().values())[0].status == Instance.Task.TaskStatus.CANCELLED
    assert list(instances[1].get_task_statuses().values())[0].status == Instance.Task.TaskStatus.CANCELLED


def test_failure(odps, setup):
    from ..errors import DagDependencyError

    def err_maker(x):
        raise ValueError(x)

    expr1 = setup.odps_df[setup.odps_df.id.map(err_maker), ].cache()
    expr2 = expr1.count()

    fs = setup.engine.execute(expr2, async_=True)
    pytest.raises(DagDependencyError, fs.result)


def test_append_id_cache(odps, setup):
    options.ml.dry_run = False

    @output(['id1'] + setup.odps_df.schema.names, ['int'] + setup.odps_df.schema.types)
    def h(row):
        yield row

    expr1 = setup.odps_df.append_id(id_col='id1').apply(h, axis=1)
    expr2 = setup.odps_df.append_id(id_col='id2')
    expr3 = expr1.join(expr2, on='id')['id1', 'id2']
    assert len(expr3.execute()) == 3


def test_append_id(odps, setup):
    options.ml.dry_run = False

    expr = setup.odps_df['name', ].distinct()
    expr = expr.append_id(id_col='id2')
    expr = expr.join(setup.odps_df, on=['name'])
    tablename = tn('pyodps_test_append_id_persist')
    odps.delete_table(tablename, if_exists=True)
    expr.persist(tablename, partitions=['name'], lifecycle=1)


def test_horz_concat(odps, setup):
    options.ml.dry_run = False

    table_name = tn('test_horz_concat_table2_xxx_yyy')
    odps.delete_table(table_name, if_exists=True)

    result_table_name = tn('test_horz_concat_result')
    odps.delete_table(result_table_name, if_exists=True)

    setup.odps_df[setup.odps_df.name, (setup.odps_df.id * 2).rename('ren_id')].persist(table_name)
    df2 = odps.get_table(table_name).to_df()
    df2 = df2[:3]
    expr = setup.odps_df.concat(df2.ren_id, axis=1)
    expr.persist(result_table_name, lifecycle=1)


def test_as_type_map_reduce(odps, setup):
    expr = setup.odps_df[setup.odps_df.exclude('id'), setup.odps_df.id.astype('float')]
    expr = expr.filter(expr.id < 10)['id', 'name']

    @output(['id', 'name'], ['float', 'string'])
    def h(group):
        def inn(row, done):
            yield row

        return inn

    expr = expr.map_reduce(reducer=h)
    expr.execute()

    expr = setup.odps_df[setup.odps_df.exclude('id'), setup.odps_df.id.astype('float')]
    expr = expr.filter(expr.id < 10).distinct('id', 'name')

    @output(['id', 'name'], ['float', 'string'])
    def h(group):
        def inn(row, done):
            yield row

        return inn

    expr = expr.map_reduce(reducer=h)
    expr.execute()
