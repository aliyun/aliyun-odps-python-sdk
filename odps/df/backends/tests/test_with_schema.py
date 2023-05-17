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
import uuid
from collections import namedtuple
from datetime import datetime

import pytest

from ....compat import Decimal
from ....errors import ODPSError
from ....models import TableSchema
from ....tests.core import force_drop_schema, tn, pandas_case, get_result, run_sub_tests_in_parallel
from ... import CollectionExpr
from ...types import validate_data_type
from ..odpssql.tests.test_engine import ODPSEngine, FakeBar
from ..odpssql.types import df_schema_to_odps_schema
from ..pd.engine import PandasEngine

TEST_CLS_SCHEMA_NAME = tn("pyodps_test_cls_df_schema")


@pytest.fixture(scope="module")
def setup_cls_schema(odps_with_schema):
    if odps_with_schema.exist_schema(TEST_CLS_SCHEMA_NAME):
        cls_schema = odps_with_schema.get_schema(TEST_CLS_SCHEMA_NAME)
    else:
        cls_schema = odps_with_schema.create_schema(TEST_CLS_SCHEMA_NAME)

    try:
        yield
    finally:
        force_drop_schema(cls_schema)


@pytest.fixture
def setup(odps_with_schema, setup_cls_schema):
    def gen_data(data=None):
        odps_with_schema.write_table(table, 0, data)
        return data

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    df_schema = TableSchema.from_lists(
        ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
    )
    table_schema = df_schema_to_odps_schema(df_schema)
    table_name = tn('pyodps_test_engine_table_%s' % str(uuid.uuid4()).replace('-', '_'))
    odps_with_schema.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)
    table = odps_with_schema.create_table(
        table_name, table_schema, schema=TEST_CLS_SCHEMA_NAME, lifecycle=1
    )
    expr = CollectionExpr(_source_data=table, _schema=df_schema)
    faked_bar = FakeBar()

    nt = namedtuple("NT", "expr table table_schema faked_bar gen_data")
    return nt(expr, table, table_schema, faked_bar, gen_data)


def test_simple_call(odps_with_schema, setup):
    odps = odps_with_schema

    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    df = setup.table.to_df()

    result_table_name = tn('pyodps_test_odps_simple_table')
    try:
        df2 = df['name', 'fid', df.id.astype('int32'), 'ismale', 'scale', 'birth']
        df3 = df2.persist(result_table_name, schema=TEST_CLS_SCHEMA_NAME, lifecycle=1)

        assert odps.exist_table(result_table_name, schema=TEST_CLS_SCHEMA_NAME) is True

        res = df3.execute()
        result = get_result(res)
        assert len(result) == 5
    finally:
        odps.delete_table(result_table_name, schema=TEST_CLS_SCHEMA_NAME, if_exists=True)


def test_odps_engine(odps_with_schema, setup):
    odps = odps_with_schema
    engine = ODPSEngine(odps)

    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    setup.gen_data(data=data)

    # tests for tunnel engine
    expr = setup.expr.count()
    res = engine._handle_cases(expr, setup.faked_bar)
    result = get_result(res)
    assert 5 == result

    def simple_persist_test(table_name):
        odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)
        try:
            with pytest.raises(ODPSError):
                engine.persist(setup.expr, table_name, create_table=False, schema=TEST_CLS_SCHEMA_NAME)

            df = engine.persist(setup.expr, table_name, schema=TEST_CLS_SCHEMA_NAME)
            assert odps.exist_table(table_name, schema=TEST_CLS_SCHEMA_NAME) is True

            res = engine.execute(df, schema=TEST_CLS_SCHEMA_NAME)
            result = get_result(res)
            assert len(result) == 5
            assert data == result

            with pytest.raises(ValueError):
                engine.persist(setup.expr, table_name, create_partition=True, schema=TEST_CLS_SCHEMA_NAME)
            with pytest.raises(ValueError):
                engine.persist(setup.expr, table_name, drop_partition=True, schema=TEST_CLS_SCHEMA_NAME)
        finally:
            odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)

    def persist_over_existing_test(table_name):
        odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)
        try:
            odps.create_table(
                table_name,
                'name string, fid double, id bigint, isMale boolean, scale decimal, birth datetime',
                schema=TEST_CLS_SCHEMA_NAME,
                lifecycle=1,
            )

            expr = setup.expr['name', 'fid', setup.expr.id.astype('int32'), 'isMale', 'scale', 'birth']
            df = engine.persist(expr, table_name, schema=TEST_CLS_SCHEMA_NAME)

            res = engine.execute(df, schema=TEST_CLS_SCHEMA_NAME)
            result = get_result(res)
            assert len(result) == 5
            assert data == [[r[0], r[2], r[1], None, None, None] for r in result]
        finally:
            odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)

    def persist_with_create_part_test(table_name):
        odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)
        try:
            t_schema = TableSchema.from_lists(setup.table_schema.names, setup.table_schema.types, ['ds'], ['string'])
            odps.create_table(table_name, t_schema, schema=TEST_CLS_SCHEMA_NAME)
            df = engine.persist(
                setup.expr, table_name, partition='ds=today', create_partition=True, schema=TEST_CLS_SCHEMA_NAME
            )
            assert odps.exist_table(table_name, schema=TEST_CLS_SCHEMA_NAME) is True

            res = engine.execute(df, schema=TEST_CLS_SCHEMA_NAME)
            result = get_result(res)
            assert len(result) == 5
            assert data == [d[:-1] for d in result]
        finally:
            odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)

    def persist_with_existing_part_test(table_name):
        odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)
        try:
            t_schema = TableSchema.from_lists(setup.table_schema.names, setup.table_schema.types, ['dsi'], ['bigint'])
            table = odps.create_table(table_name, t_schema, schema=TEST_CLS_SCHEMA_NAME)
            table.create_partition("dsi='00'")
            df = engine.persist(
                setup.expr,
                table_name,
                partition="dsi='00'",
                create_partition=True,
                schema=TEST_CLS_SCHEMA_NAME,
            )
            assert odps.exist_table(table_name, schema=TEST_CLS_SCHEMA_NAME) is True

            res = engine.execute(df, schema=TEST_CLS_SCHEMA_NAME)
            result = get_result(res)
            assert len(result) == 5
            assert data == [d[:-1] for d in result]
        finally:
            odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)

    def persist_with_dynamic_parts_test(table_name):
        try:
            engine.persist(setup.expr, table_name, partitions=['name'], schema=TEST_CLS_SCHEMA_NAME)

            assert odps.exist_table(table_name, schema=TEST_CLS_SCHEMA_NAME) is True

            t = odps.get_table(table_name, schema=TEST_CLS_SCHEMA_NAME)
            assert 2 == len(list(t.partitions))
            with t.open_reader(partition='name=name1', reopen=True) as r:
                assert 4 == r.count
            with t.open_reader(partition='name=name2', reopen=True) as r:
                assert 1 == r.count
        finally:
            odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)

    sub_tests = [
        simple_persist_test,
        persist_over_existing_test,
        persist_with_create_part_test,
        persist_with_existing_part_test,
        persist_with_dynamic_parts_test,
    ]
    base_table_name = tn('pyodps_test_odps_schema_persist_table')
    run_sub_tests_in_parallel(
        10,
        [
            functools.partial(sub_test, base_table_name + "_%d" % idx)
            for idx, sub_test in enumerate(sub_tests)
        ]
    )


@pandas_case
def test_pandas_engine(odps_with_schema, setup):
    import pandas as pd

    odps = odps_with_schema
    engine = PandasEngine(odps)
    odps_engine = ODPSEngine(odps)

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime')
    )
    table_schema = df_schema_to_odps_schema(schema)

    data = [
        ['name1', 4, 5.3, True, Decimal('3.14'), datetime(1999, 5, 25, 3, 10)],
        ['name2', 2, 3.5, False, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]
    expr = CollectionExpr(_source_data=pd.DataFrame(data), _schema=schema)

    def simple_persist_test(table_name):
        odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)
        try:
            with pytest.raises(ODPSError):
                engine.persist(expr, table_name, create_table=False, schema=TEST_CLS_SCHEMA_NAME)

            df = engine.persist(expr, table_name, schema=TEST_CLS_SCHEMA_NAME)

            res = df.to_pandas(schema=TEST_CLS_SCHEMA_NAME)
            result = get_result(res)
            assert len(result) == 5
            assert data == result

            with pytest.raises(ValueError):
                engine.persist(expr, table_name, create_partition=True, schema=TEST_CLS_SCHEMA_NAME)
            with pytest.raises(ValueError):
                engine.persist(expr, table_name, drop_partition=True, schema=TEST_CLS_SCHEMA_NAME)
        finally:
            odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)

    def persist_over_existing_test(table_name):
        try:
            odps.create_table(
                table_name,
                'name string, fid double, id bigint, isMale boolean, scale decimal, birth datetime',
                schema=TEST_CLS_SCHEMA_NAME,
                lifecycle=1,
            )

            expr1 = expr['name', 'fid', expr.id.astype('int32'), 'isMale', 'scale', 'birth']
            df = engine.persist(expr1, table_name, schema=TEST_CLS_SCHEMA_NAME)

            res = df.to_pandas(schema=TEST_CLS_SCHEMA_NAME)
            result = get_result(res)
            assert len(result) == 5
            assert data == [[r[0], r[2], r[1], r[3], r[4], r[5]] for r in result]
        finally:
            odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)

    def persist_with_create_part_test(table_name):
        try:
            schema = TableSchema.from_lists(table_schema.names, table_schema.types, ['ds'], ['string'])
            odps.create_table(table_name, schema, schema=TEST_CLS_SCHEMA_NAME)
            df = engine.persist(
                expr, table_name, partition='ds=today', create_partition=True, schema=TEST_CLS_SCHEMA_NAME
            )

            res = odps_engine.execute(df, schema=TEST_CLS_SCHEMA_NAME)
            result = get_result(res)
            assert len(result) == 5
            assert data == [d[:-1] for d in result]
        finally:
            odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)

    def persist_with_dynamic_parts(table_name):
        try:
            engine.persist(expr, table_name, partitions=['name'], schema=TEST_CLS_SCHEMA_NAME)

            t = odps.get_table(table_name, schema=TEST_CLS_SCHEMA_NAME)
            assert 2 == len(list(t.partitions))
            with t.open_reader(partition='name=name1', reopen=True) as r:
                assert 4 == r.count
            with t.open_reader(partition='name=name2', reopen=True) as r:
                assert 1 == r.count
        finally:
            odps.delete_table(table_name, if_exists=True, schema=TEST_CLS_SCHEMA_NAME)

    sub_tests = [
        simple_persist_test,
        persist_over_existing_test,
        persist_with_create_part_test,
        persist_with_dynamic_parts,
    ]
    base_table_name = tn('pyodps_test_pd_schema_persist_table')
    run_sub_tests_in_parallel(
        10,
        [
            functools.partial(sub_test, base_table_name + "_%d" % idx)
            for idx, sub_test in enumerate(sub_tests)
        ]
    )
