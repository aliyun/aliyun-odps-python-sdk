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

import uuid
from collections import namedtuple

import pytest

from .... import types, options
from ....df.backends.tests.core import tn
from ....tests.core import sqlalchemy_case
from ....models import TableSchema
from ...expr.expressions import CollectionExpr
from ...types import validate_data_type
from ..core import EngineTypes as Engines
from ..odpssql.types import df_schema_to_odps_schema
from ..odpssql.types import odps_schema_to_df_schema
from ..selecter import EngineSelecter
from ..tests.core import NumGenerators


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

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    pd_schema = TableSchema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                               datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
    schema = df_schema_to_odps_schema(pd_schema)
    table_name = tn('pyodps_test_selecter_table_%s' % str(uuid.uuid4()).replace('-', '_'))
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(name=table_name, table_schema=schema)
    expr = CollectionExpr(_source_data=table, _schema=pd_schema)

    class FakeBar(object):
        def update(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def status(self, *args, **kwargs):
            pass
    faked_bar = FakeBar()

    data = [
        ['name1', 4, 5.3, None, None, None],
        ['name2', 2, 3.5, None, None, None],
        ['name1', 4, 4.2, None, None, None],
        ['name1', 3, 2.2, None, None, None],
        ['name1', 3, 4.1, None, None, None],
    ]

    schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                [types.string, types.bigint, types.bigint])

    table_name = tn('pyodps_test_selecter_table2')
    odps.delete_table(table_name, if_exists=True)
    table2 = odps.create_table(name=table_name, table_schema=schema2)
    expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

    gen_data(data=data)

    data2 = [
        ['name1', 4, -1],
        ['name2', 1, -2]
    ]

    odps.write_table(table2, 0, data2)

    selecter = EngineSelecter()

    nt = namedtuple("NT", "expr expr2 selecter table")
    return nt(expr, expr2, selecter, table)


@sqlalchemy_case
def test_selecter(setup):
    src_max_size = options.df.seahawks.max_size
    src_seahawks_url = options.seahawks_url
    try:
        if not options.seahawks_url:
            options.seahawks_url = 'fake url'

        assert setup.selecter.select(setup.expr.to_dag(copy=False)) == Engines.SEAHAWKS

        expr = setup.expr.join(setup.expr2)
        assert setup.selecter.select(expr.to_dag(copy=False)) == Engines.SEAHAWKS

        options.df.seahawks.max_size = setup.table.size - 1

        assert setup.selecter.select(setup.expr.to_dag(copy=False)) == Engines.ODPS
        assert setup.selecter.select(expr.to_dag(copy=False)) == Engines.ODPS
    finally:
        if options.seahawks_url == 'fake url':
            options.seahawks_url = src_seahawks_url
        options.df.seahawks.max_size = src_max_size
