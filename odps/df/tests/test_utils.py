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

import pytest

from ...tests.core import tn
from ...models import TableSchema
from .. import DataFrame
from ..backends.utils import fetch_data_source_size


@pytest.fixture
def test_table(odps):
    test_table_name = tn('pyodps_test_dataframe')
    schema = TableSchema.from_lists(
        ['id', 'name'], ['bigint', 'string'], ['ds', 'mm', 'hh'], ['string'] * 3
    )

    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema)

    pt = 'ds=today,mm=now,hh=curr'
    table.create_partition(pt)
    with table.open_writer(pt) as w:
        w.write([[1, 'name1'], [2, 'name2'], [3, 'name3']])

    try:
        yield table, pt
    finally:
        table.drop()


def test_fetch_table_size(test_table):
    table, pt = test_table
    df = DataFrame(table)

    expr = df.filter_parts(pt)
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) > 0

    expr = df.filter_parts('ds=today,hh=curr,mm=now')
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) > 0

    expr = df.filter_parts('ds=today,hh=curr,mm=now2')
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) is None

    expr = df.filter_parts('ds=today,hh=curr')
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) > 0

    expr = df.filter_parts('ds=today,mm=now')
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) > 0

    expr = df.filter(df.ds == 'today', df.mm == 'now', df.hh == 'curr')
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) > 0

    expr = df.filter(df.ds == 'today', df.hh == 'curr', df.mm == 'now')
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) > 0

    expr = df.filter(df.ds == 'today', df.hh == 'curr', df.mm == 'now2')
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) is None

    expr = df.filter(df.ds == 'today', df.hh == 'curr')
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) > 0

    expr = df.filter(df.ds == 'today', df.mm == 'now')
    dag = expr.to_dag(copy=False)
    assert fetch_data_source_size(dag, df, table) > 0
