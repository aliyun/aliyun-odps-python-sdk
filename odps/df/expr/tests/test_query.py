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

from ..expressions import *
from ..tests.core import MockTable


@pytest.fixture
def src_expr(odps):
    schema = TableSchema.from_lists(['name', 'id', 'fid'], [types.string, types.int64, types.float64])
    table = MockTable(name='pyodps_test_query_table', table_schema=schema)
    table._client = odps.rest
    return CollectionExpr(_source_data=table, _schema=schema)


def test_base_query(src_expr):
    expr = src_expr

    result = expr.query('@expr.id > 0')
    assert isinstance(result, FilterCollectionExpr)

    result = expr.query('name == "test"')
    assert isinstance(result, FilterCollectionExpr)

    result = expr.query('id + fid > id * fid')
    assert isinstance(result, FilterCollectionExpr)

    result = expr.query('id ** fid <= id / fid - 1')
    assert isinstance(result, FilterCollectionExpr)


def test_chained_cmp(src_expr):
    expr = src_expr

    result = expr.query('id > 0 & fid < 10 and (name in ["test1", "test2"])')
    assert isinstance(result, FilterCollectionExpr)

    result = expr.query('id >= 0 | fid in id or name != "test"')
    assert isinstance(result, FilterCollectionExpr)


def test_local_variable(src_expr):
    expr = src_expr
    id = 1  # npqa: E722
    name = ['test1', 'test2']  # npqa: E722

    result = expr.query('id + 1 > @id & name in @name')
    assert isinstance(result, FilterCollectionExpr)

    pytest.raises(KeyError, lambda: expr.query('id == @fid'))
