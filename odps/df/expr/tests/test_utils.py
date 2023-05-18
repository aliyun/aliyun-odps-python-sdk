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

from ....df.expr.utils import *
from ..tests.core import MockTable
from ... import types
from ...expr.expressions import *
from ....models import TableSchema


def test_get_attrs():
    schema = TableSchema.from_lists(['name', 'id'], [types.string, types.int64])
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
    expr = CollectionExpr(_source_data=table, _schema=schema)

    expected = ('_lhs', '_rhs', '_data_type', '_source_data_type',
                '_ml_fields_cache', '_ml_uplink', '_ml_operations',
                '_name', '_source_name', '_deps', '_ban_optimize', '_engine',
                '_need_cache', '_mem_cache', '_id', '_args_indexes', )
    assert list(expected) == list(get_attrs(expr.id + 1))


def test_is_changed():
    schema = TableSchema.from_lists(['name', 'id'], [types.string, types.int64])
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
    expr = CollectionExpr(_source_data=table, _schema=schema)
    expr2 = CollectionExpr(_source_data=table, _schema=schema)

    assert is_changed(expr[expr.id < 3], expr.id) is False
    assert is_changed(expr[expr.id + 2,], expr.id) is True
    assert is_changed(expr[expr.id < 3], expr2.id) is None
    assert is_changed(expr.groupby('name').agg(id=expr.id.sum()), expr.id) is True
    assert is_changed(expr.groupby('name').agg(id=expr.id.sum()), expr.name) is False