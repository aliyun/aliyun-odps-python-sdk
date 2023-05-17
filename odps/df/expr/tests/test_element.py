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

import textwrap

import pytest

from ....compat import LESS_PY35
from ....models import TableSchema
from ...types import validate_data_type
from ..tests.core import MockTable
from ..expressions import *
from ..element import *


@pytest.fixture
def src_expr():
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
    )
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)

    return CollectionExpr(_source_data=table, _schema=schema)


def test_is_null(src_expr):
    assert isinstance(src_expr.fid.isnull(), IsNull)
    assert isinstance(src_expr.fid.isnull(), BooleanSequenceExpr)
    assert isinstance(src_expr.fid.sum().isnull(), BooleanScalar)


def test_not_null(src_expr):
    assert isinstance(src_expr.fid.notnull(), NotNull)
    assert isinstance(src_expr.fid.notnull(), BooleanSequenceExpr)
    assert isinstance(src_expr.fid.sum().notnull(), BooleanScalar)


def test_fill_na(src_expr):
    assert isinstance(src_expr.name.fillna('test'), FillNa)
    assert isinstance(src_expr.name.fillna('test'), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().fillna('test'), StringScalar)
    assert isinstance(src_expr.scale.fillna(0), DecimalSequenceExpr)

    pytest.raises(ValueError, lambda: src_expr.id.fillna('abc'))


def test_is_in(src_expr):
    expr = src_expr.name.isin(list('abc'))
    assert isinstance(expr, IsIn)
    assert len(expr.values) == 3
    assert [it.value for it in expr.values] == list('abc')
    assert isinstance(src_expr.name.isin(list('abc')), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().isin(list('abc')), BooleanScalar)


def test_between(src_expr):
    assert isinstance(src_expr.id.between(1, 3), Between)
    assert isinstance(src_expr.name.between(1, 3), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().between(1, 3), BooleanScalar)


def test_if_else(src_expr):
    assert isinstance((src_expr.id == 3).ifelse(src_expr.id, src_expr.fid), IfElse)
    assert isinstance((src_expr.id == 3).ifelse(src_expr.id, src_expr.fid), Float64SequenceExpr)
    assert isinstance((src_expr.id == 3).ifelse(src_expr.id.sum(), src_expr.fid), Float64SequenceExpr)


def test_switch(src_expr):
    expr = src_expr.id.switch(3, src_expr.name, src_expr.fid.abs(), src_expr.name + 'test')
    assert isinstance(expr, Switch)
    assert len(expr.conditions) == 2
    assert len(expr.thens) == 2
    assert isinstance(expr, StringSequenceExpr)
    assert not isinstance(expr, Scalar)

    expr = src_expr.id.switch((3, src_expr.name), (src_expr.fid.abs(), src_expr.name + 'test'))
    assert isinstance(expr, Switch)
    assert len(expr.conditions) == 2
    assert len(expr.thens) == 2
    assert isinstance(expr, StringSequenceExpr)
    assert not isinstance(expr, Scalar)

    expr = src_expr.id.switch(3, src_expr.name, src_expr.fid.abs(), src_expr.name + 'test',
                               default=src_expr.name.lower())
    assert isinstance(expr, Switch)
    assert len(expr.conditions) == 2
    assert len(expr.thens) == 2
    assert isinstance(expr.default, StringSequenceExpr)
    assert isinstance(expr, StringSequenceExpr)
    assert not isinstance(expr, Scalar)

    pytest.raises(ExpressionError, lambda: src_expr.switch(3, src_expr.name))

    expr = src_expr.switch(src_expr.id == 3, src_expr.name,
                            src_expr.id == 2, src_expr.name + 'test')
    assert isinstance(expr, Switch)
    assert len(expr.conditions) == 2
    assert len(expr.thens) == 2
    assert isinstance(expr, StringSequenceExpr)
    assert not isinstance(expr, Scalar)


def test_cut(src_expr):
    expr = src_expr.id.cut([5, 10], labels=['mid'])
    assert isinstance(expr, Cut)
    assert isinstance(expr, StringSequenceExpr)

    expr = src_expr.id.max().cut([5, 10], labels=['mid'])
    assert isinstance(expr, Cut)
    assert isinstance(expr, StringScalar)

    pytest.raises(ValueError, lambda: src_expr.id.cut([5]))
    pytest.raises(ValueError, lambda: src_expr.id.cut([5], include_under=True))
    pytest.raises(ValueError, lambda: src_expr.id.cut([5], include_over=True))


def test_to_datetime(src_expr):
    expr = src_expr.id.to_datetime()
    assert isinstance(expr, IntToDatetime)
    assert isinstance(expr, DatetimeSequenceExpr)

    expr = src_expr.id.max().to_datetime()
    assert isinstance(expr, IntToDatetime)
    assert isinstance(expr, DatetimeScalar)


def test_map(src_expr):
    expr = src_expr.id.map(lambda a: float(a + 1), rtype=types.float64)
    assert isinstance(expr, MappedExpr)
    assert expr._data_type is types.float64

    if not LESS_PY35:
        l = locals().copy()
        six.exec_(textwrap.dedent("""
        from typing import Optional
        
        def fun(v) -> float:
            return float(v + 1)
        expr = src_expr.id.map(fun)
        """), globals(), l)
        expr = l['expr']
        assert isinstance(expr, MappedExpr)
        assert isinstance(expr._data_type, types.Float)

        l = locals().copy()
        six.exec_(textwrap.dedent("""
        from typing import Optional
        
        def fun(v) -> Optional[float]:
            return float(v + 1)
        expr = src_expr.id.map(fun)
        """), globals(), l)
        expr = l['expr']
        assert isinstance(expr, MappedExpr)
        assert isinstance(expr._data_type, types.Float)


def test_reduce_apply(src_expr):
    expr = src_expr[src_expr.id, src_expr['name', 'id'].apply(
        lambda row: row.name + row.id, axis=1, reduce=True).rename('nameid')]

    assert isinstance(expr._fields[1], MappedExpr)

    if not LESS_PY35:
        l = locals().copy()
        six.exec_(textwrap.dedent("""
        def fun(r) -> float:
            return r.id + r.fid
        expr = src_expr[src_expr.id, src_expr['id', 'fid'].apply(fun, axis=1, reduce=True).rename('idfid')]
        """), globals(), l)
        expr = l['expr']
        assert isinstance(expr._fields[1], MappedExpr)
        assert isinstance(expr._fields[1]._data_type, types.Float)