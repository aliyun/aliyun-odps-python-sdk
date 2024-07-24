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

from ....models import TableSchema
from ...types import validate_data_type
from ..tests.core import MockTable
from ..math import *


@pytest.fixture
def src_expr(odps):
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
    )
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema, client=odps.rest)

    return CollectionExpr(_source_data=table, _schema=schema)


def test_math(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.name.sin())
    pytest.raises(AttributeError, lambda: src_expr.isMale.cos())
    pytest.raises(AttributeError, lambda: src_expr.birth.tan())

    assert isinstance(src_expr.fid.abs(), Abs)
    assert isinstance(src_expr.fid.abs(), Float64SequenceExpr)
    assert isinstance(src_expr.fid.sum().abs(), Float64Scalar)

    assert isinstance(src_expr.id.sqrt(), Sqrt)
    assert isinstance(src_expr.id.sqrt(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.sum().sqrt(), DecimalScalar)

    assert isinstance(src_expr.id.sqrt(), Sqrt)
    assert isinstance(src_expr.id.sqrt(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.sum().sqrt(), DecimalScalar)

    assert isinstance(src_expr.id.sin(), Sin)
    assert isinstance(src_expr.id.sin(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().sin(), DecimalScalar)

    assert isinstance(src_expr.id.sinh(), Sinh)
    assert isinstance(src_expr.id.sinh(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().sinh(), DecimalScalar)

    assert isinstance(src_expr.id.cos(), Cos)
    assert isinstance(src_expr.id.cos(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().cos(), DecimalScalar)

    assert isinstance(src_expr.id.cosh(), Cosh)
    assert isinstance(src_expr.id.cosh(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().cosh(), DecimalScalar)

    assert isinstance(src_expr.id.tan(), Tan)
    assert isinstance(src_expr.id.tan(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().tan(), DecimalScalar)

    assert isinstance(src_expr.id.tanh(), Tanh)
    assert isinstance(src_expr.id.tanh(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().tanh(), DecimalScalar)

    assert isinstance(src_expr.id.exp(), Exp)
    assert isinstance(src_expr.id.exp(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().exp(), DecimalScalar)

    assert isinstance(src_expr.id.expm1(), Expm1)
    assert isinstance(src_expr.id.expm1(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().expm1(), DecimalScalar)

    assert isinstance(src_expr.id.log(), Log)
    assert isinstance(src_expr.id.log(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().log(), DecimalScalar)

    assert isinstance(src_expr.id.log(2), Log)
    assert isinstance(src_expr.id.log(2), Float64SequenceExpr)
    assert src_expr.id.log(2)._base == 2
    assert isinstance(src_expr.scale.max().log(2), DecimalScalar)

    assert isinstance(src_expr.id.log10(), Log10)
    assert isinstance(src_expr.id.log10(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().log10(), DecimalScalar)

    assert isinstance(src_expr.id.log1p(), Log1p)
    assert isinstance(src_expr.id.log1p(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().log1p(), DecimalScalar)

    assert isinstance(src_expr.id.arccos(), Arccos)
    assert isinstance(src_expr.id.arccos(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().arccos(), DecimalScalar)

    assert isinstance(src_expr.id.arccosh(), Arccosh)
    assert isinstance(src_expr.id.arccosh(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().arccosh(), DecimalScalar)

    assert isinstance(src_expr.id.arcsin(), Arcsin)
    assert isinstance(src_expr.id.arcsin(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().arcsin(), DecimalScalar)

    assert isinstance(src_expr.id.arcsinh(), Arcsinh)
    assert isinstance(src_expr.id.arcsin(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().arcsin(), DecimalScalar)

    assert isinstance(src_expr.id.arctan(), Arctan)
    assert isinstance(src_expr.id.arctan(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().arctan(), DecimalScalar)

    assert isinstance(src_expr.id.arctanh(), Arctanh)
    assert isinstance(src_expr.id.sin(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().sin(), DecimalScalar)

    assert isinstance(src_expr.id.radians(), Radians)
    assert isinstance(src_expr.id.radians(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().radians(), DecimalScalar)

    assert isinstance(src_expr.id.degrees(), Degrees)
    assert isinstance(src_expr.id.degrees(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().degrees(), DecimalScalar)

    assert isinstance(src_expr.fid.ceil(), Ceil)
    assert isinstance(src_expr.fid.ceil(), Int64SequenceExpr)
    assert isinstance(src_expr.scale.max().ceil(), Int64Scalar)

    assert isinstance(src_expr.fid.floor(), Floor)
    assert isinstance(src_expr.fid.floor(), Int64SequenceExpr)
    assert isinstance(src_expr.scale.max().floor(), Int64Scalar)

    assert isinstance(src_expr.id.trunc(), Trunc)
    assert isinstance(src_expr.id.trunc(), Float64SequenceExpr)
    assert isinstance(src_expr.scale.max().trunc(), DecimalScalar)

    assert isinstance(src_expr.id.trunc(2), Trunc)
    assert isinstance(src_expr.id.trunc(2), Float64SequenceExpr)
    assert src_expr.id.trunc(2)._decimals == 2
    assert isinstance(src_expr.scale.max().trunc(2), DecimalScalar)