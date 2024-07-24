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
from ..expressions import *
from ..strings import *


@pytest.fixture
def src_expr(odps):
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
    )
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema, client=odps.rest)

    return CollectionExpr(_source_data=table, _schema=schema)


def test_strings(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.id.strip())
    pytest.raises(AttributeError, lambda: src_expr.fid.upper())
    pytest.raises(AttributeError, lambda: src_expr.isMale.lower())
    pytest.raises(AttributeError, lambda: src_expr.scale.repeat(3))
    pytest.raises(AttributeError, lambda: src_expr.birth.len())

    assert isinstance(src_expr.name.capitalize(), Capitalize)
    assert isinstance(src_expr.name.capitalize(), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().capitalize(), StringScalar)

    expr = src_expr.name.cat(src_expr.id.astype('string'), sep=',')
    assert isinstance(expr, CatStr)
    assert isinstance(expr, StringSequenceExpr)
    assert isinstance(src_expr.name.sum().cat(src_expr.id.sum().astype('string')), StringScalar)

    pytest.raises(ValueError, lambda: src_expr.name.cat(','))

    assert isinstance(src_expr.name.contains('test'), Contains)
    assert isinstance(src_expr.name.contains('test'), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().contains('test'), BooleanScalar)

    assert isinstance(src_expr.name.count('test'), Count)
    assert isinstance(src_expr.name.count('test'), Int64SequenceExpr)
    assert isinstance(src_expr.name.sum().count('test'), Int64Scalar)

    assert isinstance(src_expr.name.endswith('test'), Endswith)
    assert isinstance(src_expr.name.endswith('test'), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().endswith('test'), BooleanScalar)

    assert isinstance(src_expr.name.startswith('test'), Startswith)
    assert isinstance(src_expr.name.startswith('test'), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().startswith('test'), BooleanScalar)

    assert isinstance(src_expr.name.extract(r'[ab](\d)'), Extract)
    assert isinstance(src_expr.name.extract(r'[ab](\d)'), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().extract(r'[ab](\d)'), StringScalar)

    assert isinstance(src_expr.name.find('test'), Find)
    assert isinstance(src_expr.name.find('test'), Int64SequenceExpr)
    assert isinstance(src_expr.name.sum().find('test'), Int64Scalar)

    assert isinstance(src_expr.name.rfind('test'), RFind)
    assert isinstance(src_expr.name.rfind('test'), Int64SequenceExpr)
    assert isinstance(src_expr.name.sum().rfind('test'), Int64Scalar)

    assert isinstance(src_expr.name.replace('test', 'test2'), Replace)
    assert isinstance(src_expr.name.replace('test', 'test2'), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().replace('test', 'test2'), StringScalar)

    assert isinstance(src_expr.name.get(1), Get)
    assert isinstance(src_expr.name.get(1), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().get(1), StringScalar)
    assert isinstance(src_expr.name[1], Get)
    assert isinstance(src_expr.name[1], StringSequenceExpr)
    assert isinstance(src_expr.name.sum()[1], StringScalar)
    assert isinstance(src_expr.name[src_expr.id], Get)
    assert isinstance(src_expr.name[src_expr.id], StringSequenceExpr)
    assert isinstance(src_expr.name.sum()[src_expr.id], StringScalar)

    assert isinstance(src_expr.name.len(), Len)
    assert isinstance(src_expr.name.len(), Int64SequenceExpr)
    assert isinstance(src_expr.name.sum().len(), Int64Scalar)

    assert isinstance(src_expr.name.ljust(3), Ljust)
    assert isinstance(src_expr.name.ljust(3), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().ljust(3), StringScalar)

    assert isinstance(src_expr.name.rjust(3), Rjust)
    assert isinstance(src_expr.name.rjust(3), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().rjust(3), StringScalar)

    assert isinstance(src_expr.name.lower(), Lower)
    assert isinstance(src_expr.name.lower(), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().lower(), StringScalar)

    assert isinstance(src_expr.name.upper(), Upper)
    assert isinstance(src_expr.name.upper(), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().upper(), StringScalar)

    assert isinstance(src_expr.name.lstrip(), Lstrip)
    assert isinstance(src_expr.name.lstrip(), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().lstrip(), StringScalar)

    assert isinstance(src_expr.name.strip(), Strip)
    assert isinstance(src_expr.name.strip(), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().strip(), StringScalar)

    assert isinstance(src_expr.name.pad(4), Pad)
    assert isinstance(src_expr.name.pad(4), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().pad(4), StringScalar)

    assert isinstance(src_expr.name.repeat(4), Repeat)
    assert isinstance(src_expr.name.repeat(4), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().repeat(4), StringScalar)

    assert isinstance(src_expr.name.slice(0, 4), Slice)
    assert isinstance(src_expr.name.slice(0, 4), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().slice(0, 4), StringScalar)
    assert isinstance(src_expr.name[0: 4], Slice)
    assert isinstance(src_expr.name[0: 4], StringSequenceExpr)
    assert isinstance(src_expr.name.sum()[0: 4], StringScalar)

    assert isinstance(src_expr.name.split(','), Split)
    assert isinstance(src_expr.name.split(','), ListSequenceExpr)
    assert src_expr.name.split(',').dtype == types.validate_data_type('list<string>')

    assert isinstance(src_expr.name.swapcase(), Swapcase)
    assert isinstance(src_expr.name.swapcase(), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().swapcase(), StringScalar)

    assert isinstance(src_expr.name.title(), Title)
    assert isinstance(src_expr.name.title(), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().title(), StringScalar)

    assert isinstance(src_expr.name.zfill(5), Zfill)
    assert isinstance(src_expr.name.zfill(5), StringSequenceExpr)
    assert isinstance(src_expr.name.sum().zfill(5), StringScalar)

    assert isinstance(src_expr.name.isalnum(), Isalnum)
    assert isinstance(src_expr.name.isalnum(), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().isalnum(), BooleanScalar)

    assert isinstance(src_expr.name.isalpha(), Isalpha)
    assert isinstance(src_expr.name.isalpha(), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().isalpha(), BooleanScalar)

    assert isinstance(src_expr.name.isdigit(), Isdigit)
    assert isinstance(src_expr.name.isdigit(), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().isdigit(), BooleanScalar)

    assert isinstance(src_expr.name.isspace(), Isspace)
    assert isinstance(src_expr.name.isspace(), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().isspace(), BooleanScalar)

    assert isinstance(src_expr.name.islower(), Islower)
    assert isinstance(src_expr.name.islower(), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().islower(), BooleanScalar)

    assert isinstance(src_expr.name.isupper(), Isupper)
    assert isinstance(src_expr.name.isupper(), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().isupper(), BooleanScalar)

    assert isinstance(src_expr.name.istitle(), Istitle)
    assert isinstance(src_expr.name.istitle(), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().istitle(), BooleanScalar)

    assert isinstance(src_expr.name.isnumeric(), Isnumeric)
    assert isinstance(src_expr.name.isnumeric(), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().isnumeric(), BooleanScalar)

    assert isinstance(src_expr.name.isdecimal(), Isdecimal)
    assert isinstance(src_expr.name.isdecimal(), BooleanSequenceExpr)
    assert isinstance(src_expr.name.sum().isdecimal(), BooleanScalar)

    assert isinstance(src_expr.name.todict(), StringToDict)
    assert isinstance(src_expr.name.todict(), DictSequenceExpr)
    assert src_expr.name.todict().dtype == types.validate_data_type('dict<string, string>')
