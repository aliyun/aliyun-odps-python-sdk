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

from .... import compat
from ...types import validate_data_type
from ..tests.core import MockTable
from ..expressions import *
from ..composites import *


@pytest.fixture
def src_expr(odps):
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['id', 'name', 'relatives', 'hobbies'],
        datatypes('int64', 'string', 'dict<string, string>', 'list<string>')
    )
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema, client=odps.rest)

    return CollectionExpr(_source_data=table, _schema=schema)


def test_explode(src_expr):
    expr = src_expr.hobbies.explode()
    assert isinstance(expr, RowAppliedCollectionExpr)
    assert expr.input is src_expr
    assert expr._func == 'EXPLODE'
    assert expr.dtypes.names == [src_expr.hobbies.name]
    assert expr.dtypes.types == [src_expr.hobbies.dtype.value_type]

    expr = src_expr.hobbies.explode('exploded')
    assert expr.dtypes.names == ['exploded']

    pytest.raises(ValueError, src_expr.hobbies.explode, ['abc', 'def'])

    expr = src_expr.hobbies.explode(pos=True)
    assert isinstance(expr, RowAppliedCollectionExpr)
    assert expr.input is src_expr
    assert expr._func == 'POSEXPLODE'
    assert expr.dtypes.names == [src_expr.hobbies.name + '_pos', src_expr.hobbies.name]
    assert expr.dtypes.types == [validate_data_type('int64'), src_expr.hobbies.dtype.value_type]

    expr = src_expr.hobbies.explode(['pos', 'exploded'], pos=True)
    assert expr.dtypes.names == ['pos', 'exploded']

    expr = src_expr.hobbies.explode('exploded', pos=True)
    assert expr.dtypes.names == ['exploded_pos', 'exploded']

    expr = src_expr.relatives.explode()
    assert isinstance(expr, RowAppliedCollectionExpr)
    assert expr.input is src_expr
    assert expr._func == 'EXPLODE'
    assert expr.dtypes.names == [src_expr.relatives.name + '_key', src_expr.relatives.name + '_value']
    assert expr.dtypes.types == [src_expr.relatives.dtype.key_type, src_expr.relatives.dtype.value_type]

    expr = src_expr.relatives.explode(['k', 'v'])
    assert expr.dtypes.names == ['k', 'v']

    pytest.raises(ValueError, src_expr.relatives.explode, ['abc'])
    pytest.raises(ValueError, src_expr.relatives.explode, ['abc'], pos=True)


def test_list_methods(src_expr):
    expr = src_expr.hobbies[0]
    assert isinstance(expr, ListDictGetItem)
    assert isinstance(expr, StringSequenceExpr)
    assert expr.dtype == validate_data_type('string')

    expr = src_expr.hobbies.len()
    assert isinstance(expr, ListDictLength)
    assert isinstance(expr, Int64SequenceExpr)

    expr = src_expr.hobbies.sort()
    assert isinstance(expr, ListSort)
    assert isinstance(expr, ListSequenceExpr)
    assert expr.dtype == validate_data_type('list<string>')

    expr = src_expr.hobbies.contains('yacht')
    assert isinstance(expr, ListContains)
    assert isinstance(expr, BooleanSequenceExpr)


def test_dict_methods(src_expr):
    expr = src_expr.relatives['abc']
    assert isinstance(expr, ListDictGetItem)
    assert isinstance(expr, StringSequenceExpr)
    assert expr.dtype == validate_data_type('string')

    expr = src_expr.relatives.len()
    assert isinstance(expr, ListDictLength)
    assert isinstance(expr, Int64SequenceExpr)

    expr = src_expr.relatives.keys()
    assert isinstance(expr, DictKeys)
    assert isinstance(expr, ListSequenceExpr)
    assert expr.dtype == validate_data_type('list<string>')

    expr = src_expr.relatives.values()
    assert isinstance(expr, DictValues)
    assert isinstance(expr, ListSequenceExpr)
    assert expr.dtype == validate_data_type('list<string>')


def test_builders(src_expr):
    expr = make_list(1, 2, 3, 4)
    assert isinstance(expr, ListBuilder)
    assert isinstance(expr, ListScalar)
    assert expr.dtype == validate_data_type('list<int32>')

    expr = make_list(1, 2, 3, src_expr.id)
    assert isinstance(expr, ListBuilder)
    assert isinstance(expr, ListSequenceExpr)
    assert expr.dtype == validate_data_type('list<int64>')

    pytest.raises(TypeError, make_list, 1, 2, 'str', type='int32')
    pytest.raises(TypeError, make_list, 1, 2, 'str')
    expr = make_list(1, 2, 3, 4, type='int64')
    assert expr.dtype == validate_data_type('list<int64>')
    expr = make_list(1.1, 2.2, 3.3, 4.4)
    assert expr.dtype == validate_data_type('list<float64>')
    expr = make_list(1, 2, 3, 65535)
    assert expr.dtype == validate_data_type('list<int32>')
    expr = make_list(1, 2, 3, compat.long_type(12345678910))
    assert expr.dtype == validate_data_type('list<int64>')
    expr = make_list(1, 2, 3, 3.5)
    assert expr.dtype == validate_data_type('list<float64>')

    pytest.raises(ValueError, make_dict, 1, 2, 3)

    expr = make_dict(1, 2, 3, 4)
    assert isinstance(expr, DictBuilder)
    assert isinstance(expr, DictScalar)
    assert expr.dtype == validate_data_type('dict<int32,int32>')

    expr = make_dict(1, 2, 3, 4, key_type='int16', value_type='int64')
    assert isinstance(expr, DictBuilder)
    assert isinstance(expr, DictScalar)
    assert expr.dtype == validate_data_type('dict<int16,int64>')

    expr = make_dict(1, 2, 3, src_expr.id)
    assert isinstance(expr, DictBuilder)
    assert isinstance(expr, DictSequenceExpr)
    assert expr.dtype == validate_data_type('dict<int32,int64>')
