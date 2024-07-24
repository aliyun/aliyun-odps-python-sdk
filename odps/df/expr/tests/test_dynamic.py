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

from collections import namedtuple

import pytest

from ....models import TableSchema
from ...types import validate_data_type
from ..dynamic import *
from ..expressions import StringScalar
from ..tests.core import MockTable


@pytest.fixture
def setup(odps):
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = DynamicSchema.from_schema(
        TableSchema.from_lists(
            ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
            datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
        )
    )
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema, client=odps.rest)

    schema2 = DynamicSchema.from_schema(
        TableSchema.from_lists(
            ['name2', 'id', 'fid2'], datatypes('string', 'int64', 'float64')
        ),
        default_type=types.string
    )
    table2 = MockTable(name='pyodps_test_expr_tabl2', table_schema=schema2, client=odps.rest)

    expr = DynamicCollectionExpr(_source_data=table, _schema=schema)
    expr2 = DynamicCollectionExpr(_source_data=table2, _schema=schema2)

    nt = namedtuple("NT", "expr expr2")
    return nt(expr, expr2)


def test_dynamic(setup):
    df = setup.expr.distinct('name', 'id')
    assert not isinstance(df, DynamicMixin)
    assert not isinstance(df._schema, DynamicSchema)

    # the sequence must be definite, no need for generating dynamic sequence
    assert not isinstance(setup.expr['name'], DynamicMixin)
    # a field which does not exist is fine
    assert isinstance(setup.expr['not_exist'], DynamicMixin)
    # df's schema is not dynamic
    pytest.raises(ValueError, lambda: df['non_exist'])

    df = setup.expr.distinct('name', 'non_exist')
    assert isinstance(df, DynamicMixin)
    assert not isinstance(df._schema, DynamicSchema)

    df = setup.expr.distinct()
    assert isinstance(df, DynamicMixin)
    assert isinstance(df._schema, DynamicSchema)

    df2 = setup.expr2.distinct('name2', 'not_exist')
    assert not isinstance(df2, DynamicMixin)
    assert not isinstance(df2._schema, DynamicSchema)

    # the sequence must be definite, no need for generating dynamic sequence
    assert not isinstance(df2['name2'], DynamicMixin)
    # a field which does not exist is fine
    assert not isinstance(df2['not_exist'], DynamicMixin)
    # df2's schema is not dynamic
    pytest.raises(ValueError, lambda: df2['non_exist'])

    assert setup.expr2['non_exist'].dtype == types.string
    assert isinstance(setup.expr2['non_exist'].sum(), StringScalar)

    # projection
    df3 = setup.expr2[setup.expr2, setup.expr2.id.astype('string').rename('id2')]
    assert isinstance(df3, DynamicMixin)
    assert isinstance(df3._schema, DynamicSchema)
    # non_exist need to be checked
    assert isinstance(df3['non_exist'], DynamicMixin)
    assert not isinstance(setup.expr['id', 'name2']._schema, DynamicSchema)

    # filter
    df4 = setup.expr2.filter(setup.expr2.id < 10)
    assert isinstance(df4, DynamicMixin)
    assert isinstance(df4._schema, DynamicSchema)

    # slice
    df5 = setup.expr2[2:4]
    assert isinstance(df5, DynamicMixin)
    assert isinstance(df5._schema, DynamicSchema)

    # sort
    df6 = setup.expr2.sort('id')
    assert isinstance(df6, DynamicMixin)
    assert isinstance(df6._schema, DynamicSchema)

    # apply
    df7 = setup.expr2.apply(lambda row: row, axis=1, names=setup.expr2.schema.names)
    assert not isinstance(df7, DynamicMixin)
    assert not isinstance(df7._schema, DynamicSchema)

    # sample
    df8 = setup.expr2.sample(parts=10)
    assert isinstance(df8, DynamicMixin)
    assert isinstance(df8._schema, DynamicSchema)

    # groupby
    df9 = setup.expr2.groupby('id').agg(setup.expr2['name3'].sum())
    assert not isinstance(df9, DynamicMixin)
    assert not isinstance(df9._schema, DynamicSchema)
    df10 = setup.expr.groupby('id2').agg(setup.expr.name.sum())
    assert not isinstance(df10, DynamicMixin)
    assert not isinstance(df10._schema, DynamicSchema)

    # mutate
    df11 = setup.expr2.groupby('id').mutate(id2=lambda x: x.id.cumsum())
    assert not isinstance(df11, DynamicMixin)
    assert not isinstance(df11._schema, DynamicSchema)
    setup.expr.groupby('id').sort('id').non_exist.astype('int').cumsum()

    # join
    df12 = setup.expr.join(setup.expr2)[setup.expr, setup.expr2['id2']]
    assert isinstance(df12, DynamicMixin)
    assert isinstance(df12._schema, DynamicSchema)
    assert isinstance(df12.input, DynamicMixin)
    assert isinstance(df12.input._schema, DynamicSchema)
    df13 = setup.expr.join(setup.expr2)[setup.expr.id, setup.expr2.name2]
    assert not isinstance(df13, DynamicMixin)
    assert not isinstance(df13._schema, DynamicSchema)

    # union
    df14 = setup.expr['id', setup.expr.name.rename('name2'), setup.expr.fid.rename('fid2')]\
        .union(setup.expr2)
    assert not isinstance(df14, DynamicMixin)
    assert not isinstance(df14._schema, DynamicSchema)