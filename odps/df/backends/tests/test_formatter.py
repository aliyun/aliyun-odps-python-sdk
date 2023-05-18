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

import itertools
import random
try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters

import pytest
try:
    import pandas
except ImportError:
    pandas = None

from ....compat import Version
from ....models import TableSchema, Record
from ....utils import to_text
from ...expr.expressions import CollectionExpr
from ...expr.tests.core import MockTable
from ...expr.groupby import GroupByCollectionExpr
from ...types import validate_data_type
from ..tests.core import NumGenerators
from ..frame import ResultFrame
from ..odpssql.types import df_schema_to_odps_schema
from ..engine import MixedEngine
from ..formatter import ExprExecutionGraphFormatter


@pytest.fixture
def schema():
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    return TableSchema.from_lists(['name', 'id', 'fid', 'dt'],
                                  datatypes('string', 'int64', 'float64', 'datetime'))


def _random_values(schema):
    values = [NumGenerators.gen_random_string() if random.random() >= 0.05 else None,
              NumGenerators.gen_random_bigint(),
              NumGenerators.gen_random_double() if random.random() >= 0.05 else None,
              NumGenerators.gen_random_datetime() if random.random() >= 0.05 else None]
    schema = df_schema_to_odps_schema(schema)
    return Record(schema=schema, values=values)


@pytest.mark.skipif(not pandas, reason='Pandas not installed')
def test_small_rows_formatter(odps, schema):
    data = [_random_values(schema) for _ in range(10)]
    data[-1][0] = None
    pd = ResultFrame(data=data, schema=schema, pandas=True)
    result = ResultFrame(data=data, schema=schema, pandas=False)
    assert to_text(repr(pd)) == to_text(repr(result))
    assert to_text(pd._repr_html_()) == to_text(result._repr_html_())

    assert result._values == [r for r in result]


@pytest.mark.skipif(not pandas or Version(pandas.__version__) >= Version('0.20'),
                    reason='Pandas not installed or version too new')
def test_large_rows_formatter(schema):
    data = [_random_values(schema) for _ in range(1000)]
    pd = ResultFrame(data=data, schema=schema, pandas=True)
    result = ResultFrame(data=data, schema=schema, pandas=False)
    assert to_text(repr(pd)) == to_text(repr(result))
    assert to_text(pd._repr_html_()) == to_text(result._repr_html_())


@pytest.mark.skipif(not pandas or Version(pandas.__version__) >= Version('0.20'),
                    reason='Pandas not installed or version too new')
def test_large_columns_formatter(schema):
    names = list(itertools.chain(*[[name + str(i) for name in schema.names] for i in range(10)]))
    types = schema.types * 10

    schema = TableSchema.from_lists(names, types)
    gen_row = lambda: list(itertools.chain(*(_random_values(schema).values for _ in range(10))))
    data = [Record(schema=df_schema_to_odps_schema(schema), values=gen_row()) for _ in range(10)]

    pd = ResultFrame(data=data, schema=schema, pandas=True)
    result = ResultFrame(data=data, schema=schema, pandas=False)

    assert to_text(repr(pd)) == to_text(repr(result))
    assert to_text(pd._repr_html_()) == to_text(result._repr_html_())


def test_svg_formatter(odps, schema):
    t = MockTable(
        name='pyodps_test_svg', table_schema=schema, _client=odps.rest
    )
    expr = CollectionExpr(_source_data=t, _schema=schema)

    expr1 = expr.groupby('name').agg(id=expr['id'].sum())
    expr2 = expr1['name', expr1.id + 3]

    engine = MixedEngine(odps)
    dag = engine.compile(expr2)
    nodes = dag.nodes()
    assert len(nodes) == 1
    expr3 = nodes[0].expr
    assert isinstance(expr3, GroupByCollectionExpr)
    dot = ExprExecutionGraphFormatter(dag)._to_dot()
    assert 'Projection' not in dot

    expr1 = expr.groupby('name').agg(id=expr['id'].sum()).cache()
    expr2 = expr1['name', expr1.id + 3]

    engine = MixedEngine(odps)
    dag = engine.compile(expr2)
    nodes = dag.nodes()
    assert len(nodes) == 2
    dot = ExprExecutionGraphFormatter(dag)._to_dot()
    assert 'Projection' in dot
