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
from ....tests.core import pandas_case
from ....types import validate_data_type
from ... import DataFrame
from ...expr.tests.core import MockTable
from ...expr.collections import DistinctCollectionExpr
from ..engine import MixedEngine, available_engines


@pytest.fixture
@pandas_case
def setup(odps):
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                               datatypes('string', 'bigint', 'double', 'boolean', 'decimal', 'datetime'))
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
    tb = DataFrame(table)

    import pandas as pd

    df = pd.DataFrame([['name1', 2, 3.14], ['name2', 100, 2.7]], columns=['name', 'id', 'fid'])
    pd = DataFrame(df)

    expr = tb.join(pd, on='name')

    engine = MixedEngine(odps)

    nt = namedtuple("NT", "tb pd expr engine")
    return nt(tb, pd, expr, engine)


def test_mixed_compile(odps, setup):
    dag = setup.engine.compile(setup.expr)

    assert len(dag._graph) == 2

    topos = dag.topological_sort()
    root_node, expr_node = topos[0], topos[1]
    root = root_node.expr
    expr = expr_node.expr

    assert expr.is_ancestor(root) is True
    assert id(expr_node) in dag._graph[id(root_node)]
    assert len(available_engines(expr.data_source())) == 1


def test_cache_compile(odps, setup):
    expr = setup.tb['name', 'id'].cache()
    expr = expr.groupby('name').agg(expr.id.mean()).cache()
    expr = expr.distinct()

    dag = setup.engine.compile(expr)

    assert len(dag._graph) == 3

    topos = dag.topological_sort()
    project_node, groupby_node, distinct_node = topos[0], topos[1], topos[2]
    distincted = distinct_node.expr

    assert id(groupby_node) in dag._graph[id(project_node)]
    assert id(distinct_node) in dag._graph[id(groupby_node)]
    assert isinstance(distincted, DistinctCollectionExpr)


def test_dep(odps, setup):
    expr = setup.tb.pivot_table(rows='id', columns='name', values='fid')

    dag = setup.engine.compile(expr)

    assert len(dag._graph) == 2
    assert sum(len(v) for v in dag._graph.values()) == 1
