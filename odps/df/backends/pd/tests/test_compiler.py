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

from .....tests.core import pandas_case
from ....expr.expressions import *
from .... import types
from ...context import ExecuteContext
from ..compiler import PandasCompiler


@pandas_case
def test_pandas_compilation():
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc'))

    schema = TableSchema.from_lists(list('abc'), [types.int8] * 3)
    expr = CollectionExpr(_source_data=df, _schema=schema)

    expr = expr['a', 'b']
    _ = ExecuteContext()

    compiler = PandasCompiler(expr.to_dag())
    dag = compiler.compile(expr)

    assert len(dag._graph) == 4
    topos = dag.topological_sort()
    assert isinstance(topos[0][0], CollectionExpr)
    assert isinstance(topos[1][0], Column)
    assert isinstance(topos[2][0], Column)
    assert isinstance(topos[3][0], ProjectCollectionExpr)
