#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from odps.tests.core import TestBase, pandas_case
from odps.compat import unittest
from odps.df.expr.expressions import *
from odps.df import types
from odps.df.backends.pd.compiler import PandasCompiler
from odps.df.backends.context import ExecuteContext


@pandas_case
class Test(TestBase):
    def testPandasCompilation(self):
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list('abc'))

        schema = Schema.from_lists(list('abc'), [types.int8] * 3)
        expr = CollectionExpr(_source_data=df, _schema=schema)

        expr = expr['a', 'b']
        ctx = ExecuteContext()

        compiler = PandasCompiler(ctx.build_dag(expr, expr))
        dag = compiler.compile(expr)

        self.assertEqual(len(dag._graph), 4)
        topos = dag.topological_sort()
        self.assertIsInstance(topos[0][0], CollectionExpr)
        self.assertIsInstance(topos[1][0], Column)
        self.assertIsInstance(topos[2][0], Column)
        self.assertIsInstance(topos[3][0], ProjectCollectionExpr)

if __name__ == '__main__':
    unittest.main()
