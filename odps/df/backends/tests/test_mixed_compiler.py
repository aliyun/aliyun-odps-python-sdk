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

from odps.tests.core import TestBase
from odps.compat import unittest
from odps.models import Schema
from odps.types import validate_data_type
from odps.df.expr.tests.core import MockTable
from odps.df.expr.collections import DistinctCollectionExpr
from odps.df.backends.engine import MixedEngine, available_engines
from odps.df import DataFrame


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'bigint', 'double', 'boolean', 'decimal', 'datetime'))
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        self.tb = DataFrame(table)

        import pandas as pd

        df = pd.DataFrame([['name1', 2, 3.14], ['name2', 100, 2.7]], columns=['name', 'id', 'fid'])
        self.pd = DataFrame(df)

        self.expr = self.tb.join(self.pd, on='name')

        self.engine = MixedEngine(self.odps)

    def testMixedCompile(self):
        dag, expr, callbacks = self.engine._compile(self.expr)

        self.assertEqual(len(dag._graph), 2)

        topos = dag.topological_sort()
        root_node = root, _ = topos[0]
        expr_node = expr, _ = topos[1]

        self.assertTrue(root.is_ancestor(expr))
        self.assertIn(id(expr_node), dag._graph[id(root_node)])
        self.assertEqual(len(available_engines(expr.data_source())), 1)

    def testCacheCompile(self):
        expr = self.tb['name', 'id'].cache()
        expr = expr.groupby('name').agg(expr.id.mean()).cache()
        expr = expr.distinct()

        dag, expr, callbacks = self.engine._compile(expr)

        self.assertEqual(len(dag._graph), 3)

        topos = dag.topological_sort()
        project_node = projected, _ = topos[0]
        groupby_node = grouped, _ = topos[1]
        distinct_node = distincted, _ = topos[2]

        self.assertIn(id(groupby_node), dag._graph[id(project_node)])
        self.assertIn(id(distinct_node), dag._graph[id(groupby_node)])
        self.assertIsInstance(distincted, DistinctCollectionExpr)


if __name__ == '__main__':
    unittest.main()
