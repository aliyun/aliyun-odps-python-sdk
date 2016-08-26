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
from odps.df.types import validate_data_type
from odps.models import Schema
from odps.df.expr.tests.core import MockTable
from odps.df.expr.expressions import CollectionExpr
from odps.df.backends.context import ExecuteContext


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid'],
                                   datatypes('string', 'int64', 'float64'))
        table = MockTable(name='pyodps_test_expr_table', schema=schema)

        self.expr = CollectionExpr(_source_data=table, _schema=schema)
        self.ctx = ExecuteContext()

    def testBuildDAG(self):
        expr = self.expr[self.expr.name, self.expr.id + 1]
        expr2 = self.expr[self.expr.name, self.expr.id + 2]
        expr3 = self.expr[self.expr.name, self.expr.id + 3]
        self.ctx.build_dag(expr, expr.copy_tree())
        self.ctx.build_dag(expr2, expr2.copy_tree())
        self.ctx.build_dag(expr3, expr3.copy_tree())

        self.assertEqual(len(self.ctx._expr_to_dag), 3)
        self.assertGreater(len(list(self.ctx._expr_to_dag.values())[0].nodes()), 0)
        self.assertGreater(len(list(self.ctx._expr_to_dag.values())[1].nodes()), 0)
        self.assertGreater(len(list(self.ctx._expr_to_dag.values())[2].nodes()), 0)
        self.assertTrue(all(l is r for l, r in zip(expr.traverse(unique=True),
                                                   self.ctx.get_dag(expr).traverse(expr))))

        del expr
        self.assertEqual(len(self.ctx._expr_to_dag), 2)
        del expr2
        del expr3
        self.assertEqual(len(self.ctx._expr_to_dag), 0)


if __name__ == '__main__':
    unittest.main()