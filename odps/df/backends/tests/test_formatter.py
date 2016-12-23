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

import random
try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters
import itertools

from odps.tests.core import to_str
from odps.compat import unittest
from odps.df.backends.tests.core import TestBase
from odps.df.backends.frame import ResultFrame
from odps.df.types import validate_data_type, int64
from odps.df.backends.odpssql.types import df_schema_to_odps_schema
from odps.df.expr.expressions import CollectionExpr
from odps.df.expr.tests.core import MockTable
from odps.df.expr.groupby import GroupByCollectionExpr
from odps.df.backends.engine import MixedEngine
from odps.df.backends.formatter import ExprExecutionGraphFormatter
from odps.models import Schema, Record


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        self.schema = Schema.from_lists(['name', 'id', 'fid', 'dt'],
                                        datatypes('string', 'int64', 'float64', 'datetime'))

    def _random_values(self):
        values = [self._gen_random_string(), self._gen_random_bigint(),
                  self._gen_random_double(), self._gen_random_datetime()]
        schema = df_schema_to_odps_schema(self.schema)
        return Record(schema=schema, values=values)

    def testSmallRowsFormatter(self):
        data = [self._random_values() for _ in range(10)]
        data[-1][0] = None
        pd = ResultFrame(data=data, schema=self.schema, pandas=True)
        result = ResultFrame(data=data, schema=self.schema, pandas=False)
        self.assertEqual(to_str(repr(pd)), to_str(repr(result)))
        self.assertEqual(to_str(pd._repr_html_()), to_str(result._repr_html_()))

        self.assertEqual(result._values, [r for r in result])

    def testLargeRowsFormatter(self):
        data = [self._random_values() for _ in range(1000)]
        pd = ResultFrame(data=data, schema=self.schema, pandas=True)
        result = ResultFrame(data=data, schema=self.schema, pandas=False)
        self.assertEqual(to_str(repr(pd)), to_str(repr(result)))
        self.assertEqual(to_str(pd._repr_html_()), to_str(result._repr_html_()))

    def testLargeColumnsFormatter(self):
        names = list(itertools.chain(*[[name + str(i) for name in self.schema.names] for i in range(10)]))
        types = self.schema.types * 10

        schema = Schema.from_lists(names, types)
        gen_row = lambda: list(itertools.chain(*(self._random_values().values for _ in range(10))))
        data = [Record(schema=df_schema_to_odps_schema(schema), values=gen_row()) for _ in range(10)]

        pd = ResultFrame(data=data, schema=schema, pandas=True)
        result = ResultFrame(data=data, schema=schema, pandas=False)

        self.assertEqual(to_str(repr(pd)), to_str(repr(result)))
        self.assertEqual(to_str(pd._repr_html_()), to_str(result._repr_html_()))

    def testSVGFormatter(self):
        t = MockTable(name='pyodps_test_svg', schema=self.schema, _client=self.odps.rest)
        expr = CollectionExpr(_source_data=t, _schema=self.schema)

        expr1 = expr.groupby('name').agg(id=expr['id'].sum())
        expr2 = expr1['name', expr1.id + 3]

        engine = MixedEngine(self.odps)
        dag = engine.compile(expr2)
        nodes = dag.nodes()
        self.assertEqual(len(nodes), 1)
        expr3 = nodes[0].expr
        self.assertIsInstance(expr3, GroupByCollectionExpr)
        dot = ExprExecutionGraphFormatter(dag)._to_dot()
        self.assertNotIn('Projection', dot)

        expr1 = expr.groupby('name').agg(id=expr['id'].sum()).cache()
        expr2 = expr1['name', expr1.id + 3]

        engine = MixedEngine(self.odps)
        dag = engine.compile(expr2)
        nodes = dag.nodes()
        self.assertEqual(len(nodes), 2)
        dot = ExprExecutionGraphFormatter(dag)._to_dot()
        self.assertIn('Projection', dot)


if __name__ == '__main__':
    unittest.main()
