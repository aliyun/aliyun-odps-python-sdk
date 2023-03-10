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

import random
try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters
import itertools
try:
    import pandas
except ImportError:
    pandas = None

from odps.tests.core import to_str
from odps.compat import unittest, Version
from odps.df.backends.tests.core import TestBase
from odps.df.backends.frame import ResultFrame
from odps.df.types import validate_data_type, int64
from odps.df.backends.odpssql.types import df_schema_to_odps_schema
from odps.df.expr.expressions import CollectionExpr
from odps.df.expr.tests.core import MockTable
from odps.df.expr.groupby import GroupByCollectionExpr
from odps.df.backends.engine import MixedEngine
from odps.df.backends.formatter import ExprExecutionGraphFormatter
from odps.models import TableSchema, Record


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        self.schema = TableSchema.from_lists(['name', 'id', 'fid', 'dt'],
                                        datatypes('string', 'int64', 'float64', 'datetime'))

    def _random_values(self):
        values = [self._gen_random_string() if random.random() >= 0.05 else None,
                  self._gen_random_bigint(),
                  self._gen_random_double() if random.random() >= 0.05 else None,
                  self._gen_random_datetime() if random.random() >= 0.05 else None]
        schema = df_schema_to_odps_schema(self.schema)
        return Record(schema=schema, values=values)

    @unittest.skipIf(not pandas, 'Pandas not installed')
    def testSmallRowsFormatter(self):
        data = [self._random_values() for _ in range(10)]
        data[-1][0] = None
        pd = ResultFrame(data=data, schema=self.schema, pandas=True)
        result = ResultFrame(data=data, schema=self.schema, pandas=False)
        self.assertEqual(to_str(repr(pd)), to_str(repr(result)))
        self.assertEqual(to_str(pd._repr_html_()), to_str(result._repr_html_()))

        self.assertEqual(result._values, [r for r in result])

    @unittest.skipIf(not pandas or Version(pandas.__version__) >= Version('0.20'),
                     'Pandas not installed or version too new')
    def testLargeRowsFormatter(self):
        data = [self._random_values() for _ in range(1000)]
        pd = ResultFrame(data=data, schema=self.schema, pandas=True)
        result = ResultFrame(data=data, schema=self.schema, pandas=False)
        self.assertEqual(to_str(repr(pd)), to_str(repr(result)))
        self.assertEqual(to_str(pd._repr_html_()), to_str(result._repr_html_()))

    @unittest.skipIf(not pandas or Version(pandas.__version__) >= Version('0.20'),
                     'Pandas not installed or version too new')
    def testLargeColumnsFormatter(self):
        names = list(itertools.chain(*[[name + str(i) for name in self.schema.names] for i in range(10)]))
        types = self.schema.types * 10

        schema = TableSchema.from_lists(names, types)
        gen_row = lambda: list(itertools.chain(*(self._random_values().values for _ in range(10))))
        data = [Record(schema=df_schema_to_odps_schema(schema), values=gen_row()) for _ in range(10)]

        pd = ResultFrame(data=data, schema=schema, pandas=True)
        result = ResultFrame(data=data, schema=schema, pandas=False)

        self.assertEqual(to_str(repr(pd)), to_str(repr(result)))
        self.assertEqual(to_str(pd._repr_html_()), to_str(result._repr_html_()))

    def testSVGFormatter(self):
        t = MockTable(
            name='pyodps_test_svg', table_schema=self.schema, _client=self.odps.rest
        )
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
