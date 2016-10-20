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


from odps.tests.core import TestBase, to_str
from odps.compat import unittest
from odps.models import Schema
from odps.df import Scalar, NullScalar, BuiltinFunction, output
from odps.df.types import validate_data_type
from odps.df.expr.tests.core import MockTable
from odps.df.expr.expressions import CollectionExpr, ProjectCollectionExpr, \
    FilterCollectionExpr
from odps.df.backends.odpssql.engine import ODPSEngine
from odps.df.backends.optimize.columnpruning import ColumnPruning


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
                                   ['ds'], datatypes('string'))
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        self.expr = CollectionExpr(_source_data=table, _schema=Schema(columns=schema.columns))

        table1 = MockTable(name='pyodps_test_expr_table1', schema=schema)
        self.expr1 = CollectionExpr(_source_data=table1, _schema=Schema(columns=schema.columns))

        table2 = MockTable(name='pyodps_test_expr_table2', schema=schema)
        self.expr2 = CollectionExpr(_source_data=table2, _schema=Schema(columns=schema.columns))

        schema2 = Schema.from_lists(['name', 'id', 'fid'], datatypes('string', 'int64', 'float64'),
                                    ['part1', 'part2'], datatypes('string', 'int64'))
        table3 = MockTable(name='pyodps_test_expr_table2', schema=schema2)
        self.expr3 = CollectionExpr(_source_data=table3, _schema=Schema(columns=schema2.columns))

    def testProjectPrune(self):
        expr = self.expr.select('name', 'id')
        new_expr = ColumnPruning(expr.to_dag()).prune()
        self.assertIsInstance(new_expr, ProjectCollectionExpr)
        self.assertIsNotNone(new_expr.input._source_data)

        expected = 'SELECT t1.`name`, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(expected, ODPSEngine(self.odps).compile(expr, prettify=False))

        expr = self.expr[Scalar(3).rename('const'),
                         NullScalar('string').rename('string_const'),
                         self.expr.id]
        expected = 'SELECT 3 AS `const`, CAST(NULL AS STRING) AS `string_const`, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.select(pt=BuiltinFunction('max_pt', args=(self.expr._source_data.name,)))
        expected = "SELECT max_pt('pyodps_test_expr_table') AS `pt` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testApplyPrune(self):
        @output(['name', 'id'], ['string', 'string'])
        def h(row):
            yield row[0], row[1]

        expr = self.expr[self.expr.fid < 0].apply(h, axis=1)['id', ]
        new_expr = ColumnPruning(expr.to_dag()).prune()

        self.assertIsInstance(new_expr, ProjectCollectionExpr)
        self.assertIsInstance(new_expr.input.input, FilterCollectionExpr)
        self.assertIsNotNone(new_expr.input.input.input._source_data)

    def testFilterPrune(self):
        expr = self.expr.filter(self.expr.name == 'name1')
        expr = expr['name', 'id']

        new_expr = ColumnPruning(expr.to_dag()).prune()

        self.assertIsInstance(new_expr.input, FilterCollectionExpr)
        self.assertNotIsInstance(new_expr.input.input, ProjectCollectionExpr)
        self.assertIsNotNone(new_expr.input.input._source_data)

        expected = 'SELECT t1.`name`, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'WHERE t1.`name` == \'name1\''
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.filter(self.expr.name == 'name1')

        new_expr = ColumnPruning(expr.to_dag()).prune()

        self.assertIsInstance(new_expr, FilterCollectionExpr)
        self.assertIsNotNone(new_expr.input._source_data)

        expr = self.expr.filter(self.expr.id.isin(self.expr3.id))

        expected = 'SELECT * \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'WHERE t1.`id` IN (SELECT t3.`id` FROM (  ' \
                   'SELECT t2.`id`   FROM mocked_project.`pyodps_test_expr_table2` t2 ) t3)'
        self.assertTrue(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testFilterPartitionPrune(self):
        expr = self.expr.filter_partition('ds=today')[lambda x: x.fid < 0][
            'name', lambda x: x.id + 1]

        new_expr = ColumnPruning(expr.to_dag()).prune()
        self.assertEqual(set(new_expr.input.input.schema.names), set(['name', 'id', 'fid']))

        expected = "SELECT t2.`name`, t2.`id` + 1 AS `id` \n" \
                   "FROM (\n" \
                   "  SELECT t1.`name`, t1.`id`, t1.`fid` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "  WHERE t1.`ds` == 'today' \n" \
                   ") t2 \n" \
                   "WHERE t2.`fid` < 0"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testSlicePrune(self):
        expr = self.expr.filter(self.expr.fid < 0)[:4]['name', lambda x: x.id + 1]

        new_expr = ColumnPruning(expr.to_dag()).prune()
        self.assertIsNotNone(new_expr.input.input.input._source_data)

        expected = "SELECT t1.`name`, t1.`id` + 1 AS `id` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "WHERE t1.`fid` < 0 \n" \
                   "LIMIT 4"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testGroupbyPrune(self):
        expr = self.expr.groupby('name').agg(id=self.expr.id.max())
        expr = expr[expr.id < 0]['name', ]

        expected = "SELECT t1.`name` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "GROUP BY t1.`name` \n" \
                   "HAVING MAX(t1.`id`) < 0"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby('name').agg(id=self.expr.id.max())
        expr = expr[expr.id < 0]['id',]

        expected = "SELECT MAX(t1.`id`) AS `id` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "GROUP BY t1.`name` \n" \
                   "HAVING MAX(t1.`id`) < 0"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testMutatePrune(self):
        expr = self.expr[self.expr.exclude('birth'), self.expr.fid.astype('int').rename('new_id')]
        expr = expr[expr, expr.groupby('name').mutate(lambda x: x.new_id.cumsum().rename('new_id_sum'))]
        expr = expr[expr.new_id, expr.new_id_sum]

        expected = "SELECT t2.`new_id`, t2.`new_id_sum` \n" \
                   "FROM (\n" \
                   "  SELECT CAST(t1.`fid` AS BIGINT) AS `new_id`, " \
                   "SUM(CAST(t1.`fid` AS BIGINT)) OVER (PARTITION BY t1.`name`) AS `new_id_sum` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   ") t2"

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testValueCountsPrune(self):
        expr = self.expr.name.value_counts()['count', ]
        new_expr = ColumnPruning(expr.to_dag()).prune()

        self.assertIsInstance(new_expr.input.input, ProjectCollectionExpr)
        self.assertEqual(set(new_expr.input.input.schema.names), set(['name']))

    def testSortPrune(self):
        expr = self.expr[self.expr.exclude('name'), self.expr.name.rename('name2')].sort('name2')['id', 'fid']

        expected = "SELECT t2.`id`, t2.`fid` \n" \
                   "FROM (\n" \
                   "  SELECT t1.`id`, t1.`fid`, t1.`name` AS `name2` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "  ORDER BY name2 \n" \
                   "  LIMIT 10000\n" \
                   ") t2"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testDistinctPrune(self):
        expr = self.expr.distinct(self.expr.id + 1, self.expr.name)['name', ]

        expected = "SELECT t2.`name` \n" \
                   "FROM (\n" \
                   "  SELECT DISTINCT t1.`id` + 1 AS `id`, t1.`name` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   ") t2"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testSamplePrune(self):
        expr = self.expr['name', 'id'].sample(parts=5)['id', ]

        expected = "SELECT t1.`id` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "WHERE SAMPLE(5, 1)"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testJoinPrune(self):
        left = self.expr.select(self.expr, type='normal')
        right = self.expr3[:4]
        joined = left.left_join(right, on='id')
        expr = joined.id_x.rename('id')

        expected = "SELECT t2.`id` \n" \
                   "FROM (\n" \
                   "  SELECT t1.`id` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1\n" \
                   ") t2 \n" \
                   "LEFT OUTER JOIN \n" \
                   "  (\n" \
                   "    SELECT t3.`id` \n" \
                   "    FROM mocked_project.`pyodps_test_expr_table2` t3 \n" \
                   "    LIMIT 4\n" \
                   "  ) t4\n" \
                   "ON t2.`id` == t4.`id`"

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        joined = self.expr.join(self.expr2, 'name')

        expected = 'SELECT t1.`name`, t1.`id` AS `id_x`, t1.`fid` AS `fid_x`, ' \
                   't1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, ' \
                   't1.`birth` AS `birth_x`, t1.`ds` AS `ds_x`, t2.`id` AS `id_y`, ' \
                   't2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, ' \
                   't2.`birth` AS `birth_y`, t2.`ds` AS `ds_y` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table2` t2\n' \
                   'ON t1.`name` == t2.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

    def testUnionPrune(self):
        left = self.expr.select('name', 'id')
        right = self.expr3.select(self.expr3.fid.astype('int').rename('id'), self.expr3.name)
        expr = left.union(right)['id']

        expected = "SELECT t3.`id` \n" \
                   "FROM (\n" \
                   "  SELECT t1.`id` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "  UNION ALL\n" \
                   "    SELECT CAST(t2.`fid` AS BIGINT) AS `id` \n" \
                   "    FROM mocked_project.`pyodps_test_expr_table2` t2\n" \
                   ") t3"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.union(self.expr2)

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT * \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  UNION ALL\n' \
                   '    SELECT * \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table2` t2\n' \
                   ') t3'

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))


if __name__ == '__main__':
    unittest.main()
