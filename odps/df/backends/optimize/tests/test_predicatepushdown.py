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
from odps.df.types import validate_data_type
from odps.df.expr.tests.core import MockTable
from odps.df.expr.expressions import CollectionExpr
from odps.df.backends.odpssql.engine import ODPSEngine


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
                                   ['ds'], datatypes('string'))
        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        self.expr = CollectionExpr(_source_data=table, _schema=schema)

        table1 = MockTable(name='pyodps_test_expr_table1', schema=schema)
        self.expr1 = CollectionExpr(_source_data=table1, _schema=schema)

        table2 = MockTable(name='pyodps_test_expr_table2', schema=schema)
        self.expr2 = CollectionExpr(_source_data=table2, _schema=schema)

        schema2 = Schema.from_lists(['name', 'id', 'fid'], datatypes('string', 'int64', 'float64'),
                                    ['part1', 'part2'], datatypes('string', 'int64'))
        table3 = MockTable(name='pyodps_test_expr_table2', schema=schema2)
        self.expr3 = CollectionExpr(_source_data=table3, _schema=schema2)

        self.maxDiff = None

    def testFilterPushdownThroughProjection(self):
        expr = self.expr[self.expr.id + 1, 'name'][lambda x: x.id < 10]

        expected = 'SELECT t1.`id` + 1 AS `id`, t1.`name` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'WHERE (t1.`id` + 1) < 10'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr['name', self.expr.id ** 2]\
            .filter(lambda x: x.name == 'name1').filter(lambda x: x.id < 3)
        expected = "SELECT t1.`name`, CAST(POW(t1.`id`, 2) AS BIGINT) AS `id` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "WHERE (t1.`name` == 'name1') AND ((CAST(POW(t1.`id`, 2) AS BIGINT)) < 3)"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr['name', self.expr.id + 1].filter(lambda x: x.name == 'name1')[
            lambda x: 'tt' + x.name, 'id'
        ].filter(lambda x: x.id < 3)

        expected = "SELECT CONCAT('tt', t1.`name`) AS `name`, t1.`id` + 1 AS `id` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "WHERE (t1.`name` == 'name1') AND ((t1.`id` + 1) < 3)"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.filter(self.expr.name == 'name1').select('name', lambda x: (x.id + 1) * 2)[
            lambda x: 'tt' + x.name, 'id'
        ].filter(lambda x: x.id < 3)
        expected = "SELECT CONCAT('tt', t1.`name`) AS `name`, (t1.`id` + 1) * 2 AS `id` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "WHERE (((t1.`id` + 1) * 2) < 3) AND (t1.`name` == 'name1')"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.filter(self.expr.id.between(2, 6),
                                self.expr.name.lower().contains('pyodps', regex=False)).name.nunique()
        expected = "SELECT COUNT(DISTINCT t2.`name`) AS `name_nunique` \n" \
                   "FROM (\n" \
                   "  SELECT t1.`id`, t1.`name` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "  WHERE ((t1.`id` >= 2) AND (t1.`id` <= 6)) AND INSTR(TOLOWER(t1.`name`), 'pyodps') > 0 \n" \
                   ") t2"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testFilterPushDownThroughJoin(self):
        expr = self.expr.join(self.expr3, on='name')
        expr = expr[(expr.id_x < 10) & (expr.fid_y > 3)]

        expected = 'SELECT t2.`name`, t2.`id` AS `id_x`, t2.`fid` AS `fid_x`, ' \
                   't2.`isMale`, t2.`scale`, t2.`birth`, t2.`ds`, t4.`id` AS `id_y`, ' \
                   't4.`fid` AS `fid_y`, t4.`part1`, t4.`part2` \n' \
                   'FROM (\n' \
                   '  SELECT * \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  WHERE t1.`id` < 10\n' \
                   ') t2 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT * \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table2` t3 \n' \
                   '    WHERE t3.`fid` > 3\n' \
                   '  ) t4\n' \
                   'ON t2.`name` == t4.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.join(self.expr3, on='name')
        expr = expr[(expr.id_x < 10) & (expr.fid_y > 3) & (expr.id_x > 3)]

        expected = 'SELECT t2.`name`, t2.`id` AS `id_x`, t2.`fid` AS `fid_x`, ' \
                   't2.`isMale`, t2.`scale`, t2.`birth`, t2.`ds`, t4.`id` AS `id_y`, ' \
                   't4.`fid` AS `fid_y`, t4.`part1`, t4.`part2` \n' \
                   'FROM (\n' \
                   '  SELECT * \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  WHERE (t1.`id` < 10) AND (t1.`id` > 3)\n' \
                   ') t2 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT * \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table2` t3 \n' \
                   '    WHERE t3.`fid` > 3\n' \
                   '  ) t4\n' \
                   'ON t2.`name` == t4.`name`'

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[self.expr.name, self.expr.id + 1]
        expr2 = self.expr3['tt' + self.expr3.name, self.expr3.id.rename('id2')]
        expr = expr.join(expr2, on='name')
        expr = expr[((expr.id < 10) | (expr.id > 100)) & (expr.id2 > 3)]

        expected = "SELECT t2.`name`, t2.`id`, t4.`id2` \n" \
                   "FROM (\n" \
                   "  SELECT t1.`name`, t1.`id` + 1 AS `id` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "  WHERE ((t1.`id` + 1) < 10) OR ((t1.`id` + 1) > 100)\n" \
                   ") t2 \n" \
                   "INNER JOIN \n" \
                   "  (\n" \
                   "    SELECT CONCAT('tt', t3.`name`) AS `name`, t3.`id` AS `id2` \n" \
                   "    FROM mocked_project.`pyodps_test_expr_table2` t3 \n" \
                   "    WHERE t3.`id` > 3\n" \
                   "  ) t4\n" \
                   "ON t2.`name` == t4.`name`"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.join(self.expr3, on='name')
        expr = expr[(expr.id_x + expr.id_y < 10) & (expr.id_x > 3)]

        expected = "SELECT * \n" \
                   "FROM (\n" \
                   "  SELECT t2.`name`, t2.`id` AS `id_x`, t2.`fid` AS `fid_x`, t2.`isMale`, " \
                   "t2.`scale`, t2.`birth`, t2.`ds`, t3.`id` AS `id_y`, " \
                   "t3.`fid` AS `fid_y`, t3.`part1`, t3.`part2` \n" \
                   "  FROM (\n" \
                   "    SELECT * \n" \
                   "    FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "    WHERE t1.`id` > 3\n" \
                   "  ) t2 \n" \
                   "  INNER JOIN \n" \
                   "    mocked_project.`pyodps_test_expr_table2` t3\n" \
                   "  ON t2.`name` == t3.`name` \n" \
                   ") t4 \n" \
                   "WHERE (t4.`id_x` + t4.`id_y`) < 10"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.outer_join(self.expr3, on='name')
        expr = expr[(expr.id_x + expr.id_y < 10) & (expr.id_x > 3)]

        expected = "SELECT * \n" \
                   "FROM (\n" \
                   "  SELECT t1.`name` AS `name_x`, t1.`id` AS `id_x`, t1.`fid` AS `fid_x`, " \
                   "t1.`isMale`, t1.`scale`, t1.`birth`, t1.`ds`, t2.`name` AS `name_y`, " \
                   "t2.`id` AS `id_y`, t2.`fid` AS `fid_y`, t2.`part1`, t2.`part2` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "  FULL OUTER JOIN \n" \
                   "    mocked_project.`pyodps_test_expr_table2` t2\n" \
                   "  ON t1.`name` == t2.`name` \n" \
                   ") t3 \n" \
                   "WHERE ((t3.`id_x` + t3.`id_y`) < 10) AND (t3.`id_x` > 3)"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.join(self.expr3, on=['name', self.expr.id == self.expr3.id,
                                              self.expr.id < 10, self.expr3.name == 'name1',
                                              self.expr.id > 5])

        expected = 'SELECT t2.`name`, t2.`id`, t2.`fid` AS `fid_x`, t2.`isMale`, ' \
                   't2.`scale`, t2.`birth`, t2.`ds`, t4.`fid` AS `fid_y`, t4.`part1`, t4.`part2` \n' \
                   'FROM (\n' \
                   '  SELECT * \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  WHERE (t1.`id` < 10) AND (t1.`id` > 5)\n' \
                   ') t2 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT * \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table2` t3 \n' \
                   '    WHERE t3.`name` == \'name1\'\n' \
                   '  ) t4\n' \
                   'ON (t2.`name` == t4.`name`) AND (t2.`id` == t4.`id`)'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.left_join(self.expr3, on=['name', self.expr.id == self.expr3.id,
                                                   self.expr.id < 10, self.expr3.name == 'name1',
                                                   self.expr.id > 5])
        expected = 'SELECT t1.`name` AS `name_x`, t1.`id` AS `id_x`, t1.`fid` AS `fid_x`, t1.`isMale`, ' \
                   't1.`scale`, t1.`birth`, t1.`ds`, t2.`name` AS `name_y`, t2.`id` AS `id_y`, ' \
                   't2.`fid` AS `fid_y`, t2.`part1`, t2.`part2` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'LEFT OUTER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table2` t2\n' \
                   'ON ((((t1.`name` == t2.`name`) AND (t1.`id` == t2.`id`)) ' \
                   "AND (t1.`id` < 10)) AND (t2.`name` == 'name1')) AND (t1.`id` > 5)"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testFilterPushdownThroughUnion(self):
        expr = self.expr['name', 'id'].union(self.expr2['id', 'name'])
        expr = expr.filter(expr.id + 1 < 3)

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, t1.`id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  WHERE (t1.`id` + 1) < 3 \n' \
                   '  UNION ALL\n' \
                   '    SELECT t2.`name`, t2.`id` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table2` t2 \n' \
                   '    WHERE (t2.`id` + 1) < 3\n' \
                   ') t3'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr1 = self.expr.filter(self.expr.id == 1)['name', 'id']
        expr2 = self.expr.filter(self.expr.id == 0)['id', 'name']
        expr = expr1.union(expr2)

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, t1.`id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  WHERE t1.`id` == 1 \n' \
                   '  UNION ALL\n' \
                   '    SELECT t2.`name`, t2.`id` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t2 \n' \
                   '    WHERE t2.`id` == 0\n' \
                   ') t3'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))


if __name__ == '__main__':
    unittest.main()