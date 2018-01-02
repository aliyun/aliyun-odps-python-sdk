#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from odps.tests.core import TestBase, to_str
from odps.compat import unittest
from odps.models import Schema
from odps.df.types import validate_data_type
from odps.df.expr.tests.core import MockTable
from odps.df.expr.expressions import CollectionExpr
from odps.df.backends.odpssql.types import odps_schema_to_df_schema
from odps.df.backends.odpssql.tests.test_compiler import ODPSEngine


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

        schema3 = Schema.from_lists(['id', 'name', 'relatives', 'hobbies'],
                                    datatypes('int64', 'string', 'dict<string, string>', 'list<string>'))
        table4 = MockTable(name='pyodps_test_expr_table', schema=schema)
        self.expr4 = CollectionExpr(_source_data=table4, _schema=schema3)

        self.maxDiff = None

    def testFilterPushdownThroughProjection(self):
        expr = self.expr[self.expr.id + 1, 'name'][lambda x: x.id < 10]

        expected = 'SELECT t1.`id` + 1 AS `id`, t1.`name` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'WHERE (t1.`id` + 1) < 10'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[self.expr.id + 1, 'name', self.expr.name.isnull().rename('is_null')][lambda x: x.is_null]

        expected = 'SELECT t1.`id` + 1 AS `id`, t1.`name`, t1.`name` IS NULL AS `is_null` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'WHERE t1.`name` IS NULL'
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
                   "WHERE (t1.`name` == 'name1') AND (((t1.`id` + 1) * 2) < 3)"
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

    def testGroupbyProjection(self):
        expr = self.expr['id', 'name', 'fid']
        expr2 = expr.groupby('name').agg(count=expr.count(), id=expr.id.sum())
        expr3 = expr2['count', 'id']

        expected = "SELECT COUNT(1) AS `count`, SUM(t1.`id`) AS `id` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "GROUP BY t1.`name`"

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr3, prettify=False)))

        expr = self.expr['id', 'name', 'fid'].filter(self.expr.id < 10)['name', 'id']
        expr2 = expr.groupby('name').agg(count=expr.count(), id=expr.id.sum(), name2=expr.name.max())
        expr3 = expr2[expr2.count + 1, 'id']

        expected = "SELECT COUNT(1) + 1 AS `count`, SUM(t1.`id`) AS `id` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "WHERE t1.`id` < 10 \n" \
                   "GROUP BY t1.`name`"

        self.assertEqual(expected, ODPSEngine(self.odps).compile(expr3, prettify=False))

    def testFilterPushdownThroughMultipleProjection(self):
        schema = Schema.from_lists(list('abcde'), ['string']*5)
        table = MockTable(name='pyodps_test_expr_table3', schema=schema)
        tab = CollectionExpr(_source_data=table, _schema=odps_schema_to_df_schema(schema))

        labels2 = []
        bins2 = []
        for i in range(0, 30):
            a = str(7 * i) + '-' + str(7 * (i + 1))
            b = 7 * i
            bins2.append(b)
            labels2.append(a)

        p1 = tab.select(tab.a,
                        tab.c.astype('int').cut(bins2, labels=labels2, include_over=True).rename('c_cut'),
                        tab.e.astype('int').rename('e'),
                        tab.c.astype('int').rename('c'))
        p1['f'] = p1['e'] / p1['c']
        t = []
        l = []
        for i in range(0, 20):
            a = 1 * i
            b = str(a)
            t.append(a)
            l.append(b)
        p2 = p1.select(p1.a, p1.c_cut, p1.f.cut(bins=t, labels=l, include_over=True).rename('f_cut'))

        expected = "SELECT t1.`a`, CASE WHEN (0 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 7) THEN '0-7' " \
                   "WHEN (7 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 14) " \
                   "THEN '7-14' WHEN (14 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 21) THEN '14-21' " \
                   "WHEN (21 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 28) " \
                   "THEN '21-28' WHEN (28 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 35) THEN '28-35' " \
                   "WHEN (35 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 42) THEN '35-42' " \
                   "WHEN (42 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 49) THEN '42-49' " \
                   "WHEN (49 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 56) " \
                   "THEN '49-56' WHEN (56 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 63) THEN '56-63' " \
                   "WHEN (63 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 70) THEN '63-70' " \
                   "WHEN (70 < CAST(t1.`c` AS BIGINT)) AND (CAST(t1.`c` AS BIGINT) <= 77) " \
                   "THEN '70-77' WHEN (77 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 84) " \
                   "THEN '77-84' WHEN (84 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 91) THEN '84-91' " \
                   "WHEN (91 < CAST(t1.`c` AS BIGINT)) AND (CAST(t1.`c` AS BIGINT) <= 98) " \
                   "THEN '91-98' WHEN (98 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 105) THEN '98-105' " \
                   "WHEN (105 < CAST(t1.`c` AS BIGINT)) AND (CAST(t1.`c` AS BIGINT) <= 112) " \
                   "THEN '105-112' WHEN (112 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 119) THEN '112-119' " \
                   "WHEN (119 < CAST(t1.`c` AS BIGINT)) AND (CAST(t1.`c` AS BIGINT) <= 126) " \
                   "THEN '119-126' WHEN (126 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 133) THEN '126-133' " \
                   "WHEN (133 < CAST(t1.`c` AS BIGINT)) AND (CAST(t1.`c` AS BIGINT) <= 140) " \
                   "THEN '133-140' WHEN (140 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 147) THEN '140-147' " \
                   "WHEN (147 < CAST(t1.`c` AS BIGINT)) AND (CAST(t1.`c` AS BIGINT) <= 154) " \
                   "THEN '147-154' WHEN (154 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 161) THEN '154-161' " \
                   "WHEN (161 < CAST(t1.`c` AS BIGINT)) AND (CAST(t1.`c` AS BIGINT) <= 168) " \
                   "THEN '161-168' WHEN (168 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 175) THEN '168-175' " \
                   "WHEN (175 < CAST(t1.`c` AS BIGINT)) AND (CAST(t1.`c` AS BIGINT) <= 182) " \
                   "THEN '175-182' WHEN (182 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 189) THEN '182-189' " \
                   "WHEN (189 < CAST(t1.`c` AS BIGINT)) AND (CAST(t1.`c` AS BIGINT) <= 196) " \
                   "THEN '189-196' WHEN (196 < CAST(t1.`c` AS BIGINT)) " \
                   "AND (CAST(t1.`c` AS BIGINT) <= 203) THEN '196-203' " \
                   "WHEN 203 < CAST(t1.`c` AS BIGINT) THEN '203-210' END AS `c_cut`, " \
                   "CASE WHEN (0 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 1) THEN '0' " \
                   "WHEN (1 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 2) " \
                   "THEN '1' WHEN (2 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 3) THEN '2' " \
                   "WHEN (3 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 4) " \
                   "THEN '3' WHEN (4 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 5) THEN '4' " \
                   "WHEN (5 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 6) THEN '5' " \
                   "WHEN (6 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 7) " \
                   "THEN '6' WHEN (7 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 8) THEN '7' " \
                   "WHEN (8 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 9) THEN '8' " \
                   "WHEN (9 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 10) " \
                   "THEN '9' WHEN (10 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 11) THEN '10' " \
                   "WHEN (11 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 12) " \
                   "THEN '11' WHEN (12 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 13) THEN '12' " \
                   "WHEN (13 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 14) THEN '13' " \
                   "WHEN (14 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 15) THEN '14' " \
                   "WHEN (15 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 16) THEN '15' " \
                   "WHEN (16 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 17) THEN '16' " \
                   "WHEN (17 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 18) " \
                   "THEN '17' WHEN (18 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 19) THEN '18' " \
                   "WHEN 19 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) THEN '19' END AS `f_cut` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table3` t1 \n" \
                   "WHERE (CASE WHEN (0 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 1) THEN '0' " \
                   "WHEN (1 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 2) " \
                   "THEN '1' WHEN (2 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 3) THEN '2' " \
                   "WHEN (3 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 4) THEN '3' " \
                   "WHEN (4 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 5) THEN '4' " \
                   "WHEN (5 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 6) THEN '5' " \
                   "WHEN (6 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 7) THEN '6' " \
                   "WHEN (7 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 8) THEN '7' " \
                   "WHEN (8 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 9) THEN '8' " \
                   "WHEN (9 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 10) THEN '9' " \
                   "WHEN (10 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 11) THEN '10' " \
                   "WHEN (11 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 12) THEN '11' " \
                   "WHEN (12 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 13) THEN '12' " \
                   "WHEN (13 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 14) THEN '13' " \
                   "WHEN (14 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 15) THEN '14' " \
                   "WHEN (15 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 16) THEN '15' " \
                   "WHEN (16 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 17) THEN '16' " \
                   "WHEN (17 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 18) THEN '17' " \
                   "WHEN (18 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT))) " \
                   "AND ((CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) <= 19) THEN '18' " \
                   "WHEN 19 < (CAST(t1.`e` AS BIGINT) / CAST(t1.`c` AS BIGINT)) THEN '19' END) == '9'"

        self.assertEqual(str(expected), str(ODPSEngine(self.odps).compile(p2[p2.f_cut == '9'], prettify=False)))

    def testFilterPushdownThroughLateralView(self):
        expr = self.expr4[self.expr4.id, self.expr4.name, self.expr4.hobbies.explode('hobby')]
        expr2 = expr[expr.hobby.notnull() & (expr.id < 4)]

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t2.`id`, t2.`name`, t3.`hobby` \n' \
                   '  FROM (\n' \
                   '    SELECT * \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    WHERE t1.`id` < 4\n' \
                   '  ) t2 \n' \
                   '  LATERAL VIEW EXPLODE(t2.`hobbies`) t3 AS `hobby` \n' \
                   ') t4 \n' \
                   'WHERE t4.`hobby` IS NOT NULL'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr2, prettify=False)))

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, t1.`name`, t2.`hobby` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  LATERAL VIEW EXPLODE(t1.`hobbies`) t2 AS `hobby` \n' \
                   ') t3 \n' \
                   'WHERE (t3.`hobby` IS NOT NULL) OR (t3.`id` < 4)'
        expr3 = expr[expr.hobby.notnull() | (expr.id < 4)]
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr3, prettify=False)))


if __name__ == '__main__':
    unittest.main()