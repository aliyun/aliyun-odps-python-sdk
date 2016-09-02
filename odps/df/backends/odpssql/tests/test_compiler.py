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

import re
from datetime import datetime, timedelta
from decimal import Decimal

from odps import options
from odps.tests.core import TestBase, to_str
from odps.compat import unittest, six
from odps.udf.tools import runners
from odps.models import Schema
from odps.utils import to_timestamp, to_milliseconds
from odps.df import output
from odps.df.types import validate_data_type
from odps.df.expr.expressions import CollectionExpr, BuiltinFunction
from odps.df.backends.odpssql.engine import ODPSEngine, UDF_CLASS_NAME
from odps.df.backends.odpssql.compiler import BINARY_OP_COMPILE_DIC, \
    MATH_COMPILE_DIC, DATE_PARTS_DIC
from odps.df.backends.errors import CompileError
from odps.df.expr.tests.core import MockTable
from odps.df import Scalar, NullScalar, switch, year, month, day, hour, minute, second, millisecond

# required by cloudpickle tests
six.exec_("""
import base64
import time
import inspect
from odps.lib.cloudpickle import *
from odps.lib.importer import *
""", globals(), locals())


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
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

        # turn off the column pruning and predicate pushdown
        # for the purpose not to modify the case
        # we only need to ensure the correctness of engine execution
        options.df.optimizes.cp = False
        options.df.optimizes.pp = False

        self.maxDiff = None

    def teardown(self):
        options.df.optimizes.cp = True
        options.df.optimizes.pp = True

    def _clear_functions(self, engine):
        engine._ctx._registered_funcs.clear()
        engine._ctx._func_to_udfs.clear()

    def _testify_udf(self, expected, inputs, engine):
        udf = list(engine._ctx._func_to_udfs.values())[0]
        six.exec_(udf, globals(), locals())
        udf = locals()[UDF_CLASS_NAME]
        self.assertSequenceEqual(expected, runners.simple_run(udf, inputs))

        self._clear_functions(engine)

    def testBaseCompilation(self):
        expr = self.expr[self.expr.id < 10]['name', lambda x: x.id]
        expected = 'SELECT t1.`name`, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'WHERE t1.`id` < 10'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[Scalar(3).rename('const'),
                         NullScalar('string').rename('string_const'),
                         self.expr.id]
        expected = 'SELECT 3 AS `const`, CAST(NULL AS STRING) AS `string_const`, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[self.expr, (self.expr.id + 1).rename('id2')]
        expected = 'SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, ' \
                   't1.`scale`, t1.`birth`, t1.`id` + 1 AS `id2` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[lambda x: x.exclude('name'), (self.expr.id + 1).rename('id2')]
        expected = 'SELECT t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale`, t1.`birth`, t1.`id` + 1 AS `id2` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
        expr = self.expr[self.expr, self.expr.id.cut(list(range(0, 81, 10)), labels=labels).rename('id_group')]
        expr = expr['id', 'id_group'].distinct()[:10]
        expected = "SELECT DISTINCT t1.`id`, CASE " \
                   "WHEN (0 < t1.`id`) AND (t1.`id` <= 10) THEN '0-9' " \
                   "WHEN (10 < t1.`id`) AND (t1.`id` <= 20) THEN '10-19' " \
                   "WHEN (20 < t1.`id`) AND (t1.`id` <= 30) THEN '20-29' " \
                   "WHEN (30 < t1.`id`) AND (t1.`id` <= 40) THEN '30-39' " \
                   "WHEN (40 < t1.`id`) AND (t1.`id` <= 50) THEN '40-49' " \
                   "WHEN (50 < t1.`id`) AND (t1.`id` <= 60) THEN '50-59' " \
                   "WHEN (60 < t1.`id`) AND (t1.`id` <= 70) THEN '60-69' " \
                   "WHEN (70 < t1.`id`) AND (t1.`id` <= 80) THEN '70-79' END AS `id_group` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "LIMIT 10"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby(lambda x: x.name).agg(id_count=self.expr.id.count(),
                                                       id_mean=lambda x: x.id.mean())
        expr = expr[expr.id_count >= 100].sort(['id_mean', 'id_count'], ascending=False)

        expected = 'SELECT t1.`name`, COUNT(t1.`id`) AS `id_count`, AVG(t1.`id`) AS `id_mean` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name` \n' \
                   'HAVING COUNT(t1.`id`) >= 100 \n' \
                   'ORDER BY id_mean DESC, id_count DESC \n' \
                   'LIMIT 10000'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[self.expr.id < 100].groupby('name').agg(id=lambda x: x.id.sum()).sort('id')[:1000]['id']
        expected = 'SELECT t2.`id` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, SUM(t1.`id`) AS `id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  WHERE t1.`id` < 100 \n' \
                   '  GROUP BY t1.`name` \n' \
                   '  ORDER BY id \n' \
                   '  LIMIT 1000\n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[[self.expr.name.extract('/projects/(.*?)/', group=1).rename('project')]]\
            .groupby('project').count()
        expected = "SELECT COUNT(1) AS `count` \n" \
                   "FROM (\n" \
                   "  SELECT REGEXP_EXTRACT(t1.`name`, '/projects/(.*?)/', 1) AS `project` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   ") t2 \n" \
                   "GROUP BY t2.`project`"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby(
                self.expr.name.extract('/projects/(.*?)/', group=1).rename('project')).count()
        expected = "SELECT COUNT(1) AS `count` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "GROUP BY REGEXP_EXTRACT(t1.`name`, '/projects/(.*?)/', 1)"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.describe()
        expected = 'SELECT SUM(IF(t1.`id` IS NOT NULL, 1, 0)) AS `id_count`, ' \
                   'MIN(t1.`id`) AS `id_min`, MAX(t1.`id`) AS `id_max`, ' \
                   'AVG(t1.`id`) AS `id_mean`, STDDEV(t1.`id`) AS `id_std`, ' \
                   'SUM(IF(t1.`fid` IS NOT NULL, 1, 0)) AS `fid_count`, ' \
                   'MIN(t1.`fid`) AS `fid_min`, MAX(t1.`fid`) AS `fid_max`, ' \
                   'AVG(t1.`fid`) AS `fid_mean`, STDDEV(t1.`fid`) AS `fid_std`, ' \
                   'SUM(IF(t1.`scale` IS NOT NULL, 1, 0)) AS `scale_count`, ' \
                   'MIN(t1.`scale`) AS `scale_min`, MAX(t1.`scale`) AS `scale_max`, ' \
                   'AVG(t1.`scale`) AS `scale_mean`, STDDEV(t1.`scale`) AS `scale_std` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[self.expr, Scalar(3).rename('const')][
            lambda x: x.exclude('isMale'), lambda x: (x.const + 1).rename('const2')]
        expected = 'SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`scale`, t1.`birth`, ' \
                   '3 AS `const`, 3 + 1 AS `const2` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.select(pt=BuiltinFunction('max_pt', args=(self.expr._source_data.name, )))
        expected = "SELECT max_pt('pyodps_test_expr_table') AS `pt` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.sample(10, columns='name')
        expected = 'SELECT * \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'WHERE SAMPLE(10, 1, t1.`name`)'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.sample(parts=10, i=(2, 3, 4), columns=self.expr.id)
        expected = 'SELECT * \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'WHERE (SAMPLE(10, 3, t1.`id`) OR SAMPLE(10, 4, t1.`id`)) OR SAMPLE(10, 5, t1.`id`)'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr['name', self.expr.id + 1][lambda x: x.id < 1]
        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, t1.`id` + 1 AS `id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   ') t2 \n' \
                   'WHERE t2.`id` < 1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby('name').agg(count=self.expr.id.count()).limit(10)[lambda x: x['count'] > 10]
        expected = "SELECT * \n" \
                   "FROM (\n" \
                   "  SELECT t1.`name`, COUNT(t1.`id`) AS `count` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "  GROUP BY t1.`name` \n" \
                   "  LIMIT 10\n" \
                   ") t2 \n" \
                   "WHERE t2.`count` > 10"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testElementCompilation(self):
        expect = 'SELECT t1.`id` IS NULL AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.id.isnull(), prettify=False)))

        expect = 'SELECT t1.`id` IS NOT NULL AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.id.notnull(), prettify=False)))

        expect = 'SELECT IF(t1.`id` IS NULL, 100, t1.`id`) AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.id.fillna(100), prettify=False)))

        expr = self.expr.id.isin([1, 2, 3]).rename('id')
        expect = 'SELECT t1.`id` IN (1, 2, 3) AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.id.isin(self.expr.fid.astype('int')).rename('id')
        expect = 'SELECT t1.`id` IN ' \
                 '(SELECT CAST(t1.`fid` AS BIGINT) AS `fid` FROM mocked_project.`pyodps_test_expr_table` t1) ' \
                 'AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.filter(self.expr.id.isin(self.expr3.id))
        expect = 'SELECT * \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                 'WHERE t1.`id` IN (SELECT t2.`id` FROM mocked_project.`pyodps_test_expr_table2` t2)'
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.id.notin([1, 2, 3]).rename('id')
        expect = 'SELECT t1.`id` NOT IN (1, 2, 3) AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.id.notin(self.expr.fid.astype('int')).rename('id')
        expect = 'SELECT t1.`id` NOT IN ' \
                 '(SELECT CAST(t1.`fid` AS BIGINT) AS `fid` FROM mocked_project.`pyodps_test_expr_table` t1) ' \
                 'AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expect = 'SELECT (t1.`fid` <= t1.`id`) AND (t1.`id` <= 3) AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.id.between(self.expr.fid, 3), prettify=False)))

        expect = 'SELECT IF(t1.`id` < 5, \'test\', CONCAT(t1.`name`, \'abc\')) AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         ODPSEngine(self.odps).compile(
                             (self.expr.id < 5).ifelse('test', self.expr.name + 'abc').rename('id'),
                             prettify=False))

        expr = self.expr.name.switch('test', 'test' + self.expr.name,
                                     'test2', 'test2' + self.expr.name).rename('name')
        expect = 'SELECT CASE t1.`name` WHEN \'test\' THEN CONCAT(\'test\', t1.`name`) ' \
                 'WHEN \'test2\' THEN CONCAT(\'test2\', t1.`name`) END AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = switch(self.expr.name == 'test', 'test2', default='notest').rename('name')
        expect = "SELECT CASE WHEN t1.`name` == 'test' THEN 'test2' ELSE 'notest' END AS `name` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.id.cut([100, 200, 300], labels=['small', 'large']).rename('tp')
        expect = 'SELECT CASE WHEN (100 < t1.`id`) AND (t1.`id` <= 200) THEN \'small\' ' \
                 'WHEN (200 < t1.`id`) AND (t1.`id` <= 300) THEN \'large\' END AS `tp` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.id.cut([100, 200, 300],
                                labels=['xsmall', 'small', 'large', 'xlarge'],
                                include_under=True, include_over=True).rename('tp')
        expect = "SELECT CASE WHEN t1.`id` <= 100 THEN 'xsmall' " \
                 "WHEN (100 < t1.`id`) AND (t1.`id` <= 200) THEN 'small' " \
                 "WHEN (200 < t1.`id`) AND (t1.`id` <= 300) THEN 'large' " \
                 "WHEN 300 < t1.`id` THEN 'xlarge' END AS `tp` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.id.cut([100, 200, 300],
                                labels=['xsmall', 'small', 'large', 'xlarge'],
                                include_lowest=True,
                                include_under=True, include_over=True).rename('tp')
        expect = "SELECT CASE WHEN t1.`id` < 100 THEN 'xsmall' " \
                 "WHEN (100 <= t1.`id`) AND (t1.`id` <= 200) THEN 'small' " \
                 "WHEN (200 < t1.`id`) AND (t1.`id` <= 300) THEN 'large' " \
                 "WHEN 300 < t1.`id` THEN 'xlarge' END AS `tp` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.scale.map(lambda x: x + 1)
        engine = ODPSEngine(self.odps)
        engine.compile(expr)
        data = ['2.3', '3.3']
        self._testify_udf([str(Decimal(d) + 1) for d in data], [(d, ) for d in data], engine)

    def testArithmeticCompilation(self):
        for e in (self.expr.id + 5, self.expr.id - 5, self.expr.id * 5,
                  self.expr.id / 5, self.expr.id > 5, 5 < self.expr.id,
                  self.expr.id >= 5, 5 <= self.expr.id, self.expr.id < 5,
                  self.expr.id <= 5, self.expr.id == 5, self.expr.id != 5):
            expect = 'SELECT t1.`id` {0} 5 AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'.format(
                BINARY_OP_COMPILE_DIC[e.__class__.__name__])

            self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(e, prettify=False)))

        for e in (5 - self.expr.id, 5 * self.expr.id, 5 / self.expr.id):
            expect = 'SELECT 5 {0} t1.`id` AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'.format(
                BINARY_OP_COMPILE_DIC[e.__class__.__name__])

            self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(e, prettify=False)))

        for e in (self.expr.isMale & True, self.expr.isMale | True):
            expect = 'SELECT t1.`isMale` {0} true AS `isMale` \nFROM mocked_project.`pyodps_test_expr_table` t1'.format(
                BINARY_OP_COMPILE_DIC[e.__class__.__name__].upper())

            self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(e, prettify=False)))

        self.assertEqual(to_str('SELECT -t1.`id` AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'),
                         to_str(ODPSEngine(self.odps).compile((-self.expr.id), prettify=False)))

        now = datetime.now()
        unix_time = to_timestamp(now)
        expr = self.expr.birth < now
        expect = 'SELECT t1.`birth` < FROM_UNIXTIME(%s) AS `birth` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1' % unix_time
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.birth + year(1)
        expect = "SELECT DATEADD(t1.`birth`, 1, 'yyyy') AS `birth` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.birth + month(1)
        expect = "SELECT DATEADD(t1.`birth`, 1, 'mm') AS `birth` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.birth + day(1)
        expect = "SELECT DATEADD(t1.`birth`, 1, 'dd') AS `birth` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.birth + hour(1)
        expect = "SELECT DATEADD(t1.`birth`, 1, 'hh') AS `birth` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.birth + minute(1)
        expect = "SELECT DATEADD(t1.`birth`, 1, 'mi') AS `birth` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.birth + second(1)
        expect = "SELECT DATEADD(t1.`birth`, 1, 'ss') AS `birth` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        data = [datetime.now()]

        expr = self.expr.birth - millisecond(100)
        engine = ODPSEngine(self.odps)
        engine.compile(expr)
        self._testify_udf([to_milliseconds(d - timedelta(milliseconds=100)) for d in data],
                          [(d, 100, '-') for d in data], engine)

        expr = self.expr.birth - datetime.now()
        engine = ODPSEngine(self.odps)
        engine.compile(expr)
        now = datetime.now()
        self._testify_udf([(d - now).microseconds // 1000 for d in data],
                          [(d, now, '-') for d in data], engine)

        expr = self.expr.scale < Decimal('3.14')
        expect = "SELECT t1.`scale` < CAST('3.14' AS DECIMAL) AS `scale` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        data = [(30, 8)]

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.id // 8)
        self._testify_udf([l // r for l, r in data], [d for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile((self.expr.id // self.expr.id).rename('id'))
        self._testify_udf([l // l for l, r in data], [d for d in [(30, 30)]], engine)

        expr = 'tt' + self.expr.id.astype('string')
        expect = "SELECT CONCAT('tt', CAST(t1.`id` AS STRING)) AS `id` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testMathCompilation(self):
        for math_cls, func in MATH_COMPILE_DIC.items():
            e = getattr(self.expr.id, math_cls.lower())()

            expect = 'SELECT {0}(t1.`id`) AS `id` \n' \
                     'FROM mocked_project.`pyodps_test_expr_table` t1'.format(func.upper())

            self.assertEqual(to_str(expect), to_str(ODPSEngine(self.odps).compile(e, prettify=False)))

        expect = 'SELECT LN(t1.`id`) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.id.log(), prettify=False)))

        expect = 'SELECT LOG(2, t1.`id`) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.id.log2(), prettify=False)))

        expect = 'SELECT LOG(10, t1.`id`) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.id.log10(), prettify=False)))

        expect = 'SELECT LN(1 + t1.`id`) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.id.log1p(), prettify=False)))

        expect = 'SELECT EXP(t1.`id`) - 1 AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.id.expm1(), prettify=False)))

        expect = 'SELECT TRUNC(t1.`fid`) AS `fid` \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.fid.trunc(), prettify=False)))

        expect = 'SELECT TRUNC(t1.`fid`, 2) AS `fid` \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.fid.trunc(2), prettify=False)))

        data = [30]
        try:
            import numpy as np
        except ImportError:
            return

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.fid.arccosh())
        self._testify_udf([np.arccosh(d) for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.fid.arcsinh())
        self._testify_udf([np.arcsinh(d) for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.fid.degrees())
        self._testify_udf([np.degrees(d) for d in data], [(d,) for d in data], engine)

        data = [0.2]

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.fid.arctanh())
        self._testify_udf([np.arctanh(d) for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.fid.radians())
        self._testify_udf([np.radians(d) for d in data], [(d,) for d in data], engine)

    def testStringCompilation(self):
        expect = 'SELECT CONCAT(TOUPPER(SUBSTR(t1.`name`, 1, 1)), TOLOWER(SUBSTR(t1.`name`, 2))) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.capitalize(), prettify=False)))

        expect = 'SELECT REGEXP_INSTR(t1.`name`, \'test\') > 0 AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.contains('test'), prettify=False)))
        expect = 'SELECT INSTR(t1.`name`, \'test\') > 0 AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.contains('test', regex=False), prettify=False)))
        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.contains('test', flags=re.I))
        self._testify_udf([True, False, True], [('Test',), ('tes',), ('ToTEst',)], engine)

        expect = 'SELECT REGEXP_COUNT(t1.`name`, \'test\') AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.count('test'), prettify=False)))

        expect = 'SELECT INSTR(REVERSE(t1.`name`), REVERSE(\'test\')) == 1 AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.endswith('test'), prettify=False)))

        expect = 'SELECT INSTR(t1.`name`, \'test\') == 1 AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.startswith('test'), prettify=False)))

        expect = 'SELECT REGEXP_EXTRACT(t1.`name`, \'test\', 0) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.extract('test'), prettify=False)))
        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.extract('test\d', flags=re.I))
        self._testify_udf(['Test1', None, 'test3'], [('Test1',), ('tes',), ('test32',)], engine)

        expect = 'SELECT INSTR(t1.`name`, \'test\', 1) - 1 AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.find('test'), prettify=False)))
        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.rfind('test'))
        data = ['abcdggtest1222', 'notes', 'test']
        self._testify_udf([d.rfind('test') for d in data], [(d,) for d in data], engine)

        expect = 'SELECT REGEXP_REPLACE(t1.`name`, \'test\', \'test2\', 0) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.replace('test', 'test2'), prettify=False)))

        expect = 'SELECT SUBSTR(t1.`name`, 3, 1) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.get(2), prettify=False)))

        expect = 'SELECT LENGTH(t1.`name`) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.len(), prettify=False)))

        expect = "SELECT CONCAT(t1.`name`, IF((10 - LENGTH(t1.`name`)) >= 0, " \
                 "REPEAT(' ', 10 - LENGTH(t1.`name`)), '')) AS `name` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.ljust(10), prettify=False)))
        expect = "SELECT CONCAT(t1.`name`, IF((10 - LENGTH(t1.`name`)) >= 0, " \
                 "REPEAT('*', 10 - LENGTH(t1.`name`)), '')) AS `name` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.ljust(10, fillchar='*'), prettify=False)))

        expect = "SELECT CONCAT(IF((10 - LENGTH(t1.`name`)) >= 0, " \
                 "REPEAT(' ', 10 - LENGTH(t1.`name`)), ''), t1.`name`) AS `name` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.rjust(10), prettify=False)))

        expect = 'SELECT TOLOWER(t1.`name`) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.lower(), prettify=False)))

        expect = 'SELECT TOUPPER(t1.`name`) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.upper(), prettify=False)))

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.lstrip())
        data = [' abc\n', ' \nddd']
        self._testify_udf([d.lstrip() for d in data], [(d,) for d in data], engine)
        expect = 'SELECT LTRIM(t1.`name`) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.lstrip(to_strip=' '), prettify=False)))

        expect = 'SELECT RTRIM(t1.`name`) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.rstrip(to_strip=' '), prettify=False)))

        expect = 'SELECT TRIM(t1.`name`) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.strip(to_strip=' '), prettify=False)))

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.pad(10))
        data = ['ab', 'a' * 12]
        self._testify_udf([d.rjust(10) for d in data], [(d,) for d in data], engine)

        expect = 'SELECT REPEAT(t1.`name`, 4) AS `name` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.repeat(4), prettify=False)))

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.slice(0, 10, 2))
        data = ['ab' * 15, 'def']
        self._testify_udf([d[0: 10: 2] for d in data], [(d, 0, 10, 2) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.slice(-5, -1))
        data = ['ab' * 15, 'def']
        self._testify_udf([d[-5: -1] for d in data], [(d, -5, -1) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.slice(-5, step=1).value_counts(sort=False))
        data = ['ab' * 15, 'def']
        self._testify_udf([d[-5:] for d in data], [(d, -5, 1) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.title())
        data = ['Abc Def', 'ADEFddEE']
        self._testify_udf([d.title() for d in data], [(d,) for d in data], engine)

        expect = "SELECT CONCAT(IF((10 - LENGTH(t1.`name`)) >= 0, " \
                 "REPEAT('0', 10 - LENGTH(t1.`name`)), ''), t1.`name`) AS `name` \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.name.zfill(10), prettify=False)))

        data = ['123', 'dEf', '124df']

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.isalnum())
        self._testify_udf([d.isalnum() for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.isalpha())
        self._testify_udf([d.isalpha() for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.isdigit())
        self._testify_udf([d.isdigit() for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.isspace())
        self._testify_udf([d.isspace() for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.islower())
        self._testify_udf([d.islower() for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.isupper())
        self._testify_udf([d.isupper() for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.istitle())
        self._testify_udf([d.istitle() for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.isnumeric())
        self._testify_udf([to_str(d).isnumeric() for d in data], [(d,) for d in data], engine)

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.name.isdecimal())
        self._testify_udf([to_str(d).isdecimal() for d in data], [(d,) for d in data], engine)

    def testValueCounts(self):
        labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
        expr = self.expr.id.cut(list(range(0, 81, 10)), labels=labels).rename('id_group').value_counts()

        expected = "SELECT CASE WHEN (0 < t1.`id`) AND (t1.`id` <= 10) THEN '0-9' WHEN (10 < t1.`id`) " \
                   "AND (t1.`id` <= 20) THEN '10-19' WHEN (20 < t1.`id`) AND (t1.`id` <= 30) THEN '20-29' " \
                   "WHEN (30 < t1.`id`) AND (t1.`id` <= 40) THEN '30-39' WHEN (40 < t1.`id`) AND (t1.`id` <= 50) " \
                   "THEN '40-49' WHEN (50 < t1.`id`) AND (t1.`id` <= 60) THEN '50-59' " \
                   "WHEN (60 < t1.`id`) AND (t1.`id` <= 70) THEN '60-69' WHEN (70 < t1.`id`) AND (t1.`id` <= 80) " \
                   "THEN '70-79' END AS `id_group`, COUNT(CASE WHEN (0 < t1.`id`) AND (t1.`id` <= 10) THEN '0-9' " \
                   "WHEN (10 < t1.`id`) AND (t1.`id` <= 20) THEN '10-19' WHEN (20 < t1.`id`) AND (t1.`id` <= 30) " \
                   "THEN '20-29' WHEN (30 < t1.`id`) AND (t1.`id` <= 40) THEN '30-39' " \
                   "WHEN (40 < t1.`id`) AND (t1.`id` <= 50) THEN '40-49' WHEN (50 < t1.`id`) AND (t1.`id` <= 60) " \
                   "THEN '50-59' WHEN (60 < t1.`id`) AND (t1.`id` <= 70) THEN '60-69' WHEN (70 < t1.`id`) " \
                   "AND (t1.`id` <= 80) THEN '70-79' END) AS `count` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "GROUP BY CASE WHEN (0 < t1.`id`) AND (t1.`id` <= 10) THEN '0-9' WHEN (10 < t1.`id`) " \
                   "AND (t1.`id` <= 20) THEN '10-19' WHEN (20 < t1.`id`) AND (t1.`id` <= 30) " \
                   "THEN '20-29' WHEN (30 < t1.`id`) AND (t1.`id` <= 40) THEN '30-39' WHEN (40 < t1.`id`) " \
                   "AND (t1.`id` <= 50) THEN '40-49' WHEN (50 < t1.`id`) AND (t1.`id` <= 60) " \
                   "THEN '50-59' WHEN (60 < t1.`id`) AND (t1.`id` <= 70) THEN '60-69' WHEN (70 < t1.`id`) " \
                   "AND (t1.`id` <= 80) THEN '70-79' END \n" \
                   "ORDER BY count DESC \n" \
                   "LIMIT 10000"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.id.value_counts().sort('id')['id', 'count']
        expected = 'SELECT t2.`id`, t2.`count` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, COUNT(t1.`id`) AS `count` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  GROUP BY t1.`id` \n' \
                   '  ORDER BY count DESC \n' \
                   '  LIMIT 10000\n' \
                   ') t2 \n' \
                   'ORDER BY id \n' \
                   'LIMIT 10000'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testDatetimeCompilation(self):
        self.assertRaises(NotImplementedError,
                          lambda: ODPSEngine(self.odps).compile(self.expr.birth.time))

        for clz, date_part in DATE_PARTS_DIC.items():
            expect = 'SELECT DATEPART(t1.`birth`, \'%s\') AS `birth` \n' \
                     'FROM mocked_project.`pyodps_test_expr_table` t1' % date_part
            attr = getattr(self.expr.birth, clz.lower())
            self.assertEqual(to_str(expect),
                             to_str(ODPSEngine(self.odps).compile(attr, prettify=False)))

        self.assertRaises(NotImplementedError,
                          lambda: ODPSEngine(self.odps).compile(self.expr.birth.microsecond))

        self.assertRaises(NotImplementedError,
                          lambda: ODPSEngine(self.odps).compile(self.expr.birth.week))

        expect = 'SELECT WEEKOFYEAR(t1.`birth`) AS `birth` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.birth.weekofyear, prettify=False)))

        expect = 'SELECT WEEKDAY(t1.`birth`) AS `birth` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.birth.dayofweek, prettify=False)))
        self.assertEqual(to_str(expect),
                         to_str(ODPSEngine(self.odps).compile(self.expr.birth.weekday, prettify=False)))

        self.assertRaises(NotImplementedError,
                          lambda: ODPSEngine(self.odps).compile(self.expr.birth.dayofyear))

        self.assertRaises(NotImplementedError,
                          lambda: ODPSEngine(self.odps).compile(self.expr.birth.is_month_start))

        self.assertRaises(NotImplementedError,
                          lambda: ODPSEngine(self.odps).compile(self.expr.birth.is_month_end))

        self.assertRaises(NotImplementedError,
                          lambda: ODPSEngine(self.odps).compile(self.expr.birth.is_year_start))

        self.assertRaises(NotImplementedError,
                          lambda: ODPSEngine(self.odps).compile(self.expr.birth.is_year_end))

        data = [datetime.now()]

        engine = ODPSEngine(self.odps)
        engine.compile(self.expr.birth.strftime('%Y'))
        self._testify_udf([d.strftime('%Y') for d in data], [(d,) for d in data], engine)

    def testSortCompilation(self):
        expr = self.expr.sort(['name', -self.expr.id])[:50]

        expected = 'SELECT * \nFROM mocked_project.`pyodps_test_expr_table` t1 \nORDER BY name, id DESC \nLIMIT 50'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.sort(['name', 'id'], ascending=[True, False])[:50]
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr['id', 'name'].sort(['name'])['name', 'id']
        expected = 'SELECT t2.`name`, t2.`id` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, t1.`name` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  ORDER BY name \n' \
                   '  LIMIT 10000\n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.sort(self.expr.id + 1)[:50]
        expected = 'SELECT * \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'ORDER BY t1.`id` + 1 \n' \
                   'LIMIT 50'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testDistinctCompilation(self):
        expr = self.expr.distinct(['name', self.expr.id + 1])

        expected = 'SELECT DISTINCT t1.`name`, t1.`id` + 1 AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr['name', 'id'].distinct()

        expected = 'SELECT DISTINCT t1.`name`, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr['name'].unique()

        expected = 'SELECT t2.`name` \n' \
                   'FROM (\n' \
                   '  SELECT DISTINCT t1.`name` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testGroupByCompilation(self):
        expr = self.expr.groupby(['name', 'id'])[lambda x: x.id.min() * 2 < 10] \
            .agg(self.expr.fid.max() + 1, new_id=self.expr.id.sum())

        expected = 'SELECT t1.`name`, t1.`id`, MAX(t1.`fid`) + 1 AS `fid_max`, SUM(t1.`id`) AS `new_id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`, t1.`id` \n' \
                   'HAVING (MIN(t1.`id`) * 2) < 10'

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby('name').agg(id=self.expr.id.max())[
            lambda x: x.name.astype('float'), 'id']['id', 'name']

        expected = 'SELECT MAX(t1.`id`) AS `id`, CAST(t1.`name` AS DOUBLE) AS `name` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby('name').agg(id=self.expr.id.max()).sort('id')['id', 'name']
        expected = 'SELECT t2.`id`, t2.`name` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, MAX(t1.`id`) AS `id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  GROUP BY t1.`name` \n' \
                   '  ORDER BY id \n' \
                   '  LIMIT 10000\n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby(['name']).mutate(lambda x: x.row_number(sort='id'))

        expected = 'SELECT t1.`name`, ' \
                   'ROW_NUMBER() OVER (PARTITION BY t1.`name` ORDER BY id) AS `row_number` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.name.value_counts()[:25]
        expr2 = self.expr.name.topk(25)

        expected = 'SELECT t1.`name`, COUNT(t1.`name`) AS `count` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name` \n' \
                   'ORDER BY count DESC \n' \
                   'LIMIT 25'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr2, prettify=False)))

        expr = self.expr.groupby('name').count()
        expected = 'SELECT COUNT(1) AS `count` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby(Scalar(1)).id.max()
        expected = 'SELECT MAX(t1.`id`) AS `id_max` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY 1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby(self.expr['name'].rename('name2')).agg(id=self.expr['id'].sum())
        expr = expr[expr.name2, expr.id + 3]

        expected = 'SELECT t1.`name` AS `name2`, SUM(t1.`id`) + 3 AS `id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testMutate(self):
        expr = self.expr[self.expr.exclude('birth'), self.expr.fid.astype('int').rename('new_id')]
        expr = expr[expr, expr.groupby('name').mutate(lambda x: x.new_id.cumsum().rename('new_id_sum'))]
        expr = expr[expr.exclude('new_id_sum'), expr.new_id_sum + 1]

        expected = 'SELECT t2.`name`, t2.`id`, t2.`fid`, t2.`isMale`, t2.`scale`, ' \
                   't2.`new_id`, t2.`new_id_sum` + 1 AS `new_id_sum` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale`, ' \
                   'CAST(t1.`fid` AS BIGINT) AS `new_id`, ' \
                   'SUM(CAST(t1.`fid` AS BIGINT)) OVER (PARTITION BY t1.`name`) AS `new_id_sum` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testFilterGroupbySinkFilterCompilation(self):
        # to test the sinking of filter to `having` clause
        expr = self.expr.groupby(['name']).agg(id=self.expr.id.max())[lambda x: x.id < 10]
        expected = 'SELECT t1.`name`, MAX(t1.`id`) AS `id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name` \n' \
                   'HAVING MAX(t1.`id`) < 10'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testWindowRewrite(self):
        expr = self.expr[self.expr.id - self.expr.id.mean() < 10][
            [lambda x: x.id - x.id.max()]][[lambda x: x.id - x.id.min()]][lambda x: x.id - x.id.std(ddof=0) > 0]

        expected = "SELECT t8.`id` \n" \
                   "FROM (\n" \
                   "  SELECT t7.`id`, STDDEV(t7.`id`) OVER (PARTITION BY 1) AS `id_std_0` \n" \
                   "  FROM (\n" \
                   "    SELECT t6.`id` - t6.`id_min_1` AS `id` \n" \
                   "    FROM (\n" \
                   "      SELECT t5.`id`, MIN(t5.`id`) OVER (PARTITION BY 1) AS `id_min_1` \n" \
                   "      FROM (\n" \
                   "        SELECT t4.`id` - t4.`id_max_2` AS `id` \n" \
                   "        FROM (\n" \
                   "          SELECT t3.`id`, MAX(t3.`id`) OVER (PARTITION BY 1) AS `id_max_2` \n" \
                   "          FROM (\n" \
                   "            SELECT t2.`name`, t2.`id`, t2.`fid`, t2.`isMale`, t2.`scale`, t2.`birth` \n" \
                   "            FROM (\n" \
                   "              SELECT t1.`birth`, t1.`fid`, t1.`id`, " \
                   "AVG(t1.`id`) OVER (PARTITION BY 1) AS `id_mean_3`, t1.`isMale`, t1.`name`, t1.`scale` \n" \
                   "              FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "            ) t2 \n" \
                   "            WHERE (t2.`id` - t2.`id_mean_3`) < 10 \n" \
                   "          ) t3 \n" \
                   "        ) t4 \n" \
                   "      ) t5 \n" \
                   "    ) t6 \n" \
                   "  ) t7 \n" \
                   ") t8 \n" \
                   "WHERE (t8.`id` - t8.`id_std_0`) > 0"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testWindowRewriteInSelectCompilation(self):
        # to test rewriting the window function in select clause
        expr = self.expr.id - self.expr.id.max()
        expected = 'SELECT t2.`id` - t2.`id_max_0` AS `id` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, MAX(t1.`id`) OVER (PARTITION BY 1) AS `id_max_0` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testWindowRewriteInFilterCompilation(self):
        # to test rewriting the window function in filter clause
        expr = self.expr[self.expr.id - self.expr.id.mean() < 10]
        expected = 'SELECT t2.`name`, t2.`id`, t2.`fid`, t2.`isMale`, t2.`scale`, t2.`birth` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`birth`, t1.`fid`, t1.`id`, AVG(t1.`id`) OVER (PARTITION BY 1) AS `id_mean_0`, ' \
                   't1.`isMale`, t1.`name`, t1.`scale` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   ') t2 \n' \
                   'WHERE (t2.`id` - t2.`id_mean_0`) < 10'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testReductionCompilation(self):
        # TODO test all the reductions
        expr = self.expr.groupby(['id']).id.std(ddof=0) + 1

        expected = 'SELECT STDDEV(t1.`id`) + 1 AS `id_std` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`id`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.fid.mean()
        expected = 'SELECT AVG(t1.`fid`) AS `fid_mean` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.count()
        expected = 'SELECT COUNT(1) AS `count` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = (self.expr.id == 1).any().rename('equal1')
        expected = 'SELECT MAX(IF(t1.`id` == 1, 1, 0)) == 1 AS `equal1` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertTrue(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.isMale.all()
        expected = 'SELECT MIN(IF(t1.`isMale`, 1, 0)) == 1 AS `isMale_all` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertTrue(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.id.nunique()
        expected = 'SELECT COUNT(DISTINCT t1.`id`) AS `id_nunique` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby(['id']).agg(name=self.expr.name.sum()).count()
        expected = 'SELECT COUNT(1) AS `count` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, WM_CONCAT(\'\', t1.`name`) AS `name` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  GROUP BY t1.`id` \n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby(['id']).mean()
        expected = 'SELECT t1.`id`, AVG(t1.`fid`) AS `fid_mean`, ' \
                   'AVG(t1.`id`) AS `id_mean`, AVG(t1.`scale`) AS `scale_mean` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`id`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby(['name']).any()
        expected = 'SELECT t1.`name`, MAX(IF(t1.`isMale`, 1, 0)) == 1 AS `isMale_any` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby(['name']).name.nunique()
        expected = 'SELECT COUNT(DISTINCT t1.`name`) AS `name_nunique` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        is_decimal = False

        class Aggregator(object):
            def buffer(self):
                return [0, ]

            def __call__(self, buffer, val):
                buffer[0] += val + 1

            def merge(self, buffer, pbuffer):
                buffer[0] += pbuffer[0]

            def getvalue(self, buffer):
                if is_decimal:
                    return Decimal(buffer[0])
                return buffer[0]

        data = [1, 2, 3]

        expr = self.expr.id.agg(Aggregator)
        engine = ODPSEngine(self.odps)
        engine.compile(expr)
        self._testify_udf([9, ], [[r, ] for r in data], engine)

        data = [Decimal('2.2'), Decimal('1.1'), Decimal('3.3')]

        is_decimal = True
        expr = self.expr.scale.agg(Aggregator)
        engine = ODPSEngine(self.odps)
        engine.compile(expr)
        self._testify_udf(['9.6', ], [[str(r), ] for r in data], engine)

    def testProjectionCompact(self):
        expr = self.expr[self.expr.id.rename('new_id'), self.expr.name, self.expr.name.rename('new_name2')]
        expr = expr[expr.new_id.rename('new_id2'), expr.name.rename('new_name'), expr.new_name2]

        expected = 'SELECT t1.`id` AS `new_id2`, t1.`name` AS `new_name`, t1.`name` AS `new_name2` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testGroupbyCompact(self):
        expr = self.expr['name', self.expr.id + 1].groupby('name').count()

        expected = 'SELECT COUNT(1) AS `count` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, t1.`id` + 1 AS `id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   ') t2 \n' \
                   'GROUP BY t2.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testWindowCompilation(self):
        # TODO test all window functions
        expr = self.expr.groupby('name').id.cumcount(unique=True)

        expected = 'SELECT COUNT(DISTINCT t1.`id`) OVER (PARTITION BY t1.`name`) AS `id_count` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr.groupby('name').isMale.cumsum()
        expected = 'SELECT SUM(IF(t1.`isMale`, 1, 0)) OVER (PARTITION BY t1.`name`) AS `isMale_sum` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[self.expr.name, self.expr.id + 1][
            'name', lambda x: x.groupby('name').sort('id', ascending=False).row_number().rename('rank')]
        expected = "SELECT t1.`name`, ROW_NUMBER() OVER (PARTITION BY t1.`name` ORDER BY t1.`id` + 1 DESC) AS `rank` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expr = self.expr[self.expr.name, self.expr.groupby(Scalar(1)).id.cumcount()]
        expected = 'SELECT t1.`name`, COUNT(t1.`id`) OVER (PARTITION BY 1) AS `id_count` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(expected, ODPSEngine(self.odps).compile(expr, prettify=False))

    def testApply(self):
        expr = self.expr.groupby('name').sort(['name', 'id']).apply(
            lambda x: x, names=self.expr.schema.names)['id', 'name']

        expected = 'SELECT \n' \
                   '  t2.`id`,\n' \
                   '  t2.`name` \n' \
                   'FROM (\n' \
                   '  SELECT {0}(t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, ' \
                   'CAST(t1.`scale` AS STRING), t1.`birth`) AS (name, id, fid, isMale, scale, birth) \n' \
                   '  FROM (\n' \
                   '    SELECT *  \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    DISTRIBUTE BY \n' \
                   '      name \n' \
                   '    SORT BY \n' \
                   '      name,\n' \
                   '      id\n' \
                   '  ) t1 \n' \
                   ') t2'
        engine = ODPSEngine(self.odps)
        res = engine.compile(expr)
        fun_name = list(engine._ctx._registered_funcs.values())[0]
        self.assertEqual(to_str(expected.format(fun_name)), to_str(res))

        from odps.config import options
        options.df.quote = False

        engine = ODPSEngine(self.odps)
        res = engine.compile(expr)
        fun_name = list(engine._ctx._registered_funcs.values())[0]
        self.assertEqual(to_str(expected.replace('`', '').format(fun_name)), to_str(res))

        options.df.quote = True

    def testJoin(self):
        e = self.expr
        e1 = self.expr1
        e2 = self.expr2
        joined = e.join(e1, ['fid'])
        expected = 'SELECT t1.`name` AS `name_x`, t1.`id` AS `id_x`, t1.`fid`, t1.`isMale` AS `isMale_x`, ' \
                   't1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, t2.`name` AS `name_y` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2\nON t1.`fid` == t2.`fid`'

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined[e, e1.name], prettify=False)))

        expected = 'SELECT t1.`name` AS `name_x`, t1.`id`, t1.`fid`, t1.`isMale` AS `isMale_x`, ' \
                   't1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, t2.`name` AS `name_y`,' \
                   ' t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, t2.`birth` AS `birth_y` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2\n' \
                   'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)'

        joined = e.join(e1, ['fid', 'id'])

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

        joined = e.join(e1, ['fid', 'id'])

        joined2 = e.join(e2, ['name'])
        joined3 = joined.join(joined2, joined.name_x == joined2.name)

        expected = 'SELECT t4.`name`, t4.`id_x`, t4.`fid_x`, ' \
                   't4.`isMale_x` AS `isMale_x_y`, t4.`scale_x` AS `scale_x_y`, t4.`birth_x` AS `birth_x_y`, ' \
                   't4.`id_y`, t4.`fid_y`, ' \
                   't4.`isMale_y` AS `isMale_y_y`, t4.`scale_y` AS `scale_y_y`, ' \
                   't4.`birth_y` AS `birth_y_y`, t1.`name` AS `name_x` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2\n' \
                   'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`) \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t1.`name`, t1.`id` AS `id_x`, t1.`fid` AS `fid_x`, t1.`isMale` AS `isMale_x`, ' \
                   't1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, t3.`id` AS `id_y`, ' \
                   't3.`fid` AS `fid_y`, t3.`isMale` AS `isMale_y`, t3.`scale` AS `scale_y`, t3.`birth` AS `birth_y` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table2` t3\n' \
                   '    ON t1.`name` == t3.`name`\n' \
                   '  ) t4\n' \
                   'ON t1.`name` == t4.`name`'

        self.assertEqual(to_str(expected),
                         to_str(ODPSEngine(self.odps).compile(joined3[joined2, joined.name_x], prettify=False)))
        # test twice to check the cache
        self.assertEqual(to_str(expected),
                         to_str(ODPSEngine(self.odps).compile(joined3[joined2, joined.name_x], prettify=False)))

        expected = 'SELECT t2.`name` AS `new_name`, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \nINNER JOIN \n  ' \
                   'mocked_project.`pyodps_test_expr_table1` t2\n' \
                   'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)'

        self.assertEqual(to_str(expected),
                         to_str(ODPSEngine(self.odps).compile(joined[lambda x, y: y.name.rename('new_name'), e.id],
                                                    prettify=False)))

        joined = e.join(e1, ['fid', 'id'])

        joined = joined[e1.name, e.id]
        expected = 'SELECT t1.`name`, t1.`id` AS `id_x`, t1.`fid`, t1.`isMale`, t1.`scale`, ' \
                   't1.`birth`, t4.`name_y`, t4.`id` AS `id_y` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table2` t1 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t3.`name` AS `name_y`, t2.`id` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t2 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table1` t3\n' \
                   '    ON (t2.`fid` == t3.`fid`) AND (t2.`id` == t3.`id`)\n' \
                   '  ) t4\n' \
                   'ON t4.`name_y` == t1.`name`'

        self.assertEqual(to_str(expected),
                         to_str(ODPSEngine(self.odps).compile(e2.join(joined, joined.name_y == e2.name), prettify=False)))

        expected = 'SELECT t1.`name` AS `name_x`, t1.`id` AS `id_x`, t1.`fid`, ' \
                   't1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, ' \
                   't2.`name` AS `name_y_x`, ' \
                   't2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, t2.`birth` AS `birth_y`, ' \
                   't3.`name_y` AS `name_y_y`, t3.`id` AS `id_y` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2\n' \
                   'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`) \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t2.`name` AS `name_y`, t1.`id` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table1` t2\n' \
                   '    ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)\n' \
                   '  ) t3\n' \
                   'ON t3.`name_y` == t1.`name`'

        joined = e.join(e1, ['fid', 'id']).join(joined, lambda x, y: y.name_y == x.name_x)
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

        joined = e.join(e1, ['fid', 'id'])[e.name, e1]
        joined2 = e.join(e2, ['name'])[e.name, e.fid, e2.id]
        joined3 = joined.join(joined2, joined.name_x == joined2.name)[joined2.name, ]
        expected = 'SELECT t5.`name` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name` AS `name_x`, t2.`name` AS `name_y`, t1.`id`, ' \
                   't1.`fid`, t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, t2.`birth` AS `birth_y` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  INNER JOIN \n' \
                   '    mocked_project.`pyodps_test_expr_table1` t2\n' \
                   '  ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)\n' \
                   ') t3 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t1.`name`, t1.`fid` AS `fid_x`, t4.`id` AS `id_y` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table2` t4\n' \
                   '    ON t1.`name` == t4.`name`\n' \
                   '  ) t5\n' \
                   'ON t3.`name_x` == t5.`name`'

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined3, prettify=False)))

        joined = e.distinct('fid', 'id')[:10].join(e1, ['fid', 'id'])
        expected = 'SELECT t2.`fid`, t2.`id`, t3.`name`, ' \
                   't3.`isMale`, t3.`scale`, t3.`birth` \n' \
                   'FROM (\n' \
                   '  SELECT DISTINCT t1.`fid`, t1.`id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  LIMIT 10\n' \
                   ') t2 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t3\n' \
                   'ON (t2.`fid` == t3.`fid`) AND (t2.`id` == t3.`id`)'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

        e1 = e1.select(name2=e1.name, id2=e1.id)['id2', 'name2']
        joined = e.join(e1, on=('name', 'tt' + e1.name2))

        expected = "SELECT * \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "INNER JOIN \n" \
                   "  (\n" \
                   "    SELECT t2.`id` AS `id2`, t2.`name` AS `name2` \n" \
                   "    FROM mocked_project.`pyodps_test_expr_table1` t2\n" \
                   "  ) t3\n" \
                   "ON t1.`name` == (CONCAT('tt', t3.`name2`))"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

    def testJoinMultipleFilter(self):
        e = self.expr
        e1 = self.expr1[self.expr1.name.rename('name2'), self.expr1.id.rename('id2')]
        expr = e.join(e1, on=('id', 'id2'))
        expr = expr[expr.id.notnull()][lambda x: x.name2.isnull()]

        expected = 'SELECT * \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t2.`name` AS `name2`, t2.`id` AS `id2` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
                   '  ) t3\n' \
                   'ON t1.`id` == t3.`id2` \n' \
                   'WHERE (t3.`name2` IS NULL) AND (t1.`id` IS NOT NULL)'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

        expected = 'SELECT * \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT \n' \
                   '      t2.`name` AS `name2`,\n' \
                   '      t2.`id`   AS `id2` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
                   '  ) t3\n' \
                   'ON t1.`id` == t3.`id2` \n' \
                   'WHERE (t3.`name2` IS NULL) AND (t1.`id` IS NOT NULL)'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr)))

    def testSelfJoin(self):
        joined = self.expr.join(self.expr, 'name')
        expected = 'SELECT t1.`name`, t1.`id` AS `id_x`, t1.`fid` AS `fid_x`, t1.`isMale` AS `isMale_x`, ' \
                   't1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, t2.`id` AS `id_y`, ' \
                   't2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, t2.`birth` AS `birth_y` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table` t2\n' \
                   'ON t1.`name` == t2.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

        proj = self.expr['name', 'id']
        joined = proj.join(proj, 'name').join(proj, 'name')
        expected = 'SELECT t2.`name`, t2.`id` AS `id_x`, t3.`id` AS `id_y`, t4.`id` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, t1.`id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
                   ') t2 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t1.`name`, t1.`id` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1\n' \
                   '  ) t3\n' \
                   'ON t2.`name` == t3.`name` \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t1.`name`, t1.`id` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1\n' \
                   '  ) t4\n' \
                   'ON t2.`name` == t4.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

    def testLeftJoin(self):
        left = self.expr.select(self.expr, type='normal')
        right = self.expr[:4]
        joined = left.left_join(right, on='id')
        res = joined.id_x.rename('id')

        expected = "SELECT t2.`id` \n" \
                   "FROM (\n" \
                   "  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale`, " \
                   "t1.`birth`, 'normal' AS `type` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1\n" \
                   ") t2 \n" \
                   "LEFT OUTER JOIN \n" \
                   "  (\n" \
                   "    SELECT * \n" \
                   "    FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "    LIMIT 4\n" \
                   "  ) t3\n" \
                   "ON t2.`id` == t3.`id`"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(res, prettify=False)))

        left = self.expr.exclude('birth')
        right = self.expr[:4]
        joined = left.left_join(right, on='id')
        res = joined[left, ]

        expected = 'SELECT t2.`name` AS `name_x`, t2.`id` AS `id_x`, t2.`fid` AS `fid_x`, ' \
                   't2.`isMale` AS `isMale_x`, t2.`scale` AS `scale_x` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
                   ') t2 \n' \
                   'LEFT OUTER JOIN \n' \
                   '  (\n' \
                   '    SELECT * \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    LIMIT 4\n' \
                   '  ) t3\n' \
                   'ON t2.`id` == t3.`id`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(res, prettify=False)))

    def testMapJoin(self):
        joined = self.expr.join(self.expr1, on=[], mapjoin=True)
        expected = \
            'SELECT /*+mapjoin(t2)*/ t1.`name` AS `name_x`, t1.`id` AS `id_x`, t1.`fid` AS `fid_x`, ' \
            't1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, t2.`name` AS `name_y`, ' \
            't2.`id` AS `id_y`, t2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, ' \
            't2.`birth` AS `birth_y` \n' \
            'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
            'INNER JOIN \n' \
            '  mocked_project.`pyodps_test_expr_table1` t2'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

        joined = self.expr.join(self.expr1, on=[lambda x,y:x.name>y.name], mapjoin=True).select(self.expr1.name)
        expected = 'SELECT /*+mapjoin(t2)*/ t2.`name` AS `name_y` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2\n' \
                   'ON t1.`name` > t2.`name`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

        joined = self.expr.join(self.expr1, on=[], mapjoin=True).join(self.expr2, on=[], mapjoin=True).select(self.expr.name)
        expected = 'SELECT /*+mapjoin(t2,t3)*/ t1.`name` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table2` t3'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))
        joined = self.expr.join(self.expr1.join(self.expr2, on=[], mapjoin=True).select(self.expr2.name), mapjoin=True, on=[]).select(self.expr.name)

        expected = 'SELECT /*+mapjoin(t4)*/ t1.`name` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT /*+mapjoin(t3)*/ t3.`name` AS `name_y` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t2 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table2` t3\n' \
                   '  ) t4'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

        joined = self.expr['name', 'id'].join(self.expr1.limit(4), on=[], mapjoin=True)

        expected = 'SELECT /*+mapjoin(t4)*/ t2.`name` AS `name_x`, t2.`id` AS `id_x`, t4.`name` AS `name_y`, ' \
                   't4.`id` AS `id_y`, t4.`fid`, t4.`isMale`, t4.`scale`, t4.`birth` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, t1.`id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
                   ') t2 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT * \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t3 \n' \
                   '    LIMIT 4\n' \
                   '  ) t4'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(joined, prettify=False)))

        expr = joined[joined.id_x == 1]
        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT /*+mapjoin(t4)*/ t2.`name` AS `name_x`, t2.`id` AS `id_x`, ' \
                   't4.`name` AS `name_y`, t4.`id` AS `id_y`, t4.`fid`, t4.`isMale`, ' \
                   't4.`scale`, t4.`birth` \n' \
                   '  FROM (\n' \
                   '    SELECT t1.`name`, t1.`id` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1\n' \
                   '  ) t2 \n' \
                   '  INNER JOIN \n' \
                   '    (\n' \
                   '      SELECT * \n' \
                   '      FROM mocked_project.`pyodps_test_expr_table1` t3 \n' \
                   '      LIMIT 4\n' \
                   '    ) t4 \n' \
                   ') t5 \n' \
                   'WHERE t5.`id_x` == 1'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(expr, prettify=False)))

    def testComplexMapJoin(self):
        def distance(x1, y1, x2, y2):
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        input = self.expr[self.expr.id.rename('uid'), self.expr.fid.rename('ux'), self.expr.fid.rename('uy')]
        center = self.expr[self.expr.id.rename('cid'), self.expr.fid.rename('x'), self.expr.fid.rename('y')][:4]
        for i in range(2):
            join_tmp = input.join(center, mapjoin=True, on=[])
            join_tmp2 = join_tmp.select(join_tmp.uid, join_tmp.ux, join_tmp.uy,
                                        distance(join_tmp.ux, join_tmp.uy, join_tmp.x, join_tmp.y).rename('distance'),
                                        join_tmp.cid)
            join_tmp3 = join_tmp2['uid', 'ux', 'uy', 'cid',
                                  join_tmp2.groupby('uid').sort(join_tmp2.distance, ascending=True).row_number()
            ].filter(lambda x: x.row_number == 1)
            join_tmp4 = join_tmp3.groupby('cid').agg(join_tmp3.ux.mean().rename('x'),
                                                     join_tmp3.uy.mean().rename('y'),
                                                     cnt=join_tmp3.count())
            center = join_tmp4[join_tmp4.cid, 'x', 'y']

        expected = 'SELECT t6.`cid`, AVG(t6.`ux`) AS `x`, AVG(t6.`uy`) AS `y` \n' \
                   'FROM (\n' \
                   '  SELECT /*+mapjoin(t5)*/ t2.`uid`, t2.`ux`, t2.`uy`, t5.`cid`, ' \
                   'ROW_NUMBER() OVER (PARTITION BY t2.`uid` ' \
                   'ORDER BY (POW(t2.`ux` - t5.`x`, 2)) + (POW(t2.`uy` - t5.`y`, 2))) AS `row_number` \n' \
                   '  FROM (\n' \
                   '    SELECT t1.`id` AS `uid`, t1.`fid` AS `ux`, t1.`fid` AS `uy` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1\n' \
                   '  ) t2 \n' \
                   '  INNER JOIN \n' \
                   '    (\n' \
                   '      SELECT t4.`cid`, AVG(t4.`ux`) AS `x`, AVG(t4.`uy`) AS `y` \n' \
                   '      FROM (\n' \
                   '        SELECT /*+mapjoin(t3)*/ t2.`uid`, t2.`ux`, t2.`uy`, t3.`cid`, ' \
                   'ROW_NUMBER() OVER (PARTITION BY t2.`uid` ' \
                   'ORDER BY (POW(t2.`ux` - t3.`x`, 2)) + (POW(t2.`uy` - t3.`y`, 2))) AS `row_number` \n' \
                   '        FROM (\n' \
                   '          SELECT t1.`id` AS `uid`, t1.`fid` AS `ux`, t1.`fid` AS `uy` \n' \
                   '          FROM mocked_project.`pyodps_test_expr_table` t1\n' \
                   '        ) t2 \n' \
                   '        INNER JOIN \n' \
                   '          (\n' \
                   '            SELECT t1.`id` AS `cid`, t1.`fid` AS `x`, t1.`fid` AS `y` \n' \
                   '            FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '            LIMIT 4\n' \
                   '          ) t3 \n' \
                   '      ) t4 \n' \
                   '      WHERE t4.`row_number` == 1 \n' \
                   '      GROUP BY t4.`cid`\n' \
                   '    ) t5 \n' \
                   ') t6 \n' \
                   'WHERE t6.`row_number` == 1 \n' \
                   'GROUP BY t6.`cid`'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(center, prettify=False)))

    def testJoinMapReduce(self):
        expr = self.expr.left_join(self.expr3.select(new_id=self.expr3['id']), on=('id', 'new_id'))

        @output(expr.schema.names, expr.schema.types)
        def reducer(keys):
            def h(row):
                yield row
            return h

        expr=expr.map_reduce(reducer=reducer, group='name')

        expected = "SELECT t5.`name`, t5.`id`, t5.`fid`, t5.`isMale`, " \
                   "CAST(t5.`scale` AS DECIMAL) AS `scale`, t5.`birth`, t5.`new_id` \n" \
                   "FROM (\n" \
                   "  SELECT {0}(" \
                   "t4.`name`, t4.`id`, t4.`fid`, t4.`isMale`, CAST(t4.`scale` AS STRING), " \
                   "t4.`birth`, t4.`new_id`) AS (name, id, fid, isMale, scale, birth, new_id) \n" \
                   "  FROM (\n" \
                   "    SELECT *  \n" \
                   "    FROM (\n" \
                   "      SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, " \
                   "t1.`scale`, t1.`birth`, t3.`new_id` \n" \
                   "      FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "      LEFT OUTER JOIN \n" \
                   "        (\n" \
                   "          SELECT t2.`id` AS `new_id` \n" \
                   "          FROM mocked_project.`pyodps_test_expr_table2` t2\n" \
                   "        ) t3\n" \
                   "      ON t1.`id` == t3.`new_id` \n" \
                   "    ) t4 \n" \
                   "    DISTRIBUTE BY t4.`name` \n" \
                   "    SORT BY name\n" \
                   "  ) t4 \n" \
                   ") t5"
        engine = ODPSEngine(self.odps)
        res = engine.compile(expr, prettify=False)
        fun_name = list(engine._ctx._registered_funcs.values())[0]
        self.assertEqual(to_str(expected.format(fun_name)), to_str(res))

    def testAsType(self):
        e = self.expr
        new_e = e.id.astype('float')
        expected = 'SELECT CAST(t1.`id` AS DOUBLE) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(new_e, prettify=False)))

        self.assertRaises(CompileError, lambda: ODPSEngine(self.odps).compile(self.expr.id.astype('boolean')))

    def testUnion(self):
        e = self.expr
        e1 = self.expr1

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name` AS `name_x` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  UNION ALL\n' \
                   '    SELECT t1.`name` AS `name_x` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table1` t2\n' \
                   '    ON t1.`name` == t2.`name`\n' \
                   ') t3'
        self.assertEqual(to_str(expected),
                         to_str(ODPSEngine(self.odps).compile(
                             e.name.rename('name_x').union(e.join(e1, 'name')[e['name'].rename('name_x'), ]), False)))

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT * \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n'\
                   '  UNION ALL\n'\
                   '    SELECT * \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
                   ') t3'

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(e.union(e1), False)))

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  UNION ALL\n' \
                   '    SELECT t2.`name` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
                   ') t5'
        try:
            self.assertEqual(expected, ODPSEngine(self.odps).compile(e.name.union(e1.name), False))
        except:
            self.assertEqual(expected[:-1], ODPSEngine(self.odps).compile(e.name.union(e1.name), False)[:-1])

        e = self.expr['name', 'id']
        e1 = self.expr1['name', 'id']

        expected = 'SELECT * \nFROM (\n' \
                   '  SELECT t1.`name`, t1.`id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n'\
                   '  UNION ALL\n'\
                   '    SELECT t2.`name`, t2.`id` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
                   ') t3'
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(e.union(e1), False)))

        e1 = self.expr1['id', 'name']
        e2 = self.expr2['name', 'id']

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, t1.`name` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table1` t1 \n' \
                   '  UNION ALL\n' \
                   '    SELECT t2.`id`, t2.`name` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table2` t2\n' \
                   ') t3'

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(e1.union(e2), False)))

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, t1.`name` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table1` t1 \n' \
                   '  UNION ALL\n' \
                   '    SELECT t2.`id`, t2.`name` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table2` t2\n' \
                   ') t3'

        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(e1.concat(e2), False)))

    def testAliases(self):
        df = self.expr
        df = df[(df.id == 1) | (df.id == 2)].exclude(['fid'])
        df = df.groupby(df['id']).agg(df.name.count().rename('count')).sort('id', ascending=False)
        df = df[df, Scalar('1').rename('part')]

        expected = "SELECT t2.`id`, t2.`count`, '1' AS `part` \n" \
                   "FROM (\n" \
                   "  SELECT t1.`id`, COUNT(t1.`name`) AS `count` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "  WHERE (t1.`id` == 1) OR (t1.`id` == 2) \n" \
                   "  GROUP BY t1.`id` \n" \
                   "  ORDER BY id DESC \n" \
                   "  LIMIT 10000\n" \
                   ") t2"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(df, prettify=False)))

    def testFilterPartition(self):
        df = self.expr3
        df = df.filter_partition('part1=a/part2=1,part1=b')['id', 'name']

        expected = "SELECT t2.`id`, t2.`name` \n" \
                   "FROM (\n" \
                   "  SELECT t1.`name`, t1.`id`, t1.`fid` \n" \
                   "  FROM mocked_project.`pyodps_test_expr_table2` t1 \n" \
                   "  WHERE ((t1.`part1` == 'a') AND (CAST(t1.`part2` AS BIGINT) == 1)) OR (t1.`part1` == 'b') \n" \
                   ") t2"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(df, prettify=False)))

        df = self.expr3
        df = df.filter_partition('part1=a/part2=1,part1=b', False)['id', 'name']

        expected = "SELECT t1.`id`, t1.`name` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table2` t1 \n" \
                   "WHERE ((t1.`part1` == 'a') AND (CAST(t1.`part2` AS BIGINT) == 1)) OR (t1.`part1` == 'b')"
        self.assertEqual(to_str(expected), to_str(ODPSEngine(self.odps).compile(df, prettify=False)))

if __name__ == '__main__':
    unittest.main()
