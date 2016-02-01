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
from datetime import datetime
import base64  # noqa

import six

from odps.tests.core import TestBase, to_str
from odps.compat import unittest
from odps.udf.tools import runners
from odps.models import Schema
from odps.utils import to_timestamp
from odps.df.types import validate_data_type
from odps.df.expr.expressions import CollectionExpr
from odps.df.backends.odpssql.engine import ODPSEngine, UDF_CLASS_NAME
from odps.df.backends.odpssql.compiler import BINARY_OP_COMPILE_DIC, \
    MATH_COMPILE_DIC, DATE_PARTS_DIC
from odps.df.backends.errors import CompileError
from odps.df.expr.tests.core import MockTable
from odps.df.backends.odpssql.cloudpickle import *  # noqa
from odps.df import Scalar, switch


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

        self.engine = ODPSEngine(self.odps)
        self.maxDiff = None

    def _clear_functions(self):
        self.engine._ctx._registered_funcs.clear()
        self.engine._ctx._func_to_udfs.clear()

    def _testify_udf(self, expected, inputs):
        if six.PY3:
            # As Python UDF only supports 2.7, so we just skip the test
            return

        udf = list(self.engine._ctx._func_to_udfs.values())[0]
        exec (udf, globals(), locals())
        udf = locals()[UDF_CLASS_NAME]
        self.assertSequenceEqual(expected, runners.simple_run(udf, inputs))

        self._clear_functions()

    def testBaseCompilation(self):
        expr = self.expr[self.expr.id < 10]['name', lambda x: x.id]
        expected = 'SELECT t1.`name`, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'WHERE t1.`id` < 10'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr[Scalar(3).rename('const'), self.expr.id]
        expected = 'SELECT 3 AS const, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr[self.expr, (self.expr.id + 1).rename('id2')]
        expected = 'SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, ' \
                   't1.`scale`, t1.`birth`, t1.`id` + 1 AS id2 \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

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
                   "WHEN (70 < t1.`id`) AND (t1.`id` <= 80) THEN '70-79' END AS id_group \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "LIMIT 10"
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.groupby('name').agg(id_count=self.expr.id.count(), id_mean=self.expr.id.mean())
        expr = expr[expr.id_count >= 100].sort(['id_mean', 'id_count'], ascending=False)

        expected = 'SELECT t1.`name`, COUNT(t1.`id`) AS id_count, AVG(t1.`id`) AS id_mean \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name` \n' \
                   'HAVING COUNT(t1.`id`) >= 100 \n' \
                   'ORDER BY id_mean DESC, id_count DESC \n' \
                   'LIMIT 10000'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr[self.expr.id < 100].groupby('name').agg(id=lambda x: x.id.sum()).sort('id')[:1000]['id']
        expected = 'SELECT t2.`id` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, SUM(t1.`id`) AS id \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  WHERE t1.`id` < 100 \n' \
                   '  GROUP BY t1.`name` \n' \
                   '  ORDER BY id \n' \
                   '  LIMIT 1000\n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr[[self.expr.name.extract('/projects/(.*?)/', group=1).rename('project')]]\
            .groupby('project').count()
        expected = "SELECT REGEXP_EXTRACT(t1.`name`, '/projects/(.*?)/', 1) AS project, COUNT(1) AS count \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
                   "GROUP BY REGEXP_EXTRACT(t1.`name`, '/projects/(.*?)/', 1)"
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.describe()
        expected = 'SELECT SUM(IF(t1.`id` IS NOT NULL, 1, 0)) AS id_count, ' \
                   'MIN(t1.`id`) AS id_min, MAX(t1.`id`) AS id_max, ' \
                   'AVG(t1.`id`) AS id_mean, STDDEV(t1.`id`) AS id_std, ' \
                   'SUM(IF(t1.`fid` IS NOT NULL, 1, 0)) AS fid_count, ' \
                   'MIN(t1.`fid`) AS fid_min, MAX(t1.`fid`) AS fid_max, ' \
                   'AVG(t1.`fid`) AS fid_mean, STDDEV(t1.`fid`) AS fid_std, ' \
                   'SUM(IF(t1.`scale` IS NOT NULL, 1, 0)) AS scale_count, ' \
                   'MIN(t1.`scale`) AS scale_min, MAX(t1.`scale`) AS scale_max, ' \
                   'AVG(t1.`scale`) AS scale_mean, STDDEV(t1.`scale`) AS scale_std \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

    def testElementCompilation(self):
        expect = 'SELECT t1.`id` IS NULL AS id \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.id.isnull(), prettify=False)))

        expect = 'SELECT t1.`id` IS NOT NULL AS id \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.id.notnull(), prettify=False)))

        expect = 'SELECT IF(t1.`id` IS NULL, 100, t1.`id`) AS id \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.id.fillna(100), prettify=False)))

        expr = self.expr.id.isin([1, 2, 3]).rename('id')
        expect = 'SELECT t1.`id` IN (1, 2, 3) AS id \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.id.isin(self.expr.fid.astype('int')).rename('id')
        expect = 'SELECT t1.`id` IN ' \
                 '(SELECT CAST(t1.`fid` AS BIGINT) AS fid FROM mocked_project.`pyodps_test_expr_table` t1) ' \
                 'AS id \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.id.notin([1, 2, 3]).rename('id')
        expect = 'SELECT t1.`id` NOT IN (1, 2, 3) AS id \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.id.notin(self.expr.fid.astype('int')).rename('id')
        expect = 'SELECT t1.`id` NOT IN ' \
                 '(SELECT CAST(t1.`fid` AS BIGINT) AS fid FROM mocked_project.`pyodps_test_expr_table` t1) ' \
                 'AS id \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

        expect = 'SELECT (t1.`fid` <= t1.`id`) AND (t1.`id` <= 3) AS id \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.id.between(self.expr.fid, 3), prettify=False)))

        expect = 'SELECT IF(t1.`id` < 5, \'test\', CONCAT(t1.`name`, \'abc\')) AS id \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         self.engine.compile(
                             (self.expr.id < 5).ifelse('test', self.expr.name + 'abc').rename('id'),
                             prettify=False))

        expr = self.expr.name.switch('test', 'test' + self.expr.name,
                                     'test2', 'test2' + self.expr.name).rename('name')
        expect = 'SELECT CASE t1.`name` WHEN \'test\' THEN CONCAT(t1.`name`, \'test\') ' \
                 'WHEN \'test2\' THEN CONCAT(t1.`name`, \'test2\') END AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

        expr = switch(self.expr.name == 'test', 'test2', default='notest').rename('name')
        expect = "SELECT CASE WHEN t1.`name` == 'test' THEN 'test2' ELSE 'notest' END AS name \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.id.cut([100, 200, 300], labels=['small', 'large']).rename('tp')
        expect = 'SELECT CASE WHEN (100 < t1.`id`) AND (t1.`id` <= 200) THEN \'small\' ' \
                 'WHEN (200 < t1.`id`) AND (t1.`id` <= 300) THEN \'large\' END AS tp \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.id.cut([100, 200, 300],
                                labels=['xsmall', 'small', 'large', 'xlarge'],
                                include_under=True, include_over=True).rename('tp')
        expect = "SELECT CASE WHEN t1.`id` <= 100 THEN 'xsmall' " \
                 "WHEN (100 < t1.`id`) AND (t1.`id` <= 200) THEN 'small' " \
                 "WHEN (200 < t1.`id`) AND (t1.`id` <= 300) THEN 'large' " \
                 "WHEN 300 < t1.`id` THEN 'xlarge' END AS tp \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.id.cut([100, 200, 300],
                                labels=['xsmall', 'small', 'large', 'xlarge'],
                                include_lowest=True,
                                include_under=True, include_over=True).rename('tp')
        expect = "SELECT CASE WHEN t1.`id` < 100 THEN 'xsmall' " \
                 "WHEN (100 <= t1.`id`) AND (t1.`id` <= 200) THEN 'small' " \
                 "WHEN (200 < t1.`id`) AND (t1.`id` <= 300) THEN 'large' " \
                 "WHEN 300 < t1.`id` THEN 'xlarge' END AS tp \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

    def testArithmeticCompilation(self):
        for e in (self.expr.id + 5, 5 + self.expr.id, self.expr.id - 5, self.expr.id * 5,
                  self.expr.id / 5, self.expr.id > 5, 5 < self.expr.id,
                  self.expr.id >= 5, 5 <= self.expr.id, self.expr.id < 5,
                  self.expr.id <= 5, self.expr.id == 5, self.expr.id != 5):
            expect = 'SELECT t1.`id` {0} 5 AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'.format(
                BINARY_OP_COMPILE_DIC[e.__class__.__name__])

            self.assertEqual(to_str(expect), to_str(self.engine.compile(e, prettify=False)))

        for e in (5 - self.expr.id, 5 * self.expr.id, 5 / self.expr.id):
            expect = 'SELECT 5 {0} t1.`id` AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'.format(
                BINARY_OP_COMPILE_DIC[e.__class__.__name__])

            self.assertEqual(to_str(expect), to_str(self.engine.compile(e, prettify=False)))

        for e in (self.expr.isMale & True, self.expr.isMale | True):
            expect = 'SELECT t1.`isMale` {0} true AS isMale \nFROM mocked_project.`pyodps_test_expr_table` t1'.format(
                BINARY_OP_COMPILE_DIC[e.__class__.__name__].upper())

            self.assertEqual(to_str(expect), to_str(self.engine.compile(e, prettify=False)))

        self.assertEqual(to_str('SELECT -t1.`id` AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'),
                         to_str(self.engine.compile((-self.expr.id), prettify=False)))

        now = datetime.now()
        unix_time = to_timestamp(now)
        expr = self.expr.birth < now
        expect = 'SELECT t1.`birth` < FROM_UNIXTIME(%s) AS birth \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1' % unix_time
        self.assertEqual(to_str(expect), to_str(self.engine.compile(expr, prettify=False)))

    def testMathCompilation(self):
        for math_cls, func in MATH_COMPILE_DIC.items():
            e = getattr(self.expr.id, math_cls.lower())()

            expect = 'SELECT {0}(t1.`id`) AS id \n' \
                     'FROM mocked_project.`pyodps_test_expr_table` t1'.format(func.upper())

            self.assertEqual(to_str(expect), to_str(self.engine.compile(e, prettify=False)))

        expect = 'SELECT LN(t1.`id`) AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.id.log(), prettify=False)))

        expect = 'SELECT LOG(2, t1.`id`) AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.id.log2(), prettify=False)))

        expect = 'SELECT LOG(10, t1.`id`) AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.id.log10(), prettify=False)))

        expect = 'SELECT LN(1 + t1.`id`) AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.id.log1p(), prettify=False)))

        expect = 'SELECT EXP(t1.`id`) - 1 AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.id.expm1(), prettify=False)))

        expect = 'SELECT TRUNC(t1.`fid`) AS fid \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.fid.trunc(), prettify=False)))

        expect = 'SELECT TRUNC(t1.`fid`, 2) AS fid \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.fid.trunc(2), prettify=False)))

        data = [30]
        try:
            import numpy as np
        except ImportError:
            return

        self.engine.compile(self.expr.fid.arccosh())
        self._testify_udf([np.arccosh(d) for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.fid.arcsinh())
        self._testify_udf([np.arcsinh(d) for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.fid.degrees())
        self._testify_udf([np.degrees(d) for d in data], [(d,) for d in data])

        data = [0.2]

        self.engine.compile(self.expr.fid.arctanh())
        self._testify_udf([np.arctanh(d) for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.fid.radians())
        self._testify_udf([np.radians(d) for d in data], [(d,) for d in data])

    def testStringCompilation(self):
        expect = 'SELECT CONCAT(TOUPPER(SUBSTR(t1.`name`, 1, 1)), TOLOWER(SUBSTR(t1.`name`, 2))) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.capitalize(), prettify=False)))

        expect = 'SELECT REGEXP_INSTR(t1.`name`, \'test\') > 0 AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.contains('test'), prettify=False)))
        expect = 'SELECT INSTR(t1.`name`, \'test\') > 0 AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.contains('test', regex=False), prettify=False)))
        self.engine.compile(self.expr.name.contains('test', flags=re.I))
        self._testify_udf([True, False, True], [('Test',), ('tes',), ('ToTEst',)])

        expect = 'SELECT REGEXP_COUNT(t1.`name`, \'test\') AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.count('test'), prettify=False)))

        expect = 'SELECT INSTR(REVERSE(t1.`name`), REVERSE(\'test\')) == 1 AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.endswith('test'), prettify=False)))

        expect = 'SELECT INSTR(t1.`name`, \'test\') == 1 AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.startswith('test'), prettify=False)))

        expect = 'SELECT REGEXP_EXTRACT(t1.`name`, \'test\', 0) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.extract('test'), prettify=False)))
        self.engine.compile(self.expr.name.extract('test\d', flags=re.I))
        self._testify_udf(['Test1', None, 'test3'], [('Test1',), ('tes',), ('test32',)])

        expect = 'SELECT INSTR(t1.`name`, \'test\', 1) - 1 AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.find('test'), prettify=False)))
        self.engine.compile(self.expr.name.rfind('test'))
        data = ['abcdggtest1222', 'notes', 'test']
        self._testify_udf([d.rfind('test') for d in data], [(d,) for d in data])

        expect = 'SELECT REGEXP_REPLACE(t1.`name`, \'test\', \'test2\', 0) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.replace('test', 'test2'), prettify=False)))

        expect = 'SELECT SUBSTR(t1.`name`, 3, 1) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.get(2), prettify=False)))

        expect = 'SELECT LENGTH(t1.`name`) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.len(), prettify=False)))

        expect = "SELECT CONCAT(t1.`name`, IF((10 - LENGTH(t1.`name`)) >= 0, " \
                 "REPEAT(' ', 10 - LENGTH(t1.`name`)), '')) AS name \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.ljust(10), prettify=False)))
        expect = "SELECT CONCAT(t1.`name`, IF((10 - LENGTH(t1.`name`)) >= 0, " \
                 "REPEAT('*', 10 - LENGTH(t1.`name`)), '')) AS name \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.ljust(10, fillchar='*'), prettify=False)))

        expect = "SELECT CONCAT(IF((10 - LENGTH(t1.`name`)) >= 0, " \
                 "REPEAT(' ', 10 - LENGTH(t1.`name`)), ''), t1.`name`) AS name \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.rjust(10), prettify=False)))

        expect = 'SELECT TOLOWER(t1.`name`) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.lower(), prettify=False)))

        expect = 'SELECT TOUPPER(t1.`name`) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.upper(), prettify=False)))

        self.engine.compile(self.expr.name.lstrip())
        data = [' abc\n', ' \nddd']
        self._testify_udf([d.lstrip() for d in data], [(d,) for d in data])
        expect = 'SELECT LTRIM(t1.`name`) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.lstrip(to_strip=' '), prettify=False)))

        expect = 'SELECT RTRIM(t1.`name`) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.rstrip(to_strip=' '), prettify=False)))

        expect = 'SELECT TRIM(t1.`name`) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.strip(to_strip=' '), prettify=False)))

        self.engine.compile(self.expr.name.pad(10))
        data = ['ab', 'a' * 12]
        self._testify_udf([d.rjust(10) for d in data], [(d,) for d in data])

        expect = 'SELECT REPEAT(t1.`name`, 4) AS name \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.repeat(4), prettify=False)))

        self.engine.compile(self.expr.name.slice(0, 10, 2))
        data = ['ab' * 15, 'def']
        self._testify_udf([d[0: 10: 2] for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.slice(-5, -1))
        data = ['aa' * 15, 'd']
        self._testify_udf([d[-5: -1] for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.title())
        data = ['Abc Def', 'ADEFddEE']
        self._testify_udf([d.title() for d in data], [(d,) for d in data])

        expect = "SELECT CONCAT(IF((10 - LENGTH(t1.`name`)) >= 0, " \
                 "REPEAT('0', 10 - LENGTH(t1.`name`)), ''), t1.`name`) AS name \n" \
                 "FROM mocked_project.`pyodps_test_expr_table` t1"
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.name.zfill(10), prettify=False)))

        data = ['123', 'dEf', '124df']

        self.engine.compile(self.expr.name.isalnum())
        self._testify_udf([d.isalnum() for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.isalpha())
        self._testify_udf([d.isalpha() for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.isdigit())
        self._testify_udf([d.isdigit() for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.isspace())
        self._testify_udf([d.isspace() for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.islower())
        self._testify_udf([d.islower() for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.isupper())
        self._testify_udf([d.isupper() for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.istitle())
        self._testify_udf([d.istitle() for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.isnumeric())
        self._testify_udf([to_str(d).isnumeric() for d in data], [(d,) for d in data])

        self.engine.compile(self.expr.name.isdecimal())
        self._testify_udf([to_str(d).isdecimal() for d in data], [(d,) for d in data])

    def testDatetimeCompilation(self):
        self.assertRaises(NotImplementedError,
                          lambda: self.engine.compile(self.expr.birth.date))

        self.assertRaises(NotImplementedError,
                          lambda: self.engine.compile(self.expr.birth.time))

        for clz, date_part in DATE_PARTS_DIC.items():
            expect = 'SELECT DATEPART(t1.`birth`, \'%s\') AS birth \n' \
                     'FROM mocked_project.`pyodps_test_expr_table` t1' % date_part
            attr = getattr(self.expr.birth, clz.lower())
            self.assertEqual(to_str(expect),
                             to_str(self.engine.compile(attr, prettify=False)))

        self.assertRaises(NotImplementedError,
                          lambda: self.engine.compile(self.expr.birth.microsecond))

        self.assertRaises(NotImplementedError,
                          lambda: self.engine.compile(self.expr.birth.week))

        expect = 'SELECT WEEKOFYEAR(t1.`birth`) AS birth \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.birth.weekofyear, prettify=False)))

        expect = 'SELECT WEEKDAY(t1.`birth`) AS birth \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.birth.dayofweek, prettify=False)))
        self.assertEqual(to_str(expect),
                         to_str(self.engine.compile(self.expr.birth.weekday, prettify=False)))

        self.assertRaises(NotImplementedError,
                          lambda: self.engine.compile(self.expr.birth.dayofyear))

        self.assertRaises(NotImplementedError,
                          lambda: self.engine.compile(self.expr.birth.is_month_start))

        self.assertRaises(NotImplementedError,
                          lambda: self.engine.compile(self.expr.birth.is_month_end))

        self.assertRaises(NotImplementedError,
                          lambda: self.engine.compile(self.expr.birth.is_year_start))

        self.assertRaises(NotImplementedError,
                          lambda: self.engine.compile(self.expr.birth.is_year_end))

        data = [datetime.now()]

        self.engine.compile(self.expr.birth.strftime('%Y'))
        self._testify_udf([d.strftime('%Y') for d in data], [(d,) for d in data])

    def testSortCompilation(self):
        expr = self.expr.sort(['name', -self.expr.id])[:50]

        expected = 'SELECT * \nFROM mocked_project.`pyodps_test_expr_table` t1 \nORDER BY name, id DESC \nLIMIT 50'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.sort(['name', 'id'], ascending=[True, False])[:50]
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr['id', 'name'].sort(['name'])['name', 'id']
        expected = 'SELECT t2.`name`, t2.`id` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, t1.`name` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  ORDER BY name \n' \
                   '  LIMIT 10000\n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

    def testDistinctCompilation(self):
        expr = self.expr.distinct(['name', self.expr.id + 1])

        expected = 'SELECT DISTINCT t1.`name`, t1.`id` + 1 AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr['name', 'id'].distinct()

        expected = 'SELECT DISTINCT t1.`name`, t1.`id` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

    def testGroupByCompilation(self):
        expr = self.expr.groupby(['name', 'id'])[lambda x: x.id.min() * 2 < 10] \
            .agg(self.expr.fid.max() + 1, new_id=self.expr.id.sum())

        expected = 'SELECT t1.`name`, t1.`id`, MAX(t1.`fid`) + 1 AS fid_max, SUM(t1.`id`) AS new_id \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`, t1.`id` \n' \
                   'HAVING (MIN(t1.`id`) * 2) < 10'

        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.groupby('name').agg(id=self.expr.id.max())[
            lambda x: x.name.astype('float'), 'id']['id', 'name']

        expected = 'SELECT MAX(t1.`id`) AS id, CAST(t1.`name` AS DOUBLE) AS name \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.groupby('name').agg(id=self.expr.id.max()).sort('id')['id', 'name']
        expected = 'SELECT t2.`id`, t2.`name` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name`, MAX(t1.`id`) AS id \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  GROUP BY t1.`name` \n' \
                   '  ORDER BY id \n' \
                   '  LIMIT 10000\n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.groupby(['name']).mutate(lambda x: x.row_number())

        expected = 'SELECT t1.`name`, ROW_NUMBER() OVER (PARTITION BY t1.`name`) AS row_number \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.name.value_counts()[:25]
        expr2 = self.expr.name.topk(25)

        expected = 'SELECT t1.`name`, COUNT(t1.`name`) AS count \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name` \n' \
                   'ORDER BY count DESC \n' \
                   'LIMIT 25'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr2, prettify=False)))

        expr = self.expr.groupby('name').count()
        expected = 'SELECT t1.`name`, COUNT(1) AS count \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.groupby(Scalar(1)).id.max()
        expected = 'SELECT MAX(t1.`id`) AS id_max \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY 1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

    def testFilterGroupbySinkFilterCompilation(self):
        # to test the sinking of filter to `having` clause
        expr = self.expr.groupby(['name']).agg(id=self.expr.id.max())[lambda x: x.id < 10]
        expected = 'SELECT t1.`name`, MAX(t1.`id`) AS id \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name` \n' \
                   'HAVING MAX(t1.`id`) < 10'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

    def testWindowRewriteInSelectCompilation(self):
        # to test rewriting the window function in select clause
        expr = self.expr.id - self.expr.id.max()
        expected = 'SELECT t2.`id` - t2.`id_max_0` AS id \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, MAX(t1.`id`) OVER (PARTITION BY 1) AS id_max_0 \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

    def testWindowRewriteInFilterCompilation(self):
        # to test rewriting the window function in filter clause
        expr = self.expr[self.expr.id - self.expr.id.mean() < 10]
        expected = 'SELECT t2.`name`, t2.`id`, t2.`fid`, t2.`isMale`, t2.`scale`, t2.`birth` \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, AVG(t1.`id`) OVER (PARTITION BY 1) AS id_mean_0, t1.`name`, ' \
                   't1.`fid`, t1.`isMale`, t1.`scale`, t1.`birth` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   ') t2 \n' \
                   'WHERE (t2.`id` - t2.`id_mean_0`) < 10'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))
        # test twice to check the cache of optimization
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

    def testReductionCompilation(self):
        # TODO test all the reductions
        expr = self.expr.groupby(['id']).id.std(ddof=0) + 1

        expected = 'SELECT STDDEV(t1.`id`) + 1 AS id_std \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`id`'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.fid.mean()
        expected = 'SELECT AVG(t1.`fid`) AS fid_mean \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.count()
        expected = 'SELECT COUNT(1) AS count \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = (self.expr.id == 1).any().rename('equal1')
        expected = 'SELECT MAX(IF(t1.`id` == 1, 1, 0)) == 1 AS equal1 \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertTrue(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.isMale.all()
        expected = 'SELECT MIN(IF(t1.`isMale`, 1, 0)) == 1 AS isMale_all \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertTrue(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.groupby(['id']).agg(name=self.expr.name.sum()).count()
        expected = 'SELECT COUNT(1) AS count \n' \
                   'FROM (\n' \
                   '  SELECT t1.`id`, WM_CONCAT(\'\', t1.`name`) AS name \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  GROUP BY t1.`id` \n' \
                   ') t2'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.groupby(['id']).mean()
        expected = 'SELECT t1.`id`, AVG(t1.`fid`) AS fid_mean, ' \
                   'AVG(t1.`id`) AS id_mean, AVG(t1.`scale`) AS scale_mean \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`id`'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.groupby(['name']).any()
        expected = 'SELECT t1.`name`, MAX(IF(t1.`isMale`, 1, 0)) == 1 AS isMale_any \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'GROUP BY t1.`name`'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

    def testWindowCompilation(self):
        # TODO test all window functions
        expr = self.expr.groupby('name').id.cumcount(unique=True)

        expected = 'SELECT COUNT(DISTINCT t1.`id`) OVER (PARTITION BY t1.`name`) AS id_count \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

        expr = self.expr.groupby('name').isMale.cumsum()
        expected = 'SELECT SUM(IF(t1.`isMale`, 1, 0)) OVER (PARTITION BY t1.`name`) AS isMale_sum \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(expr, prettify=False)))

    def testJoin(self):
        e = self.expr
        e1 = self.expr1
        e2 = self.expr2
        joined = e.join(e1, ['fid'])
        expected = 'SELECT t1.`name` AS name_x, t1.`id` AS id_x, t1.`fid` AS fid_x, t1.`isMale` AS isMale_x, ' \
                   't1.`scale` AS scale_x, t1.`birth` AS birth_x, t2.`name` AS name_y \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2\nON t1.`fid` == t2.`fid`'

        self.assertEqual(to_str(expected), to_str(self.engine.compile(joined[e, e1.name], prettify=False)))

        expected = 'SELECT t1.`name` AS name_x, t1.`id` AS id_x, t1.`fid` AS fid_x, t1.`isMale` AS isMale_x, ' \
                   't1.`scale` AS scale_x, t1.`birth` AS birth_x, t2.`name` AS name_y, t2.`id` AS id_y,' \
                   ' t2.`fid` AS fid_y, t2.`isMale` AS isMale_y, t2.`scale` AS scale_y, t2.`birth` AS birth_y \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2\n' \
                   'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)'

        joined = e.join(e1, ['fid', 'id'])

        self.assertEqual(to_str(expected), to_str(self.engine.compile(joined, prettify=False)))

        joined = e.join(e1, ['fid', 'id'])

        joined2 = e.join(e2, ['name'])
        joined3 = joined.join(joined2, joined.name_x == joined2.name_x)

        expected = 'SELECT t4.`name_x` AS name_x_y, t4.`id_x` AS id_x_y, t4.`fid_x` AS fid_x_y, ' \
                   't4.`isMale_x` AS isMale_x_y, t4.`scale_x` AS scale_x_y, t4.`birth_x` AS birth_x_y, ' \
                   't4.`name_y` AS name_y_y, t4.`id_y` AS id_y_y, t4.`fid_y` AS fid_y_y, ' \
                   't4.`isMale_y` AS isMale_y_y, t4.`scale_y` AS scale_y_y, ' \
                   't4.`birth_y` AS birth_y_y, t1.`name` AS name_x_x \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2\n' \
                   'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`) \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t1.`name` AS name_x, t1.`id` AS id_x, t1.`fid` AS fid_x, t1.`isMale` AS isMale_x, ' \
                   't1.`scale` AS scale_x, t1.`birth` AS birth_x, t3.`name` AS name_y, t3.`id` AS id_y, ' \
                   't3.`fid` AS fid_y, t3.`isMale` AS isMale_y, t3.`scale` AS scale_y, t3.`birth` AS birth_y \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table2` t3\n' \
                   '    ON t1.`name` == t3.`name`\n' \
                   '  ) t4\n' \
                   'ON t1.`name` == t4.`name_x`'

        self.assertEqual(to_str(expected),
                         to_str(self.engine.compile(joined3[joined2, joined.name_x], prettify=False)))
        # test twice to check the cache
        self.assertEqual(to_str(expected),
                         to_str(self.engine.compile(joined3[joined2, joined.name_x], prettify=False)))

        expected = 'SELECT t2.`name` AS new_name, t1.`id` AS id_x \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \nINNER JOIN \n  ' \
                   'mocked_project.`pyodps_test_expr_table1` t2\n' \
                   'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)'

        self.assertEqual(to_str(expected),
                         to_str(self.engine.compile(joined[e1.name.rename('new_name'), e.id], prettify=False)))

        joined = e.join(e1, ['fid', 'id'])

        joined = joined[e1.name, e.id]
        expected = 'SELECT * \n' \
                   'FROM mocked_project.`pyodps_test_expr_table2` t3 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t2.`name` AS name_y, t1.`id` AS id_x \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table1` t2\n' \
                   '    ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)\n' \
                   '  ) t5\n' \
                   'ON t5.`name_y` == t3.`name`'

        self.assertEqual(to_str(expected),
                         to_str(self.engine.compile(e2.join(joined, joined.name_y == e2.name), prettify=False)))

        expected = 'SELECT t1.`name` AS name_x, t1.`id` AS id_x_x, t1.`fid` AS fid_x, ' \
                   't1.`isMale` AS isMale_x, t1.`scale` AS scale_x, t1.`birth` AS birth_x, ' \
                   't2.`name` AS name_y_x, t2.`id` AS id_y, t2.`fid` AS fid_y, ' \
                   't2.`isMale` AS isMale_y, t2.`scale` AS scale_y, t2.`birth` AS birth_y, ' \
                   't5.`name_y` AS name_y_y, t5.`id_x` AS id_x_y \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table1` t2\n' \
                   'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`) \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t2.`name` AS name_y, t1.`id` AS id_x \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table1` t2\n' \
                   '    ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)\n' \
                   '  ) t5\n' \
                   'ON t5.`name_y` == t1.`name`'

        joined = e.join(e1, ['fid', 'id']).join(joined, lambda x, y: y.name_y == x.name_x)
        self.assertEqual(to_str(expected), to_str(self.engine.compile(joined, prettify=False)))

        joined = e.join(e1, ['fid', 'id'])[e.name, e.id, e1]
        joined2 = e.join(e2, ['name'])[e.name, e.fid, e2.name, e2.id]
        joined3 = joined.join(joined2, joined.name_x == joined2.name_x)[joined2.name_x]
        expected = 'SELECT t7.`name_x` AS name_x_y \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name` AS name_x, t1.`id` AS id_x, t2.`name` AS name_y, t2.`id` AS id_y, ' \
                   't2.`fid` AS fid_y, t2.`isMale` AS isMale_y, t2.`scale` AS scale_y, t2.`birth` AS birth_y \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  INNER JOIN \n' \
                   '    mocked_project.`pyodps_test_expr_table1` t2\n' \
                   '  ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)\n' \
                   ') t6 \n' \
                   'INNER JOIN \n' \
                   '  (\n' \
                   '    SELECT t1.`name` AS name_x, t1.`fid` AS fid_x, t3.`name` AS name_y, t3.`id` AS id_y \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table2` t3\n' \
                   '    ON t1.`name` == t3.`name`\n' \
                   '  ) t7\n' \
                   'ON t6.`name_x` == t7.`name_x`'

        self.assertEqual(to_str(expected), to_str(self.engine.compile(joined3, prettify=False)))

    def testSelfJoin(self):
        joined = self.expr.join(self.expr, 'name')
        expected = 'SELECT t1.`name` AS name_x, t1.`id` AS id_x, t1.`fid` AS fid_x, t1.`isMale` AS isMale_x, ' \
                   't1.`scale` AS scale_x, t1.`birth` AS birth_x, t2.`name` AS name_y, t2.`id` AS id_y, ' \
                   't2.`fid` AS fid_y, t2.`isMale` AS isMale_y, t2.`scale` AS scale_y, t2.`birth` AS birth_y \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   'INNER JOIN \n' \
                   '  mocked_project.`pyodps_test_expr_table` t2\n' \
                   'ON t1.`name` == t2.`name`'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(joined, prettify=False)))

    def testAsType(self):
        e = self.expr
        new_e = e.id.astype('float')
        expected = 'SELECT CAST(t1.`id` AS DOUBLE) AS id \nFROM mocked_project.`pyodps_test_expr_table` t1'

        self.assertEqual(to_str(expected), to_str(self.engine.compile(new_e, prettify=False)))

        self.assertRaises(CompileError, lambda: self.engine.compile(self.expr.id.astype('boolean')))

    def testUnion(self):
        e = self.expr
        e1 = self.expr1

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name` AS name_x \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  UNION ALL\n' \
                   '    SELECT t1.`name` AS name_x \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '    INNER JOIN \n' \
                   '      mocked_project.`pyodps_test_expr_table1` t2\n' \
                   '    ON t1.`name` == t2.`name`\n' \
                   ') t3'
        self.assertEqual(to_str(expected),
                         to_str(self.engine.compile(e.name.rename('name_x').union(e.join(e1, 'name')['name_x']), False)))

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT * \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n'\
                   '  UNION ALL\n'\
                   '    SELECT * \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
                   ') t4'

        self.assertEqual(to_str(expected), to_str(self.engine.compile(e.union(e1), False)))

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t1.`name` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
                   '  UNION ALL\n' \
                   '    SELECT t2.`name` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
                   ') t5'
        try:
            self.assertEqual(expected, self.engine.compile(e.name.union(e1.name), False))
        except:
            self.assertEqual(expected[:-1], self.engine.compile(e.name.union(e1.name), False)[:-1])

        e = self.expr['name', 'id']
        e1 = self.expr1['name', 'id']

        expected = 'SELECT * \nFROM (\n' \
                   '  SELECT t1.`name`, t1.`id` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table` t1 \n'\
                   '  UNION ALL\n'\
                   '    SELECT t2.`name`, t2.`id` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
                   ') t6'
        self.assertEqual(to_str(expected), to_str(self.engine.compile(e.union(e1), False)))

        e1 = self.expr1['id', 'name']
        e2 = self.expr2['name', 'id']

        expected = 'SELECT * \n' \
                   'FROM (\n' \
                   '  SELECT t2.`id`, t2.`name` \n' \
                   '  FROM mocked_project.`pyodps_test_expr_table1` t2 \n' \
                   '  UNION ALL\n' \
                   '    SELECT t7.`id`, t7.`name` \n' \
                   '    FROM mocked_project.`pyodps_test_expr_table2` t7\n' \
                   ') t6'

        try:
            self.assertEqual(to_str(expected), to_str(self.engine.compile(e1.union(e2), False)))
        except AssertionError:
            self.assertEqual(to_str(expected)[:-1], to_str(self.engine.compile(e1.union(e2), False)[:-1]))

        try:
            self.assertEqual(to_str(expected), to_str(self.engine.compile(e1.concat(e2), False)))
        except AssertionError:
            self.assertEqual(to_str(expected)[:-1], to_str(self.engine.compile(e1.concat(e2), False)[:-1]))

if __name__ == '__main__':
    unittest.main()
