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

# for performance
import cProfile
import re
import sys
from collections import namedtuple
from decimal import Decimal
from datetime import datetime, timedelta
from pstats import Stats

import pytest

from ..... import options
from .....compat import six
from .....udf.tools import runners
from .....models import TableSchema
from .....utils import to_timestamp, to_milliseconds, to_text
from .... import Scalar, NullScalar, switch, year, month, day, hour, minute, second, millisecond
from .... import output, func, make_list, make_dict
from ....expr.expressions import CollectionExpr, BuiltinFunction, RandomScalar, ExpressionError
from ....expr.tests.core import MockTable
from ....types import validate_data_type
from ...errors import CompileError
from ..engine import ODPSSQLEngine, UDF_CLASS_NAME
from ..compiler import BINARY_OP_COMPILE_DIC, MATH_COMPILE_DIC, DATE_PARTS_DIC


ENABLE_PROFILE = False


class ODPSEngine(ODPSSQLEngine):

    def compile(self, expr, prettify=True, libraries=None):
        expr = self._convert_table(expr)
        expr_dag = expr.to_dag()
        self._analyze(expr_dag, expr)
        new_expr = self._rewrite(expr_dag)
        sql = self._compile(new_expr, prettify=prettify, libraries=libraries)
        if isinstance(sql, list):
            return '\n'.join(sql)
        return sql


@pytest.fixture
def exprs():
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
        datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'),
    )
    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
    expr = CollectionExpr(_source_data=table, _schema=schema)

    table1 = MockTable(name='pyodps_test_expr_table1', table_schema=schema)
    expr1 = CollectionExpr(_source_data=table1, _schema=schema)

    table2 = MockTable(name='pyodps_test_expr_table2', table_schema=schema)
    expr2 = CollectionExpr(_source_data=table2, _schema=schema)

    schema2 = TableSchema.from_lists(
        ['name', 'id', 'fid'], datatypes('string', 'int64', 'float64'),
        ['part1', 'part2'], datatypes('string', 'int64'),
    )
    table3 = MockTable(name='pyodps_test_expr_table2', table_schema=schema2)
    expr3 = CollectionExpr(_source_data=table3, _schema=schema2)

    schema3 = TableSchema.from_lists(
        ['id', 'name', 'relatives', 'hobbies'],
        datatypes('int64', 'string', 'dict<string, string>', 'list<string>'),
    )
    table4 = MockTable(name='pyodps_test_expr_table', table_schema=schema3)
    expr4 = CollectionExpr(_source_data=table4, _schema=schema3)

    # turn off the column pruning and predicate pushdown
    # for the purpose not to modify the case
    # we only need to ensure the correctness of engine execution
    options.df.optimizes.cp = False
    options.df.optimizes.pp = False

    pr = None
    if ENABLE_PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    try:
        nt = namedtuple("NT", "expr expr1 expr2 expr3 expr4 pr")
        yield nt(expr, expr1, expr2, expr3, expr4, pr)
    finally:
        options.df.optimizes.cp = True
        options.df.optimizes.pp = True

        if ENABLE_PROFILE:
            p = Stats(pr)
            p.strip_dirs()
            p.sort_stats('time')
            p.print_stats(40)


def _clear_functions(engine):
    engine._ctx._registered_funcs.clear()
    engine._ctx._func_to_udfs.clear()


def _testify_udf(expected, inputs, engine, annotation=None):
    from .....udf import runtime as udf_runtime

    udf = list(engine._ctx._func_to_udfs.values())[0]
    d = dict()
    six.exec_(udf, d, d)
    udf = d[UDF_CLASS_NAME]
    assert list(expected) == list(runners.simple_run(udf, inputs))

    if annotation:
        assert udf_runtime._annotated_classes[udf] == annotation

    _clear_functions(engine)


def test_base_compilation(odps, exprs):
    expr = exprs.expr[exprs.expr.id < 10]['name', lambda x: x.id]
    expected = 'SELECT t1.`name`, t1.`id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'WHERE t1.`id` < 10'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr[Scalar(3).rename('const'),
                     NullScalar('string').rename('string_const'),
                     exprs.expr.id]
    expected = 'SELECT 3 AS `const`, CAST(NULL AS STRING) AS `string_const`, t1.`id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr[exprs.expr, (exprs.expr.id + 1).rename('id2')]
    expected = 'SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, ' \
               't1.`scale`, t1.`birth`, t1.`id` + 1 AS `id2` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr[lambda x: x.exclude('name'), (exprs.expr.id + 1).rename('id2')]
    expected = 'SELECT t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale`, t1.`birth`, t1.`id` + 1 AS `id2` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    expr = exprs.expr[exprs.expr, exprs.expr.id.cut(list(range(0, 81, 10)), labels=labels).rename('id_group')]
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
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(lambda x: x.name).agg(id_count=exprs.expr.id.count(),
                                                   id_mean=lambda x: x.id.mean())
    expr = expr[expr.id_count >= 100].sort(['id_mean', 'id_count'], ascending=False)

    expected = 'SELECT t1.`name`, COUNT(t1.`id`) AS `id_count`, AVG(t1.`id`) AS `id_mean` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name` \n' \
               'HAVING COUNT(t1.`id`) >= 100 \n' \
               'ORDER BY id_mean DESC, id_count DESC \n' \
               'LIMIT 10000'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr[exprs.expr.id < 100].groupby('name').agg(id=lambda x: x.id.sum()).sort('id')[:1000]['id']
    expected = 'SELECT t2.`id` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, SUM(t1.`id`) AS `id` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  WHERE t1.`id` < 100 \n' \
               '  GROUP BY t1.`name` \n' \
               '  ORDER BY id \n' \
               '  LIMIT 1000\n' \
               ') t2'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr[[exprs.expr.name.map('REGEXP_EXTRACT', args=('/projects/(.*?)/', 1)).rename('project')]]\
        .groupby('project').count()
    expected = "SELECT COUNT(1) AS `count` \n" \
               "FROM (\n" \
               "  SELECT REGEXP_EXTRACT(t1.`name`, '/projects/(.*?)/', 1) AS `project` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               ") t2 \n" \
               "GROUP BY t2.`project`"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(
            exprs.expr.name.map('REGEXP_EXTRACT', args=('/projects/(.*?)/', 1)).rename('project')).count()
    expected = "SELECT COUNT(1) AS `count` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "GROUP BY REGEXP_EXTRACT(t1.`name`, '/projects/(.*?)/', 1)"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr[exprs.expr, Scalar(3).rename('const')][
        lambda x: x.exclude('isMale'), lambda x: (x.const + 1).rename('const2')]
    expected = 'SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`scale`, t1.`birth`, ' \
               '3 AS `const`, 3 + 1 AS `const2` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.select(pt=BuiltinFunction('max_pt', args=(exprs.expr._source_data.name, )))
    expected = "SELECT max_pt('pyodps_test_expr_table') AS `pt` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr[exprs.expr, RandomScalar(1).rename('random')]
    expected = "SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale`, t1.`birth`, rand(1) AS `random` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr['name', exprs.expr.id + 1][lambda x: x.id < 1]
    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, t1.`id` + 1 AS `id` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               ') t2 \n' \
               'WHERE t2.`id` < 1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').agg(count=exprs.expr.id.count()).limit(10)[lambda x: x['count'] > 10]
    expected = "SELECT * \n" \
               "FROM (\n" \
               "  SELECT t1.`name`, COUNT(t1.`id`) AS `count` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "  GROUP BY t1.`name` \n" \
               "  LIMIT 10\n" \
               ") t2 \n" \
               "WHERE t2.`count` > 10"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    engine = ODPSEngine(odps)

    expr = exprs.expr.dropna(thresh=2)
    expected = "SELECT * \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "WHERE (((((IF(t1.`name` IS NOT NULL, 1, 0) + IF(t1.`id` IS NOT NULL, 1, 0)) " \
               "+ IF((t1.`fid` IS NOT NULL) AND NOT {0}(t1.`fid`), 1, 0)) + IF(t1.`isMale` IS NOT NULL, 1, 0)) " \
               "+ IF(t1.`scale` IS NOT NULL, 1, 0)) + IF(t1.`birth` IS NOT NULL, 1, 0)) >= 2"
    res = engine.compile(expr, prettify=False)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == to_text(res)

    engine = ODPSEngine(odps)

    expr = exprs.expr.dropna(how='all')
    expected = "SELECT * \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "WHERE (((((IF(t1.`name` IS NOT NULL, 1, 0) + IF(t1.`id` IS NOT NULL, 1, 0)) " \
               "+ IF((t1.`fid` IS NOT NULL) AND NOT {0}(t1.`fid`), 1, 0)) + IF(t1.`isMale` IS NOT NULL, 1, 0)) " \
               "+ IF(t1.`scale` IS NOT NULL, 1, 0)) + IF(t1.`birth` IS NOT NULL, 1, 0)) >= 1"
    res = engine.compile(expr, prettify=False)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == to_text(res)

    engine = ODPSEngine(odps)

    expr = exprs.expr.fillna(0, subset=['id', 'fid'])
    expected = "SELECT t1.`name`, IF(t1.`id` IS NULL, 0, t1.`id`) AS `id`, " \
               "IF((t1.`fid` IS NULL) OR {0}(t1.`fid`), 0, t1.`fid`) AS `fid`, " \
               "t1.`isMale`, t1.`scale`, t1.`birth` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1"
    res = engine.compile(expr, prettify=False)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == to_text(res)

    try:
        options.df.odps.nan_handler = None
        expr = exprs.expr.fillna(0, subset=['id', 'fid'])
        expected = "SELECT t1.`name`, IF(t1.`id` IS NULL, 0, t1.`id`) AS `id`, " \
                   "IF(t1.`fid` IS NULL, 0, t1.`fid`) AS `fid`, " \
                   "t1.`isMale`, t1.`scale`, t1.`birth` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1"
        assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

        options.df.odps.nan_handler = 'bulitin'
        expr = exprs.expr.fillna(0, subset=['id', 'fid'])
        expected = "SELECT t1.`name`, IF(t1.`id` IS NULL, 0, t1.`id`) AS `id`, " \
                   "IF((t1.`fid` IS NULL) OR ISNAN(t1.`fid`), 0, t1.`fid`) AS `fid`, " \
                   "t1.`isMale`, t1.`scale`, t1.`birth` \n" \
                   "FROM mocked_project.`pyodps_test_expr_table` t1"
        assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))
    finally:
        options.df.odps.nan_handler = 'py'

    expr = exprs.expr[1 + exprs.expr.id, exprs.expr.fid]
    expr = expr[expr.id.fillna(0), expr.fid]
    expected = "SELECT IF((1 + t1.`id`) IS NULL, 0, 1 + t1.`id`) AS `id`, t1.`fid` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.filter(func.sample(5, 1, 'name', 'id', rtype='boolean'))
    expected = "SELECT * \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "WHERE sample(5, 1, 'name', 'id')"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.select(func.sample(exprs.expr.id, 1, 'name', 'id', rtype='boolean', project='udf_proj'))
    expected = "SELECT udf_proj:sample(t1.`id`, 1, 'name', 'id') AS `id` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.select(func.sample(exprs.expr.id, exprs.expr.fid, 'name', 'id', rtype='boolean',
                                        name='renamed'))
    expected = "SELECT sample(t1.`id`, t1.`fid`, 'name', 'id') AS `renamed` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.filter(exprs.expr.id < 10)[exprs.expr.name, exprs.expr.id]
    expected = "SELECT t1.`name`, t1.`id` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "WHERE t1.`id` < 10"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.copy()
    expr._proxy = None

    expr['new_id'] = expr.id + 1
    expr['new_id2'] = expr['new_id'] + 2
    expr2 = expr.filter(expr.new_id < 10)
    expr2['new_id3'] = expr2.new_id + expr2.id
    expr2['new_id2'] = expr2.new_id
    expected = "SELECT t2.`name`, t2.`id`, t2.`fid`, t2.`isMale`, t2.`scale`, " \
               "t2.`birth`, t2.`new_id`, t2.`new_id` AS `new_id2`, t2.`new_id` + t2.`id` AS `new_id3` \n" \
               "FROM (\n" \
               "  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale`, t1.`birth`, " \
               "t1.`id` + 1 AS `new_id`, (t1.`id` + 1) + 2 AS `new_id2` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               ") t2 \n" \
               "WHERE t2.`new_id` < 10"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr2, prettify=False))

    expr3 = expr.distinct()
    expr3['new_id3'] = expr3['new_id2'] * 2
    del expr3.new_id2

    expected = "SELECT t2.`name`, t2.`id`, t2.`fid`, t2.`isMale`, t2.`scale`, " \
               "t2.`birth`, t2.`new_id`, t2.`new_id2` * 2 AS `new_id3` \n" \
               "FROM (\n" \
               "  SELECT DISTINCT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, " \
               "t1.`scale`, t1.`birth`, t1.`id` + 1 AS `new_id`, " \
               "(t1.`id` + 1) + 2 AS `new_id2` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               ") t2"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr3, prettify=False))

    expr = exprs.expr['name', 'id'].distinct()
    expr = expr[expr.id > 0]
    expr = expr.exclude('id')
    expr = expr[expr, expr.apply(lambda row: row[0], axis=1, reduce=True, types='string').rename('name2')]

    engine = ODPSEngine(odps)

    expected = "SELECT t2.`name`, {0}(t2.`name`) AS `name2` \n" \
               "FROM (\n" \
               "  SELECT DISTINCT t1.`name`, t1.`id` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               ") t2 \n" \
               "WHERE t2.`id` > 0"
    res = engine.compile(expr, prettify=False)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == to_text(res)

    expr = exprs.expr['name', 'id'].distinct()
    expr = expr[expr.id > 0]
    expr = expr[expr.exclude('id'), (expr.name + 'new').rename('name3')]
    expr = expr[expr, expr.apply(lambda row: row[0], axis=1, reduce=True, types='string').rename('name2')]

    engine = ODPSEngine(odps)

    expected = "SELECT t2.`name`, CONCAT(t2.`name`, 'new') AS `name3`, " \
               "{0}(t2.`name`, CONCAT(t2.`name`, 'new')) AS `name2` \n" \
               "FROM (\n" \
               "  SELECT DISTINCT t1.`name`, t1.`id` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               ") t2 \n" \
               "WHERE t2.`id` > 0"
    res = engine.compile(expr, prettify=False)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == to_text(res)

    expr = exprs.expr.limit(5)
    expr = expr.map_reduce(mapper=lambda row: row[0],
                           mapper_output_names=['name'], mapper_output_types='string')

    engine = ODPSEngine(odps)

    expected = "SELECT {0}(t2.`name`, t2.`id`, t2.`fid`, t2.`isMale`, " \
               "CAST(t2.`scale` AS STRING), t2.`birth`) AS (`name`) \n" \
               "FROM (\n" \
               "  SELECT * \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "  LIMIT 5\n" \
               ") t2"
    res = engine.compile(expr, prettify=False)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == to_text(res)


def test_mem_cache_compilation(odps, exprs):
    cached = exprs.expr['name', exprs.expr.id + 1].cache(mem=True)
    expr = cached.groupby('name').agg(cached.id.sum())

    expected = '@c1 := CACHE ON SELECT t1.`name`, t1.`id` + 1 AS `id` ' \
               'FROM mocked_project.`pyodps_test_expr_table` t1;\n' \
               'SELECT \n' \
               '  t2.`name`,\n' \
               '  SUM(t2.`id`) AS `id_sum` \n' \
               'FROM @c1 t2 \n' \
               'GROUP BY \n' \
               '  t2.`name`'
    assert expected in repr(ODPSSQLEngine(odps).compile(expr).nodes()[0])

    cached = exprs.expr.cache(mem=True)
    cached2 = cached['name', 'id'].cache(mem=True)
    expr = cached2.id.sum()

    expected = '@c1 := CACHE ON SELECT * FROM mocked_project.`pyodps_test_expr_table` t1;\n' \
               '@c2 := CACHE ON SELECT t2.`name`, t2.`id` FROM @c1 t2;\n' \
               'SELECT \n' \
               '  SUM(t3.`id`) AS `id_sum` \n' \
               'FROM @c2 t3'

    assert expected in repr(ODPSSQLEngine(odps).compile(expr).nodes()[0])

    expr2 = exprs.expr.view().cache(mem=True)
    expr = exprs.expr.cache(mem=True).join(expr2)

    expected = '@c1 := CACHE ON SELECT * FROM mocked_project.`pyodps_test_expr_table` t1;\n' \
               'SELECT \n' \
               '  t1.`name`,\n' \
               '  t1.`id`,\n' \
               '  t1.`fid`,\n' \
               '  t1.`isMale`,\n' \
               '  t1.`scale`,\n' \
               '  t1.`birth` \n' \
               'FROM @c1 t1 \n' \
               'INNER JOIN \n' \
               '  @c1 t2\n' \
               'ON (((((t1.`name` == t2.`name`) AND (t1.`id` == t2.`id`)) ' \
               'AND (t1.`fid` == t2.`fid`)) AND (t1.`isMale` == t2.`isMale`)) ' \
               'AND (t1.`scale` == t2.`scale`)) AND (t1.`birth` == t2.`birth`)'

    assert expected in repr(ODPSSQLEngine(odps).compile(expr).nodes()[0])

    expr2 = exprs.expr.cache(mem=True)
    expr = exprs.expr.join(expr2)

    assert expected in repr(ODPSSQLEngine(odps).compile(expr).nodes()[0])

    expr2 = exprs.expr[exprs.expr.id + 1, ].cache(mem=True)
    expr = exprs.expr.join(expr2)

    expected = '@c1 := CACHE ON SELECT * FROM mocked_project.`pyodps_test_expr_table` t1;\n' \
               '@c3 := CACHE ON SELECT t1.`id` + 1 AS `id` FROM @c1 t1;\n' \
               'SELECT \n' \
               '  t2.`name`,\n' \
               '  t2.`id`,\n' \
               '  t2.`fid`,\n' \
               '  t2.`isMale`,\n' \
               '  t2.`scale`,\n' \
               '  t2.`birth` \n' \
               'FROM @c1 t2 \n' \
               'INNER JOIN \n' \
               '  @c3 t3\n' \
               'ON t2.`id` == t3.`id`'

    assert expected in repr(ODPSSQLEngine(odps).compile(expr).nodes()[0])


def test_element_compilation(odps, exprs):
    expect = 'SELECT t1.`id` IS NULL AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.id.isnull(), prettify=False))

    expect = 'SELECT t1.`id` IS NOT NULL AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.id.notnull(), prettify=False))

    expect = 'SELECT IF(t1.`id` IS NULL, 100, t1.`id`) AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.id.fillna(100), prettify=False))

    expect = 'SELECT (1 + t1.`id`) IS NULL AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile((1 + exprs.expr.id).isnull(), prettify=False))

    expect = 'SELECT (1 + t1.`id`) IS NOT NULL AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile((1 + exprs.expr.id).notnull(), prettify=False))

    expr = exprs.expr.id.isin([1, 2, 3]).rename('id')
    expect = 'SELECT t1.`id` IN (1, 2, 3) AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.id.isin(exprs.expr.fid.astype('int')).rename('id')
    expect = 'SELECT t1.`id` IN ' \
             '(SELECT CAST(t2.`fid` AS BIGINT) AS `fid` FROM mocked_project.`pyodps_test_expr_table` t2) ' \
             'AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.filter(exprs.expr.id.isin(exprs.expr3.id))
    expect = 'SELECT * \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
             'WHERE t1.`id` IN (SELECT t2.`id` FROM mocked_project.`pyodps_test_expr_table2` t2)'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.id.notin([1, 2, 3]).rename('id')
    expect = 'SELECT t1.`id` NOT IN (1, 2, 3) AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.id.notin(exprs.expr.fid.astype('int')).rename('id')
    expect = 'SELECT t1.`id` NOT IN ' \
             '(SELECT CAST(t1.`fid` AS BIGINT) AS `fid` FROM mocked_project.`pyodps_test_expr_table` t1) ' \
             'AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expect = 'SELECT (t1.`fid` <= t1.`id`) AND (t1.`id` <= 3) AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.id.between(exprs.expr.fid, 3), prettify=False))

    expect = 'SELECT IF(t1.`id` < 5, \'test\', CONCAT(t1.`name`, \'abc\')) AS `id` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == ODPSEngine(odps).compile(
                         (exprs.expr.id < 5).ifelse('test', exprs.expr.name + 'abc').rename('id'),
                         prettify=False)

    expr = exprs.expr.name.switch('test', 'test' + exprs.expr.name,
                                 'test2', 'test2' + exprs.expr.name).rename('name')
    expect = 'SELECT CASE t1.`name` WHEN \'test\' THEN CONCAT(\'test\', t1.`name`) ' \
             'WHEN \'test2\' THEN CONCAT(\'test2\', t1.`name`) END AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = switch(exprs.expr.name == 'test', 'test2', default='notest').rename('name')
    expect = "SELECT CASE WHEN t1.`name` == 'test' THEN 'test2' ELSE 'notest' END AS `name` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.id.cut([100, 200, 300], labels=['small', 'large']).rename('tp')
    expect = 'SELECT CASE WHEN (100 < t1.`id`) AND (t1.`id` <= 200) THEN \'small\' ' \
             'WHEN (200 < t1.`id`) AND (t1.`id` <= 300) THEN \'large\' END AS `tp` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.id.cut([100, 200, 300],
                            labels=['xsmall', 'small', 'large', 'xlarge'],
                            include_under=True, include_over=True).rename('tp')
    expect = "SELECT CASE WHEN t1.`id` <= 100 THEN 'xsmall' " \
             "WHEN (100 < t1.`id`) AND (t1.`id` <= 200) THEN 'small' " \
             "WHEN (200 < t1.`id`) AND (t1.`id` <= 300) THEN 'large' " \
             "WHEN 300 < t1.`id` THEN 'xlarge' END AS `tp` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.id.cut([100, 200, 300],
                            labels=['xsmall', 'small', 'large', 'xlarge'],
                            include_lowest=True,
                            include_under=True, include_over=True).rename('tp')
    expect = "SELECT CASE WHEN t1.`id` < 100 THEN 'xsmall' " \
             "WHEN (100 <= t1.`id`) AND (t1.`id` <= 200) THEN 'small' " \
             "WHEN (200 < t1.`id`) AND (t1.`id` <= 300) THEN 'large' " \
             "WHEN 300 < t1.`id` THEN 'xlarge' END AS `tp` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.birth.unix_timestamp.to_datetime()
    expect = "SELECT FROM_UNIXTIME(UNIX_TIMESTAMP(t1.`birth`)) AS `birth` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.scale.map(lambda x: x + 1)
    engine = ODPSEngine(odps)
    engine.compile(expr)
    data = ['2.3', '3.3']
    _testify_udf([str(Decimal(d) + 1) for d in data], [(d, ) for d in data], engine)


def test_arithmetic_compilation(odps, exprs):
    for e in (exprs.expr.id + 5, exprs.expr.id - 5, exprs.expr.id * 5,
              exprs.expr.id / 5, exprs.expr.id > 5, 5 < exprs.expr.id,
              exprs.expr.id >= 5, 5 <= exprs.expr.id, exprs.expr.id < 5,
              exprs.expr.id <= 5, exprs.expr.id == 5, exprs.expr.id != 5):
        expect = 'SELECT t1.`id` {0} 5 AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'.format(
            BINARY_OP_COMPILE_DIC[e.__class__.__name__])

        assert to_text(expect) == to_text(ODPSEngine(odps).compile(e, prettify=False))

    for e in (5 - exprs.expr.id, 5 * exprs.expr.id, 5 / exprs.expr.id):
        expect = 'SELECT 5 {0} t1.`id` AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'.format(
            BINARY_OP_COMPILE_DIC[e.__class__.__name__])

        assert to_text(expect) == to_text(ODPSEngine(odps).compile(e, prettify=False))

    for e in (exprs.expr.isMale & True, exprs.expr.isMale | True):
        expect = 'SELECT t1.`isMale` {0} true AS `isMale` \nFROM mocked_project.`pyodps_test_expr_table` t1'.format(
            BINARY_OP_COMPILE_DIC[e.__class__.__name__].upper())

        assert to_text(expect) == to_text(ODPSEngine(odps).compile(e, prettify=False))

    assert to_text('SELECT -t1.`id` AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1') == to_text(ODPSEngine(odps).compile((-exprs.expr.id), prettify=False))

    now = datetime.now()
    unix_time = to_timestamp(now)
    expr = exprs.expr.birth < now
    expect = 'SELECT t1.`birth` < FROM_UNIXTIME(%s) AS `birth` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1' % unix_time
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.birth + year(1)
    expect = "SELECT DATEADD(t1.`birth`, 1, 'yyyy') AS `birth` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.birth + month(1)
    expect = "SELECT DATEADD(t1.`birth`, 1, 'mm') AS `birth` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.birth + day(1)
    expect = "SELECT DATEADD(t1.`birth`, 1, 'dd') AS `birth` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.birth + hour(1)
    expect = "SELECT DATEADD(t1.`birth`, 1, 'hh') AS `birth` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.birth + minute(1)
    expect = "SELECT DATEADD(t1.`birth`, 1, 'mi') AS `birth` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.birth + second(1)
    expect = "SELECT DATEADD(t1.`birth`, 1, 'ss') AS `birth` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    data = [datetime.now()]

    expr = exprs.expr.birth - millisecond(100)
    engine = ODPSEngine(odps)
    engine.compile(expr)
    if sys.version_info[0] == 2:
        _testify_udf([to_milliseconds(d - timedelta(milliseconds=100)) for d in data],
                          [(d, 100, '-') for d in data], engine)
    else:
        _testify_udf([d - timedelta(milliseconds=100) for d in data],
                          [(d, 100, '-') for d in data], engine)

    expr = exprs.expr.birth - datetime.now()
    engine = ODPSEngine(odps)
    engine.compile(expr)
    now = datetime.now()
    _testify_udf([int((d - now).total_seconds() * 1000) for d in data],
                      [(d, now, '-') for d in data], engine)

    expr = exprs.expr.scale < Decimal('3.14')
    expect = "SELECT t1.`scale` < CAST('3.14' AS DECIMAL) AS `scale` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    data = [(30, 8)]

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.id // 8)
    _testify_udf([l // r for l, r in data], [d for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile((exprs.expr.id // exprs.expr.id).rename('id'))
    _testify_udf([l // l for l, r in data], [d for d in [(30, 30)]], engine)

    expr = 'tt' + exprs.expr.id.astype('string')
    expect = "SELECT CONCAT('tt', CAST(t1.`id` AS STRING)) AS `id` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_math_compilation(odps, exprs):
    for math_cls, func in MATH_COMPILE_DIC.items():
        e = getattr(exprs.expr.id, math_cls.lower())()

        expect = 'SELECT {0}(t1.`id`) AS `id` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1'.format(func.upper())

        assert to_text(expect) == to_text(ODPSEngine(odps).compile(e, prettify=False))

    expect = 'SELECT LN(t1.`id`) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.id.log(), prettify=False))

    expect = 'SELECT LOG(2, t1.`id`) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.id.log2(), prettify=False))

    expect = 'SELECT LOG(10, t1.`id`) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.id.log10(), prettify=False))

    expect = 'SELECT LN(1 + t1.`id`) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.id.log1p(), prettify=False))

    expect = 'SELECT EXP(t1.`id`) - 1 AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.id.expm1(), prettify=False))

    expect = 'SELECT TRUNC(t1.`fid`) AS `fid` \nFROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.fid.trunc(), prettify=False))

    expect = 'SELECT TRUNC(t1.`fid`, 2) AS `fid` \nFROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.fid.trunc(2), prettify=False))

    data = [30]
    try:
        import numpy as np
    except ImportError:
        return

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.fid.arccosh())
    _testify_udf([np.arccosh(d) for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.fid.arcsinh())
    _testify_udf([np.arcsinh(d) for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.fid.degrees())
    _testify_udf([np.degrees(d) for d in data], [(d,) for d in data], engine)

    data = [0.2]

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.fid.arctanh())
    _testify_udf([np.arctanh(d) for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.fid.radians())
    _testify_udf([np.radians(d) for d in data], [(d,) for d in data], engine)


def test_string_compilation(odps, exprs):
    expect = 'SELECT CONCAT(TOUPPER(SUBSTR(t1.`name`, 1, 1)), TOLOWER(SUBSTR(t1.`name`, 2))) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.capitalize(), prettify=False))

    expr = exprs.expr[
        exprs.expr.name.cat(exprs.expr.id.astype('string')).rename('name1'),
        exprs.expr.name.cat(exprs.expr.id.astype('string'), sep=',').rename('name2'),
        exprs.expr.name.cat([exprs.expr.id.astype('string'),
                            exprs.expr.fid.astype('string')], sep=',', na_rep='?').rename('name3')
    ]
    expect = "SELECT IF(CAST(t1.`id` AS STRING) IS NULL, t1.`name`, " \
             "CONCAT(t1.`name`, CAST(t1.`id` AS STRING))) AS `name1`, " \
             "IF(CAST(t1.`id` AS STRING) IS NULL, t1.`name`, " \
             "CONCAT(CONCAT(t1.`name`, ','), CAST(t1.`id` AS STRING))) AS `name2`, " \
             "CONCAT(IF(t1.`name` IS NULL, '?', t1.`name`), ',', IF(CAST(t1.`id` AS STRING) IS NULL, '?', " \
             "CAST(t1.`id` AS STRING)), ',', IF(CAST(t1.`fid` AS STRING) IS NULL, '?', " \
             "CAST(t1.`fid` AS STRING))) AS `name3` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.contains('test'))
    _testify_udf([True, False], [('test', 'test', True, 0), ('tes', 'test', True, 0)], engine)
    expect = 'SELECT INSTR(t1.`name`, \'test\') > 0 AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.contains('test', regex=False), prettify=False))
    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.contains('test', flags=re.I))
    _testify_udf([True, False, True], [('Test', 'test', True, re.I), ('tes', 'test', True, re.I),
                                            ('ToTEst', 'test', True, re.I)], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.count('test', re.I))
    _testify_udf([1, 2], [('testTst', 'test', re.I), ('testTest', 'test', re.I)], engine)

    expect = 'SELECT INSTR(REVERSE(t1.`name`), REVERSE(\'test\')) == 1 AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.endswith('test'), prettify=False))

    expect = 'SELECT INSTR(t1.`name`, \'test\') == 1 AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.startswith('test'), prettify=False))

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.extract('test'))
    _testify_udf(['test', None, 'test'],
                      [('test', 'test', 0, 0), ('tes', 'test', 0, 0),
                       ('test32', 'test', 0, 0)], engine)
    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.extract(r'test\d', flags=re.I))
    _testify_udf(['Test1', None, 'test3'],
                      [('Test1', r'test\d', re.I, 0),
                       ('tes', r'test\d', re.I, 0),
                       ('test32', r'test\d', re.I, 0)], engine)

    expect = 'SELECT INSTR(t1.`name`, \'test\', 1) - 1 AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.find('test'), prettify=False))
    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.rfind('test'))
    data = ['abcdggtest1222', 'notes', 'test']
    _testify_udf([d.rfind('test') for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.replace('test', 'test2'))
    data = ['test1', ]
    _testify_udf([d.replace('test', 'test2') for d in data],
                 [(d, 'test', 'test2', -1, True, 0, False) for d in data], engine)

    expect = 'SELECT SUBSTR(t1.`name`, 3, 1) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.get(2), prettify=False))

    expect = 'SELECT LENGTH(t1.`name`) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.len(), prettify=False))

    expect = "SELECT CONCAT(t1.`name`, IF((10 - LENGTH(t1.`name`)) >= 0, " \
             "REPEAT(' ', 10 - LENGTH(t1.`name`)), '')) AS `name` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.ljust(10), prettify=False))
    expect = "SELECT CONCAT(t1.`name`, IF((10 - LENGTH(t1.`name`)) >= 0, " \
             "REPEAT('*', 10 - LENGTH(t1.`name`)), '')) AS `name` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.ljust(10, fillchar='*'), prettify=False))

    expect = "SELECT CONCAT(IF((10 - LENGTH(t1.`name`)) >= 0, " \
             "REPEAT(' ', 10 - LENGTH(t1.`name`)), ''), t1.`name`) AS `name` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.rjust(10), prettify=False))

    expect = 'SELECT TOLOWER(t1.`name`) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.lower(), prettify=False))

    expect = 'SELECT TOUPPER(t1.`name`) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.upper(), prettify=False))

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.lstrip())
    data = [' abc\n', ' \nddd']
    _testify_udf([d.lstrip() for d in data], [(d,) for d in data], engine)
    expect = 'SELECT LTRIM(t1.`name`) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.lstrip(to_strip=' '), prettify=False))

    expect = 'SELECT RTRIM(t1.`name`) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.rstrip(to_strip=' '), prettify=False))

    expect = 'SELECT TRIM(t1.`name`) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.strip(to_strip=' '), prettify=False))

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.pad(10))
    data = ['ab', 'a' * 12]
    _testify_udf([d.rjust(10) for d in data], [(d, 10, ' ', 'left') for d in data], engine)

    expect = 'SELECT REPEAT(t1.`name`, 4) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.repeat(4), prettify=False))

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.slice(0, 10, 2))
    data = ['ab' * 15, 'def']
    _testify_udf([d[0: 10: 2] for d in data], [(d, 7, 0, 10, 2) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.slice(-5, -1))
    data = ['ab' * 15, 'def']
    _testify_udf([d[-5: -1] for d in data], [(d, 6, -5, -1) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.slice(-5, step=1).value_counts(sort=False))
    data = ['ab' * 15, 'def']
    _testify_udf([d[-5:] for d in data], [(d, 4, -5, 1) for d in data], engine)

    expect = 'SELECT SPLIT(t1.`name`, \'x\') AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.split('x'), prettify=False))

    expect = 'SELECT SPLIT(t1.`name`, \'\\\\\') AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.split('\\'), prettify=False))

    expect = 'SELECT IF(SIZE(SPLIT(t1.`name`, \'\\\\.\')) = 0, SPLIT(t1.`name`, \'.\'), ' \
             'SPLIT(t1.`name`, \'\\\\.\')) AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.split('.'), prettify=False))

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.title())
    data = ['Abc Def', 'ADEFddEE']
    _testify_udf([d.title() for d in data], [(d,) for d in data], engine)

    expect = "SELECT CONCAT(IF((10 - LENGTH(t1.`name`)) >= 0, " \
             "REPEAT('0', 10 - LENGTH(t1.`name`)), ''), t1.`name`) AS `name` \n" \
             "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.zfill(10), prettify=False))

    data = ['123', 'dEf', '124df']

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.isalnum())
    _testify_udf([d.isalnum() for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.isalpha())
    _testify_udf([d.isalpha() for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.isdigit())
    _testify_udf([d.isdigit() for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.isspace())
    _testify_udf([d.isspace() for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.islower())
    _testify_udf([d.islower() for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.isupper())
    _testify_udf([d.isupper() for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.istitle())
    _testify_udf([d.istitle() for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.isnumeric())
    _testify_udf([to_text(d).isnumeric() for d in data], [(d,) for d in data], engine)

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.name.isdecimal())
    _testify_udf([to_text(d).isdecimal() for d in data], [(d,) for d in data], engine)

    expect = 'SELECT STR_TO_MAP(t1.`name`, \',\', \':\') AS `name` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.name.todict(',', ':'), prettify=False))


def test_value_counts(odps, exprs):
    labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    expr = exprs.expr.id.cut(list(range(0, 81, 10)), labels=labels).rename('id_group').value_counts()

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
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.id.value_counts().sort('id')['id', 'count']
    expected = 'SELECT t3.`id`, t3.`count` \n' \
               'FROM (\n' \
               '  SELECT * \n' \
               '  FROM (\n' \
               '    SELECT t1.`id`, COUNT(t1.`id`) AS `count` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '    GROUP BY t1.`id` \n' \
               '    ORDER BY count DESC \n' \
               '    LIMIT 10000\n' \
               '  ) t2 \n' \
               '  ORDER BY id \n' \
               '  LIMIT 10000\n' \
               ') t3'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.id.value_counts(sort=True, ascending=True, dropna=True)
    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT t1.`id`, COUNT(t1.`id`) AS `count` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  GROUP BY t1.`id` \n' \
               '  ORDER BY count \n' \
               '  LIMIT 10000\n' \
               ') t2 \n' \
               'WHERE t2.`id` IS NOT NULL'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_datetime_compilation(odps, exprs):
    with pytest.raises(NotImplementedError):
        ODPSEngine(odps).compile(exprs.expr.birth.time)

    for clz, date_part in DATE_PARTS_DIC.items():
        expect = 'SELECT DATEPART(t1.`birth`, \'%s\') AS `birth` \n' \
                 'FROM mocked_project.`pyodps_test_expr_table` t1' % date_part
        attr = getattr(exprs.expr.birth, clz.lower())
        assert to_text(expect) == to_text(ODPSEngine(odps).compile(attr, prettify=False))

    with pytest.raises(NotImplementedError):
        ODPSEngine(odps).compile(exprs.expr.birth.microsecond)

    with pytest.raises(NotImplementedError):
        ODPSEngine(odps).compile(exprs.expr.birth.week)

    expect = 'SELECT WEEKOFYEAR(t1.`birth`) AS `birth` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.birth.weekofyear, prettify=False))

    expect = 'SELECT WEEKDAY(t1.`birth`) AS `birth` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.birth.dayofweek, prettify=False))
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.birth.weekday, prettify=False))

    expect = 'SELECT UNIX_TIMESTAMP(t1.`birth`) AS `birth` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr.birth.unix_timestamp, prettify=False))

    with pytest.raises(NotImplementedError):
        ODPSEngine(odps).compile(exprs.expr.birth.dayofyear)

    with pytest.raises(NotImplementedError):
        ODPSEngine(odps).compile(exprs.expr.birth.is_month_start)

    with pytest.raises(NotImplementedError):
        ODPSEngine(odps).compile(exprs.expr.birth.is_month_end)

    with pytest.raises(NotImplementedError):
        ODPSEngine(odps).compile(exprs.expr.birth.is_year_start)

    with pytest.raises(NotImplementedError):
        ODPSEngine(odps).compile(exprs.expr.birth.is_year_end)

    data = [datetime.now()]

    engine = ODPSEngine(odps)
    engine.compile(exprs.expr.birth.strftime('%Y'))
    _testify_udf([d.strftime('%Y') for d in data], [(d, '%Y') for d in data], engine)


def test_composite_compilation(odps, exprs):
    expect = 'SELECT ARRAY(5, t1.`id`, t1.`fid`) AS `arr` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    expr = make_list(5, exprs.expr.id, exprs.expr.fid).rename('arr')
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expect = 'SELECT MAP(\'id\', t1.`id`, \'fid\', t1.`fid`) AS `dict` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    expr = make_dict('id', exprs.expr.id, 'fid', exprs.expr.fid).rename('dict')
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expect = 'SELECT SIZE(t1.`hobbies`) AS `hobbies` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.hobbies.len(), prettify=False))

    expect = 'SELECT SIZE(t1.`relatives`) AS `relatives` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.relatives.len(), prettify=False))

    expect = 'SELECT SORT_ARRAY(t1.`hobbies`) AS `hobbies` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.hobbies.sort(), prettify=False))

    expect = 'SELECT ARRAY_CONTAINS(t1.`hobbies`, t1.`name`) AS `hobbies` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.hobbies.contains(exprs.expr4.name),
                                                          prettify=False))

    expect = 'SELECT MAP_KEYS(t1.`relatives`) AS `relatives` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.relatives.keys(), prettify=False))

    expect = 'SELECT MAP_VALUES(t1.`relatives`) AS `relatives` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.relatives.values(), prettify=False))

    expect = 'SELECT t1.`hobbies`[2] AS `hobbies` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.hobbies[2], prettify=False))

    expect = 'SELECT t1.`hobbies`[SIZE(t1.`hobbies`) - 2] AS `hobbies` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.hobbies[-2], prettify=False))

    expect = 'SELECT IF((t1.`id` - 1) >= 0, t1.`hobbies`[t1.`id` - 1], t1.`hobbies`[SIZE(t1.`hobbies`) + (t1.`id` - 1)]) AS `hobbies` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.hobbies[exprs.expr4.id - 1], prettify=False))

    expect = 'SELECT t1.`relatives`[\'abc\'] AS `relatives` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.relatives['abc'], prettify=False))

    expect = 'SELECT t1.`relatives`[t1.`name`] AS `relatives` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.relatives[exprs.expr4.name], prettify=False))

    expect = 'SELECT EXPLODE(t1.`relatives`) AS (`relatives_key`, `relatives_value`) \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(exprs.expr4.relatives.explode(), prettify=False))

    expect = 'SELECT t1.`id`, t1.`name`, t2.`hobbies` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
             'LATERAL VIEW EXPLODE(t1.`hobbies`) t2 AS `hobbies`'
    expr = exprs.expr4[exprs.expr4.id, exprs.expr4.name, exprs.expr4.hobbies.explode()]
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expect = 'SELECT t1.`id`, t1.`name`, t2.`hobbies_pos`, t2.`hobbies` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
             'LATERAL VIEW POSEXPLODE(t1.`hobbies`) t2 AS `hobbies_pos`, `hobbies`'
    expr = exprs.expr4[exprs.expr4.id, exprs.expr4.name, exprs.expr4.hobbies.explode(pos=True)]
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expect = 'SELECT t1.`id`, t1.`name`, t2.`relatives_key`, t2.`relatives_value` \n' \
             'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
             'LATERAL VIEW EXPLODE(t1.`relatives`) t2 AS `relatives_key`, `relatives_value`'
    expr = exprs.expr4[exprs.expr4.id, exprs.expr4.name, exprs.expr4.relatives.explode()]
    assert to_text(expect) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_sample_compiliation(odps, exprs):
    expr = exprs.expr.sample(10, columns='name')
    expected = 'SELECT * \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'WHERE SAMPLE(10, 1, t1.`name`)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.sample(parts=10, i=(2, 3, 4), columns=exprs.expr.id)
    expected = 'SELECT * \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'WHERE (SAMPLE(10, 3, t1.`id`) OR SAMPLE(10, 4, t1.`id`)) OR SAMPLE(10, 5, t1.`id`)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.sample(frac=0.1)
    expected = 'SELECT * \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'TABLESAMPLE (10 PERCENT)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.sample(n=100)
    expected = 'SELECT * \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'TABLESAMPLE (100 ROWS)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_sort_compilation(odps, exprs):
    expr = exprs.expr.sort(['name', -exprs.expr.id])[:50]

    expected = 'SELECT * \nFROM mocked_project.`pyodps_test_expr_table` t1 \nORDER BY name, id DESC \nLIMIT 50'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.sort(['name', 'id'], ascending=[True, False])[:50]
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr['id', 'name'].sort(['name'])['name', 'id']
    expected = 'SELECT t2.`name`, t2.`id` \n' \
               'FROM (\n' \
               '  SELECT t1.`id`, t1.`name` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  ORDER BY name \n' \
               '  LIMIT 10000\n' \
               ') t2'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.sort(exprs.expr.id + 1)[:50]
    expected = 'SELECT * \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'ORDER BY t1.`id` + 1 \n' \
               'LIMIT 50'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_distinct_compilation(odps, exprs):
    expr = exprs.expr.distinct(['name', exprs.expr.id + 1])

    expected = 'SELECT DISTINCT t1.`name`, t1.`id` + 1 AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr['name', 'id'].distinct()

    expected = 'SELECT DISTINCT t1.`name`, t1.`id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr['name'].unique()

    expected = 'SELECT t2.`name` \n' \
               'FROM (\n' \
               '  SELECT DISTINCT t1.`name` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               ') t2'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_groupby_compilation(odps, exprs):
    expr = exprs.expr.groupby(['name', 'id'])[lambda x: x.id.min() * 2 < 10] \
        .agg(exprs.expr.fid.max() + 1, new_id=exprs.expr.id.sum())

    expected = 'SELECT t1.`name`, t1.`id`, MAX(t1.`fid`) + 1 AS `fid_max`, SUM(t1.`id`) AS `new_id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`, t1.`id` \n' \
               'HAVING (MIN(t1.`id`) * 2) < 10'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').agg(id=exprs.expr.id.max())[
        lambda x: x.name.astype('float'), 'id']['id', 'name']

    expected = 'SELECT MAX(t1.`id`) AS `id`, CAST(t1.`name` AS DOUBLE) AS `name` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').agg(id=exprs.expr.id.max()).sort('id')['id', 'name']
    expected = 'SELECT t2.`id`, t2.`name` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, MAX(t1.`id`) AS `id` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  GROUP BY t1.`name` \n' \
               '  ORDER BY id \n' \
               '  LIMIT 10000\n' \
               ') t2'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['name']).mutate(lambda x: x.row_number(sort='id'))

    expected = 'SELECT t1.`name`, ' \
               'ROW_NUMBER() OVER (PARTITION BY t1.`name` ORDER BY t1.`id`) AS `row_number` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.name.value_counts()[:25]
    expr2 = exprs.expr.name.topk(25)

    expected = 'SELECT t1.`name`, COUNT(t1.`name`) AS `count` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name` \n' \
               'ORDER BY count DESC \n' \
               'LIMIT 25'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr2, prettify=False))

    expr = exprs.expr.groupby('name').count()
    expected = 'SELECT COUNT(1) AS `count` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(Scalar(1)).id.max()
    expected = 'SELECT MAX(t1.`id`) AS `id_max` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY 1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(exprs.expr['name'].rename('name2')).agg(id=exprs.expr['id'].sum())
    expr = expr[expr.name2, expr.id + 3]

    expected = 'SELECT t1.`name` AS `name2`, SUM(t1.`id`) + 3 AS `id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['name', 'id']).filter(exprs.expr.id.min() * 2 < 10) \
        .agg(exprs.expr.fid.max() + 1, new_id=exprs.expr.id.unique().sum())

    expected = 'SELECT t1.`name`, t1.`id`, MAX(t1.`fid`) + 1 AS `fid_max`, SUM(DISTINCT t1.`id`) AS `new_id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`, t1.`id` \n' \
               'HAVING (MIN(t1.`id`) * 2) < 10'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['name', 'id']).filter(exprs.expr.id.min() * 2 < 10) \
        .agg(exprs.expr.fid.max() + 1, new_id=exprs.expr.id.sum())

    expected = 'SELECT t1.`name`, t1.`id`, MAX(t1.`fid`) + 1 AS `fid_max`, SUM(t1.`id`) AS `new_id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`, t1.`id` \n' \
               'HAVING (MIN(t1.`id`) * 2) < 10'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').agg(exprs.expr.id.nunique(),
                                         exprs.expr['id', 'fid'].nunique().rename('nunique'),
                                         lambda x: x['id', 'name'].nunique().rename('nunique2'),
                                         lambda x: x.fid.nunique().rename('nunique3'))

    expected = 'SELECT t1.`name`, COUNT(DISTINCT t1.`id`) AS `id_nunique`, ' \
               'COUNT(DISTINCT t1.`id`, t1.`fid`) AS `nunique`, ' \
               'COUNT(DISTINCT t1.`id`, t1.`name`) AS `nunique2`, ' \
               'COUNT(DISTINCT t1.`fid`) AS `nunique3` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_mutate(odps, exprs):
    expr = exprs.expr[exprs.expr.exclude('birth'), exprs.expr.fid.astype('int').rename('new_id')]
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
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_reshuffle(odps, exprs):
    expr = exprs.expr.reshuffle(RandomScalar())

    expected = 'SELECT * \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'DISTRIBUTE BY rand()'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.reshuffle()
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.reshuffle(exprs.expr.name, sort='id', ascending=False)[lambda x: x.id + 1, ]

    expected = 'SELECT t2.`id` + 1 AS `id` \n' \
               'FROM (\n' \
               '  SELECT * \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  DISTRIBUTE BY name \n' \
               '  SORT BY id DESC\n' \
               ') t2'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.reshuffle(sort=RandomScalar())

    expected = 'SELECT * \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'DISTRIBUTE BY rand() \n' \
               'SORT BY rand()'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_filter_groupby_sink_filter_compilation(odps, exprs):
    # to test the sinking of filter to `having` clause
    expr = exprs.expr.groupby(['name']).agg(id=exprs.expr.id.max())[lambda x: x.id < 10]
    expected = 'SELECT t1.`name`, MAX(t1.`id`) AS `id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name` \n' \
               'HAVING MAX(t1.`id`) < 10'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_window_rewrite(odps, exprs):
    expr = exprs.expr[exprs.expr.id - exprs.expr.id.mean() < 10][
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
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_window_rewrite_in_select_compilation(odps, exprs):
    # to test rewriting the window function in select clause
    expr = exprs.expr.id - exprs.expr.id.max()
    expected = 'SELECT t2.`id` - t2.`id_max_0` AS `id` \n' \
               'FROM (\n' \
               '  SELECT t1.`id`, MAX(t1.`id`) OVER (PARTITION BY 1) AS `id_max_0` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               ') t2'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_window_rewrite_in_filter_compilation(odps, exprs):
    # to test rewriting the window function in filter clause
    expr = exprs.expr[exprs.expr.id - exprs.expr.id.mean() < 10]
    expected = 'SELECT t2.`name`, t2.`id`, t2.`fid`, t2.`isMale`, t2.`scale`, t2.`birth` \n' \
               'FROM (\n' \
               '  SELECT t1.`birth`, t1.`fid`, t1.`id`, AVG(t1.`id`) OVER (PARTITION BY 1) AS `id_mean_0`, ' \
               't1.`isMale`, t1.`name`, t1.`scale` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               ') t2 \n' \
               'WHERE (t2.`id` - t2.`id_mean_0`) < 10'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_window_rewrite_after_groupby_compilation(odps, exprs):
    expr = exprs.expr.groupby('name').agg(id=exprs.expr.id.sum())
    expr = expr['name', expr.id.sum().rename('id')]
    expected = "SELECT t3.`name`, t3.`id_sum_0` AS `id` \n" \
               "FROM (\n" \
               "  SELECT SUM(t2.`id`) OVER (PARTITION BY 1) AS `id_sum_0`, t2.`name` \n" \
               "  FROM (\n" \
               "    SELECT t1.`name`, SUM(t1.`id`) AS `id` \n" \
               "    FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "    GROUP BY t1.`name` \n" \
               "  ) t2 \n" \
               ") t3"

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_reduction_compilation(odps, exprs):
    # TODO test all the reductions
    expr = exprs.expr.groupby(['id']).id.std(ddof=0) + 1
    expected = 'SELECT STDDEV(t1.`id`) + 1 AS `id_std` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.fid.mean()
    expected = 'SELECT AVG(t1.`fid`) AS `fid_mean` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.fid.unique().mean()
    expected = 'SELECT AVG(DISTINCT t1.`fid`) AS `fid_mean` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).id.std() + 1
    expected = 'SELECT STDDEV_SAMP(t1.`id`) + 1 AS `id_std` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).id.var() + 1
    expected = 'SELECT (POW(STDDEV_SAMP(t1.`id`), 2)) + 1 AS `id_var` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).fid.moment(3, central=True) + 1
    expected = 'SELECT (((AVG(POW(t1.`fid`, 3))' \
               ' - ((3 * AVG(POW(t1.`fid`, 2))) * AVG(t1.`fid`)))' \
               ' + ((3 * AVG(t1.`fid`)) * (POW(AVG(t1.`fid`), 2))))' \
               ' - (POW(AVG(t1.`fid`), 3))) + 1 AS `fid_moment` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).fid.skew() + 1
    expected = 'SELECT (((((AVG(POW(t1.`fid`, 3))' \
               ' - ((3 * AVG(POW(t1.`fid`, 2))) * AVG(t1.`fid`)))' \
               ' + ((3 * AVG(t1.`fid`)) * (POW(AVG(t1.`fid`), 2))))' \
               ' - (POW(AVG(t1.`fid`), 3))) / (POW(STDDEV_SAMP(t1.`fid`), 3)))' \
               ' * (((CAST(POW(COUNT(t1.`fid`), 2) AS BIGINT))' \
               ' / (COUNT(t1.`fid`) - 1)) / (COUNT(t1.`fid`) - 2))) + 1 AS `fid_skew` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).fid.kurtosis() + 1
    expected = 'SELECT (((1.0 / (COUNT(t1.`fid`) - 2)) / (COUNT(t1.`fid`) - 3))' \
               ' * (((((COUNT(t1.`fid`) * COUNT(t1.`fid`)) - 1) * ((((AVG(POW(t1.`fid`, 4))' \
               ' - ((4 * AVG(POW(t1.`fid`, 3))) * AVG(t1.`fid`)))' \
               ' + ((6 * AVG(POW(t1.`fid`, 2))) * (POW(AVG(t1.`fid`), 2))))' \
               ' - ((4 * AVG(t1.`fid`)) * (POW(AVG(t1.`fid`), 3))))' \
               ' + (POW(AVG(t1.`fid`), 4))))' \
               ' / (POW(STDDEV(t1.`fid`), 4)))' \
               ' - (3 * (CAST(POW(COUNT(t1.`fid`) - 1, 2) AS BIGINT))))) + 1 AS `fid_kurtosis` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).fid.moment(3) + 1
    expected = 'SELECT AVG(POW(t1.`fid`, 3)) + 1 AS `fid_moment` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.count()
    expected = 'SELECT COUNT(1) AS `count` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = (exprs.expr.id == 1).any().rename('equal1')
    expected = 'SELECT MAX(IF(t1.`id` == 1, 1, 0)) == 1 AS `equal1` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected)

    expr = exprs.expr.isMale.all()
    expected = 'SELECT MIN(IF(t1.`isMale`, 1, 0)) == 1 AS `isMale_all` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected)

    expr = exprs.expr.id.nunique()
    expected = 'SELECT COUNT(DISTINCT t1.`id`) AS `id_nunique` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.name.cat(sep=',', na_rep='')
    expected = "SELECT WM_CONCAT(',', IF(t1.`name` IS NULL, '', t1.`name`)) AS `name_cat` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.name.cat(sep=',')
    expected = "SELECT WM_CONCAT(',', t1.`name`) AS `name_cat` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).agg(name=exprs.expr.name.sum()).count()
    expected = 'SELECT COUNT(1) AS `count` \n' \
               'FROM (\n' \
               '  SELECT t1.`id`, WM_CONCAT(\'\', t1.`name`) AS `name` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  GROUP BY t1.`id` \n' \
               ') t2'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).mean()
    expected = 'SELECT t1.`id`, AVG(t1.`fid`) AS `fid_mean`, ' \
               'AVG(t1.`id`) AS `id_mean`, AVG(t1.`scale`) AS `scale_mean` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).quantile(0.25)
    expected = 'SELECT t1.`id`, PERCENTILE(t1.`fid`, 0.25) AS `fid_quantile`, ' \
               'PERCENTILE(t1.`id`, 0.25) AS `id_quantile`, ' \
               'PERCENTILE(t1.`scale`, 0.25) AS `scale_quantile` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['id']).quantile([0.25, 0.5, 0.75])
    expected = 'SELECT t1.`id`, PERCENTILE(t1.`fid`, ARRAY(0.25, 0.5, 0.75)) AS `fid_quantile`, ' \
               'PERCENTILE(t1.`id`, ARRAY(0.25, 0.5, 0.75)) AS `id_quantile`, ' \
               'PERCENTILE(t1.`scale`, ARRAY(0.25, 0.5, 0.75)) AS `scale_quantile` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['name']).any()
    expected = 'SELECT t1.`name`, MAX(IF(t1.`isMale`, 1, 0)) == 1 AS `isMale_any` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['name']).name.nunique()
    expected = 'SELECT COUNT(DISTINCT t1.`name`) AS `name_nunique` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['name']).name.tolist()
    expected = 'SELECT COLLECT_LIST(t1.`name`) AS `name_tolist` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby(['name']).name.tolist(unique=True)
    expected = 'SELECT COLLECT_SET(t1.`name`) AS `name_tolist` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'GROUP BY t1.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.limit(10).count()
    expected = 'SELECT COUNT(1) AS `count` \n' \
               'FROM (\n' \
               '  SELECT * \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  LIMIT 10\n' \
               ') t2'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

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

    expr = exprs.expr.id.agg(Aggregator)
    engine = ODPSEngine(odps)
    engine.compile(expr)
    _testify_udf([9, ], [[r, ] for r in data], engine)

    data = [Decimal('2.2'), Decimal('1.1'), Decimal('3.3')]

    is_decimal = True
    expr = exprs.expr.scale.agg(Aggregator)
    engine = ODPSEngine(odps)
    engine.compile(expr)
    _testify_udf(['9.6', ], [[str(r), ] for r in data], engine)


def test_projection_compact(odps, exprs):
    expr = exprs.expr[exprs.expr.id.rename('new_id'), exprs.expr.name, exprs.expr.name.rename('new_name2')]
    expr = expr[expr.new_id.rename('new_id2'), expr.name.rename('new_name'), expr.new_name2]

    expected = 'SELECT t1.`id` AS `new_id2`, t1.`name` AS `new_name`, t1.`name` AS `new_name2` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_groupby_compact(odps, exprs):
    expr = exprs.expr['name', exprs.expr.id + 1].groupby('name').count()

    expected = 'SELECT COUNT(1) AS `count` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, t1.`id` + 1 AS `id` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               ') t2 \n' \
               'GROUP BY t2.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_window_compilation(odps, exprs):
    # TODO test all window functions
    expr = exprs.expr.groupby('name').id.cumcount(unique=True)

    expected = 'SELECT COUNT(DISTINCT t1.`id`) OVER (PARTITION BY t1.`name`) AS `id_count` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').isMale.cumsum()
    expected = 'SELECT SUM(IF(t1.`isMale`, 1, 0)) OVER (PARTITION BY t1.`name`) AS `isMale_sum` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').id.cummedian(preceding=2)
    expected = 'SELECT MEDIAN(t1.`id`) OVER (PARTITION BY t1.`name` ROWS 2 PRECEDING) AS `id_median` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').id.cummedian(preceding=2, following=3)
    expected = 'SELECT MEDIAN(t1.`id`) OVER (PARTITION BY t1.`name` ' \
               'ROWS BETWEEN 2 PRECEDING AND 3 FOLLOWING) AS `id_median` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').id.nth_value(4)
    expected = 'SELECT NTH_VALUE(t1.`id`, 5) OVER (PARTITION BY t1.`name`) AS `id_nthvalue` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').id.nth_value(4, skip_nulls=True)
    expected = 'SELECT NTH_VALUE(t1.`id`, 5, true) OVER (PARTITION BY t1.`name`) AS `id_nthvalue` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr[exprs.expr.name, exprs.expr.id + 1][
        'name', lambda x: x.groupby('name').sort('id', ascending=False).row_number().rename('rank')]
    expected = "SELECT t1.`name`, ROW_NUMBER() OVER (PARTITION BY t1.`name` ORDER BY t1.`id` + 1 DESC) AS `rank` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr[exprs.expr.name, exprs.expr.groupby(Scalar(1)).id.cumcount()]
    expected = 'SELECT t1.`name`, COUNT(t1.`id`) OVER (PARTITION BY 1) AS `id_count` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert expected == ODPSEngine(odps).compile(expr, prettify=False)

    expr = exprs.expr.groupby('name').sort('fid').id.lag(1)
    expected = 'SELECT \n' \
               '  LAG(\n' \
               '      t1.`id`,\n' \
               '      1) OVER (PARTITION BY t1.`name` ORDER BY t1.`fid`) AS `id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr))

    expected = 'SELECT LAG(t1.`id`, 1) OVER (PARTITION BY t1.`name` ORDER BY t1.`fid`) AS `id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').sort('fid').qcut(4)
    expected = 'SELECT NTILE(4) OVER (PARTITION BY t1.`name` ORDER BY t1.`fid`) - 1 AS `q_cut` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr.groupby('name').sort('fid').cume_dist()
    expected = 'SELECT CUME_DIST() OVER (PARTITION BY t1.`name` ORDER BY t1.`fid`) AS `cume_dist` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_window_setitem(odps, exprs):
    expr = exprs.expr[exprs.expr, (exprs.expr.id + 1).rename('id1')]
    expr['id1_row_number'] = expr.groupby('name').sort('id1').row_number()
    expected = 'SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale`, t1.`birth`, t1.`id` + 1 AS `id1`, ' \
               'ROW_NUMBER() OVER (PARTITION BY t1.`name` ORDER BY t1.`id` + 1) AS `id1_row_number` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_apply(odps, exprs):
    expr = exprs.expr.groupby('name').sort(['name', 'id']).apply(
        lambda x: x, names=exprs.expr.schema.names)['id', 'name']

    expected = 'SELECT \n' \
               '  t3.`id`,\n' \
               '  t3.`name` \n' \
               'FROM (\n' \
               '  SELECT {0}(t2.`name`, t2.`id`, t2.`fid`, t2.`isMale`, ' \
               'CAST(t2.`scale` AS STRING), t2.`birth`) AS (`name`, `id`, `fid`, ' \
               '`isMale`, `scale`, `birth`) \n' \
               '  FROM (\n' \
               '    SELECT * \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '    DISTRIBUTE BY \n' \
               '      name \n' \
               '    SORT BY \n' \
               '      name,\n' \
               '      id\n' \
               '  ) t2 \n' \
               ') t3'
    engine = ODPSEngine(odps)
    res = engine.compile(expr)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == to_text(res)

    expr2 = expr.join(expr.view())
    engine.compile(expr2)
    assert len(engine._ctx._registered_funcs.values()) == 1

    from .....config import options
    options.df.quote = False

    engine = ODPSEngine(odps)
    res = engine.compile(expr)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.replace('`', '').format(fun_name)) == to_text(res)

    options.df.quote = True


def test_join(odps, exprs):
    e = exprs.expr
    e1 = exprs.expr1
    e2 = exprs.expr2
    joined = e.join(e1, ['fid'])
    expected = 'SELECT t1.`name` AS `name_x`, t1.`id` AS `id_x`, t1.`fid`, t1.`isMale` AS `isMale_x`, ' \
               't1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, t2.`name` AS `name_y` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\nON t1.`fid` == t2.`fid`'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined[e, e1.name], prettify=False))

    expected = 'SELECT t1.`name` AS `name_x`, t1.`id`, t1.`fid`, t1.`isMale` AS `isMale_x`, ' \
               't1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, t2.`name` AS `name_y`,' \
               ' t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, t2.`birth` AS `birth_y` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)'

    joined = e.join(e1, ['fid', 'id'])

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    joined = e.join(e1, ['fid', 'id'])

    joined2 = e.join(e2, ['name'])
    joined3 = joined.join(joined2, joined.name_x == joined2.name)

    expected = 'SELECT t5.`name`, t5.`id_x`, t5.`fid_x`, ' \
               't5.`isMale_x` AS `isMale_x_y`, t5.`scale_x` AS `scale_x_y`, t5.`birth_x` AS `birth_x_y`, ' \
               't5.`id_y`, t5.`fid_y`, ' \
               't5.`isMale_y` AS `isMale_y_y`, t5.`scale_y` AS `scale_y_y`, ' \
               't5.`birth_y` AS `birth_y_y`, t1.`name` AS `name_x` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`) \n' \
               'INNER JOIN \n' \
               '  (\n' \
               '    SELECT t3.`name`, t3.`id` AS `id_x`, t3.`fid` AS `fid_x`, t3.`isMale` AS `isMale_x`, ' \
               't3.`scale` AS `scale_x`, t3.`birth` AS `birth_x`, t4.`id` AS `id_y`, ' \
               't4.`fid` AS `fid_y`, t4.`isMale` AS `isMale_y`, t4.`scale` AS `scale_y`, t4.`birth` AS `birth_y` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t3 \n' \
               '    INNER JOIN \n' \
               '      mocked_project.`pyodps_test_expr_table2` t4\n' \
               '    ON t3.`name` == t4.`name`\n' \
               '  ) t5\n' \
               'ON t1.`name` == t5.`name`'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined3[joined2, joined.name_x], prettify=False))
    # test twice to check the cache
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined3[joined2, joined.name_x], prettify=False))

    expected = 'SELECT t2.`name` AS `new_name`, t1.`id` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \nINNER JOIN \n  ' \
               'mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`)'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined[lambda x, y: y.name.rename('new_name'), e.id],
                                                          prettify=False))

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

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(e2.join(joined, joined.name_y == e2.name), prettify=False))

    expected = 'SELECT t1.`name` AS `name_x`, t1.`id` AS `id_x`, t1.`fid`, ' \
               't1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, ' \
               't2.`name` AS `name_y_x`, ' \
               't2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, t2.`birth` AS `birth_y`, ' \
               't5.`name_y` AS `name_y_y`, t5.`id` AS `id_y` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON (t1.`fid` == t2.`fid`) AND (t1.`id` == t2.`id`) \n' \
               'INNER JOIN \n' \
               '  (\n' \
               '    SELECT t4.`name` AS `name_y`, t3.`id` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t3 \n' \
               '    INNER JOIN \n' \
               '      mocked_project.`pyodps_test_expr_table1` t4\n' \
               '    ON (t3.`fid` == t4.`fid`) AND (t3.`id` == t4.`id`)\n' \
               '  ) t5\n' \
               'ON t5.`name_y` == t1.`name`'

    joined = e.join(e1, ['fid', 'id']).join(joined, lambda x, y: y.name_y == x.name_x)
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    joined = e.join(e1, ['fid', 'id'])[e.name, e1]
    joined2 = e.join(e2, ['name'])[e.name, e.fid, e2.id]
    joined3 = joined.join(joined2, joined.name_x == joined2.name)[joined2.name, ]
    expected = 'SELECT t6.`name` \n' \
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
               '    SELECT t4.`name`, t4.`fid` AS `fid_x`, t5.`id` AS `id_y` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t4 \n' \
               '    INNER JOIN \n' \
               '      mocked_project.`pyodps_test_expr_table2` t5\n' \
               '    ON t4.`name` == t5.`name`\n' \
               '  ) t6\n' \
               'ON t3.`name_x` == t6.`name`'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined3, prettify=False))

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
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

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
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    e, e1 = exprs.expr.filter(exprs.expr.id < 10), exprs.expr2
    rht = e1.filter(e1.id < 10)['id', lambda x: x.name.rename('name2')].distinct()
    expr = e.join(rht, on='id', suffixes=('', '_tmp'))['id', 'name', 'name2']
    expr = e.join(expr, on=['name', 'id'], suffixes=('', '_tmp'))

    expected = "SELECT t2.`name`, t2.`id`, t2.`fid`, " \
               "t2.`isMale`, t2.`scale`, t2.`birth`, t7.`name2` \n" \
               "FROM (\n" \
               "  SELECT * \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "  WHERE t1.`id` < 10\n" \
               ") t2 \n" \
               "INNER JOIN \n" \
               "  (\n" \
               "    SELECT t4.`id`, t4.`name`, t6.`name2` \n" \
               "    FROM (\n" \
               "      SELECT * \n" \
               "      FROM mocked_project.`pyodps_test_expr_table` t3 \n" \
               "      WHERE t3.`id` < 10\n" \
               "    ) t4 \n" \
               "    INNER JOIN \n" \
               "      (\n" \
               "        SELECT DISTINCT t5.`id`, t5.`name` AS `name2` \n" \
               "        FROM mocked_project.`pyodps_test_expr_table2` t5 \n" \
               "        WHERE t5.`id` < 10\n" \
               "      ) t6\n" \
               "    ON t4.`id` == t6.`id`\n" \
               "  ) t7\n" \
               "ON (t2.`name` == t7.`name`) AND (t2.`id` == t7.`id`)"

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_join_multiple_filter(odps, exprs):
    e = exprs.expr
    e1 = exprs.expr1[exprs.expr1.name.rename('name2'), exprs.expr1.id.rename('id2')]
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
               'WHERE (t1.`id` IS NOT NULL) AND (t3.`name2` IS NULL)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

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
               'WHERE (t1.`id` IS NOT NULL) AND (t3.`name2` IS NULL)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr))


def test_self_join(odps, exprs):
    joined = exprs.expr.join(exprs.expr, 'name')
    expected = 'SELECT t1.`name`, t1.`id` AS `id_x`, t1.`fid` AS `fid_x`, t1.`isMale` AS `isMale_x`, ' \
               't1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, t2.`id` AS `id_y`, ' \
               't2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, t2.`birth` AS `birth_y` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table` t2\n' \
               'ON t1.`name` == t2.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    proj = exprs.expr['name', 'id']
    joined = proj.join(proj, 'name').join(proj, 'name')
    expected = 'SELECT t2.`name`, t2.`id` AS `id_x`, t4.`id` AS `id_y`, t6.`id` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, t1.`id` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
               ') t2 \n' \
               'INNER JOIN \n' \
               '  (\n' \
               '    SELECT t3.`name`, t3.`id` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t3\n' \
               '  ) t4\n' \
               'ON t2.`name` == t4.`name` \n' \
               'INNER JOIN \n' \
               '  (\n' \
               '    SELECT t5.`name`, t5.`id` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t5\n' \
               '  ) t6\n' \
               'ON t2.`name` == t6.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    e1, e2 = exprs.expr, exprs.expr1
    expr = e1.join(e2, on=[e1['id'] == e2['id']])
    expr2 = expr.join(e1, on=[expr['fid_x'] == e1['fid']])
    expected = 'SELECT t1.`name` AS `name_x`, t1.`id` AS `id_x`, t1.`fid` AS `fid_x`, ' \
               't1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, ' \
               't1.`birth` AS `birth_x`, t2.`name` AS `name_y`, ' \
               't2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, ' \
               't2.`scale` AS `scale_y`, t2.`birth` AS `birth_y`, t3.`name`, ' \
               't3.`id` AS `id_y`, t3.`fid`, t3.`isMale`, t3.`scale`, t3.`birth` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON t1.`id` == t2.`id` \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table` t3\n' \
               'ON t1.`fid` == t3.`fid`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr2, prettify=False))


def test_left_join(odps, exprs):
    left = exprs.expr.select(exprs.expr, type='normal')
    right = exprs.expr[:4]
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
               "    FROM mocked_project.`pyodps_test_expr_table` t3 \n" \
               "    LIMIT 4\n" \
               "  ) t4\n" \
               "ON t2.`id` == t4.`id`"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(res, prettify=False))

    left = exprs.expr.exclude('birth')
    right = exprs.expr[:4]
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
               '    FROM mocked_project.`pyodps_test_expr_table` t3 \n' \
               '    LIMIT 4\n' \
               '  ) t4\n' \
               'ON t2.`id` == t4.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(res, prettify=False))

    left = exprs.expr.exclude('birth')
    right = exprs.expr[:4]
    res = left.left_join(right, on='id', merge_columns=True)

    expected = 'SELECT t2.`name` AS `name_x`, IF(t2.`id` IS NULL, t4.`id`, t2.`id`) AS `id`, ' \
               't2.`fid` AS `fid_x`, t2.`isMale` AS `isMale_x`, t2.`scale` AS `scale_x`, ' \
               't4.`name` AS `name_y`, t4.`fid` AS `fid_y`, t4.`isMale` AS `isMale_y`, ' \
               't4.`scale` AS `scale_y`, t4.`birth` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
               ') t2 \n' \
               'LEFT OUTER JOIN \n' \
               '  (\n' \
               '    SELECT * \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t3 \n' \
               '    LIMIT 4\n' \
               '  ) t4\n' \
               'ON t2.`id` == t4.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(res, prettify=False))

    left = exprs.expr.exclude('birth')
    right = exprs.expr[:4]
    res = left.left_join(right, on=['name', 'id'], merge_columns='id')

    expected = 'SELECT t2.`name` AS `name_x`, IF(t2.`id` IS NULL, t4.`id`, t2.`id`) AS `id`, ' \
               't2.`fid` AS `fid_x`, t2.`isMale` AS `isMale_x`, t2.`scale` AS `scale_x`, ' \
               't4.`name` AS `name_y`, t4.`fid` AS `fid_y`, t4.`isMale` AS `isMale_y`, ' \
               't4.`scale` AS `scale_y`, t4.`birth` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
               ') t2 \n' \
               'LEFT OUTER JOIN \n' \
               '  (\n' \
               '    SELECT * \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t3 \n' \
               '    LIMIT 4\n' \
               '  ) t4\n' \
               'ON (t2.`name` == t4.`name`) AND (t2.`id` == t4.`id`)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(res, prettify=False))

    left = exprs.expr.exclude('birth')
    right = exprs.expr[:4]
    res = left.left_join(right, on=['name', 'id'], merge_columns=dict(name=True, id='left'))

    expected = 'SELECT IF(t2.`name` IS NULL, t4.`name`, t2.`name`) AS `name`, t2.`id`, ' \
               't2.`fid` AS `fid_x`, t2.`isMale` AS `isMale_x`, t2.`scale` AS `scale_x`, ' \
               't4.`fid` AS `fid_y`, t4.`isMale` AS `isMale_y`, t4.`scale` AS `scale_y`, t4.`birth` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
               ') t2 \n' \
               'LEFT OUTER JOIN \n' \
               '  (\n' \
               '    SELECT * \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t3 \n' \
               '    LIMIT 4\n' \
               '  ) t4\n' \
               'ON (t2.`name` == t4.`name`) AND (t2.`id` == t4.`id`)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(res, prettify=False))

    left = exprs.expr.exclude('birth')
    right = exprs.expr[:4]
    joined = left.left_join(right, on=['name', 'id'], merge_columns=dict(name=True, id='left'))
    res = joined[joined.id.rename('id_joined'), right.id.rename('id_right'), left.fid * 10, right]

    expected = 'SELECT t2.`id` AS `id_joined`, t2.`id` AS `id_right`, t2.`fid` * 10 AS `fid_x`, ' \
               'IF(t2.`name` IS NULL, t4.`name`, t2.`name`) AS `name`, t2.`id`, t4.`fid` AS `fid_y`, ' \
               't4.`isMale` AS `isMale_y`, t4.`scale` AS `scale_y`, t4.`birth` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
               ') t2 \n' \
               'LEFT OUTER JOIN \n' \
               '  (\n' \
               '    SELECT * \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t3 \n' \
               '    LIMIT 4\n' \
               '  ) t4\n' \
               'ON (t2.`name` == t4.`name`) AND (t2.`id` == t4.`id`)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(res, prettify=False))

    left = exprs.expr.exclude('birth')
    right = exprs.expr[:4]
    res = left.left_join(right, on='id', merge_columns=True)
    res['rn_id'] = res.groupby('id').sort('birth').row_number()

    expected = 'SELECT t2.`name` AS `name_x`, IF(t2.`id` IS NULL, t4.`id`, t2.`id`) AS `id`, ' \
               't2.`fid` AS `fid_x`, t2.`isMale` AS `isMale_x`, t2.`scale` AS `scale_x`, ' \
               't4.`name` AS `name_y`, t4.`fid` AS `fid_y`, t4.`isMale` AS `isMale_y`, ' \
               't4.`scale` AS `scale_y`, t4.`birth`, ' \
               'ROW_NUMBER() OVER (PARTITION BY IF(t2.`id` IS NULL, t4.`id`, t2.`id`) ' \
               'ORDER BY t4.`birth`) AS `rn_id` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
               ') t2 \n' \
               'LEFT OUTER JOIN \n' \
               '  (\n' \
               '    SELECT * \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t3 \n' \
               '    LIMIT 4\n' \
               '  ) t4\n' \
               'ON t2.`id` == t4.`id`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(res, prettify=False))

    left = exprs.expr.exclude('birth')
    right = exprs.expr[:4]
    res = left.left_join(right, on='id', merge_columns=True)
    res['id1'] = res.apply(lambda row: row.id, axis=1, reduce=True)
    res['id2'] = res.apply(lambda row: row.id, axis=1, reduce=True)

    expected = 'SELECT t2.`name` AS `name_x`, IF(t2.`id` IS NULL, t4.`id`, t2.`id`) AS `id`, ' \
               't2.`fid` AS `fid_x`, t2.`isMale` AS `isMale_x`, t2.`scale` AS `scale_x`, ' \
               't4.`name` AS `name_y`, t4.`fid` AS `fid_y`, t4.`isMale` AS `isMale_y`, ' \
               't4.`scale` AS `scale_y`, t4.`birth`, {0}(t2.`name`, IF(t2.`id` IS NULL, t4.`id`, t2.`id`), t2.`fid`, t2.`isMale`, CAST(t2.`scale` AS STRING), t4.`name`, t4.`fid`, t4.`isMale`, CAST(t4.`scale` AS STRING), t4.`birth`) AS `id1`, ' \
               '{1}(t2.`name`, IF(t2.`id` IS NULL, t4.`id`, t2.`id`), t2.`fid`, t2.`isMale`, CAST(t2.`scale` AS STRING), t4.`name`, t4.`fid`, t4.`isMale`, CAST(t4.`scale` AS STRING), t4.`birth`, {0}(t2.`name`, IF(t2.`id` IS NULL, t4.`id`, t2.`id`), t2.`fid`, t2.`isMale`, CAST(t2.`scale` AS STRING), t4.`name`, t4.`fid`, t4.`isMale`, CAST(t4.`scale` AS STRING), t4.`birth`)) AS `id2` \n' \
               'FROM (\n' \
               '  SELECT t1.`name`, t1.`id`, t1.`fid`, t1.`isMale`, t1.`scale` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1\n' \
               ') t2 \n' \
               'LEFT OUTER JOIN \n' \
               '  (\n' \
               '    SELECT * \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t3 \n' \
               '    LIMIT 4\n' \
               '  ) t4\n' \
               'ON t2.`id` == t4.`id`'

    engine = ODPSEngine(odps)
    compiled = engine.compile(res, prettify=False)
    fun_names = list(engine._ctx._registered_funcs.values())
    assert to_text(expected).format(*fun_names) == to_text(compiled)


def test_map_join(odps, exprs):
    joined = exprs.expr.join(exprs.expr1, on=[], mapjoin=True)
    expected = \
        'SELECT /*+ mapjoin(t2) */ t1.`name` AS `name_x`, t1.`id` AS `id_x`, t1.`fid` AS `fid_x`, ' \
        't1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, t2.`name` AS `name_y`, ' \
        't2.`id` AS `id_y`, t2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, ' \
        't2.`birth` AS `birth_y` \n' \
        'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
        'INNER JOIN \n' \
        '  mocked_project.`pyodps_test_expr_table1` t2'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    joined = exprs.expr.join(exprs.expr1, on=[lambda x, y: x.name > y.name], mapjoin=True).select(exprs.expr1.name)
    expected = 'SELECT /*+ mapjoin(t2) */ t2.`name` AS `name_y` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON t1.`name` > t2.`name`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    joined = exprs.expr.join(exprs.expr1, on=[], mapjoin=True).join(exprs.expr2, on=[], mapjoin=True).select(exprs.expr.name)
    expected = 'SELECT /*+ mapjoin(t2, t3) */ t1.`name` AS `name_x` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table2` t3'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))
    joined = exprs.expr.join(exprs.expr1.join(exprs.expr2, on=[], mapjoin=True).select(exprs.expr2.name), mapjoin=True, on=[]).select(exprs.expr.name)

    expected = 'SELECT /*+ mapjoin(t4) */ t1.`name` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  (\n' \
               '    SELECT /*+ mapjoin(t3) */ t3.`name` AS `name_y` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table1` t2 \n' \
               '    INNER JOIN \n' \
               '      mocked_project.`pyodps_test_expr_table2` t3\n' \
               '  ) t4'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    joined = exprs.expr['name', 'id'].join(exprs.expr1.limit(4), on=[], mapjoin=True)

    expected = 'SELECT /*+ mapjoin(t4) */ t2.`name` AS `name_x`, t2.`id` AS `id_x`, t4.`name` AS `name_y`, ' \
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
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    expr = joined[joined.id_x == 1]
    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT /*+ mapjoin(t4) */ t2.`name` AS `name_x`, t2.`id` AS `id_x`, ' \
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
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_complex_map_join(odps, exprs):
    def distance(x1, y1, x2, y2):
        return (x1 - x2) ** 2 + (y1 - y2) ** 2

    input = exprs.expr[exprs.expr.id.rename('uid'), exprs.expr.fid.rename('ux'), exprs.expr.fid.rename('uy')]
    center = exprs.expr[exprs.expr.id.rename('cid'), exprs.expr.fid.rename('x'), exprs.expr.fid.rename('y')][:4]
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

    expected = 'SELECT t9.`cid`, AVG(t9.`ux`) AS `x`, AVG(t9.`uy`) AS `y` \n' \
               'FROM (\n' \
               '  SELECT /*+ mapjoin(t8) */ t2.`uid`, t2.`ux`, t2.`uy`, t8.`cid`, ' \
               'ROW_NUMBER() OVER (PARTITION BY t2.`uid` ' \
               'ORDER BY (POW(t2.`ux` - t8.`x`, 2)) + (POW(t2.`uy` - t8.`y`, 2))) AS `row_number` \n' \
               '  FROM (\n' \
               '    SELECT t1.`id` AS `uid`, t1.`fid` AS `ux`, t1.`fid` AS `uy` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t1\n' \
               '  ) t2 \n' \
               '  INNER JOIN \n' \
               '    (\n' \
               '      SELECT t7.`cid`, AVG(t7.`ux`) AS `x`, AVG(t7.`uy`) AS `y` \n' \
               '      FROM (\n' \
               '        SELECT /*+ mapjoin(t6) */ t4.`uid`, t4.`ux`, t4.`uy`, t6.`cid`, ' \
               'ROW_NUMBER() OVER (PARTITION BY t4.`uid` ' \
               'ORDER BY (POW(t4.`ux` - t6.`x`, 2)) + (POW(t4.`uy` - t6.`y`, 2))) AS `row_number` \n' \
               '        FROM (\n' \
               '          SELECT t3.`id` AS `uid`, t3.`fid` AS `ux`, t3.`fid` AS `uy` \n' \
               '          FROM mocked_project.`pyodps_test_expr_table` t3\n' \
               '        ) t4 \n' \
               '        INNER JOIN \n' \
               '          (\n' \
               '            SELECT t5.`id` AS `cid`, t5.`fid` AS `x`, t5.`fid` AS `y` \n' \
               '            FROM mocked_project.`pyodps_test_expr_table` t5 \n' \
               '            LIMIT 4\n' \
               '          ) t6 \n' \
               '      ) t7 \n' \
               '      WHERE t7.`row_number` == 1 \n' \
               '      GROUP BY t7.`cid`\n' \
               '    ) t8 \n' \
               ') t9 \n' \
               'WHERE t9.`row_number` == 1 \n' \
               'GROUP BY t9.`cid`'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(center, prettify=False))


def test_skew_join(odps, exprs):
    joined = exprs.expr.join(exprs.expr1, on=['name', 'id'], skewjoin=True)
    expected = 'SELECT /*+ skewjoin(t2) */ t1.`name`, t1.`id`, t1.`fid` AS `fid_x`, ' \
               't1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, ' \
               't2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, ' \
               't2.`birth` AS `birth_y` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON (t1.`name` == t2.`name`) AND (t1.`id` == t2.`id`)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    joined = exprs.expr.join(exprs.expr1, on=['name', 'id'], skewjoin=['id'])
    expected = 'SELECT /*+ skewjoin(t2(id)) */ t1.`name`, t1.`id`, t1.`fid` AS `fid_x`, ' \
               't1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, t1.`birth` AS `birth_x`, ' \
               't2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, ' \
               't2.`birth` AS `birth_y` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON (t1.`name` == t2.`name`) AND (t1.`id` == t2.`id`)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    joined = exprs.expr.join(exprs.expr1, on=['name', 'id'],
                            skewjoin=[{'name': 'a', 'id': 1}, {'id': 2, 'name': 'b'}])
    expected = 'SELECT /*+ skewjoin(t2(id, name)((1, \'a\'), (2, \'b\'))) */ t1.`name`, t1.`id`, ' \
               't1.`fid` AS `fid_x`, t1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, ' \
               't1.`birth` AS `birth_x`, t2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, ' \
               't2.`scale` AS `scale_y`, t2.`birth` AS `birth_y` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON (t1.`name` == t2.`name`) AND (t1.`id` == t2.`id`)'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))

    expr = joined[joined.id == 1]
    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT /*+ skewjoin(t2(id, name)((1, \'a\'), (2, \'b\'))) */ t1.`name`, ' \
               't1.`id`, t1.`fid` AS `fid_x`, t1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, ' \
               't1.`birth` AS `birth_x`, t2.`fid` AS `fid_y`, t2.`isMale` AS `isMale_y`, ' \
               't2.`scale` AS `scale_y`, t2.`birth` AS `birth_y` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  INNER JOIN \n' \
               '    mocked_project.`pyodps_test_expr_table1` t2\n' \
               '  ON (t1.`name` == t2.`name`) AND (t1.`id` == t2.`id`) \n' \
               ') t3 \n' \
               'WHERE t3.`id` == 1'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    joined = exprs.expr.join(exprs.expr1, on=['id'], skewjoin=True) \
        .join(exprs.expr2, on=[], mapjoin=True) \
        .select(exprs.expr.name)
    expected = 'SELECT /*+ mapjoin(t3), skewjoin(t2) */ t1.`name` AS `name_x` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table1` t2\n' \
               'ON t1.`id` == t2.`id` \n' \
               'INNER JOIN \n' \
               '  mocked_project.`pyodps_test_expr_table2` t3'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(joined, prettify=False))


def test_join_map_reduce(odps, exprs):
    expr = exprs.expr.left_join(exprs.expr3.select(new_id=exprs.expr3['id']), on=('id', 'new_id'))

    @output(expr.schema.names, expr.schema.types)
    def reducer(keys):
        def h(row):
            yield row
        return h

    expr = expr.map_reduce(reducer=reducer, group='name')

    expected = "SELECT t6.`name`, t6.`id`, t6.`fid`, t6.`isMale`, " \
               "CAST(t6.`scale` AS DECIMAL) AS `scale`, t6.`birth`, t6.`new_id` \n" \
               "FROM (\n" \
               "  SELECT {0}(" \
               "t5.`name`, t5.`id`, t5.`fid`, t5.`isMale`, CAST(t5.`scale` AS STRING), " \
               "t5.`birth`, t5.`new_id`) AS (`name`, `id`, `fid`, `isMale`, `scale`, `birth`, `new_id`) \n" \
               "  FROM (\n" \
               "    SELECT * \n" \
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
               "  ) t5 \n" \
               ") t6"
    engine = ODPSEngine(odps)
    res = engine.compile(expr, prettify=False)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == to_text(res)


def test_join_sort(odps, exprs):
    expr = exprs.expr.outer_join(exprs.expr1, on='id')
    expr = expr[expr['id_x'].rename('id'), 'name_x', 'name_y'].sort(['id', 'name_x'])
    engine = ODPSEngine(odps)

    expected = "SELECT t1.`id`, t1.`name` AS `name_x`, t2.`name` AS `name_y` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "FULL OUTER JOIN \n" \
               "  mocked_project.`pyodps_test_expr_table1` t2\n" \
               "ON t1.`id` == t2.`id` \n" \
               "ORDER BY id, name_x \n" \
               "LIMIT 10000"
    assert to_text(expected) == to_text(engine.compile(expr, prettify=False))


def test_left_join_merge_columns_apply_reduce(odps, exprs):
    expr = exprs.expr.left_join(exprs.expr1, on='id', merge_columns=True)

    def func(_):
        return 1

    expr['id2'] = expr.apply(func, reduce=True, axis=1, types='float')

    expected = "SELECT t1.`name` AS `name_x`, IF(t1.`id` IS NULL, t2.`id`, t1.`id`) AS `id`, " \
               "t1.`fid` AS `fid_x`, t1.`isMale` AS `isMale_x`, t1.`scale` AS `scale_x`, " \
               "t1.`birth` AS `birth_x`, t2.`name` AS `name_y`, t2.`fid` AS `fid_y`, " \
               "t2.`isMale` AS `isMale_y`, t2.`scale` AS `scale_y`, t2.`birth` AS `birth_y`, " \
               "{0}(t1.`name`, IF(t1.`id` IS NULL, t2.`id`, t1.`id`), t1.`fid`, t1.`isMale`, " \
               "CAST(t1.`scale` AS STRING), t1.`birth`, t2.`name`, t2.`fid`, t2.`isMale`, " \
               "CAST(t2.`scale` AS STRING), t2.`birth`) AS `id2` \n" \
               "FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "LEFT OUTER JOIN \n" \
               "  mocked_project.`pyodps_test_expr_table1` t2\n" \
               "ON t1.`id` == t2.`id`"

    engine = ODPSEngine(odps)
    res = engine.compile(expr, prettify=False)
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == to_text(res)


def test_as_type(odps, exprs):
    e = exprs.expr
    new_e = e.id.astype('float')
    expected = 'SELECT CAST(t1.`id` AS DOUBLE) AS `id` \nFROM mocked_project.`pyodps_test_expr_table` t1'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(new_e, prettify=False))

    with pytest.raises(CompileError):
        ODPSEngine(odps).compile(exprs.expr.id.astype('boolean'))


def test_odps2_scalar(odps, exprs):
    try:
        options.sql.use_odps2_extension = True
        expr = exprs.expr[
            exprs.expr.id,
            Scalar(1).rename("int8"),
            Scalar(1024).rename("int16"),
            Scalar(2 ** 33).rename("int64"),
        ]

        expected = 'SELECT t1.`id`, 1Y AS `int8`, 1024S AS `int16`, 8589934592L AS `int64` \n' \
                   'FROM mocked_project.`pyodps_test_expr_table` t1'
        assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))
    finally:
        options.sql.use_odps2_extension = None


def test_union(odps, exprs):
    e = exprs.expr
    e1 = exprs.expr1

    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT t1.`name` AS `name_x` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  UNION ALL\n' \
               '    SELECT t2.`name` AS `name_x` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t2 \n' \
               '    INNER JOIN \n' \
               '      mocked_project.`pyodps_test_expr_table1` t3\n' \
               '    ON t2.`name` == t3.`name`\n' \
               ') t4'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(
                         e.name.rename('name_x').union(e.join(e1, 'name')[e['name'].rename('name_x'), ]), False))

    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT * \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n'\
               '  UNION ALL\n'\
               '    SELECT * \n' \
               '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
               ') t3'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(e.union(e1), False))

    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT t1.`name` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  UNION ALL\n' \
               '    SELECT t2.`name` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
               ') t5'
    try:
        assert expected == ODPSEngine(odps).compile(e.name.union(e1.name), False)
    except:
        assert expected[:-1] == ODPSEngine(odps).compile(e.name.union(e1.name), False)[:-1]

    e = exprs.expr['name', 'id']
    e1 = exprs.expr1['name', 'id']

    expected = 'SELECT * \nFROM (\n' \
               '  SELECT t1.`name`, t1.`id` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n'\
               '  UNION ALL\n'\
               '    SELECT t2.`name`, t2.`id` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table1` t2\n' \
               ') t3'
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(e.union(e1), False))

    e1 = exprs.expr1['id', 'name']
    e2 = exprs.expr2['name', 'id']

    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT t1.`id`, t1.`name` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table1` t1 \n' \
               '  UNION ALL\n' \
               '    SELECT t2.`id`, t2.`name` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table2` t2\n' \
               ') t3'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(e1.union(e2), False))

    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT t1.`id`, t1.`name` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table1` t1 \n' \
               '  UNION ALL\n' \
               '    SELECT t2.`id`, t2.`name` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table2` t2\n' \
               ') t3'

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(e1.concat(e2), False))

    expr = e1.union(e2).union(exprs.expr['name', 'id'])
    expected = "SELECT * \n" \
               "FROM (\n" \
               "  SELECT t1.`id`, t1.`name` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table1` t1 \n" \
               "  UNION ALL\n" \
               "    SELECT t2.`id`, t2.`name` \n" \
               "    FROM mocked_project.`pyodps_test_expr_table2` t2 \n" \
               "  UNION ALL\n" \
               "    SELECT t3.`id`, t3.`name` \n" \
               "    FROM mocked_project.`pyodps_test_expr_table` t3\n" \
               ") t4"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = e1.union(e2).union(e1.union(e2))
    expected = "SELECT * \n" \
               "FROM (\n" \
               "  SELECT t1.`id`, t1.`name` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table1` t1 \n" \
               "  UNION ALL\n" \
               "    SELECT t2.`id`, t2.`name` \n" \
               "    FROM mocked_project.`pyodps_test_expr_table2` t2 \n" \
               "  UNION ALL\n" \
               "    SELECT * \n" \
               "    FROM (\n" \
               "      SELECT t3.`id`, t3.`name` \n" \
               "      FROM mocked_project.`pyodps_test_expr_table1` t3 \n" \
               "      UNION ALL\n" \
               "        SELECT t4.`id`, t4.`name` \n" \
               "        FROM mocked_project.`pyodps_test_expr_table2` t4\n" \
               "    ) t5\n" \
               ") t6"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    e = exprs.expr['name', 'id']
    e['const'] = 'cst'
    e1 = exprs.expr1['name', 'id']
    e3 = e1.groupby('name').agg(id=e1.id.sum())
    e3['const'] = 'cst'
    expr = e.union(e3['const', 'id', 'name'])

    expected = "SELECT * \n" \
               "FROM (\n" \
               "  SELECT t1.`name`, t1.`id`, 'cst' AS `const` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "  UNION ALL\n" \
               "    SELECT t2.`name`, SUM(t2.`id`) AS `id`, 'cst' AS `const` \n" \
               "    FROM mocked_project.`pyodps_test_expr_table1` t2 \n" \
               "    GROUP BY t2.`name`\n" \
               ") t3"

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = e.union(e3['const', 'id', 'name'], distinct=True)

    expected = "SELECT * \n" \
               "FROM (\n" \
               "  SELECT t1.`name`, t1.`id`, 'cst' AS `const` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table` t1 \n" \
               "  UNION\n" \
               "    SELECT t2.`name`, SUM(t2.`id`) AS `id`, 'cst' AS `const` \n" \
               "    FROM mocked_project.`pyodps_test_expr_table1` t2 \n" \
               "    GROUP BY t2.`name`\n" \
               ") t3"

    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))


def test_aliases(odps, exprs):
    df = exprs.expr
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
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(df, prettify=False))


def test_filter_parts(odps, exprs):
    df = exprs.expr3
    df = df.filter_parts('part1=a,part2=1/part1=b')['id', 'name']

    expected = "SELECT t2.`id`, t2.`name` \n" \
               "FROM (\n" \
               "  SELECT t1.`name`, t1.`id`, t1.`fid` \n" \
               "  FROM mocked_project.`pyodps_test_expr_table2` t1 \n" \
               "  WHERE ((t1.`part1` == 'a') AND (CAST(t1.`part2` AS BIGINT) == 1)) OR (t1.`part1` == 'b') \n" \
               ") t2"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(df, prettify=False))

    df = exprs.expr3
    df = df.filter_parts('part1=a,part2=1/part1=b', False)['id', 'name']

    expected = "SELECT t1.`id`, t1.`name` \n" \
               "FROM mocked_project.`pyodps_test_expr_table2` t1 \n" \
               "WHERE ((t1.`part1` == 'a') AND (CAST(t1.`part2` AS BIGINT) == 1)) OR (t1.`part1` == 'b')"
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(df, prettify=False))


def test_query(odps, exprs):
    df = exprs.expr1

    df_res = df.query('id+2 >= 3*3')
    df_cmp = df[df.id + 2 >= 9]
    assert to_text(ODPSEngine(odps).compile(df_cmp, prettify=False)) == to_text(ODPSEngine(odps).compile(df_res, prettify=False))

    df_res = df.query('name="test" & id>1 or isMale and True')
    df_cmp = df[(df.name == 'test') & (df.id > 1) | df.isMale & True]
    assert to_text(ODPSEngine(odps).compile(df_cmp, prettify=False)) == to_text(ODPSEngine(odps).compile(df_res, prettify=False))

    id = 1
    fid = 0.2
    s = 't1'
    df_res = df.query('id+1 > @id and fid**2 != @fid & name in [@s,"t2"]')
    df_cmp = df[((df.id + 1) > 1) & (df.fid ** 2 != 0.2) & (df.name.isin(['t1', 't2']))]
    assert to_text(ODPSEngine(odps).compile(df_cmp, prettify=False)) == to_text(ODPSEngine(odps).compile(df_res, prettify=False))

    l = [1, 2, 3, 4]
    df_res = df.query('@df.name="test" & id < @l[2]')
    df_cmp = df[(df.name == 'test') & (df.id < 3)]
    assert to_text(ODPSEngine(odps).compile(df_cmp, prettify=False)) == to_text(ODPSEngine(odps).compile(df_res, prettify=False))

    df_res = df.query('@df.name="test" & id in @l[:-1] or -2<-fid<-3')
    df_cmp = df[(df.name == 'test') & df.id.isin([1, 2, 3]) | ((-df.fid > -2) & (-df.fid < -3))]
    assert to_text(ODPSEngine(odps).compile(df_cmp, prettify=False)) == to_text(ODPSEngine(odps).compile(df_res, prettify=False))


def test_lateral_view(odps, exprs):
    @output(['name', 'id'], ['string', 'int64'])
    def mapper(row):
        for _ in range(10):
            yield row

    with pytest.raises(ExpressionError):
        _ = exprs.expr[
            exprs.expr['name', exprs.expr.id].apply(mapper, axis=1).apply(mapper, axis=1),
            exprs.expr.exclude('name', 'id'),
        ]

    expected = 'SELECT t1.`id` AS `r_id`, t1.`id` * 2 AS `r_id2`, t2.`name`, t2.`id`, ' \
               't1.`fid`, t1.`isMale`, t1.`scale`, t1.`birth`, 5 AS `five` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'LATERAL VIEW {0}(t1.`name`, t1.`id`) t2 AS `name`, `id`'

    expr = exprs.expr[exprs.expr.id.rename('r_id'), (exprs.expr.id * 2).rename('r_id2'),
                     exprs.expr['name', exprs.expr.id].apply(mapper, axis=1),
                     exprs.expr.exclude('name', 'id'), Scalar(5).rename('five')]
    engine = ODPSEngine(odps)
    res = to_text(engine.compile(expr, prettify=False))
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == res

    expected = 'SELECT t2.`id`, t3.`hobbies` \n' \
               'FROM (\n' \
               '  SELECT * \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  WHERE t1.`id` < 5\n' \
               ') t2 \n' \
               'LATERAL VIEW EXPLODE(t2.`hobbies`) t3 AS `hobbies`'
    expr = exprs.expr4[exprs.expr4.id < 5][exprs.expr4.id, exprs.expr4.hobbies.explode()]
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expected = 'SELECT t2.`fid`, t3.`name`, t3.`id` \n' \
               'FROM (\n' \
               '  SELECT * \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  WHERE t1.`id` < 5\n' \
               ') t2 \n' \
               'LATERAL VIEW {0}(t2.`name`, t2.`id`) t3 AS `name`, `id`'

    expr = exprs.expr[exprs.expr.id < 5][exprs.expr.fid, exprs.expr['name', exprs.expr.id].apply(mapper, axis=1)]
    engine = ODPSEngine(odps)
    res = to_text(engine.compile(expr, prettify=False))
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == res

    expected = 'SELECT t2.`name`, t2.`id`, t1.`fid`, t1.`isMale`, t1.`scale`, t1.`birth` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'LATERAL VIEW {0}(t1.`name`, t1.`id` * 2) t2 AS `name`, `id`'

    expr = exprs.expr[exprs.expr['name', exprs.expr.id * 2].apply(mapper, axis=1),
                     exprs.expr.exclude('name', 'id')]
    engine = ODPSEngine(odps)
    res = to_text(engine.compile(expr, prettify=False))
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == res

    expected = 'SELECT t1.`id` * 2 AS `r_id`, t1.`id`, t1.`name`, t1.`relatives`, t1.`hobbies`, ' \
               't2.`r_hobbies`, t3.`r_rel_key`, t3.`r_rel_value` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'LATERAL VIEW EXPLODE(t1.`hobbies`) t2 AS `r_hobbies` \n' \
               'LATERAL VIEW EXPLODE(t1.`relatives`) t3 AS `r_rel_key`, `r_rel_value`'
    expr = exprs.expr4[exprs.expr4.id.rename('r_id') * 2, exprs.expr4,
                      exprs.expr4.hobbies.explode('r_hobbies'),
                      exprs.expr4.relatives.explode(['r_rel_key', 'r_rel_value'])]
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expected = 'SELECT t1.`id` * 2 AS `r_id`, t1.`name` AS `r_name`, t2.`r_hobbies`, ' \
               't3.`r_rel_key`, t3.`r_rel_value` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'LATERAL VIEW EXPLODE(t1.`hobbies`) t2 AS `r_hobbies` \n' \
               'LATERAL VIEW EXPLODE(t1.`relatives`) t3 AS `r_rel_key`, `r_rel_value`'
    expr = exprs.expr4[exprs.expr4.id.rename('r_id') * 2, exprs.expr4.name.rename('r_name'),
                      exprs.expr4.hobbies.explode('r_hobbies'),
                      exprs.expr4.relatives.explode(['r_rel_key', 'r_rel_value'])]
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expected = 'SELECT t1.`r_id`, CONCAT(t1.`r_name`, t3.`r_rel_key`) AS `t_sum1`, ' \
               'CONCAT(t3.`r_rel_key`, t3.`r_rel_value`) AS `t_sum2` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'LATERAL VIEW EXPLODE(t1.`hobbies`) t2 AS `r_hobbies` \n' \
               'LATERAL VIEW EXPLODE(t1.`relatives`) t3 AS `r_rel_key`, `r_rel_value`'
    expr = expr[expr.r_id, (expr.r_name + expr.r_rel_key).rename('t_sum1'),
                (expr.r_rel_key + expr.r_rel_value).rename('t_sum2')]
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expected = 'SELECT t2.`name`, t2.`id`, t3.`relatives` \n' \
               'FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               'LATERAL VIEW {0}(t1.`name`, t1.`id`) t2 AS `name`, `id` \n' \
               'LATERAL VIEW EXPLODE(MAP_VALUES(t1.`relatives`)) t3 AS `relatives`'
    expr = exprs.expr4[exprs.expr4['name', 'id'].apply(mapper, axis=1),
                      exprs.expr4.relatives.values().explode()]
    engine = ODPSEngine(odps)
    res = to_text(engine.compile(expr, prettify=False))
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == res

    @output(['name_map', 'id_map'], ['string', 'int64'])
    def mapper2(row):
        for _ in range(10):
            yield row

    expected = 'SELECT t5.`id`, t5.`name`, t5.`relatives`, t5.`hobbies`, t5.`name_map`, ' \
               't5.`id_map`, t5.`rel_keyexp`, t5.`rel_valexp`, t7.`hb_key` \n' \
               'FROM (\n' \
               '  SELECT t3.`id`, t3.`name`, t3.`relatives`, t3.`hobbies`, t3.`name_map`, ' \
               't3.`id_map`, t4.`rel_keyexp`, t4.`rel_valexp` \n' \
               '  FROM (\n' \
               '    SELECT t1.`id` * 2 AS `id`, t1.`name`, t1.`relatives`, t1.`hobbies`, ' \
               't2.`name_map`, t2.`id_map` \n' \
               '    FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '    LATERAL VIEW {0}(t1.`name`, t1.`id`) t2 AS `name_map`, `id_map`\n' \
               '  ) t3 \n' \
               '  LATERAL VIEW EXPLODE(t3.`relatives`) t4 AS `rel_keyexp`, `rel_valexp`\n' \
               ') t5 \n' \
               'LATERAL VIEW EXPLODE(t5.`hobbies`) t7 AS `hb_key`'
    expr = exprs.expr4[exprs.expr4.id * 2, exprs.expr4.exclude('id'), exprs.expr4['name', 'id'].apply(mapper2, axis=1)]
    expr = expr[expr, expr.relatives.explode(['rel_keyexp', 'rel_valexp'])]
    expr = expr[expr, expr.hobbies.explode('hb_key')]
    engine = ODPSEngine(odps)
    res = to_text(engine.compile(expr, prettify=False))
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == res

    expected = 'SELECT t2.`id`, t2.`name`, t3.`hobbies` \n' \
               'FROM (\n' \
               '  SELECT * \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  WHERE t1.`id` < 10\n' \
               ') t2 \n' \
               'LATERAL VIEW EXPLODE(t2.`hobbies`) t3 AS `hobbies`'
    expr_if = exprs.expr4[exprs.expr4.id < 10]
    expr = expr_if[expr_if.id, expr_if.name, expr_if.hobbies.explode()]
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expected = 'SELECT * \n' \
               'FROM (\n' \
               '  SELECT t1.`id`, t1.`name`, t1.`hobbies`, t2.`relatives_key`, t2.`relatives_value` \n' \
               '  FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '  LATERAL VIEW EXPLODE(t1.`relatives`) t2 AS `relatives_key`, `relatives_value` \n' \
               ') t3 \n' \
               'WHERE t3.`relatives_key` IS NOT NULL'
    expr_explode = exprs.expr4[exprs.expr4.id, exprs.expr4.name, exprs.expr4.hobbies,
                              exprs.expr4.relatives.explode()]
    expr_if = expr_explode[expr_explode.relatives_key.notnull()]
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr_if, prettify=False))

    expected = 'SELECT t6.`id`, CONCAT(t6.`name`, t6.`relatives_value`) AS `nrv` \n' \
               'FROM (\n' \
               '  SELECT t4.`id`, t4.`name`, t4.`relatives_key`, t4.`relatives_value`, t5.`hobbies` \n' \
               '  FROM (\n' \
               '    SELECT * \n' \
               '    FROM (\n' \
               '      SELECT t1.`id`, t1.`name`, t1.`hobbies`, t2.`relatives_key`, t2.`relatives_value` \n' \
               '      FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '      LATERAL VIEW EXPLODE(t1.`relatives`) t2 AS `relatives_key`, `relatives_value` \n' \
               '    ) t3 \n' \
               '    WHERE t3.`relatives_key` IS NOT NULL\n' \
               '  ) t4 \n' \
               '  LATERAL VIEW EXPLODE(t4.`hobbies`) t5 AS `hobbies` \n' \
               ') t6 \n' \
               'WHERE LENGTH(t6.`hobbies`) > 5'
    expr_explode2 = expr_if[expr_if.exclude('hobbies'), expr_if.hobbies.explode()]
    expr = expr_explode2[expr_explode2.hobbies.len() > 5] \
        [expr_explode2.id, (expr_explode2.name + expr_explode2.relatives_value).rename('nrv')]
    assert to_text(expected) == to_text(ODPSEngine(odps).compile(expr, prettify=False))

    expr = exprs.expr4[exprs.expr4.id < 10][exprs.expr4.id, exprs.expr4.hobbies.explode('r_hobbies'),
                                          exprs.expr4.relatives.explode(['r_rel_key', 'r_rel_value']),
                                          Scalar(5).rename('five')]

    @output(expr.schema.names, expr.schema.types)
    def reducer(keys):
        def h(row):
            yield row
        return h

    expected = 'SELECT {0}(t6.`id`, t6.`r_hobbies`, t6.`r_rel_key`, t6.`r_rel_value`, t6.`five`) ' \
               'AS (`id`, `r_hobbies`, `r_rel_key`, `r_rel_value`, `five`) \n' \
               'FROM (\n' \
               '  SELECT * \n' \
               '  FROM (\n' \
               '    SELECT t2.`id`, t3.`r_hobbies`, t4.`r_rel_key`, t4.`r_rel_value`, 5 AS `five` \n' \
               '    FROM (\n' \
               '      SELECT * \n' \
               '      FROM mocked_project.`pyodps_test_expr_table` t1 \n' \
               '      WHERE t1.`id` < 10\n' \
               '    ) t2 \n' \
               '    LATERAL VIEW EXPLODE(t2.`hobbies`) t3 AS `r_hobbies` \n' \
               '    LATERAL VIEW EXPLODE(t2.`relatives`) t4 AS `r_rel_key`, `r_rel_value` \n' \
               '  ) t5 \n' \
               '  DISTRIBUTE BY t5.`id` \n' \
               '  SORT BY id\n' \
               ') t6'
    expr = expr.map_reduce(reducer=reducer, group='id')
    engine = ODPSEngine(odps)
    res = to_text(engine.compile(expr, prettify=False))
    fun_name = list(engine._ctx._registered_funcs.values())[0]
    assert to_text(expected.format(fun_name)) == res
