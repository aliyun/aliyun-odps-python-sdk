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

from collections import namedtuple

import pytest

from .....tests.core import numpy_case
from .....compat import six
from .....models import TableSchema
from .....udf.tools import runners
from ....types import validate_data_type
from ....expr.expressions import CollectionExpr
from ....expr.tests.core import MockTable
from ..engine import ODPSSQLEngine, UDF_CLASS_NAME


class ODPSEngine(ODPSSQLEngine):

    def compile(self, expr, prettify=True, libraries=None):
        expr = self._convert_table(expr)
        expr_dag = expr.to_dag()
        self._analyze(expr_dag, expr)
        new_expr = self._rewrite(expr_dag)
        return self._compile(new_expr, prettify=prettify, libraries=libraries)


def get_function(source, fun_name):
    d = dict()
    six.exec_(source, d, d)
    return d[fun_name]


@pytest.fixture
def setup(odps):
    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(
        ['name', 'id', 'fid'], datatypes('string', 'int64', 'float64')
    )

    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)

    expr = CollectionExpr(_source_data=table, _schema=schema)
    engine = ODPSEngine(odps)
    nt = namedtuple("NT", "expr engine")
    return nt(expr, engine)


def test_simple_lambda(setup):
    setup.engine.compile(setup.expr.id.map(lambda x: x + 1))
    udf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udf = get_function(udf, UDF_CLASS_NAME)
    assert [4] == list(runners.simple_run(udf, [(3, ), ]))


def test_simple_function(setup):
    def my_func(x):
        if x < 0:
            return -1
        elif x == 0:
            return 0
        else:
            return 1

    setup.engine.compile(setup.expr.id.map(my_func))
    udf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udf = get_function(udf, UDF_CLASS_NAME)
    assert [-1, 0, 1] == list(runners.simple_run(udf, [(-3, ), (0, ), (5, )]))


@numpy_case
def test_simple_numpy_function(setup):
    import numpy as np

    def my_func(x):
        if x < 0:
            return np.int32(-1)
        elif x == 0:
            return np.int32(0)
        else:
            return np.int32(1)

    setup.engine.compile(setup.expr.id.map(my_func))
    udf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udf = get_function(udf, UDF_CLASS_NAME)
    run_result = runners.simple_run(udf, [(-3, ), (0, ), (5, )])
    assert [-1, 0, 1] == list(run_result)
    assert any(isinstance(v, np.generic) for v in run_result) is False


def test_nest_function(setup):
    def my_func(x):
        def inner(y):
            if y < 0:
                return -2
            elif y == 0:
                return 0
            else:
                return 2
        return inner(x)

    setup.engine.compile(setup.expr.id.map(my_func))
    udf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udf = get_function(udf, UDF_CLASS_NAME)
    assert [-2, 0, 2] == list(runners.simple_run(udf, [(-3, ), (0, ), (5, )]))


def test_duplicate_function(setup):
    def gen_no_closure_func():
        def inner(x):
            return x + 1

        return inner

    def gen_func_with_closure():
        incr = 0

        def inner(x):
            return x + incr

        return inner

    expr = setup.expr[
        setup.expr.id.map(gen_no_closure_func()).rename("id1"),
        setup.expr.id.map(gen_no_closure_func()).rename("id2"),
        setup.expr.id.map(gen_func_with_closure()).rename("id3"),
        setup.expr.id.map(gen_func_with_closure()).rename("id4"),
    ]
    setup.engine.compile(expr)
    assert 3 == len(setup.engine._ctx._func_to_udfs.values())


def test_global_var_function(setup):
    global_val = 10

    def my_func(x):
        if x < global_val:
            return -1
        elif x == global_val:
            return 0
        else:
            return 1

    setup.engine.compile(setup.expr.id.map(my_func))
    udf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udf = get_function(udf, UDF_CLASS_NAME)
    assert [-1, 0, 1] == list(runners.simple_run(udf, [(-9, ), (10, ), (15, )]))


def test_ref_func_function(setup):
    global_val = 10

    def my_func1(x):
        if x < global_val:
            return -1
        elif x == global_val:
            return 0
        else:
            return 1

    def my_func(y):
        return my_func1(y)

    setup.engine.compile(setup.expr.id.map(my_func))
    udf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udf = get_function(udf, UDF_CLASS_NAME)
    assert [-1, 0, 1] == list(runners.simple_run(udf, [(-9, ), (10, ), (15, )]))


def test_apply_to_sequence_funtion(setup):
    def my_func(row):
        return row.name + str(row.id)

    setup.engine.compile(setup.expr.apply(my_func, axis=1, reduce=True).rename('test'))
    udf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udf = get_function(udf, UDF_CLASS_NAME)
    assert ['name1', 'name2'] == runners.simple_run(udf, [('name', 1, None), ('name', 2, None)])


def test_apply_function(setup):
    def my_func(row):
        return row.name, row.id

    setup.engine.compile(setup.expr.apply(my_func, axis=1, names=['name', 'id'], types=['string', 'int']))
    udtf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udtf = get_function(udtf, UDF_CLASS_NAME)
    assert [('name1', 1), ('name2', 2)] == runners.simple_run(udtf, [('name1', 1, None), ('name2', 2, None)])


@numpy_case
def test_apply_numpy_function(setup):
    import numpy as np

    def my_func(row):
        return row.name, np.int32(row.id)

    setup.engine.compile(setup.expr.apply(my_func, axis=1, names=['name', 'id'], types=['string', 'int']))
    udtf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udtf = get_function(udtf, UDF_CLASS_NAME)
    run_result = runners.simple_run(udtf, [('name1', 1, None), ('name2', 2, None)])
    assert [('name1', 1), ('name2', 2)] == run_result


def test_apply_generator_function(setup):
    def my_func(row):
        for n in row.name.split(','):
            yield n

    setup.engine.compile(setup.expr.apply(my_func, axis=1, names='name'))
    udtf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udtf = get_function(udtf, UDF_CLASS_NAME)
    assert ['name1', 'name2', 'name3', 'name4'] == runners.simple_run(udtf, [('name1,name2', 1, None), ('name3,name4', 2, None)])


def test_agg_function(setup):
    class Agg(object):
        def buffer(self):
            return ['']

        def __call__(self, buffer, val):
            if not buffer[0]:
                buffer[0] = val
            else:
                buffer[0] += ',' + val

        def merge(self, buffer, pbuffer):
            if not pbuffer[0]:
                return
            elif not buffer[0]:
                buffer[0] = pbuffer[0]
            else:
                buffer[0] += ',' + pbuffer[0]

        def getvalue(self, buffer):
            return buffer[0]

    setup.engine.compile(setup.expr.name.agg(Agg))
    udaf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udaf = get_function(udaf, UDF_CLASS_NAME)
    assert ['name1,name2,name3,name4'] == list(runners.simple_run(udaf, [('name1,name2',), ('name3,name4',)]))


@numpy_case
def test_agg_numpy_function(setup):
    import numpy as np

    class Agg(object):
        def buffer(self):
            return [np.int32(1)]

        def __call__(self, buffer, val):
            buffer[0] *= val

        def merge(self, buffer, pbuffer):
            buffer[0] *= pbuffer[0]

        def getvalue(self, buffer):
            return buffer[0]

    setup.engine.compile(setup.expr.id.agg(Agg))
    udaf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udaf = get_function(udaf, UDF_CLASS_NAME)
    result = runners.simple_run(udaf, [(3,), (6,), (5,)])
    assert [90] == result
    assert not isinstance(result, np.generic)


def test_bizarre_field(setup):
    def my_func(row):
        return getattr(row, '012') * 2.0

    datatypes = lambda *types: [validate_data_type(t) for t in types]
    schema = TableSchema.from_lists(['name', 'id', 'fid', '012'],
                               datatypes('string', 'int64', 'float64', 'float64'))

    table = MockTable(name='pyodps_test_expr_table', table_schema=schema)
    expr = CollectionExpr(_source_data=table, _schema=schema)

    setup.engine.compile(expr.apply(my_func, axis=1, names=['out_col'], types=['float64']))
    udtf = list(setup.engine._ctx._func_to_udfs.values())[0]
    udtf = get_function(udtf, UDF_CLASS_NAME)
    assert [20, 40] == runners.simple_run(udtf, [('name1', 1, None, 10), ('name2', 2, None, 20)])
