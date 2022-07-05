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

from odps.tests.core import TestBase, numpy_case
from odps.compat import unittest, six, PY26
from odps.models import Schema
from odps.udf.tools import runners
from odps.df.types import validate_data_type
from odps.df.backends.odpssql.engine import ODPSSQLEngine, UDF_CLASS_NAME
from odps.df.expr.expressions import CollectionExpr
from odps.df.expr.tests.core import MockTable


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


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid'],
                                    datatypes('string', 'int64', 'float64'))

        table = MockTable(name='pyodps_test_expr_table', schema=schema)

        self.expr = CollectionExpr(_source_data=table, _schema=schema)

        self.engine = ODPSEngine(self.odps)

    def testSimpleLambda(self):
        self.engine.compile(self.expr.id.map(lambda x: x + 1))
        udf = list(self.engine._ctx._func_to_udfs.values())[0]
        udf = get_function(udf, UDF_CLASS_NAME)
        self.assertSequenceEqual([4, ], runners.simple_run(udf, [(3, ), ]))

    def testSimpleFunction(self):
        def my_func(x):
            if x < 0:
                return -1
            elif x == 0:
                return 0
            else:
                return 1

        self.engine.compile(self.expr.id.map(my_func))
        udf = list(self.engine._ctx._func_to_udfs.values())[0]
        udf = get_function(udf, UDF_CLASS_NAME)
        self.assertSequenceEqual([-1, 0, 1], runners.simple_run(udf, [(-3, ), (0, ), (5, )]))

    @numpy_case
    def testSimpleNumpyFunction(self):
        import numpy as np

        def my_func(x):
            if x < 0:
                return np.int32(-1)
            elif x == 0:
                return np.int32(0)
            else:
                return np.int32(1)

        self.engine.compile(self.expr.id.map(my_func))
        udf = list(self.engine._ctx._func_to_udfs.values())[0]
        udf = get_function(udf, UDF_CLASS_NAME)
        run_result = runners.simple_run(udf, [(-3, ), (0, ), (5, )])
        self.assertSequenceEqual([-1, 0, 1], run_result)
        self.assertFalse(any(isinstance(v, np.generic) for v in run_result))

    def testNestFunction(self):
        def my_func(x):
            def inner(y):
                if y < 0:
                    return -2
                elif y == 0:
                    return 0
                else:
                    return 2
            return inner(x)

        self.engine.compile(self.expr.id.map(my_func))
        udf = list(self.engine._ctx._func_to_udfs.values())[0]
        udf = get_function(udf, UDF_CLASS_NAME)
        self.assertSequenceEqual([-2, 0, 2], runners.simple_run(udf, [(-3, ), (0, ), (5, )]))

    def testGlobalVarFunction(self):
        global_val = 10

        def my_func(x):
            if x < global_val:
                return -1
            elif x == global_val:
                return 0
            else:
                return 1

        self.engine.compile(self.expr.id.map(my_func))
        udf = list(self.engine._ctx._func_to_udfs.values())[0]
        udf = get_function(udf, UDF_CLASS_NAME)
        self.assertSequenceEqual([-1, 0, 1], runners.simple_run(udf, [(-9, ), (10, ), (15, )]))

    def testRefFuncFunction(self):
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

        self.engine.compile(self.expr.id.map(my_func))
        udf = list(self.engine._ctx._func_to_udfs.values())[0]
        udf = get_function(udf, UDF_CLASS_NAME)
        self.assertSequenceEqual([-1, 0, 1], runners.simple_run(udf, [(-9, ), (10, ), (15, )]))

    def testApplyToSequenceFuntion(self):
        def my_func(row):
            return row.name + str(row.id)

        self.engine.compile(self.expr.apply(my_func, axis=1, reduce=True).rename('test'))
        udf = list(self.engine._ctx._func_to_udfs.values())[0]
        udf = get_function(udf, UDF_CLASS_NAME)
        self.assertEqual(['name1', 'name2'],
                         runners.simple_run(udf, [('name', 1, None), ('name', 2, None)]))

    def testApplyFunction(self):
        def my_func(row):
            return row.name, row.id

        self.engine.compile(self.expr.apply(my_func, axis=1, names=['name', 'id'], types=['string', 'int']))
        udtf = list(self.engine._ctx._func_to_udfs.values())[0]
        udtf = get_function(udtf, UDF_CLASS_NAME)
        self.assertEqual([('name1', 1), ('name2', 2)],
                          runners.simple_run(udtf, [('name1', 1, None), ('name2', 2, None)]))

    @numpy_case
    def testApplyNumpyFunction(self):
        import numpy as np

        def my_func(row):
            return row.name, np.int32(row.id)

        self.engine.compile(self.expr.apply(my_func, axis=1, names=['name', 'id'], types=['string', 'int']))
        udtf = list(self.engine._ctx._func_to_udfs.values())[0]
        udtf = get_function(udtf, UDF_CLASS_NAME)
        run_result = runners.simple_run(udtf, [('name1', 1, None), ('name2', 2, None)])
        self.assertEqual([('name1', 1), ('name2', 2)], run_result)

    def testApplyGeneratorFunction(self):
        def my_func(row):
            for n in row.name.split(','):
                yield n

        self.engine.compile(self.expr.apply(my_func, axis=1, names='name'))
        udtf = list(self.engine._ctx._func_to_udfs.values())[0]
        udtf = get_function(udtf, UDF_CLASS_NAME)
        self.assertEqual(['name1', 'name2', 'name3', 'name4'],
                         runners.simple_run(udtf, [('name1,name2', 1, None), ('name3,name4', 2, None)]))

    def testAggFunction(self):
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

        self.engine.compile(self.expr.name.agg(Agg))
        udaf = list(self.engine._ctx._func_to_udfs.values())[0]
        udaf = get_function(udaf, UDF_CLASS_NAME)
        self.assertEqual(['name1,name2,name3,name4'],
                         runners.simple_run(udaf, [('name1,name2',), ('name3,name4',)]))

    @numpy_case
    def testAggNumpyFunction(self):
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

        self.engine.compile(self.expr.id.agg(Agg))
        udaf = list(self.engine._ctx._func_to_udfs.values())[0]
        udaf = get_function(udaf, UDF_CLASS_NAME)
        result = runners.simple_run(udaf, [(3,), (6,), (5,)])
        self.assertEqual([90], result)
        self.assertNotIsInstance(result, np.generic)

    @unittest.skipIf(PY26, 'Ignored under Python 2.6')
    def testBizarreField(self):
        def my_func(row):
            return getattr(row, '012') * 2.0

        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['name', 'id', 'fid', '012'],
                                   datatypes('string', 'int64', 'float64', 'float64'))

        table = MockTable(name='pyodps_test_expr_table', schema=schema)
        expr = CollectionExpr(_source_data=table, _schema=schema)

        self.engine.compile(expr.apply(my_func, axis=1, names=['out_col'], types=['float64']))
        udtf = list(self.engine._ctx._func_to_udfs.values())[0]
        udtf = get_function(udtf, UDF_CLASS_NAME)
        self.assertEqual([20, 40],
                         runners.simple_run(udtf, [('name1', 1, None, 10), ('name2', 2, None, 20)]))


if __name__ == '__main__':
    unittest.main()
