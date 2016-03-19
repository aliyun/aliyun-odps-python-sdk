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

import base64  # noqa
from collections import namedtuple  # noqa
import inspect  # noqa

from odps.tests.core import TestBase
from odps.compat import unittest
from odps.models import Schema
from odps.udf.tools import runners
from odps.df.types import validate_data_type
from odps.df.backends.odpssql.engine import ODPSEngine, UDF_CLASS_NAME
from odps.df.expr.expressions import CollectionExpr
from odps.df.expr.tests.core import MockTable
from odps.df.backends.odpssql.cloudpickle import * # noqa


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
        exec(udf)
        udf = locals()[UDF_CLASS_NAME]
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
        exec(udf)
        udf = locals()[UDF_CLASS_NAME]
        self.assertSequenceEqual([-1, 0, 1], runners.simple_run(udf, [(-3, ), (0, ), (5, )]))

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
        exec(udf, globals(), locals())
        udf = locals()[UDF_CLASS_NAME]
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
        exec(udf, globals(), locals())
        udf = locals()[UDF_CLASS_NAME]
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
        exec(udf, globals(), locals())
        udf = locals()[UDF_CLASS_NAME]
        self.assertSequenceEqual([-1, 0, 1], runners.simple_run(udf, [(-9, ), (10, ), (15, )]))

    def testApplyToSequenceFuntion(self):
        def my_func(row):
            return row.name + str(row.id)

        self.engine.compile(self.expr.apply(my_func, axis=1, reduce=True).rename('test'))
        udf = list(self.engine._ctx._func_to_udfs.values())[0]
        exec(udf, globals(), locals())
        udf = locals()[UDF_CLASS_NAME]
        self.assertEqual(['name1', 'name2'],
                         runners.simple_run(udf, [('name', 1, None), ('name', 2, None)]))

    def testApplyFunction(self):
        def my_func(row):
            return row.name, row.id

        self.engine.compile(self.expr.apply(my_func, axis=1, names=['name', 'id'], types=['string', 'int']))
        udtf = list(self.engine._ctx._func_to_udfs.values())[0]
        exec(udtf, globals(), locals())
        udtf = locals()[UDF_CLASS_NAME]
        self.assertEqual([('name1', 1), ('name2', 2)],
                          runners.simple_run(udtf, [('name1', 1, None), ('name2', 2, None)]))

    def testApplyGeneratorFunction(self):
        def my_func(row):
            for n in row.name.split(','):
                yield n

        self.engine.compile(self.expr.apply(my_func, axis=1, names='name'))
        udtf = list(self.engine._ctx._func_to_udfs.values())[0]
        exec(udtf, globals(), locals())
        udtf = locals()[UDF_CLASS_NAME]
        self.assertEqual(['name1', 'name2', 'name3', 'name4'],
                         runners.simple_run(udtf, [('name1,name2', 1, None), ('name3,name4', 2, None)]))

if __name__ == '__main__':
    unittest.main()
