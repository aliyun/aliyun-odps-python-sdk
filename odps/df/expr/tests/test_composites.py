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

from odps.tests.core import TestBase
from odps import compat
from odps.df.types import validate_data_type
from odps.df.expr.tests.core import MockTable
from odps.df.expr.expressions import *
from odps.df.expr.composites import *


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = Schema.from_lists(['id', 'name', 'relatives', 'hobbies'],
                                   datatypes('int64', 'string', 'dict<string, string>', 'list<string>'))
        table = MockTable(name='pyodps_test_expr_table', schema=schema)

        self.expr = CollectionExpr(_source_data=table, _schema=schema)

    def testExplode(self):
        expr = self.expr.hobbies.explode()
        self.assertIsInstance(expr, RowAppliedCollectionExpr)
        self.assertIs(expr.input, self.expr)
        self.assertEqual(expr._func, 'EXPLODE')
        self.assertEqual(expr.dtypes.names, [self.expr.hobbies.name])
        self.assertEqual(expr.dtypes.types, [self.expr.hobbies.dtype.value_type])

        expr = self.expr.hobbies.explode('exploded')
        self.assertEqual(expr.dtypes.names, ['exploded'])

        self.assertRaises(ValueError, self.expr.hobbies.explode, ['abc', 'def'])

        expr = self.expr.hobbies.explode(pos=True)
        self.assertIsInstance(expr, RowAppliedCollectionExpr)
        self.assertIs(expr.input, self.expr)
        self.assertEqual(expr._func, 'POSEXPLODE')
        self.assertEqual(expr.dtypes.names,
                         [self.expr.hobbies.name + '_pos', self.expr.hobbies.name])
        self.assertEqual(expr.dtypes.types,
                         [validate_data_type('int64'), self.expr.hobbies.dtype.value_type])

        expr = self.expr.hobbies.explode(['pos', 'exploded'], pos=True)
        self.assertEqual(expr.dtypes.names, ['pos', 'exploded'])

        expr = self.expr.hobbies.explode('exploded', pos=True)
        self.assertEqual(expr.dtypes.names, ['exploded_pos', 'exploded'])

        expr = self.expr.relatives.explode()
        self.assertIsInstance(expr, RowAppliedCollectionExpr)
        self.assertIs(expr.input, self.expr)
        self.assertEqual(expr._func, 'EXPLODE')
        self.assertEqual(expr.dtypes.names,
                         [self.expr.relatives.name + '_key', self.expr.relatives.name + '_value'])
        self.assertEqual(expr.dtypes.types,
                         [self.expr.relatives.dtype.key_type, self.expr.relatives.dtype.value_type])

        expr = self.expr.relatives.explode(['k', 'v'])
        self.assertEqual(expr.dtypes.names, ['k', 'v'])

        self.assertRaises(ValueError, self.expr.relatives.explode, ['abc'])
        self.assertRaises(ValueError, self.expr.relatives.explode, ['abc'], pos=True)

    def testListMethods(self):
        expr = self.expr.hobbies[0]
        self.assertIsInstance(expr, ListDictGetItem)
        self.assertIsInstance(expr, StringSequenceExpr)
        self.assertEqual(expr.dtype, validate_data_type('string'))

        expr = self.expr.hobbies.len()
        self.assertIsInstance(expr, ListDictLength)
        self.assertIsInstance(expr, Int64SequenceExpr)

        expr = self.expr.hobbies.sort()
        self.assertIsInstance(expr, ListSort)
        self.assertIsInstance(expr, ListSequenceExpr)
        self.assertEqual(expr.dtype, validate_data_type('list<string>'))

        expr = self.expr.hobbies.contains('yacht')
        self.assertIsInstance(expr, ListContains)
        self.assertIsInstance(expr, BooleanSequenceExpr)

    def testDictMethods(self):
        expr = self.expr.relatives['abc']
        self.assertIsInstance(expr, ListDictGetItem)
        self.assertIsInstance(expr, StringSequenceExpr)
        self.assertEqual(expr.dtype, validate_data_type('string'))

        expr = self.expr.relatives.len()
        self.assertIsInstance(expr, ListDictLength)
        self.assertIsInstance(expr, Int64SequenceExpr)

        expr = self.expr.relatives.keys()
        self.assertIsInstance(expr, DictKeys)
        self.assertIsInstance(expr, ListSequenceExpr)
        self.assertEqual(expr.dtype, validate_data_type('list<string>'))

        expr = self.expr.relatives.values()
        self.assertIsInstance(expr, DictValues)
        self.assertIsInstance(expr, ListSequenceExpr)
        self.assertEqual(expr.dtype, validate_data_type('list<string>'))

    def testBuilders(self):
        expr = make_list(1, 2, 3, 4)
        self.assertIsInstance(expr, ListBuilder)
        self.assertIsInstance(expr, ListScalar)
        self.assertEqual(expr.dtype, validate_data_type('list<int32>'))

        expr = make_list(1, 2, 3, self.expr.id)
        self.assertIsInstance(expr, ListBuilder)
        self.assertIsInstance(expr, ListSequenceExpr)
        self.assertEqual(expr.dtype, validate_data_type('list<int64>'))

        self.assertRaises(TypeError, make_list, 1, 2, 'str', type='int32')
        self.assertRaises(TypeError, make_list, 1, 2, 'str')
        expr = make_list(1, 2, 3, 4, type='int64')
        self.assertEqual(expr.dtype, validate_data_type('list<int64>'))
        expr = make_list(1.1, 2.2, 3.3, 4.4)
        self.assertEqual(expr.dtype, validate_data_type('list<float64>'))
        expr = make_list(1, 2, 3, 65535)
        self.assertEqual(expr.dtype, validate_data_type('list<int32>'))
        expr = make_list(1, 2, 3, compat.long_type(12345678910))
        self.assertEqual(expr.dtype, validate_data_type('list<int64>'))
        expr = make_list(1, 2, 3, 3.5)
        self.assertEqual(expr.dtype, validate_data_type('list<float64>'))

        self.assertRaises(ValueError, make_dict, 1, 2, 3)

        expr = make_dict(1, 2, 3, 4)
        self.assertIsInstance(expr, DictBuilder)
        self.assertIsInstance(expr, DictScalar)
        self.assertEqual(expr.dtype, validate_data_type('dict<int32,int32>'))

        expr = make_dict(1, 2, 3, 4, key_type='int16', value_type='int64')
        self.assertIsInstance(expr, DictBuilder)
        self.assertIsInstance(expr, DictScalar)
        self.assertEqual(expr.dtype, validate_data_type('dict<int16,int64>'))

        expr = make_dict(1, 2, 3, self.expr.id)
        self.assertIsInstance(expr, DictBuilder)
        self.assertIsInstance(expr, DictSequenceExpr)
        self.assertEqual(expr.dtype, validate_data_type('dict<int32,int64>'))
