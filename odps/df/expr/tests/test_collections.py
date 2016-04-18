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


from odps.tests.core import TestBase
from odps.compat import unittest
from odps.models import Schema
from odps.df.expr.expressions import *
from odps.df import output


class Test(TestBase):
    def setup(self):
        schema = Schema.from_lists(types._data_types.keys(), types._data_types.values())
        self.expr = CollectionExpr(_source_data=None, _schema=schema)

    def testSort(self):
        sorted_expr = self.expr.sort(self.expr.int64)
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [True])

        sorted_expr = self.expr.sort_values([self.expr.float32, 'string'])
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [True] * 2)

        sorted_expr = self.expr.sort([self.expr.decimal, 'boolean', 'string'], ascending=False)
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [False] * 3)

        sorted_expr = self.expr.sort([self.expr.int8, 'datetime', 'float64'],
                                     ascending=[False, True, False])
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [False, True, False])

        sorted_expr = self.expr.sort([-self.expr.int8, 'datetime', 'float64'])
        self.assertIsInstance(sorted_expr, CollectionExpr)
        self.assertEqual(sorted_expr._schema, self.expr._schema)
        self.assertSequenceEqual(sorted_expr._ascending, [False, True, True])

    def testDistinct(self):
        distinct = self.expr.distinct()
        self.assertIsInstance(distinct, CollectionExpr)
        self.assertEqual(distinct._schema, self.expr._schema)

        distinct = self.expr.distinct(self.expr.string)
        self.assertIsInstance(distinct, CollectionExpr)
        self.assertEqual(distinct._schema, self.expr[[self.expr.string]]._schema)

        distinct = self.expr.distinct([self.expr.boolean, 'decimal'])
        self.assertIsInstance(distinct, CollectionExpr)
        self.assertEqual(distinct._schema, self.expr[['boolean', 'decimal']]._schema)

        self.assertRaises(ExpressionError, lambda: self.expr['boolean', self.expr.string.unique()])

    def testMapReduce(self):
        @output(['id', 'name', 'rating'], ['int', 'string', 'int'])
        def mapper(row):
            yield row.int64, row.string, row.int32

        @output(['name', 'rating'], ['string', 'int'])
        def reducer(_):
            i = [0]
            def h(row):
                if i[0] <= 1:
                    yield row.name, row.rating
            return h

        expr = self.expr.map_reduce(mapper, reducer, group='name',
                                    sort='rating', ascending=False)
        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr._sort_fields), 2)
        self.assertTrue(expr._sort_fields[0]._ascending)
        self.assertFalse(expr._sort_fields[1]._ascending)

        expr = self.expr.map_reduce(mapper, reducer, group=lambda x: x.name,
                                    sort=lambda x: -x.rating)
        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr._sort_fields), 2)
        self.assertTrue(expr._sort_fields[0]._ascending)
        self.assertFalse(expr._sort_fields[1]._ascending)

        expr = self.expr.map_reduce(mapper, reducer, group='name',
                                    sort=['rating', 'id'], ascending=[False, True])

        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr._sort_fields), 3)
        self.assertTrue(expr._sort_fields[0]._ascending)
        self.assertFalse(expr._sort_fields[1]._ascending)
        self.assertTrue(expr._sort_fields[2]._ascending)

        expr = self.expr.map_reduce(mapper, reducer, group='name',
                                    sort=['rating', 'id'], ascending=False)

        self.assertEqual(expr.schema.names, ['name', 'rating'])
        self.assertEqual(len(expr._sort_fields), 3)
        self.assertTrue(expr._sort_fields[0]._ascending)
        self.assertFalse(expr._sort_fields[1]._ascending)
        self.assertFalse(expr._sort_fields[2]._ascending)


if __name__ == '__main__':
    unittest.main()
