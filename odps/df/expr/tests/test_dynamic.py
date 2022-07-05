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

from odps.tests.core import TestBase
from odps.compat import unittest
from odps.models import Schema
from odps.df.types import validate_data_type
from odps.df.expr.tests.core import MockTable
from odps.df.expr.expressions import StringScalar
from odps.df.expr.dynamic import *


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = DynamicSchema.from_schema(
            Schema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                              datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        )
        table = MockTable(name='pyodps_test_expr_table', schema=schema)

        schema2 = DynamicSchema.from_schema(
            Schema.from_lists(['name2', 'id', 'fid2'],
                              datatypes('string', 'int64', 'float64')),
            default_type=types.string
        )
        table2 = MockTable(name='pyodps_test_expr_tabl2', schema=schema2)

        self.expr = DynamicCollectionExpr(_source_data=table, _schema=schema)
        self.expr2 = DynamicCollectionExpr(_source_data=table2, _schema=schema2)

    def testDynamic(self):
        df = self.expr.distinct('name', 'id')
        self.assertNotIsInstance(df, DynamicMixin)
        self.assertNotIsInstance(df._schema, DynamicSchema)

        # the sequence must be definite, no need for generating dynamic sequence
        self.assertNotIsInstance(self.expr['name'], DynamicMixin)
        # a field which does not exist is fine
        self.assertIsInstance(self.expr['not_exist'], DynamicMixin)
        # df's schema is not dynamic
        self.assertRaises(ValueError, lambda: df['non_exist'])

        df = self.expr.distinct('name', 'non_exist')
        self.assertIsInstance(df, DynamicMixin)
        self.assertNotIsInstance(df._schema, DynamicSchema)

        df = self.expr.distinct()
        self.assertIsInstance(df, DynamicMixin)
        self.assertIsInstance(df._schema, DynamicSchema)

        df2 = self.expr2.distinct('name2', 'not_exist')
        self.assertNotIsInstance(df2, DynamicMixin)
        self.assertNotIsInstance(df2._schema, DynamicSchema)

        # the sequence must be definite, no need for generating dynamic sequence
        self.assertNotIsInstance(df2['name2'], DynamicMixin)
        # a field which does not exist is fine
        self.assertNotIsInstance(df2['not_exist'], DynamicMixin)
        # df2's schema is not dynamic
        self.assertRaises(ValueError, lambda: df2['non_exist'])

        self.assertEqual(self.expr2['non_exist'].dtype, types.string)
        self.assertIsInstance(self.expr2['non_exist'].sum(), StringScalar)

        # projection
        df3 = self.expr2[self.expr2, self.expr2.id.astype('string').rename('id2')]
        self.assertIsInstance(df3, DynamicMixin)
        self.assertIsInstance(df3._schema, DynamicSchema)
        # non_exist need to be checked
        self.assertIsInstance(df3['non_exist'], DynamicMixin)
        self.assertNotIsInstance(self.expr['id', 'name2']._schema, DynamicSchema)

        # filter
        df4 = self.expr2.filter(self.expr2.id < 10)
        self.assertIsInstance(df4, DynamicMixin)
        self.assertIsInstance(df4._schema, DynamicSchema)

        # slice
        df5 = self.expr2[2:4]
        self.assertIsInstance(df5, DynamicMixin)
        self.assertIsInstance(df5._schema, DynamicSchema)

        # sort
        df6 = self.expr2.sort('id')
        self.assertIsInstance(df6, DynamicMixin)
        self.assertIsInstance(df6._schema, DynamicSchema)

        # apply
        df7 = self.expr2.apply(lambda row: row, axis=1, names=self.expr2.schema.names)
        self.assertNotIsInstance(df7, DynamicMixin)
        self.assertNotIsInstance(df7._schema, DynamicSchema)

        # sample
        df8 = self.expr2.sample(parts=10)
        self.assertIsInstance(df8, DynamicMixin)
        self.assertIsInstance(df8._schema, DynamicSchema)

        # groupby
        df9 = self.expr2.groupby('id').agg(self.expr2['name3'].sum())
        self.assertNotIsInstance(df9, DynamicMixin)
        self.assertNotIsInstance(df9._schema, DynamicSchema)
        df10 = self.expr.groupby('id2').agg(self.expr.name.sum())
        self.assertNotIsInstance(df10, DynamicMixin)
        self.assertNotIsInstance(df10._schema, DynamicSchema)

        # mutate
        df11 = self.expr2.groupby('id').mutate(id2=lambda x: x.id.cumsum())
        self.assertNotIsInstance(df11, DynamicMixin)
        self.assertNotIsInstance(df11._schema, DynamicSchema)
        self.expr.groupby('id').sort('id').non_exist.astype('int').cumsum()

        # join
        df12 = self.expr.join(self.expr2)[self.expr, self.expr2['id2']]
        self.assertIsInstance(df12, DynamicMixin)
        self.assertIsInstance(df12._schema, DynamicSchema)
        self.assertIsInstance(df12.input, DynamicMixin)
        self.assertIsInstance(df12.input._schema, DynamicSchema)
        df13 = self.expr.join(self.expr2)[self.expr.id, self.expr2.name2]
        self.assertNotIsInstance(df13, DynamicMixin)
        self.assertNotIsInstance(df13._schema, DynamicSchema)

        # union
        df14 = self.expr['id', self.expr.name.rename('name2'), self.expr.fid.rename('fid2')]\
            .union(self.expr2)
        self.assertNotIsInstance(df14, DynamicMixin)
        self.assertNotIsInstance(df14._schema, DynamicSchema)

if __name__ == '__main__':
    unittest.main()