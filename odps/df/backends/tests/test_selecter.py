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

import uuid

from odps.df.backends.tests.core import TestBase, tn
from odps.tests.core import sqlalchemy_case
from odps.models import TableSchema
from odps import types, options
from odps.df.types import validate_data_type
from odps.df.backends.odpssql.types import df_schema_to_odps_schema
from odps.df.expr.expressions import CollectionExpr
from odps.df.backends.odpssql.types import odps_schema_to_df_schema
from odps.df.backends.selecter import EngineSelecter
from odps.df.backends.core import EngineTypes as Engines


class Test(TestBase):
    def setup(self):
        datatypes = lambda *types: [validate_data_type(t) for t in types]
        schema = TableSchema.from_lists(['name', 'id', 'fid', 'isMale', 'scale', 'birth'],
                                   datatypes('string', 'int64', 'float64', 'boolean', 'decimal', 'datetime'))
        self.schema = df_schema_to_odps_schema(schema)
        table_name = tn('pyodps_test_selecter_table_%s' % str(uuid.uuid4()).replace('-', '_'))
        self.odps.delete_table(table_name, if_exists=True)
        self.table = self.odps.create_table(name=table_name, table_schema=self.schema)
        self.expr = CollectionExpr(_source_data=self.table, _schema=schema)

        class FakeBar(object):
            def update(self, *args, **kwargs):
                pass

            def inc(self, *args, **kwargs):
                pass

            def status(self, *args, **kwargs):
                pass
        self.faked_bar = FakeBar()

        data = [
            ['name1', 4, 5.3, None, None, None],
            ['name2', 2, 3.5, None, None, None],
            ['name1', 4, 4.2, None, None, None],
            ['name1', 3, 2.2, None, None, None],
            ['name1', 3, 4.1, None, None, None],
        ]

        schema2 = TableSchema.from_lists(['name', 'id2', 'id3'],
                                    [types.string, types.bigint, types.bigint])

        table_name = tn('pyodps_test_selecter_table2')
        self.odps.delete_table(table_name, if_exists=True)
        table2 = self.odps.create_table(name=table_name, table_schema=schema2)
        self.expr2 = CollectionExpr(_source_data=table2, _schema=odps_schema_to_df_schema(schema2))

        self._gen_data(data=data)

        data2 = [
            ['name1', 4, -1],
            ['name2', 1, -2]
        ]

        self.odps.write_table(table2, 0, data2)

        self.selecter = EngineSelecter()

    def _gen_data(self, rows=None, data=None, nullable_field=None, value_range=None):
        if data is None:
            data = []
            for _ in range(rows):
                record = []
                for t in self.schema.types:
                    method = getattr(self, '_gen_random_%s' % t.name)
                    if t.name == 'bigint':
                        record.append(method(value_range=value_range))
                    else:
                        record.append(method())
                data.append(record)

            if nullable_field is not None:
                j = self.schema._name_indexes[nullable_field]
                for i, l in enumerate(data):
                    if i % 2 == 0:
                        data[i][j] = None

        self.odps.write_table(self.table, 0, data)
        return data

    @sqlalchemy_case
    def testSelecter(self):
        src_max_size = options.df.seahawks.max_size
        src_seahawks_url = options.seahawks_url
        try:
            if not options.seahawks_url:
                options.seahawks_url = 'fake url'

            self.assertEqual(self.selecter.select(self.expr.to_dag(copy=False)), Engines.SEAHAWKS)

            expr = self.expr.join(self.expr2)
            self.assertEqual(self.selecter.select(expr.to_dag(copy=False)), Engines.SEAHAWKS)

            options.df.seahawks.max_size = self.table.size - 1

            self.assertEqual(self.selecter.select(self.expr.to_dag(copy=False)), Engines.ODPS)
            self.assertEqual(self.selecter.select(expr.to_dag(copy=False)), Engines.ODPS)
        finally:
            if options.seahawks_url == 'fake url':
                options.seahawks_url = src_seahawks_url
            options.df.seahawks.max_size = src_max_size
