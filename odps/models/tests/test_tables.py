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

from datetime import datetime
import itertools

import six

from odps.tests.core import TestBase
from odps.compat import unittest
from odps.models import Schema, Record


class Test(TestBase):

    def testTables(self):
        self.assertIs(self.odps.get_project().tables, self.odps.get_project().tables)
        size = len(list(itertools.islice(self.odps.list_tables(), 0, 200)))
        self.assertGreaterEqual(size, 0)

        if size > 0:
            table = next(iter(self.odps.list_tables()))

            tables = list(self.odps.list_tables(prefix=table.name))
            self.assertGreaterEqual(len(tables), 1)
            for t in tables:
                self.assertTrue(t.name.startswith(table.name))

            tables = list(self.odps.list_tables(owner=table.owner))
            self.assertGreaterEqual(len(tables), 1)
            for t in tables:
                self.assertEqual(t.owner, table.owner)

    def testTableExists(self):
        non_exists_table = 'a_non_exists_table'
        self.assertFalse(self.odps.exist_table(non_exists_table))

    def testTable(self):
        tables = self.odps.list_tables()
        table = next(tables)

        self.assertIs(table, self.odps.get_table(table.name))

        self.assertIsNone(table._getattr('format'))
        self.assertIsNone(table._getattr('schema'))
        self.assertIsNone(table._getattr('comment'))
        self.assertIsNotNone(table._getattr('owner'))
        self.assertIsNone(table._getattr('table_label'))
        self.assertIsNone(table._getattr('creation_time'))
        self.assertIsNone(table._getattr('last_modified_time'))
        self.assertIsNone(table._getattr('last_meta_modified_time'))
        self.assertIsNone(table._getattr('is_virtual_view'))
        self.assertIsNone(table._getattr('lifecycle'))
        self.assertIsNone(table._getattr('view_text'))
        self.assertIsNone(table._getattr('size'))
        self.assertIsNone(table._getattr('is_archived'))
        self.assertIsNone(table._getattr('physical_size'))
        self.assertIsNone(table._getattr('file_num'))

        self.assertFalse(table.is_loaded)
        self.assertFalse(table._is_extend_info_loaded)

        self.assertIsInstance(table.is_archived, bool)
        self.assertGreaterEqual(table.physical_size, 0)
        self.assertGreaterEqual(table.file_num, 0)
        self.assertIsInstance(table.schema, Schema)
        self.assertGreater(len(table.schema.get_columns()), 0)
        self.assertGreaterEqual(len(table.schema.get_partitions()), 0)
        self.assertIsInstance(table.owner, six.string_types)
        self.assertIsInstance(table.table_label, six.string_types)
        self.assertIsInstance(table.creation_time, datetime)
        self.assertIsInstance(table.last_modified_time, datetime)
        self.assertIsInstance(table.last_meta_modified_time, datetime)
        self.assertIsInstance(table.is_virtual_view, bool)
        self.assertGreaterEqual(table.lifecycle, -1)
        self.assertGreaterEqual(table.size, 0)

        self.assertTrue(table.is_loaded)

    def testCreateDeleteTable(self):
        test_table_name = 'pyodps_t_tmp_create_table'
        schema = Schema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds', ], ['string',])

        tables = self.odps._project.tables

        tables.delete(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        table = tables.create(test_table_name, schema, lifecycle=10)
        self.assertEqual(table.name, test_table_name)
        self.assertEqual(table.schema, schema)
        self.assertEqual(table.lifecycle, 10)

        tables.delete(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        table = self.odps.create_table(test_table_name, schema, shard_num=10, hub_lifecycle=5)
        self.assertEqual(table.name, test_table_name)
        self.assertEqual(table.schema, schema)
        self.assertNotEqual(table.lifecycle, 10)
        self.assertEqual(table.shard.shard_num, 10)

        self.odps.delete_table(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

    def testReadWriteTable(self):
        test_table_name = 'pyodps_t_tmp_read_write_table'
        schema = Schema.from_lists(['id', 'name'], ['bigint', 'string'])

        self.odps.delete_table(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        table = self.odps.create_table(test_table_name, schema)
        data = [[111, 'aaa'],
                [222, 'bbb'],
                [333, 'ccc']]
        records = [Record(schema=schema, values=values) for values in data]

        self.odps.write_table(table, 0, records)
        self.assertSequenceEqual(data, [record.values for record in self.odps.read_table(table, 3)])

        self.assertSequenceEqual(data, [record.values for record in table.read(3)])

        self.odps.delete_table(test_table_name)
        self.assertFalse(self.odps.exist_table(test_table_name))

if __name__ == '__main__':
    unittest.main()
