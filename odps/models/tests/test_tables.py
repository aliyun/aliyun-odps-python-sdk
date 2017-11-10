#!/usr/bin/env python
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

from datetime import datetime
import itertools
import textwrap

from odps.tests.core import TestBase, to_str, tn
from odps.compat import unittest, six
from odps.errors import NoSuchObject
from odps.models import Schema, Record, Column, Partition


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
        try:
            table = next(t for t in tables if not t.is_loaded)
        except StopIteration:
            return

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
        self.assertIsInstance(table.creation_time, datetime)
        self.assertIsInstance(table.schema, Schema)
        self.assertGreater(len(table.schema.simple_columns), 0)
        self.assertGreaterEqual(len(table.schema.partitions), 0)
        self.assertIsInstance(table.owner, six.string_types)
        self.assertIsInstance(table.table_label, six.string_types)
        self.assertIsInstance(table.last_modified_time, datetime)
        self.assertIsInstance(table.last_meta_modified_time, datetime)
        self.assertIsInstance(table.is_virtual_view, bool)
        self.assertGreaterEqual(table.lifecycle, -1)
        self.assertGreaterEqual(table.size, 0)

        self.assertTrue(table.is_loaded)

    def testCreateTableDDL(self):
        test_table_name = tn('pyodps_t_tmp_table_ddl')
        schema = Schema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds', ], ['string',])
        self.odps.delete_table(test_table_name, if_exists=True)
        table = self.odps.create_table(test_table_name, schema, lifecycle=10)

        ddl = table.get_ddl()
        for col in table.schema.names:
            self.assertIn(col, ddl)

    def testCreateDeleteTable(self):
        test_table_name = tn('pyodps_t_tmp_create_table')
        schema = Schema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds', ], ['string',])

        tables = self.odps._project.tables

        tables.delete(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        table = tables.create(test_table_name, schema, lifecycle=10)

        self.assertIsNone(table._getattr('owner'))
        self.assertIsNotNone(table.owner)

        self.assertEqual(table.name, test_table_name)
        self.assertEqual(table.schema, schema)
        self.assertEqual(table.lifecycle, 10)

        tables.delete(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        str_schema = ('id bigint, name string', 'ds string')
        table = tables.create(test_table_name, str_schema, lifecycle=10)

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

    def testCreateTableWithChineseColumn(self):
        test_table_name = tn('pyodps_t_tmp_create_table_with_chinese_columns')
        columns = [
            Column(name='序列', type='bigint', comment='注释'),
            Column(name=u'值', type=u'string', comment=u'注释2'),
        ]
        partitions = [
            Partition(name='ds', type='string', comment='分区注释'),
            Partition(name=u'ds2', type=u'string', comment=u'分区注释2'),
        ]
        schema = Schema(columns=columns, partitions=partitions)

        columns_repr = "[<column 序列, type bigint>, <column 值, type string>]"
        partitions_repr = "[<partition ds, type string>, <partition ds2, type string>]"
        schema_repr = textwrap.dedent("""
        odps.Schema {
          序列    bigint      # 注释
          值      string      # 注释2
        }
        Partitions {
          ds      string      # 分区注释
          ds2     string      # 分区注释2
        }
        """).strip()
        ddl_string_comment = textwrap.dedent(u"""
        CREATE TABLE `table_name` (
          `序列` BIGINT COMMENT '注释',
          `值` STRING COMMENT '注释2'
        ) PARTITIONED BY (
          `ds` STRING COMMENT '分区注释',
          `ds2` STRING COMMENT '分区注释2'
        )""").strip()
        ddl_string = textwrap.dedent(u"""
        CREATE TABLE `table_name` (
          `序列` BIGINT,
          `值` STRING
        ) PARTITIONED BY (
          `ds` STRING,
          `ds2` STRING
        )""").strip()

        self.assertEqual(repr(columns), columns_repr)
        self.assertEqual(repr(partitions), partitions_repr)
        self.assertEqual(repr(schema).strip(), schema_repr)
        self.assertEqual(schema.get_table_ddl().strip(), ddl_string_comment)
        self.assertEqual(schema.get_table_ddl(with_comments=False).strip(), ddl_string)

        self.odps.delete_table(test_table_name, if_exists=True)

        table = self.odps.create_table(test_table_name, schema)
        self.assertSequenceEqual([to_str(col.name) for col in table.schema.columns],
                                 [to_str(col.name) for col in schema.columns])
        self.assertSequenceEqual([to_str(col.comment) for col in table.schema.columns],
                                 [to_str(col.comment) for col in schema.columns])

    def testRecordReadWriteTable(self):
        test_table_name = tn('pyodps_t_tmp_read_write_table')
        schema = Schema.from_lists(['id', 'name', 'right'], ['bigint', 'string', 'boolean'])

        self.odps.delete_table(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        table = self.odps.create_table(test_table_name, schema)
        data = [[111, 'aaa', True],
                [222, 'bbb', False],
                [333, 'ccc', True],
                [5940813139082772990, '中文', False]]
        length = len(data)
        records = [Record(schema=schema, values=values) for values in data]

        texted_data = [[it[0], to_str(it[1]), it[2]] for it in data]

        self.odps.write_table(table, 0, records)
        self.assertSequenceEqual(texted_data, [record.values for record in self.odps.read_table(table, length)])
        self.assertSequenceEqual(texted_data[::2],
                                 [record.values for record in self.odps.read_table(table, length, step=2)])

        self.assertSequenceEqual(texted_data, [record.values for record in table.head(length)])

        table.truncate()
        self.assertEqual([], list(self.odps.read_table(table)))

        self.odps.delete_table(test_table_name)
        self.assertFalse(self.odps.exist_table(test_table_name))

    def testArrayReadWriteTable(self):
        test_table_name = tn('pyodps_t_tmp_read_write_table')
        schema = Schema.from_lists(['id', 'name', 'right'], ['bigint', 'string', 'boolean'])

        self.odps.delete_table(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        table = self.odps.create_table(test_table_name, schema)
        data = [[111, 'aaa', True],
                [222, 'bbb', False],
                [333, 'ccc', True],
                [444, '中文', False]]
        length = len(data)

        texted_data = [[it[0], to_str(it[1]), it[2]] for it in data]

        self.odps.write_table(table, 0, data)
        self.assertSequenceEqual(texted_data, [record.values for record in self.odps.read_table(table, length)])
        self.assertSequenceEqual(texted_data[::2],
                                 [record.values for record in self.odps.read_table(table, length, step=2)])

        self.assertSequenceEqual(texted_data, [record.values for record in table.head(length)])

        self.odps.delete_table(test_table_name)
        self.assertFalse(self.odps.exist_table(test_table_name))

    def testReadWritePartitionTable(self):
        test_table_name = tn('pyodps_t_tmp_read_write_partition_table')
        schema = Schema.from_lists(['id', 'name'], ['bigint', 'string'], ['pt'], ['string'])

        self.odps.delete_table(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        table = self.odps.create_table(test_table_name, schema)
        table._upload_ids = dict()

        pt1 = 'pt=20151122'
        pt2 = 'pt=20151123'
        table.create_partition(pt1)
        table.create_partition(pt2)

        with table.open_reader(pt1) as reader:
            self.assertEqual(len(list(reader)), 0)

        with table.open_writer(pt1, commit=False) as writer:
            record = table.new_record([1, 'name1'])
            writer.write(record)

            record = table.new_record()
            record[0] = 3
            record[1] = 'name3'
            writer.write(record)

        self.assertEqual(len(table._upload_ids), 1)
        upload_id = list(table._upload_ids.values())[0]
        with table.open_writer(pt1):
            self.assertEqual(len(table._upload_ids), 1)
            self.assertEqual(upload_id, list(table._upload_ids.values())[0])

        with table.open_writer(pt2) as writer:
            writer.write([2, 'name2'])

        with table.open_reader(pt1, reopen=True) as reader:
            records = list(reader)
            self.assertEqual(len(records), 2)
            self.assertEqual(sum(r[0] for r in records), 4)

        with table.open_reader(pt2, reopen=True) as reader:
            records = list(reader)
            self.assertEqual(len(records), 1)
            self.assertEqual(sum(r[0] for r in records), 2)

        table.drop()

    def testSimpleRecordReadWriteTable(self):
        test_table_name = tn('pyodps_t_tmp_simpe_read_write_table')
        schema = Schema.from_lists(['num'], ['string'], ['pt'], ['string'])

        self.odps.delete_table(test_table_name, if_exists=True)

        table = self.odps.create_table(test_table_name, schema)
        partition = 'pt=20151122'
        table.create_partition(partition)

        with table.open_writer(partition) as writer:
            record = table.new_record()
            record[0] = '1'
            writer.write(record)

        with table.open_reader(partition) as reader:
            self.assertEqual(reader.count, 1)
            record = next(reader)
            self.assertEqual(record[0], '1')
            self.assertEqual(record.num, '1')

        partition = 'pt=20151123'
        self.assertRaises(NoSuchObject, lambda: table.open_writer(partition, create_partition=False))

        with table.open_writer(partition, create_partition=True) as writer:
            record = table.new_record()
            record[0] = '1'
            writer.write(record)

        with table.open_reader(partition) as reader:
            self.assertEqual(reader.count, 1)
            record = next(reader)
            self.assertEqual(record[0], '1')
            self.assertEqual(record.num, '1')

        table.drop()

    def testSimpleArrayReadWriteTable(self):
        test_table_name = tn('pyodps_t_tmp_simpe_read_write_table')
        schema = Schema.from_lists(['num'], ['string'], ['pt'], ['string'])

        self.odps.delete_table(test_table_name, if_exists=True)

        table = self.odps.create_table(test_table_name, schema)
        partition = 'pt=20151122'
        table.create_partition(partition)

        with table.open_writer(partition) as writer:
            writer.write(['1', ])

        with table.open_reader(partition) as reader:
            self.assertEqual(reader.count, 1)
            record = next(reader)
            self.assertEqual(record[0], '1')
            self.assertEqual(record.num, '1')

        table.drop()

    def testTableWriteError(self):
        test_table_name = tn('pyodps_t_tmp_test_table_write')
        schema = Schema.from_lists(['name'], ['string'])

        self.odps.delete_table(test_table_name, if_exists=True)

        table = self.odps.create_table(test_table_name, schema)
        try:
            with table.open_writer() as writer:
                writer.write([['Content']])
                raise ValueError('Mock error')
        except ValueError as ex:
            self.assertEqual(str(ex), 'Mock error')


if __name__ == '__main__':
    unittest.main()
