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

import itertools
import pickle
import textwrap
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
except (AttributeError, ImportError):
    np = pd = pa = None

import mock
import pytest

from odps.tests.core import TestBase, to_str, tn
from odps.compat import unittest, six
from odps.errors import NoSuchObject
from odps.models import TableSchema, Record, Column, Partition


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

    def testSchemaPickle(self):
        schema = TableSchema.from_lists(['name'], ['string'])
        self.assertEqual(schema, pickle.loads(pickle.dumps(schema)))

    def testTableExists(self):
        non_exists_table = 'a_non_exists_table'
        self.assertFalse(self.odps.exist_table(non_exists_table))

    def testTable(self):
        tables = self.odps.list_tables()
        try:
            table = next(t for t in tables if not t.is_loaded)
        except StopIteration:
            return

        self.assertRaises(ValueError, lambda: self.odps.get_table(""))

        self.assertIs(table, self.odps.get_table(table.name))

        self.assertIsNone(table._getattr('format'))
        self.assertIsNone(table._getattr('table_schema'))
        self.assertIsNone(table._getattr('comment'))
        self.assertIsNotNone(table._getattr('owner'))
        self.assertIsNone(table._getattr('table_label'))
        self.assertIsNone(table._getattr('creation_time'))
        self.assertIsNone(table._getattr('last_data_modified_time'))
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

        test_table_name = tn('pyodps_t_tmp_test_table_attrs')
        schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds', ], ['string',])
        self.odps.delete_table(test_table_name, if_exists=True)
        table = self.odps.create_table(test_table_name, schema, lifecycle=1)

        try:
            self.assertIs(table, self.odps.get_table(table.name))

            self.assertIsNone(table._getattr('format'))
            self.assertIsNone(table._getattr('table_schema'))
            self.assertIsNone(table._getattr('comment'))
            self.assertIsNone(table._getattr('owner'))
            self.assertIsNone(table._getattr('table_label'))
            self.assertIsNone(table._getattr('creation_time'))
            self.assertIsNone(table._getattr('last_data_modified_time'))
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
            self.assertIsInstance(table.table_schema, TableSchema)
            self.assertGreater(len(table.table_schema.simple_columns), 0)
            self.assertGreaterEqual(len(table.table_schema.partitions), 0)
            self.assertIsInstance(table.owner, six.string_types)
            self.assertIsInstance(table.table_label, six.string_types)
            self.assertIsInstance(table.last_data_modified_time, datetime)
            self.assertIsInstance(table.last_meta_modified_time, datetime)
            self.assertIsInstance(table.is_virtual_view, bool)
            self.assertGreaterEqual(table.lifecycle, -1)
            self.assertGreaterEqual(table.size, 0)

            self.assertTrue(table.is_loaded)

            self.assertIs(table, self.odps.get_table(table.full_table_name))
        finally:
            self.odps.delete_table(test_table_name, if_exists=True)

    def testCreateTableDDL(self):
        from odps.compat import OrderedDict
        from odps.models import Table

        test_table_name = tn('pyodps_t_tmp_table_ddl')
        schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds', ], ['string',])
        self.odps.delete_table(test_table_name, if_exists=True)
        table = self.odps.create_table(test_table_name, schema, lifecycle=10)

        ddl = table.get_ddl()
        self.assertNotIn('EXTERNAL', ddl)
        self.assertNotIn('NOT EXISTS', ddl)
        for col in table.table_schema.names:
            self.assertIn(col, ddl)

        ddl = table.get_ddl(if_not_exists=True)
        self.assertIn('NOT EXISTS', ddl)

        ddl = Table.gen_create_table_sql(
            'test_external_table', schema, comment='TEST_COMMENT',
            storage_handler='com.aliyun.odps.CsvStorageHandler',
            serde_properties=OrderedDict([('name1', 'value1'), ('name2', 'value2')]),
            location='oss://mock_endpoint/mock_bucket/mock_path/',
        )
        self.assertEqual(ddl, textwrap.dedent("""
        CREATE EXTERNAL TABLE `test_external_table` (
          `id` BIGINT,
          `name` STRING
        )
        COMMENT 'TEST_COMMENT'
        PARTITIONED BY (
          `ds` STRING
        )
        STORED BY 'com.aliyun.odps.CsvStorageHandler'
        WITH SERDEPROPERTIES (
          'name1' = 'value1',
          'name2' = 'value2'
        )
        LOCATION 'oss://mock_endpoint/mock_bucket/mock_path/'
        """).strip())

    def testCreateDeleteTable(self):
        test_table_name = tn('pyodps_t_tmp_create_table')
        schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds', ], ['string',])

        tables = self.odps._project.tables

        tables.delete(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        table = tables.create(test_table_name, schema, lifecycle=10)

        self.assertIsNone(table._getattr('owner'))
        self.assertIsNotNone(table.owner)

        self.assertEqual(table.name, test_table_name)
        self.assertEqual(table.table_schema, schema)
        self.assertEqual(table.lifecycle, 10)

        tables.delete(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        str_schema = ('id bigint, name string', 'ds string')
        table = tables.create(test_table_name, str_schema, lifecycle=10)

        self.assertEqual(table.name, test_table_name)
        self.assertEqual(table.table_schema, schema)
        self.assertEqual(table.lifecycle, 10)

        tables.delete(test_table_name, if_exists=True)
        self.assertFalse(self.odps.exist_table(test_table_name))

        table = self.odps.create_table(test_table_name, schema, shard_num=10, hub_lifecycle=5)
        self.assertEqual(table.name, test_table_name)
        self.assertEqual(table.table_schema, schema)
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
        schema = TableSchema(columns=columns, partitions=partitions)

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
        )
        PARTITIONED BY (
          `ds` STRING COMMENT '分区注释',
          `ds2` STRING COMMENT '分区注释2'
        )""").strip()
        ddl_string = textwrap.dedent(u"""
        CREATE TABLE `table_name` (
          `序列` BIGINT,
          `值` STRING
        )
        PARTITIONED BY (
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
        self.assertSequenceEqual([to_str(col.name) for col in table.table_schema.columns],
                                 [to_str(col.name) for col in schema.columns])
        self.assertSequenceEqual([to_str(col.comment) for col in table.table_schema.columns],
                                 [to_str(col.comment) for col in schema.columns])

    def testRecordReadWriteTable(self):
        test_table_name = tn('pyodps_t_tmp_read_write_table')
        schema = TableSchema.from_lists(['id', 'name', 'right'], ['bigint', 'string', 'boolean'])

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
        schema = TableSchema.from_lists(['id', 'name', 'right'], ['bigint', 'string', 'boolean'])

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
        schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['pt'], ['string'])

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
        schema = TableSchema.from_lists(['num'], ['string'], ['pt'], ['string'])

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

        if pd is not None:
            with table.open_reader(partition, reopen=True) as reader:
                pd_data = reader.to_pandas()
                self.assertEqual(len(pd_data), 1)

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
        schema = TableSchema.from_lists(['num'], ['string'], ['pt'], ['string'])

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

        with table.open_reader(partition, async_mode=True, reopen=True) as reader:
            self.assertEqual(reader.count, 1)
            record = next(reader)
            self.assertEqual(record[0], '1')
            self.assertEqual(record.num, '1')

        table.drop()

    def testTableWriteError(self):
        test_table_name = tn('pyodps_t_tmp_test_table_write')
        schema = TableSchema.from_lists(['name'], ['string'])

        self.odps.delete_table(test_table_name, if_exists=True)

        table = self.odps.create_table(test_table_name, schema)
        try:
            with table.open_writer() as writer:
                writer.write([['Content']])
                raise ValueError('Mock error')
        except ValueError as ex:
            self.assertEqual(str(ex), 'Mock error')

    @unittest.skipIf(pd is None, "Need pandas to run this test")
    def testMultiProcessToPandas(self):
        from odps.tunnel.tabletunnel import TableDownloadSession

        test_table_name = tn('pyodps_t_tmp_mproc_read_table')
        schema = TableSchema.from_lists(['num'], ['bigint'])

        self.odps.delete_table(test_table_name, if_exists=True)

        table = self.odps.create_table(test_table_name, schema)
        with table.open_writer(arrow=True) as writer:
            writer.write(pd.DataFrame({"num": np.random.randint(0, 1000, 1000)}))

        with table.open_reader() as reader:
            pd_data = reader.to_pandas(n_process=2)
            assert len(pd_data) == 1000

        orginal_meth = TableDownloadSession.open_record_reader

        def patched(self, start, *args, **kwargs):
            if start != 0:
                raise ValueError("Intentional error")
            return orginal_meth(self, start, *args, **kwargs)

        with pytest.raises(ValueError):
            with mock.patch(
                "odps.tunnel.tabletunnel.TableDownloadSession.open_record_reader", new=patched
            ):
                with table.open_reader() as reader:
                    reader.to_pandas(n_process=2)

        with table.open_reader(arrow=True) as reader:
            pd_data = reader.to_pandas(n_process=2)
            assert len(pd_data) == 1000

    @unittest.skipIf(pd is None, "Need pandas to run this test")
    def testColumnSelectToPandas(self):
        test_table_name = tn('pyodps_t_tmp_col_select_table')
        schema = TableSchema.from_lists(['num1', 'num2'], ['bigint', 'bigint'])

        self.odps.delete_table(test_table_name, if_exists=True)

        table = self.odps.create_table(test_table_name, schema)
        with table.open_writer(arrow=True) as writer:
            writer.write(pd.DataFrame({
                "num1": np.random.randint(0, 1000, 1000),
                "num2": np.random.randint(0, 1000, 1000),
            }))

        with table.open_reader(columns=["num1"]) as reader:
            pd_data = reader.to_pandas()
            assert len(pd_data) == 1000
            assert len(pd_data.columns) == 1

        with table.open_reader(columns=["num1"], arrow=True) as reader:
            pd_data = reader.to_pandas()
            assert len(pd_data) == 1000
            assert len(pd_data.columns) == 1

    @unittest.skipIf(pa is None, "Need pyarrow to run this test")
    def testSimpleArrowReadWriteTable(self):
        test_table_name = tn('pyodps_t_tmp_simple_arrow_read_write_table')
        schema = TableSchema.from_lists(['num'], ['string'], ['pt'], ['string'])

        self.odps.delete_table(test_table_name, if_exists=True)

        table = self.odps.create_table(test_table_name, schema)
        partition = 'pt=20151122'
        table.create_partition(partition)

        with table.open_writer(partition, arrow=True) as writer:
            writer.write(pd.DataFrame({"num": list("ABCDE")}))

        with table.open_reader(partition, arrow=True) as reader:
            self.assertEqual(reader.count, 5)
            batches = list(reader)
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0].num_rows, 5)

        with table.open_reader(partition, reopen=True, arrow=True) as reader:
            pd_data = reader.to_pandas()
            self.assertEqual(len(pd_data), 5)

        table.drop()

    def testRunSQLClearCache(self):
        test_table_name = tn('pyodps_t_tmp_statement_cache_clear')
        self.odps.delete_table(test_table_name, if_exists=True)

        self.odps.create_table(test_table_name, "col string")
        self.odps.get_table(test_table_name)

        self.odps.execute_sql("ALTER TABLE %s ADD COLUMN col2 string" % test_table_name)
        assert "col2" in self.odps.get_table(test_table_name).table_schema

        self.odps.execute_sql(
            "ALTER TABLE %s.%s ADD COLUMN col3 string" % (self.odps.project, test_table_name)
        )
        assert "col3" in self.odps.get_table(test_table_name).table_schema

        self.odps.delete_table(test_table_name)

    def testMaxPartition(self):
        test_table_name = tn('pyodps_t_tmp_max_partition')
        self.odps.delete_table(test_table_name, if_exists=True)

        table = self.odps.create_table(test_table_name, ("col string", "pt1 string, pt2 string"))
        for pt1, pt2 in (("a", "a"), ("a", "b"), ("b", "c")):
            part_spec = "pt1=%s,pt2=%s" % (pt1, pt2)
            self.odps.write_table(
                test_table_name, [["value"]], partition=part_spec, create_partition=True
            )
        table.create_partition("pt1=b,pt2=d")
        table.create_partition("pt1=c,pt2=e")

        assert tuple(table.get_max_partition().partition_spec.values()) == ("b", "c")
        assert tuple(
            table.get_max_partition(skip_empty=False).partition_spec.values()
        ) == ("c", "e")
        assert tuple(
            table.get_max_partition("pt1=a").partition_spec.values()
        ) == ("a", "b")
        assert table.get_max_partition("pt1=c") is None
        assert table.get_max_partition("pt1=d") is None

        with pytest.raises(ValueError):
            table.get_max_partition("pt2=a")
        with pytest.raises(ValueError):
            table.get_max_partition("pt1=c,pt2=e")

        table.drop()

    def testSchemaArgBackwardCompat(self):
        if six.PY2:
            from odps.models import Schema
        else:
            with pytest.deprecated_call():
                from odps.models import Schema

        columns = [Column(name='num', type='bigint', comment='the column'),
                   Column(name='num2', type='double', comment='the column2')]
        schema = Schema(columns=columns)

        table_name = tn('test_backward_compat')

        with pytest.deprecated_call():
            table = self.odps.create_table(table_name, schema=schema, lifecycle=1)
        assert self.odps.exist_table(table_name)
        with pytest.deprecated_call():
            assert isinstance(table.schema, Schema)
        with pytest.deprecated_call():
            getattr(table, "last_modified_time")

        table.drop()


if __name__ == '__main__':
    unittest.main()
