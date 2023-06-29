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

import errno
import itertools
import pickle
import textwrap
import time
from collections import OrderedDict
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
except (AttributeError, ImportError):
    np = pd = pa = None

import mock
import pytest

from ...compat import six
from ...config import options
from ...errors import NoSuchObject
from ...tests.core import tn, pandas_case
from ...utils import to_text
from .. import TableSchema, Record, Column, Partition


def test_tables(odps):
    assert odps.get_project().tables is odps.get_project().tables
    size = len(list(itertools.islice(odps.list_tables(), 0, 200)))
    assert size >= 0

    if size > 0:
        table = next(iter(odps.list_tables()))

        tables = list(itertools.islice(odps.list_tables(prefix=table.name), 0, 10))
        assert len(tables) >= 1
        for t in tables:
            assert t.name.startswith(table.name) is True

        tables = list(itertools.islice(odps.list_tables(owner=table.owner), 0, 10))
        assert len(tables) >= 1
        for t in tables:
            assert t.owner == table.owner


def test_schema_pickle():
    schema = TableSchema.from_lists(['name'], ['string'])
    assert schema == pickle.loads(pickle.dumps(schema))


def test_table_exists(odps):
    non_exists_table = 'a_non_exists_table'
    assert odps.exist_table(non_exists_table) is False


def test_table(odps):
    tables = odps.list_tables()
    try:
        table = next(t for t in tables if not t.is_loaded)
    except StopIteration:
        return

    pytest.raises(ValueError, lambda: odps.get_table(""))

    assert table is odps.get_table(table.name)

    assert table._getattr('format') is None
    assert table._getattr('table_schema') is None
    assert table._getattr('comment') is None
    assert table._getattr('owner') is not None
    assert table._getattr('table_label') is None
    assert table._getattr('creation_time') is None
    assert table._getattr('last_data_modified_time') is None
    assert table._getattr('last_meta_modified_time') is None
    assert table._getattr('is_virtual_view') is None
    assert table._getattr('lifecycle') is None
    assert table._getattr('view_text') is None
    assert table._getattr('size') is None
    assert table._getattr('is_archived') is None
    assert table._getattr('physical_size') is None
    assert table._getattr('file_num') is None

    assert table.is_loaded is False
    assert table._is_extend_info_loaded is False

    test_table_name = tn('pyodps_t_tmp_test_table_attrs')
    schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds', ], ['string',])
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema, lifecycle=1)

    try:
        assert table is odps.get_table(table.name)

        assert table._getattr('format') is None
        assert table._getattr('table_schema') is None
        assert table._getattr('comment') is None
        assert table._getattr('owner') is None
        assert table._getattr('table_label') is None
        assert table._getattr('creation_time') is None
        assert table._getattr('last_data_modified_time') is None
        assert table._getattr('last_meta_modified_time') is None
        assert table._getattr('is_virtual_view') is None
        assert table._getattr('lifecycle') is None
        assert table._getattr('view_text') is None
        assert table._getattr('size') is None
        assert table._getattr('is_archived') is None
        assert table._getattr('physical_size') is None
        assert table._getattr('file_num') is None

        assert table.is_loaded is False
        assert table._is_extend_info_loaded is False

        assert isinstance(table.is_archived, bool)
        assert table.physical_size >= 0
        assert table.file_num >= 0
        assert isinstance(table.creation_time, datetime)
        assert isinstance(table.table_schema, TableSchema)
        assert len(table.table_schema.simple_columns) > 0
        assert len(table.table_schema.partitions) >= 0
        assert isinstance(table.owner, six.string_types)
        assert isinstance(table.table_label, six.string_types)
        assert isinstance(table.last_data_modified_time, datetime)
        assert isinstance(table.last_meta_modified_time, datetime)
        assert isinstance(table.is_virtual_view, bool)
        assert table.lifecycle >= -1
        assert table.size >= 0

        assert table.is_loaded is True

        assert table is odps.get_table(table.full_table_name)
    finally:
        odps.delete_table(test_table_name, if_exists=True)


def test_create_table_ddl(odps):
    from .. import Table

    test_table_name = tn('pyodps_t_tmp_table_ddl')
    schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds', ], ['string',])
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema, lifecycle=10)

    ddl = table.get_ddl()
    assert 'EXTERNAL' not in ddl
    assert 'NOT EXISTS' not in ddl
    for col in table.table_schema.names:
        assert col in ddl

    ddl = table.get_ddl(if_not_exists=True)
    assert 'NOT EXISTS' in ddl

    ddl = Table.gen_create_table_sql(
        'test_external_table', schema, comment='TEST_COMMENT',
        storage_handler='com.aliyun.odps.CsvStorageHandler',
        serde_properties=OrderedDict([('name1', 'value1'), ('name2', 'value2')]),
        location='oss://mock_endpoint/mock_bucket/mock_path/',
    )
    assert ddl == textwrap.dedent("""
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
    """).strip()


def test_create_delete_table(odps):
    test_table_name = tn('pyodps_t_tmp_create_table')
    schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['ds', ], ['string',])

    tables = odps._project.tables

    tables.delete(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = tables.create(test_table_name, schema, lifecycle=10)

    assert table._getattr('owner') is None
    assert table.owner is not None

    assert table.name == test_table_name
    assert table.table_schema == schema
    assert table.lifecycle == 10

    tables.delete(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    str_schema = ('id bigint, name string', 'ds string')
    table = tables.create(test_table_name, str_schema, lifecycle=10)

    assert table.name == test_table_name
    assert table.table_schema == schema
    assert table.lifecycle == 10

    tables.delete(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(test_table_name, schema, shard_num=10, hub_lifecycle=5)
    assert table.name == test_table_name
    assert table.table_schema == schema
    assert table.lifecycle != 10
    assert table.shard.shard_num == 10

    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False


def test_create_table_with_chinese_column(odps):
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

    assert repr(columns) == columns_repr
    assert repr(partitions) == partitions_repr
    assert repr(schema).strip() == schema_repr
    assert schema.get_table_ddl().strip() == ddl_string_comment
    assert schema.get_table_ddl(with_comments=False).strip() == ddl_string

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    assert [to_text(col.name) for col in table.table_schema.columns] == [to_text(col.name) for col in schema.columns]
    assert [to_text(col.comment) for col in table.table_schema.columns] == [to_text(col.comment) for col in schema.columns]


def test_record_read_write_table(odps):
    test_table_name = tn('pyodps_t_tmp_read_write_table')
    schema = TableSchema.from_lists(['id', 'name', 'right'], ['bigint', 'string', 'boolean'])

    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(test_table_name, schema)
    data = [[111, 'aaa', True],
            [222, 'bbb', False],
            [333, 'ccc', True],
            [5940813139082772990, '中文', False]]
    length = len(data)
    records = [Record(schema=schema, values=values) for values in data]

    texted_data = [[it[0], to_text(it[1]), it[2]] for it in data]

    odps.write_table(table, 0, records)
    assert texted_data == [record.values for record in odps.read_table(table, length)]
    assert texted_data[::2] == [record.values for record in odps.read_table(table, length, step=2)]

    assert texted_data == [record.values for record in table.head(length)]

    table.truncate()
    assert [] == list(odps.read_table(table))

    odps.delete_table(test_table_name)
    assert odps.exist_table(test_table_name) is False


def test_array_read_write_table(odps):
    test_table_name = tn('pyodps_t_tmp_read_write_table')
    schema = TableSchema.from_lists(['id', 'name', 'right'], ['bigint', 'string', 'boolean'])

    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(test_table_name, schema)
    data = [[111, 'aaa', True],
            [222, 'bbb', False],
            [333, 'ccc', True],
            [444, '中文', False]]
    length = len(data)

    texted_data = [[it[0], to_text(it[1]), it[2]] for it in data]

    odps.write_table(table, 0, data)
    assert texted_data == [record.values for record in odps.read_table(table, length)]
    assert texted_data[::2] == [record.values for record in odps.read_table(table, length, step=2)]

    assert texted_data == [record.values for record in table.head(length)]

    odps.delete_table(test_table_name)
    assert odps.exist_table(test_table_name) is False


def test_read_write_partition_table(odps):
    test_table_name = tn('pyodps_t_tmp_read_write_partition_table')
    schema = TableSchema.from_lists(['id', 'name'], ['bigint', 'string'], ['pt'], ['string'])

    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(test_table_name, schema)
    table._upload_ids = dict()

    pt1 = 'pt=20151122'
    pt2 = 'pt=20151123'
    table.create_partition(pt1)
    table.create_partition(pt2)

    with table.open_reader(pt1) as reader:
        assert len(list(reader)) == 0

    with table.open_writer(pt1, commit=False) as writer:
        record = table.new_record([1, 'name1'])
        writer.write(record)

        record = table.new_record()
        record[0] = 3
        record[1] = 'name3'
        writer.write(record)

    assert len(table._upload_ids) == 1
    upload_id = list(table._upload_ids.values())[0]
    with table.open_writer(pt1):
        assert len(table._upload_ids) == 1
        assert upload_id == list(table._upload_ids.values())[0]

    with table.open_writer(pt2) as writer:
        writer.write([2, 'name2'])

    with table.open_reader(pt1, reopen=True) as reader:
        records = list(reader)
        assert len(records) == 2
        assert sum(r[0] for r in records) == 4

    with table.open_reader(pt2, reopen=True) as reader:
        records = list(reader)
        assert len(records) == 1
        assert sum(r[0] for r in records) == 2

    table.drop()


def test_simple_record_read_write_table(odps):
    test_table_name = tn('pyodps_t_tmp_simpe_read_write_table')
    schema = TableSchema.from_lists(['num'], ['string'], ['pt'], ['string'])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    partition = 'pt=20151122'
    table.create_partition(partition)

    with table.open_writer(partition) as writer:
        record = table.new_record()
        record[0] = '1'
        writer.write(record)

    with table.open_reader(partition) as reader:
        assert reader.count == 1
        record = next(reader)
        assert record[0] == '1'
        assert record.num == '1'

    if pd is not None:
        with table.open_reader(partition, reopen=True) as reader:
            pd_data = reader.to_pandas()
            assert len(pd_data) == 1

    partition = 'pt=20151123'
    pytest.raises(NoSuchObject, lambda: table.open_writer(partition, create_partition=False))

    with table.open_writer(partition, create_partition=True) as writer:
        record = table.new_record()
        record[0] = '1'
        writer.write(record)

    with table.open_reader(partition) as reader:
        assert reader.count == 1
        record = next(reader)
        assert record[0] == '1'
        assert record.num == '1'

    table.drop()


def test_simple_array_read_write_table(odps):
    test_table_name = tn('pyodps_t_tmp_simpe_read_write_table')
    schema = TableSchema.from_lists(['num'], ['string'], ['pt'], ['string'])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    partition = 'pt=20151122'
    table.create_partition(partition)

    with table.open_writer(partition) as writer:
        writer.write(['1', ])

    with table.open_reader(partition) as reader:
        assert reader.count == 1
        record = next(reader)
        assert record[0] == '1'
        assert record.num == '1'

    with table.open_reader(partition, async_mode=True, reopen=True) as reader:
        assert reader.count == 1
        record = next(reader)
        assert record[0] == '1'
        assert record.num == '1'

    table.drop()


def test_table_write_error(odps):
    test_table_name = tn('pyodps_t_tmp_test_table_write')
    schema = TableSchema.from_lists(['name'], ['string'])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    try:
        with table.open_writer() as writer:
            writer.write([['Content']])
            raise ValueError('Mock error')
    except ValueError as ex:
        assert str(ex) == 'Mock error'


@pandas_case
def test_multi_process_to_pandas(odps):
    from ...tunnel.tabletunnel import TableDownloadSession

    if pa is None:
        pytest.skip("Need pyarrow to run the test.")

    test_table_name = tn('pyodps_t_tmp_mproc_read_table')
    schema = TableSchema.from_lists(['num'], ['bigint'])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
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


@pandas_case
def test_column_select_to_pandas(odps):
    if pa is None:
        pytest.skip("Need pyarrow to run the test.")

    test_table_name = tn('pyodps_t_tmp_col_select_table')
    schema = TableSchema.from_lists(['num1', 'num2'], ['bigint', 'bigint'])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
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


@pandas_case
def test_complex_type_to_pandas(odps):
    test_table_name = tn("pyodps_t_tmp_complex_type_to_pd")
    schema = TableSchema.from_lists(
        ['cp1', 'cp2', 'cp3'], [
            'array<string>', 'map<string,bigint>', 'struct<a: string, b: bigint>'
        ]
    )

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    row = [
        ["abc", "def"],
        OrderedDict([("uvw", 1), ("xyz", 2)]),
        OrderedDict([("a", "data"), ("b", 1)]),
    ]
    with table.open_writer() as writer:
        writer.write([row])

    with table.open_reader() as reader:
        pd_data = reader.to_pandas()
        assert pd_data.iloc[0].to_list() == row

    if pa is not None:
        with table.open_reader(arrow=True) as reader:
            pd_data = reader.to_pandas()
            assert [
                pd_data.iloc[0, 0].tolist(),
                OrderedDict(pd_data.iloc[0, 1]),
                OrderedDict(pd_data.iloc[0, 2]),
            ] == row

@pandas_case
def test_record_to_pandas_batches(odps):
    test_table_name = tn('pyodps_t_read_in_batches')
    odps.delete_table(test_table_name, if_exists=True)
    rec_count = 37

    data = [[idx, "str_" + str(idx)] for idx in range(rec_count)]

    table = odps.create_table(test_table_name, "col1 bigint, col2 string")
    with table.open_writer() as writer:
        writer.write(data)

    try:
        options.tunnel.read_row_batch_size = 5
        options.tunnel.batch_merge_threshold = 5
        with table.open_reader() as reader:
            pd_result = reader.to_pandas()
        assert len(pd_result) == rec_count

        with table.open_reader() as reader:
            pd_result = reader[:10].to_pandas()
        assert len(pd_result) == 10
    finally:
        options.tunnel.read_row_batch_size = 1024
        options.tunnel.batch_merge_threshold = 128


@pytest.mark.skipif(pa is None, reason="Need pyarrow to run this test")
def test_simple_arrow_read_write_table(odps):
    test_table_name = tn('pyodps_t_tmp_simple_arrow_read_write_table')
    schema = TableSchema.from_lists(['num'], ['string'], ['pt'], ['string'])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    partition = 'pt=20151122'
    table.create_partition(partition)

    with table.open_writer(partition, arrow=True) as writer:
        writer.write(pd.DataFrame({"num": list("ABCDE")}))

    with table.open_reader(partition, arrow=True) as reader:
        assert reader.count == 5
        batches = list(reader)
        assert len(batches) == 1
        assert batches[0].num_rows == 5

    with table.open_reader(partition, reopen=True, arrow=True) as reader:
        pd_data = reader.to_pandas()
        assert len(pd_data) == 5

    # now test corner case of empty table
    table.truncate(partition)

    with table.open_reader(partition, arrow=True) as reader:
        batches = list(reader)
        assert len(batches) == 0

    with table.open_reader(partition, reopen=True, arrow=True) as reader:
        pd_data = reader.to_pandas()
        assert len(pd_data.columns) == 1
        assert len(pd_data) == 0

    table.drop()


def test_run_sql_clear_cache(odps):
    test_table_name = tn('pyodps_t_tmp_statement_cache_clear')
    odps.delete_table(test_table_name, if_exists=True)

    odps.create_table(test_table_name, "col string")
    odps.get_table(test_table_name)

    odps.execute_sql("ALTER TABLE %s ADD COLUMN col2 string" % test_table_name)
    assert "col2" in odps.get_table(test_table_name).table_schema

    odps.execute_sql(
        "ALTER TABLE %s.%s ADD COLUMN col3 string" % (odps.project, test_table_name)
    )
    assert "col3" in odps.get_table(test_table_name).table_schema

    odps.delete_table(test_table_name)


def test_max_partition(odps):
    test_table_name = tn('pyodps_t_tmp_max_partition')
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, ("col string", "pt1 string, pt2 string"))
    for pt1, pt2 in (("a", "a"), ("a", "b"), ("b", "c")):
        part_spec = "pt1=%s,pt2=%s" % (pt1, pt2)
        odps.write_table(
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


def test_read_with_retry(odps):
    test_table_name = tn('pyodps_t_tmp_read_with_retry')
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, "col string")

    try:
        data = [["str%d" % idx] for idx in range(10)]
        with table.open_writer() as writer:
            writer.write(data)

        from ..readers import TunnelRecordReader
        original = TunnelRecordReader._open_and_iter_reader
        exc_type = ConnectionResetError if six.PY3 else OSError
        raised = []

        def raise_conn_reset():
            exc = exc_type("Connection reset")
            exc.errno = errno.ECONNRESET
            raised.append(True)
            raise exc

        def wrapped(self, start, *args, **kwargs):
            for idx, rec in enumerate(original(self, start, *args, **kwargs)):
                yield rec
                if idx == 2:
                    raise_conn_reset()

        with mock.patch.object(
            TunnelRecordReader, "_open_and_iter_reader", new=wrapped
        ):
            with table.open_reader() as reader:
                assert data == sorted([rec[0]] for rec in reader)
        assert len(raised) > 1

        with pytest.raises(exc_type) as exc_info:
            with table.open_reader(reopen=True) as reader:
                for idx, _ in enumerate(reader):
                    if idx == 2:
                        raise_conn_reset()
        assert exc_info.value.errno == errno.ECONNRESET
    finally:
        table.drop()


def test_write_record_with_interval(odps):
    test_table_name = tn('pyodps_t_tmp_write_rec_interval')
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, "col string")
    try:
        options.table_auto_flush_time = 2
        with table.open_writer() as writer:
            writer.write([["str1"], ["str2"]], block_id=0)
            time.sleep(4.5)
            # as auto_flush_time arrives, a new block should be generated
            assert len(writer._id_to_blocks) == 2
            writer.write([["str3"], ["str4"]], block_id=0)
        strs = [rec[0] for rec in table.head(4)]
        assert sorted(strs) == ["str1", "str2", "str3", "str4"]
        assert not writer._daemon_thread.is_alive()
    finally:
        options.table_auto_flush_time = 150
        table.drop()


@pytest.mark.skipif(pa is None, reason="Need pyarrow to run this test")
def test_write_arrow_with_interval(odps):
    test_table_name = tn('pyodps_t_tmp_write_arrow_interval')
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, "col string")
    try:
        options.table_auto_flush_time = 2
        with table.open_writer(arrow=True) as writer:
            writer.write(pd.DataFrame({"col": ["str1", "str2"]}))
            time.sleep(4.5)
            # as auto_flush_time arrives, a new block should be generated
            assert len(writer._id_to_blocks) == 2
            writer.write(pd.DataFrame({"col": ["str3", "str4"]}))
        strs = [rec[0] for rec in table.head(4)]
        assert sorted(strs) == ["str1", "str2", "str3", "str4"]
        assert not writer._daemon_thread.is_alive()
    finally:
        options.table_auto_flush_time = 150
        table.drop()


def test_schema_arg_backward_compat(odps):
    if six.PY2:
        from .. import Schema
    else:
        with pytest.deprecated_call():
            from .. import Schema

    columns = [Column(name='num', type='bigint', comment='the column'),
               Column(name='num2', type='double', comment='the column2')]
    schema = Schema(columns=columns)

    table_name = tn('test_backward_compat')

    with pytest.deprecated_call():
        table = odps.create_table(table_name, schema=schema, lifecycle=1)
    assert odps.exist_table(table_name)
    with pytest.deprecated_call():
        assert isinstance(table.schema, Schema)
    with pytest.deprecated_call():
        getattr(table, "last_modified_time")

    table.drop()
