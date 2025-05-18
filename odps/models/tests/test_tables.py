#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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
import time
from collections import OrderedDict
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
except (AttributeError, ImportError):
    np = pd = pa = None

import pytest

from ... import types as odps_types
from ...compat import six
from ...config import options
from ...tests.core import tn
from ...utils import to_text
from .. import Column, Partition, Table, TableSchema
from ..cluster_info import ClusterInfo, ClusterSortOrder, ClusterType
from ..storage_tier import StorageTier


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
    schema = TableSchema.from_lists(["name"], ["string"])
    assert schema == pickle.loads(pickle.dumps(schema))


def test_table_exists(odps):
    non_exists_table = "a_non_exists_table"
    assert odps.exist_table(non_exists_table) is False


def test_table(odps):
    tables = odps.list_tables()
    try:
        table = next(t for t in tables if not t.is_loaded)
    except StopIteration:
        return

    pytest.raises(ValueError, lambda: odps.get_table(""))

    assert table is odps.get_table(table.name)

    assert table._getattr("format") is None
    assert table._getattr("table_schema") is None
    assert table._getattr("comment") is None
    assert table._getattr("owner") is not None
    assert table._getattr("table_label") is None
    assert table._getattr("creation_time") is None
    assert table._getattr("last_data_modified_time") is None
    assert table._getattr("last_meta_modified_time") is None
    assert table._getattr("is_virtual_view") is None
    assert table._getattr("lifecycle") is None
    assert table._getattr("view_text") is None
    assert table._getattr("size") is None
    assert table._getattr("is_archived") is None
    assert table._getattr("physical_size") is None
    assert table._getattr("file_num") is None

    assert table.is_loaded is False
    assert table._is_extend_info_loaded is False

    test_table_name = tn("pyodps_t_tmp_test_table_attrs")
    schema = TableSchema.from_lists(
        ["id", "name"], ["bigint", "string"], ["ds"], ["string"]
    )
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema, lifecycle=1)

    try:
        assert table is odps.get_table(table.name)

        assert table._getattr("format") is None
        assert table._getattr("table_schema") is None
        assert table._getattr("comment") is None
        assert table._getattr("owner") is None
        assert table._getattr("table_label") is None
        assert table._getattr("creation_time") is None
        assert table._getattr("last_data_modified_time") is None
        assert table._getattr("last_meta_modified_time") is None
        assert table._getattr("is_virtual_view") is None
        assert table._getattr("lifecycle") is None
        assert table._getattr("view_text") is None
        assert table._getattr("size") is None
        assert table._getattr("is_archived") is None
        assert table._getattr("physical_size") is None
        assert table._getattr("file_num") is None

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
    test_table_name = tn("pyodps_t_tmp_table_ddl")
    schema = TableSchema.from_lists(
        ["id", "name"], ["bigint", "string"], ["ds"], ["string"]
    )
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema, lifecycle=10)

    ddl = table.get_ddl()
    assert "EXTERNAL" not in ddl
    assert "NOT EXISTS" not in ddl
    for col in table.table_schema.names:
        assert col in ddl

    ddl = table.get_ddl(if_not_exists=True)
    assert "NOT EXISTS" in ddl

    # make sure ddl works
    odps.delete_table(test_table_name, if_exists=True)
    odps.execute_sql(ddl)
    assert odps.exist_table(test_table_name)

    ddl = Table.gen_create_table_sql(
        "test_trans_table",
        schema,
        comment="TEST_COMMENT",
        transactional=True,
        table_properties={"table.format.version": True},
    )
    assert (
        ddl
        == textwrap.dedent(
            """
    CREATE TABLE `test_trans_table` (
      `id` BIGINT,
      `name` STRING
    )
    COMMENT 'TEST_COMMENT'
    PARTITIONED BY (
      `ds` STRING
    )
    TBLPROPERTIES (
      'table.format.version' = 'true',
      'transactional' = 'true'
    )"""
        ).strip()
    )

    ddl = Table.gen_create_table_sql(
        "test_external_table",
        schema,
        comment="TEST_COMMENT",
        storage_handler="CsvStorageHandler",
        serde_properties=OrderedDict([("name1", "value1"), ("name2", "value2")]),
        location="oss://mock_endpoint/mock_bucket/mock_path/",
    )
    assert (
        ddl
        == textwrap.dedent(
            """
    CREATE EXTERNAL TABLE `test_external_table` (
      `id` BIGINT,
      `name` STRING
    )
    COMMENT 'TEST_COMMENT'
    PARTITIONED BY (
      `ds` STRING
    )
    STORED BY 'CsvStorageHandler'
    WITH SERDEPROPERTIES (
      'name1' = 'value1',
      'name2' = 'value2'
    )
    LOCATION 'oss://mock_endpoint/mock_bucket/mock_path/'
    """
        ).strip()
    )

    ddl = Table.gen_create_table_sql(
        "test_external_table",
        schema,
        comment="TEST_COMMENT",
        row_format_serde="org.apache.hadoop.hive.serde2.OpenCSVSerde",
        stored_as="textfile",
        serde_properties=OrderedDict([("name1", "value1"), ("name2", "value2")]),
        location="oss://mock_endpoint/mock_bucket/mock_path/",
    )
    assert (
        ddl
        == textwrap.dedent(
            """
    CREATE EXTERNAL TABLE `test_external_table` (
      `id` BIGINT,
      `name` STRING
    )
    COMMENT 'TEST_COMMENT'
    PARTITIONED BY (
      `ds` STRING
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    WITH SERDEPROPERTIES (
      'name1' = 'value1',
      'name2' = 'value2'
    )
    STORED AS textfile
    LOCATION 'oss://mock_endpoint/mock_bucket/mock_path/'
    """
        ).strip()
    )


def test_create_table_ddl_with_auto_parts(odps_daily):
    odps = odps_daily
    test_table_name = tn("pyodps_t_tmp_table_ddl_with_auto_parts")
    odps.delete_table(test_table_name, if_exists=True)

    schema = TableSchema(
        columns=[
            Column(name="dt", type="datetime"),
            Column(name="test_col", type="string"),
        ],
        partitions=[
            Column(name="pt", type="string"),
            Column(
                name="pt_d", type="string", generate_expression="trunc_time(dt, 'hour')"
            ),
        ],
    )
    with pytest.raises(ValueError):
        odps.create_table(test_table_name, schema, lifecycle=10)

    schema = TableSchema(
        columns=[
            Column(name="dt", type="datetime"),
            Column(name="test_col", type="string"),
        ],
        partitions=[
            Column(
                name="pt_d", type="string", generate_expression="trunc_time(dt, 'hour')"
            ),
        ],
    )
    table = odps.create_table(test_table_name, schema, lifecycle=10)
    ddl = table.get_ddl()
    assert "AUTO PARTITIONED BY" in ddl

    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(
        test_table_name,
        ("dt datetime, test_col string", "trunc_time(dt, 'day') as pt"),
        lifecycle=10,
    )
    ddl = table.get_ddl()
    assert "AUTO PARTITIONED BY" in ddl


def test_create_delete_table(odps):
    test_table_name = tn("pyodps_t_tmp_create_table")
    schema = TableSchema.from_lists(
        ["id", "name"], ["bigint", "string"], ["ds"], ["string"]
    )

    tables = odps._project.tables

    tables.delete(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = tables.create(test_table_name, schema, lifecycle=10)

    assert table._getattr("owner") is None
    assert table.owner is not None

    assert table.name == test_table_name
    assert table.type == Table.Type.MANAGED_TABLE
    assert table.table_schema == schema
    assert table.lifecycle == 10

    tables.delete(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    str_schema = ("id bigint, name string", "ds string")
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
    test_table_name = tn("pyodps_t_tmp_create_table_with_chinese_columns")
    columns = [
        Column(name="序列", type="bigint", comment="注释"),
        Column(name=u"值", type=u"string", comment=u"注释2"),
        Column(name=u"值2", type=u"string", comment=u"注释'3"),
        Column(name=u"值3", type=u"string", comment=u"注释\"4"),
    ]
    partitions = [
        Partition(name="ds", type="string", comment="分区注释"),
        Partition(name=u"ds2", type=u"string", comment=u"分区注释2"),
    ]
    schema = TableSchema(columns=columns, partitions=partitions)

    columns_repr = (
        "[<column 序列, type bigint>, <column 值, type string>, "
        "<column 值2, type string>, <column 值3, type string>]"
    )
    partitions_repr = "[<partition ds, type string>, <partition ds2, type string>]"
    schema_repr = textwrap.dedent(
        """
    odps.Schema {
      序列    bigint      # 注释
      值      string      # 注释2
      值2     string      # 注释'3
      值3     string      # 注释"4
    }
    Partitions {
      ds      string      # 分区注释
      ds2     string      # 分区注释2
    }
    """
    ).strip()

    ddl_string_comment = textwrap.dedent(
        u"""
    CREATE TABLE `table_name` (
      `序列` BIGINT COMMENT '注释',
      `值` STRING COMMENT '注释2',
      `值2` STRING COMMENT '注释\\'3',
      `值3` STRING COMMENT '注释\\"4'
    )
    PARTITIONED BY (
      `ds` STRING COMMENT '分区注释',
      `ds2` STRING COMMENT '分区注释2'
    )"""
    ).strip()
    ddl_string = textwrap.dedent(
        u"""
    CREATE TABLE `table_name` (
      `序列` BIGINT,
      `值` STRING,
      `值2` STRING,
      `值3` STRING
    )
    PARTITIONED BY (
      `ds` STRING,
      `ds2` STRING
    )"""
    ).strip()

    assert repr(columns) == columns_repr
    assert repr(partitions) == partitions_repr
    assert repr(schema).strip() == schema_repr
    assert schema.get_table_ddl().strip() == ddl_string_comment
    assert schema.get_table_ddl(with_comments=False).strip() == ddl_string

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    table.reload()
    assert [to_text(col.name) for col in table.table_schema.columns] == [
        to_text(col.name) for col in schema.columns
    ]
    assert [to_text(col.comment) for col in table.table_schema.columns] == [
        to_text(col.comment) for col in schema.columns
    ]

    # test repr with not null columns
    schema[u"序列"].nullable = False
    columns_repr = (
        "[<column 序列, type bigint, not null>, <column 值, type string>, "
        "<column 值2, type string>, <column 值3, type string>]"
    )
    schema_repr = textwrap.dedent(
        """
    odps.Schema {
      序列    bigint      not null    # 注释
      值      string                  # 注释2
      值2     string                  # 注释'3
      值3     string                  # 注释"4
    }
    Partitions {
      ds      string      # 分区注释
      ds2     string      # 分区注释2
    }
    """
    ).strip()
    assert repr(columns) == columns_repr
    assert repr(schema).strip() == schema_repr


def test_create_transactional_table(odps):
    test_table_name = tn("pyodps_t_tmp_transactional")
    schema = TableSchema.from_lists(["key", "value"], ["string", "string"])
    schema["key"].comment = "comment_text"
    schema["value"].comment = "comment_text2"

    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(test_table_name, schema, transactional=True, lifecycle=1)
    table.reload()

    assert table.is_transactional
    assert "transactional" in table.get_ddl()
    assert "PRIMARY KEY" not in table.get_ddl()
    with table.open_writer() as writer:
        writer.write([["abc", "def"]])
    with table.open_reader() as reader:
        assert [r.values for r in reader] == [["abc", "def"]]
    table.drop()


def test_create_transactional_table_with_keys(odps):
    test_table_name = tn("pyodps_t_tmp_transactional_with_keys")
    schema = TableSchema.from_lists(["key", "value"], ["string", "string"])
    schema["key"].nullable = False
    schema["key"].comment = "comment_text"
    schema["value"].comment = "comment_text2"

    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(
        test_table_name, schema, transactional=True, primary_key="key", lifecycle=1
    )
    table.reload()
    assert not table.table_schema["key"].nullable
    assert table.is_transactional
    assert table.primary_key == ["key"]
    assert "transactional" in table.get_ddl()
    assert "PRIMARY KEY" in table.get_ddl()
    table.drop()

    options.sql.ignore_fields_not_null = True
    try:
        table = odps.create_table(test_table_name, schema)
        table.reload()
        assert table.table_schema["key"].nullable
        table.drop()
    finally:
        options.sql.ignore_fields_not_null = False


def test_create_tier_table(odps_with_storage_tier):
    odps = odps_with_storage_tier

    test_table_name = tn("pyodps_t_tmp_tiered")
    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(
        test_table_name, "col string", storage_tier="standard", lifecycle=1
    )
    assert table.storage_tier_info.storage_tier == StorageTier.STANDARD
    assert "storagetier" in table.get_ddl()
    table.set_storage_tier("low_frequency")
    assert table.storage_tier_info.storage_tier == StorageTier.LOWFREQENCY
    table.drop()


def test_create_clustered_table(odps):
    test_table_name = tn("pyodps_t_tmp_clustered")
    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    odps.execute_sql(
        "create table %s (a STRING, b STRING, c BIGINT) "
        "partitioned by (dt STRING) "
        "clustered by (c) sorted by (c) into 10 buckets lifecycle 1" % test_table_name
    )
    table = odps.get_table(test_table_name)
    assert table.cluster_info.cluster_type == ClusterType.HASH
    assert table.cluster_info.bucket_num == 10
    assert table.cluster_info.cluster_cols == ["c"]
    assert table.cluster_info.sort_cols[0].name == "c"
    assert table.cluster_info.sort_cols[0].order == ClusterSortOrder.ASC
    assert "CLUSTERED BY" in table.get_ddl()
    table.drop()

    test_table_name = tn("pyodps_t_tmp_range_clustered")
    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    odps.execute_sql(
        "create table %s (a STRING, b STRING, c BIGINT) "
        "partitioned by (dt STRING) "
        "range clustered by (c) sorted by (c) lifecycle 1" % test_table_name
    )
    table = odps.get_table(test_table_name)
    assert table.cluster_info.cluster_type == ClusterType.RANGE
    assert table.cluster_info.bucket_num == 0
    assert "RANGE CLUSTERED BY" in table.get_ddl()
    assert "BUCKETS" not in table.get_ddl()

    table.set_cluster_info(
        ClusterInfo(cluster_type=ClusterType.RANGE, cluster_cols=["c"])
    )
    assert table.cluster_info.cluster_type == ClusterType.RANGE
    table.reload()
    assert table.cluster_info.cluster_type == ClusterType.RANGE

    table.set_cluster_info(None)
    assert table.cluster_info is None
    table.reload()
    assert table.cluster_info is None

    table.drop()


def test_create_view(odps):
    test_table_name = tn("pyodps_t_tmp_view_source")
    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False
    table = odps.create_table(test_table_name, ("col string, col2 string", "pt string"))

    test_view_name = tn("pyodps_v_tmp_view")
    odps.delete_view(test_view_name, if_exists=True)
    odps.execute_sql(
        "create view %s comment 'comment_text' "
        "as select * from %s" % (test_view_name, test_table_name)
    )
    view = odps.get_table(test_view_name)
    assert view.type == Table.Type.VIRTUAL_VIEW
    assert view.comment == "comment_text"
    assert "CREATE VIEW" in view.get_ddl()
    view.drop()

    test_view_name = tn("pyodps_v_tmp_mt_view")
    odps.delete_materialized_view(test_view_name, if_exists=True)
    odps.execute_sql(
        "create materialized view %s "
        "disable rewrite "
        "partitioned on (pt) "
        "as select * from %s" % (test_view_name, test_table_name)
    )
    view = odps.get_table(test_view_name)
    assert view.type == Table.Type.MATERIALIZED_VIEW
    assert "CREATE MATERIALIZED VIEW" in view.get_ddl()
    view.drop()

    table.drop()


def test_run_sql_clear_cache(odps):
    test_table_name = tn("pyodps_t_tmp_statement_cache_clear")
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
    test_table_name = tn("pyodps_t_tmp_max_partition")
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
    assert tuple(table.get_max_partition(skip_empty=False).partition_spec.values()) == (
        "c",
        "e",
    )
    assert tuple(table.get_max_partition("pt1=a").partition_spec.values()) == ("a", "b")
    assert table.get_max_partition("pt1=c") is None
    assert table.get_max_partition("pt1=d") is None

    with pytest.raises(ValueError):
        table.get_max_partition("pt2=a")
    with pytest.raises(ValueError):
        table.get_max_partition("pt1=c,pt2=e")

    table.drop()


def test_schema_arg_backward_compat(odps):
    if six.PY2:
        from .. import Schema
    else:
        with pytest.deprecated_call():
            from .. import Schema

    columns = [
        Column(name="num", type="bigint", comment="the column"),
        Column(name="num2", type="double", comment="the column2"),
    ]
    schema = Schema(columns=columns)

    table_name = tn("test_backward_compat")

    with pytest.deprecated_call():
        table = odps.create_table(table_name, schema=schema, lifecycle=1)
    assert odps.exist_table(table_name)
    with pytest.deprecated_call():
        assert isinstance(table.schema, Schema)
    with pytest.deprecated_call():
        getattr(table, "last_modified_time")

    table.drop()


def test_alter_table_options(odps):
    table_name = tn("test_alter_table_options")
    table_name2 = tn("test_alter_table_options2")
    odps.delete_table(table_name, if_exists=True)
    odps.delete_table(table_name2, if_exists=True)

    test_table = odps.create_table(table_name, "col1 string, col2 bigint", lifecycle=3)

    last_modify_time = test_table.last_data_modified_time
    time.sleep(0.1)
    test_table.touch()
    assert last_modify_time != test_table.last_data_modified_time

    test_table.set_lifecycle(1)
    assert 1 == test_table.lifecycle
    test_table.reload()
    assert 1 == test_table.lifecycle

    test_table.set_comment("TABLE'COMMENT")
    assert "TABLE'COMMENT" == test_table.comment
    test_table.reload()
    assert "TABLE'COMMENT" == test_table.comment

    test_table.add_columns("col3 double")
    assert ["col1", "col2", "col3"] == [c.name for c in test_table.table_schema.columns]

    test_table.add_columns(odps_types.Column("col4", "datetime"))
    assert ["col1", "col2", "col3", "col4"] == [
        c.name for c in test_table.table_schema.columns
    ]

    test_table.delete_columns("col4")
    assert ["col1", "col2", "col3"] == [c.name for c in test_table.table_schema.columns]

    test_table.rename_column("col3", "col3_1")
    assert ["col1", "col2", "col3_1"] == [
        c.name for c in test_table.table_schema.columns
    ]

    test_table.rename_column("col3_1", "col3_2", comment="new'col'comment")
    assert ["col1", "col2", "col3_2"] == [
        c.name for c in test_table.table_schema.columns
    ]
    assert test_table.table_schema["col3_2"].comment == "new'col'comment"

    test_table.rename(table_name2)
    assert test_table.name == table_name2
    assert odps.exist_table(table_name2)

    test_table.set_owner(test_table.owner)

    odps.delete_table(table_name2, if_exists=True)
