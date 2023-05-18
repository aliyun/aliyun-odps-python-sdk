# encoding: utf-8
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

from __future__ import print_function

import pytest

from ... import types as odps_types
from ...df import DataFrame
from ...df.backends.odpssql.types import df_schema_to_odps_schema
from ...models import Partition, TableSchema
from ...tests.core import tn
from .base import MLTestUtil

IRIS_TABLE = tn('pyodps_test_ml_iris_persist')
SIMPLE_PERSIST_TABLE = tn('pyodps_test_ml_simple_persist_table')
EXISTING_PERSIST_TABLE = tn('pyodps_test_ml_existing_persist_table')
STATIC_PART_TABLE = tn('pyodps_test_ml_static_part_table')
DYNAMIC_PART_TABLE = tn('pyodps_test_ml_dynamic_part_table')


@pytest.fixture
def utils(odps, tunnel):
    return MLTestUtil(odps, tunnel)


def test_simple_persist(odps, utils):
    utils.create_iris(IRIS_TABLE)
    df = DataFrame(odps.get_table(IRIS_TABLE))
    df.append_id().persist(SIMPLE_PERSIST_TABLE, lifecycle=1, drop_table=True)
    assert odps.exist_table(SIMPLE_PERSIST_TABLE) is True


def test_existing_persist(odps, utils):
    utils.create_iris(IRIS_TABLE)
    df = DataFrame(odps.get_table(IRIS_TABLE)).append_id()

    odps_schema = df_schema_to_odps_schema(df.schema)
    cols = list(reversed(odps_schema.columns))
    odps_schema = TableSchema.from_lists([c.name for c in cols], [c.type for c in cols])

    odps.delete_table(EXISTING_PERSIST_TABLE, if_exists=True)
    odps.create_table(EXISTING_PERSIST_TABLE, odps_schema)
    df.persist(EXISTING_PERSIST_TABLE)


def test_static_partition(odps, utils):
    utils.create_iris(IRIS_TABLE)
    df = DataFrame(odps.get_table(IRIS_TABLE))
    id_df = df.append_id()

    src_schema = df_schema_to_odps_schema(id_df.schema)
    schema = TableSchema(
        columns=src_schema.simple_columns, partitions=[Partition(name='ds', type=odps_types.string)]
    )
    odps.delete_table(STATIC_PART_TABLE, if_exists=True)
    dest_table = odps.create_table(STATIC_PART_TABLE, schema, lifecycle=1)

    id_df.persist(STATIC_PART_TABLE, partition='ds=20170314', lifecycle=1)
    assert dest_table.exist_partition('ds=20170314') is True


def test_dynamic_partition(odps, utils):
    utils.create_iris(IRIS_TABLE)
    df = DataFrame(odps.get_table(IRIS_TABLE))
    id_df = df.append_id()
    odps.delete_table(DYNAMIC_PART_TABLE, if_exists=True)
    id_df.persist(DYNAMIC_PART_TABLE, partitions='category', lifecycle=1, drop_table=True)

    assert odps.exist_table(DYNAMIC_PART_TABLE) is True
    t = odps.get_table(DYNAMIC_PART_TABLE)
    assert 'category' in [pt.name for pt in t.table_schema.partitions]
