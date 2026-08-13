# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import pytest

from ...config import options
from ...tests.core import get_test_unique_name, tn
from ..base import MaxStorageClient

_BLOB_HINTS = {
    "odps.sql.type.system.odps2": "true",
    "odps.table.append2.enable": "true",
}
_BLOB_TABLE_PROPS = {"table.format.version": "2", "transactional": "true"}


def _write_data(table, partition_spec, rows):
    """Write rows to a partition via tunnel, creating the partition if needed."""
    table.create_partition(partition_spec, if_not_exists=True)
    with table.open_writer(partition=partition_spec, overwrite=True) as writer:
        writer.write(rows)


def _count_rows(read_session):
    """Count total rows across all splits in a read session."""
    total = 0
    for split in read_session.splits:
        reader = read_session.open_arrow_reader(split)
        for batch in reader:
            total += batch.num_rows
        reader.close()
    return total


@pytest.fixture
def maxstorage_client(odps):
    """MaxStorageClient wired to a fresh 4-column BIGINT table with a partition.

    The storage endpoint is resolved from the tunnel endpoint discovered by
    ``BaseTunnel._get_tunnel_server`` — no explicit endpoint needed.
    """
    prev_enable_schema = options.enable_schema
    options.enable_schema = True

    test_table_name = tn(f"test_maxstorage_{get_test_unique_name(5)}")
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(
        test_table_name,
        ("a BIGINT, b BIGINT, c BIGINT, d BIGINT", "pt string"),
        if_not_exists=True,
    )
    try:
        yield MaxStorageClient(odps), table
    finally:
        table.drop(async_=True)
        options.enable_schema = prev_enable_schema


@pytest.fixture
def maxstorage_blob_client(odps):
    """MaxStorageClient (api_version=3) wired to a BLOB table."""
    prev_enable_schema = options.enable_schema
    options.enable_schema = True

    test_table_name = tn(f"test_maxstorage_blob_{get_test_unique_name(5)}")
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(
        test_table_name,
        ("a BIGINT, b BLOB", "pt string"),
        hints=_BLOB_HINTS,
        table_properties=_BLOB_TABLE_PROPS,
        if_not_exists=True,
    )
    try:
        yield MaxStorageClient(odps, api_version="3"), table
    finally:
        table.drop(async_=True)
        options.enable_schema = prev_enable_schema


@pytest.fixture
def maxstorage_nested_blob_client(odps):
    """MaxStorageClient (api_version=3) wired to a table with nested BLOB columns."""
    prev_enable_schema = options.enable_schema
    options.enable_schema = True

    test_table_name = tn(f"test_maxstorage_nblob_{get_test_unique_name(5)}")
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(
        test_table_name,
        ("a BIGINT, b ARRAY<BLOB>", "pt string"),
        hints=_BLOB_HINTS,
        table_properties=_BLOB_TABLE_PROPS,
        if_not_exists=True,
    )
    try:
        yield MaxStorageClient(odps, api_version="3"), table
    finally:
        table.drop(async_=True)
        options.enable_schema = prev_enable_schema


@pytest.fixture
def maxstorage_delta_client(odps):
    """MaxStorageClient wired to a transactional PK table (Delta table).

    The table has a BIGINT PK ``id`` and two value columns, no partition.
    ``__operation`` is a system column on Delta tables.
    """
    prev_enable_schema = options.enable_schema
    options.enable_schema = True

    test_table_name = tn(f"test_maxstorage_delta_{get_test_unique_name(5)}")
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(
        test_table_name,
        "id BIGINT NOT NULL, v1 BIGINT, v2 STRING",
        transactional=True,
        primary_key=["id"],
        hints={"odps.sql.type.system.odps2": "true"},
        if_not_exists=True,
    )
    try:
        yield MaxStorageClient(odps), table
    finally:
        table.drop(async_=True)
        options.enable_schema = prev_enable_schema


@pytest.fixture
def maxstorage_delta_blob_client(odps):
    """MaxStorageClient (api_version=3) wired to a transactional PK table
    with a BLOB column (Delta + BLOB)."""
    prev_enable_schema = options.enable_schema
    options.enable_schema = True

    test_table_name = tn(f"test_maxstorage_dblob_{get_test_unique_name(5)}")
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(
        test_table_name,
        "id BIGINT NOT NULL, b BLOB",
        transactional=True,
        primary_key=["id"],
        hints=_BLOB_HINTS,
        table_properties=_BLOB_TABLE_PROPS,
        if_not_exists=True,
    )
    try:
        yield MaxStorageClient(odps, api_version="3"), table
    finally:
        table.drop(async_=True)
        options.enable_schema = prev_enable_schema
