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

"""End-to-end tests for odps.maxstorage read path against a real MaxCompute
service."""

import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None

from ..models.enums import SplitMode
from ..options import SplitOptions
from .conftest import _count_rows, _write_data

pytestmark = pytest.mark.skipif(pa is None, reason="Need pyarrow to run E2E tests")


def test_read_session_and_arrow_read(maxstorage_client):
    """Write data via tunnel, then read it back via MaxStorage read session."""
    client, table = maxstorage_client

    _write_data(table, "pt=test_read", [[i, i * 2, i * 3, i * 4] for i in range(100)])

    # ---- Create read session ----
    read_session = client.create_table_read_session(
        table,
        partitions=["pt=test_read"],
        split_options=SplitOptions(split_mode=SplitMode.SIZE),
    )
    assert read_session.id is not None
    assert len(read_session.splits) > 0

    # ---- Read all splits ----
    assert _count_rows(read_session) == 100


def test_record_reader(maxstorage_client):
    """Read rows via get_as_record_reader and verify Record values + count."""
    client, table = maxstorage_client

    _write_data(table, "pt=test_rr", [[i, i * 2, i * 3, i * 4] for i in range(50)])

    read_session = client.create_table_read_session(
        table,
        partitions=["pt=test_rr"],
        split_options=SplitOptions(split_mode=SplitMode.SIZE),
    )
    assert len(read_session.splits) > 0

    total_rows = 0
    for split in read_session.splits:
        reader = read_session.open_arrow_reader(split)
        rr = reader.get_as_record_reader()

        # Record reader exposes schema and count (total available, not cursor)
        assert [c.name for c in rr.schema.columns] == ["a", "b", "c", "d", "pt"]
        assert rr.count == reader.count

        for rec in rr:
            a = rec[0]
            assert rec[1] == a * 2
            assert rec[2] == a * 3
            assert rec[3] == a * 4
            total_rows += 1
        # count is the total available, unaffected by iteration
        assert rr.count == reader.count
        reader.close()

    assert total_rows == 50


def test_read_session_row_offset_split(maxstorage_client):
    """Read using ROW_OFFSET split mode with explicit split_number."""
    client, table = maxstorage_client

    row_count = 200
    _write_data(table, "pt=test_rowoffset", [[i, i, i, i] for i in range(row_count)])

    read_session = client.create_table_read_session(
        table,
        partitions=["pt=test_rowoffset"],
        split_options=SplitOptions(
            split_mode=SplitMode.ROW_OFFSET,
            split_number=50,
        ),
    )
    assert read_session.split_mode == SplitMode.ROW_OFFSET
    assert len(read_session.splits) >= 1

    assert _count_rows(read_session) == row_count


def test_preview_table(maxstorage_client):
    """Preview returns rows from the table."""
    client, table = maxstorage_client

    _write_data(table, "pt=test_preview", [[i, i, i, i] for i in range(20)])

    reader = client.preview_table(table, partition_spec="pt=test_preview", limit=10)
    total_rows = 0
    for batch in reader:
        total_rows += batch.num_rows
    reader.close()
    assert 0 < total_rows <= 10


def test_instance_read_session(maxstorage_client, odps):
    """Read SQL instance results via instance read session."""
    client, table = maxstorage_client

    _write_data(table, "pt=test_instance", [[i, i, i, i] for i in range(10)])
    instance = odps.execute_sql(
        f"SELECT * FROM {table.name} WHERE pt='test_instance' LIMIT 5"
    )

    read_session = client.create_instance_read_session(instance)
    assert read_session.id is not None

    reader = read_session.open_arrow_reader()
    total_rows = 0
    for batch in reader:
        total_rows += batch.num_rows
    reader.close()
    assert total_rows == 5
