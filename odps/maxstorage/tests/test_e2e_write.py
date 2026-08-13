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

"""End-to-end tests for odps.maxstorage write path against a real MaxCompute
service."""

import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None

from ..models.enums import WriteMode
from .conftest import _count_rows

pytestmark = pytest.mark.skipif(pa is None, reason="Need pyarrow to run E2E tests")


def _make_batch(num_rows=100, offset=0):
    """Build a 4-column BIGINT RecordBatch matching the test table schema."""
    return pa.RecordBatch.from_arrays(
        [
            pa.array(list(range(offset, offset + num_rows)), pa.int64()),
            pa.array(list(range(offset, offset + num_rows)), pa.int64()),
            pa.array(list(range(offset, offset + num_rows)), pa.int64()),
            pa.array(list(range(offset, offset + num_rows)), pa.int64()),
        ],
        names=["a", "b", "c", "d"],
    )


def test_write_and_read_back(maxstorage_client):
    """Write via MaxStorage write session, read back via MaxStorage read session."""
    client, table = maxstorage_client

    # ---- Write session ----
    write_session = client.create_table_write_session(
        table,
        partition_spec="pt=test_write_read",
        write_mode=WriteMode.BATCH,
    )
    assert write_session.id is not None

    writer = write_session.open_arrow_writer("stream-0")
    writer.write_batch(_make_batch(100))
    writer.close()
    write_session.commit()

    # ---- Read back ----
    read_session = client.create_table_read_session(
        table,
        partitions=["pt=test_write_read"],
    )
    assert len(read_session.splits) > 0

    assert _count_rows(read_session) == 100


def test_write_multiple_batches(maxstorage_client):
    """Write multiple Arrow batches in a single stream."""
    client, table = maxstorage_client

    write_session = client.create_table_write_session(
        table,
        partition_spec="pt=test_multi_batch",
    )
    writer = write_session.open_arrow_writer("stream-0")

    for i in range(5):
        writer.write_batch(_make_batch(20, offset=i * 20))

    writer.close()
    write_session.commit()

    # Read back and verify row count
    read_session = client.create_table_read_session(
        table,
        partitions=["pt=test_multi_batch"],
    )
    assert _count_rows(read_session) == 100


def test_write_compressed(maxstorage_client):
    """Write with ZSTD compression, then read back."""
    client, table = maxstorage_client

    write_session = client.create_table_write_session(
        table,
        partition_spec="pt=test_write_zstd",
    )
    writer = write_session.open_arrow_writer("stream-0", compress_algo="zstd")
    writer.write_batch(_make_batch(50))
    writer.close()
    write_session.commit()

    read_session = client.create_table_read_session(
        table,
        partitions=["pt=test_write_zstd"],
    )
    assert _count_rows(read_session) == 50


def test_abort_write_session(maxstorage_client):
    """Abort discards all written data without error."""
    client, table = maxstorage_client

    write_session = client.create_table_write_session(
        table,
        partition_spec="pt=test_abort",
    )
    writer = write_session.open_arrow_writer("stream-0")
    writer.write_batch(_make_batch(10))
    writer.close()

    # Abort should not raise
    write_session.abort()


def test_write_multiple_partitions(maxstorage_client):
    """Write to two different partitions in separate sessions."""
    client, table = maxstorage_client

    for pt_val in ["p1", "p2"]:
        write_session = client.create_table_write_session(
            table,
            partition_spec=f"pt={pt_val}",
        )
        writer = write_session.open_arrow_writer("stream-0")
        writer.write_batch(_make_batch(30))
        writer.close()
        write_session.commit()

    read_session = client.create_table_read_session(
        table,
        partitions=["pt=p1", "pt=p2"],
    )
    assert _count_rows(read_session) == 60


def test_record_writer(maxstorage_client):
    """Write via the record-oriented writer interface on a plain table.

    No ``auto_upload_blobs`` — the table has no BLOB columns, so a plain
    :class:`TableArrowWriter` suffices.
    """
    client, table = maxstorage_client

    write_session = client.create_table_write_session(
        table,
        partition_spec="pt=test_record_writer",
    )
    writer = write_session.open_arrow_writer("stream-0")
    record_writer = writer.get_as_record_writer(row_count_per_batch=10)

    for i in range(25):
        record_writer.write([i, i * 2, i * 3, i * 4])

    record_writer.close()
    write_session.commit()
    read_session = client.create_table_read_session(
        table,
        partitions=["pt=test_record_writer"],
    )
    assert _count_rows(read_session) == 25


def test_delta_table_record_writer(maxstorage_delta_client):
    """Write UPSERTs and DELETEs to a Delta (PK) table via the record writer.

    Exercises ``DeltaTableRecordWriter`` — ``write()`` stamps
    ``__operation='U'`` and ``delete()`` stamps ``__operation='D'`` per
    record.  Both operations are freely interleaved on a single writer
    instance.
    """
    client, table = maxstorage_delta_client

    write_session = client.create_table_write_session(table)
    writer = write_session.open_arrow_writer("stream-0")
    rw = writer.get_as_record_writer(row_count_per_batch=10)

    # UPSERT three rows
    rw.write([1, 100, "alice"])
    rw.write([2, 200, "bob"])
    rw.write([3, 300, "carol"])
    # UPSERT on existing key → overwrites
    rw.write([2, 999, "bob-updated"])
    # DELETE key 3
    rw.delete([3, None, None])

    rw.close()
    write_session.commit()

    # Read back via SQL (PK tables require SELECT)
    inst = client._odps.execute_sql(f"SELECT * FROM {table.name}")
    with inst.open_reader() as reader:
        rows = sorted(([v for v in rec.values] for rec in reader), key=lambda r: r[0])

    assert rows == [
        [1, 100, "alice"],
        [2, 999, "bob-updated"],
    ]
