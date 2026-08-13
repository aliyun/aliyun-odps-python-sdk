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

"""Unit tests for odps.maxstorage read path: ArrowReader, splits, sessions."""

import io

import mock
import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None
    pytestmark = pytest.mark.skip("Need pyarrow to run this test")

from ..models.schema import ReadSchema
from ..read.reader import ArrowReader
from ..read.record_reader import ArrowRecordReader

_ASYNC = pytest.mark.parametrize("async_read", [False, True])

# ---------------------------------------------------------------------------
# ArrowReader
# ---------------------------------------------------------------------------


def _make_arrow_stream(schema, batches):
    sink = io.BytesIO()
    writer = pa.ipc.new_stream(sink, schema)
    for batch in batches:
        writer.write_batch(batch)
    writer.close()
    return sink.getvalue()


def _make_mock_resp(stream_bytes, request_id=""):
    resp = mock.Mock()
    resp.headers = {"x-odps-request-id": request_id} if request_id else {}
    resp.raw = pa.BufferReader(stream_bytes)
    resp.status_code = 200
    resp.close = mock.Mock()
    return resp


@_ASYNC
def test_arrow_reader_single_batch(async_read):
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], pa.int64()), pa.array(["a", "b", "c"], pa.string())],
        schema=schema,
    )
    stream_bytes = _make_arrow_stream(schema, [batch])
    resp = _make_mock_resp(stream_bytes, request_id="req-1")

    reader = ArrowReader(
        resp, schema, compress_option=None, request_id="req-1", async_read=async_read
    )
    batches = list(reader)
    assert len(batches) == 1
    assert batches[0].num_rows == 3
    assert reader.get_status() == "OK"
    assert reader.get_request_id() == "req-1"
    reader.close()


@_ASYNC
def test_arrow_reader_multiple_batches(async_read):
    schema = pa.schema([("id", pa.int64())])
    batch1 = pa.RecordBatch.from_arrays([pa.array([1, 2], pa.int64())], schema=schema)
    batch2 = pa.RecordBatch.from_arrays([pa.array([3, 4], pa.int64())], schema=schema)
    stream_bytes = _make_arrow_stream(schema, [batch1, batch2])
    resp = _make_mock_resp(stream_bytes)

    reader = ArrowReader(resp, schema, compress_option=None, async_read=async_read)
    batches = list(reader)
    assert [b.num_rows for b in batches] == [2, 2]
    reader.close()


@_ASYNC
def test_arrow_reader_schema_property(async_read):
    schema = pa.schema([("id", pa.int64())])
    batch = pa.RecordBatch.from_arrays([pa.array([1], pa.int64())], schema=schema)
    stream_bytes = _make_arrow_stream(schema, [batch])
    resp = _make_mock_resp(stream_bytes)

    reader = ArrowReader(resp, schema, compress_option=None, async_read=async_read)
    assert reader.arrow_schema == schema
    reader.close()


@_ASYNC
def test_arrow_reader_read_returns_none_at_end(async_read):
    schema = pa.schema([("id", pa.int64())])
    batch = pa.RecordBatch.from_arrays([pa.array([1], pa.int64())], schema=schema)
    stream_bytes = _make_arrow_stream(schema, [batch])
    resp = _make_mock_resp(stream_bytes)

    reader = ArrowReader(resp, schema, compress_option=None, async_read=async_read)
    first = reader.read()
    assert first is not None and first.num_rows == 1
    assert reader.read() is None  # end of stream
    reader.close()


@_ASYNC
def test_arrow_reader_count_is_total_available(async_read):
    schema = pa.schema([("id", pa.int64())])
    batch1 = pa.RecordBatch.from_arrays([pa.array([1, 2], pa.int64())], schema=schema)
    batch2 = pa.RecordBatch.from_arrays([pa.array([3], pa.int64())], schema=schema)
    stream_bytes = _make_arrow_stream(schema, [batch1, batch2])
    resp = _make_mock_resp(stream_bytes)

    # count defaults to 0 when not given
    reader = ArrowReader(resp, schema, compress_option=None, async_read=async_read)
    assert reader.count == 0
    reader.close()

    # count is the fixed total, unaffected by iteration
    resp = _make_mock_resp(stream_bytes)
    reader = ArrowReader(
        resp, schema, compress_option=None, count=3, async_read=async_read
    )
    assert reader.count == 3
    list(reader)
    assert reader.count == 3
    reader.close()


def _odps_schema():
    return ReadSchema.from_dict(
        {"DataColumns": [{"Name": "id", "Type": "bigint", "Comment": ""}]}
    )


def test_record_reader_count_delegates_to_arrow_reader():
    """ArrowRecordReader.count mirrors the underlying ArrowReader.count."""
    arrow_schema = pa.schema([("id", pa.int64())])
    odps_schema = _odps_schema()
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3], pa.int64())], schema=arrow_schema
    )
    stream_bytes = _make_arrow_stream(arrow_schema, [batch])
    resp = _make_mock_resp(stream_bytes)

    reader = ArrowReader(resp, odps_schema, compress_option=None, count=3)
    rr = reader.get_as_record_reader()
    assert isinstance(rr, ArrowRecordReader)
    assert rr.count == 3 == reader.count

    # iteration does not mutate count (total available, not cursor)
    rows = list(rr)
    assert len(rows) == 3
    assert rr.count == 3 == reader.count
    reader.close()
