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

"""Unit tests for odps.maxstorage read path: ArrowReader, extended Arrow IPC
streams, splits, sessions."""

import io

import mock
import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None
    pytestmark = pytest.mark.skip("Need pyarrow to run this test")

from ..io.arrow_reader import ExtendedArrowStreamReader
from ..models.requests import CreateTableReadSessionRequest
from ..models.responses import CreateTableReadSessionResponse
from ..models.schema import ReadSchema
from ..read.reader import ArrowReader
from ..read.record_reader import ArrowRecordReader
from ..read.table_session import TableReadSession

_ASYNC = pytest.mark.parametrize("async_read", [False, True])

DATA_TAG = b"\x01"
METRICS_TAG = b"\x02"

_SESSION_RESPONSE = {
    "SessionId": "s1",
    "SessionStatus": "NORMAL",
    "SplitsCount": 1,
    "SplitMode": "Size",
}

# ---------------------------------------------------------------------------
# helpers
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


def _value_stream(batches):
    """IPC bytes of an ``a`` int64 column over ``batches`` (lists of rows)."""
    schema = pa.schema([("a", pa.int64())])
    return _make_arrow_stream(
        schema,
        [
            pa.RecordBatch.from_arrays([pa.array(rows, pa.int64())], schema=schema)
            for rows in batches
        ],
    )


def _extended_wire(*frames):
    """Wire body of tagged substreams.

    Each frame is ``(DATA_TAG, batches)`` for a data substream or
    ``(METRICS_TAG, metadata)`` for a metrics substream.
    """
    sink = io.BytesIO()
    for tag, payload in frames:
        sink.write(tag)
        if tag == METRICS_TAG:
            sink.write(_make_arrow_stream(pa.schema([], metadata=payload), []))
        else:
            sink.write(_value_stream(payload))
    return sink.getvalue()


def _extended_reader(*frames):
    """An :class:`ExtendedArrowStreamReader` over ``frames``."""
    return ExtendedArrowStreamReader(_make_mock_resp(_extended_wire(*frames)))


def _arrow_reader(batches, async_read=False, request_id="", **kwargs):
    """An :class:`ArrowReader` over an ``a`` int64 stream of ``batches``."""
    return ArrowReader(
        _make_mock_resp(_value_stream(batches), request_id=request_id),
        _odps_schema(),
        async_read=async_read,
        request_id=request_id,
        **kwargs,
    )


def _odps_schema():
    """The ODPS schema matching :func:`_value_stream`'s wire schema."""
    return ReadSchema.from_dict(
        {"DataColumns": [{"Name": "a", "Type": "bigint", "Comment": ""}]}
    )


def _batch_rows(batches):
    """``(first value, row count)`` of every batch."""
    return [(b.column("a")[0].as_py(), b.num_rows) for b in batches]


# ---------------------------------------------------------------------------
# ArrowReader
# ---------------------------------------------------------------------------


@_ASYNC
@pytest.mark.parametrize(
    "batches,expected_rows",
    [
        pytest.param([[1, 2, 3]], [3], id="single-batch"),
        pytest.param([[1, 2], [3, 4]], [2, 2], id="multiple-batches"),
        pytest.param([], [], id="schema-only-stream"),
    ],
)
def test_arrow_reader_iterates_batches(async_read, batches, expected_rows):
    """The schema is readable up front and iteration ends with ``None``."""
    reader = _arrow_reader(batches, async_read=async_read)
    assert reader.arrow_schema == pa.schema([("a", pa.int64())])
    assert [b.num_rows for b in reader] == expected_rows
    assert reader.read() is None
    reader.close()


@_ASYNC
def test_arrow_reader_count_and_stream_metadata(async_read):
    """``count`` is the fixed total available, not an iteration cursor."""
    reader = _arrow_reader(
        [[1, 2], [3]], async_read=async_read, request_id="req-1", count=3
    )
    assert (reader.count, reader.get_status(), reader.get_request_id()) == (
        3,
        "OK",
        "req-1",
    )
    assert [b.num_rows for b in reader] == [2, 1]
    assert reader.count == 3
    reader.close()

    default = _arrow_reader([[1]], async_read=async_read)
    assert default.count == 0  # no total given
    default.close()


def test_record_reader_count_delegates_to_arrow_reader():
    """ArrowRecordReader.count mirrors the underlying ArrowReader.count."""
    reader = _arrow_reader([[1, 2, 3]], count=3)
    rr = reader.get_as_record_reader()
    assert isinstance(rr, ArrowRecordReader)
    assert rr.count == 3 == reader.count
    assert len(list(rr)) == 3
    assert rr.count == 3 == reader.count  # iteration does not mutate count
    reader.close()


# ---------------------------------------------------------------------------
# Extended Arrow IPC (tagged substreams)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frames,expected_batches,expected_metrics",
    [
        pytest.param([], [], {}, id="empty-input"),
        pytest.param(
            [
                (DATA_TAG, [[42]]),
                (
                    METRICS_TAG,
                    {
                        b"server_processing_duration_us": b"10",
                        b"rate_limit_duration_us": b"5",
                        b"other_key": b"abc",
                    },
                ),
                (DATA_TAG, [[43]]),
            ],
            [(42, 1), (43, 1)],
            {
                "server_processing_duration_us": "10",
                "rate_limit_duration_us": "5",
                "other_key": "abc",
            },
            id="data-metrics-data",
        ),
        pytest.param(
            [
                (METRICS_TAG, {b"server_processing_duration_us": b"10"}),
                (
                    METRICS_TAG,
                    {
                        b"server_processing_duration_us": b"20",
                        b"rate_limit_duration_us": b"7",
                        b"replaced_key": b"first",
                    },
                ),
                (METRICS_TAG, {b"replaced_key": b"second"}),
                (DATA_TAG, [[7]]),
            ],
            [(7, 1)],
            {
                "server_processing_duration_us": "30",
                "rate_limit_duration_us": "7",
                "replaced_key": "second",
            },
            id="durations-summed-other-keys-replaced",
        ),
        pytest.param(
            [
                (DATA_TAG, [[1]]),
                (METRICS_TAG, {b"server_processing_duration_us": b"3"}),
                (DATA_TAG, [[2]]),
                (DATA_TAG, [[3]]),
            ],
            [(1, 1), (2, 1), (3, 1)],
            {"server_processing_duration_us": "3"},
            id="metrics-between-data",
        ),
        pytest.param(
            [(DATA_TAG, [[1, 2], [1, 2]])],
            [(1, 2), (1, 2)],
            {},
            id="multiple-batches-per-substream",
        ),
    ],
)
def test_extended_reader_tagged_substreams(frames, expected_batches, expected_metrics):
    """Data substreams yield rows; metrics frames are consumed, never rows."""
    reader = _extended_reader(*frames)
    assert _batch_rows(list(reader)) == expected_batches
    assert reader.metrics_metadata == expected_metrics
    reader.close()


def test_extended_reader_tolerates_non_numeric_duration_metric():
    reader = _extended_reader(
        (METRICS_TAG, {b"server_processing_duration_us": b"10"}),
        (METRICS_TAG, {b"server_processing_duration_us": b"1.5ms"}),
        (DATA_TAG, [[7]]),
    )
    # A malformed duration keeps the raw value instead of aborting the read.
    assert _batch_rows(list(reader)) == [(7, 1)]
    assert reader.metrics_metadata == {"server_processing_duration_us": "1.5ms"}
    reader.close()


def test_extended_reader_arrow_schema_from_first_data_stream():
    reader = _extended_reader((METRICS_TAG, {b"k": b"v"}), (DATA_TAG, [[42]]))
    assert reader.arrow_schema.names == ["a"]  # resolved past leading metrics frames
    assert _batch_rows(list(reader)) == [(42, 1)]
    reader.close()


@_ASYNC
def test_arrow_reader_extended_streams(async_read):
    wire = _extended_wire(
        (DATA_TAG, [[42]]),
        (METRICS_TAG, {b"server_processing_duration_us": b"10"}),
        (DATA_TAG, [[43]]),
    )
    reader = ArrowReader(
        _make_mock_resp(wire), schema=None, extended=True, async_read=async_read
    )
    assert _batch_rows(list(reader)) == [(42, 1), (43, 1)]
    assert reader.metrics_metadata == {"server_processing_duration_us": "10"}
    reader.close()


@_ASYNC
def test_arrow_reader_ordinary_streams_have_no_metrics(async_read):
    reader = _arrow_reader([[1]], async_read=async_read)
    assert reader.metrics_metadata is None
    assert _batch_rows(list(reader)) == [(1, 1)]
    reader.close()


# ---------------------------------------------------------------------------
# TableReadSession
# ---------------------------------------------------------------------------


def _make_table_session(payload, stream_bytes=b"", **request_kwargs):
    stub = mock.Mock()
    stub.create_table_read_session.return_value = (
        CreateTableReadSessionResponse.from_dict(payload)
    )
    stub.create_table_read_stream.return_value = _make_mock_resp(stream_bytes)
    return TableReadSession(
        stub, mock.Mock(), CreateTableReadSessionRequest(**request_kwargs)
    )


def test_table_read_session_extended_read_wiring():
    session = _make_table_session(
        {
            **_SESSION_RESPONSE,
            "ExtendedArrowIPCEnabled": True,
            "LatestTxnId": "txn-9",
        },
        _extended_wire(
            (DATA_TAG, [[42]]),
            (METRICS_TAG, {b"server_processing_duration_us": b"10"}),
        ),
        enable_extended_arrow_ipc=True,
    )
    assert (session.extended_arrow_ipc_enabled, session.latest_txn_id) == (
        True,
        "txn-9",
    )

    reader = session.open_arrow_reader(session.splits[0])
    assert _batch_rows(list(reader)) == [(42, 1)]
    assert reader.metrics_metadata == {"server_processing_duration_us": "10"}
    reader.close()


def test_table_read_session_v2_fields_default_when_absent():
    session = _make_table_session(dict(_SESSION_RESPONSE))
    assert (session.extended_arrow_ipc_enabled, session.latest_txn_id) == (False, None)


@pytest.mark.parametrize(
    "payload,expected",
    [
        pytest.param({}, -1, id="absent-count-is-unknown"),
        pytest.param({"RecordCount": 5}, 5, id="reported-count"),
        pytest.param({"RecordCount": None}, -1, id="null-count-is-unknown"),
    ],
)
def test_table_read_session_record_count(payload, expected):
    session = _make_table_session({**_SESSION_RESPONSE, **payload})
    assert session.record_count == expected
