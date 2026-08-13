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

"""Unit tests for odps.maxstorage write path: session, writer, record_writer."""

import base64
import io
import json
import struct
import threading

import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None
    pytestmark = pytest.mark.skip("Need pyarrow to run this test")

from ..io.arrow_writer import RawArrowRequestBody, serialize_batch
from ..io.compress import CompressOption
from ..models.enums import DataFormat, WriteMode
from ..models.identifier import TableIdentifier
from ..models.responses import (
    CreateTableWriteSessionResponse,
    CreateWriteStreamResponse,
    GetTableWriteSessionResponse,
    GetWriteStreamResponse,
    WriteStreamResponse,
)
from ..models.schema import WriteSchema
from ..write.record_writer import AppendTableRecordWriter, DeltaTableRecordWriter
from ..write.session import TableWriteSession
from ..write.writer import TableArrowBlobUploadWriter
from ._helpers import CloseTrackingStream, TrackedStream

# ---------------------------------------------------------------------------
# Fake stub
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, data=None, headers=None):
        self._data = data if data is not None else b""
        self.headers = headers or {}
        self.content = self._data
        self.status_code = 200

    def json(self):
        return json.loads(self._data) if self._data else {}


def _make_write_table_schema():
    return {
        "DataColumns": [
            {
                "columnType": {
                    "MemberName": "id",
                    "ColumnId": 0,
                    "Nullable": True,
                    "Type": 0,
                }
            },
            {
                "columnType": {
                    "MemberName": "name",
                    "ColumnId": 1,
                    "Nullable": True,
                    "Type": 4,
                }
            },
            {
                "columnType": {
                    "MemberName": "data",
                    "ColumnId": 2,
                    "Nullable": True,
                    "Type": 22,
                }
            },
        ],
        "SystemColumns": [],
    }


def _make_delta_table_schema():
    """Write schema for a PK/delta table (includes ``__operation``)."""
    schema = _make_write_table_schema()
    schema["SystemColumns"] = [
        {
            "columnType": {
                "MemberName": "__operation",
                "ColumnId": 3,
                "Nullable": True,
                "Type": 6,
            }
        }
    ]
    return schema


class FakeWriteStub:
    def __init__(self):
        self.calls = []
        self.route_token = "rt-123"
        self.write_record_counts = []

    def create_table_write_session(self, table_id, request, write_mode):
        self.calls.append(("create_session",))
        r = CreateTableWriteSessionResponse.from_dict({"SessionId": "sess-1"})
        r.route_token = self.route_token
        return r

    def get_table_write_session(self, table_id, session_id, write_mode):
        self.calls.append(("get_session", table_id, session_id))
        r = GetTableWriteSessionResponse.from_dict({})
        r.route_token = self.route_token
        return r

    def create_table_write_stream(
        self, table_id, session_id, request, route_token, write_mode
    ):
        self.calls.append(("create_stream",))
        return CreateWriteStreamResponse.from_dict(
            {
                "TableId": "t1",
                "SchemaVersion": 1,
                "TableSchema": _make_write_table_schema(),
            }
        )

    def get_write_stream(self, table_id, request, route_token, write_mode):
        return GetWriteStreamResponse.from_dict({})

    def write_table(
        self,
        table_id,
        session_id,
        stream_id,
        stream_version,
        record_count,
        body,
        route_token,
        **kwargs
    ):
        self.calls.append(("write_table",))
        self.write_record_counts.append(record_count)
        return _FakeResp(
            data=json.dumps({"StagingId": "stg-1"}).encode(),
            headers={
                "x-odps-request-id": "req-1",
                "x-odps-max-storage-route-token": self.route_token,
            },
        )

    def parse_write_stream_response(self, resp):
        return WriteStreamResponse.from_dict(resp.json())

    def close_write_stream(self, table_id, request, route_token, write_mode):
        self.calls.append(("close_stream",))

    def commit_table_write_session(self, *a, **kw):
        self.calls.append(("commit",))

    def abort_table_write_session(self, *a, **kw):
        self.calls.append(("abort",))

    def table_batch_write_blob(self, table_id, params, data, route_token, **kwargs):
        # Consume the generator from RequestsIO to unblock the streaming pipeline.
        consumed = b"".join(data) if not isinstance(data, (bytes, bytearray)) else data
        self.calls.append(("batch_blob",))
        # Count blob frames in the consumed data to return matching refs.
        # Each frame is [HeaderLen LE64][Header JSON][DataLen LE64][Data]...
        num_items = self._count_blob_frames(consumed)
        refs = [base64.b64encode(b"ref%d" % i).decode() for i in range(num_items)]
        return _FakeResp(data=json.dumps({"BlobReferences": refs}).encode())

    @staticmethod
    def _count_blob_frames(data):
        """Count blob write frames in a streamed batch body."""
        count = 0
        offset = 0
        while offset + 8 <= len(data):
            header_len = struct.unpack("<q", data[offset : offset + 8])[0]
            offset += 8
            if header_len < 0 or offset + header_len > len(data):
                break
            offset += header_len  # skip header JSON
            if offset + 8 > len(data):
                break
            data_len = struct.unpack("<q", data[offset : offset + 8])[0]
            offset += 8
            if data_len < 0 or offset + data_len > len(data):
                break
            offset += data_len  # skip data
            # Skip footer if present
            if offset + 8 <= len(data):
                footer_len = struct.unpack("<q", data[offset : offset + 8])[0]
                if 0 <= footer_len and offset + 8 + footer_len <= len(data):
                    offset += 8 + footer_len
            count += 1
        return count


@pytest.fixture
def table_id():
    return TableIdentifier("proj", "tbl", "sch")


@pytest.fixture
def arrow_schema():
    return pa.schema([("id", pa.int64()), ("name", pa.string()), ("data", pa.binary())])


@pytest.fixture
def arrow_batch(arrow_schema):
    return pa.RecordBatch.from_arrays(
        [
            pa.array([1, 2], pa.int64()),
            pa.array(["a", "b"], pa.string()),
            pa.array([b"x", b"y"], pa.binary()),
        ],
        schema=arrow_schema,
    )


# ---------------------------------------------------------------------------
# Session creation paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "write_mode,session_id,expected_id,expect_call",
    [
        (WriteMode.BATCH, None, "sess-1", "create_session"),
        (WriteMode.STREAMING, None, "default", None),
        (WriteMode.BATCH, "existing", "existing", "get_session"),
    ],
)
def test_session_creation_paths(
    table_id, write_mode, session_id, expected_id, expect_call
):
    """Exercise the three session-creation paths: create, default, reload."""
    stub = FakeWriteStub()
    kwargs = dict(write_mode=write_mode, api_version="2")
    if session_id is not None:
        kwargs["session_id"] = session_id
    sess = TableWriteSession(stub, table_id, **kwargs)
    assert sess.id == expected_id
    if expect_call == "create_session":
        assert sess.route_token == "rt-123"
    if expect_call is None:
        assert len(stub.calls) == 0
    elif expect_call == "get_session":
        assert ("get_session", table_id, "existing") in stub.calls


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_writer_write_and_close(table_id, arrow_batch):
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)
    assert writer.stream_id == "stream-0"
    assert writer.write_schema is not None
    writer.write_batch(arrow_batch)
    writer.close()
    call_names = [c[0] for c in stub.calls]
    assert "write_table" in call_names
    assert "close_stream" in call_names


def test_streaming_default_skips_close(table_id, arrow_batch):
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.STREAMING, api_version="2"
    )
    writer = sess.open_arrow_writer("default", compress_option=None)
    writer.write_batch(arrow_batch)
    writer.close()
    assert "close_stream" not in [c[0] for c in stub.calls]


def test_required_data_format_forwarded_to_request(table_id):
    """required_data_format reaches the wire as RequiredDataFormat."""
    captured = {}

    class _CapturingStub(FakeWriteStub):
        def create_table_write_session(self, table_id, request, write_mode):
            captured["request"] = request
            return super().create_table_write_session(table_id, request, write_mode)

    # Set -> serialized as RequiredDataFormat
    TableWriteSession(
        _CapturingStub(),
        table_id,
        write_mode=WriteMode.BATCH,
        required_data_format=DataFormat("Arrow", "V5"),
        api_version="2",
    )
    assert captured["request"].to_dict()["RequiredDataFormat"] == {
        "Type": "Arrow",
        "Version": "V5",
    }

    # None -> omitted (server applies its default)
    TableWriteSession(
        _CapturingStub(),
        table_id,
        write_mode=WriteMode.BATCH,
        api_version="2",
    )
    assert "RequiredDataFormat" not in captured["request"].to_dict()


def test_writer_writes_multi_block_table(table_id, arrow_batch):
    """A chunked pa.Table must flush all batches, not just the first chunk."""
    table = pa.Table.from_batches([arrow_batch, arrow_batch])
    assert len(table.to_batches()) == 2  # sanity: two real chunks

    # auto_upload_blobs=False: no blob upload, both chunks flushed as one batch.
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)
    writer.write_batch(table)
    writer.close()
    assert stub.write_record_counts == [4]
    assert not any(c[0] == "batch_blob" for c in stub.calls)

    # auto_upload_blobs=True: both chunks processed -> two batch blob uploads.
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )
    writer.write_batch(table)
    writer.close()
    assert stub.write_record_counts == [4]
    assert sum(1 for c in stub.calls if c[0] == "batch_blob") == 2


def test_writer_writes_record_batch(table_id, arrow_batch):
    """A single pa.RecordBatch input must flush with the correct row count."""
    # auto_upload_blobs=False: no blob upload.
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)
    writer.write_batch(arrow_batch)
    writer.close()
    assert stub.write_record_counts == [arrow_batch.num_rows]
    assert not any(c[0] == "batch_blob" for c in stub.calls)

    # auto_upload_blobs=True: blob upload triggered.
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )
    writer.write_batch(arrow_batch)
    writer.close()
    assert stub.write_record_counts == [arrow_batch.num_rows]
    assert any(c[0] == "batch_blob" for c in stub.calls)


def test_writer_writes_empty_table(table_id, arrow_schema):
    """An empty pa.Table must not raise and must not flush any rows."""
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)
    empty_cols = [pa.array([], type=f.type) for f in arrow_schema]
    writer.write_batch(pa.Table.from_arrays(empty_cols, schema=arrow_schema))
    writer.close()
    assert stub.write_record_counts == []


@pytest.mark.parametrize(
    "compress_option",
    [
        None,
        CompressOption(CompressOption.CompressAlgorithm.ODPS_ZLIB),
        CompressOption(CompressOption.CompressAlgorithm.ODPS_ZSTD),
    ],
    ids=["uncompressed", "zlib", "zstd"],
)
def test_raw_arrow_request_body_multi_block_round_trip(compress_option):
    """Multiple pre-serialized Arrow blocks survive a single IPC stream.

    Exercises :class:`RawArrowRequestBody` with several distinct
    ``RecordBatch`` blocks (the multi-arrow-block write scenario) under both
    the uncompressed fast path and the compressed re-serialization path.
    The compressed path previously built a malformed stream (empty schema
    message) and raised ``ArrowInvalid``; this test guards against that
    regression by round-tripping every block.
    """
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    batch0 = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], pa.int64()), pa.array(["a", "b"], pa.string())],
        schema=schema,
    )
    batch1 = pa.RecordBatch.from_arrays(
        [pa.array([3], pa.int64()), pa.array(["c"], pa.string())],
        schema=schema,
    )
    batch2 = pa.RecordBatch.from_arrays(
        [pa.array([4, 5, 6], pa.int64()), pa.array(["d", "e", "f"], pa.string())],
        schema=schema,
    )
    batch_bytes = [serialize_batch(b) for b in (batch0, batch1, batch2)]

    body = RawArrowRequestBody(
        schema, batch_bytes, compress_option=compress_option
    ).serialize()

    reader = pa.ipc.open_stream(pa.BufferReader(body))
    round_tripped = reader.read_all()
    # All three blocks (2 + 1 + 3 = 6 rows) survived intact and in order.
    assert round_tripped.num_rows == 6
    assert round_tripped.column(0).to_pylist() == [1, 2, 3, 4, 5, 6]
    assert round_tripped.column(1).to_pylist() == list("abcdef")


# ---------------------------------------------------------------------------
# Record writer
# ---------------------------------------------------------------------------


def test_record_writer_write(table_id, arrow_schema):
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )
    rw = writer.get_as_record_writer()
    rw.write([3, "c", b"zdata"])
    rw.close()


def test_record_writer_on_plain_writer(table_id, arrow_schema):
    """get_as_record_writer works on a plain TableArrowWriter (no auto_upload).

    Previously this raised StorageClientError.  Now it must succeed: the
    record writer wraps any TableArrowWriter.  When auto_upload_blobs is
    False, BLOB cells pass through unchanged — no batch_blob call is made.
    """
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    # Plain writer — auto_upload_blobs defaults to False
    writer = sess.open_arrow_writer("stream-0", compress_option=None)
    assert not isinstance(writer, TableArrowBlobUploadWriter)

    # This used to raise; now it must succeed.
    rw = writer.get_as_record_writer()
    assert isinstance(rw, AppendTableRecordWriter)

    # Writing a row with bytes in the BLOB column must NOT trigger a blob
    # upload — the bytes pass through inline (the caller is responsible
    # for placing reference bytes).
    rw.write([3, "c", b"raw-bytes"])
    rw.close()

    call_names = [c[0] for c in stub.calls]
    assert "batch_blob" not in call_names
    assert "write_table" in call_names


def test_record_writer_auto_upload_blobs_flag_controls_interception(table_id):
    """auto_upload_blobs=True triggers blob upload; False does not."""
    # With auto_upload_blobs=True
    stub_on = FakeWriteStub()
    sess_on = TableWriteSession(
        stub_on, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer_on = sess_on.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )
    rw_on = writer_on.get_as_record_writer()
    rw_on.write([1, "a", b"payload"])
    rw_on.close()
    assert "batch_blob" in [c[0] for c in stub_on.calls]

    # With auto_upload_blobs=False (plain writer)
    stub_off = FakeWriteStub()
    sess_off = TableWriteSession(
        stub_off, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer_off = sess_off.open_arrow_writer("stream-0", compress_option=None)
    rw_off = writer_off.get_as_record_writer()
    rw_off.write([1, "a", b"payload"])
    rw_off.close()
    assert "batch_blob" not in [c[0] for c in stub_off.calls]


def test_record_writer_type_detection(table_id):
    """Writer type is determined by presence of __operation system column."""
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )
    # No __operation column -> AppendTableRecordWriter
    writer._write_schema._system_columns = []
    assert isinstance(writer.get_as_record_writer(), AppendTableRecordWriter)

    # Inject __operation system column -> DeltaTableRecordWriter
    delta_schema = WriteSchema.from_dict(_make_delta_table_schema())
    writer._write_schema._system_columns = delta_schema.system_columns
    assert isinstance(writer.get_as_record_writer(), DeltaTableRecordWriter)


def test_delta_record_writer_write_and_delete_stamp_operation(table_id):
    """DeltaTableRecordWriter.write() stamps UPSERT, delete() stamps DELETE.

    Verifies per-record operation injection: both operations can be
    freely interleaved on a single writer instance.
    """
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)

    # Inject __operation system column to simulate a PK/delta table.
    delta_schema = WriteSchema.from_dict(_make_delta_table_schema())
    writer._write_schema._system_columns = delta_schema.system_columns

    rw = writer.get_as_record_writer()
    assert isinstance(rw, DeltaTableRecordWriter)

    # write() → UPSERT, delete() → DELETE, interleaved on one writer.
    # The schema is id, name, data (3 data cols) + __operation (system).
    rw.write([1, "alice", 100])
    rw.delete([1, None, None])

    # _flush_records converts buffered rows to an Arrow batch and hands
    # it to write_batch — inspect the cached batch before calling
    # flush() which sends it to the server.
    rw._flush_records()

    assert len(writer._cached_record_batches) >= 1
    batch = writer._cached_record_batches[0]
    op_idx = batch.schema.get_field_index("__operation")
    assert op_idx >= 0
    ops = batch.column(op_idx).to_pylist()
    assert ops == [ord("U"), ord("D")]


# ---------------------------------------------------------------------------
# Commit / abort
# ---------------------------------------------------------------------------


def test_commit(table_id):
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    sess.commit(["stream-0"], [0])
    assert any(c[0] == "commit" for c in stub.calls)


def test_auto_abort_on_close(table_id):
    """Uncommitted session auto-aborts on close()."""
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    sess.close()
    assert any(c[0] == "abort" for c in stub.calls)


# ---------------------------------------------------------------------------
# v3 gating
# ---------------------------------------------------------------------------


def test_v3_gating_min_uncommitted_staging_id(table_id):
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.STREAMING, api_version="2"
    )
    with pytest.raises(NotImplementedError):
        sess.get_min_uncommitted_staging_id()


# ---------------------------------------------------------------------------
# Per-row blob metadata
# ---------------------------------------------------------------------------


def _parse_blob_frame_headers(raw, include_data=False):
    """Parse all blob frame headers from a serialized batch body.

    Returns a list of dicts (one per frame): ``{ContentType, CustomFileName, ...}``.
    When *include_data* is True, returns a list of ``(header, data, footer)``
    tuples instead, so callers can verify actual blob bytes without
    re-implementing the frame-parsing loop.
    """
    buf = io.BytesIO(raw)
    frames = []
    while True:
        hlen_bytes = buf.read(8)
        if len(hlen_bytes) < 8:
            break
        hlen = struct.unpack("<q", hlen_bytes)[0]
        header = json.loads(buf.read(hlen).decode("utf-8"))
        # data
        dlen = struct.unpack("<q", buf.read(8))[0]
        data = buf.read(dlen)
        # footer
        flen = struct.unpack("<q", buf.read(8))[0]
        footer = buf.read(flen)
        if include_data:
            frames.append((header, data, footer))
        else:
            frames.append(header)
    return frames


class _StreamingCaptureStub(FakeWriteStub):
    """FakeWriteStub that captures both single-stream and batch blob bodies."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_stream_bodies = []
        self.captured_batch_bodies = []

    def table_write_blob(
        self, table_id, params, data, route_token, content_encoding=None, **kwargs
    ):
        if not isinstance(data, (bytes, bytearray)):
            body = b"".join(data)
        else:
            body = bytes(data)
        self.captured_stream_bodies.append(body)
        return _FakeResp(
            data=json.dumps(
                {"BlobReference": base64.b64encode(b"stream_ref").decode()}
            ).encode()
        )

    def table_batch_write_blob(self, table_id, params, data, route_token, **kwargs):
        if not isinstance(data, (bytes, bytearray)):
            body = b"".join(data)
        else:
            body = bytes(data)
        self.captured_batch_bodies.append(body)
        num_items = self._count_blob_frames(body)
        refs = [base64.b64encode(b"ref%d" % i).decode() for i in range(num_items)]
        return _FakeResp(data=json.dumps({"BlobReferences": refs}).encode())


def test_per_row_blob_metadata_via_callback(table_id):
    """blob_metadata_callback supplies per-row mime_type / custom_file_name.

    Each blob cell in a single write_batch should carry its own metadata
    from the callback, not the session-level default.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="3"
    )

    blob_schema = pa.schema([("id", pa.int64()), ("data", pa.binary())])
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], pa.int64()), pa.array([b"aaa", b"bbb"], pa.binary())],
        schema=blob_schema,
    )

    call_log = []

    def metadata_fn(row_index, column_name, blob_data):
        call_log.append((row_index, column_name, blob_data))
        return ("image/png" if row_index == 0 else "text/plain", f"f{row_index}.bin")

    writer = sess.open_arrow_writer(
        "stream-0",
        auto_upload_blobs=True,
        blob_metadata_callback=metadata_fn,
        compress_option=None,
    )
    writer.write_batch(batch)
    writer.close()

    # Callback was invoked once per blob cell
    assert len(call_log) == 2
    assert call_log[0] == (0, "data", b"aaa")
    assert call_log[1] == (1, "data", b"bbb")

    # Parse the serialized frames and verify per-row metadata
    assert len(stub.captured_batch_bodies) >= 1
    headers = _parse_blob_frame_headers(stub.captured_batch_bodies[0])
    assert len(headers) == 2
    assert headers[0].get("ContentType") == "image/png"
    assert headers[0].get("CustomFileName") == "f0.bin"
    assert headers[1].get("ContentType") == "text/plain"
    assert headers[1].get("CustomFileName") == "f1.bin"


def test_per_row_blob_metadata_fallback_to_default(table_id):
    """When blob_metadata_callback returns None, fall back to session-level defaults."""
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="3"
    )

    blob_schema = pa.schema([("id", pa.int64()), ("data", pa.binary())])
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1], pa.int64()), pa.array([b"xxx"], pa.binary())],
        schema=blob_schema,
    )

    def metadata_fn(row_index, column_name, blob_data):
        return None  # fall back to defaults

    writer = sess.open_arrow_writer(
        "stream-0",
        auto_upload_blobs=True,
        blob_mime_type="application/octet-stream",
        blob_custom_file_name="default.bin",
        blob_metadata_callback=metadata_fn,
        compress_option=None,
    )
    writer.write_batch(batch)
    writer.close()

    headers = _parse_blob_frame_headers(stub.captured_batch_bodies[0])
    assert len(headers) == 1
    assert headers[0].get("ContentType") == "application/octet-stream"
    assert headers[0].get("CustomFileName") == "default.bin"


def test_record_writer_blob_metadata_callback_receives_original_value(table_id):
    """Record API: callback receives the original file-like object, not bytes.

    The callback must be called exactly once per blob, with the value the
    user passed (bytes or file-like), before the record writer reads it to
    bytes for Arrow serialization.  The resolved metadata must appear in the
    serialized frame headers.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="3"
    )

    file_obj = io.BytesIO(b"file-data")
    call_log = []

    def metadata_fn(row_index, column_name, blob_data):
        call_log.append((row_index, column_name, blob_data))
        return "image/png", f"file_{row_index}.png"

    writer = sess.open_arrow_writer(
        "stream-0",
        auto_upload_blobs=True,
        blob_metadata_callback=metadata_fn,
        compress_option=None,
    )
    rw = writer.get_as_record_writer()
    # Row 0: file-like object, Row 1: raw bytes
    rw.write([1, "a", file_obj])
    rw.write([2, "b", b"raw-data"])
    rw.close()

    # Callback called exactly once per blob (2 blobs total)
    assert len(call_log) == 2

    # Row 0: callback received the original file-like object (not bytes)
    assert call_log[0][2] is file_obj
    # Row 1: callback received the original bytes
    assert call_log[1][2] == b"raw-data"

    # Verify metadata in serialized frames
    assert len(stub.captured_batch_bodies) >= 1
    headers = _parse_blob_frame_headers(stub.captured_batch_bodies[0])
    assert len(headers) == 2
    assert headers[0].get("ContentType") == "image/png"
    assert headers[0].get("CustomFileName") == "file_0.png"
    assert headers[1].get("ContentType") == "image/png"
    assert headers[1].get("CustomFileName") == "file_1.png"


def test_record_writer_blob_metadata_callback_called_once_per_blob(table_id):
    """Callback must not be re-invoked in _process_top_level_blob for record path."""
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="3"
    )

    call_count = [0]

    def metadata_fn(row_index, column_name, blob_data):
        call_count[0] += 1
        return "text/plain", f"doc_{row_index}.txt"

    writer = sess.open_arrow_writer(
        "stream-0",
        auto_upload_blobs=True,
        blob_metadata_callback=metadata_fn,
        compress_option=None,
    )
    rw = writer.get_as_record_writer()
    rw.write([1, "a", b"blob1"])
    rw.write([2, "b", b"blob2"])
    rw.write([3, "c", b"blob3"])
    rw.close()

    # Exactly 3 calls — one per blob, not re-invoked in _process_top_level_blob
    assert call_count[0] == 3


def test_batch_blob_writer_multi_block_data_integrity(table_id, arrow_schema):
    """Multi-block write: blob data from all blocks is uploaded intact.

    Verifies that the actual blob bytes (not just frame counts) survive the
    multi-chunk path — each chunk's blobs appear in the captured body with
    the correct data length and can be round-tripped.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )

    blob_a = b"\x00\x01\x02\x03"
    blob_b = b"\x10\x20\x30\x40"
    blob_c = b"\xaa\xbb\xcc\xdd"
    blob_d = b"\xee\xff\x00\x11"

    chunk0 = pa.RecordBatch.from_arrays(
        [
            pa.array([1, 2], pa.int64()),
            pa.array(["a", "b"], pa.string()),
            pa.array([blob_a, blob_b], pa.binary()),
        ],
        schema=arrow_schema,
    )
    chunk1 = pa.RecordBatch.from_arrays(
        [
            pa.array([3, 4], pa.int64()),
            pa.array(["c", "d"], pa.string()),
            pa.array([blob_c, blob_d], pa.binary()),
        ],
        schema=arrow_schema,
    )
    table = pa.Table.from_batches([chunk0, chunk1])
    assert len(table.to_batches()) == 2

    writer.write_batch(table)
    writer.close()

    # Two chunks -> two batch blob uploads.
    assert len(stub.captured_batch_bodies) == 2
    # All 4 rows in a single write_table call.
    assert stub.write_record_counts == [4]

    # Parse frame data (not just headers) and verify blob bytes.
    all_blobs = [
        data
        for body in stub.captured_batch_bodies
        for _header, data, _footer in _parse_blob_frame_headers(body, include_data=True)
    ]
    assert all_blobs == [blob_a, blob_b, blob_c, blob_d]


# ---------------------------------------------------------------------------
# Async flush (MR review round-3: future deque, restore order, auto-flush)
# ---------------------------------------------------------------------------


class _ControllableWriteStub(FakeWriteStub):
    """FakeWriteStub whose write_table can fail/succeed on demand.

    ``write_results`` is a list of outcomes consumed in call order:
    ``None`` = success, ``Exception`` = raise that exception.
    ``write_table_calls`` records the route_token and row_offset of each call.
    """

    def __init__(self, write_results=None, route_tokens=None):
        super().__init__()
        self._write_results = list(write_results) if write_results else []
        self.write_table_calls = []
        self._route_tokens = route_tokens or []

    def write_table(
        self,
        table_id,
        session_id,
        stream_id,
        stream_version,
        record_count,
        body,
        route_token,
        **kwargs
    ):
        self.write_table_calls.append(
            {
                "route_token": route_token,
                "row_offset": kwargs.get("row_offset"),
                "record_count": record_count,
            }
        )
        if self._write_results:
            outcome = self._write_results.pop(0)
        else:
            outcome = None
        if isinstance(outcome, Exception):
            raise outcome
        resp = _FakeResp(
            data=json.dumps({"StagingId": "stg-1"}).encode(),
            headers={
                "x-odps-request-id": "req-%d" % len(self.write_table_calls),
                "x-odps-max-storage-route-token": "rt-%d" % len(self.write_table_calls),
            },
        )
        return resp


def _make_writer(table_id, stub, **kwargs):
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    return sess.open_arrow_writer("stream-0", compress_option=None, **kwargs)


def test_flush_async_empty_returns_completed_future(table_id):
    """flush_async with nothing cached returns a completed Future."""
    stub = FakeWriteStub()
    writer = _make_writer(table_id, stub)
    fut = writer.flush_async()
    assert fut is not None
    assert fut.done()
    writer.close()


def test_async_failure_surfaces_on_close(table_id, arrow_batch):
    """A failed async flush must surface its exception from close/flush."""
    stub = _ControllableWriteStub(write_results=[RuntimeError("upload failed")])
    writer = _make_writer(table_id, stub, max_pending_buffers=2)
    writer.write_batch(arrow_batch)
    writer.flush_async()
    # Wait for the async flush to complete, then close must surface the error.
    with pytest.raises(RuntimeError, match="upload failed"):
        writer.close()


def test_multi_pending_future_failure_not_erased(table_id, arrow_batch):
    """Buffer 1 failure must not be erased by buffer 2 success (MR 236258561).

    With the old scalar _pending_future, buf2's success would clear the
    future and buf1's failure would be lost.  The deque ensures the
    failure surfaces (at the second flush_async, flush, or close) even
    though buf2 succeeded.
    """
    stub = _ControllableWriteStub(write_results=[RuntimeError("buf1 failed"), None])
    writer = _make_writer(table_id, stub, max_pending_buffers=2)
    writer.write_batch(arrow_batch)
    writer.flush_async()
    surfaced = None
    try:
        writer.write_batch(arrow_batch)
        writer.flush_async()
    except RuntimeError as e:
        surfaced = e
    if surfaced is None:
        try:
            writer.close()
        except RuntimeError as e:
            surfaced = e
    else:
        writer.close()
    assert surfaced is not None and "buf1 failed" in str(surfaced)


def test_restore_preserves_submission_order(table_id, arrow_batch):
    """Failed buffers must be restored after live cache, not before (MR 236258563).

    Both flushes fail; the restored cache must hold both batches in
    submission order so a retry sends them as one write_table with
    row_count == 4 (2+2).  A gate ensures both batches are submitted and
    queued before either upload runs, so both are in the deque when they
    fail.
    """
    gate = threading.Event()
    fail_count = [0]

    class _GatedStub(_ControllableWriteStub):
        def write_table(self, *a, **kw):
            gate.wait()
            fail_count[0] += 1
            raise RuntimeError("fail%d" % fail_count[0])

    stub = _GatedStub()
    writer = _make_writer(table_id, stub, max_pending_buffers=2)
    writer.write_batch(arrow_batch)
    writer.flush_async()
    writer.write_batch(arrow_batch)
    writer.flush_async()
    # Release both queued uploads; both fail and restore to the cache.
    gate.set()
    # Surface pending failures (may raise at write_batch/flush_async/flush).
    try:
        writer.flush()
    except RuntimeError:
        pass

    # The restored cache holds both batches in order; a sync flush sends
    # them as one write_table with row_count == 4 (2+2).
    class _OkStub(_ControllableWriteStub):
        pass

    writer._stub = _OkStub()
    writer._stub._write_results = [None]
    writer.flush()
    assert writer._stub.write_table_calls, "no retry write_table call"
    assert writer._stub.write_table_calls[-1]["record_count"] == 4
    writer.close()


def test_auto_flush_uses_flush_async_not_flush(table_id):
    """auto-flush at threshold must not block write_batch (MR 236258571)."""
    block = threading.Event()
    block.set()  # start unblocked

    class _BlockingStub(_ControllableWriteStub):
        def write_table(self, *a, **kw):
            block.wait()  # blocks until released
            return super().write_table(*a, **kw)

    stub = _BlockingStub()
    # buffer_size=1 so a single row triggers auto-flush.
    writer = _make_writer(
        table_id,
        stub,
        buffer_size=1,
        max_pending_buffers=2,
    )
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1], pa.int64())], schema=pa.schema([("id", pa.int64())])
    )
    block.clear()  # now block uploads
    # write_batch triggers auto-flush via flush_async; it must NOT block.
    done = []

    def _do_write():
        writer.write_batch(batch)
        done.append(True)

    t = threading.Thread(target=_do_write)
    t.start()
    t.join(timeout=2.0)
    assert done, "write_batch blocked on auto-flush — flush_async not used"
    block.set()  # release the upload
    writer.close()
    t.join(timeout=5.0)


@pytest.mark.parametrize(
    "payload_size",
    [300 * 1024, 500 * 1024],
    ids=["300k", "500k"],
)
def test_record_writer_file_like_streamed_not_materialized(table_id, payload_size):
    """File-like BLOB values are streamed in chunks, not read to bytes.

    Regression: previously ``_intercept_blob_leaf`` called ``value.read()``
    with no argument, fully materializing the file-like object into memory
    before upload.  Now the file-like object is queued and streamed directly
    by ``BlobWriteItem.write_frame_to`` in 256 KiB chunks.

    Proof: a ``BytesIO`` subclass tracks every ``read`` call.  If the
    file-like were materialized, ``read()`` (no-arg) would be called once
    in the record writer.  Instead, ``read(chunk_size)`` is called in the
    blob-writer loop — never ``read()`` with no args.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    blob_content = b"\xab" * payload_size  # > 256 KiB chunk boundary

    file_obj = TrackedStream(blob_content)

    writer = sess.open_arrow_writer(
        "stream-0",
        auto_upload_blobs=True,
        compress_option=None,
        auto_close_files=True,
    )
    rw = writer.get_as_record_writer()
    rw.write([1, "a", file_obj])
    rw.close()

    # The blob data must appear intact in the captured body.
    assert len(stub.captured_batch_bodies) >= 1
    frames = _parse_blob_frame_headers(stub.captured_batch_bodies[0], include_data=True)
    assert len(frames) == 1
    assert frames[0][1] == blob_content

    # Proof of streaming: read called with explicit chunk sizes (>0).
    assert file_obj.read_calls, "read() was never called — file not streamed"
    assert all(
        isinstance(s, int) and s > 0 for s in file_obj.read_calls
    ), "read() called with no size arg — file was materialized, not streamed"

    # File-like object closed after streaming.
    assert file_obj.closed


def test_record_writer_mixed_bytes_and_file_like_stream(table_id):
    """Mixed bytes and file-like values in one batch both upload intact.

    Ensures the per-column file queue stays in lock-step with the upload
    iteration order when bytes (queue placeholder None) and file-like
    objects are interleaved across rows.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )

    file_a = io.BytesIO(b"file-aaa")
    bytes_b = b"bytes-bbb"
    file_c = io.BytesIO(b"file-ccc")

    writer = sess.open_arrow_writer(
        "stream-0",
        auto_upload_blobs=True,
        compress_option=None,
        auto_close_files=True,
    )
    rw = writer.get_as_record_writer()
    rw.write([1, "a", file_a])
    rw.write([2, "b", bytes_b])
    rw.write([3, "c", file_c])
    rw.close()

    assert len(stub.captured_batch_bodies) >= 1
    frames = _parse_blob_frame_headers(stub.captured_batch_bodies[0], include_data=True)
    assert len(frames) == 3
    all_data = [f[1] for f in frames]
    assert all_data == [b"file-aaa", b"bytes-bbb", b"file-ccc"]

    # File-like objects must be closed after streaming.
    assert file_a.closed
    assert file_c.closed


def test_record_writer_str_ref_passes_through_without_upload(table_id):
    """str blob refs pass through unchanged — no upload, no placeholder.

    When ``auto_upload_blobs=True``, ``str`` values in BLOB columns are
    treated as already-uploaded references.  They must NOT trigger a blob
    upload, and they must appear in the write_table batch as-is (converted
    to bytes by Arrow).  Only file-like / bytes values are uploaded.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )

    writer = sess.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )
    rw = writer.get_as_record_writer()
    # Row 0: str ref (passes through), Row 1: raw bytes (uploaded)
    rw.write([1, "a", "existing_ref"])
    rw.write([2, "b", b"raw-data"])
    rw.close()

    # Only 1 blob uploaded (the raw bytes), not 2.
    # _StreamingCaptureStub captures bodies but does not record in calls.
    assert len(stub.captured_batch_bodies) == 1
    frames = _parse_blob_frame_headers(stub.captured_batch_bodies[0], include_data=True)
    assert len(frames) == 1
    assert frames[0][1] == b"raw-data"

    # write_table called once with both rows.
    assert "write_table" in [c[0] for c in stub.calls]
    assert stub.write_record_counts == [2]


def test_record_writer_auto_upload_blobs_false_uses_plain_writer(table_id):
    """auto_upload_blobs=False: no blob upload, refs pass through to write_table.

    The record writer wraps a plain TableArrowWriter.  BLOB cells pass
    through unchanged — no batch_blob call is made.  The caller is
    responsible for placing reference bytes.
    """
    stub = FakeWriteStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)
    assert not isinstance(writer, TableArrowBlobUploadWriter)

    rw = writer.get_as_record_writer()
    rw.write([1, "a", b"some-ref-bytes"])
    rw.close()

    call_names = [c[0] for c in stub.calls]
    assert "batch_blob" not in call_names


# ---------------------------------------------------------------------------
# Streaming test harness: verify chunked streaming across all blob paths
# ---------------------------------------------------------------------------


def test_write_blob_stream_streams_payload(table_id):
    """``write_blob_stream`` streams data via chunked transfer-encoding.

    Data is streamed to the server without being fully materialized in a
    single buffer.  Whether written in one call or multiple, the server
    receives the concatenated payload.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)

    # Single write, > 256 KiB
    payload = b"\xab" * (300 * 1024)
    sw = writer.write_blob_stream(column_name="data")
    sw.write(payload)
    resp = sw.finish()
    assert resp.blob_reference is not None
    assert len(stub.captured_stream_bodies) == 1
    assert stub.captured_stream_bodies[0] == payload

    # 5 × 100 KiB writes = 500 KiB
    chunk = b"\x00" * (100 * 1024)
    sw = writer.write_blob_stream(column_name="data")
    for _ in range(5):
        sw.write(chunk)
    resp = sw.finish()
    assert resp.blob_reference is not None
    assert len(stub.captured_stream_bodies) == 2
    assert stub.captured_stream_bodies[1] == chunk * 5


def _assert_blob_batch_streamed(writer, stub, items_data):
    """Shared body for ``write_blob_batch`` streaming checks."""
    tracked = []
    items = []
    for data in items_data:
        if isinstance(data, (bytes, bytearray)) and len(data) > 1000:
            # Large bytes → treat as file-like to prove streaming
            obj = TrackedStream(data)
            tracked.append(obj)
            items.append(writer.build_blob_write_item(obj, column_name="data"))
        else:
            # Small bytes → direct bytes path
            items.append(writer.build_blob_write_item(data, column_name="data"))
    resp = writer.write_blob_batch(items)

    assert len(resp.blob_references) == len(items)
    assert len(stub.captured_batch_bodies) == 1
    frames = _parse_blob_frame_headers(stub.captured_batch_bodies[0], include_data=True)
    assert len(frames) == len(items)

    # Verify data integrity and streaming proof for file-like items.
    for i, (data, obj) in enumerate(zip(items_data, tracked)):
        assert frames[i][1] == data
        assert obj.read_calls, "read() was never called — file not streamed"
        assert all(s > 0 for s in obj.read_calls)
        # write_blob_batch does not close file-like objects (caller's job).
        assert not obj.closed


def test_write_blob_batch_streams_items(table_id):
    """``write_blob_batch`` streams file-like items in chunks.

    ``BlobWriteItem.write_frame_to`` reads file-like data in 256 KiB
    chunks — never ``read()`` with no args.  Bytes items use the direct
    ``stream.write(self.data)`` path.  Both produce correct frames.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)

    # Two file-like items
    _assert_blob_batch_streamed(
        writer, stub, [b"\xaa" * (300 * 1024), b"\xbb" * (200 * 1024)]
    )

    # Mixed: bytes + file-like
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)
    _assert_blob_batch_streamed(
        writer, stub, [b"raw-bytes-payload", b"\xcc" * (300 * 1024)]
    )


def test_record_writer_non_seekable_stream(table_id):
    """Non-seekable file-like objects are streamed without seek().

    ``BlobWriteItem._get_data_size`` requires ``seek``/``tell`` to
    determine size.  A non-seekable stream without ``__len__`` would
    fail — but only if it reaches ``_get_data_size``.  This test
    documents that seekable streams are required for the batch path
    (the wire format needs a data-length prefix).
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )
    rw = writer.get_as_record_writer()

    class _NonSeekableStream:
        """File-like with read() but no seek()/tell()/__len__."""

        def __init__(self, data):
            self._data = data
            self._pos = 0

        def read(self, size=-1):
            if size is None or size < 0:
                chunk = self._data[self._pos :]
            else:
                chunk = self._data[self._pos : self._pos + size]
            self._pos += len(chunk)
            return chunk

        def close(self):
            pass

    payload = b"non-seekable-data"
    stream = _NonSeekableStream(payload)

    # Non-seekable streams cannot provide a data-length prefix, so the
    # batch upload path raises ValueError from _get_data_size.
    rw.write([1, "a", stream])
    with pytest.raises(ValueError, match="Cannot determine data size"):
        rw.close()

    # The single-stream path (write_blob_stream) does NOT need a size
    # prefix — it streams via BlobStreamWriter.write().  Non-seekable
    # streams work there.
    writer2 = sess.open_arrow_writer("stream-1", compress_option=None)
    sw = writer2.write_blob_stream(column_name="data")
    sw.write(payload)
    resp = sw.finish()
    assert resp.blob_reference is not None
    assert stub.captured_stream_bodies[-1] == payload


def test_record_writer_multiple_batches_stream_all_blobs(table_id):
    """Multiple flush cycles each stream their pending blobs independently.

    When ``row_count_per_batch`` is small, the record writer flushes
    multiple times.  Each flush batch-uploads its own pending blob items
    and clears the pending list.  No blob data leaks across batches.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )
    rw = writer.get_as_record_writer(row_count_per_batch=1)

    payloads = [b"batch0_data", b"batch1_data", b"batch2_data"]
    for i, p in enumerate(payloads):
        rw.write([i, chr(ord("a") + i), p])
    rw.close()

    # 3 flushes → 3 batch_blob uploads, each with 1 frame.
    assert len(stub.captured_batch_bodies) == 3
    for i, body in enumerate(stub.captured_batch_bodies):
        frames = _parse_blob_frame_headers(body, include_data=True)
        assert len(frames) == 1
        assert frames[0][1] == payloads[i]

    # All 3 rows flushed in a single write_table call.
    assert stub.write_record_counts == [3]


def test_record_writer_nested_array_blob_file_like_streamed(table_id):
    """ARRAY<BLOB> column with file-like elements: each streamed in chunks.

    Verifies that nested blob values in an ARRAY<BLOB> column are
    individually streamed via the record writer's batch upload path.
    """
    nested_schema = {
        "DataColumns": [
            {
                "columnType": {
                    "MemberName": "id",
                    "ColumnId": 0,
                    "Nullable": True,
                    "Type": 0,
                }
            },
            {
                "columnType": {
                    "Type": 17,  # ARRAY
                    "Nullable": True,
                    "SubTypes": [
                        {"Type": 22, "ColumnId": 2, "Nullable": True}
                    ],  # BLOB element
                }
            },
        ],
        "SystemColumns": [],
    }
    ws = WriteSchema.from_dict(nested_schema)
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="3"
    )
    writer = sess.open_arrow_writer(
        "stream-0",
        auto_upload_blobs=True,
        compress_option=None,
        auto_close_files=True,
    )
    # Patch the write schema on the writer since the fake stub returns a default.
    writer._write_schema = ws
    rw = writer.get_as_record_writer()

    payload_a = b"\xaa" * (300 * 1024)
    payload_b = b"\xbb" * (200 * 1024)
    file_a = TrackedStream(payload_a)
    file_b = TrackedStream(payload_b)

    rw.write([1, [file_a, file_b]])
    rw.close()

    assert len(stub.captured_batch_bodies) == 1
    frames = _parse_blob_frame_headers(stub.captured_batch_bodies[0], include_data=True)
    assert len(frames) == 2
    assert frames[0][1] == payload_a
    assert frames[1][1] == payload_b

    for f in (file_a, file_b):
        assert f.read_calls
        assert all(s > 0 for s in f.read_calls)
    assert file_a.closed
    assert file_b.closed


def test_delta_record_writer_file_like_streamed(table_id):
    """DeltaTableRecordWriter streams file-like blobs in UPSERT rows.

    The DeltaTableRecordWriter delegates to AppendTableRecordWriter.write
    after stamping __operation='U'.  File-like blob values must still
    be streamed, not materialized.
    """
    stub = _StreamingCaptureStub()
    # Inject __operation system column for delta detection.
    delta_schema = WriteSchema.from_dict(_make_delta_table_schema())

    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0",
        auto_upload_blobs=True,
        compress_option=None,
        auto_close_files=True,
    )
    writer._write_schema._system_columns = delta_schema.system_columns
    rw = writer.get_as_record_writer()
    assert isinstance(rw, DeltaTableRecordWriter)

    payload = b"\xdd" * (300 * 1024)
    file_obj = TrackedStream(payload)
    rw.write([1, "a", file_obj])  # UPSERT
    rw.close()

    assert len(stub.captured_batch_bodies) == 1
    frames = _parse_blob_frame_headers(stub.captured_batch_bodies[0], include_data=True)
    assert len(frames) == 1
    assert frames[0][1] == payload

    assert file_obj.read_calls
    assert all(s > 0 for s in file_obj.read_calls)
    assert file_obj.closed


# ---------------------------------------------------------------------------
# auto_close_files
# ---------------------------------------------------------------------------


def test_auto_close_files_closes_after_write_blob_batch(table_id):
    """``auto_close_files=True`` closes file-like items after ``write_blob_batch``."""
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", compress_option=None, auto_close_files=True
    )

    file_a = CloseTrackingStream(b"payload-a")
    file_b = CloseTrackingStream(b"payload-b")
    items = [
        writer.build_blob_write_item(file_a, column_name="data"),
        writer.build_blob_write_item(file_b, column_name="data"),
    ]
    resp = writer.write_blob_batch(items)

    assert len(resp.blob_references) == 2
    assert file_a.close_count == 1
    assert file_b.close_count == 1


def test_auto_close_files_default_keeps_files_open(table_id):
    """Default ``auto_close_files=False`` leaves file-like items open."""
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer("stream-0", compress_option=None)

    file_a = CloseTrackingStream(b"payload-a")
    items = [writer.build_blob_write_item(file_a, column_name="data")]
    writer.write_blob_batch(items)

    assert file_a.close_count == 0
    assert not file_a.closed


def test_auto_close_files_skips_bytes_and_closed_files(table_id):
    """``auto_close_files`` skips raw ``bytes`` and already-closed files."""
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", compress_option=None, auto_close_files=True
    )

    class _AlreadyClosedStream(io.BytesIO):
        """File-like that reports ``closed=True`` but still reads."""

        @property
        def closed(self):
            return True

        def close(self):
            # Should never be called because _close_file_items skips
            # objects whose ``closed`` attr is truthy.
            raise AssertionError("close() called on an already-closed file")

    closed_file = _AlreadyClosedStream(b"already-closed")

    items = [
        writer.build_blob_write_item(b"raw-bytes", column_name="data"),
        writer.build_blob_write_item(closed_file, column_name="data"),
    ]
    writer.write_blob_batch(items)


def test_auto_close_files_closes_after_auto_upload_write_batch(table_id):
    """``auto_close_files=True`` closes file-like BLOB cells after upload.

    Exercises the ``TableArrowBlobUploadWriter._batch_upload_blobs`` path
    used when ``auto_upload_blobs=True``.  File-like objects reach this
    path through the record writer, which wraps them in
    :class:`BlobWriteItem` before pyarrow serialization — so we drive it
    via the record writer and verify files are closed exactly once by
    ``_batch_upload_blobs`` (the record writer itself never closes).
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0",
        auto_upload_blobs=True,
        compress_option=None,
        auto_close_files=True,
    )
    rw = writer.get_as_record_writer()

    file_a = CloseTrackingStream(b"payload-a")
    file_b = CloseTrackingStream(b"payload-b")
    rw.write([1, "a", file_a])
    rw.write([2, "b", file_b])
    rw.close()

    # Each file closed exactly once by _batch_upload_blobs via the flag;
    # the record writer itself never closes file-like objects.
    assert file_a.close_count == 1
    assert file_b.close_count == 1


def test_auto_close_files_false_record_writer_keeps_files_open(table_id):
    """Default ``auto_close_files=False``: record writer leaves files open.

    The record writer no longer closes file-like objects on its own — only
    ``_batch_upload_blobs`` closes them when the flag is ``True``.  With
    the default (``False``), the caller owns the file lifecycle.
    """
    stub = _StreamingCaptureStub()
    sess = TableWriteSession(
        stub, table_id, write_mode=WriteMode.BATCH, api_version="2"
    )
    writer = sess.open_arrow_writer(
        "stream-0", auto_upload_blobs=True, compress_option=None
    )
    rw = writer.get_as_record_writer()

    file_a = CloseTrackingStream(b"payload-a")
    file_b = CloseTrackingStream(b"payload-b")
    rw.write([1, "a", file_a])
    rw.write([2, "b", file_b])
    rw.close()

    assert file_a.close_count == 0
    assert file_b.close_count == 0
    assert not file_a.closed
    assert not file_b.closed
