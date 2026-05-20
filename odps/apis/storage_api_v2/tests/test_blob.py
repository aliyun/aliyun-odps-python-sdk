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

import base64
import hashlib
import json
import struct
import time
import zlib
from io import BytesIO

import pytest

from ..client import (
    BlobDataIterator,
    BlobStreamReader,
    BlobWriteItem,
    ChecksumType,
    SessionStatus,
    SplitOptions,
)

try:
    import pyarrow as pa
except ImportError:
    pa = None
    pytestmark = pytest.mark.skip("Need pyarrow to run this test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_frame(raw, pos=0):
    """Parse one [HeaderLen][Header][DataLen][Data][FooterLen][Footer] frame.

    Returns (header_dict, data_bytes, footer_dict, new_pos).
    """
    header_len = struct.unpack("<q", raw[pos : pos + 8])[0]
    pos += 8
    header = json.loads(raw[pos : pos + header_len].decode("utf-8"))
    pos += header_len
    data_len = struct.unpack("<q", raw[pos : pos + 8])[0]
    pos += 8
    data = raw[pos : pos + data_len]
    pos += data_len
    footer_len = struct.unpack("<q", raw[pos : pos + 8])[0]
    pos += 8
    footer = json.loads(raw[pos : pos + footer_len].decode("utf-8"))
    pos += footer_len
    return header, data, footer, pos


def _build_clean_stream(items):
    """Build a clean (no CRC, no LZ4) protocol frame stream from item dicts.

    Each item: {"data": bytes, "mime_type": str|None,
                "partition_values": list|None, "column_index": int}
    """
    buf = BytesIO()
    for item in items:
        header = {"ColumnIndex": item.get("column_index", 0)}
        if item.get("partition_values"):
            header["PartitionValues"] = item["partition_values"]
        if item.get("mime_type") is not None:
            header["ContentType"] = item["mime_type"]
        footer = {"Checksum": {"Type": 0}}

        header_bytes = json.dumps(header).encode("utf-8")
        footer_bytes = json.dumps(footer).encode("utf-8")
        data = item["data"]

        buf.write(struct.pack("<q", len(header_bytes)))
        buf.write(header_bytes)
        buf.write(struct.pack("<q", len(data)))
        buf.write(data)
        buf.write(struct.pack("<q", len(footer_bytes)))
        buf.write(footer_bytes)
    buf.seek(0)
    return buf


def _make_iterator(stream):
    """Create a BlobDataIterator wired directly to a clean stream (bypassing CRC/LZ4)."""
    it = BlobDataIterator.__new__(BlobDataIterator)
    it._raw_stream = None
    it._current_stream = stream
    it._finished = False
    it._first = True
    it._framed = (
        BlobDataIterator._is_framed(stream.getvalue())
        if hasattr(stream, "getvalue")
        else True
    )
    return it


# ---------------------------------------------------------------------------
# Unit tests (no live cluster needed)
# ---------------------------------------------------------------------------


def test_blob_write_item():
    # basic serialization round-trip
    item = BlobWriteItem(
        data=b"hello world",
        column_index=2,
        partition_values=["pt=2024"],
        mime_type="text/plain",
    )
    header, data, footer, _ = _parse_frame(item.serialize())
    assert header["ColumnIndex"] == 2
    assert header["PartitionValues"] == ["pt=2024"]
    assert header["ContentType"] == "text/plain"
    assert data == b"hello world"
    assert footer["Checksum"]["Type"] == 0
    assert "Crc32" not in footer["Checksum"]
    assert "MD5" not in footer["Checksum"]

    # checksums: NONE, CRC32, MD5
    item_none = BlobWriteItem(data=b"check me", checksum_type=ChecksumType.NONE)
    _, _, footer_none, _ = _parse_frame(item_none.serialize())
    assert footer_none["Checksum"]["Type"] == 0
    assert "Crc32" not in footer_none["Checksum"]
    assert "MD5" not in footer_none["Checksum"]

    item_crc = BlobWriteItem(data=b"check me", checksum_type=ChecksumType.CRC32)
    _, _, footer_crc, _ = _parse_frame(item_crc.serialize())
    assert footer_crc["Checksum"]["Type"] == 1
    assert footer_crc["Checksum"]["Crc32"] == (zlib.crc32(b"check me") & 0xFFFFFFFF)

    item_md5 = BlobWriteItem(data=b"check me", checksum_type=ChecksumType.MD5)
    _, _, footer_md5, _ = _parse_frame(item_md5.serialize())
    assert footer_md5["Checksum"]["Type"] == 2
    assert footer_md5["Checksum"]["MD5"] == hashlib.md5(b"check me").hexdigest()

    # optional fields omitted when not set
    header_min = BlobWriteItem(data=b"x", column_index=0)._build_header()
    assert header_min["PartitionValues"] == []
    assert "DistributionKey" not in header_min
    assert "ContentType" not in header_min
    assert header_min["ColumnIndex"] == 0

    # distribution key present when set
    header_dk = BlobWriteItem(data=b"x", distribution_key="abc123")._build_header()
    assert header_dk["DistributionKey"] == "abc123"

    # multiple items via write_blobs
    items = [
        BlobWriteItem(data=b"first", column_index=0),
        BlobWriteItem(data=b"second", column_index=1, mime_type="image/png"),
    ]
    raw = BlobWriteItem.write_blobs(items)
    header1, data1, _, pos = _parse_frame(raw)
    assert data1 == b"first"
    assert header1["ColumnIndex"] == 0
    header2, data2, _, _ = _parse_frame(raw, pos)
    assert data2 == b"second"
    assert header2["ContentType"] == "image/png"

    # empty list
    assert BlobWriteItem.write_blobs([]) == b""


@pytest.mark.parametrize(
    "data_fn, checksum_type",
    [
        # bytes with all checksum types
        (lambda: b"check me", ChecksumType.NONE),
        (lambda: b"check me", ChecksumType.CRC32),
        (lambda: b"check me", ChecksumType.MD5),
        # BytesIO (seekable) with CRC32 and MD5
        (lambda: BytesIO(b"stream data"), ChecksumType.CRC32),
        (lambda: BytesIO(b"stream data"), ChecksumType.MD5),
    ],
)
def test_blob_write_item_write_frame_to(data_fn, checksum_type):
    """write_frame_to produces output compatible with serialize() for bytes,
    and valid frames with correct checksums for file-like data."""
    raw_data = b"check me" if isinstance(data_fn(), bytes) else b"stream data"
    data = data_fn()
    item = BlobWriteItem(data=data, checksum_type=checksum_type)

    buf = BytesIO()
    item.write_frame_to(buf)
    output = buf.getvalue()

    if isinstance(data, bytes):
        # For bytes data, write_frame_to must match serialize()
        assert output == item.serialize()

    # Parse the frame and verify structure
    header, parsed_data, footer, _ = _parse_frame(output)
    assert header["ColumnIndex"] == 0
    assert parsed_data == raw_data
    assert footer["Checksum"]["Type"] == checksum_type.value

    if checksum_type == ChecksumType.CRC32:
        assert footer["Checksum"]["Crc32"] == (zlib.crc32(raw_data) & 0xFFFFFFFF)
    elif checksum_type == ChecksumType.MD5:
        assert footer["Checksum"]["MD5"] == hashlib.md5(raw_data).hexdigest()


def test_blob_write_item_write_frame_to_non_seekable():
    """Non-seekable stream with explicit size works via write_frame_to."""
    raw_data = b"non-seekable stream data"

    class NonSeekableStream:
        def __init__(self, data):
            self._buf = BytesIO(data)

        def read(self, size=-1):
            return self._buf.read(size)

    stream = NonSeekableStream(raw_data)
    item = BlobWriteItem(
        data=stream, checksum_type=ChecksumType.CRC32, size=len(raw_data)
    )

    buf = BytesIO()
    item.write_frame_to(buf)
    output = buf.getvalue()

    header, parsed_data, footer, _ = _parse_frame(output)
    assert parsed_data == raw_data
    assert footer["Checksum"]["Crc32"] == (zlib.crc32(raw_data) & 0xFFFFFFFF)


def test_blob_write_item_write_frame_to_multiple():
    """Multiple items streamed via write_frame_to match write_blobs()."""
    items = [
        BlobWriteItem(data=b"first", column_index=0),
        BlobWriteItem(data=b"second", column_index=1, mime_type="image/png"),
    ]

    buf = BytesIO()
    for item in items:
        item.write_frame_to(buf)
    output = buf.getvalue()

    assert output == BlobWriteItem.write_blobs(items)


def test_blob_write_item_no_size_error():
    """Non-seekable stream without size raises ValueError."""

    class NonSeekableStream:
        def read(self, size=-1):
            return b""

    stream = NonSeekableStream()
    item = BlobWriteItem(data=stream, checksum_type=ChecksumType.NONE)

    with pytest.raises(ValueError, match="Cannot determine data size"):
        item._get_data_size()


def test_blob_data_iterator():
    # single blob
    stream = _build_clean_stream(
        [{"data": b"hello world", "mime_type": "text/plain", "column_index": 2}]
    )
    results = list(_make_iterator(stream))
    assert len(results) == 1
    assert results[0] == (b"hello world", "text/plain")

    # multiple blobs with and without mime type
    stream = _build_clean_stream(
        [
            {"data": b"blob1", "mime_type": "text/plain", "column_index": 0},
            {"data": b"blob2", "mime_type": "image/png", "column_index": 1},
            {"data": b"blob3", "column_index": 2},
        ]
    )
    results = list(_make_iterator(stream))
    assert results == [
        (b"blob1", "text/plain"),
        (b"blob2", "image/png"),
        (b"blob3", None),
    ]

    # empty stream
    assert list(_make_iterator(BytesIO(b""))) == []

    # round-trip: BlobWriteItem -> BlobDataIterator
    items = [
        BlobWriteItem(
            data=b"first blob",
            partition_values=["pt=2024"],
            column_index=1,
            mime_type="application/octet-stream",
        ),
        BlobWriteItem(data=b"second blob", column_index=0),
    ]
    raw = BlobWriteItem.write_blobs(items)
    results = list(_make_iterator(BytesIO(raw)))
    assert results == [
        (b"first blob", "application/octet-stream"),
        (b"second blob", None),
    ]


def test_blob_stream_reader_framed():
    """Test BlobStreamReader with multiple framed blobs."""
    stream = _build_clean_stream(
        [
            {"data": b"hello world", "mime_type": "text/plain", "column_index": 0},
            {"data": b"second blob", "mime_type": "image/png", "column_index": 1},
            {"data": b"third", "column_index": 2},
        ]
    )
    it = _make_iterator(stream)
    reader = BlobStreamReader(it)

    # First blob: mime_type, incremental read
    assert reader.mime_type == "text/plain"
    chunk1 = reader.read(5)
    assert chunk1 == b"hello"
    chunk2 = reader.read(6)
    assert chunk2 == b" world"
    # Blob exhausted, read returns empty
    assert reader.read() == b""

    # Cannot call next() before exhausting current blob is ok since it IS exhausted
    reader2 = reader.next()
    assert reader2 is not None

    # Second blob
    assert reader2.mime_type == "image/png"
    assert reader2.read() == b"second blob"

    # Advance to third blob
    reader3 = reader2.next()
    assert reader3 is not None
    assert reader3.mime_type is None
    assert reader3.read() == b"third"

    # No more blobs
    reader4 = reader3.next()
    assert reader4 is None


def test_blob_stream_reader_next_before_exhausted():
    """BlobStreamReader.next() raises IOError if current blob not fully read."""
    stream = _build_clean_stream(
        [
            {"data": b"hello world", "mime_type": "text/plain", "column_index": 0},
            {"data": b"second", "column_index": 1},
        ]
    )
    it = _make_iterator(stream)
    reader = BlobStreamReader(it)

    # Read only part of the first blob
    reader.read(5)

    # next() should raise because blob is not exhausted
    with pytest.raises(IOError, match="not been fully read"):
        reader.next()


def test_blob_stream_reader_single_raw():
    """Test BlobStreamReader with a single raw (unframed) blob."""
    stream = _build_clean_stream([{"data": b"raw blob data", "column_index": 0}])
    it = _make_iterator(stream)
    reader = BlobStreamReader(it)

    # Raw blobs have no mime_type
    assert reader.mime_type is None
    assert reader.read() == b"raw blob data"

    # No more blobs
    assert reader.next() is None


def test_blob_stream_reader_empty():
    """Test BlobStreamReader with empty stream."""
    it = _make_iterator(BytesIO(b""))
    reader = BlobStreamReader(it)

    # Empty stream: finished immediately
    assert reader.read() == b""
    assert reader.next() is None


# ---------------------------------------------------------------------------
# Integration tests (require live ODPS cluster)
# ---------------------------------------------------------------------------


def test_blob_write_and_read(storage_api_blob_client):
    """Write blobs via single-blob streaming upload, then read them back."""
    client = storage_api_blob_client

    # ---- Create write session ----
    write_resp = client.create_write_session(partial_partition_spec="pt=test_blob_v2")
    assert write_resp.session_id is not None
    session_id = write_resp.session_id

    # ---- Create write stream ----
    stream_resp = client.create_write_stream(
        session_id, stream_id="stream-1", stream_version=1
    )
    assert stream_resp.request_id != ""

    # ---- Upload blobs one-by-one via single-blob streaming ----
    blob_data_list = [b"hello world! this is a blob!", b"another blob content"]
    blob_refs = []
    for idx, blob_data in enumerate(blob_data_list):
        writer = client.write_blob_stream(
            session_id, stream_id="stream-1", stream_version=1, column_index=2
        )
        writer.write(blob_data)
        resp = writer.finish()
        assert resp is not None
        assert resp.blob_reference is not None
        blob_refs.append(resp.blob_reference)

    # ---- Write arrow row data with blob references ----
    bigint_list = list(range(len(blob_refs)))
    ref_bytes_list = [base64.b64decode(ref) for ref in blob_refs]
    record_batch = pa.RecordBatch.from_arrays(
        [pa.array(bigint_list), pa.array(ref_bytes_list)],
        names=["a", "b"],
    )

    writer = client.write_rows_arrow(
        session_id,
        stream_id="stream-1",
        stream_version=1,
        record_count=len(blob_refs),
    )
    assert writer.write(record_batch) is True
    _, suc = writer.finish()
    assert suc is True

    # ---- Close write stream and commit ----
    client.close_write_stream(session_id, stream_id="stream-1", stream_version=1)
    client.commit_write_session(session_id)

    # ---- Create read session with ROW_OFFSET split mode ----
    split_opts = SplitOptions(
        split_mode=SplitOptions.SplitMode.ROW_OFFSET,
        split_number=256 * 1024 * 1024,
    )
    read_resp = client.create_read_session(
        required_partitions=["pt=test_blob_v2"],
        split_options=split_opts,
    )
    assert read_resp.session_id is not None
    read_session_id = read_resp.session_id
    read_route_token = read_resp.route_token

    # ---- Poll read session until NORMAL ----
    for _ in range(60):
        read_resp = client.get_read_session(read_session_id)
        if read_resp.session_status != SessionStatus.INIT:
            break
        time.sleep(1)
    if read_resp.route_token:
        read_route_token = read_resp.route_token

    # ---- Read rows using Offset+Count (ROW_OFFSET mode) ----
    record_count = read_resp.record_count or len(blob_data_list)
    split_number = split_opts.split_number
    all_blob_refs = []
    for offset in range(0, record_count, split_number):
        count = min(split_number, record_count - offset)
        buf = b""
        reader = client.read_rows_stream(
            session_id=read_session_id,
            row_offset=offset,
            row_count=count,
            max_batch_rows=4096,
            route_token=read_route_token,
        )
        while True:
            data = reader.read(65536)
            if len(data) == 0:
                break
            buf += data
        reader.close()

        if buf:
            with pa.ipc.open_stream(buf) as arrow_reader:
                for batch in arrow_reader:
                    for ref in batch.column(1).to_pylist():
                        if ref is not None:
                            all_blob_refs.append(ref)

    assert len(all_blob_refs) >= 2

    # The arrow VarBinaryVector stores blob references as UTF-8 encoded
    # reference strings.  The BlobRead API expects these reference strings
    # directly (not base64-encoded).  The server transforms references between
    # write and storage, so we must use the references read from arrow data,
    # not the write-time references.
    read_blob_refs = [ref.decode("utf-8") for ref in all_blob_refs]
    downloaded = [data for data, _ in client.read_blobs(blob_references=read_blob_refs)]

    assert len(downloaded) >= 2
    for d in downloaded:
        assert d in blob_data_list


def test_blob_batch_write_and_read(storage_api_blob_client):
    """Write blobs via batch upload with MIME types, then read them back."""
    client = storage_api_blob_client

    # ---- Create write session ----
    write_resp = client.create_write_session(
        partial_partition_spec="pt=test_blob_batch"
    )
    assert write_resp.session_id is not None
    session_id = write_resp.session_id

    # ---- Create write stream ----
    stream_id = "stream-batch-1"
    stream_version = 1
    stream_resp = client.create_write_stream(
        session_id, stream_id=stream_id, stream_version=stream_version
    )
    assert stream_resp.request_id != ""

    # ---- Upload multiple blobs in a single batch with MIME types ----
    blob_data_list = [
        (b"batch blob 1 - text data", "text/plain"),
        (b"batch blob 2 - image data", "image/png"),
        (b"batch blob 3 - json data", "application/json"),
    ]
    items = [
        BlobWriteItem(
            data=data,
            column_index=2,
            mime_type=mime,
        )
        for data, mime in blob_data_list
    ]
    batch_resp = client.write_blob_batch(
        items=items,
        session_id=session_id,
        stream_id=stream_id,
        stream_version=stream_version,
    )
    assert batch_resp is not None
    assert batch_resp.blob_references is not None
    assert len(batch_resp.blob_references) == len(blob_data_list)
    blob_refs = batch_resp.blob_references

    # ---- Write arrow row data with blob references ----
    bigint_list = list(range(len(blob_refs)))
    ref_bytes_list = [base64.b64decode(ref) for ref in blob_refs]
    record_batch = pa.RecordBatch.from_arrays(
        [pa.array(bigint_list), pa.array(ref_bytes_list)],
        names=["a", "b"],
    )

    writer = client.write_rows_arrow(
        session_id,
        stream_id=stream_id,
        stream_version=stream_version,
        record_count=len(blob_refs),
    )
    assert writer.write(record_batch) is True
    _, suc = writer.finish()
    assert suc is True

    # ---- Close write stream and commit ----
    client.close_write_stream(
        session_id, stream_id=stream_id, stream_version=stream_version
    )
    client.commit_write_session(session_id)

    # ---- Create read session ----
    split_opts = SplitOptions(
        split_mode=SplitOptions.SplitMode.ROW_OFFSET,
        split_number=256 * 1024 * 1024,
    )
    read_resp = client.create_read_session(
        required_partitions=["pt=test_blob_batch"],
        split_options=split_opts,
    )
    assert read_resp.session_id is not None
    read_session_id = read_resp.session_id
    read_route_token = read_resp.route_token

    # ---- Poll read session until NORMAL ----
    if read_resp.session_status == SessionStatus.INIT:
        for _ in range(60):
            read_resp = client.get_read_session(read_session_id)
            if read_resp.session_status != SessionStatus.INIT:
                break
            time.sleep(1)
        if read_resp.route_token:
            read_route_token = read_resp.route_token

    # ---- Read rows ----
    record_count = read_resp.record_count or len(blob_data_list)
    split_number = split_opts.split_number
    all_blob_refs = []
    for offset in range(0, record_count, split_number):
        count = min(split_number, record_count - offset)
        buf = b""
        reader = client.read_rows_stream(
            session_id=read_session_id,
            row_offset=offset,
            row_count=count,
            max_batch_rows=4096,
            route_token=read_route_token,
        )
        while True:
            data = reader.read(65536)
            if len(data) == 0:
                break
            buf += data
        reader.close()

        if buf:
            with pa.ipc.open_stream(buf) as arrow_reader:
                for batch in arrow_reader:
                    for ref in batch.column(1).to_pylist():
                        if ref is not None:
                            all_blob_refs.append(ref)

    assert len(all_blob_refs) >= len(blob_data_list)

    # ---- Read blobs back and verify MIME types ----
    downloaded_with_mime = [
        (data, mime) for data, mime in client.read_blobs(blob_references=all_blob_refs)
    ]

    assert len(downloaded_with_mime) >= len(blob_data_list)
    downloaded_data = [data for data, _ in downloaded_with_mime]
    for data, _ in blob_data_list:
        assert data in downloaded_data

    # Verify MIME types are preserved on read
    downloaded_mime_types = [mime for _, mime in downloaded_with_mime]
    expected_mime_types = [mime for _, mime in blob_data_list]
    # The server may not always return MIME types, so check that
    # when they are present they match
    for expected, actual in zip(expected_mime_types, downloaded_mime_types):
        if actual is not None:
            assert expected == actual


def test_commit_write_session_with_streams(storage_api_blob_client):
    """Verify commit_write_session accepts stream_ids and stream_versions."""
    client = storage_api_blob_client

    # ---- Create write session ----
    write_resp = client.create_write_session(
        partial_partition_spec="pt=test_commit_streams"
    )
    assert write_resp.session_id is not None
    session_id = write_resp.session_id

    # ---- Create write stream ----
    stream_id = "stream-commit-1"
    stream_version = 1
    stream_resp = client.create_write_stream(
        session_id, stream_id=stream_id, stream_version=stream_version
    )
    assert stream_resp.request_id != ""

    # ---- Write a simple blob ----
    blob_data = b"commit stream test blob"
    writer = client.write_blob_stream(
        session_id, stream_id=stream_id, stream_version=stream_version, column_index=2
    )
    writer.write(blob_data)
    resp = writer.finish()
    assert resp is not None
    assert resp.blob_reference is not None
    blob_ref = resp.blob_reference

    # ---- Write arrow row data ----
    ref_bytes = base64.b64decode(blob_ref)
    record_batch = pa.RecordBatch.from_arrays(
        [pa.array([0]), pa.array([ref_bytes])],
        names=["a", "b"],
    )
    arrow_writer = client.write_rows_arrow(
        session_id,
        stream_id=stream_id,
        stream_version=stream_version,
        record_count=1,
    )
    assert arrow_writer.write(record_batch) is True
    _, suc = arrow_writer.finish()
    assert suc is True

    # ---- Close write stream ----
    client.close_write_stream(
        session_id, stream_id=stream_id, stream_version=stream_version
    )

    # ---- Commit with stream_ids and stream_versions ----
    client.commit_write_session(
        session_id,
        stream_ids=[stream_id],
        stream_versions=[stream_version],
    )

    # ---- Verify data is visible by reading ----
    split_opts = SplitOptions(
        split_mode=SplitOptions.SplitMode.ROW_OFFSET,
        split_number=256 * 1024 * 1024,
    )
    read_resp = client.create_read_session(
        required_partitions=["pt=test_commit_streams"],
        split_options=split_opts,
    )
    assert read_resp.session_id is not None
    read_session_id = read_resp.session_id
    read_route_token = read_resp.route_token

    for _ in range(60):
        read_resp = client.get_read_session(read_session_id)
        if read_resp.session_status != SessionStatus.INIT:
            break
        time.sleep(1)
    if read_resp.route_token:
        read_route_token = read_resp.route_token

    record_count = read_resp.record_count or 1
    buf = b""
    reader = client.read_rows_stream(
        session_id=read_session_id,
        row_offset=0,
        row_count=record_count,
        max_batch_rows=4096,
        route_token=read_route_token,
    )
    while True:
        data = reader.read(65536)
        if len(data) == 0:
            break
        buf += data
    reader.close()

    assert len(buf) > 0
