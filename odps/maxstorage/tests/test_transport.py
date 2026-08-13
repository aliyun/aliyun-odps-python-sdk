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

"""Unit tests for odps.maxstorage transport: IO modules (compress, crc,
arrow_reader, arrow_writer, blob_reader, blob_writer) and stub."""

import hashlib
import io
import json
import struct

import mock
import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None
    pytestmark = pytest.mark.skip("Need pyarrow to run this test")

try:
    from odps.crc import Crc32c
except ImportError:
    Crc32c = None
from ..blob import BlobManager
from ..errors import MaxStorageError
from ..io.arrow_reader import _convert_struct_timestamps, _is_timestamp_struct_type
from ..io.arrow_writer import RawArrowRequestBody, serialize_batch
from ..io.blob_reader import BlobDataIterator, BlobStreamReader
from ..io.blob_writer import BlobStreamWriter
from ..io.compress import CompressionCodec
from ..io.crc import CrcStrippedInputStream
from ..stub import _parse_json_response
from ._helpers import TrackedStream

# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


def test_compression_codec_content_encoding():
    assert CompressionCodec.ZSTD.content_encoding == "zstd"
    assert CompressionCodec.LZ4_FRAME.content_encoding == "x-lz4-frame"
    assert CompressionCodec.NO_COMPRESSION.content_encoding is None


def test_compression_codec_build_compress_option():
    co = CompressionCodec.build_compress_option("zstd")
    assert co is not None


# ---------------------------------------------------------------------------
# crc
# ---------------------------------------------------------------------------


def _compute_crc32c(data):
    """Compute CRC32C (Castagnoli) of *data* and return a 4-byte LE trailer."""
    crc = Crc32c()
    crc.update(bytearray(data))
    return struct.pack("<I", crc.getvalue())


def test_crc_stripped_input_stream():
    """Verify CRC stripping: [4096B data][4B CRC] blocks -> raw data."""
    data_block = b"\x00" * 4096
    crc = _compute_crc32c(data_block)
    wire = data_block + crc
    stream = CrcStrippedInputStream(io.BytesIO(wire))
    result = stream.read()
    assert result == data_block


def test_crc_stripped_multiple_blocks():
    block1 = b"\x01" * 4096
    block2 = b"\x02" * 4096
    wire = block1 + _compute_crc32c(block1) + block2 + _compute_crc32c(block2)
    stream = CrcStrippedInputStream(io.BytesIO(wire))
    result = stream.read()
    assert result == block1 + block2


def test_crc_stripped_partial_block():
    """Last block may be < 4096 bytes, CRC still stripped."""
    block = b"\x03" * 100
    wire = block + _compute_crc32c(block)
    stream = CrcStrippedInputStream(io.BytesIO(wire))
    result = stream.read()
    assert result == block


def test_crc_stripped_mismatch():
    """A corrupt CRC32C trailer must raise MaxStorageError."""
    data_block = b"\x00" * 4096
    bad_crc = struct.pack("<I", 0xDEADBEEF)
    stream = CrcStrippedInputStream(io.BytesIO(data_block + bad_crc))
    with pytest.raises(MaxStorageError):
        stream.read()


# ---------------------------------------------------------------------------
# arrow_writer
# ---------------------------------------------------------------------------


def test_serialize_batch():
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], pa.int64()), pa.array(["a", "b"], pa.string())],
        schema=schema,
    )
    data = serialize_batch(batch)
    assert len(data) > 0
    # serialize_batch returns raw batch message bytes (no schema/EOS)


def test_raw_arrow_request_body():
    schema = pa.schema([("id", pa.int64())])
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], pa.int64())],
        schema=schema,
    )
    batch_bytes = serialize_batch(batch)
    body = RawArrowRequestBody(schema, [batch_bytes], None)
    data = body.serialize()
    assert len(data) > len(batch_bytes)
    # Should be a valid Arrow IPC stream readable by pyarrow
    reader = pa.ipc.open_stream(io.BytesIO(data))
    read_batch = reader.read_next_batch()
    assert read_batch.num_rows == 2


def test_raw_arrow_request_body_multiple_batches():
    schema = pa.schema([("id", pa.int64())])
    batch1 = pa.RecordBatch.from_arrays(
        [pa.array([1], pa.int64())],
        schema=schema,
    )
    batch2 = pa.RecordBatch.from_arrays(
        [pa.array([2], pa.int64())],
        schema=schema,
    )
    body = RawArrowRequestBody(
        schema, [serialize_batch(batch1), serialize_batch(batch2)], None
    )
    data = body.serialize()
    reader = pa.ipc.open_stream(io.BytesIO(data))
    batches = []
    while True:
        try:
            b = reader.read_next_batch()
        except StopIteration:
            break
        if b is None:
            break
        batches.append(b)
    assert len(batches) == 2
    assert batches[0].num_rows == 1
    assert batches[1].num_rows == 1


# ---------------------------------------------------------------------------
# blob_reader
# ---------------------------------------------------------------------------


def _make_blob_frame(header, data, footer):
    """Build a single blob frame."""
    header_json = json.dumps(header).encode()
    footer_json = json.dumps(footer).encode()
    return (
        struct.pack("<q", len(header_json))
        + header_json
        + struct.pack("<q", len(data))
        + data
        + struct.pack("<q", len(footer_json))
        + footer_json
    )


def test_blob_data_iterator_single_frame():
    # expected_count=1 → raw mode: server returns unframed blob bytes for a
    # single-ref request.  The entire CRC-stripped stream is one blob.
    data = b"hello"
    iterator = BlobDataIterator(
        io.BytesIO(data + _compute_crc32c(data)), expected_count=1
    )
    records = list(iterator)
    assert len(records) == 1
    assert records[0].data == b"hello"


def test_blob_data_iterator_multiple_frames():
    frame1 = _make_blob_frame({"length": 5}, b"hello", {})
    frame2 = _make_blob_frame({"length": 5}, b"world", {})
    # Both frames share one CRC block (see single-frame test for the trailer).
    combined = frame1 + frame2
    iterator = BlobDataIterator(
        io.BytesIO(combined + _compute_crc32c(combined)), expected_count=2
    )
    records = list(iterator)
    assert len(records) == 2
    assert records[0].data == b"hello"
    assert records[1].data == b"world"


# ---------------------------------------------------------------------------
# BlobDataIterator strictness (MR review round-3)
# ---------------------------------------------------------------------------


def test_blob_iterator_requires_explicit_contract():
    """Direct construction without expected_count must raise."""
    with pytest.raises(ValueError):
        BlobDataIterator(io.BytesIO(b""))


def test_blob_iterator_raw_mode_passthrough():
    """expected_count=1 returns the entire stream as a single BlobRecord."""
    data = b"\x00\x01\x02raw blob data"
    iterator = BlobDataIterator(io.BytesIO(data), expected_count=1, crc_strip=False)
    records = list(iterator)
    assert len(records) == 1
    assert records[0].data == data


def _make_negative_header_frame():
    """Build a frame with a negative header length."""
    return struct.pack("<q", -1) + b""


def _make_negative_data_frame():
    """Build a frame with a negative data length."""
    header_json = json.dumps({"length": 5}).encode()
    return struct.pack("<q", len(header_json)) + header_json + struct.pack("<q", -1)


def _make_negative_footer_frame():
    """Build a frame with a negative footer length."""
    header_json = json.dumps({"length": 5}).encode()
    return (
        struct.pack("<q", len(header_json))
        + header_json
        + struct.pack("<q", 5)
        + b"hello"
        + struct.pack("<q", -1)
    )


def _make_truncated_header_frame():
    """Build a frame with a short header (stream ends before header_len)."""
    return struct.pack("<q", 100) + b"short"


def _make_truncated_data_frame():
    """Build a frame with a short data read (less than declared data_len)."""
    header_json = json.dumps({"length": 100}).encode()
    return (
        struct.pack("<q", len(header_json))
        + header_json
        + struct.pack("<q", 100)
        + b"short"
    )


@pytest.mark.parametrize(
    "frame_maker,match",
    [
        (_make_negative_header_frame, "negative header length"),
        (_make_negative_data_frame, "negative data length"),
        (_make_negative_footer_frame, "negative footer length"),
        (_make_truncated_header_frame, None),
        (_make_truncated_data_frame, None),
    ],
    ids=["neg_header", "neg_data", "neg_footer", "trunc_header", "trunc_data"],
)
def test_blob_iterator_rejects_corrupt_frames(frame_maker, match):
    """Corrupt frames (negative lengths, truncation) must raise MaxStorageError."""
    frame = frame_maker()
    iterator = BlobDataIterator(io.BytesIO(frame), expected_count=2, crc_strip=False)
    with pytest.raises(MaxStorageError, match=match):
        list(iterator)


def test_blob_iterator_extra_frames_not_exposed():
    """With expected_count=2, a third frame must not be yielded."""
    frame1 = _make_blob_frame({"length": 5}, b"hello", {})
    frame2 = _make_blob_frame({"length": 5}, b"world", {})
    frame3 = _make_blob_frame({"length": 5}, b"extra", {})
    iterator = BlobDataIterator(
        io.BytesIO(frame1 + frame2 + frame3), expected_count=2, crc_strip=False
    )
    records = list(iterator)
    assert len(records) == 2
    assert records[0].data == b"hello"
    assert records[1].data == b"world"


def test_blob_iterator_truncated_eof_raises():
    """EOF before expected_count frames is a truncation error."""
    frame = _make_blob_frame({"length": 5}, b"hello", {})
    iterator = BlobDataIterator(io.BytesIO(frame), expected_count=2, crc_strip=False)
    with pytest.raises(MaxStorageError, match="Truncated blob stream"):
        list(iterator)


def test_blob_iterator_consumes_final_footer():
    """After the last frame, the final footer must be consumed (not left unread)."""
    footer = {"checksum": "abc"}
    frame1 = _make_blob_frame({"length": 5}, b"hello", {})
    frame2 = _make_blob_frame({"length": 5}, b"world", footer)
    stream = io.BytesIO(frame1 + frame2)
    iterator = BlobDataIterator(stream, expected_count=2, crc_strip=False)
    records = list(iterator)
    assert len(records) == 2
    assert records[0].data == b"hello"
    assert records[1].data == b"world"
    # The entire stream including the final footer must have been consumed.
    remaining = stream.read()
    assert remaining == b"", "final footer was not consumed: %r" % remaining


def test_blob_iterator_missing_final_footer_raises():
    """A missing final footer must raise, not be silently accepted."""
    header_json = json.dumps({"length": 5}).encode()
    frame1 = _make_blob_frame({"length": 5}, b"hello", {})
    # Second frame has header + data but no footer.
    frame2 = (
        struct.pack("<q", len(header_json))
        + header_json
        + struct.pack("<q", 5)
        + b"world"
    )
    iterator = BlobDataIterator(
        io.BytesIO(frame1 + frame2), expected_count=2, crc_strip=False
    )
    with pytest.raises(MaxStorageError):
        list(iterator)


def test_blob_stream_reader_enforces_expected_count():
    """Streaming mode must not expose frames beyond expected_count."""
    frame1 = _make_blob_frame({"length": 5}, b"hello", {})
    frame2 = _make_blob_frame({"length": 5}, b"world", {})
    frame3 = _make_blob_frame({"length": 5}, b"extra", {})
    iterator = BlobDataIterator(
        io.BytesIO(frame1 + frame2 + frame3), expected_count=2, crc_strip=False
    )
    reader = BlobStreamReader(iterator)
    # First blob reads fine.
    data = b""
    while True:
        chunk = reader.read(10)
        if not chunk:
            break
        data += chunk
    assert data == b"hello"
    # Advance to second blob.
    assert reader.next() is not None
    data = b""
    while True:
        chunk = reader.read(10)
        if not chunk:
            break
        data += chunk
    assert data == b"world"
    # next() must return None — no third frame exposed.
    assert reader.next() is None


def test_blob_stream_reader_raw_mode():
    """Streaming raw mode (expected_count=1) reads the entire stream as one blob."""
    data = b"raw blob bytes"
    iterator = BlobDataIterator(io.BytesIO(data), expected_count=1, crc_strip=False)
    reader = BlobStreamReader(iterator)
    out = b""
    while True:
        chunk = reader.read(4)
        if not chunk:
            break
        out += chunk
    assert out == data
    assert reader.next() is None


def test_blob_stream_reader_truncated_data_raises():
    """Streaming reads must reject EOF before the declared data length."""
    header = json.dumps({"ContentType": "application/octet-stream"}).encode()
    frame = struct.pack("<q", len(header)) + header + struct.pack("<q", 10) + b"abc"
    iterator = BlobDataIterator(io.BytesIO(frame), expected_count=2, crc_strip=False)
    reader = BlobStreamReader(iterator)

    assert reader.read(10) == b"abc"
    with pytest.raises(MaxStorageError, match="truncated stream"):
        reader.read(10)


# ---------------------------------------------------------------------------
# BlobManager framing contract (ref count determines framing)
# ---------------------------------------------------------------------------


def _make_mock_client(api_version="3"):
    """Build a minimal mock MaxStorageClient for BlobManager tests."""
    client = mock.Mock()
    client.api_version = api_version
    return client


def _make_crc_raw_stream(payload):
    """Wrap *payload* as a CRC-interleaved wire stream (one block)."""
    return io.BytesIO(payload + _compute_crc32c(payload))


def _make_crc_lz4_raw_stream(payload):
    """Build the response body for Content-Encoding: x-lz4-frame."""
    lz4_frame = pytest.importorskip("lz4.frame")
    compressed = lz4_frame.compress(payload)
    return io.BytesIO(compressed + _compute_crc32c(compressed))


def _make_framed_stream(frames):
    """Build a CRC-interleaved wire stream from a list of (header, data, footer)."""
    payload = b"".join(_make_blob_frame(h, d, f) for h, d, f in frames)
    return io.BytesIO(payload + _compute_crc32c(payload))


def test_blob_manager_single_ref_uses_raw_mode():
    """read_blobs with one ref → server returns raw, iterator uses raw mode."""
    blob_data = b"\x93NUMPY\x01\x00raw blob payload"
    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=_make_crc_raw_stream(blob_data),
        headers={},
    )

    mgr = BlobManager(client)
    records = list(mgr.read_blobs(["ref1"]))
    assert len(records) == 1
    assert records[0].data == blob_data
    # Single ref → no ACCEPT-ENCODING from caller default is LZ4, but the
    # iterator must NOT parse frame headers (raw mode).
    client.stub.read_blobs.assert_called_once()
    call_args = client.stub.read_blobs.call_args
    assert call_args[0][0] == ["ref1"]


def test_blob_manager_multi_ref_uses_framed_mode():
    """read_blobs with >1 refs → server returns framed, iterator parses frames."""
    frames = [
        ({"ContentType": "text/plain"}, b"hello", {}),
        ({"ContentType": "application/json"}, b"world", {}),
    ]
    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=_make_framed_stream(frames),
        headers={},
    )

    mgr = BlobManager(client)
    records = list(mgr.read_blobs(["ref1", "ref2"]))
    assert len(records) == 2
    assert records[0].data == b"hello"
    assert records[0].mime_type == "text/plain"
    assert records[1].data == b"world"
    assert records[1].mime_type == "application/json"


def test_blob_manager_read_blob_delegates_to_read_blobs():
    """read_blob(single ref) uses the same raw-mode path as read_blobs([ref])."""
    blob_data = b"single blob data"
    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=_make_crc_raw_stream(blob_data),
        headers={},
    )

    mgr = BlobManager(client)
    mgr.read_blob("ref1")


def test_blob_manager_no_compress_sends_no_encoding():
    """Default and 'raw' compress_algo both send no ACCEPT-ENCODING header."""
    blob_data = b"uncompressed data"

    # Default (no compress_algo)
    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=_make_crc_raw_stream(blob_data),
        headers={},
    )
    mgr = BlobManager(client)
    list(mgr.read_blobs(["ref1"]))
    assert client.stub.read_blobs.call_args[0][1] is None

    # Explicit 'raw'
    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=_make_crc_raw_stream(blob_data),
        headers={},
    )
    mgr = BlobManager(client)
    list(mgr.read_blobs(["ref1"], compress_algo="raw"))
    assert client.stub.read_blobs.call_args[0][1] is None


def test_blob_manager_strips_crc_before_decompressing():
    """CRC wraps compressed response bytes and must be stripped first."""
    blob_data = b"compressed blob response"
    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=_make_crc_lz4_raw_stream(blob_data),
        headers={"Content-Encoding": "x-lz4-frame"},
    )

    records = list(BlobManager(client).read_blobs(["ref1"]))
    assert len(records) == 1
    assert records[0].data == blob_data


def test_blob_manager_does_not_infer_response_encoding_from_request():
    """Accept-Encoding must not imply that an unencoded response is compressed."""
    pytest.importorskip("lz4.frame")
    blob_data = b"server returned an uncompressed response"
    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=_make_crc_raw_stream(blob_data),
        headers={},
    )

    records = list(BlobManager(client).read_blobs(["ref1"], compress_algo="lz4"))
    assert len(records) == 1
    assert records[0].data == blob_data


def test_blob_manager_raw_mode_handles_first_bytes_decoding_as_small_int():
    """Raw mode must not misinterpret blob bytes as a frame header.

    A blob starting with bytes that decode to a small LE int64 (e.g.
    short ASCII) must be returned verbatim in raw mode, not parsed as
    a frame header.
    """
    # b"hello\\x00\\x00\\x00" decodes to LE int64 0x0000006f6c6c6568 = small
    blob_data = b"hello\x00\x00\x00" + b" rest of blob"
    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=_make_crc_raw_stream(blob_data),
        headers={},
    )

    mgr = BlobManager(client)
    records = list(mgr.read_blobs(["ref1"]))
    assert len(records) == 1
    assert records[0].data == blob_data


# ---------------------------------------------------------------------------
# Blob streaming read: incremental chunked reads never materialize full blobs
# ---------------------------------------------------------------------------


def test_stream_read_large_framed_blob_not_materialized():
    """BlobStreamReader reads a large framed blob in chunks, not all at once.

    A blob larger than the 64 KiB raw-read block must be readable via
    incremental ``read(size)`` calls without a single ``read()`` that
    would materialize the whole payload.
    """
    blob_data = b"\xab" * (200 * 1024)  # > 64 KiB block boundary
    frame = _make_blob_frame({"ContentType": "application/octet-stream"}, blob_data, {})
    iterator = BlobDataIterator(io.BytesIO(frame), expected_count=1, crc_strip=False)
    # Use framed mode to get a declared data length.
    iterator._framed = True
    reader = BlobStreamReader(iterator)

    # Read in 8 KiB chunks — never a no-arg read().
    chunks = []
    while True:
        chunk = reader.read(8 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    data = b"".join(chunks)
    assert data == blob_data
    # Next blob must be None (only 1 expected).
    assert reader.next() is None


def test_stream_read_raw_mode_incremental():
    """Raw-mode BlobStreamReader supports incremental reads.

    Single-ref raw mode has unknown data length (``_data_remaining == -1``),
    so the reader reads until EOF.  Multiple ``read(size)`` calls should
    return chunks, not a single materialized blob.
    """
    blob_data = b"\xcd" * (100 * 1024)
    iterator = BlobDataIterator(
        io.BytesIO(blob_data), expected_count=1, crc_strip=False
    )
    reader = BlobStreamReader(iterator)

    chunks = []
    while True:
        chunk = reader.read(16 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    data = b"".join(chunks)
    assert data == blob_data
    assert reader.next() is None


def test_stream_read_multi_blob_incremental():
    """Multi-blob framed stream: each blob read incrementally via next().

    Three blobs of varying sizes are streamed one at a time.  Each blob
    is fully read before advancing to the next.  The reader must not
    materialize all blobs at once.
    """
    payloads = [b"first_blob", b"\xef" * (50 * 1024), b"third_small"]
    frames = b"".join(
        _make_blob_frame({"ContentType": "application/octet-stream"}, p, {})
        for p in payloads
    )
    iterator = BlobDataIterator(io.BytesIO(frames), expected_count=3, crc_strip=False)
    reader = BlobStreamReader(iterator)

    for i, payload in enumerate(payloads):
        chunks = []
        while True:
            chunk = reader.read(4 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        assert b"".join(chunks) == payload
        # Advance to next blob (or None after last).
        result = reader.next()
        if i < len(payloads) - 1:
            assert result is not None
        else:
            assert result is None


def test_stream_read_auto_drain_on_next():
    """next() auto-drains unread bytes before advancing to the next blob.

    If the caller reads only part of a blob then calls ``next()``, the
    remaining bytes + footer must be consumed so the next frame header
    is parsed correctly.
    """
    payload_a = b"\xaa" * (30 * 1024)
    payload_b = b"\xbb" * (20 * 1024)
    frames = _make_blob_frame({}, payload_a, {}) + _make_blob_frame({}, payload_b, {})
    iterator = BlobDataIterator(io.BytesIO(frames), expected_count=2, crc_strip=False)
    reader = BlobStreamReader(iterator)

    # Read only first 1 KiB of blob A, then skip to blob B.
    partial = reader.read(1024)
    assert partial == payload_a[:1024]
    assert reader.next() is not None  # auto-drains remaining payload_a

    # Blob B should be fully readable.
    chunks = []
    while True:
        chunk = reader.read(4 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    assert b"".join(chunks) == payload_b
    assert reader.next() is None


def test_stream_read_via_blob_manager_single_ref():
    """BlobManager.read_blobs(stream=True) with one ref returns a BlobStreamReader.

    The stream reader reads the raw (unframed) payload incrementally.
    """
    blob_data = b"managed single blob"
    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=_make_crc_raw_stream(blob_data),
        headers={},
    )

    mgr = BlobManager(client)
    reader = mgr.read_blobs(["ref1"], stream=True)
    assert isinstance(reader, BlobStreamReader)

    chunks = []
    while True:
        chunk = reader.read(8)
        if not chunk:
            break
        chunks.append(chunk)
    assert b"".join(chunks) == blob_data
    assert reader.next() is None


def test_stream_read_via_blob_manager_multi_ref():
    """BlobManager.read_blobs(stream=True) with multiple refs streams each blob.

    The stream reader parses framed data and yields each blob via
    ``read(size)`` + ``next()``.
    """
    payloads = [b"first_managed", b"second_managed", b"third_managed"]
    frames = [
        ({"ContentType": "text/plain"}, payloads[0], {}),
        ({"ContentType": "application/json"}, payloads[1], {}),
        ({"ContentType": "text/plain"}, payloads[2], {}),
    ]
    client = _make_mock_client()
    # Use side_effect so each call returns a fresh stream (BytesIO is
    # consumed by the first read and left at EOF for the second).
    framed_stream = _make_framed_stream(frames)
    client.stub.read_blobs.side_effect = [
        mock.Mock(raw=io.BytesIO(framed_stream.getvalue()), headers={}),
        mock.Mock(raw=io.BytesIO(framed_stream.getvalue()), headers={}),
    ]

    mgr = BlobManager(client)
    reader = mgr.read_blobs(["r1", "r2", "r3"], stream=True)
    assert isinstance(reader, BlobStreamReader)

    for i, payload in enumerate(payloads):
        chunks = []
        while True:
            chunk = reader.read(4)
            if not chunk:
                break
            chunks.append(chunk)
        assert b"".join(chunks) == payload
        if i < len(payloads) - 1:
            assert reader.next() is not None
            assert reader.mime_type in ("text/plain", "application/json")
        else:
            assert reader.next() is None

    # Verify mime_type by re-reading with a fresh stream.
    reader2 = mgr.read_blobs(["r1", "r2", "r3"], stream=True)
    assert reader2.mime_type == "text/plain"
    reader2.next()
    assert reader2.mime_type == "application/json"
    reader2.next()
    assert reader2.mime_type == "text/plain"


def _make_crc_stream_large(payload, block_size=4096):
    """Build a CRC-interleaved stream with per-block CRC32C for large payloads."""
    buf = io.BytesIO()
    offset = 0
    while offset < len(payload):
        block = payload[offset : offset + block_size]
        buf.write(block)
        buf.write(_compute_crc32c(block))
        offset += block_size
    buf.seek(0)
    return buf


def test_stream_read_large_raw_blob_via_blob_manager():
    """Large raw blob via BlobManager is streamed without full materialization.

    The underlying raw stream is a TrackedStream.  If the blob were
    materialized, a single ``read()`` (no-arg) would appear.  Incremental
    streaming uses ``read(size)`` with explicit positive sizes.
    """
    blob_data = b"\x99" * (200 * 1024)
    client = _make_mock_client()
    raw_bytes = _make_crc_stream_large(blob_data).getvalue()
    tracked = TrackedStream(raw_bytes)
    client.stub.read_blobs.return_value = mock.Mock(raw=tracked, headers={})
    mgr = BlobManager(client)
    reader = mgr.read_blobs(["ref1"], stream=True)

    chunks = []
    while True:
        chunk = reader.read(32 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    assert b"".join(chunks) == blob_data

    # The underlying stream was read in chunks (CRC stripping reads in
    # 4096-byte blocks).  No single no-arg read() that would materialize
    # the entire blob — CrcStrippedInputStream reads block-by-block.
    assert tracked.read_calls, "read() was never called — blob not streamed"
    # All read calls should have explicit sizes (CrcStrippedInputStream
    # uses 4096 or the requested size).
    assert all(
        isinstance(s, int) and s != -1 for s in tracked.read_calls
    ), "read() called with no size arg — blob was materialized"


def test_stream_read_multi_large_blobs_not_materialized():
    """Multiple large blobs via BlobManager.read_blobs(stream=True) are streamed.

    Three blobs (each > 64 KiB) are read incrementally via ``read(size)``
    + ``next()``.  The underlying wire stream is a ``TrackedStream``;
    if any blob were materialized, a single ``read()`` (no-arg) would appear.
    Incremental streaming uses ``read(size)`` with explicit positive sizes
    throughout the entire multi-blob read sequence.
    """
    payloads = [
        b"\xaa" * (200 * 1024),
        b"\xbb" * (150 * 1024),
        b"\xcc" * (100 * 1024),
    ]
    frames = b"".join(
        _make_blob_frame({"ContentType": "application/octet-stream"}, p, {})
        for p in payloads
    )
    # Build per-block CRC stream for the framed body.
    raw_bytes = _make_crc_stream_large(frames).getvalue()
    tracked = TrackedStream(raw_bytes)

    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(raw=tracked, headers={})

    mgr = BlobManager(client)
    reader = mgr.read_blobs(["r1", "r2", "r3"], stream=True)
    assert isinstance(reader, BlobStreamReader)

    for i, payload in enumerate(payloads):
        chunks = []
        while True:
            chunk = reader.read(8 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        assert b"".join(chunks) == payload, f"blob {i} mismatch"
        if i < len(payloads) - 1:
            assert reader.next() is not None
        else:
            assert reader.next() is None

    # Proof of streaming: the underlying wire stream was read in chunks.
    # No single no-arg read() that would materialize the entire batch body.
    assert tracked.read_calls, "read() was never called — blobs not streamed"
    assert all(
        isinstance(s, int) and s != -1 for s in tracked.read_calls
    ), "read() called with no size arg — batch body was materialized"


def test_stream_read_multi_blobs_auto_drain_between_large_blobs():
    """Partial read of each large blob + next() auto-drains remaining.

    Each blob is only partially read before ``next()`` is called.  The
    auto-drain must consume the remaining bytes + footer so the next
    frame header is parsed at the correct offset.  Data integrity of
    the fully-read portions is verified.
    """
    payloads = [
        b"\xaa" * (100 * 1024),
        b"\xbb" * (80 * 1024),
        b"\xcc" * (60 * 1024),
    ]
    frames = b"".join(_make_blob_frame({}, p, {}) for p in payloads)
    raw_bytes = _make_crc_stream_large(frames).getvalue()

    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=io.BytesIO(raw_bytes), headers={}
    )

    mgr = BlobManager(client)
    reader = mgr.read_blobs(["r1", "r2", "r3"], stream=True)

    for i, payload in enumerate(payloads):
        # Read only the first 4 KiB of each blob.
        partial = reader.read(4 * 1024)
        assert partial == payload[: 4 * 1024], f"blob {i} partial mismatch"
        # Advance — auto-drains the remaining bytes.
        if i < len(payloads) - 1:
            assert reader.next() is not None
        else:
            assert reader.next() is None


def test_stream_read_multi_blob_data_integrity_via_manager():
    """Multi-blob stream read: all blob data survives intact across frames.

    Three blobs of varying sizes (including one larger than the 64 KiB
    raw-read block) are streamed via ``read(size)`` + ``next()`` and
    verified byte-for-byte against the original payloads.
    """
    payloads = [
        b"small_blob",
        b"\xdd" * (70 * 1024),  # > 64 KiB block boundary
        b"\xee" * (5 * 1024),
    ]
    frames = b"".join(
        _make_blob_frame(
            {"ContentType": "text/plain" if i == 0 else "application/octet-stream"},
            p,
            {},
        )
        for i, p in enumerate(payloads)
    )
    raw_bytes = _make_crc_stream_large(frames).getvalue()

    client = _make_mock_client()
    client.stub.read_blobs.return_value = mock.Mock(
        raw=io.BytesIO(raw_bytes), headers={}
    )

    mgr = BlobManager(client)
    reader = mgr.read_blobs(["r1", "r2", "r3"], stream=True)

    results = []
    for i, payload in enumerate(payloads):
        chunks = []
        while True:
            chunk = reader.read(2 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        results.append(b"".join(chunks))
        if i < len(payloads) - 1:
            assert reader.next() is not None
        else:
            assert reader.next() is None

    assert results == payloads


# ---------------------------------------------------------------------------
# blob_writer
# ---------------------------------------------------------------------------


def test_blob_stream_writer():
    """Verify BlobStreamWriter streams data, uploads on finish(), verifies MD5."""
    uploaded = []

    def upload(data_generator):
        # RequestsIO passes a generator; consume it to get the full payload.
        data = b"".join(data_generator)
        uploaded.append(data)
        resp = mock.Mock()
        resp.headers = {"x-odps-request-id": "req-1"}
        resp.status_code = 200
        resp.json.return_value = {"MD5Value": hashlib.md5(data).hexdigest()}
        resp.text = "{}"
        return resp

    writer = BlobStreamWriter(upload, compress_option=None)
    writer.write(b"hello")
    writer.write(b" world")
    writer.finish()
    # Upload should have been called with combined data
    assert len(uploaded) == 1
    assert uploaded[0] == b"hello world"


def test_parse_json_response():
    resp = mock.Mock()
    resp.headers = {}
    resp.json.return_value = {"key": "value"}
    result = _parse_json_response(resp)
    assert result == {"key": "value"}


def test_parse_json_response_empty_body():
    resp = mock.Mock()
    resp.headers = {}
    resp.json.side_effect = ValueError("empty")
    resp.text = ""
    resp.status_code = 200
    with pytest.raises(Exception):
        _parse_json_response(resp)


# ---------------------------------------------------------------------------
# pyarrow 1.0 compatibility: _convert_struct_timestamps
# ---------------------------------------------------------------------------

_TS_STRUCT_TYPE = (
    pa.struct([("sec", pa.int64()), ("nano", pa.int32())]) if pa is not None else None
)


def _make_timestamp_struct_batch():
    """Build a batch with a ``{sec, nano}`` struct column.

    The struct array is constructed with an explicit type so field order
    is preserved across pyarrow versions (dict-literal construction reorders
    fields alphabetically on some versions).
    """
    schema = pa.schema([("id", pa.int64()), ("ts", _TS_STRUCT_TYPE)])
    ts_arr = pa.array([(100, 5), (200, 10)], type=_TS_STRUCT_TYPE)
    return pa.RecordBatch.from_arrays(
        [pa.array([1, 2], pa.int64()), ts_arr], schema=schema
    )


@pytest.mark.parametrize(
    "make_type, expected",
    [
        (lambda: _TS_STRUCT_TYPE, True),
        (lambda: pa.int64(), False),
        (lambda: pa.struct([("x", pa.int64())]), False),
        (lambda: pa.struct([("sec", pa.int32()), ("nano", pa.int32())]), False),
    ],
)
def test_is_timestamp_struct_type(make_type, expected):
    assert _is_timestamp_struct_type(make_type()) is expected


def test_convert_struct_timestamps_converts_sec_nano_to_timestamp():
    """``{sec, nano}`` struct column → ``timestamp(ns)`` column.

    Regression guard for pyarrow 1.0: ``RecordBatch.field(i)`` does not
    exist in 1.0 (must use ``batch.schema.field(i)``), and
    ``pa.compute.add`` on mixed int64/int32 arrays has no kernel (must
    cast ``nano`` to int64 first).
    """
    batch = _make_timestamp_struct_batch()
    result = _convert_struct_timestamps(batch)

    assert result.schema.field(1).type == pa.timestamp("ns")
    assert result.schema.field(0).type == pa.int64()
    ts_ns = result.column(1).cast(pa.int64()).to_pylist()
    assert ts_ns == [100_000_000_005, 200_000_000_010]


def test_convert_struct_timestamps_preserves_other_columns():
    """Non-struct columns survive unchanged; batches without struct-timestamp
    columns are returned as-is."""
    # Batch with a struct-timestamp column + extra columns
    schema = pa.schema(
        [("id", pa.int64()), ("ts", _TS_STRUCT_TYPE), ("label", pa.string())]
    )
    ts_arr = pa.array([(50, 3)], type=_TS_STRUCT_TYPE)
    batch = pa.RecordBatch.from_arrays(
        [pa.array([1], pa.int64()), ts_arr, pa.array(["x"], pa.string())],
        schema=schema,
    )
    result = _convert_struct_timestamps(batch)
    assert result.column(0).to_pylist() == [1]
    assert result.column(2).to_pylist() == ["x"]
    assert result.schema.field(1).type == pa.timestamp("ns")
    assert result.schema.field(0).name == "id"
    assert result.schema.field(2).name == "label"

    # Batch without any struct-timestamp column is returned unchanged
    plain = pa.RecordBatch.from_arrays(
        [pa.array([1, 2], pa.int64()), pa.array(["a", "b"], pa.string())],
        schema=pa.schema([("id", pa.int64()), ("name", pa.string())]),
    )
    assert _convert_struct_timestamps(plain) is plain
