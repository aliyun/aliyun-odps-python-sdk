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

import hashlib
import logging
import os
import struct
import time
from io import BytesIO

import mock
import pytest

from ....core import ODPS
from ....models import Instance

try:
    import pyarrow as pa
except ImportError:
    pa = None

if pa is not None:
    from ..client import (
        BlobDataIterator,
        SessionStatus,
        Status,
        StorageApiArrowClient,
        StorageApiClient,
    )
    from ..stream_io import BlobStreamWriter, StreamReader, _CRCStrippingStream
else:
    pytestmark = pytest.mark.skip("Need pyarrow to run this test")

logger = logging.getLogger(__name__)


def test_storage_api(storage_api_client):
    """End-to-end lifecycle: write session -> write stream -> read session -> read."""

    # ---- Create write session ----
    write_resp = storage_api_client.create_write_session(
        partial_partition_spec="pt=test_write"
    )

    assert write_resp.session_id is not None
    session_id = write_resp.session_id
    logger.info("Write session created: %s", session_id)

    # ---- Get write session ----
    get_resp = storage_api_client.get_write_session(session_id)
    assert get_resp.request_id != ""
    logger.info("Write session retrieved, streams: %s", get_resp.streams)

    # ---- Create write stream ----
    stream_resp = storage_api_client.create_write_stream(
        session_id=session_id, stream_id="stream-1", stream_version=1
    )
    assert stream_resp.request_id != ""
    logger.info("Write stream created, data_schema: %s", stream_resp.data_schema)

    # ---- Write arrow data via streaming upload ----
    bigint_list = list(range(4096))
    record_batch = pa.RecordBatch.from_arrays(
        [
            pa.array(bigint_list),
            pa.array(bigint_list),
            pa.array(bigint_list),
            pa.array(bigint_list),
        ],
        names=["a", "b", "c", "d"],
    )

    writer = storage_api_client.write_rows_arrow(
        session_id=session_id,
        stream_id="stream-1",
        stream_version=1,
        record_count=4096,
    )

    for _ in range(10):
        suc = writer.write(record_batch)
        assert suc is True

    commit_message, suc = writer.finish()
    assert suc is True
    logger.info(
        "Write rows finished, commit_message present: %s", commit_message is not None
    )

    # ---- Close write stream ----
    close_resp = storage_api_client.close_write_stream(
        session_id=session_id, stream_id="stream-1", stream_version=1
    )
    logger.info("Write stream closed, warning: %s", close_resp.warning_message)

    # ---- Commit write session ----
    storage_api_client.commit_write_session(session_id)
    logger.info("Write session committed")

    # ---- Create read session ----
    read_resp = storage_api_client.create_read_session(
        required_partitions=["pt=test_write"]
    )
    assert read_resp.session_id is not None
    read_session_id = read_resp.session_id
    logger.info("Read session created: %s", read_session_id)

    # ---- Poll read session until NORMAL ----
    for _ in range(60):
        read_resp = storage_api_client.get_read_session(read_session_id)
        if read_resp.session_status != SessionStatus.INIT:
            break
        logger.info("Read session still INIT, waiting...")
        time.sleep(1)

    splits_count = read_resp.splits_count
    assert splits_count is not None and splits_count > 0
    logger.info("Read session NORMAL, splits_count: %d", splits_count)

    # ---- Read all splits ----
    read_size = 65536
    buf = b""
    for i in range(splits_count):
        reader = storage_api_client.read_rows_stream(
            session_id=read_session_id,
            split_index=i,
            max_batch_rows=4096,
        )

        while True:
            data = reader.read(read_size)
            if len(data) == 0:
                break
            buf += data

        reader.close()
        assert reader.get_status() == Status.OK
        logger.info("Read split %d done", i)

    # Verify data can be deserialized as arrow IPC
    with pa.ipc.open_stream(buf) as arrow_reader:
        schema = arrow_reader.schema
        batches = [b for b in arrow_reader]
    logger.info("Schema: %s, batches: %d", schema, len(batches))
    assert len(batches) > 0


def test_abort_write_session(storage_api_client):
    """Test abort write session API."""
    write_resp = storage_api_client.create_write_session(
        partial_partition_spec="pt=test_abort"
    )

    assert write_resp.session_id is not None
    session_id = write_resp.session_id
    logger.info("Write session created for abort test: %s", session_id)

    # Abort should not raise
    storage_api_client.abort_write_session(session_id)
    logger.info("Write session aborted successfully")


# ---------------------------------------------------------------------------
# Instance-based client unit tests (mock, no server needed)
# ---------------------------------------------------------------------------


def _make_instance_client():
    """Build a StorageApiClient wired to a mock Instance."""
    odps = mock.MagicMock(spec=ODPS)
    instance = mock.MagicMock(spec=Instance)
    instance.id = "20250420000000000xxxxx"
    instance.project.name = "test_project"
    return StorageApiClient(odps, instance)


def test_instance_client_write_guard():
    """Write methods must raise ValueError when the client is instance-based."""
    client = _make_instance_client()

    with pytest.raises(ValueError, match="not supported"):
        client.create_write_session()

    with pytest.raises(ValueError, match="not supported"):
        client.get_write_session("sid")

    with pytest.raises(ValueError, match="not supported"):
        client.commit_write_session("sid")

    with pytest.raises(ValueError, match="not supported"):
        client.abort_write_session("sid")

    with pytest.raises(ValueError, match="not supported"):
        client.create_write_stream(
            session_id="sid",
            stream_id="1",
        )

    with pytest.raises(ValueError, match="not supported"):
        client.get_write_stream("sid", "1", 0)

    with pytest.raises(ValueError, match="not supported"):
        client.write_rows_stream(
            session_id="sid",
            stream_id="1",
        )

    with pytest.raises(ValueError, match="not supported"):
        client.close_write_stream(
            session_id="sid",
            stream_id="1",
        )

    with pytest.raises(ValueError, match="not supported"):
        client.preview_table()


# ---------------------------------------------------------------------------
# Instance-based client e2e test
# ---------------------------------------------------------------------------


def test_storage_api_read_instance(storage_api_client, odps):
    """End-to-end: read SQL instance results via StorageApiClient with instance."""
    table = storage_api_client.table

    # Run a SELECT against the table to produce an instance
    instance = odps.execute_sql(f"SELECT * FROM {table.name} LIMIT 100")
    instance.wait_for_success()
    logger.info("SQL instance created: %s", instance.id)

    # Build an instance-based client
    api_endpoint = os.getenv("ODPS_STORAGE_API_ENDPOINT")
    inst_client = StorageApiArrowClient(odps, instance, rest_endpoint=api_endpoint)

    # ---- Create read session ----
    read_resp = inst_client.create_read_session()
    assert read_resp.session_id is not None
    read_session_id = read_resp.session_id
    logger.info("Instance read session created: %s", read_session_id)

    # Instance reads use a single stream without split indices
    reader = inst_client.read_rows_stream(
        session_id=read_session_id,
        max_batch_rows=4096,
    )

    buf = b""
    while True:
        data = reader.read(65536)
        if len(data) == 0:
            break
        buf += data

    reader.close()
    assert reader.get_status() == Status.OK

    # Verify data can be deserialized as arrow IPC
    with pa.ipc.open_stream(buf) as arrow_reader:
        schema = arrow_reader.schema
        batches = [b for b in arrow_reader]
    logger.info("Instance schema: %s, batches: %d", schema, len(batches))
    assert schema is not None and len(schema) > 0


# ---------------------------------------------------------------------------
# previewTable e2e test
# ---------------------------------------------------------------------------


def test_preview_table(storage_api_client):
    """End-to-end: write data, then preview it via preview_table_arrow."""

    # ---- Write data first ----
    write_resp = storage_api_client.create_write_session(
        partial_partition_spec="pt=test_preview_v2"
    )
    assert write_resp.session_id is not None
    session_id = write_resp.session_id

    storage_api_client.create_write_stream(
        session_id=session_id, stream_id="stream-1", stream_version=1
    )

    bigint_list = list(range(100))
    record_batch = pa.RecordBatch.from_arrays(
        [
            pa.array(bigint_list),
            pa.array(bigint_list),
            pa.array(bigint_list),
            pa.array(bigint_list),
        ],
        names=["a", "b", "c", "d"],
    )

    writer = storage_api_client.write_rows_arrow(
        session_id=session_id,
        stream_id="stream-1",
        stream_version=1,
        record_count=100,
    )
    for _ in range(1):
        writer.write(record_batch)
    writer.finish()

    storage_api_client.close_write_stream(
        session_id=session_id, stream_id="stream-1", stream_version=1
    )
    storage_api_client.commit_write_session(session_id)

    # ---- Preview with limit ----
    reader = storage_api_client.preview_table_arrow(
        limit=15, partition="pt=test_preview_v2"
    )
    batch = reader.read()
    assert batch is not None
    assert batch.num_rows == 15
    logger.info("Preview with limit=15, rows: %d", batch.num_rows)

    # ---- Preview with column selection ----
    reader = storage_api_client.preview_table_arrow(
        columns=["a", "b"], partition="pt=test_preview_v2"
    )
    batch = reader.read()
    assert batch is not None
    assert batch.num_columns == 2
    logger.info("Preview with columns=[a,b], columns: %d", batch.num_columns)


# ---------------------------------------------------------------------------
# _CRCStrippingStream unit tests (no server needed)
# ---------------------------------------------------------------------------


def _make_crc_wire_payload(clean_data, block_size=4096, crc_size=4):
    """Build a wire-format payload by appending a dummy CRC after each block."""
    # Allow overriding defaults for smaller test payloads
    block_size = block_size or _CRCStrippingStream._CRC_BLOCK_SIZE
    crc_size = crc_size or _CRCStrippingStream._CRC_SIZE

    out = BytesIO()
    offset = 0
    while offset < len(clean_data):
        chunk = clean_data[offset : offset + block_size]
        out.write(chunk)
        out.write(b"\x00" * crc_size)  # dummy CRC
        offset += block_size
    return out.getvalue()


def test_crc_stripping_stream_read_all():
    """Reading with size=-1 returns all clean data."""
    clean = b"Hello, world!" * 1000  # > 4096 bytes
    wire = _make_crc_wire_payload(clean)
    raw = BytesIO(wire)

    s = _CRCStrippingStream(raw)
    result = s.read()
    assert result == clean


def test_crc_stripping_stream_read_small_chunks():
    """Reading in small chunks still yields correct data."""
    clean = b"ABCDEFGHIJ" * 500  # 5000 bytes
    wire = _make_crc_wire_payload(clean)
    raw = BytesIO(wire)

    s = _CRCStrippingStream(raw)
    got = b""
    while True:
        chunk = s.read(7)
        if not chunk:
            break
        got += chunk
    assert got == clean


def test_crc_stripping_stream_read_across_block_boundary():
    """A read that spans a CRC block boundary returns seamless data."""
    # Use a small block size (16 data + 4 CRC = 20 bytes per block)
    block_size = 16
    crc_size = 4
    clean = b"A" * block_size + b"B" * block_size + b"C" * 8  # 2 full + 1 tail
    wire = _make_crc_wire_payload(clean, block_size=block_size, crc_size=crc_size)
    raw = BytesIO(wire)

    s = _CRCStrippingStream(raw)
    s._CRC_BLOCK_SIZE = block_size
    s._CRC_SIZE = crc_size
    s._FULL_BLOCK_TOTAL = block_size + crc_size

    # Read 20 bytes which crosses the first block boundary
    chunk1 = s.read(20)
    assert chunk1 == b"A" * 16 + b"B" * 4
    # Read the rest
    rest = s.read()
    assert rest == b"B" * 12 + b"C" * 8


def test_crc_stripping_stream_peek():
    """peek() returns data without consuming it."""
    clean = b"Hello, peek world! " * 200  # > 4096 bytes
    wire = _make_crc_wire_payload(clean)
    raw = BytesIO(wire)

    s = _CRCStrippingStream(raw)

    # Peek first 8 bytes
    p1 = s.peek(8)
    assert p1 == clean[:8]

    # peek again — same result, stream not advanced
    p2 = s.peek(8)
    assert p2 == clean[:8]

    # read() after peek returns the same bytes
    r1 = s.read(8)
    assert r1 == clean[:8]

    # subsequent peek sees the next bytes
    p3 = s.peek(4)
    assert p3 == clean[8:12]

    # peek with size larger than remaining data still works
    rest_peek = s.peek(len(clean))
    assert rest_peek == clean[8:]

    # read the rest
    rest = s.read()
    assert rest == clean[8:]

    # peek on exhausted stream
    assert s.peek(8) == b""


def test_crc_stripping_stream_empty():
    """An empty raw stream yields empty data."""
    s = _CRCStrippingStream(BytesIO(b""))
    assert s.read() == b""


def test_crc_stripping_stream_partial_tail():
    """A tail block shorter than a full block is handled correctly."""
    # One full block + a tail of 5 data bytes + 4 CRC = 9 bytes
    block_size = 8
    crc_size = 4
    clean = b"ABCDEFGH" + b"XYZAB"
    wire = _make_crc_wire_payload(clean, block_size=block_size, crc_size=crc_size)
    raw = BytesIO(wire)

    s = _CRCStrippingStream(raw)
    s._CRC_BLOCK_SIZE = block_size
    s._CRC_SIZE = crc_size
    s._FULL_BLOCK_TOTAL = block_size + crc_size

    result = s.read()
    assert result == clean


def test_crc_stripping_stream_read_zero():
    """read(0) returns empty bytes without advancing the stream."""
    clean = b"some data here"
    wire = _make_crc_wire_payload(clean)
    raw = BytesIO(wire)

    s = _CRCStrippingStream(raw)
    assert s.read(0) == b""
    # Subsequent read-all should still return the full payload
    assert s.read() == clean


def test_blob_data_iterator_framed():
    """BlobDataIterator correctly parses framed data with CRC stripping."""
    # Build a framed payload: [HeaderLen][Header][DataLen][Data][FooterLen][Footer]
    header = b'{"ContentType": "text/plain"}'
    data = b"Hello, framed world!"
    footer = b'{"Checksum": {"Type": 0}}'

    framed = BytesIO()
    framed.write(struct.pack("<q", len(header)))
    framed.write(header)
    framed.write(struct.pack("<q", len(data)))
    framed.write(data)
    framed.write(struct.pack("<q", len(footer)))
    framed.write(footer)
    clean_data = framed.getvalue()

    wire = _make_crc_wire_payload(clean_data)
    raw = BytesIO(wire)

    it = BlobDataIterator(raw)
    results = list(it)
    assert len(results) == 1
    assert results[0] == (data, "text/plain")


def test_blob_data_iterator_raw():
    """BlobDataIterator correctly yields raw (unframed) data."""
    # Data that does NOT look like a framed header (first 8 bytes decode to
    # a value >= 1024 or < 0)
    clean = b"\xff" * 8 + b"raw blob payload"
    wire = _make_crc_wire_payload(clean)
    raw = BytesIO(wire)

    it = BlobDataIterator(raw)
    results = list(it)
    assert len(results) == 1
    assert results[0] == (clean, None)


# ---------------------------------------------------------------------------
# StreamReader unit tests (no server needed)
# ---------------------------------------------------------------------------


def _make_stream_reader(data, chunk_size=65536):
    """Build a StreamReader that reads from the given bytes."""
    raw_reader = mock.MagicMock()
    raw_reader.raw.read = mock.MagicMock()
    # Simulate reading in chunks
    offset = 0

    def _read(size):
        nonlocal offset
        chunk = data[offset : offset + size]
        offset += size
        return chunk

    raw_reader.raw.read.side_effect = _read
    raw_reader.headers = {"x-odps-request-id": "test-request-id"}

    reader = StreamReader(lambda: raw_reader)
    reader._chunk_size = chunk_size
    return reader


@pytest.mark.parametrize(
    "data,chunk_size,read_size",
    [
        (b"Hello, StreamReader!" * 100, 65536, None),  # read all
        (b"ABCDEFGHIJ" * 100, 64, 32),  # read in small chunks
        (b"short data", 65536, None),  # short data, EOF after one read
    ],
    ids=["read_all", "read_in_chunks", "short_eof"],
)
def test_stream_reader_read(data, chunk_size, read_size):
    """StreamReader reads data correctly and reaches EOF."""
    reader = _make_stream_reader(data, chunk_size=chunk_size)

    if read_size is None:
        result = reader.read()
        assert result == data
    else:
        got = b""
        while True:
            chunk = reader.read(read_size)
            if not chunk:
                break
            got += chunk
        assert got == data

    # After consuming all data, subsequent reads return empty bytes
    assert reader.read() == b""
    assert reader.read(100) == b""

    # The _eof flag should be set
    assert reader._eof is True


def test_stream_reader_close_stops_reads():
    """After close(), read() returns empty bytes immediately."""
    reader = _make_stream_reader(b"some data")

    reader.close()
    assert reader.read() == b""
    assert reader.get_status() == Status.OK


def test_stream_reader_get_request_id():
    """get_request_id returns the header value after close."""
    reader = _make_stream_reader(b"data")

    # Before close, returns None
    assert reader.get_request_id() is None

    reader.close()
    assert reader.get_request_id() == "test-request-id"


def test_stream_reader_status():
    """get_status returns RUNNING during read, OK after close."""
    reader = _make_stream_reader(b"data")
    assert reader.get_status() == Status.RUNNING
    reader.close()
    assert reader.get_status() == Status.OK


# ---------------------------------------------------------------------------
# BlobStreamWriter unit tests (no server needed)
# ---------------------------------------------------------------------------


def _make_blob_stream_writer(mock_response=None):
    """Build a BlobStreamWriter with mocked internals."""
    writer = BlobStreamWriter.__new__(BlobStreamWriter)
    writer._stopped = False
    writer._res = None
    writer._md5_digest = hashlib.md5()

    # Mock RequestsIO
    writer._req_io = mock.MagicMock()
    writer._req_io.start = mock.MagicMock()

    if mock_response is not None:
        writer._req_io.finish.return_value = mock_response
    else:
        writer._req_io.finish.return_value = None

    # Mock compressor — just pass data through
    writer._compressor = mock.MagicMock()

    return writer


def test_blob_stream_writer_write_and_md5():
    """write() updates the internal MD5 digest and returns True."""
    writer = _make_blob_stream_writer()

    assert writer.write(b"hello ") is True
    assert writer.write(b"world") is True

    expected_md5 = hashlib.md5(b"hello world").hexdigest()
    assert writer._md5_digest.hexdigest() == expected_md5


def test_blob_stream_writer_write_after_finish():
    """write() returns False after finish() has been called."""
    writer = _make_blob_stream_writer()

    md5_hex = writer._md5_digest.hexdigest()
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"MD5Value": md5_hex}
    resp.headers = {"x-odps-request-id": "blob-req-123"}
    writer._req_io.finish.return_value = resp
    writer.finish()

    assert writer.write(b"more data") is False
    assert writer.writable() is False


@pytest.mark.parametrize(
    "md5_hex,status_code,expected_result,expect_checksum_error",
    [
        # MD5 matches
        (lambda: hashlib.md5(b"test data").hexdigest(), 200, True, False),
        # No MD5 in response
        (lambda: None, 200, True, False),
        # MD5 mismatch
        (lambda: "badauthmd5value00000000000000000", 200, None, True),
        # Non-200 status
        (lambda: hashlib.md5(b"test data").hexdigest(), 500, None, False),
    ],
    ids=["md5_match", "no_md5_in_response", "md5_mismatch", "non_ok_status"],
)
def test_blob_stream_writer_finish(
    md5_hex, status_code, expected_result, expect_checksum_error
):
    """finish() handles various server response scenarios."""
    from ....errors import ChecksumError

    writer = _make_blob_stream_writer()
    writer.write(b"test data")

    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.headers = {"x-odps-request-id": "blob-req-123"}
    md5_val = md5_hex()
    if md5_val is not None:
        resp.json.return_value = {"MD5Value": md5_val}
    else:
        resp.json.return_value = {}
    writer._req_io.finish.return_value = resp

    if expect_checksum_error:
        with pytest.raises(ChecksumError, match="MD5 value mismatch"):
            writer.finish()
    else:
        result = writer.finish()
        if expected_result is True:
            assert result is not None
            assert writer.get_status() == Status.OK
        else:
            assert result is None
