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
    BlobRecord,
    BlobStreamReader,
    BlobWriteItem,
    ChecksumType,
    SessionStatus,
    SplitOptions,
)
from ..models import WriteSchema, parse_write_schema
from ..stream_io import _str_version_ge

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


def _serialize_item(item):
    """Serialize a single BlobWriteItem to bytes via write_frame_to."""
    buf = BytesIO()
    item.write_frame_to(buf)
    return buf.getvalue()


def _build_clean_stream(items):
    """Build a clean (no CRC, no LZ4) protocol frame stream from item dicts.

    Each item: {"data": bytes, "mime_type": str|None,
                "custom_file_name": str|None,
                "partition_values": list|None, "column_index": int}
    """
    buf = BytesIO()
    for item in items:
        header = {"ColumnIndex": item.get("column_index", 1)}
        if item.get("partition_values"):
            header["PartitionValues"] = item["partition_values"]
        if item.get("mime_type") is not None:
            header["ContentType"] = item["mime_type"]
        if item.get("custom_file_name") is not None:
            header["CustomFileName"] = item["custom_file_name"]
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


def _make_iterator(stream, api_version="3"):
    """Create a BlobDataIterator wired directly to a clean stream (bypassing CRC/LZ4)."""
    it = BlobDataIterator.__new__(BlobDataIterator)
    it._raw_stream = None
    it._current_stream = stream
    it._finished = False
    it._first = True
    it._api_version = api_version
    it._supports_custom_file_name = _str_version_ge(api_version, 3)
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
    item = BlobWriteItem(
        data=b"hello world",
        column_index=2,
        partition_values=["pt=2024"],
        mime_type="text/plain",
        custom_file_name="report.csv",
        api_version="3",
    )
    header, data, footer, _ = _parse_frame(_serialize_item(item))
    assert header["ColumnIndex"] == 2
    assert header["PartitionValues"] == ["pt=2024"]
    assert header["ContentType"] == "text/plain"
    assert header["CustomFileName"] == "report.csv"
    assert data == b"hello world"
    assert footer["Checksum"]["Type"] == 0
    assert "Crc32" not in footer["Checksum"]
    assert "MD5" not in footer["Checksum"]

    # checksums: NONE, CRC32, MD5
    item_none = BlobWriteItem(data=b"check me", checksum_type=ChecksumType.NONE)
    _, _, footer_none, _ = _parse_frame(_serialize_item(item_none))
    assert footer_none["Checksum"]["Type"] == 0
    assert "Crc32" not in footer_none["Checksum"]
    assert "MD5" not in footer_none["Checksum"]

    item_crc = BlobWriteItem(data=b"check me", checksum_type=ChecksumType.CRC32)
    _, _, footer_crc, _ = _parse_frame(_serialize_item(item_crc))
    assert footer_crc["Checksum"]["Type"] == 1
    assert footer_crc["Checksum"]["Crc32"] == (zlib.crc32(b"check me") & 0xFFFFFFFF)

    item_md5 = BlobWriteItem(data=b"check me", checksum_type=ChecksumType.MD5)
    _, _, footer_md5, _ = _parse_frame(_serialize_item(item_md5))
    assert footer_md5["Checksum"]["Type"] == 2
    assert footer_md5["Checksum"]["MD5"] == hashlib.md5(b"check me").hexdigest()

    # optional fields omitted when not set
    header_min = BlobWriteItem(data=b"x", column_index=1)._build_header()
    assert header_min["PartitionValues"] == []
    assert "DistributionKey" not in header_min
    assert "ContentType" not in header_min
    assert "CustomFileName" not in header_min
    assert header_min["ColumnIndex"] == 1

    # distribution key present when set
    header_dk = BlobWriteItem(data=b"x", distribution_key="abc123")._build_header()
    assert header_dk["DistributionKey"] == "abc123"

    # multiple items via serialize
    items = [
        BlobWriteItem(data=b"first", column_index=1),
        BlobWriteItem(data=b"second", column_index=2, mime_type="image/png"),
    ]
    raw = b"".join(_serialize_item(item) for item in items)
    header1, data1, _, pos = _parse_frame(raw)
    assert data1 == b"first"
    assert header1["ColumnIndex"] == 1
    header2, data2, _, _ = _parse_frame(raw, pos)
    assert data2 == b"second"
    assert header2["ContentType"] == "image/png"

    # empty list
    assert b"".join(_serialize_item(item) for item in []) == b""


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
        # For bytes data, write_frame_to must match _serialize_item
        assert output == _serialize_item(item)

    # Parse the frame and verify structure
    header, parsed_data, footer, _ = _parse_frame(output)
    assert header["ColumnIndex"] == 1
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
    """Multiple items streamed via write_frame_to match _serialize_item."""
    items = [
        BlobWriteItem(data=b"first", column_index=1),
        BlobWriteItem(data=b"second", column_index=2, mime_type="image/png"),
    ]

    buf = BytesIO()
    for item in items:
        item.write_frame_to(buf)
    output = buf.getvalue()

    assert output == b"".join(_serialize_item(item) for item in items)


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
    assert results[0] == BlobRecord(b"hello world", "text/plain")

    # multiple blobs with and without mime type
    stream = _build_clean_stream(
        [
            {"data": b"blob1", "mime_type": "text/plain", "column_index": 1},
            {"data": b"blob2", "mime_type": "image/png", "column_index": 2},
            {"data": b"blob3", "column_index": 3},
        ]
    )
    results = list(_make_iterator(stream))
    assert results == [
        BlobRecord(b"blob1", "text/plain"),
        BlobRecord(b"blob2", "image/png"),
        BlobRecord(b"blob3", None),
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
        BlobWriteItem(data=b"second blob", column_index=1),
    ]
    raw = b"".join(_serialize_item(item) for item in items)
    results = list(_make_iterator(BytesIO(raw)))
    assert results == [
        BlobRecord(b"first blob", "application/octet-stream"),
        BlobRecord(b"second blob", None),
    ]


def test_blob_stream_reader_framed():
    """Test BlobStreamReader with multiple framed blobs."""
    stream = _build_clean_stream(
        [
            {"data": b"hello world", "mime_type": "text/plain", "column_index": 1},
            {"data": b"second blob", "mime_type": "image/png", "column_index": 2},
            {"data": b"third", "column_index": 3},
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
            {"data": b"hello world", "mime_type": "text/plain", "column_index": 1},
            {"data": b"second", "column_index": 2},
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
    stream = _build_clean_stream([{"data": b"raw blob data", "column_index": 1}])
    it = _make_iterator(stream)
    reader = BlobStreamReader(it)

    # Raw blobs have no mime_type or custom_file_name
    assert reader.mime_type is None
    assert reader.custom_file_name is None
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


def test_blob_data_iterator_custom_file_name():
    """BlobDataIterator yields custom_file_name as the third tuple element."""
    # single blob with custom_file_name only
    stream = _build_clean_stream(
        [{"data": b"hello", "custom_file_name": "data.csv", "column_index": 1}]
    )
    results = list(_make_iterator(stream))
    assert results == [BlobRecord(b"hello", None, "data.csv")]

    # multiple blobs: mix of mime_type and custom_file_name
    stream = _build_clean_stream(
        [
            {
                "data": b"b1",
                "mime_type": "text/plain",
                "custom_file_name": "a.txt",
                "column_index": 1,
            },
            {"data": b"b2", "mime_type": "image/png", "column_index": 2},
            {"data": b"b3", "custom_file_name": "c.json", "column_index": 3},
            {"data": b"b4", "column_index": 4},
        ]
    )
    results = list(_make_iterator(stream))
    assert results == [
        BlobRecord(b"b1", "text/plain", "a.txt"),
        BlobRecord(b"b2", "image/png", None),
        BlobRecord(b"b3", None, "c.json"),
        BlobRecord(b"b4", None, None),
    ]

    items = [
        BlobWriteItem(
            data=b"first",
            column_index=1,
            mime_type="application/json",
            custom_file_name="payload.json",
            api_version="3",
        ),
        BlobWriteItem(
            data=b"second", column_index=1, custom_file_name="notes.md", api_version="3"
        ),
        BlobWriteItem(data=b"third", column_index=1, api_version="3"),
    ]
    raw = b"".join(_serialize_item(item) for item in items)
    results = list(_make_iterator(BytesIO(raw)))
    assert results == [
        BlobRecord(b"first", "application/json", "payload.json"),
        BlobRecord(b"second", None, "notes.md"),
        BlobRecord(b"third", None, None),
    ]


def test_blob_stream_reader_custom_file_name():
    """BlobStreamReader exposes custom_file_name per blob."""
    stream = _build_clean_stream(
        [
            {
                "data": b"hello world",
                "mime_type": "text/plain",
                "custom_file_name": "greeting.txt",
                "column_index": 1,
            },
            {"data": b"second", "custom_file_name": "second.bin", "column_index": 2},
            {"data": b"third", "column_index": 3},
        ]
    )
    it = _make_iterator(stream)
    reader = BlobStreamReader(it)

    # First blob: both mime_type and custom_file_name
    assert reader.mime_type == "text/plain"
    assert reader.custom_file_name == "greeting.txt"
    assert reader.read() == b"hello world"

    # Second blob: custom_file_name only
    reader2 = reader.next()
    assert reader2 is not None
    assert reader2.mime_type is None
    assert reader2.custom_file_name == "second.bin"
    assert reader2.read() == b"second"

    # Third blob: neither set
    reader3 = reader2.next()
    assert reader3 is not None
    assert reader3.mime_type is None
    assert reader3.custom_file_name is None
    assert reader3.read() == b"third"

    # No more blobs
    assert reader3.next() is None


def test_custom_file_name_omitted_on_v2():
    """custom_file_name is gated behind api_version >= 3 at every layer."""
    item = BlobWriteItem(data=b"hello", column_index=1, custom_file_name="data.csv")
    # _build_header: v2 omits, v3 includes
    assert "CustomFileName" not in item._build_header()
    item.api_version = "3"
    assert item._build_header()["CustomFileName"] == "data.csv"

    # write_frame_to round-trip
    item.api_version = "2"
    assert "CustomFileName" not in _parse_frame(_serialize_item(item))[0]
    item.api_version = "3"
    assert _parse_frame(_serialize_item(item))[0]["CustomFileName"] == "data.csv"

    # read side: a stream carrying CustomFileName in its header
    stream = _build_clean_stream(
        [
            {
                "data": b"hello",
                "mime_type": "text/plain",
                "custom_file_name": "a.txt",
                "column_index": 1,
            }
        ]
    )
    # v2 iterator/reader ignore it; v3 parse it
    assert list(_make_iterator(stream, api_version="2")) == [
        BlobRecord(b"hello", "text/plain", None)
    ]
    stream2 = _build_clean_stream(
        [
            {
                "data": b"hello",
                "mime_type": "text/plain",
                "custom_file_name": "a.txt",
                "column_index": 1,
            }
        ]
    )
    assert list(_make_iterator(stream2, api_version="3")) == [
        BlobRecord(b"hello", "text/plain", "a.txt")
    ]
    reader = BlobStreamReader(
        _make_iterator(
            _build_clean_stream(
                [{"data": b"hello", "custom_file_name": "a.txt", "column_index": 1}]
            ),
            api_version="2",
        )
    )
    assert reader.custom_file_name is None
    assert reader.read() == b"hello"


# Nested blob write-schema parsing (v3+)

_BIGINT = 0
_STRING = 4
_ARRAY = 17
_MAP = 18
_STRUCT = 19
_BLOB = 22


def _ct(name, type_code, column_id, sub_types=None):
    node = {"MemberName": name, "ColumnId": column_id, "Type": type_code}
    if sub_types is not None:
        node["SubTypes"] = sub_types
    return node


def _col(name, type_code, column_id, sub_types=None):
    return {
        "comment": "",
        "label": "",
        "columnType": _ct(name, type_code, column_id, sub_types),
    }


def _schema(columns):
    return {"DataColumns": columns}


@pytest.mark.parametrize(
    "columns,expected",
    [
        # c1 BIGINT, c2 BLOB
        ([_col("c1", _BIGINT, 1), _col("c2", _BLOB, 2)], {"c2": 2}),
        # c2 ARRAY<BLOB>
        (
            [_col("c1", _BIGINT, 1), _col("c2", _ARRAY, 2, [_ct("element", _BLOB, 3)])],
            {"c2.element": 3},
        ),
        # c2 STRUCT<f1:STRING, f2:BLOB>
        (
            [
                _col("c1", _BIGINT, 1),
                _col("c2", _STRUCT, 2, [_ct("f1", _STRING, 3), _ct("f2", _BLOB, 4)]),
            ],
            {"c2.f2": 4},
        ),
        # c2 ARRAY<STRUCT<f1:STRING, f2:BLOB>>
        (
            [
                _col("c1", _BIGINT, 1),
                _col(
                    "c2",
                    _ARRAY,
                    2,
                    [
                        _ct(
                            "element",
                            _STRUCT,
                            3,
                            [_ct("f1", _STRING, 4), _ct("f2", _BLOB, 5)],
                        )
                    ],
                ),
            ],
            {"c2.element.f2": 5},
        ),
        # c2 STRUCT<f1:STRING, f2:ARRAY<BLOB>>
        (
            [
                _col("c1", _BIGINT, 1),
                _col(
                    "c2",
                    _STRUCT,
                    2,
                    [
                        _ct("f1", _STRING, 3),
                        _ct("f2", _ARRAY, 4, [_ct("element", _BLOB, 5)]),
                    ],
                ),
            ],
            {"c2.f2.element": 5},
        ),
        # c2 MAP<STRING, BLOB>
        (
            [
                _col("c1", _BIGINT, 1),
                _col("c2", _MAP, 2, [_ct("key", _STRING, 3), _ct("value", _BLOB, 4)]),
            ],
            {"c2.value": 4},
        ),
        # no blob column at all
        ([_col("c1", _BIGINT, 1), _col("c2", _STRING, 2)], {}),
    ],
    ids=[
        "top-level-blob",
        "array-blob",
        "struct-blob",
        "array-struct-blob",
        "struct-array-blob",
        "map-string-blob",
        "no-blob",
    ],
)
def test_parse_write_schema_finds_nested_blob_column_ids(columns, expected):
    """find_all_blob_column_ids resolves dot-paths for nested BLOB columns."""
    ws = parse_write_schema(_schema(columns))
    assert isinstance(ws, WriteSchema)
    assert ws.find_all_blob_column_ids() == expected


def test_parse_write_schema_nested_column_ids_map():
    """nested_column_ids records every node, not just BLOBs."""
    ws = parse_write_schema(
        _schema(
            [
                _col("c1", _BIGINT, 1),
                _col(
                    "c2",
                    _ARRAY,
                    2,
                    [
                        _ct(
                            "element",
                            _STRUCT,
                            3,
                            [_ct("f1", _STRING, 4), _ct("f2", _BLOB, 5)],
                        )
                    ],
                ),
            ]
        )
    )
    assert ws.nested_column_ids == {
        "c1": 1,
        "c2": 2,
        "c2.element": 3,
        "c2.element.f1": 4,
        "c2.element.f2": 5,
    }
    assert ws.get_nested_column_id("c2.element.f2") == 5
    assert ws.get_nested_column_id("missing") is None
    assert ws.columns[0]["columnType"]["MemberName"] == "c1"
    assert ws.raw == {"DataColumns": ws.columns}


def test_parse_write_schema_none_and_empty():
    """parse_write_schema returns None for falsy input, parses empty schema."""
    assert parse_write_schema(None) is None
    assert parse_write_schema({}) is None
    ws = parse_write_schema({"DataColumns": []})
    assert ws is not None
    assert ws.find_all_blob_column_ids() == {}
    assert ws.nested_column_ids == {}


def test_parse_write_schema_rejects_non_dict():
    """parse_write_schema rejects a non-dict TableSchema."""
    with pytest.raises(ValueError, match="Expected a dict"):
        parse_write_schema(["not", "a", "dict"])


def test_parse_write_schema_rejects_bad_array_arity():
    """ARRAY with != 1 sub-type is rejected at parse time."""
    with pytest.raises(ValueError, match="ARRAY"):
        parse_write_schema(
            _schema([_col("c2", _ARRAY, 2, [_ct("a", _BLOB, 3), _ct("b", _BLOB, 4)])])
        )


def test_blob_record_tuple_compat():
    """BlobRecord behaves like a 2-tuple (data, mime_type) for backward compat.

    Tuple-style access is deprecated; the warning is suspended here so the
    legacy contract is exercised without failing under ``-W error``.
    """
    import warnings

    r = BlobRecord(b"data", "text/plain", "file.txt")

    # attribute access is the primary, non-deprecated interface
    assert r.data == b"data"
    assert r.mime_type == "text/plain"
    assert r.custom_file_name == "file.txt"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # 2-unpack and iteration yield only (data, mime_type)
        data, mime_type = r
        assert (data, mime_type) == (b"data", "text/plain")
        assert list(r) == [b"data", "text/plain"]

        # length, indexing and slicing follow the 2-tuple view
        assert len(r) == 2
        assert r[0] == b"data"
        assert r[1] == "text/plain"
        assert r[:] == (b"data", "text/plain")
        assert r[-1] == "text/plain"

        # 3-unpack is NOT supported (custom_file_name is attribute-only)
        with pytest.raises(ValueError):
            _x, _y, _z = r  # noqa: F841

        # equality with bare tuples uses the 2-tuple view; a record whose
        # custom_file_name is set does NOT equal the 2-tuple
        assert r != (b"data", "text/plain")
        # a record with custom_file_name None equals the 2-tuple
        assert BlobRecord(b"data", "text/plain", None) == (b"data", "text/plain")
        # 3-tuples never compare equal
        assert r != (b"data", "text/plain", "file.txt")
        assert r != (b"data", "text/plain", None)

        # bare tuple on the left reflects to BlobRecord.__eq__
        assert (b"data", "text/plain") == BlobRecord(b"data", "text/plain", None)
        assert (b"data", "text/plain") != r

    # BlobRecord vs BlobRecord compares all three fields (not deprecated)
    assert r == BlobRecord(b"data", "text/plain", "file.txt")
    assert r != BlobRecord(b"data", "text/plain", None)
    assert r != BlobRecord(b"other", "text/plain", "file.txt")
    # non-tuple/non-BlobRecord equality is NotImplemented -> False
    assert not (r == 5)
    assert not (r == "data")


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
    downloaded = [
        record.data for record in client.read_blobs(blob_references=read_blob_refs)
    ]

    assert len(downloaded) >= 2
    for d in downloaded:
        assert d in blob_data_list


def test_blob_batch_write_and_read(storage_api_blob_client):
    """Write blobs via batch upload with MIME types and custom file names, then read them back."""
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

    # ---- Upload multiple blobs in a single batch with MIME types and custom file names ----
    blob_data_list = [
        (b"batch blob 1 - text data", "text/plain", "data1.txt"),
        (b"batch blob 2 - image data", "image/png", "photo2.png"),
        (b"batch blob 3 - json data", "application/json", "payload3.json"),
    ]
    items = [
        BlobWriteItem(
            data=data,
            column_index=2,
            mime_type=mime,
            custom_file_name=cfn,
        )
        for data, mime, cfn in blob_data_list
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

    # ---- Read blobs back and verify MIME types and custom file names ----
    downloaded = list(client.read_blobs(blob_references=all_blob_refs))

    assert len(downloaded) >= len(blob_data_list)
    downloaded_data = [record.data for record in downloaded]
    for data, _, _ in blob_data_list:
        assert data in downloaded_data

    # Verify MIME types are preserved on read
    downloaded_mime_types = [record.mime_type for record in downloaded]
    expected_mime_types = [mime for _, mime, _ in blob_data_list]
    # The server may not always return MIME types, so check that
    # when they are present they match
    for expected, actual in zip(expected_mime_types, downloaded_mime_types):
        if actual is not None:
            assert expected == actual

    # Verify custom file names are preserved on read
    downloaded_cfn = [record.custom_file_name for record in downloaded]
    expected_cfn = [cfn for _, _, cfn in blob_data_list]
    for expected, actual in zip(expected_cfn, downloaded_cfn):
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


# Nested blob e2e (requires live ODPS cluster, v3 API)


ARRAY_BLOB_DATA = [
    [b"array-blob-0-0", b"array-blob-0-1"],
    [b"array-blob-1-0", b"array-blob-1-1"],
]

# c3 ARRAY<STRUCT<f1:STRING, f2:BLOB>> — one struct per row, each with a blob
ARRAY_STRUCT_BLOB_DATA = [
    [{"f1": "row0", "f2": b"arr-struct-blob-0"}],
    [{"f1": "row1", "f2": b"arr-struct-blob-1"}],
]


def test_nested_array_blob_write_and_read(storage_api_nested_blob_client):
    """Write nested-blob columns end to end, then read the blobs back.

    Resolves ``c2.element`` (ARRAY<BLOB>) and ``c3.element.f2``
    (ARRAY<STRUCT<f1:STRING, f2:BLOB>>) column IDs, uploads blobs with
    each ID, writes arrow rows, commits, then verifies round-trip.
    """
    client = storage_api_nested_blob_client
    partition = "pt=test_nested_array_blob"

    # ---- Create write session + stream ----
    write_resp = client.create_write_session(partial_partition_spec=partition)
    session_id = write_resp.session_id
    stream_resp = client.create_write_stream(
        session_id, stream_id="stream-1", stream_version=1
    )
    assert stream_resp.request_id != ""

    # ---- Resolve the nested blob column ID for c2.element ----
    blob_ids = client.get_nested_blob_column_ids(stream_response=stream_resp)
    assert (
        "c2.element" in blob_ids
    ), f"expected nested blob column id for c2.element, got {blob_ids}"
    blob_column_id = blob_ids["c2.element"]

    # ---- Resolve the nested blob column ID for c3.element.f2 ----
    assert (
        "c3.element.f2" in blob_ids
    ), f"expected nested blob column id for c3.element.f2, got {blob_ids}"
    struct_blob_column_id = blob_ids["c3.element.f2"]

    # ---- Upload blobs for c2 (ARRAY<BLOB>) and c3 (ARRAY<STRUCT<...,BLOB>>) ----
    blob_refs = []  # c2 refs in row-major order
    for row_blobs in ARRAY_BLOB_DATA:
        for blob_data in row_blobs:
            writer = client.write_blob_stream(
                session_id,
                stream_id="stream-1",
                stream_version=1,
                column_index=blob_column_id,
            )
            writer.write(blob_data)
            resp = writer.finish()
            assert resp is not None
            assert resp.blob_reference is not None
            blob_refs.append(resp.blob_reference)

    struct_blob_refs = []  # c3 refs, one per struct's f2 field
    for row_structs in ARRAY_STRUCT_BLOB_DATA:
        for item in row_structs:
            writer = client.write_blob_stream(
                session_id,
                stream_id="stream-1",
                stream_version=1,
                column_index=struct_blob_column_id,
            )
            writer.write(item["f2"])
            resp = writer.finish()
            assert resp is not None
            assert resp.blob_reference is not None
            struct_blob_refs.append(resp.blob_reference)

    # ---- Build arrow rows with blob refs as varbinary ----
    ref_iter = iter(blob_refs)
    struct_ref_iter = iter(struct_blob_refs)
    c1_values = list(range(len(ARRAY_BLOB_DATA)))
    c2_values = [
        [base64.b64decode(next(ref_iter)) for _ in row] for row in ARRAY_BLOB_DATA
    ]
    c3_values = [
        [
            {"f1": item["f1"], "f2": base64.b64decode(next(struct_ref_iter))}
            for item in row_structs
        ]
        for row_structs in ARRAY_STRUCT_BLOB_DATA
    ]
    struct_type = pa.struct([("f1", pa.string()), ("f2", pa.binary())])
    record_batch = pa.RecordBatch.from_arrays(
        [
            pa.array(c1_values, type=pa.int64()),
            pa.array(c2_values),
            pa.array(c3_values, type=pa.list_(struct_type)),
        ],
        names=["c1", "c2", "c3"],
    )

    writer = client.write_rows_arrow(
        session_id,
        stream_id="stream-1",
        stream_version=1,
        record_count=len(ARRAY_BLOB_DATA),
    )
    assert writer.write(record_batch) is True
    _, suc = writer.finish()
    assert suc is True

    # ---- Close stream and commit ----
    client.close_write_stream(session_id, stream_id="stream-1", stream_version=1)
    client.commit_write_session(session_id)

    # ---- Create read session and poll until NORMAL ----
    split_opts = SplitOptions(
        split_mode=SplitOptions.SplitMode.ROW_OFFSET,
        split_number=256 * 1024 * 1024,
    )
    read_resp = client.create_read_session(
        required_partitions=[partition],
        split_options=split_opts,
    )
    read_session_id = read_resp.session_id
    read_route_token = read_resp.route_token
    for _ in range(60):
        read_resp = client.get_read_session(read_session_id)
        if read_resp.session_status != SessionStatus.INIT:
            break
        time.sleep(1)
    if read_resp.route_token:
        read_route_token = read_resp.route_token

    # ---- Read rows back and collect nested blob references ----
    record_count = read_resp.record_count or len(ARRAY_BLOB_DATA)
    split_number = split_opts.split_number
    all_blob_refs = []
    for offset in range(0, record_count, split_number):
        count = min(split_number, record_count - offset)
        reader = client.read_rows_stream(
            session_id=read_session_id,
            row_offset=offset,
            row_count=count,
            max_batch_rows=4096,
            route_token=read_route_token,
        )
        buf = b""
        while True:
            data = reader.read(65536)
            if len(data) == 0:
                break
            buf += data
        reader.close()
        if buf:
            with pa.ipc.open_stream(buf) as arrow_reader:
                for batch in arrow_reader:
                    # c2 (col 1): list of varbinary refs
                    for row_refs in batch.column(1).to_pylist():
                        if row_refs is None:
                            continue
                        for ref in row_refs:
                            if ref is not None:
                                all_blob_refs.append(ref)
                    # c3 (col 2): list of struct<f1, f2(varbinary)>
                    for row_structs in batch.column(2).to_pylist():
                        if row_structs is None:
                            continue
                        for item in row_structs:
                            if item and item.get("f2") is not None:
                                all_blob_refs.append(item["f2"])

    expected_count = sum(len(r) for r in ARRAY_BLOB_DATA) + len(ARRAY_STRUCT_BLOB_DATA)
    assert len(all_blob_refs) >= expected_count

    # ---- Download the blobs and verify contents round-trip ----
    # The arrow varbinary column stores blob references as UTF-8 reference
    # strings; the BlobRead API expects these reference strings directly.
    read_blob_refs = [
        ref.decode("utf-8") if isinstance(ref, bytes) else ref for ref in all_blob_refs
    ]
    downloaded = [
        record.data for record in client.read_blobs(blob_references=read_blob_refs)
    ]

    expected = {blob for row in ARRAY_BLOB_DATA for blob in row}
    expected.update(item["f2"] for row in ARRAY_STRUCT_BLOB_DATA for item in row)
    for d in downloaded:
        assert d in expected
    assert len(downloaded) >= len(expected)
