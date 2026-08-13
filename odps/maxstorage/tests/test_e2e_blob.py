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

"""End-to-end tests for odps.maxstorage blob path against a real MaxCompute
service.

Two blob write patterns are exercised:

**Inline pattern** (``auto_upload_blobs=True``):
  Raw ``bytes`` (or ``BytesIO`` file-like objects via the record writer) are
  placed directly in the Arrow BLOB column.  The writer batch-uploads them
  and replaces cells with references transparently.

**Direct-reference pattern** (primary pattern for large blobs):
  Blob data is uploaded separately via ``TableArrowWriter.write_blob_stream``
  (streaming single upload with MD5 verification) or
  ``TableArrowWriter.write_blob_batch`` (batch upload), obtaining blob
  references.  Arrow rows are then written whose BLOB column holds the
  decoded references.

File-like objects (``io.BytesIO``) are used in several tests to verify the
streaming / chunked write paths.  Streaming reads are exercised via
``BlobManager.read_blobs(stream=True)``.
"""

import io

import pytest

from ..models.enums import SplitMode
from ..options import SplitOptions

try:
    import pyarrow as pa
except ImportError:
    pa = None

pytestmark = pytest.mark.skipif(pa is None, reason="Need pyarrow to run E2E tests")

_ROW_OFFSET_SPLIT = SplitOptions(split_mode=SplitMode.ROW_OFFSET, split_number=1000)

_BLOB_SCHEMA = (
    pa.schema([("a", pa.int64()), ("b", pa.binary())]) if pa is not None else None
)

_NESTED_BLOB_SCHEMA = (
    pa.schema(
        [
            ("a", pa.int64()),
            ("b", pa.list_(pa.binary())),
        ]
    )
    if pa is not None
    else None
)


def _read_all_rows(client, table, partition):
    """Read all rows from a partition via a read session.

    Returns a list of ``(a_val, b_val)`` tuples sorted by ``a``.
    """
    read_session = client.create_table_read_session(
        table,
        partitions=[partition],
        split_options=_ROW_OFFSET_SPLIT,
    )
    assert len(read_session.splits) > 0

    rows = []
    for split in read_session.splits:
        reader = read_session.open_arrow_reader(split)
        for batch in reader:
            a_col = batch.column(batch.schema.get_field_index("a"))
            b_col = batch.column(batch.schema.get_field_index("b"))
            for i in range(len(a_col)):
                rows.append((a_col[i].as_py(), b_col[i].as_py()))
        reader.close()
    rows.sort(key=lambda r: r[0])
    return rows


def _read_all_rows_instance(client, table, columns="a, b"):
    """Read all rows from a PK table via an instance read session.

    Returns a list of tuples sorted by the first column.
    """
    inst = client._odps.execute_sql(f"SELECT {columns} FROM {table.name}")
    read_session = client.create_instance_read_session(inst)
    reader = read_session.open_arrow_reader()
    rows = []
    while True:
        batch = reader.read()
        if batch is None:
            break
        for i in range(batch.num_rows):
            rows.append(
                tuple(batch.column(j)[i].as_py() for j in range(batch.num_columns))
            )
    reader.close()
    rows.sort(key=lambda r: r[0])
    return rows


def _download_blobs(client, blob_refs, stream=False, chunk_size=13, table=None):
    """Download blobs and return a list of ``bytes``.

    When ``stream=True``, uses ``read_blobs(stream=True)`` and reads each blob
    in ``chunk_size`` chunks via :class:`BlobStreamReader`.
    """
    blob_manager = client.open_blob_manager(table)
    if not stream:
        return [record.data for record in blob_manager.read_blobs(blob_refs)]

    reader = blob_manager.read_blobs(blob_refs, stream=True)
    results = []
    while reader is not None:
        buf = b""
        while True:
            chunk = reader.read(chunk_size)
            if not chunk:
                break
            buf += chunk
        results.append(buf)
        reader = reader.next()
    return results


# ---------------------------------------------------------------------------
# Inline pattern — raw bytes / BytesIO in Arrow BLOB column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "use_bytesio",
    [False, True],
    ids=["bytes", "bytesio"],
)
def test_blob_inline_write_and_read(maxstorage_blob_client, use_bytesio):
    """Write rows with BLOB data inline, then read back and verify.

    When ``use_bytesio`` is True, the record writer path is used with
    ``BytesIO`` file-like objects — exercising ``_intercept_blob_leaf``'s
    file-like → bytes conversion.
    """
    client, table = maxstorage_blob_client
    partition = f"pt=test_inline_{'bytesio' if use_bytesio else 'bytes'}"
    blob_payloads = [b"inline blob 0", b"inline blob 1", b"inline blob 2"]

    write_session = client.create_table_write_session(table, partition_spec=partition)
    writer = write_session.open_arrow_writer("stream-0", auto_upload_blobs=True)

    if use_bytesio:
        record_writer = writer.get_as_record_writer(row_count_per_batch=10)
        for i, payload in enumerate(blob_payloads):
            record_writer.write([i, io.BytesIO(payload)])
        record_writer.close()
    else:
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(list(range(len(blob_payloads))), pa.int64()),
                pa.array(blob_payloads, pa.binary()),
            ],
            schema=_BLOB_SCHEMA,
        )
        writer.write_batch(batch)
        writer.close()

    write_session.commit()

    rows = _read_all_rows(client, table, partition)
    assert len(rows) == len(blob_payloads)

    blob_refs = [ref for _, ref in rows]
    downloaded = _download_blobs(client, blob_refs)
    assert downloaded == blob_payloads


# ---------------------------------------------------------------------------
# Direct-reference pattern — upload blobs separately, write refs in Arrow rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "upload_method,use_bytesio,stream_read",
    [
        ("stream", False, False),
        ("stream", True, False),
        ("batch", False, False),
        ("batch", True, True),
    ],
    ids=[
        "stream-bytes",
        "stream-bytesio",
        "batch-bytes",
        "batch-bytesio",
    ],
)
def test_blob_direct_ref_and_read(
    maxstorage_blob_client, upload_method, use_bytesio, stream_read
):
    """Upload blobs separately, write refs in Arrow rows, read back and verify.

    Parametrized over:
    - ``upload_method``: ``"stream"`` (write_blob_stream) or ``"batch"``
      (write_blob_batch)
    - ``use_bytesio``: raw ``bytes`` or ``BytesIO`` file-like input
    - ``stream_read``: normal ``read_blobs`` or ``read_blobs(stream=True)``
      with chunked reads
    """
    client, table = maxstorage_blob_client
    label = f"{upload_method}-{'bytesio' if use_bytesio else 'bytes'}"
    if stream_read:
        label += "-streamread"
    partition = f"pt=test_ref_{label}"
    blob_payloads = [
        b"ref blob alpha - hello world",
        b"ref blob beta - the quick brown fox",
        b"ref blob gamma - lorem ipsum dolor sit amet",
    ]

    write_session = client.create_table_write_session(table, partition_spec=partition)
    writer = write_session.open_arrow_writer(f"stream-{label}")

    ref_bytes_list = []
    if upload_method == "stream":
        for idx, blob_data in enumerate(blob_payloads):
            stream_writer = writer.write_blob_stream(column_name="b")
            if use_bytesio:
                bio = io.BytesIO(blob_data)
                while True:
                    chunk = bio.read(7)
                    if not chunk:
                        break
                    stream_writer.write(chunk)
            else:
                stream_writer.write(blob_data)
            resp = stream_writer.finish()
            assert resp is not None
            assert resp.blob_reference is not None
            ref_bytes_list.append(resp.blob_reference)
    else:
        items = []
        for idx, blob_data in enumerate(blob_payloads):
            data = io.BytesIO(blob_data) if use_bytesio else blob_data
            items.append(writer.build_blob_write_item(data, column_name="b"))
        batch_resp = writer.write_blob_batch(items)
        assert len(batch_resp.blob_references) == len(blob_payloads)
        ref_bytes_list = list(batch_resp.blob_references)

    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(list(range(len(ref_bytes_list))), pa.int64()),
            pa.array(ref_bytes_list, pa.binary()),
        ],
        schema=_BLOB_SCHEMA,
    )
    writer.write_batch(batch)
    writer.close()
    write_session.commit()

    rows = _read_all_rows(client, table, partition)
    assert len(rows) == len(blob_payloads)

    blob_refs = [ref for _, ref in rows]
    downloaded = _download_blobs(client, blob_refs, stream=stream_read)
    assert downloaded == blob_payloads


def test_blob_batch_upload_with_metadata(maxstorage_blob_client):
    """Batch upload with MIME type and custom file name, verify round-trip."""
    client, table = maxstorage_blob_client
    partition = "pt=test_blob_meta"
    blob_payloads = [
        (b"batch blob 0 - text data", "text/plain", "data0.txt"),
        (b"batch blob 1 - image data", "image/png", "photo1.png"),
        (b"batch blob 2 - json data", "application/json", "payload2.json"),
    ]

    write_session = client.create_table_write_session(table, partition_spec=partition)
    writer = write_session.open_arrow_writer("stream-meta")

    items = [
        writer.build_blob_write_item(
            data,
            column_name="b",
            mime_type=mime,
            custom_file_name=cfn,
        )
        for data, mime, cfn in blob_payloads
    ]
    batch_resp = writer.write_blob_batch(items)
    assert len(batch_resp.blob_references) == len(blob_payloads)
    ref_bytes_list = list(batch_resp.blob_references)

    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(list(range(len(ref_bytes_list))), pa.int64()),
            pa.array(ref_bytes_list, pa.binary()),
        ],
        schema=_BLOB_SCHEMA,
    )
    writer.write_batch(batch)
    writer.close()
    write_session.commit()

    rows = _read_all_rows(client, table, partition)
    assert len(rows) == len(blob_payloads)

    blob_refs = [ref for _, ref in rows]
    blob_manager = client.open_blob_manager()
    downloaded = list(blob_manager.read_blobs(blob_refs))
    assert len(downloaded) == len(blob_payloads)

    downloaded_data = [record.data for record in downloaded]
    for data, _, _ in blob_payloads:
        assert data in downloaded_data

    for (_, expected_mime, expected_cfn), record in zip(blob_payloads, downloaded):
        if record.mime_type is not None:
            assert record.mime_type == expected_mime
        if record.custom_file_name is not None:
            assert record.custom_file_name == expected_cfn


# ---------------------------------------------------------------------------
# Nested blob — ARRAY<BLOB> direct-reference pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "use_bytesio",
    [False, True],
    ids=["bytes", "bytesio"],
)
def test_nested_blob_direct_ref_and_read(maxstorage_nested_blob_client, use_bytesio):
    """Upload nested blobs (array<blob>) via batch upload, write refs, read back.

    Each element in the ARRAY<BLOB> column is a blob reference uploaded
    separately via ``writer.write_blob_batch``.  When ``use_bytesio`` is True,
    every third element uses a ``BytesIO`` file-like object.
    """
    client, table = maxstorage_nested_blob_client
    partition = f"pt=test_nested_{'bytesio' if use_bytesio else 'bytes'}"
    row_blob_data = [
        [b"nested_0_a", b"nested_0_b"],
        [b"nested_1_a", b"nested_1_b", b"nested_1_c"],
    ]

    write_session = client.create_table_write_session(table, partition_spec=partition)
    writer = write_session.open_arrow_writer("stream-nested")

    all_blob_data = [d for row in row_blob_data for d in row]
    items = []
    for idx, d in enumerate(all_blob_data):
        if use_bytesio and idx % 3 == 2:
            items.append(
                writer.build_blob_write_item(io.BytesIO(d), column_name="b.element")
            )
        else:
            items.append(writer.build_blob_write_item(d, column_name="b.element"))

    batch_resp = writer.write_blob_batch(items)
    assert len(batch_resp.blob_references) == len(all_blob_data)
    all_ref_bytes = list(batch_resp.blob_references)

    ref_iter = iter(all_ref_bytes)
    row_ref_bytes = []
    for row in row_blob_data:
        row_ref_bytes.append([next(ref_iter) for _ in row])

    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([0, 1], pa.int64()),
            pa.array(row_ref_bytes, pa.list_(pa.binary())),
        ],
        schema=_NESTED_BLOB_SCHEMA,
    )
    writer.write_batch(batch)
    writer.close()
    write_session.commit()

    rows = _read_all_rows(client, table, partition)
    assert len(rows) == 2

    for (_, blob_ref_list), expected_list in zip(rows, row_blob_data):
        assert len(blob_ref_list) == len(expected_list)
        downloaded = _download_blobs(client, blob_ref_list, table=table)
        assert downloaded == expected_list


# ---------------------------------------------------------------------------
# Nested blob — ARRAY<BLOB> inline pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "use_bytesio",
    [False, True],
    ids=["bytes", "bytesio"],
)
def test_nested_blob_inline_write_and_read(maxstorage_nested_blob_client, use_bytesio):
    """Write rows with raw blob data inline in an ARRAY<BLOB> column.

    The ``TableArrowBlobUploadWriter`` transparently batch-uploads each
    nested BLOB cell and replaces it with a reference before writing the
    Arrow batch.  When ``use_bytesio`` is True, the record-writer path is
    used with ``BytesIO`` file-like objects, exercising the nested-blob
    interception + file-like → bytes conversion.
    """
    client, table = maxstorage_nested_blob_client
    partition = f"pt=test_nested_inline_{'bytesio' if use_bytesio else 'bytes'}"
    row_blob_data = [
        [b"n_inline_0_a", b"n_inline_0_b"],
        [b"n_inline_1_a", b"n_inline_1_b", b"n_inline_1_c"],
    ]

    write_session = client.create_table_write_session(table, partition_spec=partition)
    writer = write_session.open_arrow_writer(
        "stream-nested-inline", auto_upload_blobs=True
    )

    if use_bytesio:
        record_writer = writer.get_as_record_writer(row_count_per_batch=10)
        for i, row_blobs in enumerate(row_blob_data):
            record_writer.write([i, [io.BytesIO(d) for d in row_blobs]])
        record_writer.close()
    else:
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(list(range(len(row_blob_data))), pa.int64()),
                pa.array(row_blob_data, pa.list_(pa.binary())),
            ],
            schema=_NESTED_BLOB_SCHEMA,
        )
        writer.write_batch(batch)
        writer.close()

    write_session.commit()

    rows = _read_all_rows(client, table, partition)
    assert len(rows) == len(row_blob_data)

    for (_, blob_ref_list), expected_list in zip(rows, row_blob_data):
        assert len(blob_ref_list) == len(expected_list)
        downloaded = _download_blobs(client, blob_ref_list, table=table)
        assert downloaded == expected_list


# ---------------------------------------------------------------------------
# Single-blob download — the server returns raw unframed bytes for
# single-ref requests and framed data for multi-ref requests.  These
# tests exercise the single-ref raw path and verify that blob payloads
# whose first 8 bytes decode to a large LE int64 (e.g. npz magic bytes)
# are returned verbatim, not mis-parsed as frame headers.
# ---------------------------------------------------------------------------


def _write_single_blob(client, table, partition, stream_label, payload):
    """Upload *payload* as a single blob, write one Arrow row referencing it,
    commit, and return the blob reference read back from the table.
    """
    write_session = client.create_table_write_session(table, partition_spec=partition)
    writer = write_session.open_arrow_writer(stream_label)

    stream_writer = writer.write_blob_stream(column_name="b")
    stream_writer.write(payload)
    resp = stream_writer.finish()
    ref_bytes = resp.blob_reference

    batch = pa.RecordBatch.from_arrays(
        [pa.array([0], pa.int64()), pa.array([ref_bytes], pa.binary())],
        schema=_BLOB_SCHEMA,
    )
    writer.write_batch(batch)
    writer.close()
    write_session.commit()

    rows = _read_all_rows(client, table, partition)
    assert len(rows) == 1
    return rows[0][1]


@pytest.mark.parametrize(
    "download_method,compress_algo,payload",
    [
        ("read_blob", None, b"single blob download via read_blob"),
        # compress_algo="raw" requests an uncompressed response.
        ("read_blob", "raw", b"single blob with compress_algo=raw"),
        # npz magic bytes: first 8 bytes decode to a huge LE int64 that
        # would crash the frame parser if framing were forced.
        ("read_blobs", None, b"\x93NUMPY\x01\x00" + b"\x00" * 200),
        # Streaming raw mode — distinct BlobStreamReader code path.
        ("read_blob_stream", None, b"streaming single blob " * 50),
    ],
    ids=["read_blob", "read_blob-raw", "read_blobs-npz", "stream"],
)
def test_single_blob_download(
    maxstorage_blob_client, download_method, compress_algo, payload
):
    """Single-ref download uses raw mode (no frame parsing).

    The server returns raw unframed bytes for single-ref requests.  The
    iterator must never call the frame parser on such responses, even when
    the blob's first 8 bytes decode to a large LE int64 (e.g. npz magic).
    """
    client, table = maxstorage_blob_client
    label = download_method.replace("_", "-")
    if compress_algo:
        label += f"-{compress_algo}"
    partition = f"pt=test_single_{label.replace('-', '_')}"
    blob_ref = _write_single_blob(client, table, partition, f"stream-{label}", payload)

    blob_manager = client.open_blob_manager()
    kwargs = {} if compress_algo is None else {"compress_algo": compress_algo}

    if download_method == "read_blob":
        result = blob_manager.read_blob(blob_ref, **kwargs)
        assert result.read() == payload
    elif download_method == "read_blobs":
        records = list(blob_manager.read_blobs([blob_ref], **kwargs))
        assert len(records) == 1
        assert records[0].data == payload
    elif download_method == "read_blob_stream":
        downloaded = _download_blobs(
            client, [blob_ref], stream=True, chunk_size=37, **kwargs
        )
        assert len(downloaded) == 1
        assert downloaded[0] == payload


# ---------------------------------------------------------------------------
# Delta (PK) table with BLOB — record writer auto-upload + UPSERT/DELETE
# ---------------------------------------------------------------------------


def test_delta_blob_record_writer(maxstorage_delta_blob_client):
    """Write UPSERTs with inline BLOB to a Delta+BLOB table via record writer.

    Exercises ``DeltaTableRecordWriter`` on a table with a BLOB column:
    the writer auto-uploads BLOB cells.  ``write()`` stamps
    ``__operation='U'`` and ``delete()`` stamps ``__operation='D'`` per
    record.  Both operations are interleaved on a single writer instance.
    """
    client, table = maxstorage_delta_blob_client
    blob_payloads = {1: b"alice-blob", 2: b"bob-blob", 3: b"carol-blob"}
    blob_updated = b"bob-blob-v2"

    write_session = client.create_table_write_session(table)
    writer = write_session.open_arrow_writer("stream-0", auto_upload_blobs=True)
    rw = writer.get_as_record_writer(row_count_per_batch=10)

    # UPSERT three rows with inline BLOB data
    rw.write([1, blob_payloads[1]])
    rw.write([2, blob_payloads[2]])
    rw.write([3, blob_payloads[3]])
    # UPSERT on existing key → overwrites blob
    rw.write([2, blob_updated])
    # DELETE key 3
    rw.delete([3, None])

    rw.close()
    write_session.commit()

    rows = _read_all_rows_instance(client, table, columns="id, b")

    assert [r[0] for r in rows] == [1, 2]

    # Download blobs and verify content
    blob_refs = [r[1] for r in rows]
    downloaded = _download_blobs(client, blob_refs, table=table)
    assert downloaded == [blob_payloads[1], blob_updated]


# ---------------------------------------------------------------------------
# Record reader — get_as_record_reader returns Record objects with BLOB refs
# ---------------------------------------------------------------------------


def test_blob_record_reader(maxstorage_blob_client):
    """Read BLOB rows via get_as_record_reader; verify refs are bytes.

    The record reader returns BLOB columns as raw ``bytes`` (the server-side
    blob reference).  The actual payload is fetched separately via
    ``BlobManager.read_blobs``.
    """
    client, table = maxstorage_blob_client
    partition = "pt=test_rr"
    blob_payloads = [b"alpha", b"beta", b"gamma"]

    write_session = client.create_table_write_session(table, partition_spec=partition)
    writer = write_session.open_arrow_writer("stream-0", auto_upload_blobs=True)

    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([1, 2, 3], pa.int64()),
            pa.array(blob_payloads, pa.binary()),
        ],
        schema=_BLOB_SCHEMA,
    )
    writer.write_batch(batch)
    writer.close()
    write_session.commit()

    # Read back via record reader
    read_session = client.create_table_read_session(
        table, partitions=[partition], split_options=_ROW_OFFSET_SPLIT
    )
    assert len(read_session.splits) > 0

    rows = []
    for split in read_session.splits:
        reader = read_session.open_arrow_reader(split)
        rr = reader.get_as_record_reader()
        for rec in rr:
            rows.append((rec[0], rec[1]))
        reader.close()
    rows.sort(key=lambda r: r[0])

    assert [r[0] for r in rows] == [1, 2, 3]

    # BLOB column is the reference bytes, not the payload
    blob_refs = [r[1] for r in rows]
    assert all(isinstance(ref, bytes) for ref in blob_refs)

    downloaded = _download_blobs(client, blob_refs, table=table)
    assert downloaded == blob_payloads
