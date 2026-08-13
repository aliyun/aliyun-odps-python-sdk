# -*- coding: utf-8 -*-
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

"""Record writers for :mod:`odps.maxstorage`.

Row-oriented writers wrapping :class:`TableArrowBlobUploadWriter`.  Converts
Record -> Arrow batches, intercepting BLOB columns to read raw data and
replace with references at write_batch time.
"""

import base64
import io
import logging
from typing import TYPE_CHECKING

try:
    import pyarrow as pa
except ImportError:
    pa = None

from ...tunnel.io.types import odps_type_to_arrow_type
from ...types import Array, Blob, Map, Struct
from ..errors import StorageClientError
from ..options import BlobWriteItem

if TYPE_CHECKING:
    from .writer import TableArrowWriter

logger = logging.getLogger(__name__)

OPERATION_COLUMN_NAME = "__operation"
OPERATION_UPSERT = ord("U")
OPERATION_DELETE = ord("D")


def _extract_record_values(record) -> list:
    """Return the values list from a record-like object.

    Handles ``odps.types.Record`` (``values``), alternate record objects
    (``_values``), and plain iterables (``list``/``tuple``).
    """
    if hasattr(record, "values"):
        return list(record.values)
    elif hasattr(record, "_values"):
        return list(record._values)
    else:
        return list(record)


def _replace_pending_refs(value, ref_map):
    """Recursively replace ``_PendingBlobRef`` placeholders with ref bytes.

    Walks nested ``list``/``dict`` structures in-place, replacing every
    ``_PendingBlobRef`` it encounters with the corresponding reference
    from ``ref_map``.  Plain values are returned unchanged.
    """
    if isinstance(value, _PendingBlobRef):
        return ref_map[value.item_index]
    if isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = _replace_pending_refs(v, ref_map)
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = _replace_pending_refs(v, ref_map)
    return value


def _odps_type_to_arrow_with_blob(data_type):
    """Like :func:`odps_type_to_arrow_type` but maps BLOB to ``pa.binary()``
    and recurses into ARRAY/STRUCT/MAP so nested BLOBs are handled too.
    """
    if isinstance(data_type, Blob):
        return pa.binary()
    if isinstance(data_type, Array):
        return pa.list_(_odps_type_to_arrow_with_blob(data_type.value_type))
    if isinstance(data_type, Map):
        return pa.map_(
            _odps_type_to_arrow_with_blob(data_type.key_type),
            _odps_type_to_arrow_with_blob(data_type.value_type),
        )
    if isinstance(data_type, Struct):
        fields = [
            pa.field(fn, _odps_type_to_arrow_with_blob(ft))
            for fn, ft in zip(data_type.field_names, data_type.field_types)
        ]
        return pa.struct(fields)
    return odps_type_to_arrow_type(data_type)


def _write_schema_to_arrow_schema(write_schema) -> "pa.schema":
    """Build a ``pa.schema`` from a :class:`WriteSchema`.

    Like ``odps_schema_to_arrow_schema`` but maps ``BLOB`` to ``pa.binary()``
    and recurses into ARRAY/STRUCT/MAP so nested BLOBs are handled too.
    System columns (e.g. ``__operation`` as ``int8``) are appended after
    data columns so the record writer can populate them per-row.
    """
    fields = []
    data_names = set()
    for col in write_schema.columns:
        fields.append(
            pa.field(
                col.name, _odps_type_to_arrow_with_blob(col.type), nullable=col.nullable
            )
        )
        data_names.add(col.name)
    # Append system columns (e.g. __operation → int8) that are NOT already
    # present in data columns — the server may return __operation in both.
    for col in getattr(write_schema, "system_columns", []):
        if col.name not in data_names:
            fields.append(
                pa.field(
                    col.name, _odps_type_to_arrow_with_blob(col.type), nullable=True
                )
            )
    return pa.schema(fields)


class _PendingBlobRef:
    """Placeholder for a blob value pending batch upload.

    Created by ``_intercept_blob_leaf`` for file-like objects and raw
    ``bytes``/``bytearray`` values.  The actual :class:`BlobWriteItem`
    is appended to ``_pending_blob_items``; this placeholder carries
    the index and is replaced with the server-returned reference bytes
    in ``_flush_records`` after ``_batch_upload_blobs`` completes.
    ``str`` references and ``None`` values do not create placeholders —
    they pass through unchanged.
    """

    __slots__ = ("item_index",)

    def __init__(self, item_index: int):
        self.item_index = item_index


class AppendTableRecordWriter:
    """Row-oriented writer wrapping :class:`TableArrowWriter`.

    Converts Record -> Arrow batches.  When the underlying writer has
    ``auto_upload_blobs=True``, BLOB columns are intercepted: file-like
    objects and raw ``bytes``/``bytearray`` are batch-uploaded by the
    record writer itself and replaced with
    references pass through unchanged.  When ``auto_upload_blobs=False``,
    BLOB cells must already contain reference bytes.
    """

    def __init__(
        self,
        arrow_writer: "TableArrowWriter",
        row_count_per_batch: int = 1024,
        blob_batch_file_num: int = 1000,
    ):
        self._arrow_writer = arrow_writer
        self._schema = arrow_writer.write_schema  # WriteSchema with column_id
        self._row_count_per_batch = row_count_per_batch
        self._blob_batch_file_num = blob_batch_file_num
        self._row_buffer = []
        self._row_count = 0
        self._blob_file_count = 0
        self._pending_blob_items = []

        # Build arrow schema from the write schema.
        # We can't reuse tunnel's odps_schema_to_arrow_schema because it does
        # not know about the BLOB type — map BLOB to pa.binary() here.
        if self._schema is not None and pa is not None:
            self._arrow_schema = _write_schema_to_arrow_schema(self._schema)
        else:
            self._arrow_schema = None

        # Detect complex types with nested blobs (v3 gating)
        self._has_complex_blob = (
            self._schema.has_nested_blob_columns() if self._schema else False
        )
        if self._has_complex_blob and not arrow_writer._supports_v3():
            raise StorageClientError(
                "Complex types containing BLOB columns (e.g. array<blob>, "
                "struct<f:blob>) require API v3+. Pass api_version='3' to "
                "MaxStorageClient."
            )

    def write(self, record) -> None:
        """Write a single record.

        Accepts ``odps.types.Record``, ``list``, or ``tuple``.  BLOB
        columns may contain: ``bytes``/``bytearray`` (batch-uploaded
        by the record writer), file-like (streamed in chunks, never
        fully materialized), ``str`` (blob reference, passed through),
        ``None`` (null).

        Auto-flushes when ``row_count >= row_count_per_batch`` or when the
        accumulated blob file count >= ``blob_batch_file_num``.
        """
        values = _extract_record_values(record)

        values, file_count = self._intercept_blobs(values)
        self._row_buffer.append(values)
        self._row_count += 1
        self._blob_file_count += file_count
        if (
            self._row_count >= self._row_count_per_batch
            or self._blob_file_count >= self._blob_batch_file_num
        ):
            self._flush_records()

    def _intercept_blobs(self, values):
        """Intercept blob data in all BLOB-bearing columns (top-level + nested).

        Only active when the underlying writer is a
        :class:`TableArrowBlobUploadWriter` (i.e. ``auto_upload_blobs=True``);
        otherwise values pass through unchanged.

        For primary-key (Delta) tables, a distribution key is computed
        from the current row's PK column values and attached to every
        blob item built from this row — the server rejects blob uploads
        to PK tables that omit it.
        """
        from .writer import TableArrowBlobUploadWriter

        file_count = 0
        if self._schema is None or not isinstance(
            self._arrow_writer, TableArrowBlobUploadWriter
        ):
            return values, 0
        distribution_key = self._compute_distribution_key(values)
        cols = self._schema.columns
        for i, col in enumerate(cols):
            if i >= len(values):
                break
            values[i], n = self._intercept_value(
                values[i], col.type, col.name, col.column_id, distribution_key
            )
            file_count += n
        return values, file_count

    def _intercept_value(
        self, value, data_type, path, column_id, distribution_key=None
    ):
        """Recursively intercept blob values.  Returns (intercepted_value, file_count)."""
        if value is None:
            return None, 0
        if isinstance(data_type, Blob):
            return self._intercept_blob_leaf(path, column_id, value, distribution_key)
        if isinstance(data_type, Array):
            elem_type = data_type.value_type
            result = []
            total = 0
            if value is not None:
                for e in value:
                    v, n = self._intercept_value(
                        e, elem_type, path + ".element", column_id, distribution_key
                    )
                    result.append(v)
                    total += n
            return result, total
        if isinstance(data_type, Struct):
            result = {}
            total = 0
            for fname, ftype in zip(data_type.field_names, data_type.field_types):
                child_path = path + "." + fname
                child_id = None
                if self._schema is not None:
                    child_id = self._schema.get_nested_column_id(child_path)
                v, n = self._intercept_value(
                    value.get(fname) if isinstance(value, dict) else None,
                    ftype,
                    child_path,
                    child_id,
                    distribution_key,
                )
                result[fname] = v
                total += n
            return result, total
        if isinstance(data_type, Map):
            val_type = data_type.value_type
            result = {}
            total = 0
            if value is not None:
                for k, v in value.items():
                    iv, n = self._intercept_value(
                        v, val_type, path + ".value", column_id, distribution_key
                    )
                    result[k] = iv
                    total += n
            return result, total
        return value, 0

    def _intercept_blob_leaf(self, path, column_id, value, distribution_key=None):
        """Prepare a single BLOB value for batch upload by the record writer.

        File-like objects (anything with a ``read`` method) and raw
        ``bytes``/``bytearray`` are wrapped in a :class:`BlobWriteItem`
        and appended to ``_pending_blob_items``.  A
        :class:`_PendingBlobRef` placeholder is returned; it is replaced
        with the server-returned reference bytes in ``_flush_records``
        after ``_batch_upload_blobs`` completes.  File-like objects are
        streamed in chunks by ``BlobWriteItem.write_frame_to``, never
        fully materialized.

        ``str`` references (already-uploaded blob refs) and ``None``
        values pass through unchanged — no upload, no placeholder.

        ``distribution_key`` is forwarded to the blob item so the server
        can route blobs on primary-key (Delta) tables; ``None`` for
        non-PK tables.

        If a ``blob_metadata_callback`` is set on the underlying arrow
        writer, it is called here — exactly once per blob — with the
        *original* value.
        """
        if value is None or isinstance(value, str):
            return value, 0

        # Resolve metadata while the original value is still available.
        callback = getattr(self._arrow_writer, "_blob_metadata_callback", None)
        mime_type = None
        custom_file_name = None
        if callback is not None:
            result = callback(self._blob_file_count, path, value)
            if result is not None:
                mime_type, custom_file_name = result

        item = self._arrow_writer.build_blob_write_item(
            value,
            column_name=path,
            distribution_key=distribution_key,
            mime_type=mime_type,
            custom_file_name=custom_file_name,
            checksum_type=getattr(self._arrow_writer, "_blob_checksum_type", None)
            or BlobWriteItem.ChecksumType.NONE,
        )
        idx = len(self._pending_blob_items)
        self._pending_blob_items.append(item)
        return _PendingBlobRef(idx), 1

    def _compute_distribution_key(self, values):
        """Build a distribution key for the current row on a PK table.

        Mirrors :meth:`TableArrowBlobUploadWriter._generate_distribution_key`:
        serializes the row's primary-key columns to a single-row Arrow IPC
        stream and base64-encodes the result.  Returns ``None`` for
        non-PK tables (no distribution-key columns) or when pyarrow is
        unavailable.
        """
        if self._schema is None or pa is None:
            return None
        pk_cols = [
            (i, col)
            for i, col in enumerate(self._schema.columns)
            if getattr(col, "is_distribution_key", False)
        ]
        if not pk_cols:
            return None
        pk_arrays = []
        pk_names = []
        for i, col in pk_cols:
            if i >= len(values):
                break
            arrow_type = _odps_type_to_arrow_with_blob(col.type)
            pk_arrays.append(pa.array([values[i]], type=arrow_type))
            pk_names.append(col.name)
        if not pk_arrays:
            return None
        pk_table = pa.Table.from_arrays(pk_arrays, names=pk_names)
        sink = io.BytesIO()
        writer = pa.ipc.new_stream(sink, pk_table.schema)
        writer.write_table(pk_table)
        writer.close()
        return base64.b64encode(sink.getvalue()).decode("ascii")

    def _flush_records(self):
        """Upload pending blobs, convert buffered rows to an Arrow batch."""
        if self._row_count == 0 or pa is None or self._arrow_schema is None:
            return

        # Batch-upload all pending blob items (streamed via chunked
        # transfer-encoding).  File-like objects are streamed in chunks,
        # never fully materialized.
        if self._pending_blob_items:
            self._upload_pending_blobs()

        cols = []
        for i, (name, arrow_type) in enumerate(
            zip(self._arrow_schema.names, self._arrow_schema.types)
        ):
            col_values = [row[i] if i < len(row) else None for row in self._row_buffer]
            try:
                cols.append(pa.array(col_values, type=arrow_type))
            except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
                # Fallback: let pyarrow infer the type
                cols.append(pa.array(col_values))
        table = pa.Table.from_arrays(cols, names=self._arrow_schema.names)

        # Call the base TableArrowWriter.write_batch directly — all BLOB
        # cells now contain reference bytes, so the blob-upload override
        # is unnecessary.  This avoids re-uploading references.
        from .writer import TableArrowWriter

        TableArrowWriter.write_batch(self._arrow_writer, table)

        self._row_buffer = []
        self._row_count = 0
        self._blob_file_count = 0
        self._pending_blob_items = []

    def _upload_pending_blobs(self):
        """Batch-upload pending blob items and replace placeholders with refs."""
        items = self._pending_blob_items

        refs = self._arrow_writer._batch_upload_blobs(items)

        # Replace _PendingBlobRef placeholders in the row buffer with refs.
        ref_map = {i: refs[i] for i in range(len(refs))}
        for row in self._row_buffer:
            _replace_pending_refs(row, ref_map)

    def flush(self) -> None:
        """Flush buffered records, then flush the underlying writer."""
        self._flush_records()
        self._arrow_writer.flush()

    def close(self) -> None:
        """Flush remaining records, then close the underlying writer."""
        self.flush()
        self._arrow_writer.close()

    def __enter__(self) -> "AppendTableRecordWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DeltaTableRecordWriter(AppendTableRecordWriter):
    """Record writer for Delta Tables (primary key tables).

    ``write()`` stamps ``__operation='U'`` (UPSERT) and ``delete()`` stamps
    ``__operation='D'`` (DELETE) on each record.  Both operations can be
    freely interleaved on a single writer instance.
    """

    def write(self, record) -> None:
        """Write a record as an UPSERT (``__operation='U'``)."""
        self._write_with_operation(record, OPERATION_UPSERT)

    def delete(self, record) -> None:
        """Write a record as a DELETE (``__operation='D'``)."""
        self._write_with_operation(record, OPERATION_DELETE)

    def _write_with_operation(self, record, operation) -> None:
        """Inject ``__operation`` into the record values, then delegate."""
        values = _extract_record_values(record)

        if self._schema is not None:
            # If __operation is a regular column, set it in-place.
            for i, col in enumerate(self._schema.columns):
                if col.name == OPERATION_COLUMN_NAME:
                    while len(values) <= i:
                        values.append(None)
                    values[i] = operation
                    break
            else:
                # __operation is a system column — append at the end.
                values.append(operation)

        if hasattr(record, "values") or hasattr(record, "_values"):

            class _Rec:
                def __init__(self, vals):
                    self.values = vals

            super().write(_Rec(values))
        else:
            super().write(values)
