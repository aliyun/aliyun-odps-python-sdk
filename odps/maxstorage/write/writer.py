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

"""Table Arrow writers for :mod:`odps.maxstorage`.

:class:`TableArrowWriter` caches serialized Arrow batches and flushes them as
a single Arrow IPC stream.  :class:`TableArrowBlobUploadWriter` (subclass)
handles tables with BLOB columns by batch-uploading blob data and replacing
cells with references before flushing.
"""

import base64
import io
import logging
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional, Union

try:
    import pyarrow as pa
except ImportError:
    pa = None

from ..errors import StorageClientError
from ..io.arrow_writer import RawArrowRequestBody
from ..io.blob_writer import BlobStreamWriter, stream_blob_batch
from ..io.compress import CompressionCodec, resolve_compress_option
from ..models.enums import WriteMode
from ..models.requests import (
    CloseWriteStreamRequest,
    CreateWriteStreamRequest,
    GetWriteStreamRequest,
)
from ..models.schema import WriteSchema, _contains_blob
from ..options import (
    AUTO_COMMIT_DEFAULT_STREAM_ID,
    AUTO_COMMIT_SESSION_ID,
    BlobWriteItem,
    _normalize_partition_spec,
    _supports_v3,
)
from ..stub import ROUTE_TOKEN_HEADER
from .record_writer import (
    OPERATION_COLUMN_NAME,
    OPERATION_DELETE,
    OPERATION_UPSERT,
    AppendTableRecordWriter,
    DeltaTableRecordWriter,
)

if TYPE_CHECKING:
    from ...tunnel.io.stream import CompressOption
    from ..models.identifier import TableIdentifier
    from ..models.responses import WriteBlobResponse
    from ..stub import StorageStub
    from .session import TableWriteSession

logger = logging.getLogger(__name__)


def _completed_future():
    """Return an already-completed :class:`~concurrent.futures.Future`.

    Used by :meth:`flush_async` when there is nothing to flush so callers
    can uniformly ``.result()`` the return value without None-checks.
    """
    f = Future()
    f.set_result(None)
    return f


def _build_compressed_stream(arrow_schema, record_batches, compress_option):
    """Build a full Arrow IPC stream with built-in compression.

    Uses ``pa.ipc.new_stream`` with ``IpcWriteOptions(compression=...)``.
    No HTTP ``Content-Encoding`` is needed — the codec is embedded in the IPC format.
    """
    codec = CompressionCodec.from_compress_option(compress_option)
    arrow_codec = None
    if codec == CompressionCodec.ZSTD:
        arrow_codec = "zstd"
    elif codec == CompressionCodec.LZ4_FRAME:
        arrow_codec = "lz4"

    sink = io.BytesIO()
    writer = pa.ipc.new_stream(
        sink,
        arrow_schema,
        options=pa.ipc.IpcWriteOptions(compression=arrow_codec),
    )
    for batch in record_batches:
        writer.write_batch(batch)
    writer.close()
    return sink.getvalue()


def _close_file_items(items) -> None:
    """Best-effort close file-like objects in *items*.

    Each element may be a :class:`BlobWriteItem` (with a ``data`` attribute)
    or a bare file-like object.  Objects without a ``close`` attribute
    (e.g. raw ``bytes``) and objects that report themselves as already
    closed are skipped.  Errors raised by ``close()`` are swallowed.
    """
    for item in items:
        data = getattr(item, "data", item)
        if data is None:
            continue
        close = getattr(data, "close", None)
        if close is None:
            continue
        closed = getattr(data, "closed", False)
        if closed:
            continue
        try:
            close()
        except Exception:
            pass


class TableArrowWriter:
    """Arrow batch writer for a table write stream.

    Caches serialized Arrow batches and flushes them as a single Arrow IPC
    stream body via :meth:`stub.write_table`.

    Parameters
    ----------
    stub : StorageStub
        The RPC layer.
    table_id : TableIdentifier
        Target table.
    session_id : str
        Write session ID.
    stream_id : str
        Write stream ID (use different IDs for parallel writers).
    stream_version : int, default 1
        Write stream version.
    write_mode : WriteMode, default WriteMode.BATCH
        Batch or streaming write mode.
    route_token : str, optional
        Server routing token for session affinity.
    compress_option : odps.tunnel.CompressOption, optional
        Arrow IPC compression for write batches.
    api_version : str, default "2"
        Storage API version (``"2"`` or ``"3"``).

    Example
    -------
    >>> import pyarrow as pa
    >>> writer = write_session.open_arrow_writer(stream_id="0")
    >>> batch = pa.RecordBatch.from_arrays(
    ...     [pa.array([1, 2], pa.int64()), pa.array(["a", "b"], pa.string())],
    ...     schema=pa.schema([("id", pa.int64()), ("name", pa.string())]),
    ... )
    >>> writer.write_batch(batch)
    >>> writer.close()
    >>> write_session.commit()
    """

    def __init__(
        self,
        stub: "StorageStub",
        table_id: "TableIdentifier",
        session_id: str,
        stream_id: str,
        stream_version: int = 1,
        *,
        write_mode: WriteMode = WriteMode.BATCH,
        route_token: Optional[str] = None,
        buffer_size: int = 64 * 1024 * 1024,
        auto_flush_enabled: bool = True,
        auto_close_files: bool = False,
        resume: bool = False,
        max_pending_buffers: int = 1,
        exactly_once_mode: bool = False,
        compress_option: Optional["CompressOption"] = None,
        api_version: str = "2",
        session: Optional["TableWriteSession"] = None,
    ):
        if pa is None:
            raise ValueError("pyarrow is required for Arrow write")

        if stream_id is None:
            raise ValueError("stream_id must not be None")

        self._stub = stub
        self._table_id = table_id
        self._session_id = session_id
        self._stream_id = str(stream_id)
        self._stream_version = stream_version
        self._write_mode = write_mode
        self._route_token = route_token
        self._buffer_size = buffer_size
        self._auto_flush_enabled = auto_flush_enabled
        if max_pending_buffers < 1:
            raise ValueError(
                "max_pending_buffers must be >= 1, got %d" % max_pending_buffers
            )
        self._max_pending_buffers = max_pending_buffers
        self._exactly_once_mode = exactly_once_mode
        self._compress_option = compress_option
        self._api_version = str(api_version)
        self._session = session
        self._auto_close_files = auto_close_files

        self._cached_batches = []
        self._cached_record_batches = []
        self._cached_size = 0
        self._cached_row_count = 0
        self._arrow_schema = None
        self._closed = False
        self._last_staging_id = None
        self._last_request_id = None
        self._row_offset = 0
        self._access_token = None
        self._table_id_from_stream = None
        self._schema_version_from_stream = None
        self._write_schema = None

        # EO / streaming fields captured from create-stream response
        self._create_stream(stream_id, stream_version, resume)

        # Async flush machinery
        self._flush_executor = None
        self._flush_semaphore = None
        self._pending_flush = False
        self._pending_futures = deque()
        self._flush_lock = threading.Lock()

    def _supports_v3(self) -> bool:
        return _supports_v3(self._api_version)

    def _create_stream(self, stream_id, stream_version, resume):
        """Call createTableWriteStream or getWriteStream."""
        if resume:
            request = GetWriteStreamRequest(
                session_id=self._session_id,
                stream_id=str(stream_id),
                stream_version=stream_version,
                exactly_once_mode=self._exactly_once_mode,
            )
            resp = self._stub.get_write_stream(
                self._table_id, request, self._route_token, self._write_mode
            )
            if resp.route_token:
                self._route_token = resp.route_token
            if resp.row_offset is not None:
                self._row_offset = resp.row_offset
            if resp.access_token:
                self._access_token = resp.access_token
            self._table_id_from_stream = resp.table_id
            self._schema_version_from_stream = resp.schema_version
            if resp.data_schema:
                self._write_schema = WriteSchema.from_dict(resp.data_schema)
        else:
            request = CreateWriteStreamRequest(
                stream_id=str(stream_id),
                stream_version=stream_version,
                exactly_once_mode=self._exactly_once_mode,
            )
            resp = self._stub.create_table_write_stream(
                self._table_id,
                self._session_id,
                request,
                self._route_token,
                self._write_mode,
            )
            if resp.route_token:
                self._route_token = resp.route_token
            self._table_id_from_stream = resp.table_id
            self._schema_version_from_stream = resp.schema_version
            if self._exactly_once_mode:
                if not resp.access_token:
                    raise StorageClientError(
                        "Exactly-once mode requested but server did not return "
                        "an AccessToken in CreateWriteStreamResponse."
                    )
                self._access_token = resp.access_token
            if resp.data_schema:
                self._write_schema = WriteSchema.from_dict(resp.data_schema)

    @property
    def write_schema(self) -> WriteSchema:
        if self._write_schema is None:
            self._write_schema = self._stub.get_write_schema(
                self._table_id, self._session_id, self._route_token
            )
        return self._write_schema

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def stream_version(self) -> int:
        return self._stream_version

    # ---- Write ----

    def write_batch(self, batch) -> None:
        """Serialize a batch to Arrow IPC bytes and cache it.

        ``batch`` may be a :class:`pa.Table` or :class:`pa.RecordBatch`.
        Auto-flushes at ``buffer_size``.

        BLOB column values are sent inline as-is.  To auto-upload raw
        ``bytes`` and replace them with references, use
        :class:`TableArrowBlobUploadWriter` via ``auto_upload_blobs=True``.
        """
        if self._closed:
            raise StorageClientError("Writer has been closed")

        # Non-blocking: surface any completed async-flush failure before
        # accepting more data, but do NOT block on an in-flight flush —
        # the semaphore in flush_async provides backpressure and blocking
        # here would serialize writes, defeating max_pending_buffers > 1.
        self._surface_failed_futures()

        # Normalize to RecordBatch
        if isinstance(batch, pa.Table):
            batches = batch.to_batches()
        else:
            batches = [batch]

        for batch in batches:
            # Validate and normalize the __operation column (mandatory on
            # PK/delta tables, nulls filled with UPSERT client-side).
            batch = self._validate_operation_column(batch)

            batch_bytes = self._serialize_batch(batch)
            self._cached_batches.append(batch_bytes)
            self._cached_record_batches.append(batch)
            self._cached_size += len(batch_bytes)
            self._cached_row_count += batch.num_rows
            if self._arrow_schema is None:
                self._arrow_schema = batch.schema

            if self._auto_flush_enabled and self._cached_size >= self._buffer_size:
                self.flush_async()

    def _requires_operation_column(self):
        """True if the write schema declares ``__operation`` (PK/delta table).

        Mirrors the detection in :meth:`get_as_record_writer`: a table
        requires the ``__operation`` column iff it appears in either
        ``columns`` or ``system_columns`` of the :class:`WriteSchema`.
        """
        schema = self._write_schema
        if schema is None and self._session is not None:
            schema = self._stub.get_write_schema(
                self._table_id, self._session_id, self._route_token
            )
            self._write_schema = schema
        if schema is None:
            return False
        for col in list(schema.columns) + list(schema.system_columns):
            if col.name == OPERATION_COLUMN_NAME:
                return True
        return False

    def _validate_operation_column(self, batch):
        """Validate and normalize the ``__operation`` column.

        On PK/delta tables (where :meth:`_requires_operation_column` is True)
        ``__operation`` is mandatory, must be an int8 vector, and null entries
        are filled with UPSERT before serialization.  This method enforces:

        * If the column is required but absent from the batch → raise.
        * If present, fill nulls with ``OPERATION_UPSERT`` and validate that
          non-null values are ``'U'`` (UPSERT) or ``'D'`` (DELETE).

        Returns the (possibly rebuilt) ``RecordBatch``.
        """
        required = self._requires_operation_column()
        try:
            op_idx = batch.schema.get_field_index(OPERATION_COLUMN_NAME)
        except Exception:
            op_idx = -1
        if op_idx < 0:
            if required:
                raise StorageClientError(
                    "__operation column is required for PK/delta tables but "
                    "is absent from the batch."
                )
            return batch
        col = batch.column(op_idx)
        if col.type != pa.int8():
            raise StorageClientError(
                f"__operation column must be Arrow int8 (TinyInt), got " f"{col.type}."
            )
        # Build a replacement list, filling nulls with UPSERT and validating.
        new_vals = []
        for i in range(len(col)):
            val = col[i].as_py()
            if val is None:
                new_vals.append(OPERATION_UPSERT)
                continue
            if isinstance(val, int):
                if val != OPERATION_UPSERT and val != OPERATION_DELETE:
                    raise StorageClientError(
                        f"Invalid operation value: {val}. Must be 'U' (upsert) "
                        f"or 'D' (delete)."
                    )
            elif isinstance(val, str):
                if val not in ("U", "D"):
                    raise StorageClientError(
                        f"Invalid operation value: '{val}'. Must be 'U' or 'D'."
                    )
            new_vals.append(val)
        new_col = pa.array(new_vals, type=col.type)
        new_cols = list(batch.columns)
        new_cols[op_idx] = new_col
        return pa.RecordBatch.from_arrays(new_cols, schema=batch.schema)

    def _serialize_batch(self, batch):
        """Serialize a RecordBatch to raw Arrow IPC batch-message bytes.

        Returns only the batch record message (no schema message, no EOS).
        Compression is applied later by :class:`RawArrowRequestBody` on the
        full assembled stream, not per-batch.
        """
        return batch.serialize()

    # ---- Flush ----

    def _surface_failed_futures(self):
        """Non-blocking: re-raise exceptions from any completed async flush.

        Removes completed futures from the pending deque and calls
        ``.result()`` on each so an async exception surfaces in the
        caller's context.  In-flight (unfinished) futures are left in the
        deque — the semaphore provides backpressure, not this method.
        """
        with self._flush_lock:
            pending = list(self._pending_futures)
            still_running = deque()
            for fut in pending:
                if fut.done():
                    self._pending_futures.remove(fut)
                else:
                    still_running.append(fut)
            if not self._pending_futures:
                self._pending_flush = False
        # result() outside the lock to avoid blocking other lock holders
        for fut in pending:
            if fut not in still_running and fut.exception() is not None:
                fut.result()

    def _await_pending_flush(self):
        """Block until all in-flight async flushes resolve; re-raise failures.

        Awaits every pending Future in submission order so that
        ``flush()``, ``close()``, and ``set_row_offset()`` honor the
        documented guarantee of awaiting *all* pending uploads, not just
        the most recent one.  After this returns, no async flush is in
        flight.
        """
        with self._flush_lock:
            futures = list(self._pending_futures)
            self._pending_futures.clear()
            self._pending_flush = False
        first_error = None
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                if first_error is None:
                    first_error = e
        if first_error is not None:
            raise first_error

    def _snapshot_buffer_state(self):
        """Snapshot the current buffer state without clearing.

        Must be called under ``_flush_lock`` so concurrent ``write_batch``
        appends cannot interleave with the snapshot.
        """
        return {
            "batches": self._cached_batches,
            "record_batches": self._cached_record_batches,
            "size": self._cached_size,
            "row_count": self._cached_row_count,
            "arrow_schema": self._arrow_schema,
        }

    def _clear_buffer_state(self):
        """Clear the cached buffer state.

        Must be called under ``_flush_lock``.  Used after a successful
        flush so a failed RPC leaves the data in cache for retry.
        """
        self._cached_batches = []
        self._cached_record_batches = []
        self._cached_size = 0
        self._cached_row_count = 0

    def _capture_buffer_state(self):
        """Snapshot the current buffer state and clear it atomically.

        Must be called under ``_flush_lock`` so concurrent ``write_batch``
        appends cannot interleave with the snapshot/clear.  Used by
        ``flush_async`` where the swap-before-flush is needed for write
        concurrency — async failures are surfaced via the Future.
        """
        state = self._snapshot_buffer_state()
        self._clear_buffer_state()
        return state

    def _restore_buffer_state(self, state):
        """Restore captured *state* to the live cache after an async failure.

        Must be called under ``_flush_lock``.  Appends the captured rows
        *after* the current live cache so original submission order is
        preserved across retries — FIFO failures that restore in reverse
        would reorder append data and break exactly-once semantics.
        ``arrow_schema`` is restored only when the live schema is still
        unset.
        """
        self._cached_batches = self._cached_batches + state["batches"]
        self._cached_record_batches = (
            self._cached_record_batches + state["record_batches"]
        )
        self._cached_size += state["size"]
        self._cached_row_count += state["row_count"]
        if self._arrow_schema is None:
            self._arrow_schema = state["arrow_schema"]

    def _do_flush(self, state):
        """Send one Arrow IPC stream body built from a captured *state* dict.

        Shared by sync :meth:`flush` and async :meth:`flush_async`.  Reads
        the *immutable* buffer payload (batches, row count, schema) from
        *state* so the worker operates on a consistent snapshot.  The
        *mutable* transport fields (route token, row offset, access token)
        are read live from ``self`` under ``_flush_lock`` at execution time
        — not captured at submission — so that with the single-thread
        executor each FIFO task uses the values returned by the previous
        task's response, preserving exactly-once sequencing.
        Response fields (route token, staging id, row offset) are written
        back to ``self`` under the same lock.
        """
        arrow_schema = state["arrow_schema"]
        record_batches = state["record_batches"]
        batches = state["batches"]
        compress_option = self._compress_option

        if compress_option is not None:
            arrow_body = _build_compressed_stream(
                arrow_schema, record_batches, compress_option
            )
        else:
            body = RawArrowRequestBody(arrow_schema, batches, None)
            arrow_body = body.serialize()

        # Read mutable transport fields live under the lock so the async
        # worker picks up values updated by the previous task's response.
        with self._flush_lock:
            route_token = self._route_token
            if self._exactly_once_mode:
                row_offset = self._row_offset
                access_token = self._access_token
            else:
                row_offset = -1
                access_token = None

        resp = self._stub.write_table(
            self._table_id,
            self._session_id,
            self._stream_id,
            self._stream_version,
            state["row_count"],
            arrow_body,
            route_token,
            streaming_table_id=self._table_id_from_stream,
            streaming_schema_version=self._schema_version_from_stream,
            row_offset=row_offset,
            access_token=access_token,
            write_mode=self._write_mode,
            compress_option=compress_option,
        )

        # Write back mutable response fields under the lock.
        with self._flush_lock:
            rt = resp.headers.get(ROUTE_TOKEN_HEADER)
            if rt:
                self._route_token = rt
                if self._session is not None:
                    self._session._route_token = rt

            write_resp = self._stub.parse_write_stream_response(resp)
            if write_resp.warning_message:
                logger.warning(write_resp.warning_message)
            if write_resp.staging_id:
                self._last_staging_id = write_resp.staging_id
            if (
                self._exactly_once_mode
                and write_resp.exactly_once_row_offset is not None
            ):
                self._row_offset = write_resp.exactly_once_row_offset

            self._last_request_id = resp.headers.get("x-odps-request-id")

    def flush(self) -> None:
        """Send all cached batches as one Arrow IPC stream body.

        If an async flush is in flight, awaits and surfaces its exception
        first, then sync-flushes the remaining buffer.
        """
        self._await_pending_flush()
        if not self._cached_batches:
            return
        if self._closed:
            raise StorageClientError("Writer has been closed")

        with self._flush_lock:
            state = self._snapshot_buffer_state()
        self._do_flush(state)
        # Clear the cache only after a successful RPC so a network failure
        # leaves the data in cache for the caller to retry.
        with self._flush_lock:
            self._clear_buffer_state()

    def flush_async(self) -> Future:
        """Async flush using a ThreadPoolExecutor + Semaphore for backpressure.

        Captures all buffer state (including ``_cached_record_batches`` needed
        for compressed writes), submits the flush to a background worker, and
        retains the Future in a pending deque so failures surface from the
        next ``write_batch`` / ``flush`` / ``close``.  The semaphore provides
        real backpressure: when ``max_pending_buffers`` flushes are in flight
        the next ``flush_async`` blocks until one completes.

        Every submitted Future is tracked in ``_pending_futures`` (not a
        single scalar) so that ``flush()``, ``close()``, and
        ``set_row_offset()`` await *all* pending uploads, and a failure in
        one buffer is not erased by a success in another.

        On async failure the captured rows are restored to the live cache
        (appended after any live data to preserve submission order) so the
        caller can retry.  Returns the Future (a completed Future when there
        is nothing to flush) so callers can always ``.result()`` it.
        """
        # Non-blocking: surface any completed async-flush failure before
        # submitting a new one.  Do NOT block — the semaphore below provides
        # the real backpressure for max_pending_buffers > 1.
        self._surface_failed_futures()

        if self._closed:
            raise StorageClientError("Writer has been closed")

        if not self._cached_batches:
            # Nothing to flush — return a completed Future so callers can
            # uniformly ``.result()`` without None-checks.
            return _completed_future()

        if self._flush_executor is None:
            self._flush_executor = ThreadPoolExecutor(max_workers=1)
            self._flush_semaphore = threading.Semaphore(self._max_pending_buffers)

        self._flush_semaphore.acquire()
        with self._flush_lock:
            state = self._capture_buffer_state()

        def _worker():
            try:
                self._do_flush(state)
                # Remove this future from the pending deque on success.
                # On failure, leave it so _await_pending_flush() /
                # _surface_failed_futures() can retrieve and re-raise the
                # exception, and restore the captured rows to the live
                # cache for retry.
                with self._flush_lock:
                    if fut in self._pending_futures:
                        self._pending_futures.remove(fut)
                    if not self._pending_futures:
                        self._pending_flush = False
            except Exception:
                # Restore the captured rows so the caller can retry — the
                # async flush already detached them from the live cache.
                with self._flush_lock:
                    self._restore_buffer_state(state)
                raise
            finally:
                if self._flush_semaphore:
                    self._flush_semaphore.release()

        self._pending_flush = True
        fut = self._flush_executor.submit(_worker)
        with self._flush_lock:
            self._pending_futures.append(fut)
        return fut

    # ---- Close ----

    def close(self) -> None:
        """Flush remaining data, then closeWriteStream.

        Skips closeWriteStream when (a) streaming with session_id == "default"
        OR (b) stream_id == "default"; otherwise closes the stream.

        The async-flush executor is always shut down (even if flush or
        closeWriteStream raises) so its worker thread is joined.
        """
        if self._closed:
            return
        flush_error = None
        try:
            try:
                self.flush()
            except Exception as e:
                flush_error = e
            legacy_default = (
                self._write_mode.is_streaming()
                and self._session_id == AUTO_COMMIT_SESSION_ID
            )
            skip_close = (
                legacy_default or self._stream_id == AUTO_COMMIT_DEFAULT_STREAM_ID
            )

            if not skip_close:
                try:
                    request = CloseWriteStreamRequest(
                        session_id=self._session_id,
                        stream_id=self._stream_id,
                        stream_version=self._stream_version,
                    )
                    self._stub.close_write_stream(
                        self._table_id,
                        request,
                        self._route_token,
                        self._write_mode,
                    )
                except Exception:
                    if flush_error is None:
                        raise
                    logger.debug("closeWriteStream also failed", exc_info=True)
        finally:
            self._closed = True
            # Shut down the async-flush executor so its worker thread is
            # joined, even when flush or closeWriteStream raised above.
            if self._flush_executor is not None:
                self._flush_executor.shutdown(wait=True)
        if flush_error is not None:
            raise flush_error

    # ---- Blob operations ----

    def build_blob_write_item(
        self,
        data: Union[bytes, io.IOBase],
        *,
        column_name: Optional[str] = None,
        partition_spec: Optional[str] = None,
        distribution_key: Optional[str] = None,
        mime_type: Optional[str] = None,
        custom_file_name: Optional[str] = None,
        checksum_type=BlobWriteItem.ChecksumType.NONE,
    ) -> BlobWriteItem:
        """Build a :class:`BlobWriteItem` from writer-resolved state.

        ``column_name`` is resolved to the server-assigned column ID via
        :meth:`WriteSchema.find_all_blob_column_ids` (same dot-path keys).
        When omitted, the sole top-level pure-BLOB column is auto-selected
        (raises if the schema has multiple BLOB columns or any nested BLOB
        column).  ``partition_spec`` accepts the same types as
        :class:`~odps.types.PartitionSpec` (str, dict, or PartitionSpec).
        ``api_version`` is stamped from the writer so v3-only fields
        (``custom_file_name``) are gated automatically.

        Example
        -------
        >>> writer = write_session.open_arrow_writer(stream_id="0")
        >>> items = [
        ...     writer.build_blob_write_item(
        ...         b"payload", column_name="blob", mime_type="image/png",
        ...     )
        ... ]
        """
        column_name, column_id = self.write_schema.resolve_blob_column_name(column_name)
        partition_values = _normalize_partition_spec(partition_spec) or []

        return BlobWriteItem(
            data=data,
            column_id=column_id,
            partition_values=partition_values,
            distribution_key=distribution_key,
            mime_type=mime_type,
            custom_file_name=custom_file_name,
            checksum_type=checksum_type,
            api_version=self._api_version,
        )

    def write_blob_stream(
        self,
        *,
        column_name: Optional[str] = None,
        partition_spec: Optional[str] = None,
        compress_option: Optional["CompressOption"] = None,
        compress_algo=None,
        compress_level=None,
    ) -> BlobStreamWriter:
        """Streaming single blob upload with MD5 verification.

        Returns a :class:`BlobStreamWriter` that streams data to the server
        via chunked transfer-encoding — the full blob is never materialized.

        ``column_name`` is the dot-path name (a key of
        :meth:`WriteSchema.find_all_blob_column_ids`) resolved to the
        server-assigned column ID internally.  When omitted, the sole
        top-level pure-BLOB column is auto-selected (raises if the schema
        has multiple BLOB columns or any nested BLOB column).

        ``partition_spec`` accepts the same types as
        :class:`~odps.types.PartitionSpec` (str, dict, or PartitionSpec
        instance), e.g. ``"pt=20230101"`` or ``{"pt": "20230101"}``.

        Example
        -------
        >>> writer = write_session.open_arrow_writer(stream_id="0")
        >>> sw = writer.write_blob_stream(column_name="blob")
        >>> sw.write(b"payload data")
        >>> resp = sw.finish()
        >>> print(resp.blob_reference)
        """
        co = resolve_compress_option(compress_option, compress_algo, compress_level)
        codec = CompressionCodec.from_compress_option(co)

        # Resolve column_name -> server-assigned column ID.
        column_name, column_id = self.write_schema.resolve_blob_column_name(column_name)
        partition_values = ",".join(_normalize_partition_spec(partition_spec) or [])

        params = {
            "SessionId": self._session_id,
            "StreamId": self._stream_id,
            "StreamVersion": str(self._stream_version),
            "PartitionValues": partition_values,
            "ColumnIndex": str(column_id),
        }
        content_encoding = codec.content_encoding

        def upload(data_generator):
            return self._stub.table_write_blob(
                self._table_id,
                params,
                data_generator,
                self._route_token,
                content_encoding=content_encoding,
            )

        return BlobStreamWriter(
            upload, compress_option=co, api_version=self._api_version
        )

    def _batch_blob_params(self):
        """Common request params for batch blob uploads."""
        return {
            "SessionId": self._session_id,
            "StreamId": self._stream_id,
            "StreamVersion": str(self._stream_version),
            "Mode": "Batch",
        }

    def write_blob_batch(
        self,
        items,
        *,
        compress_option: Optional["CompressOption"] = None,
        compress_algo=None,
        compress_level=None,
    ) -> "WriteBlobResponse":
        """Batch blob upload.  Returns :class:`WriteBlobResponse`.

        Streams frames to the server via chunked transfer-encoding — the
        full batch body is never materialized.

        Example
        -------
        >>> writer = write_session.open_arrow_writer(stream_id="0")
        >>> items = [
        ...     writer.build_blob_write_item(b"a", column_name="blob"),
        ...     writer.build_blob_write_item(b"b", column_name="blob"),
        ... ]
        >>> resp = writer.write_blob_batch(items)
        >>> print(len(resp.blob_references))
        """
        co = resolve_compress_option(compress_option, compress_algo, compress_level)
        codec = CompressionCodec.from_compress_option(co)
        params = self._batch_blob_params()

        def upload(data_generator):
            return self._stub.table_batch_write_blob(
                self._table_id,
                params,
                data_generator,
                self._route_token,
                content_encoding=codec.content_encoding,
            )

        resp = stream_blob_batch(
            items, upload, compress_option=co, api_version=self._api_version
        )
        if self._auto_close_files:
            _close_file_items(items)
        return resp

    # ---- Record writer ----

    def get_as_record_writer(
        self,
        row_count_per_batch: int = 1024,
        blob_batch_file_num: int = 1000,
    ):
        """Return a record-oriented writer wrapping this writer.

        Returns :class:`DeltaTableRecordWriter` if an ``__operation`` column
        is found in either ``system_columns`` or ``columns``; otherwise
        :class:`AppendTableRecordWriter`.

        On a plain :class:`TableArrowWriter`, BLOB cells must already
        contain reference bytes.  On a
        :class:`TableArrowBlobUploadWriter`, raw ``bytes`` / file-like
        values are auto-uploaded and replaced with references.

        :param row_count_per_batch: rows per Arrow batch flush.
        :param blob_batch_file_num: max blobs per batch upload.
        """
        schema = self._write_schema
        has_operation = False
        if schema:
            for col in list(schema.columns) + list(schema.system_columns):
                if col.name == OPERATION_COLUMN_NAME:
                    has_operation = True
                    break

        if has_operation:
            return DeltaTableRecordWriter(
                self,
                row_count_per_batch=row_count_per_batch,
                blob_batch_file_num=blob_batch_file_num,
            )
        return AppendTableRecordWriter(
            self,
            row_count_per_batch=row_count_per_batch,
            blob_batch_file_num=blob_batch_file_num,
        )

    # ---- Status / introspection ----

    def get_last_staging_id(self) -> Optional[str]:
        return self._last_staging_id

    def get_last_request_id(self) -> Optional[str]:
        return self._last_request_id

    def has_pending_flush(self) -> bool:
        return self._pending_flush

    def get_cached_size(self) -> int:
        return self._cached_size

    def get_row_offset(self) -> int:
        """In EO mode, fetch the latest server-side RowOffset; else return local."""
        if self._exactly_once_mode:
            request = GetWriteStreamRequest(
                session_id=self._session_id,
                stream_id=self._stream_id,
                stream_version=self._stream_version,
                exactly_once_mode=True,
            )
            resp = self._stub.get_write_stream(
                self._table_id, request, self._route_token, self._write_mode
            )
            if resp.row_offset is not None:
                self._row_offset = resp.row_offset
            if resp.access_token:
                self._access_token = resp.access_token
        return self._row_offset

    def set_row_offset(self, new_row_offset: int) -> None:
        """Flush cached data with current offset, then update (EO mode only).

        Awaits any in-flight async flush first so the detached flush cannot
        race with the offset change and write rows at the wrong position.
        """
        self._await_pending_flush()
        if self._cached_batches:
            self.flush()
        self._row_offset = new_row_offset

    def is_exactly_once_mode(self) -> bool:
        return self._exactly_once_mode

    def __enter__(self) -> "TableArrowWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _NestedBlobEntry:
    """A nested BLOB column path + its server-assigned column ID."""

    __slots__ = ("path", "column_id")

    def __init__(self, path, column_id):
        self.path = path
        self.column_id = column_id


class TableArrowBlobUploadWriter(TableArrowWriter):
    """Arrow writer that auto-uploads BLOB cells and replaces with references.

    Subclass of :class:`TableArrowWriter`.  ``write_batch`` is overridden
    to batch-upload raw ``bytes`` / file-like BLOB cells via
    ``table_batch_write_blob`` and replace them with reference bytes
    before forwarding to ``super().write_batch``.

    Use this writer when you want inline BLOB content uploaded
    automatically — create it via ``auto_upload_blobs=True`` on
    :meth:`TableWriteSession.open_arrow_writer`.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to :class:`TableArrowWriter`.
    blob_checksum_type : BlobWriteItem.ChecksumType, optional
        Checksum algorithm for blob uploads.
    blob_mime_type : str, optional
        Session-level default MIME type for blobs.
    blob_custom_file_name : str, optional
        Session-level default custom file name (API v3 only).
    blob_metadata_callback : callable, optional
        ``callback(row_index, column_name, blob_data) -> (mime_type,
        custom_file_name) | None`` invoked once per inline BLOB cell.
    """

    def __init__(
        self,
        *args,
        blob_checksum_type=None,
        blob_mime_type: Optional[str] = None,
        blob_custom_file_name: Optional[str] = None,
        blob_metadata_callback: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._blob_checksum_type = blob_checksum_type
        self._blob_mime_type = blob_mime_type
        self._blob_custom_file_name = blob_custom_file_name
        self._blob_metadata_callback = blob_metadata_callback

        # Resolve blob column IDs from WriteSchema
        self._blob_column_ids = {}  # {dot_path: column_id}
        self._top_level_blob = []  # [(col_index, column_id)]
        self._nested_blob = []  # [NestedBlobEntry]
        self._pk_indices = []

        ws = self._write_schema
        if ws is None and self._session is not None:
            # Fetch write schema on demand
            ws = self._stub.get_write_schema(
                self._table_id, self._session_id, self._route_token
            )
            self._write_schema = ws

        if ws is not None:
            all_blob_ids = ws.find_all_blob_column_ids()
            self._blob_column_ids = all_blob_ids
            # Resolve top-level vs nested
            for col in ws.columns:
                if _contains_blob(col.type):
                    cid = all_blob_ids.get(col.name)
                    if cid is not None:
                        # Top-level blob column — col_index resolved at
                        # write_batch time from the batch schema.
                        self._top_level_blob.append((None, cid))
                    # Check for nested blobs
                    if "." not in col.name:
                        for path, pid in all_blob_ids.items():
                            if path.startswith(col.name + ".") and pid is not None:
                                self._nested_blob.append(_NestedBlobEntry(path, pid))

            # Also resolve nested blob column IDs that appear only in
            # all_blob_ids (not discoverable via top-level column iteration).
            for path, pid in all_blob_ids.items():
                if "." in path and not any(e.path == path for e in self._nested_blob):
                    self._nested_blob.append(_NestedBlobEntry(path, pid))

            for i, col in enumerate(ws.columns):
                if getattr(col, "is_distribution_key", False):
                    self._pk_indices.append(i)

    def write_batch(self, batch) -> None:
        """Override: batch-upload BLOB cells, replace with refs, then super().

        A ``pa.Table`` may contain multiple record-batch chunks; each chunk is
        processed independently.
        """
        if isinstance(batch, pa.Table):
            batches = list(batch.to_batches())
            schema = batch.schema
        else:
            batches = [batch]
            schema = batch.schema

        if not batches:
            super().write_batch(batch)
            return

        for batch in batches:
            if batch is None or len(batch) == 0:
                continue
            batch = self._intercept_blobs(batch, schema)
            super().write_batch(batch)

    def _intercept_blobs(self, batch, schema):
        """Batch-upload BLOB cells and replace with references."""
        for col in self._write_schema.columns if self._write_schema else []:
            if not _contains_blob(col.type):
                continue
            cid = self._blob_column_ids.get(col.name)
            if cid is None:
                continue
            try:
                idx = schema.get_field_index(col.name)
            except Exception:
                continue
            if idx < 0:
                continue
            batch = self._process_top_level_blob(batch, idx, cid, col.name)

        for entry in self._nested_blob:
            batch = self._process_nested_blob(batch, entry)

        return batch

    def _resolve_blob_metadata(self, column_name, row_index, blob_data):
        """Resolve per-blob ``mime_type`` / ``custom_file_name``.

        If a ``blob_metadata_callback`` is set, invoke it with the cell
        value.  If the callback returns ``None`` or is unset, fall back to
        the session-level ``blob_mime_type`` / ``blob_custom_file_name``.
        """
        if self._blob_metadata_callback is not None:
            result = self._blob_metadata_callback(row_index, column_name, blob_data)
            if result is not None:
                return result
        return self._blob_mime_type, self._blob_custom_file_name

    def _batch_upload_blobs(self, items, *, compress_option=None):
        """Batch-upload blob items via chunked transfer-encoding.

        Returns a list of reference bytes (one per item).  Raises
        :class:`StorageClientError` if the server returns a mismatched
        reference count.
        """
        params = self._batch_blob_params()

        def upload(data_generator):
            return self._stub.table_batch_write_blob(
                self._table_id,
                params,
                data_generator,
                self._route_token,
            )

        resp = stream_blob_batch(
            items,
            upload,
            compress_option=compress_option,
            api_version=self._api_version,
        )
        refs = list(resp.blob_references or [])
        if len(refs) != len(items):
            raise StorageClientError(
                "Blob batch upload returned %d references for %d items; "
                "expected exact match." % (len(refs), len(items))
            )
        if self._auto_close_files:
            _close_file_items(items)
        return refs

    def _process_top_level_blob(self, batch, col_idx, column_id, col_name):
        """Iterate non-null BLOB cells, batch-upload, replace with refs.

        Every non-null cell is treated as raw blob data and uploaded.
        Returns a new ``RecordBatch`` with blob cells replaced by their
        blob-reference bytes.  Returns the original batch unchanged if no
        cells were uploaded.
        """
        col = batch.column(col_idx)
        items = []
        indices = []
        for i in range(len(col)):
            val = col[i].as_py()
            if val is None:
                continue
            dk = self._generate_distribution_key(batch, i)
            mime_type, custom_file_name = self._resolve_blob_metadata(col_name, i, val)
            item = self.build_blob_write_item(
                val,
                column_name=col_name,
                mime_type=mime_type,
                custom_file_name=custom_file_name,
                distribution_key=dk,
                checksum_type=self._blob_checksum_type
                or BlobWriteItem.ChecksumType.NONE,
            )
            items.append(item)
            indices.append(i)
        refs = self._batch_upload_blobs(items)

        new_vals = []
        for i in range(len(col)):
            val = col[i].as_py()
            new_vals.append(val)
        for j, idx in enumerate(indices):
            new_vals[idx] = refs[j]

        new_col = pa.array(new_vals, type=col.type)
        new_cols = list(batch.columns)
        new_cols[col_idx] = new_col
        return pa.RecordBatch.from_arrays(new_cols, schema=batch.schema)

    def _process_nested_blob(self, batch, entry):
        """Resolve a nested BLOB vector, batch-upload, and rebuild the column.

        Every non-null cell is treated as raw blob data and uploaded.
        Returns a new ``RecordBatch`` with nested BLOB cells replaced by
        their reference bytes.  Returns the original batch unchanged if no
        cells were uploaded.
        """
        parts = entry.path.split(".")
        try:
            col_idx = batch.schema.get_field_index(parts[0])
        except Exception:
            return batch
        if col_idx < 0:
            return batch

        col = batch.column(col_idx)
        items = []

        def collect(arr, path_parts, depth):
            """Walk nested arrays collecting non-null blob bytes."""
            if depth >= len(path_parts):
                for i in range(len(arr)):
                    val = arr[i].as_py()
                    if val is None:
                        continue
                    mime_type, custom_file_name = self._resolve_blob_metadata(
                        entry.path, i, val
                    )
                    items.append(
                        self.build_blob_write_item(
                            val,
                            column_name=entry.path,
                            mime_type=mime_type,
                            custom_file_name=custom_file_name,
                            checksum_type=self._blob_checksum_type
                            or BlobWriteItem.ChecksumType.NONE,
                        )
                    )
                return
            if pa.types.is_list(arr.type):
                try:
                    collect(arr.values, path_parts, depth + 1)
                except Exception:
                    pass
            elif pa.types.is_map(arr.type):
                # MapVector → value child (arr.items flattens keys/nulls;
                # path_parts[depth] is always "value" — ignore, like list
                # ignores "element").
                try:
                    collect(arr.items, path_parts, depth + 1)
                except Exception:
                    pass
            elif pa.types.is_struct(arr.type):
                try:
                    collect(arr.field(path_parts[depth]), path_parts, depth + 1)
                except Exception:
                    pass

        collect(col, parts, 1)
        if not items:
            return batch
        refs = self._batch_upload_blobs(items)

        # Rebuild the column via Python conversion, replacing blob cells
        # with their reference bytes at the correct depth.  ``refs`` are
        # already decoded ``bytes`` by the response.
        py_col = col.to_pylist()
        ref_iter = iter(refs)

        def replace_in_value(val, depth):
            if val is None:
                return None
            if depth >= len(parts):
                return next(ref_iter)
            if isinstance(val, list):
                # MAP<string, ...> cells round-trip via to_pylist() as a
                # list of (key, value) tuples; ARRAY<...> cells are a list
                # of plain values.  Distinguish by the tuple check so map
                # values are rebuilt without corrupting array elements.
                if (
                    depth < len(parts)
                    and parts[depth] == "value"
                    and val
                    and isinstance(val[0], tuple)
                ):
                    return [(k, replace_in_value(v, depth + 1)) for k, v in val]
                return [replace_in_value(e, depth + 1) for e in val]
            if isinstance(val, dict):
                fname = parts[depth]
                return {
                    k: replace_in_value(v, depth + 1) if k == fname else v
                    for k, v in val.items()
                }
            return val

        new_py_col = [replace_in_value(v, 1) for v in py_col]
        new_col = pa.array(new_py_col, type=col.type)
        new_cols = list(batch.columns)
        new_cols[col_idx] = new_col
        return pa.RecordBatch.from_arrays(new_cols, schema=batch.schema)

    def _resolve_blob_vector(self, batch, path):
        """Walk the Arrow vector tree for a dot-path (List/Map/StructVector)."""
        parts = path.split(".")
        try:
            idx = batch.schema.get_field_index(parts[0])
        except Exception:
            return None
        if idx < 0:
            return None
        vec = batch.column(idx)
        for i in range(1, len(parts)):
            if pa.types.is_list(vec.type):
                # ListVector -> dataVector (ignore parts[i], always .element)
                try:
                    vec = vec.values
                except Exception:
                    return None
            elif pa.types.is_map(vec.type):
                # MapVector -> value child (ignore parts[i], always .value)
                try:
                    vec = vec.items
                except Exception:
                    return None
            elif pa.types.is_struct(vec.type):
                try:
                    vec = vec.field(parts[i])
                except Exception:
                    return None
            else:
                return None
        return vec

    def _generate_distribution_key(self, batch, row_idx):
        """Generate a distribution key by slicing PK columns to Arrow IPC + base64."""
        if not self._pk_indices:
            return None
        # Slice one row
        sliced = batch.slice(row_idx, 1)
        # Select PK columns
        pk_arrays = [
            sliced.column(i) for i in self._pk_indices if i < sliced.num_columns
        ]
        pk_names = [
            sliced.schema.field(i).name
            for i in self._pk_indices
            if i < sliced.num_columns
        ]
        if not pk_arrays:
            return None
        pk_table = pa.Table.from_arrays(pk_arrays, names=pk_names)
        # Serialize to Arrow IPC stream
        sink = io.BytesIO()
        writer = pa.ipc.new_stream(sink, pk_table.schema)
        writer.write_table(pk_table)
        writer.close()
        data = sink.getvalue()
        return base64.b64encode(data).decode("ascii")
