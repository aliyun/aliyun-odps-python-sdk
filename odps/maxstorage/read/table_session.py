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

"""Table read session for :mod:`odps.maxstorage`.

The constructor creates the session (or reloads one by id), polls
``getTableReadSession`` every second until the status is ``NORMAL`` (or the
``session_ready_timeout`` elapses), then materializes the splits according
to ``split_mode``:

* ``SIZE`` → one :class:`IndexedInputSplit` per ``splits_count`` (0..N-1).
* ``ROW_OFFSET`` → :class:`RowRangeInputSplit` chunks of ``record_count`` in
  steps of ``split_options.split_number``.
* ``PARALLELISM`` / ``BUCKET`` → raise :class:`NotImplementedError`.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

from ...config import options as _options
from ...tunnel.io.types import odps_schema_to_arrow_schema
from ...types import OdpsSchema
from ..errors import StorageClientError
from ..io.compress import CompressionCodec, resolve_compress_option
from ..models.descriptor import (
    SessionDescriptor,
    incremental_options_to_wire,
    read_schema_to_wire,
)
from ..models.enums import SessionKind, SessionStatus, SplitMode
from ..models.requests import CreateTableReadStreamRequest
from ..models.responses import CreateTableReadSessionResponse
from ..models.schema import ReadSchema
from .reader import ArrowReader

try:
    import pyarrow as pa
except ImportError:
    pa = None

if TYPE_CHECKING:
    from ...tunnel.io.stream import CompressOption
    from ..models.enums import DataFormat, SessionStats
    from ..models.identifier import TableIdentifier
    from ..models.requests import CreateTableReadSessionRequest
    from ..models.schema import StorageSchema
    from ..stub import StorageStub


# ---------------------------------------------------------------------------
# Input splits
# ---------------------------------------------------------------------------


@dataclass
class IndexedInputSplit:
    """A split addressed by index — used when ``split_mode == SIZE``.

    Attributes
    ----------
    session_id : str
        The read-session id this split belongs to.
    split_index : int
        Zero-based index in ``[0, splits_count)``.
    """

    session_id: str
    split_index: int


@dataclass
class RowRangeInputSplit:
    """A split addressed by ``[offset, offset+length)`` — used when
    ``split_mode == ROW_OFFSET``.

    Attributes
    ----------
    session_id : str
        The read-session id this split belongs to.
    offset : int
        Starting row offset within the session.
    length : int
        Number of rows in this split.
    """

    session_id: str
    offset: int
    length: int


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class TableReadSession:
    """A table read session.

    Created by ``MaxStorageClient.create_table_read_session``.  After
    construction the session is guaranteed to be in the ``NORMAL`` status and
    its ``splits`` are materialized.

    Parameters
    ----------
    stub : StorageStub
        The RPC layer.
    table_id : TableIdentifier
        Target table.
    request : CreateTableReadSessionRequest
        Creation request (partitions, split options, data format, ...).
    session_id : str, optional
        If given, reload an existing session instead of creating a new one.
    session_ready_timeout : int, optional
        Override ``options.maxstorage.session_ready_timeout`` (default 3600s).

    Example
    -------
    >>> session = client.create_table_read_session("my_table")
    >>> for split in session.splits:
    ...     reader = session.open_arrow_reader(split)
    ...     while True:
    ...         batch = reader.read()
    ...         if batch is None:
    ...             break
    ...         process(batch)
    ...     reader.close()
    """

    _POLL_INTERVAL = 1.0  # seconds

    def __init__(
        self,
        stub: "StorageStub",
        table_id: "TableIdentifier",
        request: "CreateTableReadSessionRequest",
        session_id: Optional[str] = None,
        session_ready_timeout: Optional[int] = None,
        *,
        _response: Optional["CreateTableReadSessionResponse"] = None,
        _rows_per_split: int = 0,
    ):
        self._stub = stub
        self._table_id = table_id
        self._request = request
        self._incremental_read = (
            bool(getattr(request, "incremental_read", False))
            if request is not None
            else False
        )

        if _response is not None:
            # Locally restored from a session descriptor; no RPC is issued.
            response = _response
        elif session_id is not None:
            response = stub.get_table_read_session(table_id, session_id, refresh=False)
        else:
            response = stub.create_table_read_session(table_id, request)
            response = self._poll_until_ready(table_id, response, session_ready_timeout)

        self._response = response
        self._rows_per_split = _rows_per_split or self._resolve_rows_per_split()
        self._splits = self._materialize_splits()

    @classmethod
    def from_descriptor(
        cls, stub: "StorageStub", descriptor: "SessionDescriptor"
    ) -> "TableReadSession":
        """Restore a read session locally from a descriptor, without any RPC.

        The descriptor is authoritative for schema, filters, incremental
        options, and split planning; no remote session is created or
        reloaded, and the supplied ``stub`` is only used for subsequent
        reader calls.
        """
        if descriptor.kind not in (
            SessionKind.BATCH_READ,
            SessionKind.INCREMENTAL_READ,
        ):
            raise ValueError("A read session descriptor is required")

        response = CreateTableReadSessionResponse.from_dict(descriptor.metadata)
        response.route_token = descriptor.route_token
        rows_per_split = descriptor.rows_per_split
        if (
            SplitMode.from_string(response.split_mode) == SplitMode.ROW_OFFSET
            and (response.record_count or 0) > 0
            and rows_per_split <= 0
        ):
            raise ValueError("ROW_OFFSET descriptor requires positive rowsPerSplit")

        session = cls(
            stub,
            descriptor.table_identifier,
            None,
            _response=response,
            _rows_per_split=rows_per_split,
        )
        session._incremental_read = descriptor.kind == SessionKind.INCREMENTAL_READ
        return session

    def to_descriptor(self) -> "SessionDescriptor":
        """Export an immutable, credential-free description of this session.

        The returned descriptor snapshots the current session state; later
        mutations of this session do not affect it.
        """
        resp = self._response
        # The server may omit the incremental options it accepted; preserve
        # the request's so an incremental plan survives the round trip.
        options = resp.incremental_read_options
        if not options and self._request is not None:
            options = getattr(self._request, "incremental_read_options", None)
        metadata = {
            "SessionId": resp.session_id,
            "SessionStatus": resp.session_status,
            "DataSchema": read_schema_to_wire(resp.data_schema),
            "SupportedDataFormat": [
                fmt.to_dict() for fmt in (resp.supported_data_format or [])
            ],
            "SplitMode": resp.split_mode,
            "SplitsCount": resp.splits_count,
            "RecordCount": resp.record_count,
            "SplitBucketId": resp.split_bucket_id,
            "SessionStats": (
                None
                if resp.session_stats is None
                else {
                    "EstimatedSize": resp.session_stats.estimated_size,
                    "EstimatedRowCount": resp.session_stats.estimated_row_count,
                }
            ),
            "EnableLargeString": resp.enable_large_string,
            "LatestVersion": resp.latest_version,
            "ExpirationTime": resp.expiration_time,
            "IncrementalReadOptions": incremental_options_to_wire(options),
        }
        kind = (
            SessionKind.INCREMENTAL_READ
            if self._incremental_read or metadata["IncrementalReadOptions"]
            else SessionKind.BATCH_READ
        )
        return SessionDescriptor.create(
            kind,
            self._table_id,
            metadata,
            route_token=resp.route_token,
            rows_per_split=self._descriptor_rows_per_split(),
        )

    def _descriptor_rows_per_split(self):
        """The row-offset chunk size to record, or ``None`` when unused.

        The request may have defaulted the chunk size, so fall back to the
        materialized splits to keep the plan reproducible.
        """
        if SplitMode.from_string(self._response.split_mode) != SplitMode.ROW_OFFSET:
            return None
        if self._rows_per_split:
            return self._rows_per_split
        if self._splits:
            return self._splits[0].length
        return None

    # -- session lifecycle -------------------------------------------------

    def _poll_until_ready(self, table_id, response, session_ready_timeout):
        """Poll ``getTableReadSession`` until ``NORMAL`` or timeout.

        Returns the final ``NORMAL`` response.  Always passes
        ``session_refresh=False`` during polling.
        """
        timeout = (
            session_ready_timeout
            if session_ready_timeout is not None
            else _options.maxstorage.session_ready_timeout
        )
        if timeout is None:
            timeout = 3600

        deadline = time.monotonic() + timeout
        status = SessionStatus.from_string(response.session_status)

        while status != SessionStatus.NORMAL:
            if status == SessionStatus.INIT:
                if time.monotonic() >= deadline:
                    raise StorageClientError(
                        "Table read session %s is still INIT after %ss"
                        % (response.session_id, timeout)
                    )
                time.sleep(self._POLL_INTERVAL)
                response = self._stub.get_table_read_session(
                    table_id, response.session_id, refresh=False
                )
                status = SessionStatus.from_string(response.session_status)
            else:
                # CRITICAL / EXPIRED / COMMITTING / COMMITTED / UNKNOWN ...
                raise StorageClientError(
                    "Table read session %s is in an unexpected status: %s"
                    % (response.session_id, status)
                )

        return response

    # -- split materialization --------------------------------------------

    def _materialize_splits(self):
        """Build the split list according to ``split_mode``."""
        session_id = self._response.session_id
        split_mode = SplitMode.from_string(self._response.split_mode)

        if split_mode == SplitMode.SIZE:
            count = self._response.splits_count or 0
            return [IndexedInputSplit(session_id, i) for i in range(count)]

        if split_mode == SplitMode.ROW_OFFSET:
            record_count = self._response.record_count or 0
            step = self._split_number
            splits = []
            offset = 0
            while offset < record_count:
                length = min(step, record_count - offset)
                splits.append(RowRangeInputSplit(session_id, offset, length))
                offset += length
            return splits

        # PARALLELISM / BUCKET — not supported.  Use NotImplementedError so callers can catch it distinctly.
        raise NotImplementedError(
            "Split mode %s is not supported by TableReadSession" % split_mode
        )

    def _resolve_rows_per_split(self):
        """Resolve the row-offset chunk size from the creation request.

        Returns ``0`` unless the session uses ``ROW_OFFSET`` splits and the
        request carries an explicit split number.
        """
        if SplitMode.from_string(self._response.split_mode) != SplitMode.ROW_OFFSET:
            return 0
        request = self._request
        options = (
            getattr(request, "split_options", None) if request is not None else None
        )
        return options.split_number if options and options.split_number else 0

    @property
    def _split_number(self):
        """Resolve the row-offset chunk size used to materialize splits.

        Prefers the descriptor/request ``rowsPerSplit``; a request that only
        asked the server to plan splits exposes no chunk size, and a session
        reloaded by id carries no request at all — both degenerate to a
        single split covering the whole record range.
        """
        if self._rows_per_split:
            return self._rows_per_split
        return self._response.record_count or 1

    # -- public properties -------------------------------------------------

    @property
    def id(self) -> Optional[str]:
        """The session id."""
        return self._response.session_id

    @property
    def table_schema(self) -> "StorageSchema":
        """The :class:`ReadSchema` from the response ``DataSchema``."""
        return self._response.data_schema or ReadSchema()

    @property
    def arrow_schema(self) -> "pa.Schema":
        """Arrow schema over all columns (data + partition)."""
        schema = self.table_schema
        # odps_schema_to_arrow_schema iterates simple_columns, so build a
        # schema whose "data" columns are all columns (data + partitions).
        all_columns = list(schema.columns) if hasattr(schema, "columns") else []
        return odps_schema_to_arrow_schema(OdpsSchema(columns=all_columns))

    @property
    def splits(self) -> list:
        """The materialized list of input splits."""
        return self._splits

    @property
    def split_mode(self) -> SplitMode:
        """The :class:`SplitMode` of this session."""
        return SplitMode.from_string(self._response.split_mode)

    @property
    def record_count(self) -> int:
        """Total row count in this session (from ``RecordCount``).

        ``-1`` when the server does not report a record count — an absent
        count is unknown, not zero.
        """
        count = self._response.record_count
        return -1 if count is None else count

    @property
    def session_stats(self) -> Optional["SessionStats"]:
        """Estimated size / row count (:class:`SessionStats` or ``None``)."""
        return self._response.session_stats

    @property
    def expiration_time(self) -> Optional[int]:
        """Session expiration time (from ``ExpirationTime``)."""
        return self._response.expiration_time

    @property
    def latest_version(self) -> Optional[int]:
        """Latest data version (from ``LatestVersion``, used by incremental reads)."""
        return self._response.latest_version

    @property
    def supported_data_formats(self) -> List["DataFormat"]:
        """List of :class:`DataFormat` supported by this session."""
        return self._response.supported_data_format

    @property
    def enable_large_string(self) -> bool:
        """Whether the server uses Arrow large-string columns."""
        return self._response.enable_large_string

    @property
    def extended_arrow_ipc_enabled(self) -> bool:
        """Whether the session negotiated the extended Arrow IPC format."""
        return self._response.extended_arrow_ipc_enabled

    @property
    def latest_txn_id(self) -> Optional[str]:
        """Latest transaction id (from ``LatestTxnId``, used by incremental reads)."""
        return self._response.latest_txn_id

    # -- reader -----------------------------------------------------------

    def open_arrow_reader(
        self,
        split: Union[IndexedInputSplit, RowRangeInputSplit],
        *,
        max_batch_rows: int = 4096,
        skip_row_num: int = 0,
        max_batch_raw_size: int = 0,
        data_format: Optional["DataFormat"] = None,
        data_columns: Optional[List[str]] = None,
        data_columns_unordered: bool = False,
        compress_option: Optional["CompressOption"] = None,
        compress_algo=None,
        compress_level=None,
        async_read: bool = False,
        async_queue_size: int = 2,
    ) -> ArrowReader:
        """Open an :class:`ArrowReader` for ``split``.

        Compression follows the tunnel pattern: pass ``compress_option``
        (a :class:`odps.tunnel.CompressOption`) or the shorthand
        ``compress_algo``/``compress_level``.  Default ``None`` = uncompressed.
        ``compress_option`` takes priority over the shorthand kwargs.

        When compression is requested, an ``ACCEPT-ENCODING`` header is sent
        on the stream-create request so the server compresses the Arrow stream;
        the reader then decompresses the response via
        :func:`odps.tunnel.io.stream.get_decompress_stream`.

        Example
        -------
        >>> read_session = client.create_table_read_session("my_table")
        >>> for split in read_session.splits:
        ...     reader = read_session.open_arrow_reader(split, max_batch_rows=1024)
        ...     while True:
        ...         batch = reader.read()
        ...         if batch is None:
        ...             break
        ...         print(batch.to_pandas())
        ...     reader.close()
        >>>
        >>> # Compressed read via shorthand
        >>> reader = read_session.open_arrow_reader(
        ...     read_session.splits[0], compress_algo="zstd",
        ... )
        >>> reader.close()
        """
        compress_option = resolve_compress_option(
            compress_option, compress_algo, compress_level
        )

        request = CreateTableReadStreamRequest(
            max_batch_rows=max_batch_rows,
            skip_row_num=skip_row_num,
            max_batch_raw_size=max_batch_raw_size,
            data_format=data_format,
            data_columns=list(data_columns) if data_columns else [],
            data_columns_unordered=data_columns_unordered,
        )

        accept_encoding = None
        if compress_option is not None:
            codec = CompressionCodec.from_compress_option(compress_option)
            accept_encoding = codec.accept_encoding

        raw_response = self._stub.create_table_read_stream(
            self._table_id,
            split,
            request,
            route_token=self._response.route_token,
            accept_encoding=accept_encoding,
        )

        request_id = ""
        if hasattr(raw_response, "headers"):
            request_id = raw_response.headers.get("x-odps-request-id", "")

        return ArrowReader(
            raw_response,
            schema=self.table_schema,
            compress_option=compress_option,
            request_id=request_id,
            count=getattr(split, "length", self.record_count),
            async_read=async_read,
            async_queue_size=async_queue_size,
            extended=self.extended_arrow_ipc_enabled,
        )

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Close the session.  Best-effort; the server expires sessions on its own."""
        # No explicit close — this is a no-op for symmetry with
        # InstanceReadSession and resource-manager friendliness.
        return None
