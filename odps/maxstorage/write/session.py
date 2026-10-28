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

"""Table write session for :mod:`odps.maxstorage`.

The constructor handles four creation paths:
- ``STREAMING``/``STREAMING_REALTIME`` (with ``session_id`` ``None`` or
  ``"default"``) -> session id ``"default"``, no API call.
- Existing ``session_id`` (not ``"default"``) -> ``getTableWriteSession``
  for route_token.
- New -> ``createTableWriteSession``, extract id + route_token.
- ``BATCH_COMPATIBLE`` -> ``createBatchCompatibleSession`` /
  ``getBatchCompatibleSession``, extract id + route_token + data schema +
  max block number.
"""

import logging
from typing import TYPE_CHECKING, Collection, List, Optional

from ..errors import StorageClientError
from ..io.compress import resolve_compress_option
from ..models.descriptor import (
    SessionDescriptor,
    write_arrow_units,
    write_schema_to_wire,
)
from ..models.enums import SessionKind, WriteMode
from ..models.requests import (
    BatchCompatibleCreateSessionRequest,
    BatchCompatibleDynamicPartitionOptions,
    CreateTableWriteSessionRequest,
    CreateWriteStreamRequest,
)
from ..models.responses import BatchCompatibleSessionResponse
from ..options import AUTO_COMMIT_SESSION_ID, BatchCompatibleOptions, _supports_v3
from .block_writer import BlockWriteResult, TableBlockWriter
from .writer import TableArrowBlobUploadWriter, TableArrowWriter

if TYPE_CHECKING:
    from ...tunnel.io.stream import CompressOption
    from ..models.enums import DataFormat
    from ..models.identifier import TableIdentifier
    from ..stub import StorageStub

logger = logging.getLogger(__name__)


class TableWriteSession:
    """A write session for a MaxCompute table.

    Created by :meth:`MaxStorageClient.create_table_write_session`.

    Parameters
    ----------
    stub : StorageStub
        The RPC layer.
    table_id : TableIdentifier
        Target table.
    session_id : str, optional
        Reload an existing session instead of creating a new one.
    partition_spec : str, optional
        Partition to write, e.g. ``"pt=20230101"``.
    write_mode : WriteMode, default WriteMode.BATCH
        Batch or streaming write mode.
    api_version : str, default "2"
        Storage API version (``"2"`` or ``"3"``).

    Example
    -------
    >>> import pyarrow as pa
    >>> session = client.create_table_write_session("my_table")
    >>> writer = session.open_arrow_writer(stream_id="0")
    >>> batch = pa.RecordBatch.from_arrays(
    ...     [pa.array([1], pa.int64()), pa.array(["x"], pa.string())],
    ...     schema=pa.schema([("id", pa.int64()), ("name", pa.string())]),
    ... )
    >>> writer.write_batch(batch)
    >>> writer.close()
    >>> session.commit()
    """

    def __init__(
        self,
        stub: "StorageStub",
        table_id: "TableIdentifier",
        *,
        session_id: Optional[str] = None,
        partition_spec: Optional[str] = None,
        overwrite: bool = False,
        write_mode: WriteMode = WriteMode.BATCH,
        quota_name: Optional[str] = None,
        enable_schema_evolution: bool = False,
        required_data_format: Optional["DataFormat"] = None,
        batch_compatible_options: Optional[BatchCompatibleOptions] = None,
        api_version: str = "2",
        _batch_compatible_response: Optional["BatchCompatibleSessionResponse"] = None,
    ):
        self._stub = stub
        self._table_id = table_id
        self._write_mode = write_mode
        self._api_version = str(api_version)
        self._route_token = None
        self._committed = False
        self._aborted = False
        self._closed = False

        # Batch-compatible session state
        self._max_block_number = 0
        self._batch_compatible_schema = None
        self._enhance_write_check = False
        self._last_session_status = None

        # Descriptor-restored sessions may only open block writers; commit
        # and abort stay with the coordinator that created the session.
        self._descriptor_only = False
        self._commit_started = False

        # Build the partition spec string
        spec = ""
        if partition_spec:
            if isinstance(partition_spec, str):
                spec = partition_spec
            else:
                spec = ",".join(f"{k}={v}" for k, v in partition_spec)

        flags = {}
        if overwrite:
            flags["overwrite"] = "true"
        if quota_name:
            flags["quota_name"] = quota_name
        if enable_schema_evolution:
            flags["enable_schema_evolution"] = "true"

        if write_mode == WriteMode.BATCH_COMPATIBLE:
            if _batch_compatible_response is not None:
                # Locally restored from a session descriptor; no RPC is issued.
                self._bind_batch_compatible(_batch_compatible_response)
            else:
                self._init_batch_compatible(
                    stub,
                    table_id,
                    session_id,
                    spec,
                    overwrite,
                    batch_compatible_options or BatchCompatibleOptions.create_default(),
                )
        elif write_mode.is_streaming() and (
            session_id is None or session_id == AUTO_COMMIT_SESSION_ID
        ):
            # Streaming auto-commit path: session id is "default", no API call.
            self._session_id = AUTO_COMMIT_SESSION_ID
        elif session_id is not None and session_id != AUTO_COMMIT_SESSION_ID:
            # Existing session — get route_token
            self._session_id = session_id
            resp = stub.get_table_write_session(table_id, session_id, write_mode)
            if resp.route_token:
                self._route_token = resp.route_token
        else:
            # New session
            request = CreateTableWriteSessionRequest(
                partial_partition_spec=spec,
                flags=flags,
                required_data_format=required_data_format,
            )
            resp = stub.create_table_write_session(table_id, request, write_mode)
            self._session_id = resp.session_id
            if resp.route_token:
                self._route_token = resp.route_token

    def _init_batch_compatible(
        self, stub, table_id, session_id, spec, overwrite, options
    ):
        """Create or reload a batch-compatible session."""
        if session_id is not None and session_id != AUTO_COMMIT_SESSION_ID:
            resp = stub.get_batch_compatible_session(
                table_id, session_id, route_token=None
            )
        else:
            request = BatchCompatibleCreateSessionRequest(
                partition_spec=spec,
                overwrite=overwrite,
                dynamic_partition_options=BatchCompatibleDynamicPartitionOptions(
                    dynamic_partition_limit=options.dynamic_partition_limit,
                ),
                max_field_size=options.max_field_size,
                enhance_write_check=options.enhance_write_check,
            )
            resp = stub.create_batch_compatible_session(table_id, request)
            if resp is None or not resp.session_id:
                raise StorageClientError(
                    "Create batch-compatible session returned no session identifier"
                )
        self._bind_batch_compatible(resp)

    def _bind_batch_compatible(self, resp):
        """Adopt a batch-compatible session response as this session's state."""
        if resp is None:
            raise StorageClientError(
                "Get batch-compatible session returned an empty response"
            )
        if resp.data_schema is None:
            raise StorageClientError(
                "Batch-compatible session response did not contain DataSchema"
            )

        self._session_id = resp.session_id
        if resp.route_token:
            self._route_token = resp.route_token
        self._max_block_number = resp.max_block_number
        self._batch_compatible_schema = resp.data_schema
        self._enhance_write_check = resp.enhance_write_check
        self._last_session_status = resp.session_status

    @property
    def id(self) -> Optional[str]:
        return self._session_id

    @property
    def descriptor_only(self) -> bool:
        """Whether this session was restored from a descriptor.

        Descriptor-restored sessions support block data operations only;
        global commit and abort stay with the coordinator session.
        """
        return self._descriptor_only

    def to_descriptor(self) -> "SessionDescriptor":
        """Export an immutable, credential-free description of this session.

        Only ``BATCH_COMPATIBLE`` sessions support descriptors; other write
        modes are rejected until their distinct recovery contracts exist.

        A descriptor exported after a commit was attempted cannot reopen
        writers on the restoring side, even when the last observed session
        status is still ``NORMAL``.
        """
        if self._write_mode != WriteMode.BATCH_COMPATIBLE:
            raise NotImplementedError(
                "Session descriptors require WriteMode.BATCH_COMPATIBLE"
            )
        metadata = {
            "SessionId": self._session_id,
            "SessionStatus": self._descriptor_status(),
            "DataSchema": write_schema_to_wire(self._batch_compatible_schema),
            "MaxBlockNumber": self._max_block_number,
            "EnhanceWriteCheck": self._enhance_write_check,
            "CommitStarted": self._commit_started,
        }
        return SessionDescriptor.create(
            SessionKind.BATCH_WRITE,
            self._table_id,
            metadata,
            route_token=self._route_token,
            arrow=write_arrow_units(),
        )

    @classmethod
    def from_descriptor(
        cls, stub: "StorageStub", descriptor: "SessionDescriptor"
    ) -> "TableWriteSession":
        """Restore a batch-compatible write session locally, without any RPC.

        The restored session is *descriptor-only*: it can open block writers
        (which reserve quota per attempt) but cannot commit or abort the
        global session — those operations stay with the coordinator that
        created it.  Closing the restored session never aborts the global
        session.
        """
        if descriptor.kind != SessionKind.BATCH_WRITE:
            raise ValueError("A batch-write session descriptor is required")
        response = BatchCompatibleSessionResponse.from_dict(descriptor.metadata)
        if not response.session_id:
            raise StorageClientError(
                "Batch-compatible session response did not contain SessionId"
            )
        if response.data_schema is None:
            raise StorageClientError(
                "Batch-compatible session response did not contain DataSchema"
            )
        response.route_token = descriptor.route_token

        session = cls(
            stub,
            descriptor.table_identifier,
            write_mode=WriteMode.BATCH_COMPATIBLE,
            _batch_compatible_response=response,
        )
        session._descriptor_only = True
        session._commit_started = bool(descriptor.metadata.get("CommitStarted"))
        return session

    def _descriptor_status(self):
        """The session status to record in an exported descriptor."""
        if self._committed:
            return "COMMITTED"
        if self._aborted:
            return "ABORTED"
        if self._closed:
            return "UNKNOWN"
        return self._last_session_status or "UNKNOWN"

    def _reject_descriptor_only(self):
        """Reject global session mutations on a descriptor-restored session."""
        if self._descriptor_only:
            raise StorageClientError(
                "Descriptor-restored workers cannot commit or abort a session; "
                "use the coordinator"
            )

    @property
    def write_mode(self) -> WriteMode:
        return self._write_mode

    @property
    def route_token(self) -> Optional[str]:
        return self._route_token

    @property
    def max_block_number(self) -> Optional[int]:
        """Exclusive block-number limit for batch-compatible sessions.

        Returns ``None`` for non-batch-compatible modes or when the service
        omitted it.  Valid block numbers are ``[0, max_block_number)``.
        """
        return self._max_block_number if self._max_block_number > 0 else None

    def _supports_v3(self) -> bool:
        return _supports_v3(self._api_version)

    def get_min_uncommitted_staging_id(self) -> int:
        """Streaming only.  Raises ``NotImplementedError`` on API v2."""
        if not self._supports_v3():
            raise NotImplementedError(
                "get_min_uncommitted_staging_id requires API version 3 or later"
            )
        resp = self._stub.get_table_write_session(
            self._table_id, self._session_id, self._write_mode
        )
        if resp.route_token:
            self._route_token = resp.route_token
        return resp.min_uncommitted_staging_id

    def open_arrow_writer(
        self,
        stream_id: str,
        stream_version: int = 1,
        *,
        buffer_size: int = 64 * 1024 * 1024,
        auto_upload_blobs: bool = False,
        blob_checksum_type=None,
        blob_mime_type: Optional[str] = None,
        blob_custom_file_name: Optional[str] = None,
        blob_metadata_callback: Optional[callable] = None,
        auto_flush_enabled: bool = True,
        auto_close_files: bool = False,
        resume: bool = False,
        max_pending_buffers: int = 1,
        exactly_once_mode: bool = False,
        compress_option: Optional["CompressOption"] = None,
        compress_algo=None,
        compress_level=None,
    ) -> "TableArrowWriter":
        """Open a :class:`TableArrowWriter` for this session.

        If ``resume=True``, calls ``getWriteStream``; otherwise calls
        ``createTableWriteStream``.  Returns
        :class:`TableArrowBlobUploadWriter` if ``auto_upload_blobs``,
        else :class:`TableArrowWriter`.

        Both writer classes support :meth:`~TableArrowWriter.get_as_record_writer`
        and the full blob-upload helper API (``build_blob_write_item``,
        ``write_blob_stream``, ``write_blob_batch``).  The difference is
        whether BLOB cells in ``write_batch`` / ``get_as_record_writer``
        are auto-uploaded:

        * ``auto_upload_blobs=False`` (default): BLOB columns must already
          contain reference bytes — upload them yourself via
          ``write_blob_batch`` first and place the returned references.
        * ``auto_upload_blobs=True``: returns a
          :class:`TableArrowBlobUploadWriter` that batch-uploads raw
          ``bytes``/file-like BLOB cells and replaces them with references.

        :param stream_id: write stream identifier (string).
        :param stream_version: write stream version (default 1).
        :keyword buffer_size: Arrow IPC buffer size in bytes (default 64 MiB).
        :keyword auto_upload_blobs: when ``True``, return a
            :class:`TableArrowBlobUploadWriter` that auto-uploads BLOB cells
            inline and replaces them with references.
        :keyword blob_checksum_type: checksum algorithm for blob uploads.
        :keyword blob_mime_type: session-level default MIME type, used as
            fallback when ``blob_metadata_callback`` is unset or returns
            ``None``.
        :keyword blob_custom_file_name: session-level default custom file
            name (API v3 only), same fallback semantics as ``blob_mime_type``.
        :keyword blob_metadata_callback: optional callable
            ``callback(row_index, column_name, blob_data) -> (mime_type,
            custom_file_name) | None`` invoked once per inline BLOB cell.
        :keyword auto_flush_enabled: whether to auto-flush when the buffer
            is full (default ``True``).
        :keyword auto_close_files: when ``True``, the writer closes file-like
            objects passed to it (e.g. via ``write_batch`` BLOB cells or
            ``write_blob_batch`` items) after they are consumed; objects
            without a ``close`` attribute or already closed are skipped.
        :keyword resume: when ``True``, resume an existing write stream
            instead of creating a new one.
        :keyword max_pending_buffers: maximum number of pending async flush
            buffers for back-pressure (default 1).
        :keyword exactly_once_mode: enable exactly-once write semantics.
        :keyword compress_option: :class:`~odps.tunnel.CompressOption` for
            Arrow IPC compression.
        :keyword compress_algo: shorthand for ``compress_option`` —
            ``"zstd"`` / ``"lz4"``.
        :keyword compress_level: compression level.

        Example
        -------
        >>> import pyarrow as pa
        >>> session = client.create_table_write_session("my_table")
        >>> # Plain Arrow write
        >>> writer = session.open_arrow_writer(stream_id="0")
        >>> batch = pa.RecordBatch.from_arrays(
        ...     [pa.array([1, 2], pa.int64()), pa.array(["a", "b"], pa.string())],
        ...     schema=pa.schema([("id", pa.int64()), ("name", pa.string())]),
        ... )
        >>> writer.write_batch(batch)
        >>> writer.close()
        >>> session.commit()
        >>>
        >>> # Auto-upload writer: inline BLOB cells are auto-uploaded
        >>> writer = session.open_arrow_writer(
        ...     stream_id="0", auto_upload_blobs=True,
        ... )
        >>> batch = pa.RecordBatch.from_arrays(
        ...     [pa.array([0], pa.int64()), pa.array([b"payload"], pa.binary())],
        ...     schema=pa.schema([("id", pa.int64()), ("blob", pa.binary())]),
        ... )
        >>> writer.write_batch(batch)
        >>> writer.close()
        >>> session.commit()
        """
        if self._write_mode == WriteMode.BATCH_COMPATIBLE:
            raise NotImplementedError(
                "BATCH_COMPATIBLE uses block writers; call "
                "open_block_writer(block_number, attempt_number)"
            )
        if stream_id is None:
            raise ValueError("stream_id must not be None")
        stream_id = str(stream_id)
        co = resolve_compress_option(compress_option, compress_algo, compress_level)

        if auto_upload_blobs:
            return TableArrowBlobUploadWriter(
                self._stub,
                self._table_id,
                self._session_id,
                stream_id,
                stream_version,
                write_mode=self._write_mode,
                route_token=self._route_token,
                auto_close_files=auto_close_files,
                buffer_size=buffer_size,
                blob_checksum_type=blob_checksum_type,
                blob_mime_type=blob_mime_type,
                blob_custom_file_name=blob_custom_file_name,
                blob_metadata_callback=blob_metadata_callback,
                auto_flush_enabled=auto_flush_enabled,
                resume=resume,
                max_pending_buffers=max_pending_buffers,
                exactly_once_mode=exactly_once_mode,
                compress_option=co,
                api_version=self._api_version,
                session=self,
            )
        else:
            return TableArrowWriter(
                self._stub,
                self._table_id,
                self._session_id,
                stream_id,
                stream_version,
                write_mode=self._write_mode,
                route_token=self._route_token,
                auto_close_files=auto_close_files,
                buffer_size=buffer_size,
                auto_flush_enabled=auto_flush_enabled,
                resume=resume,
                max_pending_buffers=max_pending_buffers,
                exactly_once_mode=exactly_once_mode,
                compress_option=co,
                api_version=self._api_version,
                session=self,
            )

    def open_block_writer(
        self,
        block_number: int,
        attempt_number: int,
        *,
        compress_option: Optional["CompressOption"] = None,
        compress_algo=None,
        compress_level=None,
    ) -> TableBlockWriter:
        """Create a :class:`TableBlockWriter` for ``BATCH_COMPATIBLE`` mode.

        Each call reserves a one-time upload quota for the given
        ``block_number + attempt_number`` pair.  Retry the same block with
        a new ``attempt_number``; do not reuse a committed or aborted
        writer.

        Compression follows the other write interfaces: pass
        ``compress_option`` (a :class:`odps.tunnel.CompressOption`) or the
        shorthand ``compress_algo``/``compress_level``.  Default ``None`` =
        uncompressed.  When set, the block body is serialized as an Arrow
        IPC stream with the built-in codec (``"zstd"`` / ``"lz4"``),
        matching :meth:`open_arrow_writer` — no HTTP ``Content-Encoding``.

        :param block_number: zero-based block number.
        :param attempt_number: zero-based attempt number for this block.
        :keyword compress_option: :class:`~odps.tunnel.CompressOption` for
            Arrow IPC compression.
        :keyword compress_algo: shorthand for ``compress_option`` —
            ``"zstd"`` / ``"lz4"``.
        :keyword compress_level: compression level.
        :raises NotImplementedError: if this session is not batch-compatible.
        :raises StorageClientError: if the service cannot reserve quota.

        Descriptor-restored sessions reject new writers once a commit has
        been attempted on the session or the last observed session status
        is no longer ``NORMAL``.
        """
        if self._committed or self._aborted or self._closed:
            raise StorageClientError("Session is already closed")
        if self._write_mode != WriteMode.BATCH_COMPATIBLE:
            raise NotImplementedError(
                "open_block_writer requires WriteMode.BATCH_COMPATIBLE"
            )
        if self._descriptor_only and (
            self._commit_started
            or (self._last_session_status or "").upper() != "NORMAL"
        ):
            raise StorageClientError(
                "Descriptor cannot create writers after commit started or "
                "outside the NORMAL session state"
            )
        if block_number < 0:
            raise ValueError("block_number must not be negative")
        if attempt_number < 0:
            raise ValueError("attempt_number must not be negative")
        if self._max_block_number > 0 and block_number >= self._max_block_number:
            raise ValueError(
                f"block_number must be less than the session "
                f"max_block_number {self._max_block_number}"
            )
        if self._batch_compatible_schema is None:
            raise StorageClientError(
                "Batch-compatible session response did not contain DataSchema"
            )

        # Validate before the quota reservation so bad algorithms never
        # reach the service.
        co = resolve_compress_option(compress_option, compress_algo, compress_level)

        request = CreateWriteStreamRequest(
            stream_id=f"block-{block_number}-attempt-{attempt_number}",
            stream_version=1,
        )
        reservation = self._stub.create_table_write_stream(
            self._table_id,
            self._session_id,
            request,
            self._route_token,
            WriteMode.BATCH_COMPATIBLE,
        )
        if not reservation or not reservation.quota_token:
            raise StorageClientError(
                "Batch-compatible quota reservation returned no quota token"
            )
        if not reservation.route_token:
            raise StorageClientError(
                "Batch-compatible quota reservation returned no route token"
            )
        return TableBlockWriter(
            self._stub,
            self._table_id,
            self._session_id,
            block_number,
            attempt_number,
            reservation.route_token,
            reservation.quota_token,
            self._batch_compatible_schema,
            self._enhance_write_check,
            compress_option=co,
        )

    def commit(
        self,
        stream_ids: Optional[List[str]] = None,
        stream_versions: Optional[List[int]] = None,
        block_results: Optional[Collection[BlockWriteResult]] = None,
        wait_flying_writers_timeout_seconds: Optional[int] = None,
    ) -> None:
        """Commit the session to finalize all uploaded data.

        For ``BATCH_COMPATIBLE`` mode, pass ``block_results`` (a collection
        of :class:`BlockWriteResult` from successful block writers).  The
        ``stream_ids`` / ``stream_versions`` parameters are not used in
        batch-compatible mode.

        ``wait_flying_writers_timeout_seconds`` (batch-compatible mode
        only) bounds how long the server waits for in-flight writers
        before finalizing the commit; valid range is 1..600 seconds.
        ``None`` keeps the server default.
        """
        if self._committed or self._aborted or self._closed:
            raise StorageClientError("Session is already closed")
        self._reject_descriptor_only()

        if (
            wait_flying_writers_timeout_seconds is not None
            and self._write_mode != WriteMode.BATCH_COMPATIBLE
        ):
            raise ValueError(
                "wait_flying_writers_timeout_seconds only applies to "
                f"batch-compatible mode, got write mode: {self._write_mode}"
            )

        if self._write_mode == WriteMode.BATCH_COMPATIBLE:
            self._commit_batch_compatible(
                block_results, wait_flying_writers_timeout_seconds
            )
            return

        if (
            self._write_mode.is_streaming()
            and self._session_id == AUTO_COMMIT_SESSION_ID
        ):
            self._committed = True
            return

        self._stub.commit_table_write_session(
            self._table_id,
            self._session_id,
            stream_ids,
            stream_versions,
            self._write_mode,
            route_token=self._route_token,
        )
        self._committed = True

    def abort(self) -> None:
        """Abort the session to discard all uploaded data."""
        if self._aborted:
            return
        self._reject_descriptor_only()
        if (
            self._write_mode.is_streaming()
            and self._session_id == AUTO_COMMIT_SESSION_ID
        ):
            self._aborted = True
            return
        self._stub.abort_table_write_session(
            self._table_id,
            self._session_id,
            self._write_mode,
            route_token=self._route_token,
        )
        self._aborted = True

    def _commit_batch_compatible(
        self, block_results, wait_flying_writers_timeout_seconds=None
    ):
        """Validate and commit block results for batch-compatible mode."""
        if block_results is None:
            raise ValueError("block_results cannot be None; use an empty collection")

        commit_messages = []
        committed_blocks = set()
        for result in block_results:
            if result is None:
                raise ValueError("block_results cannot contain None values")
            if result.session_id != self._session_id:
                raise ValueError(
                    f"block {result.block_number} belongs to a different "
                    f"write session"
                )
            if result.block_number in committed_blocks:
                raise ValueError(
                    f"block_results contain duplicate blockNumber "
                    f"{result.block_number}"
                )
            if result.commit_message is None:
                raise ValueError(
                    f"block {result.block_number} does not contain a " f"commit message"
                )
            committed_blocks.add(result.block_number)
            commit_messages.append(result.commit_message)

        # Once a commit has been attempted its outcome may be unknown even
        # if the call fails; record that before issuing the request.
        self._commit_started = True

        response = self._stub.commit_batch_compatible_session(
            self._table_id,
            self._session_id,
            self._route_token,
            commit_messages,
            wait_flying_writers_timeout_seconds=wait_flying_writers_timeout_seconds,
        )
        if response is None:
            raise StorageClientError(
                "Batch-compatible commit returned an empty response"
            )
        if response.route_token:
            self._route_token = response.route_token
        status = response.session_status
        if status is None or status.upper() != "COMMITTED":
            raise StorageClientError(
                f"Batch-compatible commit returned session status {status} "
                f"for session {self._session_id}"
            )
        self._committed = True

    def close(self) -> None:
        """Auto-aborts if not committed.

        Once a batch-compatible commit has been attempted, its outcome may
        be unknown even if the call failed; closing then performs no remote
        abort, and the coordinator must reconcile the session state
        separately.  Descriptor-restored workers only close their local
        handle: they never abort the global session.

        Either way the handle is finalized: no writer or commit is accepted
        afterwards, and an exported descriptor records the session as
        ``ABORTED`` (abort confirmed) or ``UNKNOWN`` (outcome not observed).
        """
        if self._descriptor_only or self._commit_started:
            self._closed = True
            return
        if not self._committed and not self._aborted:
            try:
                self.abort()
            except Exception:
                logger.debug("Failed to abort session during close", exc_info=True)
        # The handle is finished either way; a failed abort leaves the remote
        # state unknown, so later calls must not reopen writers on it.
        self._closed = True
