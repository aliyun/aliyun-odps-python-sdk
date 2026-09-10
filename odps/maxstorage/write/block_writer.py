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

"""Block writer for :class:`WriteMode.BATCH_COMPATIBLE`.

The block protocol accepts one Arrow IPC stream per block/attempt pair.
Batches are buffered until :meth:`TableBlockWriter.commit` or
:meth:`TableBlockWriter.close` finalizes the writer.
"""

import json
import logging
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Union

from ...tunnel.io.types import odps_type_to_arrow_type
from ...types import validate_data_type
from ..io.arrow_writer import RawArrowRequestBody, serialize_batch

try:
    import pyarrow as pa
except ImportError:
    pa = None

if TYPE_CHECKING:
    import pyarrow as pa

    from ...tunnel.io.stream import CompressOption
    from ..models.identifier import TableIdentifier
    from ..models.responses import (
        BatchCompatibleDataSchema,
        BatchCompatibleWriteResponse,
    )
    from ..stub import StorageStub

logger = logging.getLogger(__name__)


def _data_schema_to_arrow_schema(data_schema: "BatchCompatibleDataSchema"):
    """Build a ``pa.schema`` from a :class:`BatchCompatibleDataSchema`.

    Columns from both ``data_columns`` and ``partition_columns`` are
    included, matching the server's combined schema.
    """
    if pa is None:
        raise ValueError("pyarrow is required for batch-compatible write")

    columns = []
    if data_schema is not None:
        for col in (data_schema.data_columns or []) + (
            data_schema.partition_columns or []
        ):
            odps_type = validate_data_type(col.type)
            arrow_type = odps_type_to_arrow_type(odps_type)
            columns.append(pa.field(col.name, arrow_type, nullable=col.nullable))
    return pa.schema(columns)


class BlockWriteResult:
    """Result produced by a successful :meth:`TableBlockWriter.commit`.

    The service commit message remains an implementation detail.  Applications
    should retain or transport this object and pass it to
    :meth:`TableWriteSession.commit` with the ``block_results`` parameter.
    """

    __slots__ = (
        "_session_id",
        "_block_number",
        "_attempt_number",
        "_record_count",
        "_commit_message",
    )

    def __init__(
        self,
        session_id: str,
        block_number: int,
        attempt_number: int,
        record_count: int,
        commit_message: str,
    ):
        self._session_id = session_id
        self._block_number = block_number
        self._attempt_number = attempt_number
        self._record_count = record_count
        self._commit_message = commit_message

    @property
    def block_number(self) -> int:
        return self._block_number

    @property
    def attempt_number(self) -> int:
        return self._attempt_number

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def commit_message(self) -> Optional[str]:
        return self._commit_message

    @property
    def session_id(self) -> str:
        return self._session_id

    def __repr__(self):
        return (
            f"BlockWriteResult(block_number={self._block_number}, "
            f"attempt_number={self._attempt_number}, "
            f"record_count={self._record_count})"
        )


class TableBlockWriter:
    """Arrow writer for one block in :class:`WriteMode.BATCH_COMPATIBLE`.

    Each instance corresponds to one ``block_number + attempt_number`` pair.
    Batches are buffered in memory until :meth:`commit` or :meth:`close`
    finalizes the upload as a single Arrow IPC stream.

    When constructed with a ``compress_option``, the uploaded stream uses
    Arrow IPC's built-in codec (``zstd`` / ``lz4``), matching the writer
    returned by :meth:`~odps.maxstorage.TableWriteSession.open_arrow_writer`.

    Prefer :meth:`commit` when the returned :class:`BlockWriteResult` will be
    used directly.  Calling :meth:`abort` discards buffered batches without
    creating a remote writer result; the service releases the unused quota
    reservation after it expires.

    Do not reuse a writer after it has been committed, aborted, or closed.
    Retry the same block with a new ``attempt_number``.
    """

    def __init__(
        self,
        stub: "StorageStub",
        table_id: "TableIdentifier",
        session_id: str,
        block_number: int,
        attempt_number: int,
        route_token: str,
        quota_token: str,
        data_schema: "BatchCompatibleDataSchema",
        enhance_write_check: bool,
        compress_option: Optional["CompressOption"] = None,
    ):
        if pa is None:
            raise ValueError("pyarrow is required for batch-compatible write")

        self._stub = stub
        self._table_id = table_id
        self._session_id = session_id
        self._block_number = block_number
        self._attempt_number = attempt_number
        self._route_token = route_token
        self._quota_token = quota_token
        self._arrow_schema = _data_schema_to_arrow_schema(data_schema)
        self._enhance_write_check = enhance_write_check
        self._compress_option = compress_option

        self._cached_batches: List[bytes] = []
        self._cached_size = 0
        self._record_count = 0
        self._bytes_written = 0
        self._closed = False
        self._result: Optional[BlockWriteResult] = None

    @property
    def block_number(self) -> int:
        return self._block_number

    @property
    def attempt_number(self) -> int:
        return self._attempt_number

    @property
    def record_count(self) -> int:
        return self._record_count

    def create_vector_schema_root(self) -> "pa.Table":
        """Create a :class:`pyarrow.VectorSchemaRoot` matching the session schema."""
        return pa.Table.from_pylist([], schema=self._arrow_schema)

    @property
    def schema(self) -> "pa.Schema":
        """The Arrow schema for this block."""
        return self._arrow_schema

    def write_batch(self, root: Union["pa.Table", "pa.RecordBatch"]) -> None:
        """Buffer one Arrow batch (``RecordBatch`` or ``Table``).

        Empty batches are silently skipped.  The input schema must match
        the session schema.
        """
        if self._closed:
            raise IOError("Block writer is already closed")

        # Accept both RecordBatch and Table; normalize to RecordBatch list.
        if root is None:
            return
        if isinstance(root, pa.Table):
            batches = root.to_batches()
        else:
            batches = [root]

        for batch in batches:
            if batch.num_rows == 0:
                continue
            if not self._arrow_schema.equals(batch.schema):
                raise IOError(
                    "Arrow schema does not match the batch-compatible " "session schema"
                )
            batch_bytes = serialize_batch(batch)
            self._cached_batches.append(batch_bytes)
            self._cached_size += len(batch_bytes)
            self._record_count += batch.num_rows

    def commit(self) -> BlockWriteResult:
        """Finalize the block upload and return the commit result.

        After :meth:`commit`, the writer is closed and cannot be reused.
        Raises :class:`IOError` if the writer was aborted.
        """
        self._upload()
        if self._result is None:
            raise IOError("Block writer was aborted and cannot be committed")
        return self._result

    def abort(self) -> None:
        """Discard locally buffered data without sending it to the service."""
        if self._closed:
            return
        self._cached_batches.clear()
        self._cached_size = 0
        self._closed = True

    def close(self) -> None:
        """Finalize the writer, uploading the block if not already done.

        If :meth:`abort` was called, this is a no-op.  Otherwise it uploads
        the buffered batches.  Prefer :meth:`commit` when the result is
        needed.
        """
        if self._closed:
            return
        self._upload()

    def _upload(self) -> None:
        """Upload buffered batches as one Arrow IPC stream."""
        if self._closed:
            return
        try:
            body = RawArrowRequestBody(
                self._arrow_schema, self._cached_batches, self._compress_option
            ).serialize()
            response = self._stub.write_batch_compatible_block(
                self._table_id,
                self._session_id,
                self._block_number,
                self._attempt_number,
                body,
                self._route_token,
                self._quota_token,
            )
            self._validate_response(response)
            self._bytes_written += self._cached_size
            self._result = BlockWriteResult(
                self._session_id,
                self._block_number,
                self._attempt_number,
                response.record_count,
                response.commit_message,
            )
        except Exception as e:
            if isinstance(e, IOError):
                raise
            raise IOError(
                f"Failed to write block {self._block_number} "
                f"attempt {self._attempt_number}: {e}"
            ) from e
        finally:
            self._cached_batches.clear()
            self._cached_size = 0
            self._closed = True

    def _validate_response(
        self, response: Optional["BatchCompatibleWriteResponse"]
    ) -> None:
        """Check the server response for consistency."""
        if response is None:
            raise IOError("Batch-compatible write returned an empty response")
        if response.record_count != self._record_count:
            raise IOError(
                f"Unexpected record count, expected {self._record_count} "
                f"but got {response.record_count}"
            )
        if response.commit_message is None:
            raise IOError("Batch-compatible write did not return a commit message")

        if not self._enhance_write_check:
            return

        try:
            message = json.loads(response.commit_message)
            if not isinstance(message, dict):
                raise ValueError("Commit message is not a JSON object")
            checks = [
                ("BlockNumber", message.get("BlockNumber"), self._block_number),
                ("AttemptNumber", message.get("AttemptNumber"), self._attempt_number),
            ]
            for field, actual, expected in checks:
                if actual is not None and actual != expected:
                    raise IOError(f"Commit message {field} does not match the writer")
            stats = message.get("WriterStats") or {}
            if (
                stats.get("RecordNum") is not None
                and stats["RecordNum"] != self._record_count
            ):
                raise IOError("Commit message record count does not match the writer")
        except IOError:
            raise
        except Exception as e:
            raise IOError(
                f"Cannot validate the batch-compatible commit message: {e}"
            ) from e

    def bytes_written(self) -> int:
        """Total bytes written (uploaded + currently cached)."""
        return self._bytes_written + self._cached_size

    def __enter__(self) -> "TableBlockWriter":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        if exc_type is not None:
            self.abort()
        else:
            self.close()
        return False
