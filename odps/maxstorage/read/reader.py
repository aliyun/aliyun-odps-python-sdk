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

"""Arrow reader for :mod:`odps.maxstorage`.

``ArrowReader`` wraps an :class:`~odps.maxstorage.io.arrow_reader.ArrowStreamReader`
over the HTTP response's (optionally decompressed) Arrow IPC stream.  The
iterator protocol yields :class:`pyarrow.RecordBatch`; ``get_as_record_reader``
returns a :class:`~odps.maxstorage.read.record_reader.ArrowRecordReader` that
yields :class:`odps.models.Record` (row-oriented API).
"""
from typing import TYPE_CHECKING, Optional

from ..io.arrow_reader import ArrowStreamReader, AsyncArrowStreamReader
from .record_reader import ArrowRecordReader

if TYPE_CHECKING:
    try:
        import pyarrow as pa
    except ImportError:
        pa = None  # type: ignore

    from ...tunnel.io.stream import CompressOption  # noqa: F401
    from ..models.schema import StorageSchema


class ArrowReader:
    """Arrow reader over a Storage-API read stream.

    Parameters
    ----------
    raw_response : requests.Response
        Streaming HTTP response returned by the stub's ``create_*_read_stream``.
    schema : odps.maxstorage.models.schema.StorageSchema
        The session's table schema (``ReadSchema`` or ``WriteSchema``).  Used by
        ``get_as_record_reader`` to build :class:`odps.models.Record` instances
        whose columns match the table schema.
    compress_option : odps.tunnel.CompressOption, optional
        When set, the response stream is decompressed via
        :func:`odps.tunnel.io.stream.get_decompress_stream`.
    request_id : str, optional
        ODPS request id captured from the stream-create response headers.
    count : int, optional
        Total number of rows available to this reader (the split's row range
        length, or the instance read's requested count).  Defaults to ``0``
        (unknown).  Exposed via the :attr:`count` property.
    async_read : bool, optional
        When ``True``, a background thread pre-reads Arrow batches into a
        bounded queue, overlapping network I/O with batch processing.
        Default ``False``.
    async_queue_size : int, optional
        Maximum number of batches buffered by the async reader (default 2).
        Larger values smooth latency spikes at the cost of memory.

    Example
    -------
    >>> reader = read_session.open_arrow_reader(read_session.splits[0])
    >>> while True:
    ...     batch = reader.read()
    ...     if batch is None:
    ...         break
    ...     print(batch.to_pandas())
    >>> reader.close()
    """

    def __init__(
        self,
        raw_response,
        schema: Optional["StorageSchema"],
        compress_option: "CompressOption" = None,
        request_id: str = "",
        count: int = 0,
        async_read: bool = False,
        async_queue_size: int = 2,
    ):
        if async_read:
            self._stream_reader = AsyncArrowStreamReader(
                raw_response,
                compress_option=compress_option,
                queue_size=async_queue_size,
            )
        else:
            self._stream_reader = ArrowStreamReader(
                raw_response, compress_option=compress_option
            )
        self._schema = schema
        self._compress_option = compress_option
        self._request_id = request_id
        self._count = count
        self._closed = False

    # -- status / metadata ------------------------------------------------

    @property
    def arrow_schema(self) -> "pa.Schema":
        """The :class:`pyarrow.Schema` of the underlying IPC stream."""
        return self._stream_reader.arrow_schema

    @property
    def table_schema(self) -> Optional["StorageSchema"]:
        """The session's ODPS (Storage) schema."""
        return self._schema

    def get_status(self) -> str:
        """Return the reader status.

        The Python port has no async-read lifecycle, so this is always
        ``"OK"`` once the stream is openable.
        """
        return "OK"

    def get_request_id(self) -> str:
        """ODPS request id of the stream-create request."""
        return self._request_id

    @property
    def count(self) -> int:
        """Total number of rows available to this reader (set at open time)."""
        return self._count

    # -- batch iteration ---------------------------------------------------

    def __iter__(self) -> "ArrowReader":
        return self

    def __next__(self) -> "pa.RecordBatch":
        batch = self._stream_reader.read()
        if batch is None:
            raise StopIteration
        return batch

    def read(self) -> Optional["pa.RecordBatch"]:
        """Read the next :class:`pyarrow.RecordBatch`, or ``None`` at EOS."""
        return self._stream_reader.read()

    def read_next_batch(self) -> Optional["pa.RecordBatch"]:
        """Alias for :meth:`read`, matching the tunnel ArrowReader contract."""
        return self.read()

    @property
    def schema(self) -> "StorageSchema":
        """The session's ODPS schema — matches the tunnel ArrowReader contract."""
        return self._schema

    # -- row-oriented view -------------------------------------------------

    def get_as_record_reader(self) -> ArrowRecordReader:
        """Return a record reader yielding :class:`odps.models.Record`.

        See :class:`~odps.maxstorage.read.record_reader.ArrowRecordReader`
        for type-conversion and BLOB-handling details.
        """
        return ArrowRecordReader(self)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream_reader.close()

    def __enter__(self) -> "ArrowReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
