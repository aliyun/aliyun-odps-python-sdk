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

"""Blob manager for :mod:`odps.maxstorage`.

``BlobManager`` provides standalone blob read/write operations outside the
table writer context.  Blob references are raw ``bytes``/``str`` obtained from
previous upload operations — no ``Blob`` wrapper class.

The read path supports two access modes:
- **Iterable**: ``read_blobs(...)`` returns a :class:`BlobDataIterator` that
  yields :class:`BlobRecord` instances (materializes each blob).
- **Streaming**: ``read_blobs(..., stream=True)`` returns a
  :class:`BlobStreamReader` for file-like incremental ``read(size)`` per blob.

The write path supports:
- **Single stream upload**: ``write_blob_stream(...)`` → :class:`BlobStreamWriter`
  with MD5 verification.
- **Batch upload**: ``write_blob_batch(...)`` → :class:`WriteBlobResponse`.
"""

import io
import logging
from typing import TYPE_CHECKING, List, Optional, Union

from ..tunnel.io.stream import CompressOption, get_decompress_stream
from ..types import PartitionSpec
from .errors import StorageClientError
from .io.blob_reader import BlobDataIterator, BlobRecord, BlobStreamReader
from .io.blob_writer import BlobStreamWriter, stream_blob_batch
from .io.compress import CompressionCodec, resolve_compress_option
from .io.crc import CrcStrippedInputStream
from .options import BlobWriteItem, _normalize_partition_spec

if TYPE_CHECKING:
    from .base import MaxStorageClient
    from .models.identifier import TableIdentifier
    from .models.responses import WriteBlobResponse

logger = logging.getLogger(__name__)

__all__ = [
    "BlobManager",
    "BlobRecord",
    "BlobDataIterator",
    "BlobStreamReader",
    "BlobStreamWriter",
]


class BlobManager:
    """Standalone blob read/write manager.

    Constructed from a :class:`MaxStorageClient`; reuses its
    :class:`StorageStub` for all HTTP operations.

    Parameters
    ----------
    client : MaxStorageClient
        The parent client.
    table_id : TableIdentifier, optional
        Table identifier — required for write operations
        (``write_blob_stream``, ``write_blob_batch``).

    Example
    -------
    >>> blob_manager = client.open_blob_manager("my_table")
    >>> # Read blobs by reference
    >>> for record in blob_manager.read_blobs([ref1, ref2]):
    ...     print(len(record.data))
    >>> # Stream a single blob
    >>> reader = blob_manager.read_blobs([ref], stream=True)
    >>> while True:
    ...     chunk = reader.read(4096)
    ...     if not chunk:
    ...         break
    ...     process(chunk)
    """

    def __init__(
        self, client: "MaxStorageClient", table_id: Optional["TableIdentifier"] = None
    ):
        self._client = client
        self._api_version = client.api_version
        self._table_id = table_id

    # ---- Read ----

    def _accept_encoding(self, co: Optional[CompressOption]) -> Optional[str]:
        """Return the ``ACCEPT-ENCODING`` header value for a download request.

        Defaults to ``None`` (no ``ACCEPT-ENCODING`` header, uncompressed)
        so the server returns the response uncompressed.  When the caller
        passes an explicit ``compress_option`` its codec is used —
        ``NO_COMPRESSION`` also returns ``None``, while ``LZ4_FRAME`` /
        ``ZSTD`` return the matching encoding token so the server compresses
        the response and the CRC-stripped stream is decompressed client-side.
        """
        if co is not None:
            codec = CompressionCodec.from_compress_option(co)
            return codec.accept_encoding
        return None

    def _wrap_download_stream(self, resp):
        """Wrap a blob-download response into a frame-ready stream.

        Wire order is **CRC-strip first, then decompress**: the server emits
        ``[data block][CRC32C]`` chunks and compresses the whole
        CRC-interleaved byte stream, so the CRC block boundaries (4096+4)
        exist only in the compressed domain.  We strip CRC from the raw
        response, then decompress the CRC-free stream.

        Decompression is driven **only** by the response ``Content-Encoding``
        header — the server is authoritative about how it encoded the body.
        The request's ``compress_option`` / ``compress_algo`` only controls
        the ``ACCEPT-ENCODING`` header sent upstream (a hint the server may
        ignore); it must never be used to infer how to decode the response.
        When ``Content-Encoding`` is absent the body is uncompressed.
        """
        crc_stream = CrcStrippedInputStream(resp.raw)
        content_encoding = resp.headers.get("Content-Encoding")
        if not content_encoding:
            return crc_stream
        algo = CompressOption.CompressAlgorithm.from_encoding(content_encoding)
        resp_co = CompressOption(compress_algo=algo)
        return get_decompress_stream(crc_stream, resp_co, requests=False)

    def read_blobs(
        self,
        blob_references: List[Union[str, bytes]],
        *,
        compress_option: Optional[CompressOption] = None,
        compress_algo=None,
        stream: bool = False,
    ) -> Union[BlobDataIterator, BlobStreamReader]:
        """Download blobs by their references.

        The server's response framing depends on the number of references:

        - **Single reference**: the server returns raw unframed blob bytes
          (no ``[HeaderLen][Header][DataLen][Data]`` frames).  The entire
          decompressed payload is yielded as one :class:`BlobRecord`.
        - **Multiple references**: the server returns a framed stream with
          one frame per blob.  Each :class:`BlobRecord` carries the blob's
          MIME type and custom file name (v3+) from the frame header.

        References must be UTF-8 decodable strings or bytes.  Binary
        (non-UTF-8) references returned for nested BLOB columns are not
        supported by the generic ``BlobRead`` endpoint; reading them raises
        :class:`StorageClientError`.  Compression follows the tunnel
        pattern: ``compress_option`` (default ``None`` = uncompressed) or
        the shorthand ``compress_algo``.  Pass ``compress_algo="lz4"`` /
        ``"zstd"`` to request a compressed response. When ``stream=True``,
        returns a :class:`BlobStreamReader` for incremental file-like
        reads; otherwise returns a :class:`BlobDataIterator` yielding
        :class:`BlobRecord`.

        Example
        -------
        >>> blob_manager = client.open_blob_manager()
        >>> # Materialize each blob
        >>> for record in blob_manager.read_blobs([ref1, ref2]):
        ...     print(len(record.data))
        >>>
        >>> # Incremental streaming read
        >>> reader = blob_manager.read_blobs([ref], stream=True)
        >>> while True:
        ...     chunk = reader.read(4096)
        ...     if not chunk:
        ...         break
        ...     process(chunk)
        ...     reader = reader.next()
        """
        co = resolve_compress_option(compress_option, compress_algo)
        accept_encoding = self._accept_encoding(co)

        refs = blob_references or []
        str_refs = []
        for ref in refs:
            if isinstance(ref, bytes):
                try:
                    str_refs.append(ref.decode("utf-8"))
                except UnicodeDecodeError:
                    raise StorageClientError(
                        "Blob reference contains non-UTF-8 bytes (likely a "
                        "nested BLOB column).  The generic BlobRead endpoint "
                        "does not accept binary references; binary-ref "
                        "batch download is not yet supported."
                    )
            else:
                str_refs.append(ref)

        resp = self._client.stub.read_blobs(str_refs, accept_encoding)
        raw_stream = self._wrap_download_stream(resp)

        iterator = BlobDataIterator(
            raw_stream,
            api_version=self._api_version,
            expected_count=len(str_refs),
            crc_strip=False,
        )
        if stream:
            return BlobStreamReader(iterator)
        return iterator

    def read_blob(
        self,
        blob_reference: Union[str, bytes],
        *,
        compress_option: Optional[CompressOption] = None,
        compress_algo=None,
    ) -> io.BytesIO:
        """Download a single blob and return a :class:`io.BytesIO` over its data.

        Delegates to :meth:`read_blobs` with a single-element reference list.
        The server returns a raw unframed stream for single-ref requests, so
        no frame headers are parsed.  Compression follows the same pattern as
        :meth:`read_blobs`: ``compress_option`` (default ``None`` =
        uncompressed) or the shorthand ``compress_algo``.  The reference
        must be UTF-8 decodable.

        Example
        -------
        >>> blob_manager = client.open_blob_manager()
        >>> bio = blob_manager.read_blob(ref)
        >>> data = bio.read()
        >>> print(len(data))
        """
        iterator = self.read_blobs(
            [blob_reference],
            compress_option=compress_option,
            compress_algo=compress_algo,
        )
        records = list(iterator)
        if not records:
            return io.BytesIO(b"")
        return io.BytesIO(records[0].data)

    # ---- Write (standalone) ----

    def write_blob_stream(
        self,
        session_id: str,
        stream_id: str,
        *,
        stream_version: int = 1,
        column_name: Optional[str] = None,
        partition_spec: Optional[Union[str, dict, PartitionSpec]] = None,
        compress_option: Optional[CompressOption] = None,
        compress_algo=None,
        compress_level=None,
    ) -> BlobStreamWriter:
        """Create a streaming writer for a single blob upload.

        Uses chunked transfer-encoding; compression via ``compress_option``
        (default ``None`` = uncompressed; ``ZSTD``/``LZ4_FRAME`` opt-in via
        Content-Encoding).  The returned :class:`BlobStreamWriter` computes
        MD5 incrementally and verifies on ``finish()``.

        ``column_name`` is the dot-path name (a key of
        :meth:`WriteSchema.find_all_blob_column_ids`), resolved to the
        server-assigned column ID via the write schema.  When omitted, the
        sole top-level pure-BLOB column is auto-selected (raises if the
        schema has multiple BLOB columns or any nested BLOB column).

        ``partition_spec`` accepts the same types as
        :class:`~odps.types.PartitionSpec` (str, dict, or PartitionSpec).

        Example
        -------
        >>> blob_manager = client.open_blob_manager("my_table")
        >>> write_session = client.create_table_write_session("my_table")
        >>> sw = blob_manager.write_blob_stream(
        ...     write_session.id, stream_id="0", column_name="blob",
        ... )
        >>> sw.write(b"payload data")
        >>> resp = sw.finish()
        >>> print(resp.blob_reference)
        """
        if stream_id is None:
            raise ValueError("stream_id must not be None")
        stream_id = str(stream_id)
        co = resolve_compress_option(compress_option, compress_algo, compress_level)
        codec = CompressionCodec.from_compress_option(co)

        column_id = self._resolve_column_id(session_id, column_name)
        partition_values = ",".join(_normalize_partition_spec(partition_spec) or [])

        table_id = self._table_id
        params_extra = {
            "SessionId": session_id,
            "StreamId": stream_id,
            "StreamVersion": str(stream_version),
            "PartitionValues": partition_values,
            "ColumnIndex": str(column_id),
        }

        content_encoding = (
            codec.content_encoding if codec != CompressionCodec.NO_COMPRESSION else None
        )

        def upload(data_generator):
            return self._client.stub.table_write_blob(
                table_id,
                params_extra,
                data_generator,
                content_encoding=content_encoding,
            )

        return BlobStreamWriter(
            upload, compress_option=co, api_version=self._api_version
        )

    def build_blob_write_item(
        self,
        data,
        session_id,
        *,
        column_name=None,
        partition_spec=None,
        distribution_key=None,
        mime_type=None,
        custom_file_name=None,
        checksum_type=BlobWriteItem.ChecksumType.NONE,
    ) -> BlobWriteItem:
        """Build a :class:`BlobWriteItem` from writer-resolved state.

        ``column_name`` is resolved to the server-assigned column ID via
        the write schema.  When omitted, the sole top-level pure-BLOB
        column is auto-selected (raises if the schema has multiple BLOB
        columns or any nested BLOB column).  ``partition_spec`` accepts
        the same types as :class:`~odps.types.PartitionSpec`.
        ``api_version`` is stamped from the client so v3-only fields are
        gated automatically.

        Example
        -------
        >>> blob_manager = client.open_blob_manager("my_table")
        >>> write_session = client.create_table_write_session("my_table")
        >>> items = [
        ...     blob_manager.build_blob_write_item(
        ...         b"payload", write_session.id, column_name="blob",
        ...         mime_type="application/octet-stream",
        ...     )
        ... ]
        """
        column_id = self._resolve_column_id(session_id, column_name)

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

    def _resolve_column_id(self, session_id, column_name):
        """Resolve a dot-path column name to a server-assigned column ID.

        When *column_name* is ``None``, the sole top-level pure-BLOB column
        (if one exists and there are no nested BLOB columns) is selected
        automatically.
        """
        schema = self._client.stub.get_write_schema(self._table_id, session_id, None)
        _, column_id = schema.resolve_blob_column_name(column_name)
        return column_id

    def write_blob_batch(
        self,
        items: List[BlobWriteItem],
        session_id: str,
        stream_id: str,
        *,
        stream_version: int = 1,
        compress_option: Optional[CompressOption] = None,
        compress_algo=None,
        compress_level=None,
    ) -> "WriteBlobResponse":
        """Batch-upload multiple blobs in one request.

        Each :class:`BlobWriteItem` is framed and streamed to the server
        via chunked transfer-encoding — the full batch body is never
        materialized.  Compression via ``compress_option`` (default
        ``None`` = uncompressed).  Returns :class:`WriteBlobResponse`
        with blob_references matching input order.

        Example
        -------
        >>> blob_manager = client.open_blob_manager("my_table")
        >>> write_session = client.create_table_write_session("my_table")
        >>> items = [
        ...     blob_manager.build_blob_write_item(
        ...         b"a", write_session.id, column_name="blob",
        ...     ),
        ...     blob_manager.build_blob_write_item(
        ...         b"b", write_session.id, column_name="blob",
        ...     ),
        ... ]
        >>> resp = blob_manager.write_blob_batch(
        ...     items, write_session.id, stream_id="0",
        ... )
        >>> print(len(resp.blob_references))
        """
        if stream_id is None:
            raise ValueError("stream_id must not be None")
        stream_id = str(stream_id)
        co = resolve_compress_option(compress_option, compress_algo, compress_level)
        codec = CompressionCodec.from_compress_option(co)

        table_id = self._table_id
        params_extra = {
            "SessionId": session_id,
            "StreamId": stream_id,
            "StreamVersion": str(stream_version),
            "Mode": "Batch",
        }

        content_encoding = (
            codec.content_encoding if codec != CompressionCodec.NO_COMPRESSION else None
        )

        def upload(data_generator):
            return self._client.stub.table_batch_write_blob(
                table_id,
                params_extra,
                data_generator,
                content_encoding=content_encoding,
            )

        return stream_blob_batch(
            items, upload, compress_option=co, api_version=self._api_version
        )
