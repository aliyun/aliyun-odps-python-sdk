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

"""Blob write framing helpers for :mod:`odps.maxstorage`.

``BlobWriteItem`` is defined in :mod:`odps.maxstorage.options` (public API).
This module provides the :class:`BlobStreamWriter` for streaming single-blob
uploads with MD5 verification, and :func:`stream_blob_batch` for streaming
batch uploads.  Both use :class:`odps.tunnel.io.RequestsIO` for true chunked
transfer-encoding — data is sent to the server as it is written, never
materialized in full before the HTTP call.
"""

import hashlib
from io import IOBase
from typing import Callable, List, Optional

from ...errors import ChecksumError
from ...tunnel.io import RequestsIO
from ...tunnel.io.stream import CompressOption, get_compress_stream
from ..errors import MaxStorageError
from ..io.compress import CompressionCodec
from ..models.enums import Status
from ..models.responses import WriteBlobResponse
from ..options import BlobWriteItem
from ..stub import _parse_json_response, _update_request_id

_CODEC_TO_TUNNEL = {
    CompressionCodec.ZSTD: CompressOption.CompressAlgorithm.ODPS_ZSTD,
    CompressionCodec.LZ4_FRAME: CompressOption.CompressAlgorithm.ODPS_LZ4,
}


def _build_compress_stream(req_io, compress_option):
    """Build a compressor pipeline on top of ``req_io``.

    Returns ``req_io`` itself for NO_COMPRESSION, otherwise wraps it with
    the tunnel compress stream for the codec's corresponding algorithm.
    """
    codec = CompressionCodec.from_compress_option(compress_option)
    if codec == CompressionCodec.NO_COMPRESSION:
        return req_io
    co = CompressOption(compress_algo=_CODEC_TO_TUNNEL[codec])
    return get_compress_stream(req_io, co)


class BlobStreamWriter(IOBase):
    """Stream writer for single blob upload with MD5 checksum verification.

    Wraps a :class:`RequestsIO` + compressor pipeline.  Data written via
    ``write()`` is compressed (if requested) and streamed to the server in
    chunks via chunked transfer-encoding — the full blob is never held in
    memory.  MD5 is computed incrementally on the uncompressed data and
    verified against the server response on ``finish()``.
    """

    def __init__(
        self,
        upload: Callable,
        compress_option: Optional[CompressOption] = None,
        api_version: str = "2",
    ):
        self._compress_option = compress_option
        self._req_io = RequestsIO(upload, chunk_size=256 * 1024)
        self._req_io.start()
        self._res = None
        self._stopped = False
        self._md5_digest = hashlib.md5()
        self._api_version = api_version

        self._compressor = _build_compress_stream(self._req_io, compress_option)

    def writable(self) -> bool:
        return not self._stopped

    def write(self, data) -> int:
        if self._stopped:
            return 0
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._md5_digest.update(data)
        self._compressor.write(data)
        return len(data)

    def finish(self) -> WriteBlobResponse:
        """Finish writing, send the data, and verify MD5.

        Returns the parsed :class:`WriteBlobResponse` on success.
        Raises :class:`odps.errors.ChecksumError` on MD5 mismatch, or
        :class:`MaxStorageError` if the upload fails with a non-200 response.
        """
        self._stopped = True

        # Flush any buffered data through the compressor → RequestsIO.
        if hasattr(self._compressor, "flush"):
            self._compressor.flush()
        else:
            self._req_io.flush()

        self._res = self._req_io.finish()

        if self._res is not None and self._res.status_code == 200:
            resp_json = _parse_json_response(self._res)
            response = WriteBlobResponse.from_dict(resp_json)
            _update_request_id(response, self._res)

            # Verify MD5 checksum (server returns MD5 of uncompressed data)
            md5_value = resp_json.get("MD5Value")
            if md5_value and md5_value != self._md5_digest.hexdigest():
                raise ChecksumError(
                    f"MD5 value mismatch, expected: {md5_value}, "
                    f"actual: {self._md5_digest.hexdigest()}"
                )
            return response
        raise MaxStorageError(
            "Blob upload failed: HTTP %s, response: %s"
            % (
                getattr(self._res, "status_code", "None"),
                getattr(self._res, "text", "")[:500] if self._res is not None else "",
            )
        )

    def get_status(self) -> Status:
        if not self._stopped:
            return Status.RUNNING
        return Status.OK

    def get_request_id(self) -> Optional[str]:
        if not self._stopped:
            return None
        if self._res is not None:
            return self._res.headers.get("x-odps-request-id")
        return None


def stream_blob_batch(
    items: List[BlobWriteItem],
    upload: Callable,
    compress_option: Optional[CompressOption] = None,
    api_version: str = "2",
) -> WriteBlobResponse:
    """Stream a batch of :class:`BlobWriteItem` items to the server.

    Each item is written via :meth:`BlobWriteItem.write_frame_to` directly
    to the :class:`RequestsIO` pipeline in chunks — the full batch body is
    never materialized.

    Parameters
    ----------
    items : list[BlobWriteItem]
        The blob items to upload.
    upload : callable
        ``upload(data_generator) -> response`` — typically a closure over
        ``StorageStub.table_batch_write_blob``.
    compress_option : CompressOption, optional
        Compression to apply to the batch body.  ``None`` = uncompressed.
    api_version : str
        API version, stamped onto each item for v3 feature gating.

    Returns
    -------
    WriteBlobResponse
        Parsed server response with blob references.
    """
    req_io = RequestsIO(upload, chunk_size=256 * 1024)
    req_io.start()

    out_stream = _build_compress_stream(req_io, compress_option)

    for item in items:
        item.api_version = api_version
        item.write_frame_to(out_stream)

    if hasattr(out_stream, "flush"):
        out_stream.flush()
    else:
        req_io.flush()

    resp = req_io.finish()

    resp_json = _parse_json_response(resp)
    response = WriteBlobResponse.from_dict(resp_json)
    _update_request_id(response, resp)
    return response
