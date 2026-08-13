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

"""Arrow IPC stream writer for :mod:`odps.maxstorage`.

Assembles an Arrow IPC stream (schema message + batch messages + EOS marker)
from cached byte arrays.  Content-Type ``application/octet-stream``.

Compression uses Arrow IPC's built-in codec (``IpcWriteOptions``) — NOT HTTP
``Content-Encoding``.
"""

import struct
from io import BytesIO

from ...tunnel.io.stream import CompressOption

try:
    import pyarrow as pa
except ImportError:
    pa = None

# IPC end-of-stream marker: [continuation token 0xFFFFFFFF][zero-length 0x00000000]
_EOS_MARKER = struct.pack("<II", 0xFFFFFFFF, 0x00000000)


def _compress_option_to_arrow_codec(compress_option):
    """Map a tunnel ``CompressOption`` to a pyarrow codec name (or ``None``).

    Returns ``"zstd"`` / ``"lz4"`` for the supported codecs, ``None`` for
    no compression.  Only ZSTD and LZ4_FRAME are supported (matching
    :class:`CompressionCodec`).
    """
    if compress_option is None:
        return None
    algo = compress_option.algorithm
    if algo == CompressOption.CompressAlgorithm.ODPS_ZSTD:
        return "zstd"
    if algo == CompressOption.CompressAlgorithm.ODPS_LZ4:
        return "lz4"
    return None


class RawArrowRequestBody:
    """Assembles an Arrow IPC stream body from pre-serialized batches.

    Layout: ``[schema message][batch1 message]...[batchN message][EOS marker]``.
    When ``compress_option`` is set, the Arrow IPC built-in codec is used
    (via ``IpcWriteOptions``) — no HTTP ``Content-Encoding`` is applied.
    """

    CONTENT_TYPE = "application/octet-stream"

    def __init__(self, arrow_schema, batch_bytes_list, compress_option=None):
        if pa is None:
            raise ValueError("pyarrow is required for Arrow write")
        self._arrow_schema = arrow_schema
        self._batch_bytes_list = batch_bytes_list
        self._compress_option = compress_option
        self._total_bytes = sum(len(b) for b in batch_bytes_list)

    @property
    def content_type(self):
        return self.CONTENT_TYPE

    def get_total_bytes(self):
        """Sum of batch byte lengths (excludes schema + EOS overhead, for logging)."""
        return self._total_bytes

    def serialize(self):
        """Return the full Arrow IPC stream as ``bytes``.

        When ``compress_option`` is set, the schema and batch messages are
        re-serialized with Arrow IPC's built-in compression codec.  Otherwise
        the pre-serialized batch messages are concatenated directly.
        """
        codec = _compress_option_to_arrow_codec(self._compress_option)

        if codec is not None:
            return self._serialize_compressed(codec)

        return self._serialize_uncompressed()

    def _serialize_uncompressed(self):
        """Assemble: schema (no EOS) + batch messages + single EOS."""
        sink = BytesIO()
        # Schema message (new_stream + close writes schema + EOS; strip EOS)
        schema_stream = pa.ipc.new_stream(sink, self._arrow_schema)
        schema_stream.close()
        schema_bytes = sink.getvalue()
        if len(schema_bytes) >= 8 and schema_bytes[-8:] == _EOS_MARKER:
            schema_bytes = schema_bytes[:-8]

        out = BytesIO()
        out.write(schema_bytes)
        for batch_bytes in self._batch_bytes_list:
            out.write(batch_bytes)
        out.write(_EOS_MARKER)
        return out.getvalue()

    def _serialize_compressed(self, codec):
        """Re-serialize all batches with Arrow IPC built-in compression.

        The pre-serialized batch messages are uncompressed, so each is
        deserialized back to a ``RecordBatch`` (via ``read_message`` +
        ``read_record_batch`` with the known schema) and re-written into a
        fresh compressed IPC stream.
        """
        sink = BytesIO()
        writer = pa.ipc.new_stream(
            sink,
            self._arrow_schema,
            options=pa.ipc.IpcWriteOptions(compression=codec),
        )
        for batch_bytes in self._batch_bytes_list:
            # batch_bytes may be ``bytes`` or a ``pa.Buffer`` (serialize_batch
            # returns the latter); BufferReader accepts both, unlike BytesIO.
            message = pa.ipc.read_message(pa.BufferReader(batch_bytes))
            batch = pa.ipc.read_record_batch(message, self._arrow_schema)
            writer.write_batch(batch)
        writer.close()
        return sink.getvalue()

    def get_stream(self):
        """Return a file-like object yielding the stream."""
        return BytesIO(self.serialize())


def serialize_batch(batch, compress_option=None):
    """Serialize a single ``RecordBatch`` to raw Arrow IPC batch-message bytes.

    Returns only the batch record message (no schema message, no EOS marker)
    so the caller can accumulate multiple batches for
    :class:`RawArrowRequestBody`, which prepends the schema and appends EOS.

    Compression is NOT applied per-batch; it is handled by
    :class:`RawArrowRequestBody` on the full assembled stream.
    """
    if pa is None:
        raise ValueError("pyarrow is required for Arrow write")

    return batch.serialize()
