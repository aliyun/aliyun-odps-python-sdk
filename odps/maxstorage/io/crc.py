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

"""CRC32C-stripping stream for blob downloads.

Strips per-block CRC32C checksums from the wire format:
``[4096 bytes data][4 bytes CRC32C]`` repeated, with a potentially shorter
final block ``[N bytes data (1<=N<=4096)][4 bytes CRC32C]``.

Each block's 4-byte little-endian CRC32C trailer is verified against the
computed Castagnoli CRC32C of the preceding data before being stripped.
A mismatch raises :class:`~odps.maxstorage.errors.MaxStorageError`, so a
corrupt download fails fast rather than silently returning bad data.
"""

import struct
from io import BytesIO

from ...crc import Crc32c
from ..errors import MaxStorageError

_CRC_BLOCK_SIZE = 4096
_CRC_SIZE = 4
_FULL_BLOCK_TOTAL = _CRC_BLOCK_SIZE + _CRC_SIZE  # 4100


def read_exact(stream, n, strict=True):
    """Read exactly *n* bytes from a stream.

    When *strict* (default), a short read (stream ended before *n* bytes)
    raises :class:`MaxStorageError` so truncation is loud.  When
    *strict* is ``False``, a short read returns whatever was read (used
    by peek-style callers that tolerate partial data).
    """
    bio = BytesIO()
    while bio.tell() < n:
        chunk = stream.read(n - bio.tell())
        if not chunk:
            break
        bio.write(chunk)
    data = bio.getvalue()
    if strict and len(data) < n:
        raise MaxStorageError(
            f"Corrupt blob stream: expected {n} bytes, got {len(data)} "
            f"(truncated stream)."
        )
    return data


def read_le_long(stream):
    """Read an 8-byte little-endian signed int64 from a stream.

    Returns ``None`` at a clean EOF (0 bytes).  A partial read of 1-7
    bytes indicates a truncated/corrupt stream and raises
    :class:`MaxStorageError` rather than silently returning ``None``.
    """
    data = stream.read(8)
    if not data:
        return None
    if len(data) < 8:
        raise MaxStorageError(
            "Corrupt blob stream: expected 8 bytes for a length prefix, "
            f"got {len(data)} (truncated stream)."
        )
    return struct.unpack("<q", data)[0]


class CrcStrippedInputStream:
    """File-like wrapper that strips CRC32C checksums from the underlying stream.

    Wire format: ``[4096 bytes data][4 bytes CRC32C]`` repeated, with a
    potentially shorter final block ``[N bytes data (1<=N<=4096)][4 bytes CRC32C]``.
    """

    def __init__(self, raw_stream):
        self._raw_stream = raw_stream
        self._buffer = bytearray()
        self._finished = False

    def _fill(self):
        if self._finished:
            return
        block = read_exact(self._raw_stream, _FULL_BLOCK_TOTAL, strict=False)
        if not block:
            self._finished = True
            return
        if len(block) == _FULL_BLOCK_TOTAL:
            data_slice = block[:_CRC_BLOCK_SIZE]
            trailer = block[_CRC_BLOCK_SIZE:_FULL_BLOCK_TOTAL]
        else:
            if len(block) > _CRC_SIZE:
                data_slice = block[:-_CRC_SIZE]
                trailer = block[-_CRC_SIZE:]
            else:
                raise MaxStorageError(
                    f"Corrupt blob stream: final block of {len(block)} bytes "
                    f"is shorter than the {_CRC_SIZE}-byte CRC trailer."
                )
            self._finished = True

        # Verify the CRC32C trailer before stripping.
        expected = struct.unpack("<I", trailer)[0]
        actual = Crc32c()
        actual.update(bytearray(data_slice))
        if actual.getvalue() != expected:
            raise MaxStorageError(
                f"CRC32C mismatch: expected 0x{expected:08x}, "
                f"got 0x{actual.getvalue():08x}."
            )
        self._buffer += data_slice

    def peek(self, size):
        if size <= 0:
            return b""
        while len(self._buffer) < size and not self._finished:
            self._fill()
        return bytes(self._buffer[:size])

    def read(self, size=-1):
        if size == 0:
            return b""
        while True:
            if size < 0:
                if not self._finished:
                    self._fill()
                    continue
                break
            else:
                if len(self._buffer) >= size or self._finished:
                    break
                self._fill()
        if size < 0:
            result = bytes(self._buffer)
            self._buffer = bytearray()
            return result
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result
