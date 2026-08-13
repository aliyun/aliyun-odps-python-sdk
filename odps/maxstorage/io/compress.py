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

"""Compression utilities for :mod:`odps.maxstorage`.

Thin re-export of tunnel ``get_compress_stream`` / ``get_decompress_stream``
plus the ``CompressionCodec`` enum that restricts the storage API to the
three supported codecs (``NO_COMPRESSION`` / ``ZSTD`` / ``LZ4_FRAME``),
rejecting ``ODPS_ZLIB`` / ``ODPS_SNAPPY``.
"""

import enum

from ...tunnel.io.stream import (
    CompressOption,
    get_compress_stream,
    get_decompress_stream,
)

__all__ = [
    "CompressionCodec",
    "CompressOption",
    "get_compress_stream",
    "get_decompress_stream",
    "resolve_compress_option",
]


class CompressionCodec(enum.Enum):
    """Compression codecs supported by the Storage API v2.

    Only ``NO_COMPRESSION``, ``ZSTD``, and ``LZ4_FRAME`` are supported.
    Tunnel algorithms like ``ODPS_ZLIB`` / ``ODPS_SNAPPY`` are NOT accepted.
    """

    NO_COMPRESSION = ""
    ZSTD = "ZSTD"
    LZ4_FRAME = "LZ4_FRAME"

    @property
    def content_encoding(self):
        """HTTP ``Content-Encoding`` / ``ACCEPT-ENCODING`` value.

        ``None`` when uncompressed.
        """
        if self == CompressionCodec.NO_COMPRESSION:
            return None
        elif self == CompressionCodec.ZSTD:
            return "zstd"
        elif self == CompressionCodec.LZ4_FRAME:
            return "x-lz4-frame"

    @property
    def accept_encoding(self):
        """Value for the ``ACCEPT-ENCODING`` request header (alias)."""
        return self.content_encoding

    @classmethod
    def from_compress_option(cls, compress_option):
        """Map an :class:`odps.tunnel.CompressOption` (or ``None``) to a codec.

        Raises ``ValueError`` for unsupported algorithms (ZLIB, SNAPPY, etc.).
        """
        if compress_option is None:
            return cls.NO_COMPRESSION
        algo = compress_option.algorithm
        if algo == CompressOption.CompressAlgorithm.ODPS_ZSTD:
            return cls.ZSTD
        if algo in (
            CompressOption.CompressAlgorithm.ODPS_LZ4,
            CompressOption.CompressAlgorithm.ODPS_ARROW_LZ4,
        ):
            return cls.LZ4_FRAME
        if algo == CompressOption.CompressAlgorithm.ODPS_RAW:
            return cls.NO_COMPRESSION
        raise ValueError(
            f"Storage API v2 does not support compression algorithm {algo}. "
            f"Supported: NO_COMPRESSION, ZSTD, LZ4_FRAME."
        )

    @classmethod
    def build_compress_option(cls, compress_algo, compress_level=None):
        """Build a ``CompressOption`` from a shorthand algorithm string/value.

        ``compress_algo`` may be ``None`` (uncompressed), a
        ``CompressOption.CompressAlgorithm``, or a string name.
        Returns a :class:`CompressOption` or ``None``.
        """
        if compress_algo is None:
            return None
        if isinstance(compress_algo, CompressOption):
            return compress_algo
        return CompressOption(compress_algo=compress_algo, level=compress_level)


def resolve_compress_option(
    compress_option=None, compress_algo=None, compress_level=None
):
    """Resolve the effective ``CompressOption`` from either form.

    ``compress_option`` takes priority over the shorthand ``compress_algo`` /
    ``compress_level``.  Returns ``None`` when both are ``None`` (uncompressed).
    Validates the result against the 3 supported codecs.
    """
    if compress_option is not None:
        CompressionCodec.from_compress_option(compress_option)  # validate
        return compress_option
    if compress_algo is not None:
        co = CompressionCodec.build_compress_option(compress_algo, compress_level)
        if co is not None:
            CompressionCodec.from_compress_option(co)  # validate
        return co
    return None
