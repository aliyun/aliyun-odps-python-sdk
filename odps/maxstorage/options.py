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

"""Public options / configuration dataclasses for :mod:`odps.maxstorage`.

``BlobWriteItem`` is defined here (not in ``io/blob_writer.py``)
because it is part of the public API surface and referenced widely.

The ``maxstorage.*`` option namespace is registered in
:mod:`odps.config` via :func:`register_options`.
"""

import enum
import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from typing import List, Optional, Union

from ..types import PartitionSpec

# SplitMode is imported lazily to avoid a circular import
# (models/__init__ -> requests -> options -> models.enums).


@dataclass
class SplitOptions:
    """Controls how a read session splits data.

    Fields
    ------
    split_mode : SplitMode
        Strategy: ``SIZE`` (default), ``ROW_OFFSET``, ``PARALLELISM``, ``BUCKET``.
    split_unit : str
        Unit of ``split_number``.  Default ``"ByteSize"``.
    split_number : int
        Target split size (bytes) for ``SIZE`` mode, or row count for
        ``ROW_OFFSET``.  Default 256 MiB.
    cross_partition : bool
        Whether splits may cross partition boundaries.  Default ``True``.
    """

    split_mode: object = None
    split_unit: str = "ByteSize"
    split_number: int = 256 * 1024 * 1024
    cross_partition: bool = True

    def __post_init__(self):
        if self.split_mode is None:
            from .models.enums import (  # noqa: F401  # avoid circular import
                SplitMode as _SM,
            )

            self.split_mode = _SM.SIZE

    def to_dict(self):
        return {
            "SplitMode": self.split_mode.value,
            "SplitUnit": self.split_unit,
            "SplitNumber": self.split_number,
            "CrossPartition": self.cross_partition,
        }


@dataclass
class IncrementalReadOptions:
    """Incremental-read configuration.

    Wire fields: ``{Mode, From, To}``.  ``version`` selects the incremental
    mode; ``from_``/``to`` are version boundaries (int) or timestamps (str).
    """

    version: Optional[str] = None
    from_: Optional[Union[int, str]] = None
    to: Optional[Union[int, str]] = None

    def to_dict(self):
        d = {}
        if self.version is not None:
            d["Mode"] = self.version
        if self.from_ is not None:
            d["From"] = self.from_
        if self.to is not None:
            d["To"] = self.to
        return d


# ---------------------------------------------------------------------------
# BlobWriteItem
# ---------------------------------------------------------------------------


class BlobWriteItem:
    """A single blob item for batch upload.

    .. note::

        Prefer :meth:`TableArrowWriter.build_blob_write_item` or
        :meth:`BlobManager.build_blob_write_item` to construct items —
        they resolve ``column_id``, normalize ``partition_spec``, and
        stamp ``api_version`` automatically.  Constructing
        :class:`BlobWriteItem` directly requires knowledge of
        server-assigned column IDs and wire-format details.
    """

    class ChecksumType(enum.Enum):
        NONE = 0
        CRC32 = 1
        MD5 = 2

    def __init__(
        self,
        data,
        *,
        column_id,
        partition_values=None,
        distribution_key=None,
        mime_type=None,
        custom_file_name=None,
        checksum_type=ChecksumType.NONE,
        api_version="2",
    ):
        self.data = data
        self.column_id = column_id
        self.partition_values = partition_values or []
        self.distribution_key = distribution_key
        self.mime_type = mime_type
        self.custom_file_name = custom_file_name
        self.checksum_type = checksum_type
        self.api_version = api_version

    # -- size helpers -------------------------------------------------------

    def _get_data_size(self):
        if isinstance(self.data, (bytes, bytearray)):
            return len(self.data)
        if hasattr(self.data, "__len__"):
            return len(self.data)
        if hasattr(self.data, "seek") and hasattr(self.data, "tell"):
            pos = self.data.tell()
            self.data.seek(0, 2)
            size = self.data.tell()
            self.data.seek(pos)
            return size - pos
        raise ValueError(
            "Cannot determine data size for stream. "
            "Pass a seekable stream or provide bytes."
        )

    def _is_stream(self):
        return not isinstance(self.data, (bytes, bytearray))

    # -- frame building -----------------------------------------------------

    def _build_header(self):
        # Wire-format header frame: {PartitionValues, ColumnIndex,
        # DistributionKey, ContentType, CustomFileName (v3 only)}.
        header = {
            "PartitionValues": self.partition_values if self.partition_values else [],
            "ColumnIndex": self.column_id,
        }
        if self.distribution_key is not None:
            header["DistributionKey"] = self.distribution_key
        if self.mime_type is not None:
            header["ContentType"] = self.mime_type
        if self.custom_file_name is not None and _str_version_ge(self.api_version, 3):
            header["CustomFileName"] = self.custom_file_name
        return header

    def _build_footer(self, crc32_value=None, md5_hex=None):
        # Wire-format footer frame: {Checksum: {Type, Crc32|MD5}}.
        checksum = {"Type": self.checksum_type.value}
        if self.checksum_type == self.ChecksumType.CRC32:
            checksum["Crc32"] = crc32_value & 0xFFFFFFFF
        elif self.checksum_type == self.ChecksumType.MD5:
            checksum["MD5"] = md5_hex
        return {"Checksum": checksum}

    def write_frame_to(self, stream, chunk_size=256 * 1024):
        """Write this item to a file-like stream.

        For file-like ``data``, reads and writes in chunks, computing
        checksums incrementally to avoid loading the entire payload.
        """
        # Wire format: [8-byte LE header_len][header JSON]
        #              [8-byte LE data_len  ][data bytes    ]
        #              [8-byte LE footer_len ][footer JSON   ]
        header_bytes = json.dumps(self._build_header()).encode("utf-8")
        data_size = self._get_data_size()

        stream.write(struct.pack("<q", len(header_bytes)))
        stream.write(header_bytes)
        stream.write(struct.pack("<q", data_size))

        crc32_value = 0
        md5_digest = hashlib.md5()
        has_checksum = self.checksum_type != self.ChecksumType.NONE

        if self._is_stream():
            while True:
                chunk = self.data.read(chunk_size)
                if not chunk:
                    break
                if has_checksum:
                    if self.checksum_type == self.ChecksumType.CRC32:
                        crc32_value = zlib.crc32(chunk, crc32_value)
                    elif self.checksum_type == self.ChecksumType.MD5:
                        md5_digest.update(chunk)
                stream.write(chunk)
        else:
            if has_checksum:
                if self.checksum_type == self.ChecksumType.CRC32:
                    crc32_value = zlib.crc32(self.data, 0)
                elif self.checksum_type == self.ChecksumType.MD5:
                    md5_digest.update(self.data)
            stream.write(self.data)

        footer_bytes = json.dumps(
            self._build_footer(
                crc32_value, md5_digest.hexdigest() if has_checksum else None
            )
        ).encode("utf-8")
        stream.write(struct.pack("<q", len(footer_bytes)))
        stream.write(footer_bytes)


def _str_version_ge(version, major):
    """Return True when a string API version is >= the given major number."""
    try:
        return int(version) >= major
    except (TypeError, ValueError):
        return False


def _supports_v3(api_version) -> bool:
    """True when ``api_version >= 3``.  Gates v3-era features only."""
    return _str_version_ge(api_version, 3)


def _normalize_partition_spec(partition_spec) -> Optional[List[str]]:
    """Normalize a partition spec into a list of ``'key=value'`` strings.

    Returns ``None`` when *partition_spec* is ``None`` or a blank string.
    Accepts ``str``, ``dict``, or :class:`~odps.types.PartitionSpec`.
    """
    if partition_spec is not None and not (
        isinstance(partition_spec, str) and not partition_spec.strip()
    ):
        spec = PartitionSpec(partition_spec)
        return [f"{k}={v}" for k, v in spec.items()]
    return None


AUTO_COMMIT_SESSION_ID = "default"
"""Sentinel session id meaning "no explicit session" (auto-commit)."""

AUTO_COMMIT_DEFAULT_STREAM_ID = "default"
"""Sentinel stream id meaning "no explicit stream" (auto-commit)."""
