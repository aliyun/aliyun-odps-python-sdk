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

"""Enumerations and small value objects for the storage API.

Enums for storage models, settings, and ``WriteMode``.
``CompressionCodec`` lives in :mod:`odps.maxstorage.io.compress` because it
depends on tunnel ``CompressOption``.
"""

import enum
import warnings


class DataFormat:
    """Arrow data-format descriptor.

    JSON: ``{"Type": "Arrow", "Version": "V5"}`` (defaults).
    Not an ``enum`` — it carries two free-form string fields.
    """

    def __init__(self, type="Arrow", version="V5"):
        self.type = type
        self.version = version

    @classmethod
    def default(cls):
        return cls()

    def to_dict(self):
        return {"Type": self.type, "Version": self.version}

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(type=d.get("Type", "Arrow"), version=d.get("Version", "V5"))

    def __eq__(self, other):
        return (
            isinstance(other, DataFormat)
            and self.type == other.type
            and self.version == other.version
        )

    def __repr__(self):
        return f"DataFormat(type={self.type!r}, version={self.version!r})"


class SessionStatus(enum.Enum):
    """Read/write session lifecycle status."""

    INIT = "INIT"
    NORMAL = "NORMAL"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    CRITICAL = "CRITICAL"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value):
        """Fall back to UNKNOWN instead of raising ValueError."""
        warnings.warn(
            f"Unknown {cls.__name__} value: {value!r}, falling back to UNKNOWN",
        )
        return cls.UNKNOWN

    @classmethod
    def from_string(cls, value):
        """Parse a status string; ``None``/empty -> ``UNKNOWN``."""
        if not value:
            return cls.UNKNOWN
        return cls(value.upper())


class SplitMode(enum.Enum):
    """Data-splitting strategy for read sessions.

    JSON values: ``"Size"``, ``"Parallelism"``, ``"RowOffset"``, ``"Bucket"``.
    """

    SIZE = "Size"
    PARALLELISM = "Parallelism"
    ROW_OFFSET = "RowOffset"
    BUCKET = "Bucket"

    @classmethod
    def from_string(cls, value):
        if not value:
            return cls.SIZE
        try:
            return cls(value)
        except ValueError:
            warnings.warn(
                f"Unknown {cls.__name__} value: {value!r}, falling back to SIZE",
                stacklevel=2,
            )
            return cls.SIZE


class TimestampUnit(enum.Enum):
    """Arrow timestamp precision.

    JSON: ``"second"``, ``"milli"``, ``"micro"``, ``"nano"``.
    """

    SECOND = "second"
    MILLI = "milli"
    MICRO = "micro"
    NANO = "nano"


class WriteMode(enum.Enum):
    """Write-session mode.

    ``BATCH`` / ``BATCH_COMPATIBLE`` -- data visible only after commit.
    ``STREAMING`` / ``STREAMING_REALTIME`` -- data visible immediately on flush.
    """

    BATCH = "Batch"
    BATCH_COMPATIBLE = "BatchCompatible"
    STREAMING = "Streaming"
    STREAMING_REALTIME = "StreamingRealtime"

    def is_streaming(self):
        """True for ``STREAMING`` / ``STREAMING_REALTIME``."""
        return self in (WriteMode.STREAMING, WriteMode.STREAMING_REALTIME)


class Status(enum.Enum):
    """Writer / stream status."""

    INIT = "INIT"
    OK = "OK"
    WAIT = "WAIT"
    RUNNING = "RUNNING"


class SessionStats:
    """Estimated session size / row count from a read-session response."""

    def __init__(self, estimated_size=None, estimated_row_count=None):
        self.estimated_size = estimated_size
        self.estimated_row_count = estimated_row_count

    @classmethod
    def from_dict(cls, d):
        if not d:
            return None
        return cls(
            estimated_size=d.get("EstimatedSize"),
            estimated_row_count=d.get("EstimatedRowCount"),
        )

    def __repr__(self):
        return (
            f"SessionStats(estimated_size={self.estimated_size!r}, "
            f"estimated_row_count={self.estimated_row_count!r})"
        )
