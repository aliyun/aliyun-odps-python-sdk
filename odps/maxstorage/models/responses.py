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

"""Response DTOs for the storage API.

Each class has a ``from_dict(d)`` classmethod parsing the wire JSON body.
``request_id`` is NOT in the JSON body — it comes from the
``x-odps-request-id`` response header, set by the stub after parsing.
``route_token`` likewise comes from the ``x-odps-max-storage-route-token``
response header.
"""

import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .enums import DataFormat, SessionStats
from .schema import ReadSchema, WriteSchema


@dataclass
class CreateTableReadSessionResponse:
    """Response from ``TableCreateReadSession``.

    ``session_id``, ``session_status``, ``splits_count``, ``record_count``,
    ``data_schema`` (ReadSchema), ``expiration_time``, ``route_token``.
    """

    session_id: Optional[str] = None
    session_status: Optional[str] = None
    session_type: Optional[str] = None
    splits_count: int = 0
    record_count: int = 0
    data_schema: Optional[ReadSchema] = None
    expiration_time: Optional[int] = None
    split_mode: Optional[str] = None
    split_bucket_id: Optional[List[int]] = None
    session_stats: Optional[SessionStats] = None
    latest_version: Optional[int] = None
    supported_data_format: List[DataFormat] = field(default_factory=list)
    message: Optional[str] = None
    enable_large_string: bool = False
    incremental_read_options: Optional[Dict] = None
    request_id: str = ""
    route_token: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(
            session_id=d.get("SessionId"),
            session_status=d.get("SessionStatus"),
            session_type=d.get("SessionType"),
            splits_count=d.get("SplitsCount", 0),
            record_count=d.get("RecordCount", 0),
            data_schema=ReadSchema.from_dict(d.get("DataSchema")),
            expiration_time=d.get("ExpirationTime"),
            split_mode=d.get("SplitMode"),
            split_bucket_id=d.get("SplitBucketId"),
            session_stats=SessionStats.from_dict(d.get("SessionStats")),
            latest_version=d.get("LatestVersion"),
            supported_data_format=[
                DataFormat.from_dict(df) for df in (d.get("SupportedDataFormat") or [])
            ],
            message=d.get("Message"),
            enable_large_string=d.get("EnableLargeString", False),
            incremental_read_options=d.get("IncrementalReadOptions"),
        )


# GetTableReadSessionResponse has the same shape as create response.
GetTableReadSessionResponse = CreateTableReadSessionResponse


@dataclass
class CreateInstanceReadSessionResponse:
    """Response from ``InstanceCreateReadSession`` / ``InstanceGetReadSession``.

    The session-id wire field is ``DownloadID`` (not ``SessionId``);
    ``status`` is a raw string (not ``SessionStatus`` enum); ``TableSchema``
    uses the **WriteSchema** format (nested camelCase with int type codes).
    """

    download_id: Optional[str] = None
    record_count: int = 0
    status: Optional[str] = None
    table_schema: Optional[WriteSchema] = None
    quota_name: Optional[str] = None
    request_id: str = ""

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(
            download_id=d.get("DownloadID"),
            record_count=d.get("RecordCount", 0),
            status=d.get("Status"),
            table_schema=WriteSchema.from_dict(d.get("TableSchema")),
            quota_name=d.get("QuotaName"),
        )


@dataclass
class CreateTableWriteSessionResponse:
    """Response from ``TableCreateWriteSession``."""

    session_id: Optional[str] = None
    warning_message: Optional[str] = None
    request_id: str = ""
    route_token: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(
            session_id=d.get("SessionId"),
            warning_message=d.get("WarningMessage"),
        )


@dataclass
class GetTableWriteSessionResponse:
    """Response from ``TableGetWriteSession``."""

    streams: Optional[Dict] = None
    warning_message: Optional[str] = None
    min_uncommitted_staging_id: Optional[str] = None
    request_id: str = ""
    route_token: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(
            streams=d.get("Streams"),
            warning_message=d.get("WarningMessage"),
            min_uncommitted_staging_id=d.get("MinUncommittedStagingId"),
        )


@dataclass
class CreateWriteStreamResponse:
    """Response from ``TableCreateWriteStream``.

    ``data_schema`` is the raw ``TableSchema`` dict (WriteSchema format).
    ``table_id`` / ``schema_version`` are used by streaming writes on every
    flush.  ``access_token`` is present only in exactly-once mode.
    """

    data_schema: Optional[dict] = None
    table_id: Optional[str] = None
    schema_version: Optional[int] = None
    access_token: Optional[str] = None
    request_id: str = ""
    route_token: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(
            data_schema=d.get("TableSchema"),
            table_id=d.get("TableId"),
            schema_version=d.get("SchemaVersion"),
            access_token=d.get("AccessToken"),
        )


@dataclass
class GetWriteStreamResponse(CreateWriteStreamResponse):
    """Response from ``TableGetWriteStream``.

    Extends :class:`CreateWriteStreamResponse` with ``status``,
    ``record_count``, ``latest_schema_version``, ``row_offset``.
    """

    status: Optional[str] = None
    record_count: Optional[int] = None
    latest_schema_version: Optional[int] = None
    row_offset: Optional[int] = None

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        # Note: "status" and "recordCount" use lower/camelCase while other
        # fields (TableSchema, TableId, etc.) use PascalCase.  This mixed
        # casing matches the server wire format — the server serializes
        # these two fields by their raw property name, not a PascalCase
        # override.  Do not "fix" them to PascalCase; parsing will break.
        return cls(
            data_schema=d.get("TableSchema"),
            table_id=d.get("TableId"),
            schema_version=d.get("SchemaVersion"),
            access_token=d.get("AccessToken"),
            status=d.get("status"),
            record_count=d.get("recordCount"),
            latest_schema_version=d.get("LatestSchemaVersion"),
            row_offset=d.get("RowOffset"),
        )


@dataclass
class CloseWriteStreamResponse:
    """Response from ``TableCloseWriteStream``."""

    warning_message: Optional[str] = None
    request_id: str = ""

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(warning_message=d.get("WarningMessage"))


@dataclass
class WriteStreamResponse:
    """Response body from ``TableWrite`` (parsed by ``parse_write_stream_response``).

    Fields: ``WarningMessage``, ``ExactlyOnceRowOffset`` (EO only),
    ``StagingId`` (streaming).
    """

    warning_message: Optional[str] = None
    exactly_once_row_offset: Optional[int] = None
    staging_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(
            warning_message=d.get("WarningMessage"),
            exactly_once_row_offset=d.get("ExactlyOnceRowOffset"),
            staging_id=d.get("StagingId"),
        )


@dataclass
class WriteBlobResponse:
    """Response from blob write operations (stream or batch).

    The server returns blob references as base64-encoded strings.  They are
    stored raw in ``blob_reference_b64`` / ``blob_references_b64``; the
    ``blob_reference`` / ``blob_references`` properties decode them to
    ``bytes`` for direct use (e.g. placing into an Arrow ``binary`` column).

    ``request_id`` is NOT in the JSON body — sourced from the HTTP response
    header.
    """

    blob_reference_b64: Optional[str] = None
    blob_references_b64: List[str] = field(default_factory=list)
    warning_message: Optional[str] = None
    size: Optional[int] = None
    request_id: str = ""

    @property
    def blob_reference(self) -> Optional[bytes]:
        """Decoded single blob reference (``bytes``), or ``None``."""
        if not self.blob_reference_b64:
            return None
        return base64.b64decode(self.blob_reference_b64)

    @property
    def blob_references(self) -> List[bytes]:
        """Decoded blob references (``list[bytes]``), in input order."""
        return [base64.b64decode(r) for r in self.blob_references_b64]

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(
            blob_reference_b64=d.get("BlobReference"),
            blob_references_b64=d.get("BlobReferences") or [],
            warning_message=d.get("WarningMessage"),
            size=d.get("Size"),
        )
