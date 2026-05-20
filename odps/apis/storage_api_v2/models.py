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

"""Data models, request/response classes, and enums for the Storage API V2."""

import hashlib
import json
import struct
import zlib
from enum import Enum
from io import BytesIO

from ... import serializers
from ...errors import ParseError
from ...models.core import JSONRemoteModel

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Status(Enum):
    INIT = "INIT"
    OK = "OK"
    WAIT = "WAIT"
    RUNNING = "RUNNING"


class SessionStatus(Enum):
    INIT = "INIT"
    NORMAL = "NORMAL"
    CRITICAL = "CRITICAL"
    EXPIRED = "EXPIRED"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"


class Compression(Enum):
    """Compression algorithm for Storage API V2 data streams.

    Attributes
    ----------
    UNCOMPRESSED
        No compression. Default for all operations.
    ZSTD
        Zstandard compression. Requires the ``zstandard`` library.
    LZ4_FRAME
        LZ4 frame compression. Requires the ``lz4`` library.
    """

    UNCOMPRESSED = 0
    ZSTD = 1
    LZ4_FRAME = 2

    def to_compression_name(self):
        if self.value == 0:
            return None
        elif self.value == 1:
            return "zstd"
        elif self.value == 2:
            return "lz4"
        else:
            return "unknown"


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class SplitOptions(JSONRemoteModel):
    class SplitMode(str, Enum):
        SIZE = "Size"
        PARALLELISM = "Parallelism"
        ROW_OFFSET = "RowOffset"
        BUCKET = "Bucket"

    split_mode = serializers.JSONNodeField(
        "SplitMode", parse_callback=lambda s: SplitOptions.SplitMode(s)
    )
    split_unit = serializers.JSONNodeField("SplitUnit")
    split_number = serializers.JSONNodeField("SplitNumber")
    cross_partition = serializers.JSONNodeField("CrossPartition")

    def __init__(self, **kwargs):
        super(SplitOptions, self).__init__(**kwargs)

        self.split_mode = (
            self.split_mode
            if self.split_mode is not None
            else SplitOptions.SplitMode.SIZE
        )
        self.split_unit = self.split_unit if self.split_unit is not None else "ByteSize"
        self.split_number = (
            self.split_number if self.split_number is not None else 256 * 1024 * 1024
        )
        self.cross_partition = (
            self.cross_partition if self.cross_partition is not None else True
        )


class ArrowOptions(JSONRemoteModel):
    class TimestampUnit(str, Enum):
        SECOND = "second"
        MILLI = "milli"
        MICRO = "micro"
        NANO = "nano"

    timestamp_unit = serializers.JSONNodeField(
        "TimestampUnit", parse_callback=lambda s: ArrowOptions.TimestampUnit(s)
    )
    date_time_unit = serializers.JSONNodeField(
        "DatetimeUnit", parse_callback=lambda s: ArrowOptions.TimestampUnit(s)
    )

    def __init__(self, **kwargs):
        super(ArrowOptions, self).__init__(**kwargs)

        self.timestamp_unit = (
            self.timestamp_unit
            if self.timestamp_unit is not None
            else ArrowOptions.TimestampUnit.NANO
        )
        self.date_time_unit = (
            self.date_time_unit
            if self.date_time_unit is not None
            else ArrowOptions.TimestampUnit.MILLI
        )


class Column(JSONRemoteModel):
    name = serializers.JSONNodeField("Name")
    type = serializers.JSONNodeField("Type")
    comment = serializers.JSONNodeField("Comment")
    nullable = serializers.JSONNodeField("Nullable")


class DataSchema(JSONRemoteModel):
    data_columns = serializers.JSONNodesReferencesField(Column, "DataColumns")
    partition_columns = serializers.JSONNodesReferencesField(Column, "PartitionColumns")


class DataFormat(JSONRemoteModel):
    type = serializers.JSONNodeField("Type")
    version = serializers.JSONNodeField("Version")

    def __init__(self, **kwargs):
        super(DataFormat, self).__init__(**kwargs)

        self.type = self.type if self.type is not None else "Arrow"
        self.version = self.version if self.version is not None else "V5"


class IncrementalReadOptions(JSONRemoteModel):
    mode = serializers.JSONNodeField("Mode")
    start_time_stamp = serializers.JSONNodeField("StartTimeStamp")
    end_time_stamp = serializers.JSONNodeField("EndTimeStamp")
    start_version = serializers.JSONNodeField("StartVersion")
    end_version = serializers.JSONNodeField("EndVersion")


class SessionStats(JSONRemoteModel):
    estimated_size = serializers.JSONNodeField("EstimatedSize")
    estimated_row_count = serializers.JSONNodeField("EstimatedRowCount")


# ---------------------------------------------------------------------------
# Request classes
# ---------------------------------------------------------------------------


class CreateReadSessionRequest(serializers.JSONSerializableModel):
    """Request for creating a read session.

    Attributes
    ----------
    required_data_columns : list of str
        List of column names to read. If empty, all columns are returned.
    required_partition_columns : list of str
        Partition columns to include in the result.
    required_partitions : list of str
        Specific partition values to read (e.g., ['pt=20230101']).
    required_bucket_ids : list of str
        Bucket IDs to read for bucket-based tables.
    split_options : SplitOptions
        Controls how data is split into chunks. Defaults to size-based
        splitting with 256MB chunks.
    arrow_options : ArrowOptions
        Arrow format settings like timestamp precision.
    filter_predicate : str
        SQL-like filter condition to apply during reading.
    filter_predicate_fallback : bool
        Whether to fallback to server-side filtering if predicate pushdown fails.
    split_max_file_num : int
        Maximum number of files per split for file-based splitting.
    incremental_read : bool
        Enable incremental reading mode for capturing table changes.
    incremental_read_options : IncrementalReadOptions
        Options for incremental read mode (version range, timestamp range).
    """

    required_data_columns = serializers.JSONNodeField("RequiredDataColumns")
    required_partition_columns = serializers.JSONNodeField("RequiredPartitionColumns")
    required_partitions = serializers.JSONNodeField("RequiredPartitions")
    required_bucket_ids = serializers.JSONNodeField("RequiredBucketIds")
    split_options = serializers.JSONNodeReferenceField(SplitOptions, "SplitOptions")
    arrow_options = serializers.JSONNodeReferenceField(ArrowOptions, "ArrowOptions")
    filter_predicate = serializers.JSONNodeField("FilterPredicate")
    filter_predicate_fallback = serializers.JSONNodeField("FilterPredicateFallback")
    split_max_file_num = serializers.JSONNodeField("SplitMaxFileNum")
    incremental_read = serializers.JSONNodeField("IncrementalRead")
    incremental_read_options = serializers.JSONNodeReferenceField(
        IncrementalReadOptions, "IncrementalReadOptions"
    )

    def __init__(self, **kwargs):
        super(CreateReadSessionRequest, self).__init__(**kwargs)

        self.required_data_columns = (
            self.required_data_columns if self.required_data_columns is not None else []
        )
        self.required_partition_columns = (
            self.required_partition_columns
            if self.required_partition_columns is not None
            else []
        )
        self.required_partitions = (
            self.required_partitions if self.required_partitions is not None else []
        )
        self.required_bucket_ids = (
            self.required_bucket_ids if self.required_bucket_ids is not None else []
        )
        self.split_options = (
            self.split_options if self.split_options is not None else SplitOptions()
        )
        self.arrow_options = (
            self.arrow_options if self.arrow_options is not None else ArrowOptions()
        )
        self.filter_predicate = (
            self.filter_predicate if self.filter_predicate is not None else ""
        )
        self.filter_predicate_fallback = (
            self.filter_predicate_fallback
            if self.filter_predicate_fallback is not None
            else False
        )
        self.split_max_file_num = (
            self.split_max_file_num if self.split_max_file_num is not None else 0
        )
        self.incremental_read = (
            self.incremental_read if self.incremental_read is not None else False
        )


class CreateWriteSessionRequest(serializers.JSONSerializableModel):
    """Request for creating a write session.

    Attributes
    ----------
    partial_partition_spec : str
        Partition specification for writing to a specific partition.
        Format: 'pt=20230101' or 'pt=20230101,region=us-west'.
        If empty, writes to the table's default location.
    flags : dict
        Additional flags for session configuration. Common flags
        include 'overwrite' to replace existing partition data.
    """

    partial_partition_spec = serializers.JSONNodeField("PartialPartitionSpec")
    flags = serializers.JSONNodeField("Flags")

    def __init__(self, **kwargs):
        super(CreateWriteSessionRequest, self).__init__(**kwargs)

        self.partial_partition_spec = (
            self.partial_partition_spec
            if self.partial_partition_spec is not None
            else ""
        )
        self.flags = self.flags if self.flags is not None else {}


class ReadStreamRequest:
    """Request for reading data from a specific split in a read session.

    Attributes
    ----------
    session_id : str
        The read session identifier from create_read_session.
    split_index : int, optional
        Which split to read (0 to splits_count-1). If None,
        reads all data in the session.
    row_offset : int, optional
        Starting row offset within the split. Defaults to 0.
    row_count : int, optional
        Maximum number of rows to read. If None, reads all
        rows in the split.
    max_batch_rows : int
        Maximum rows per Arrow batch in the stream. Controls
        memory usage during reading. Default 4096.
    skip_row_num : int
        Number of rows to skip before reading. Default 0.
    max_batch_raw_size : int
        Maximum raw byte size per batch. 0 means no limit. Default 0.
    data_format : DataFormat, optional
        Format of returned data (Arrow V5 is default).
    data_columns : list of str
        Specific columns to read. Must match session schema.
    compression : Compression, default None
        Compression algorithm for the stream data. None means
        Compression.UNCOMPRESSED.
    """

    def __init__(
        self,
        session_id,
        split_index=None,
        row_offset=None,
        row_count=None,
        max_batch_rows=4096,
        skip_row_num=0,
        max_batch_raw_size=0,
        data_format=None,
        data_columns=None,
        compression=None,
    ):
        self.session_id = session_id
        self.split_index = split_index
        self.row_offset = row_offset
        self.row_count = row_count
        self.max_batch_rows = max_batch_rows
        self.skip_row_num = skip_row_num
        self.max_batch_raw_size = max_batch_raw_size
        self.data_format = data_format or DataFormat()
        self.data_columns = data_columns or []
        self.compression = (
            compression if compression is not None else Compression.UNCOMPRESSED
        )


class CreateWriteStreamRequest:
    """Request for creating a write stream within an active write session.

    Attributes
    ----------
    session_id : str
        The write session identifier from create_write_session.
    stream_id : str or int
        Unique identifier for this stream within the session.
        Typically an integer (0, 1, 2, ...) for parallel streams.
    stream_version : int
        Version number for the stream. Increment if retrying
        after a failed upload to the same stream_id. Default 0.
    exactly_once_mode : bool
        Whether to enable Exactly-Once semantics for this stream.
        When enabled, the server returns an access token and tracks
        row offsets for idempotent writes. Default False.
    """

    def __init__(
        self, session_id, stream_id, stream_version=0, exactly_once_mode=False
    ):
        self.session_id = session_id
        self.stream_id = stream_id
        self.stream_version = stream_version
        self.exactly_once_mode = exactly_once_mode


class WriteStreamRequest:
    """Request for writing row data to a write stream via streaming upload.

    Attributes
    ----------
    session_id : str
        The write session identifier.
    stream_id : str or int
        The stream identifier from create_write_stream.
    stream_version : int
        Version of the stream (should match create_write_stream). Default 0.
    record_count : int
        Total number of records to be written. Set this to the
        expected count for validation, or 0 if unknown. Default 0.
    compression : Compression, default None
        Compression algorithm for the uploaded data stream. None means
        Compression.UNCOMPRESSED.
    row_offset : int
        Row offset for Exactly-Once mode. Set to -1 (default) to not
        send the offset. In Exactly-Once mode, this tracks the starting
        row position for idempotent writes.
    access_token : str
        Access token for Exactly-Once mode. Obtained from
        CreateWriteStreamResponse or GetWriteStreamResponse.
    """

    def __init__(
        self,
        session_id,
        stream_id,
        stream_version=0,
        record_count=0,
        compression=None,
        row_offset=-1,
        access_token=None,
    ):
        self.session_id = session_id
        self.stream_id = stream_id
        self.stream_version = stream_version
        self.record_count = record_count
        self.compression = (
            compression if compression is not None else Compression.UNCOMPRESSED
        )
        self.row_offset = row_offset
        self.access_token = access_token


class CloseWriteStreamRequest:
    """Request for closing a write stream to finalize the data upload.

    Attributes
    ----------
    session_id : str
        The write session identifier.
    stream_id : str or int
        The stream identifier to close.
    stream_version : int
        Version of the stream (should match create_write_stream). Default 0.
    """

    def __init__(self, session_id, stream_id, stream_version=0):
        self.session_id = session_id
        self.stream_id = stream_id
        self.stream_version = stream_version


class PreviewTableRequest:
    """Request for previewing table data without creating a session.

    Attributes
    ----------
    limit : int, optional
        Maximum number of rows to preview. If None, returns a
        small default sample (typically 100-1000 rows).
    partition : str, optional
        Partition specification to preview specific partition data.
        Format: 'pt=20230101' or 'pt=20230101,region=us-west'.
    columns : list of str
        Specific columns to preview. If empty, all columns are returned.
    """

    def __init__(self, limit=None, partition=None, columns=None):
        self.limit = limit
        self.partition = partition
        self.columns = columns or []


# ---------------------------------------------------------------------------
# Response classes
# ---------------------------------------------------------------------------


class CreateReadSessionResponse(serializers.JSONSerializableModel):
    """Response from creating a read session.

    Attributes
    ----------
    session_id : str
        Unique identifier for this read session. Required for all
        subsequent read operations.
    session_status : SessionStatus
        Current status (INIT, NORMAL, EXPIRED, etc).
    splits_count : int
        Number of data splits available for parallel reading.
    record_count : int
        Total number of records across all splits.
    data_schema : DataSchema
        Schema information for data and partition columns.
    expiration_time : str
        When the session expires and needs refresh.
    download_id : str
        Download ID (used for instance-based reads, mapped to session_id).
    status : SessionStatus
        Status field for instance-based reads (mapped to session_status).
    session_type : str
        Type of the read session.
    supported_data_format : list of DataFormat
        List of data formats supported by the server.
    split_mode : str
        Mode used for splitting the data.
    split_bucket_id : str
        Bucket ID used for bucket-based splitting.
    session_stats : SessionStats
        Statistics about the session (estimated size, row count).
    latest_version : int
        Latest data version for incremental reads.
    message : str
        Server message, if any.
    enable_large_string : bool
        Whether large string support is enabled.
    incremental_read_options : IncrementalReadOptions
        Options for incremental read from the server response.
    request_id : str
        Request ID for debugging (set from response headers).
    """

    __slots__ = ("request_id", "route_token")

    session_id = serializers.JSONNodeField("SessionId")
    download_id = serializers.JSONNodeField("DownloadID")
    session_status = serializers.JSONNodeField(
        "SessionStatus",
        parse_callback=lambda s: SessionStatus(s.upper()) if s else None,
    )
    status = serializers.JSONNodeField(
        "Status",
        parse_callback=lambda s: SessionStatus(s.upper()) if s else None,
    )
    session_type = serializers.JSONNodeField("SessionType")
    expiration_time = serializers.JSONNodeField("ExpirationTime")
    splits_count = serializers.JSONNodeField("SplitsCount")
    record_count = serializers.JSONNodeField("RecordCount")
    data_schema = serializers.JSONNodeReferenceField(DataSchema, "DataSchema")
    supported_data_format = serializers.JSONNodesReferencesField(
        DataFormat, "SupportedDataFormat"
    )
    split_mode = serializers.JSONNodeField("SplitMode")
    split_bucket_id = serializers.JSONNodeField("SplitBucketId")
    session_stats = serializers.JSONNodeReferenceField(SessionStats, "SessionStats")
    latest_version = serializers.JSONNodeField("LatestVersion")
    message = serializers.JSONNodeField("Message")
    enable_large_string = serializers.JSONNodeField("EnableLargeString")
    incremental_read_options = serializers.JSONNodeReferenceField(
        IncrementalReadOptions, "IncrementalReadOptions"
    )

    def __init__(self):
        super(CreateReadSessionResponse, self).__init__()

        self.request_id = ""
        self.route_token = None


class CreateWriteSessionResponse(serializers.JSONSerializableModel):
    """Response from creating a write session.

    Attributes
    ----------
    session_id : str
        Unique identifier for this write session. Required for
        creating write streams and committing the session.
    warning_message : str
        Warning message if the session has any issues.
    route_token : str
        Routing token for load balancing, extracted from response headers.
        Pass this token to subsequent operations (create_write_stream,
        commit_write_session, etc.) to ensure session affinity.
    request_id : str
        Request ID for debugging (set from response headers).
    """

    __slots__ = ("request_id", "route_token")

    session_id = serializers.JSONNodeField("SessionId")
    warning_message = serializers.JSONNodeField("WarningMessage")

    def __init__(self, **kwargs):
        super(CreateWriteSessionResponse, self).__init__(**kwargs)
        self.request_id = ""
        self.route_token = None


class GetWriteSessionResponse(serializers.JSONSerializableModel):
    """Response from getting write session status.

    Attributes
    ----------
    streams : list of dict
        Information about all write streams created in this session.
        Each stream dict contains stream_id, stream_version, and
        stream status.
    warning_message : str
        Warning message if any streams have issues.
    route_token : str
        Routing token for load balancing, extracted from response headers.
        Pass this token to subsequent operations to ensure session affinity.
    request_id : str
        Request ID for debugging (set from response headers).
    """

    __slots__ = ("request_id", "route_token")

    streams = serializers.JSONNodeField("Streams")
    warning_message = serializers.JSONNodeField("WarningMessage")

    def __init__(self, **kwargs):
        super(GetWriteSessionResponse, self).__init__(**kwargs)
        self.request_id = ""
        self.route_token = None


class GetWriteStreamResponse(serializers.JSONSerializableModel):
    """Response from getting write stream status.

    Attributes
    ----------
    status : str
        Current stream state (OPEN, CLOSED, ERROR).
    record_count : int
        Number of records written to this stream.
    byte_size : int
        Total bytes uploaded through this stream.
    error_code : str
        Error code if stream is in ERROR state.
    error_message : str
        Detailed error message if stream failed.
    request_id : str
        Request ID for debugging (set from response headers).
    """

    __slots__ = ("request_id", "route_token")

    status = serializers.JSONNodeField("Status")
    record_count = serializers.JSONNodeField("RecordCount")
    byte_size = serializers.JSONNodeField("ByteSize")
    error_code = serializers.JSONNodeField("ErrorCode")
    error_message = serializers.JSONNodeField("ErrorMessage")
    latest_schema_version = serializers.JSONNodeField("LatestSchemaVersion")
    row_offset = serializers.JSONNodeField("RowOffset")
    access_token = serializers.JSONNodeField("AccessToken")

    def __init__(self, **kwargs):
        super(GetWriteStreamResponse, self).__init__(**kwargs)
        self.request_id = ""
        self.route_token = None


class CreateWriteStreamResponse(serializers.JSONSerializableModel):
    """Response from creating a write stream.

    Attributes
    ----------
    data_schema : dict
        The table schema for this stream, including column names
        and types. Use this to validate data before writing.
    table_id : str
        Internal table identifier.
    schema_version : str
        Schema version number.
    route_token : str
        Routing token for load balancing, extracted from response headers.
    request_id : str
        Request ID for debugging (set from response headers).
    """

    __slots__ = ("request_id", "route_token")

    data_schema = serializers.JSONNodeField("TableSchema")
    table_id = serializers.JSONNodeField("TableId")
    schema_version = serializers.JSONNodeField("SchemaVersion")
    access_token = serializers.JSONNodeField("AccessToken")

    def __init__(self, **kwargs):
        super(CreateWriteStreamResponse, self).__init__(**kwargs)
        self.request_id = ""
        self.route_token = None


class CloseWriteStreamResponse(serializers.JSONSerializableModel):
    """Response from closing a write stream.

    Attributes
    ----------
    warning_message : str
        Warning message if the stream closure has any issues.
        Check this for potential problems even if close succeeds.
    request_id : str
        Request ID for debugging (set from response headers).
    """

    __slots__ = ("request_id",)

    warning_message = serializers.JSONNodeField("WarningMessage")

    def __init__(self, **kwargs):
        super(CloseWriteStreamResponse, self).__init__(**kwargs)
        self.request_id = ""


class WriteStreamResponse(serializers.JSONSerializableModel):
    """Response from a write operation in Exactly-Once mode.

    Attributes
    ----------
    warning_message : str
        Warning message if the write operation has any issues.
    exactly_once_row_offset : int
        The server-side row offset after a successful write in
        Exactly-Once mode. Used to track the committed position
        for idempotent writes.
    """

    warning_message = serializers.JSONNodeField("WarningMessage")
    exactly_once_row_offset = serializers.JSONNodeField("ExactlyOnceRowOffset")


# ---------------------------------------------------------------------------
# Blob model classes
# ---------------------------------------------------------------------------


class ChecksumType(Enum):
    NONE = 0
    CRC32 = 1
    MD5 = 2


class BlobWriteItem:
    """A single blob item for batch upload.

    Wire format when serialized:
        [8-byte LE header_len][header JSON][8-byte LE data_len][data bytes][8-byte LE footer_len][footer JSON]
    """

    def __init__(
        self,
        data,
        partition_values=None,
        column_index=0,
        distribution_key=None,
        mime_type=None,
        checksum_type=ChecksumType.NONE,
        size=None,
    ):
        self.data = data
        self.partition_values = partition_values or []
        self.column_index = column_index
        self.distribution_key = distribution_key
        self.mime_type = mime_type
        self.checksum_type = checksum_type
        self._size = size

    def _get_data_size(self):
        """Return the size of data in bytes."""
        if isinstance(self.data, (bytes, bytearray)):
            return len(self.data)
        if hasattr(self.data, "__len__"):
            return len(self.data)
        if self._size is not None:
            return self._size
        if hasattr(self.data, "seek") and hasattr(self.data, "tell"):
            pos = self.data.tell()
            self.data.seek(0, 2)
            size = self.data.tell()
            self.data.seek(pos)
            return size
        raise ValueError(
            "Cannot determine data size for stream. "
            "Pass a seekable stream or provide the 'size' parameter."
        )

    def _is_stream(self):
        """Return True if data is a file-like object (not bytes)."""
        return not isinstance(self.data, (bytes, bytearray))

    def _build_header(self):
        header = {
            "PartitionValues": self.partition_values if self.partition_values else [],
            "ColumnIndex": self.column_index,
        }
        if self.distribution_key is not None:
            header["DistributionKey"] = self.distribution_key
        if self.mime_type is not None:
            header["ContentType"] = self.mime_type
        return header

    def _build_footer(self):
        checksum = {"Type": self.checksum_type.value}
        if self.checksum_type == ChecksumType.CRC32:
            checksum["Crc32"] = zlib.crc32(self.data) & 0xFFFFFFFF
        elif self.checksum_type == ChecksumType.MD5:
            checksum["MD5"] = hashlib.md5(self.data).hexdigest()
        return {"Checksum": checksum}

    def serialize(self):
        """Serialize this item into the binary frame format."""
        header_bytes = json.dumps(self._build_header()).encode("utf-8")
        footer_bytes = json.dumps(self._build_footer()).encode("utf-8")
        buf = BytesIO()
        buf.write(struct.pack("<q", len(header_bytes)))
        buf.write(header_bytes)
        buf.write(struct.pack("<q", len(self.data)))
        buf.write(self.data)
        buf.write(struct.pack("<q", len(footer_bytes)))
        buf.write(footer_bytes)
        return buf.getvalue()

    def write_frame_to(self, stream, chunk_size=256 * 1024):
        """Write this item's frame to a file-like stream.

        Writes: [8-byte LE header_len][header JSON][8-byte LE data_len][data][8-byte LE footer_len][footer JSON]

        For bytes data, writes directly. For file-like data, reads and writes
        in chunks, computing checksums incrementally.
        """
        header_bytes = json.dumps(self._build_header()).encode("utf-8")
        data_size = self._get_data_size()

        stream.write(struct.pack("<q", len(header_bytes)))
        stream.write(header_bytes)
        stream.write(struct.pack("<q", data_size))

        crc32_value = 0
        md5_digest = hashlib.md5()
        has_checksum = self.checksum_type != ChecksumType.NONE

        if self._is_stream():
            while True:
                chunk = self.data.read(chunk_size)
                if not chunk:
                    break
                if has_checksum:
                    if self.checksum_type == ChecksumType.CRC32:
                        crc32_value = zlib.crc32(chunk, crc32_value)
                    elif self.checksum_type == ChecksumType.MD5:
                        md5_digest.update(chunk)
                stream.write(chunk)
        else:
            if has_checksum:
                if self.checksum_type == ChecksumType.CRC32:
                    crc32_value = zlib.crc32(self.data, 0)
                elif self.checksum_type == ChecksumType.MD5:
                    md5_digest.update(self.data)
            stream.write(self.data)

        footer = {"Checksum": {"Type": self.checksum_type.value}}
        if self.checksum_type == ChecksumType.CRC32:
            footer["Checksum"]["Crc32"] = crc32_value & 0xFFFFFFFF
        elif self.checksum_type == ChecksumType.MD5:
            footer["Checksum"]["MD5"] = md5_digest.hexdigest()

        footer_bytes = json.dumps(footer).encode("utf-8")
        stream.write(struct.pack("<q", len(footer_bytes)))
        stream.write(footer_bytes)

    @staticmethod
    def write_blobs(items):
        """Serialize a list of BlobWriteItems into a single byte array."""
        if not items:
            return b""
        parts = []
        for item in items:
            parts.append(item.serialize())
        return b"".join(parts)


class WriteBlobRequest:
    """Request for uploading blob data (streaming or batch).

    Attributes
    ----------
    session_id : str
        The write session identifier from create_write_session.
    stream_id : str or int
        Stream identifier for this upload.
    stream_version : int
        Version number for the stream. Default 0.
    partition_values : list of str
        Partition values for the blob location.
        Format: ['pt=20230101', 'region=us-west'].
    column_index : int
        Column index in the table schema where the blob will be stored. Default 0.
    """

    def __init__(
        self,
        session_id,
        stream_id,
        stream_version=0,
        partition_values=None,
        column_index=0,
    ):
        self.session_id = session_id
        self.stream_id = stream_id
        self.stream_version = stream_version
        self.partition_values = partition_values or []
        self.column_index = column_index


class WriteBlobResponse(serializers.JSONSerializableModel):
    """Response from a blob write operation (stream or batch).

    Attributes
    ----------
    blob_reference : str
        Single blob reference if one blob was uploaded.
    blob_references : list of str
        List of blob references matching the input items order.
        Use these references to read the blobs later.
    warning_message : str
        Warning message if any blob uploads had issues.
    size : int
        Total bytes uploaded.
    request_id : str
        Request ID for debugging (set from response headers).
    """

    __slots__ = ("request_id",)

    blob_reference = serializers.JSONNodeField("BlobReference")
    blob_references = serializers.JSONNodeField("BlobReferences")
    warning_message = serializers.JSONNodeField("WarningMessage")
    size = serializers.JSONNodeField("Size")

    def __init__(self, **kwargs):
        super(WriteBlobResponse, self).__init__(**kwargs)
        self.request_id = ""


class ReadBlobRequest:
    """Request for downloading binary blobs by their references.

    Attributes
    ----------
    blob_references : list of str
        List of blob reference strings obtained from previous
        upload operations (write_blob_stream or write_blob_batch).
        These references uniquely identify the blobs to download.
        Bytes references are automatically decoded to UTF-8 strings.
    """

    def __init__(self, blob_references):
        self.blob_references = [
            ref.decode("utf-8") if isinstance(ref, bytes) else ref
            for ref in blob_references
        ]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _update_request_id(response, resp):
    if "x-odps-request-id" in resp.headers:
        response.request_id = resp.headers["x-odps-request-id"]


def _parse_json_response(resp):
    """Parse JSON body from a tunnel REST response."""
    try:
        if hasattr(resp, "json"):
            return resp.json()
        return json.loads(resp.text)
    except (ValueError, TypeError) as exc:
        snippet = resp.text[:200] if hasattr(resp, "text") else repr(resp)
        raise ParseError(
            f"Failed to parse JSON response: {exc}. Response snippet: {snippet}"
        ) from exc
