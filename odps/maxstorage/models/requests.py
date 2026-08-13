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

"""Request DTOs for the storage API.

Plain dataclasses with JSON field-name mapping.
Each class has a ``to_dict()`` producing the wire JSON body.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..options import IncrementalReadOptions, SplitOptions
from .enums import DataFormat


@dataclass
class CreateTableReadSessionRequest:
    """Wire: ``{Table, Partitions[], SplitOptions{...}, DataFormat{...},
    RequiredDataFormat, QuotaName, EnableDynamicColumns, IncrementalReadOptions{...}}``.
    """

    required_data_columns: List[str] = field(default_factory=list)
    required_partition_columns: List[str] = field(default_factory=list)
    required_partitions: List[str] = field(default_factory=list)
    required_bucket_ids: List[int] = field(default_factory=list)
    split_options: Optional[SplitOptions] = None
    arrow_options: Optional[Dict] = None
    filter_predicate: str = ""
    filter_predicate_fallback: bool = False
    split_max_file_num: int = 0
    incremental_read: bool = False
    incremental_read_options: Optional[IncrementalReadOptions] = None

    def to_dict(self):
        arrow_opts = self.arrow_options or {
            "TimestampUnit": "nano",
            "DatetimeUnit": "milli",
        }
        d = {
            "RequiredDataColumns": self.required_data_columns,
            "RequiredPartitionColumns": self.required_partition_columns,
            "RequiredPartitions": self.required_partitions,
            "RequiredBucketIds": self.required_bucket_ids,
            "SplitOptions": (self.split_options or SplitOptions()).to_dict(),
            "ArrowOptions": arrow_opts,
            "FilterPredicate": self.filter_predicate,
            "FilterPredicateFallback": self.filter_predicate_fallback,
            "SplitMaxFileNum": self.split_max_file_num,
            "IncrementalRead": self.incremental_read,
        }
        if self.incremental_read_options is not None:
            d["IncrementalReadOptions"] = self.incremental_read_options.to_dict()
        return d


@dataclass
class CreateTableReadStreamRequest:
    """Wire: ``{Index, SplitId, SessionId, DataColumns[], DataColumnsUnordered,
    SkipRowNum, MaxBatchRows, MaxBatchRawSize, DataFormat, RouteToken}``.

    No ``CompressMode`` field — arrow read compression is negotiated via the
    HTTP ``ACCEPT-ENCODING`` header.
    """

    max_batch_rows: int = 4096
    skip_row_num: int = 0
    max_batch_raw_size: int = 0
    data_format: Optional[DataFormat] = None
    data_columns: List[str] = field(default_factory=list)
    data_columns_unordered: bool = False

    def to_dict(self):
        df = self.data_format or DataFormat.default()
        return {
            "MaxBatchRows": self.max_batch_rows,
            "SkipRowNum": self.skip_row_num,
            "MaxBatchRawSize": self.max_batch_raw_size,
            "DataFormat": df.to_dict(),
            "DataColumns": self.data_columns,
            "DataColumnsUnordered": self.data_columns_unordered,
        }


@dataclass
class CreateInstanceReadSessionRequest:
    """Wire: ``{Instance, TaskName, QueryId, EnableLimit, Columns[], ...}``."""

    enable_limit: bool = False

    def to_dict(self):
        return {"EnableLimit": self.enable_limit}


@dataclass
class CreateInstanceReadStreamRequest:
    """Wire: ``{TaskName, QueryId, EnableLimit, Columns[]}``."""

    task_name: str = "AnonymousSQLTask"
    query_id: int = 0
    enable_limit: bool = False
    columns: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "TaskName": self.task_name,
            "QueryId": self.query_id,
            "EnableLimit": self.enable_limit,
            "Columns": self.columns,
        }


@dataclass
class CreateTableWriteSessionRequest:
    """Wire: ``{Table, Partitions[], WriteMode, QuotaName, Overwrite,
    EnableSchemaEvolution, RequiredDataFormat, ...}``."""

    partial_partition_spec: str = ""
    flags: Dict[str, str] = field(default_factory=dict)
    required_data_format: Optional[DataFormat] = None

    def to_dict(self):
        d = {
            "PartialPartitionSpec": self.partial_partition_spec,
            "Flags": self.flags,
        }
        if self.required_data_format is not None:
            d["RequiredDataFormat"] = self.required_data_format.to_dict()
        return d


@dataclass
class CreateWriteStreamRequest:
    """Wire: ``{SessionId, StreamId, StreamVersion, ExactlyOnceMode, RouteToken}``."""

    stream_id: Optional[str] = None
    stream_version: int = 0
    exactly_once_mode: bool = False

    def to_dict(self):
        d = {
            "StreamId": self.stream_id,
            "StreamVersion": self.stream_version,
        }
        if self.exactly_once_mode:
            d["ExactlyOnceMode"] = True
        return d


@dataclass
class GetWriteStreamRequest:
    """Wire: ``{SessionId, StreamId, RouteToken}`` (body fields TableId,
    StreamId, StreamVersion, ExactlyOnceMode)."""

    session_id: Optional[str] = None
    stream_id: Optional[str] = None
    stream_version: int = 0
    table_id: Optional[str] = None
    exactly_once_mode: Optional[bool] = None

    def to_dict(self):
        d = {
            "StreamId": self.stream_id,
            "StreamVersion": self.stream_version,
        }
        if self.table_id is not None:
            d["TableId"] = self.table_id
        if self.exactly_once_mode is not None:
            d["ExactlyOnceMode"] = self.exactly_once_mode
        return d


@dataclass
class CloseWriteStreamRequest:
    """Wire: ``{SessionId, StreamId, StreamVersion, RouteToken}``."""

    session_id: Optional[str] = None
    stream_id: Optional[str] = None
    stream_version: int = 0

    def to_dict(self):
        return {
            "SessionId": self.session_id,
            "StreamId": self.stream_id,
            "StreamVersion": self.stream_version,
        }


@dataclass
class TablePreviewRequest:
    """Wire: ``{Limit, Partition, Columns[]}``."""

    limit: Optional[int] = None
    partition: Optional[str] = None
    columns: List[str] = field(default_factory=list)

    def to_dict(self):
        d = {}
        if self.limit is not None:
            d["Limit"] = self.limit
        if self.partition is not None:
            d["Partition"] = self.partition
        if self.columns:
            d["Columns"] = self.columns
        return d


@dataclass
class BlobWriteRequest:
    """Wire: ``{"BlobReferences": [str, ...]}`` — JSON body for blob DOWNLOAD.

    Despite the "Write" name, this is the download request DTO.
    """

    blob_references: List[str] = field(default_factory=list)

    def to_dict(self):
        return {"BlobReferences": self.blob_references}
