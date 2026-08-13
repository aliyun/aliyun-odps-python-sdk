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

"""odps.maxstorage — PyODPS Storage API v2 module.

High-throughput Arrow-based table/instance read sessions, table write sessions
(batch / streaming), table preview, and blob download/upload — all built on
the existing PyODPS RestClient, account signing, options, and type system.
"""

from .base import MaxStorageClient
from .blob import (
    BlobDataIterator,
    BlobManager,
    BlobRecord,
    BlobStreamReader,
    BlobStreamWriter,
)
from .errors import (
    BlobDownloadError,
    MaxStorageError,
    StorageClientError,
    StorageServiceError,
)
from .io.compress import CompressionCodec
from .models import (
    DataFormat,
    InstanceIdentifier,
    SessionStatus,
    SplitMode,
    Status,
    TableIdentifier,
    TimestampUnit,
    WriteMode,
)
from .options import BlobWriteItem, IncrementalReadOptions, SplitOptions

# Read path — the submodule handles optional pyarrow internally, so the only
# legitimate ImportError here is a missing pyarrow install.  Any other
# import failure (typo, broken transitive dependency) is a real bug and must
# propagate instead of being silently swallowed.  On a genuine pyarrow
# absence the names are set to None so __all__ stays consistent and users
# get a clear AttributeError at call time.
_read_names = [
    "ArrowReader",
    "ArrowRecordReader",
    "InstanceReadSession",
    "TableReadSession",
]
_read_split_names = ["IndexedInputSplit", "RowRangeInputSplit"]
try:
    from .read import (
        ArrowReader,
        ArrowRecordReader,
        IndexedInputSplit,
        InstanceReadSession,
        RowRangeInputSplit,
        TableReadSession,
    )
except ImportError as _e:
    _missing = getattr(_e, "name", None) or ""
    if _missing == "pyarrow" or _missing.startswith("pyarrow."):
        ArrowReader = ArrowRecordReader = InstanceReadSession = TableReadSession = None
        IndexedInputSplit = RowRangeInputSplit = None
    else:
        raise

# Write path — same rationale as the read path above.
_write_names = [
    "AppendTableRecordWriter",
    "DeltaTableRecordWriter",
    "TableArrowBlobUploadWriter",
    "TableArrowWriter",
    "TableWriteSession",
]
try:
    from .write import (
        AppendTableRecordWriter,
        DeltaTableRecordWriter,
        TableArrowBlobUploadWriter,
        TableArrowWriter,
        TableWriteSession,
    )
except ImportError as _e:
    _missing = getattr(_e, "name", None) or ""
    if _missing == "pyarrow" or _missing.startswith("pyarrow."):
        AppendTableRecordWriter = DeltaTableRecordWriter = None
        TableArrowBlobUploadWriter = TableArrowWriter = TableWriteSession = None
    else:
        raise

__all__ = (
    [
        "MaxStorageClient",
        "BlobManager",
        "BlobRecord",
        "BlobDataIterator",
        "BlobStreamReader",
        "BlobStreamWriter",
        "BlobWriteItem",
        "MaxStorageError",
        "StorageServiceError",
        "StorageClientError",
        "BlobDownloadError",
        "CompressionCodec",
        "DataFormat",
        "InstanceIdentifier",
        "SessionStatus",
        "SplitMode",
        "Status",
        "TableIdentifier",
        "TimestampUnit",
        "WriteMode",
        "SplitOptions",
        "IncrementalReadOptions",
    ]
    + _read_names
    + _read_split_names
    + _write_names
)
