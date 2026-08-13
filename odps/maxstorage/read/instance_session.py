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

"""Instance read session for :mod:`odps.maxstorage`.

Unlike the table read session, the instance session has no polling — it
returns immediately after create/get.  The session id wire field is
``DownloadID`` (not ``SessionId``); ``status`` is a raw string (not the
``SessionStatus`` enum); the schema is deserialized with the **WriteSchema**
deserializer.

Readers are opened with ``offset`` / ``count`` instead of splits.
"""

import warnings
from typing import TYPE_CHECKING, List, Optional

from ...tunnel.io.types import odps_schema_to_arrow_schema
from ...types import OdpsSchema
from ..io.compress import CompressionCodec, resolve_compress_option
from ..models.requests import (
    CreateInstanceReadSessionRequest,
    CreateInstanceReadStreamRequest,
)
from ..models.schema import WriteSchema
from .reader import ArrowReader

if TYPE_CHECKING:
    try:
        import pyarrow as pa
    except ImportError:
        pa = None  # type: ignore

    from ...tunnel.io.stream import CompressOption  # noqa: F401
    from ..models.identifier import InstanceIdentifier
    from ..models.schema import StorageSchema  # noqa: F401
    from ..stub import StorageStub


class InstanceReadSession:
    """An instance read session.

    Created by ``MaxStorageClient.create_instance_read_session``.  After
    construction the session is ready for reads; there is no split material
    and no polling.

    Parameters
    ----------
    stub : StorageStub
        The RPC layer.
    instance_id : InstanceIdentifier
        Target instance.
    request : CreateInstanceReadSessionRequest
        Creation request.
    session_id : str, optional
        If given, reload an existing session (``DownloadID``) instead of
        creating a new one.
    """

    def __init__(
        self,
        stub: "StorageStub",
        instance_id: "InstanceIdentifier",
        request: CreateInstanceReadSessionRequest,
        session_id: Optional[str] = None,
    ):
        self._stub = stub
        self._instance_id = instance_id

        if session_id is not None:
            response = stub.get_instance_read_session(instance_id, session_id)
        else:
            response = stub.create_instance_read_session(instance_id, request)

        self._response = response

    # -- public properties -------------------------------------------------

    @property
    def id(self) -> Optional[str]:
        """The download id (wire field ``DownloadID``, not ``SessionId``)."""
        return self._response.download_id

    @property
    def table_schema(self) -> "StorageSchema":
        """The :class:`WriteSchema` deserialized from ``TableSchema``."""
        return self._response.table_schema or WriteSchema()

    @property
    def arrow_schema(self) -> "pa.Schema":
        """Arrow schema over all columns."""
        schema = self.table_schema
        all_columns = list(schema.columns) if hasattr(schema, "columns") else []
        return odps_schema_to_arrow_schema(OdpsSchema(columns=all_columns))

    @property
    def record_count(self) -> int:
        """Total row count (from ``RecordCount``)."""
        return self._response.record_count or 0

    @property
    def quota_name(self) -> Optional[str]:
        """The quota name (from ``QuotaName``), or ``None``."""
        return self._response.quota_name

    # -- reader -----------------------------------------------------------

    def open_arrow_reader(
        self,
        offset: int = 0,
        count: Optional[int] = None,
        *,
        columns: Optional[List[str]] = None,
        task_name: str = "AnonymousSQLTask",
        query_id: int = 0,
        enable_limit: bool = False,
        compress_option: "CompressOption" = None,
        compress_algo=None,
        compress_level=None,
        async_read: bool = False,
        async_queue_size: int = 2,
    ) -> ArrowReader:
        """Open an :class:`ArrowReader` for this instance read session.

        ``task_name`` defaults to ``"AnonymousSQLTask"`` and warns when unset
        (i.e. left at the default) to surface the likely-missing task
        association.

        Compression follows the same tunnel pattern as ``TableReadSession``:
        ``compress_option`` (default ``None`` = uncompressed) or the shorthand
        ``compress_algo``/``compress_level``.  Uncompressed is the default;
        LZ4 and ZSTD are also supported.

        Example
        -------
        >>> instance = odps.execute_sql("SELECT * FROM my_table LIMIT 100")
        >>> session = client.create_instance_read_session(instance)
        >>> reader = session.open_arrow_reader(offset=0, count=100)
        >>> while True:
        ...     batch = reader.read()
        ...     if batch is None:
        ...         break
        ...     print(batch.to_pandas())
        >>> reader.close()
        """
        if task_name is None or task_name == "AnonymousSQLTask":
            warnings.warn(
                "Instance read session is using the default task name "
                "'AnonymousSQLTask'; pass an explicit task_name for clarity.",
                stacklevel=2,
            )

        compress_option = resolve_compress_option(
            compress_option, compress_algo, compress_level
        )

        request = CreateInstanceReadStreamRequest(
            task_name=task_name or "AnonymousSQLTask",
            query_id=query_id,
            enable_limit=enable_limit,
            columns=list(columns) if columns else [],
        )

        accept_encoding = None
        if compress_option is not None:
            codec = CompressionCodec.from_compress_option(compress_option)
            accept_encoding = codec.accept_encoding

        raw_response = self._stub.create_instance_read_stream(
            self._instance_id,
            self._response.download_id,
            count,
            offset,
            request,
            accept_encoding=accept_encoding,
        )

        request_id = ""
        if hasattr(raw_response, "headers"):
            request_id = raw_response.headers.get("x-odps-request-id", "")

        if count is not None:
            reader_count = count
        else:
            reader_count = max(self.record_count - offset, 0)

        return ArrowReader(
            raw_response,
            schema=self.table_schema,
            compress_option=compress_option,
            request_id=request_id,
            count=reader_count,
            async_read=async_read,
            async_queue_size=async_queue_size,
        )

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Close the session.  No-op (added for symmetry)."""
        return None
