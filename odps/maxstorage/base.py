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

"""Entry point for :mod:`odps.maxstorage`.

:class:`MaxStorageClient` is constructed from an :class:`odps.Odps` entry
(reuses ``RestClient`` + account) or standalone with explicit account/endpoint.
It resolves the tunnel endpoint, creates a dedicated ``RestClient`` tagged
``MAXSTORAGE``, and wraps a :class:`StorageStub`.
"""

import logging
import threading
from typing import TYPE_CHECKING, List, Optional

from ..config import options as config_options
from ..errors import ODPSError
from ..rest import RestClient
from ..tunnel.base import BaseTunnel
from .blob import BlobManager
from .models.enums import WriteMode
from .models.identifier import InstanceIdentifier, TableIdentifier
from .models.requests import (
    CreateInstanceReadSessionRequest,
    CreateTableReadSessionRequest,
    TablePreviewRequest,
)
from .options import _supports_v3
from .read.instance_session import InstanceReadSession
from .read.reader import ArrowReader
from .read.table_session import TableReadSession
from .stub import StorageStub
from .write.session import TableWriteSession

if TYPE_CHECKING:
    from ..accounts import BaseAccount
    from ..core import ODPS
    from ..models import Table  # noqa: F401
    from .models.enums import DataFormat
    from .options import IncrementalReadOptions, SplitOptions

logger = logging.getLogger(__name__)

DEFAULT_API_VERSION = "2"


class MaxStorageClient:
    """Entry point for the MaxCompute Storage API.

    Constructed from an :class:`odps.Odps` entry (reuses ``RestClient`` +
    account) or standalone with explicit account/endpoint.

    Parameters
    ----------
    odps : ODPS, optional
        The ODPS entry object.  If given, the account/endpoint/project are
        derived from it.
    account : Account, optional
        Explicit account (when ``odps`` is not given).
    endpoint : str, optional
        Storage/tunnel endpoint override.
    project : str, optional
        Default project.
    schema : str, optional
        Default schema.
    region : str, optional
        Region name for V4 signing.
    quota_name : str, optional
        Quota name for resource management.
    tunnel_endpoint : str, optional
        Tunnel endpoint override (bypasses discovery).
    user_agent : str, optional
        Custom user agent string.
    rest_client : RestClient, optional
        Pre-configured RestClient (bypasses endpoint discovery).
    api_version : str, default "2"
        API version: ``"2"`` → ``api/storage/v2``, ``"3"`` → ``api/storage/v3``.
    """

    def __init__(
        self,
        odps: Optional["ODPS"] = None,
        *,
        account: Optional["BaseAccount"] = None,
        endpoint: Optional[str] = None,
        project: Optional[str] = None,
        schema: Optional[str] = None,
        region: Optional[str] = None,
        quota_name: Optional[str] = None,
        tunnel_endpoint: Optional[str] = None,
        user_agent: Optional[str] = None,
        rest_client: Optional["RestClient"] = None,
        api_version: str = DEFAULT_API_VERSION,
    ):
        self._api_version = str(api_version)
        self._quota_name = quota_name

        if odps is not None:
            self._odps = odps
            self._rest_client = odps.rest
            self._account = odps.rest.account
            self._project = project or odps.project
            self._schema = schema or odps.schema
            self._region = region or odps.rest.region_name
            self._user_agent = user_agent or odps.rest._user_agent
            # Resolve tunnel endpoint
            self._tunnel_endpoint = tunnel_endpoint or getattr(
                odps, "_tunnel_endpoint", None
            )
        elif rest_client is not None:
            self._rest_client = rest_client
            self._account = rest_client.account
            self._project = project
            self._schema = schema
            self._region = region or rest_client.region_name
            self._user_agent = user_agent or rest_client._user_agent
            self._tunnel_endpoint = tunnel_endpoint or endpoint
            self._odps = None
        else:
            self._account = account
            self._project = project
            self._schema = schema
            self._region = region
            self._user_agent = user_agent
            self._tunnel_endpoint = tunnel_endpoint or endpoint
            self._rest_client = None
            self._odps = None

        self._storage_rest = None
        self._storage_rest_lock = threading.RLock()
        self._stub = None

    @property
    def api_version(self) -> str:
        """The negotiated API version (``"2"`` or ``"3"``)."""
        return self._api_version

    def _supports_v3(self) -> bool:
        """True when ``api_version >= 3``.  Gates v3-era features only."""
        return _supports_v3(self._api_version)

    def _resolve_tunnel_endpoint(self) -> Optional[str]:
        """Resolve the tunnel endpoint via discovery (same as BaseTunnel)."""
        if self._tunnel_endpoint:
            return self._tunnel_endpoint

        if self._odps is None:
            return None

        # Reuse BaseTunnel logic
        tunnel = BaseTunnel(
            self._odps,
            project=self._project,
            endpoint=self._tunnel_endpoint,
            quota_name=self._quota_name,
        )
        self._tunnel_endpoint = (
            tunnel._get_tunnel_server(tunnel._project)
            if tunnel._endpoint is None
            else tunnel._endpoint
        )
        return self._tunnel_endpoint

    @property
    def storage_rest(self) -> "RestClient":
        """The dedicated RestClient tagged ``MAXSTORAGE``."""
        if self._storage_rest is not None:
            return self._storage_rest

        with self._storage_rest_lock:
            if self._storage_rest is not None:
                return self._storage_rest

            endpoint = self._resolve_tunnel_endpoint()
            if endpoint is None:
                raise ODPSError(
                    "Cannot resolve storage endpoint. "
                    "Pass tunnel_endpoint= or odps= to MaxStorageClient."
                )

            kw = dict(tag="MAXSTORAGE")
            if self._odps is not None:
                kw["namespace"] = self._odps.namespace
            if config_options.data_proxy is not None:
                kw["proxy"] = config_options.data_proxy
            if self._rest_client and self._rest_client.app_account is not None:
                kw["app_account"] = self._rest_client.app_account

            self._storage_rest = RestClient(
                self._account,
                endpoint,
                self._project,
                self._schema,
                user_agent=self._user_agent,
                region_name=self._region,
                **kw,
            )
            return self._storage_rest

    @property
    def stub(self) -> StorageStub:
        """The :class:`StorageStub` for RPC operations."""
        if self._stub is None:
            self._stub = StorageStub(self.storage_rest, self._api_version)
        return self._stub

    # ---- Public API ----

    def open_blob_manager(self, table: Optional["Table"] = None) -> BlobManager:
        """Return a :class:`BlobManager` for standalone blob read/write.

        ``table`` is optional — needed for write operations
        (``write_blob_stream``, ``write_blob_batch``) to resolve the table
        identifier.  Read operations (``read_blobs``, ``read_blob``) do not
        need it.

        Parameters
        ----------
        table : str or Table, optional
            Table name or :class:`odps.models.Table`.  Required for write
            operations; ignored for reads.

        Example
        -------
        >>> blob_manager = client.open_blob_manager("my_table")
        >>> records = blob_manager.read_blobs([ref1, ref2])
        >>> for record in records:
        ...     print(len(record.data))
        """
        table_id = self._resolve_table_id(table) if table is not None else None
        # Always return a fresh BlobManager.  Caching a single instance and
        # mutating its _table_id would silently retarget any previously
        # returned reference still held by the caller.
        return BlobManager(self, table_id)

    def _resolve_table_id(self, table) -> TableIdentifier:
        """Build a :class:`TableIdentifier` from a Table or string."""
        if isinstance(table, TableIdentifier):
            return table
        if hasattr(table, "name"):
            return TableIdentifier.from_table(table)
        # Assume (project, table) or (project, table, schema) tuple.
        # Note: a 2-tuple does NOT inherit the client's default schema —
        # pass a 3-tuple or use a string table name to get the default.
        if isinstance(table, (list, tuple)):
            return TableIdentifier(*table)
        # String — need project context
        project = self._project
        if project is None:
            raise ValueError(
                "Cannot resolve table identifier from a string table name "
                "without a default project; pass project= to MaxStorageClient "
                "or use a (project, table[, schema]) tuple."
            )
        return TableIdentifier(project, table, self._schema)

    def _resolve_instance_id(self, instance) -> InstanceIdentifier:
        """Build an :class:`InstanceIdentifier` from an Instance or string."""
        if isinstance(instance, InstanceIdentifier):
            return instance
        if hasattr(instance, "id"):
            return InstanceIdentifier.from_instance(instance)
        project = self._project
        return InstanceIdentifier(project, instance)

    def preview_table(
        self,
        table,
        partition_spec: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> ArrowReader:
        """Preview initial rows of a table as an :class:`ArrowReader`.

        Uncompressed (preview sends no ``ACCEPT-ENCODING`` header); for
        compressed reads use ``create_table_read_session`` + ``open_arrow_reader``.

        Parameters
        ----------
        table : str or Table
            Table name or :class:`odps.models.Table`.
        partition_spec : str, optional
            Partition spec, e.g. ``"pt=20230101"``.
        columns : list[str], optional
            Columns to project; ``None`` returns all.
        limit : int, optional
            Maximum rows to preview.

        Example
        -------
        >>> reader = client.preview_table("my_table", limit=10)
        >>> while True:
        ...     batch = reader.read()
        ...     if batch is None:
        ...         break
        ...     print(batch.to_pandas())
        >>> reader.close()
        """

        table_id = self._resolve_table_id(table)
        request = TablePreviewRequest(
            limit=limit,
            partition=partition_spec,
            columns=columns or [],
        )
        resp = self.stub.preview(table_id, request)
        return ArrowReader(resp, schema=None)

    def create_table_read_session(
        self,
        table,
        *,
        session_id: Optional[str] = None,
        columns: Optional[List[str]] = None,
        partition_columns: Optional[List[str]] = None,
        partitions: Optional[List[str]] = None,
        bucket_ids: Optional[List[int]] = None,
        split_options: Optional["SplitOptions"] = None,
        arrow_options: Optional[dict] = None,
        filter_predicate: Optional[str] = None,
        filter_predicate_fallback: bool = False,
        split_max_file_num: int = 0,
        incremental_read_options: Optional["IncrementalReadOptions"] = None,
        incremental_read_enabled: bool = False,
        session_ready_timeout: Optional[int] = None,
    ) -> TableReadSession:
        """Create (or reload via ``session_id``) a table read session.

        Polls until ``NORMAL`` (1s interval, ``session_ready_timeout`` default
        3600s).

        Parameters
        ----------
        table : str or Table
            Table name or :class:`odps.models.Table`.
        session_id : str, optional
            Reload an existing session instead of creating a new one.
        columns : list[str], optional
            Data columns to project.
        partition_columns : list[str], optional
            Partition key columns to project (read partition values as data).
        partitions : list[str], optional
            Partition specs to filter, e.g. ``["pt=20230101"]``.
        bucket_ids : list[int], optional
            Restrict reads to specific buckets of a bucketed table.
        split_options : SplitOptions, optional
            Split strategy (size, row-offset, or parallelism).
        arrow_options : ArrowOptions, optional
            Arrow read options (e.g. timestamp unit overrides).
        filter_predicate : str, optional
            Server-side filter predicate (SQL expression).
        filter_predicate_fallback : bool, default False
            Fall back to client-side filtering when the server rejects
            ``filter_predicate``.
        split_max_file_num : int, default 0
            Maximum number of files per split (0 = server default).
        incremental_read_options : IncrementalReadOptions, optional
            Incremental-read configuration (mode + version range).
        incremental_read_enabled : bool, default False
            Enable incremental read for this session.
        session_ready_timeout : int, optional
            Override the default 3600s poll timeout.

        Example
        -------
        >>> read_session = client.create_table_read_session("my_table")
        >>> for split in read_session.splits:
        ...     reader = read_session.open_arrow_reader(split)
        ...     while True:
        ...         batch = reader.read()
        ...         if batch is None:
        ...             break
        ...         print(batch.to_pandas())
        ...     reader.close()
        """

        table_id = self._resolve_table_id(table)
        timeout = (
            session_ready_timeout or config_options.maxstorage.session_ready_timeout
        )
        request = CreateTableReadSessionRequest(
            required_data_columns=columns or [],
            required_partition_columns=partition_columns or [],
            required_partitions=partitions or [],
            required_bucket_ids=bucket_ids or [],
            split_options=split_options,
            arrow_options=arrow_options,
            filter_predicate=filter_predicate or "",
            filter_predicate_fallback=filter_predicate_fallback,
            split_max_file_num=split_max_file_num,
            incremental_read=incremental_read_enabled,
            incremental_read_options=incremental_read_options,
        )
        return TableReadSession(
            self.stub,
            table_id,
            request,
            session_id=session_id,
            session_ready_timeout=timeout,
        )

    def create_table_write_session(
        self,
        table,
        *,
        session_id: Optional[str] = None,
        partition_spec: Optional[str] = None,
        overwrite: bool = False,
        write_mode: WriteMode = WriteMode.BATCH,
        quota_name: Optional[str] = None,
        enable_schema_evolution: bool = False,
        required_data_format: Optional["DataFormat"] = None,
    ) -> TableWriteSession:
        """Create (or reload via ``session_id``) a table write session.

        Parameters
        ----------
        table : str or Table
            Table name or :class:`odps.models.Table`.
        session_id : str, optional
            Reload an existing session instead of creating a new one.
        partition_spec : str, optional
            Partition to write, e.g. ``"pt=20230101"``.
        overwrite : bool, default False
            Overwrite existing data in the partition.
        write_mode : WriteMode, default WriteMode.BATCH
            Batch or streaming write mode.
        required_data_format : DataFormat, optional
            Required data format for the session (e.g. ``DataFormat("Arrow", "V5")``).
            When ``None`` the server applies its default. Forwarded as the
            ``RequiredDataFormat`` wire field consumed by the server.

        Example
        -------
        >>> import pyarrow as pa
        >>> write_session = client.create_table_write_session("my_table")
        >>> writer = write_session.open_arrow_writer(stream_id="0")
        >>> batch = pa.RecordBatch.from_arrays(
        ...     [pa.array([1, 2], pa.int64()), pa.array(["a", "b"], pa.string())],
        ...     schema=pa.schema([("id", pa.int64()), ("name", pa.string())]),
        ... )
        >>> writer.write_batch(batch)
        >>> writer.close()
        >>> write_session.commit()
        """

        table_id = self._resolve_table_id(table)
        return TableWriteSession(
            self.stub,
            table_id,
            session_id=session_id,
            partition_spec=partition_spec,
            overwrite=overwrite,
            write_mode=write_mode,
            quota_name=quota_name,
            enable_schema_evolution=enable_schema_evolution,
            required_data_format=required_data_format,
            api_version=self._api_version,
        )

    def create_instance_read_session(
        self,
        instance,
        *,
        session_id: Optional[str] = None,
        enable_limit: bool = False,
    ) -> InstanceReadSession:
        """Create (or reload via ``session_id``) an instance read session.

        Parameters
        ----------
        instance : Instance or str
            :class:`odps.models.Instance` or instance ID.
        session_id : str, optional
            Reload an existing session instead of creating a new one.
        enable_limit : bool, default False
            Allow ``LIMIT`` pushdown on the instance.

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

        instance_id = self._resolve_instance_id(instance)
        request = CreateInstanceReadSessionRequest(enable_limit=enable_limit)
        return InstanceReadSession(
            self.stub,
            instance_id,
            request,
            session_id=session_id,
        )
