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

"""Client for interacting with the MaxCompute Storage API V2."""

from typing import List, Union

try:
    import pyarrow as pa
except ImportError:
    pa = None

from ... import ODPS, options
from ...models import Instance, Table
from ...tunnel.io.stream import CompressOption, RequestsIO, get_decompress_stream
from ...tunnel.tabletunnel import TableTunnel
from .models import *  # noqa: F401 F403
from .models import (
    BlobWriteItem,
    CloseWriteStreamRequest,
    CloseWriteStreamResponse,
    Compression,
    CreateReadSessionRequest,
    CreateReadSessionResponse,
    CreateWriteSessionRequest,
    CreateWriteSessionResponse,
    CreateWriteStreamRequest,
    CreateWriteStreamResponse,
    DataFormat,
    GetWriteSessionResponse,
    GetWriteStreamResponse,
    PreviewTableRequest,
    ReadBlobRequest,
    ReadStreamRequest,
    WriteBlobRequest,
    WriteBlobResponse,
    WriteStreamRequest,
    _parse_json_response,
    _update_request_id,
)
from .stream_io import (
    ArrowReader,
    ArrowWriter,
    BlobDataIterator,
    BlobStreamReader,
    BlobStreamWriter,
    StreamReader,
    StreamWriter,
)

STORAGE_V2_VERSION = "2"
URL_PREFIX_V2 = "/api/storage/v" + STORAGE_V2_VERSION
ROUTE_TOKEN_HEADER = "x-odps-max-storage-route-token"
WRITE_ACCESS_TOKEN_HEADER = "x-odps-max-storage-write-access-token"


class StorageApiClient:
    """Client for the MaxCompute Storage API V2."""

    def __init__(
        self,
        odps: ODPS,
        table_or_instance=None,
        rest_endpoint: str = None,
        quota_name: str = None,
        tags: Union[None, str, List[str]] = None,
    ):
        if not isinstance(odps, ODPS):
            raise ValueError("Please input odps configuration")
        if table_or_instance is None:
            raise ValueError("Please input table or instance")
        if isinstance(table_or_instance, Instance):
            instance, table = table_or_instance, None
        elif isinstance(table_or_instance, Table):
            instance, table = None, table_or_instance
        else:
            raise ValueError("Please input valid table or instance")

        self._odps = odps
        self._table = table
        self._instance = instance
        self._quota_name = quota_name
        self._rest_endpoint = rest_endpoint
        self._tunnel_rest = None
        self._route_token = None

        self._tags = tags or options.tunnel.tags
        if isinstance(self._tags, str):
            self._tags = self._tags.split(",")

    @property
    def table(self):
        return self._table

    @property
    def route_token(self):
        """The stored route token for session affinity.

        This is automatically updated from response headers. Pass
        ``route_token=...`` to individual methods to override, or
        rely on this stored value when not specified.
        """
        return self._route_token

    @property
    def tunnel_rest(self):
        if self._tunnel_rest is not None:
            return self._tunnel_rest

        tunnel = TableTunnel(
            self._odps, endpoint=self._rest_endpoint, quota_name=self._quota_name
        )
        self._tunnel_rest = tunnel.tunnel_rest
        return self._tunnel_rest

    # ---- internal helpers ----

    def _resolve_route_token(self, route_token=None):
        """Return the effective route token: explicit arg overrides stored value."""
        return route_token or self._route_token

    def _update_route_token(self, route_token):
        """Update the stored route token if a new one is provided."""
        if route_token:
            self._route_token = route_token

    def _check_not_instance(self):
        if self._instance is not None:
            raise ValueError(
                "Write operations are not supported for instance-based client"
            )

    def _build_target(self) -> str:
        if self._instance is not None:
            inst = self._instance
            project = (
                inst.project.name
                if hasattr(inst, "project") and hasattr(inst.project, "name")
                else inst.project
            )
            return f"projects.{project}.instances.{inst.id}"
        t = self._table
        project = t.project.name if hasattr(t.project, "name") else t.project
        schema = t._get_schema_name() if hasattr(t, "_get_schema_name") else None
        schema = schema or "default"
        return f"projects.{project}.schemas.{schema}.tables.{t.name}"

    def _get_v2_url(self) -> str:
        endpoint = self.tunnel_rest.endpoint + URL_PREFIX_V2
        return endpoint

    def _fill_common_headers(self, raw_headers=None, route_token=None):
        headers = raw_headers or {}
        headers.setdefault("Content-Type", "application/json; charset=utf-8")
        if self._tags:
            headers["odps-tunnel-tags"] = ",".join(self._tags)
        if route_token:
            headers[ROUTE_TOKEN_HEADER] = route_token
        return headers

    @staticmethod
    def _compression_to_compress_algo(compression):
        """Map a Compression enum value to a CompressOption.CompressAlgorithm.

        Returns None for Compression.UNCOMPRESSED.
        """
        if compression == Compression.UNCOMPRESSED:
            return None
        elif compression == Compression.ZSTD:
            return CompressOption.CompressAlgorithm.ODPS_ZSTD
        elif compression == Compression.LZ4_FRAME:
            return CompressOption.CompressAlgorithm.ODPS_LZ4
        else:
            raise ValueError(
                f"Unsupported compression type: {compression}. "
                f"Supported values are Compression.ZSTD, Compression.LZ4_FRAME, "
                f"or Compression.UNCOMPRESSED."
            )

    def _request(self, action, extra_params=None, body=None, route_token=None):
        """Issue a JSON POST to the V2 endpoint and return parsed JSON dict + raw response."""
        url = self._get_v2_url()
        params = {"Action": action, "Target": self._build_target()}
        if self._quota_name:
            params["quotaName"] = self._quota_name
        if extra_params:
            params.update(extra_params)
        headers = self._fill_common_headers(route_token=route_token)
        data = body if body is not None else "{}"

        resp = self.tunnel_rest.post(url, data=data, params=params, headers=headers)
        resp_json = _parse_json_response(resp)
        return resp_json, resp

    # ---- Read Session ----

    def create_read_session(
        self,
        required_data_columns=None,
        required_partition_columns=None,
        required_partitions=None,
        required_bucket_ids=None,
        split_options=None,
        arrow_options=None,
        filter_predicate=None,
        filter_predicate_fallback=None,
        split_max_file_num=None,
        incremental_read=None,
        incremental_read_options=None,
    ) -> CreateReadSessionResponse:
        """
        Create a read session for table or instance data retrieval.

        A read session is a prerequisite for reading data from a MaxCompute table
        or SQL instance result. The session determines how data will be split into
        readable chunks and what data schema will be returned. Sessions have an
        expiration time and must be refreshed if they expire during long-running
        read operations.

        Parameters
        ----------
        required_data_columns : list of str, optional
            List of column names to read. If empty, all columns are returned.
        required_partition_columns : list of str, optional
            Partition columns to include in the result.
        required_partitions : list of str, optional
            Specific partition values to read (e.g., ['pt=20230101']).
        required_bucket_ids : list of str, optional
            Bucket IDs to read for bucket-based tables.
        split_options : SplitOptions, optional
            Controls how data is split into chunks. Defaults to size-based
            splitting with 256MB chunks.
        arrow_options : ArrowOptions, optional
            Arrow format settings like timestamp precision.
        filter_predicate : str, optional
            SQL-like filter condition to apply during reading.
        filter_predicate_fallback : bool, optional
            Whether to fallback to server-side filtering if predicate pushdown fails.
        split_max_file_num : int, optional
            Maximum number of files per split for file-based splitting.
        incremental_read : bool, optional
            Enable incremental reading mode for capturing table changes.
        incremental_read_options : IncrementalReadOptions, optional
            Options for incremental read mode (version range, timestamp range).

        Returns
        -------
        CreateReadSessionResponse
            Response with session_id, session_status, splits_count, record_count,
            data_schema, and expiration_time. See :class:`CreateReadSessionResponse`.

        Raises
        ------
        ValueError
            If the table or instance is not properly configured.

        See Also
        --------
        get_read_session : Get current read session status.
        read_rows_stream : Read data from a specific split.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient, SplitOptions
        ... )
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Create a basic read session that reads all columns with default splitting:

        >>> response = client.create_read_session()
        >>> print(f"Session ID: {response.session_id}")
        Session ID: session_12345
        >>> print(f"Available splits: {response.splits_count}")
        Available splits: 5

        Specify which columns to read and how to split the data. Use
        ``split_options`` to control parallelism by setting the split number
        instead of split size:

        >>> split_opts = SplitOptions()
        >>> split_opts.split_mode = SplitOptions.SplitMode.PARALLELISM
        >>> split_opts.split_number = 10  # Create 10 splits
        >>> response = client.create_read_session(
        ...     required_data_columns=["id", "name", "value"],
        ...     split_options=split_opts,
        ... )
        >>> print(f"Created {response.splits_count} splits")
        Created 10 splits

        Read from a specific partition by providing partition values in the
        ``required_partitions`` parameter:

        >>> response = client.create_read_session(
        ...     required_partitions=["pt=20230101", "region=us-west"]
        ... )
        >>> print(f"Records in partition: {response.record_count}")
        Records in partition: 1000

        For incremental data capture, enable ``incremental_read`` to read
        changes since a specific version or timestamp:

        >>> from odps.apis.storage_api_v2 import IncrementalReadOptions
        >>> incr_opts = IncrementalReadOptions()
        >>> incr_opts.start_version = 100
        >>> response = client.create_read_session(
        ...     incremental_read=True,
        ...     incremental_read_options=incr_opts,
        ... )
        >>> print(f"Latest version: {response.latest_version}")
        Latest version: 150
        """
        request = CreateReadSessionRequest(
            required_data_columns=required_data_columns,
            required_partition_columns=required_partition_columns,
            required_partitions=required_partitions,
            required_bucket_ids=required_bucket_ids,
            split_options=split_options,
            arrow_options=arrow_options,
            filter_predicate=filter_predicate,
            filter_predicate_fallback=filter_predicate_fallback,
            split_max_file_num=split_max_file_num,
            incremental_read=incremental_read,
            incremental_read_options=incremental_read_options,
        )

        action = (
            "InstanceCreateReadSession"
            if self._instance is not None
            else "TableCreateReadSession"
        )
        json_str = request.serialize()
        resp_json, resp = self._request(action, body=json_str)

        response = CreateReadSessionResponse()
        response.parse(resp, obj=response)
        _update_request_id(response, resp)
        route_token = resp.headers.get(ROUTE_TOKEN_HEADER)
        if route_token:
            response.route_token = route_token
            self._update_route_token(route_token)
        if self._instance is not None:
            if response.session_id is None and response.download_id is not None:
                response.session_id = response.download_id
            if response.session_status is None and response.status is not None:
                response.session_status = response.status

        return response

    def get_read_session(
        self, session_id: str, refresh: bool = False
    ) -> CreateReadSessionResponse:
        """
        Get current status and metadata of an existing read session.

        This method retrieves the current state of a read session, including
        its status, split count, schema, and expiration time. Use this to
        check if a session is still valid before attempting to read data,
        or to refresh a session that is approaching expiration.

        Parameters
        ----------
        session_id : str
            The unique identifier of the read session to retrieve.
            This is obtained from create_read_session response.
        refresh : bool, default False
            Whether to refresh the session expiration time. Set to True
            if the session is about to expire or has expired. When True,
            the server extends the session lifetime, allowing continued
            reading operations.

        Returns
        -------
        CreateReadSessionResponse
            Response with session_id, session_status, splits_count, record_count,
            data_schema, and expiration_time. See :class:`CreateReadSessionResponse`.

        Raises
        ------
        ValueError
            If session_id is None or empty.

        See Also
        --------
        create_read_session : Create a new read session.
        read_rows_stream : Read data from a specific split.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import StorageApiClient
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        First create a read session, then check its status:

        >>> create_response = client.create_read_session()
        >>> session_id = create_response.session_id
        >>> status_response = client.get_read_session(session_id)
        >>> print(f"Session status: {status_response.session_status}")
        Session status: NORMAL
        >>> print(f"Expiration: {status_response.expiration_time}")
        Expiration: 2023-01-01T12:00:00Z

        When a session expires during a long-running read operation,
        use the ``refresh`` parameter to extend its lifetime:

        >>> # Check if session expired
        >>> response = client.get_read_session(session_id)
        >>> if response.session_status == SessionStatus.EXPIRED:
        ...     # Refresh the session to continue reading
        ...     refreshed_response = client.get_read_session(session_id, refresh=True)
        ...     print(f"New expiration: {refreshed_response.expiration_time}")
        New expiration: 2023-01-01T14:00:00Z

        Monitor session status during a parallel read operation to
        ensure all splits are processed before expiration:

        >>> create_response = client.create_read_session()
        >>> for split_index in range(create_response.splits_count):
        ...     # Check session health before each split
        ...     status = client.get_read_session(create_response.session_id)
        ...     if status.session_status == SessionStatus.EXPIRED:
        ...         client.get_read_session(create_response.session_id, refresh=True)
        ...     # Read the split data...
        ...     reader = client.read_rows_stream(...)
        """
        action = (
            "InstanceGetReadSession"
            if self._instance is not None
            else "TableGetReadSession"
        )
        extra_params = {"SessionId": session_id}
        if refresh:
            extra_params["session_refresh"] = "true"

        resp_json, resp = self._request(action, extra_params=extra_params, body="{}")

        response = CreateReadSessionResponse()
        response.parse(resp, obj=response)
        _update_request_id(response, resp)
        route_token = resp.headers.get(ROUTE_TOKEN_HEADER)
        if route_token:
            response.route_token = route_token
            self._update_route_token(route_token)

        # Normalize instance response fields
        if self._instance is not None:
            if response.session_id is None and response.download_id is not None:
                response.session_id = response.download_id
            if response.session_status is None and response.status is not None:
                response.session_status = response.status

        return response

    def read_rows_stream(
        self,
        session_id=None,
        split_index=None,
        row_offset=None,
        row_count=None,
        max_batch_rows=4096,
        skip_row_num=0,
        max_batch_raw_size=0,
        data_format=None,
        data_columns=None,
        compression=None,
        route_token=None,
    ) -> StreamReader:
        """
        Read data from a specific split in a read session.

        This method reads a chunk of data from a table or instance result by
        specifying which split to read. The data is returned as a stream that
        can be processed incrementally, supporting efficient handling of large
        datasets. Each split can be read independently, enabling parallel
        processing of the same session across multiple workers.

        Parameters
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
        max_batch_rows : int, default 4096
            Maximum rows per Arrow batch in the stream. Controls
            memory usage during reading.
        skip_row_num : int, default 0
            Number of rows to skip before reading.
        max_batch_raw_size : int, default 0
            Maximum raw byte size per batch. 0 means no limit.
        data_format : DataFormat, optional
            Format of returned data (Arrow V5 is default).
        data_columns : list of str, optional
            Specific columns to read. Must match session schema.
        compression : Compression, default None
            Compression algorithm for the stream data. None means
            Compression.UNCOMPRESSED.

        Returns
        -------
        StreamReader
            Stream reader with read(), get_status(), get_request_id(),
            and close() methods. See :class:`StreamReader`.

        Raises
        ------
        ValueError
            If the session or split parameters are invalid.

        See Also
        --------
        create_read_session : Create a read session first.
        ArrowReader : Wrap StreamReader to read Arrow batches.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient, ArrowReader
        ... )
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Read data from a specific split using ArrowReader for convenient
        batch processing:

        >>> reader = client.read_rows_stream(session_id, split_index=0)
        >>> arrow_reader = ArrowReader(reader)
        >>> while True:
        ...     batch = arrow_reader.read()
        ...     if batch is None:
        ...         break
        ...     df = batch.to_pandas()
        ...     # Process dataframe
        >>> arrow_reader.get_request_id()  # Get request ID after completion

        For parallel processing, distribute splits across multiple workers.
        Each worker reads a different split index from the same session:

        >>> # Worker 1 reads split 0
        >>> reader1 = client.read_rows_stream(session_id, split_index=0)
        >>> # Worker 2 reads split 1
        >>> reader2 = client.read_rows_stream(session_id, split_index=1)

        Control memory usage by limiting batch size with ``max_batch_rows``.
        Smaller batches reduce memory footprint but may have lower throughput:

        >>> reader = client.read_rows_stream(session_id, split_index=0, max_batch_rows=1024)
        >>> arrow_reader = ArrowReader(reader)
        >>> # Read batches one at a time to control memory
        >>> while True:
        ...     batch = arrow_reader.read()
        ...     if batch is None:
        ...         break
        ...     df = batch.to_pandas()
        ...     # Process df and then discard to free memory

        Read a specific range of rows within a split by using ``row_offset``
        and ``row_count`` parameters:

        >>> reader = client.read_rows_stream(
        ...     session_id,
        ...     split_index=0,
        ...     row_offset=1000,  # Skip first 1000 rows
        ...     row_count=500,    # Read 500 rows
        ... )
        """
        request = ReadStreamRequest(
            session_id=session_id,
            split_index=split_index,
            row_offset=row_offset,
            row_count=row_count,
            max_batch_rows=max_batch_rows,
            skip_row_num=skip_row_num,
            max_batch_raw_size=max_batch_raw_size,
            data_format=data_format,
            data_columns=data_columns,
            compression=compression,
        )

        action = "InstanceRead" if self._instance is not None else "TableRead"
        url = self._get_v2_url()
        params = {
            "Action": action,
            "Target": self._build_target(),
            "SessionId": request.session_id,
        }
        if request.split_index is not None:
            params["Index"] = str(request.split_index)
        if request.row_offset is not None:
            params["Offset"] = str(request.row_offset)
        if request.row_count is not None and request.row_count > 0:
            params["Count"] = str(request.row_count)
        if self._quota_name:
            params["quotaName"] = self._quota_name

        data_format = request.data_format or DataFormat()
        body_dict = {
            "MaxBatchRows": request.max_batch_rows,
            "SkipRowNum": request.skip_row_num,
            "MaxBatchRawSize": request.max_batch_raw_size,
            "DataFormat": {
                "Type": data_format.type,
                "Version": data_format.version,
            },
            "DataColumns": request.data_columns or [],
            "DataColumnsUnordered": False,
        }

        body = json.dumps(body_dict)

        extra_headers = {}
        compress_algo = self._compression_to_compress_algo(request.compression)
        if compress_algo is not None:
            encoding = compress_algo.get_encoding()
            if encoding:
                extra_headers["ACCEPT-ENCODING"] = encoding

        headers = self._fill_common_headers(
            extra_headers,
            route_token=self._resolve_route_token(route_token),
        )

        def download():
            return self.tunnel_rest.post(
                url, data=body, stream=True, params=params, headers=headers
            )

        return StreamReader(download)

    def preview_table(self, limit=None, partition=None, columns=None) -> StreamReader:
        """
        Preview table data without creating a session.

        This method provides a lightweight way to sample table data without
        the overhead of creating a read session. Unlike read_rows_stream,
        preview_table directly returns an Arrow IPC stream, making it ideal
        for quick data exploration, schema inspection, or testing table
        connectivity.

        Parameters
        ----------
        limit : int, optional
            Maximum number of rows to preview. If None, returns a
            small default sample (typically 100-1000 rows).
        partition : str, optional
            Partition specification to preview specific partition data.
            Format: 'pt=20230101' or 'pt=20230101,region=us-west'.
        columns : list of str, optional
            Specific columns to preview. If empty, all columns are returned.

        Returns
        -------
        StreamReader
            Stream reader with read(), get_status(), get_request_id(),
            and close() methods. See :class:`StreamReader`.

        Raises
        ------
        ValueError
            If called on an instance-based client (preview only works on tables).

        See Also
        --------
        read_rows_stream : Read full table data with session management.
        create_read_session : Create a session for large-scale reading.

        Notes
        -----
        Preview is optimized for quick sampling and may not return exact
        row counts specified in limit. For production data reading with
        guaranteed row counts and retry support, use create_read_session
        and read_rows_stream instead.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient, ArrowReader
        ... )
        >>> import pyarrow as pa
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Preview the first few rows of a table to quickly explore its data:

        >>> stream_reader = client.preview_table(limit=10)
        >>> arrow_reader = ArrowReader(stream_reader)
        >>> batch = arrow_reader.read()
        >>> if batch is not None:
        ...     df = batch.to_pandas()
        ...     print(df)
           id    name    value
        0   1  Alice      100
        1   2    Bob      200
        2   3  Carol      150

        Preview specific columns to check their data types and values
        without downloading all columns:

        >>> reader = ArrowReader(client.preview_table(
        ...     limit=5, columns=["id", "name"]
        ... ))
        >>> batch = reader.read()
        >>> if batch is not None:
        ...     print(batch.schema)
        id: int64
        name: string

        Preview data from a specific partition to test partition filtering
        before running a full query:

        >>> reader = ArrowReader(client.preview_table(
        ...     limit=20, partition="pt=20230101"
        ... ))
        >>> batch = reader.read()
        >>> df = batch.to_pandas() if batch is not None else None
        >>> print(f"Previewed {len(df) if df is not None else 0} rows")
        Previewed 20 rows

        Use preview to inspect table schema by requesting a small sample.
        This is useful for understanding column names and types before
        creating a read session:

        >>> reader = ArrowReader(client.preview_table(limit=1))
        >>> batch = reader.read()
        >>> if batch is not None:
        ...     for field in batch.schema:
        ...         print(f"{field.name}: {field.type}")
        id: int64
        name: string
        value: double
        pt: string
        """
        request = PreviewTableRequest(limit=limit, partition=partition, columns=columns)

        self._check_not_instance()

        url = self._get_v2_url()
        params = {
            "Action": "TablePreview",
            "Target": self._build_target(),
        }
        if request.limit is not None:
            params["Limit"] = str(request.limit)
        if request.partition is not None:
            params["Partition"] = request.partition
        if self._quota_name:
            params["quotaName"] = self._quota_name

        body_dict = {}
        if request.limit is not None:
            body_dict["Limit"] = request.limit
        if request.partition is not None:
            body_dict["Partition"] = request.partition
        if request.columns:
            body_dict["Columns"] = request.columns

        body = json.dumps(body_dict)
        headers = self._fill_common_headers()

        def download():
            return self.tunnel_rest.post(
                url, data=body, stream=True, params=params, headers=headers
            )

        return StreamReader(download)

    # ---- Write Session ----

    def create_write_session(
        self, partial_partition_spec=None, flags=None
    ) -> CreateWriteSessionResponse:
        """
        Create a write session for uploading data to a table.

        A write session is the first step in the data upload workflow. It
        establishes a transactional context for writing data to a MaxCompute
        table, ensuring atomicity and consistency. After creating a session,
        you must create one or more write streams within it, upload data to
        those streams, close the streams, and finally commit the session.

        Parameters
        ----------
        partial_partition_spec : str, optional
            Partition specification for writing to a specific partition.
            Format: 'pt=20230101' or 'pt=20230101,region=us-west'.
            If empty, writes to the table's default location.
        flags : dict, optional
            Additional flags for session configuration. Common flags
            include 'overwrite' to replace existing partition data.

        Returns
        -------
        CreateWriteSessionResponse
            Response with session_id, warning_message, and request_id.
            See :class:`CreateWriteSessionResponse`.

        Raises
        ------
        ValueError
            If called on an instance-based client (writes only support tables).

        See Also
        --------
        create_write_stream : Create a stream within the session.
        commit_write_session : Commit the session to finalize writes.
        abort_write_session : Abort the session to discard all writes.

        Notes
        -----
        Write sessions are transactional. All writes within a session are
        committed atomically when commit_write_session is called. If the
        session is aborted, all uploaded data is discarded. Sessions have
        a limited lifetime and must be committed before expiration.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import StorageApiClient
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Create a basic write session for uploading data to the table's
        default location:

        >>> response = client.create_write_session()
        >>> session_id = response.session_id
        >>> print(f"Write session ID: {session_id}")
        Write session ID: write_session_12345

        Write to a specific partition by specifying the partition spec.
        This is required for partitioned tables unless writing to the
        default partition:

        >>> response = client.create_write_session(
        ...     partial_partition_spec="pt=20230101"
        ... )
        >>> session_id = response.session_id
        >>> print(f"Writing to partition pt=20230101")

        Use the 'overwrite' flag to replace existing data in a partition
        instead of appending:

        >>> response = client.create_write_session(
        ...     partial_partition_spec="pt=20230101",
        ...     flags={"overwrite": True}
        ... )
        >>> session_id = response.session_id
        >>> # After committing, existing partition data will be replaced

        Complete write workflow: create session, create stream, write data,
        close stream, and commit:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> # Create write stream (see create_write_stream example)
        >>> # Write data (see write_rows_stream example)
        >>> # Close stream (see close_write_stream example)
        >>> # Commit session to finalize
        >>> client.commit_write_session(session_id)

        If an error occurs during the write process, abort the session
        to discard all uploaded data and release resources:

        >>> session_resp = client.create_write_session()
        >>> try:
        ...     # Write data...
        ...     # If error occurs:
        ...     client.abort_write_session(session_resp.session_id)
        ... except Exception as e:
        ...     client.abort_write_session(session_resp.session_id)
        ...     raise
        """
        self._check_not_instance()
        request = CreateWriteSessionRequest(
            partial_partition_spec=partial_partition_spec,
            flags=flags,
        )

        json_str = request.serialize()
        resp_json, resp = self._request("TableCreateWriteSession", body=json_str)

        response = CreateWriteSessionResponse()
        response.parse(resp, obj=response)
        _update_request_id(response, resp)
        route_token = resp.headers.get(ROUTE_TOKEN_HEADER)
        if route_token:
            response.route_token = route_token
            self._update_route_token(route_token)

        if response.warning_message:
            logger.warning(response.warning_message)

        return response

    def get_write_session(self, session_id: str) -> GetWriteSessionResponse:
        """
        Get the current status and stream information of a write session.

        This method retrieves metadata about a write session, including the
        list of active write streams and their states. Use this to monitor
        the progress of a multi-stream upload operation or to verify that
        all streams have been properly closed before committing the session.

        Parameters
        ----------
        session_id : str
            The unique identifier of the write session to query.
            This is obtained from create_write_session response.

        Returns
        -------
        GetWriteSessionResponse
            Response with streams (list of stream info dicts), warning_message,
            and request_id. See :class:`GetWriteSessionResponse`.

        Raises
        ------
        ValueError
            If session_id is None or empty, or if called on an
            instance-based client.

        See Also
        --------
        create_write_session : Create a write session.
        commit_write_session : Commit the session after all streams closed.
        create_write_stream : Create a new stream in the session.

        Notes
        -----
        Before calling commit_write_session, ensure all write streams are
        closed. Use get_write_session to verify that no streams are still
        in the RUNNING or OPEN state.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient
        ... )
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Create a write session and check its status before creating streams:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> status_resp = client.get_write_session(session_id)
        >>> print(f"Active streams: {status_resp.streams}")
        Active streams: None

        After creating multiple write streams for parallel upload, use
        get_write_session to track which streams are active:

        >>> # Create multiple streams (see create_write_stream examples)
        >>> stream_resp1 = client.create_write_stream(...)
        >>> stream_resp2 = client.create_write_stream(...)
        >>> status = client.get_write_session(session_id)
        >>> print(f"Streams: {len(status.streams) if status.streams else 0}")
        Streams: 2

        Before committing the session, verify all streams are closed to
        ensure data upload is complete:

        >>> # Close all streams after writing data
        >>> client.close_write_stream(...)
        >>> client.close_write_stream(...)
        >>> # Verify all streams are closed
        >>> status = client.get_write_session(session_id)
        >>> if status.streams:
        ...     # Check if any stream is still open
        ...     open_streams = [s for s in status.streams if s.get('Status') == 'OPEN']
        ...     if open_streams:
        ...         print(f"Warning: {len(open_streams)} streams still open")
        ...     else:
        ...         # All streams closed, safe to commit
        ...         client.commit_write_session(session_id)
        ... else:
        ...     client.commit_write_session(session_id)

        Monitor the progress of a long-running parallel upload by
        periodically checking stream status:

        >>> import time
        >>> while True:
        ...     status = client.get_write_session(session_id)
        ...     if status.streams:
        ...         closed_count = sum(
        ...             1 for s in status.streams
        ...             if s.get('Status') == 'CLOSED'
        ...         )
        ...         print(f"Progress: {closed_count}/{len(status.streams)} streams closed")
        ...         if closed_count == len(status.streams):
        ...             break
        ...     time.sleep(5)
        >>> # All streams closed, commit the session
        >>> client.commit_write_session(session_id)
        """
        self._check_not_instance()
        extra_params = {"SessionId": session_id}
        resp_json, resp = self._request(
            "TableGetWriteSession", extra_params=extra_params, body="{}"
        )

        response = GetWriteSessionResponse()
        response.parse(resp, obj=response)
        _update_request_id(response, resp)
        route_token = resp.headers.get(ROUTE_TOKEN_HEADER)
        if route_token:
            response.route_token = route_token
            self._update_route_token(route_token)

        if response.warning_message:
            logger.warning(response.warning_message)

        return response

    def commit_write_session(
        self, session_id: str, stream_ids=None, stream_versions=None, route_token=None
    ):
        """
        Commit a write session to finalize all uploaded data.

        This is the final step in the write workflow. Calling commit
        atomically makes all data uploaded through the session's write
        streams visible in the table. The commit operation ensures that
        either all writes succeed (transaction committed) or none of
        them appear in the table (transaction aborted).

        Parameters
        ----------
        session_id : str
            The unique identifier of the write session to commit.
            All write streams in this session must be closed before
            committing.
        stream_ids : list of str, optional
            List of stream identifiers to commit. Used for
            transactional/delta tables to specify which streams to
            include in the commit. When provided, stream_versions
            must also be provided with matching length.
        stream_versions : list of int, optional
            List of stream version numbers corresponding to stream_ids.
            Must be provided together with stream_ids and must have
            the same length.

        Returns
        -------
        None
            Method returns None on successful commit. The commit operation
            makes all uploaded data visible in the table atomically.

        Raises
        ------
        ValueError
            If session_id is None or empty, or if called on an
            instance-based client.
        errors.ODPSError
            If any write streams are still open or if the session
            has expired.

        See Also
        --------
        create_write_session : Create a write session.
        close_write_stream : Close all streams before committing.
        abort_write_session : Abort instead of commit to discard writes.
        get_write_session : Verify all streams are closed.

        Notes
        -----
        Before committing, ensure all write streams are closed. Use
        get_write_session to verify stream states. If commit fails,
        call abort_write_session to clean up resources.

        Commit is a synchronous operation. Once it returns successfully,
        the data is immediately visible in the table and cannot be
        rolled back.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient, ArrowWriter, Compression
        ... )
        >>> import pyarrow as pa
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Complete write workflow: create session, create stream, write
        Arrow data, close stream, and commit:

        >>> # 1. Create write session
        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id

        >>> # 2. Create write stream
        >>> stream_resp = client.create_write_stream(session_id, stream_id=0)

        >>> # 3. Write data using Arrow writer
        >>> schema = pa.schema([
        ...     pa.field("id", pa.int64()),
        ...     pa.field("name", pa.string()),
        ... ])
        >>> batch = pa.record_batch([
        ...     pa.array([1, 2, 3]),
        ...     pa.array(["Alice", "Bob", "Carol"]),
        ... ], schema=schema)
        >>> writer = ArrowWriter(
        ...     client.write_rows_stream(session_id, stream_id=0, record_count=3),
        ...     Compression.UNCOMPRESSED
        ... )
        >>> writer.write(batch)
        >>> commit_msg, success = writer.finish()

        >>> # 4. Close the write stream
        >>> client.close_write_stream(session_id, stream_id=0)

        >>> # 5. Commit the session to finalize all writes
        >>> client.commit_write_session(session_id)
        >>> print("Data successfully uploaded to table")
        Data successfully uploaded to table

        For transactional/delta tables, specify stream_ids and
        stream_versions when committing:

        >>> client.commit_write_session(
        ...     session_id,
        ...     stream_ids=["stream-1"],
        ...     stream_versions=[1],
        ... )

        For multi-stream uploads, ensure all streams are closed before
        committing. Use get_write_session to verify:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> # Create multiple streams and write data to each
        >>> for stream_id in range(3):
        ...     # Create stream, write data, close stream...
        ...     pass
        >>> # Verify all streams are closed
        >>> status = client.get_write_session(session_id)
        >>> all_closed = all(
        ...     s.get('Status') == 'CLOSED'
        ...     for s in (status.streams or [])
        ... )
        >>> if all_closed:
        ...     # Commit session to make data visible
        ...     client.commit_write_session(session_id)
        ...     print("Session committed successfully")
        Session committed successfully

        If commit fails due to errors, abort the session to clean up:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> try:
        ...     # Write data...
        ...     client.commit_write_session(session_id)
        ... except Exception as e:
        ...     print(f"Commit failed: {e}")
        ...     # Abort to discard all writes and release resources
        ...     client.abort_write_session(session_id)
        Commit failed: [error message]
        """
        self._check_not_instance()
        extra_params = {"SessionId": session_id}
        body_dict = {}
        if stream_ids is not None and stream_versions is not None:
            body_dict["StreamIds"] = stream_ids
            body_dict["StreamVersions"] = stream_versions
        self._request(
            "TableCommitWriteSession",
            extra_params=extra_params,
            body=json.dumps(body_dict),
            route_token=self._resolve_route_token(route_token),
        )

    def abort_write_session(self, session_id: str, route_token=None):
        """
        Abort a write session to discard all uploaded data.

        This method cancels an active write session and discards all data
        uploaded through its streams. Use abort when you encounter errors
        during the upload process or when you want to cancel the operation
        without making the data visible in the table. Aborting releases all
        resources associated with the session.

        Parameters
        ----------
        session_id : str
            The unique identifier of the write session to abort.
            This can be any active or expired session.

        Returns
        -------
        None
            Method returns None after aborting the session. All uploaded
            data is discarded and session resources are released.

        Raises
        ------
        ValueError
            If session_id is None or empty, or if called on an
            instance-based client.

        See Also
        --------
        create_write_session : Create a write session.
        commit_write_session : Commit the session instead of aborting.
        get_write_session : Check session status before aborting.

        Notes
        -----
        After aborting, the session cannot be committed. All data uploaded
        is permanently discarded. Call abort as soon as you detect errors
        to release resources quickly. It's safe to abort a session multiple
        times if the first abort attempt fails.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import StorageApiClient
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Abort a session when an error occurs during data upload to
        prevent partial data from being committed:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> try:
        ...     # Create stream and write data
        ...     stream_resp = client.create_write_stream(...)
        ...     # Write data that fails due to schema mismatch
        ...     writer = client.write_rows_stream(...)
        ...     writer.write(invalid_data)
        ...     writer.finish()
        ... except Exception as e:
        ...     print(f"Upload failed: {e}")
        ...     # Abort to discard all uploaded data
        ...     client.abort_write_session(session_id)
        ...     print("Session aborted, no data committed to table")
        Upload failed: Schema mismatch error
        Session aborted, no data committed to table

        Use abort in cleanup code to ensure resources are released even
        if commit never happens:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> try:
        ...     # Upload data...
        ...     # Commit if everything succeeds
        ...     client.commit_write_session(session_id)
        ... finally:
        ...     # If commit didn't happen (exception or early return),
        ...     # ensure session is aborted to release resources
        ...     try:
        ...         status = client.get_write_session(session_id)
        ...         if status.streams is None or len(status.streams) == 0:
        ...             client.abort_write_session(session_id)
        ...     except:
        ...         # Session might already be committed or aborted
        ...         pass

        Abort a session after detecting validation errors in the data
        before attempting to commit:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> # Write data to streams
        >>> writer = client.write_rows_stream(...)
        >>> # Validate the data locally
        >>> if data_has_errors:
        ...     print("Data validation failed, aborting session")
        ...     client.abort_write_session(session_id)
        ... else:
        ...     # Data is valid, proceed with commit
        ...     client.close_write_stream(...)
        ...     client.commit_write_session(session_id)
        """
        self._check_not_instance()
        extra_params = {"SessionId": session_id}
        self._request(
            "TableAbortWriteSession",
            extra_params=extra_params,
            body="{}",
            route_token=self._resolve_route_token(route_token),
        )

    # ---- Write Stream ----

    def create_write_stream(
        self,
        session_id=None,
        stream_id=None,
        stream_version=0,
        route_token=None,
    ) -> CreateWriteStreamResponse:
        """
        Create a write stream within an active write session.

        A write stream is a data upload channel within a write session. You
        can create multiple streams in parallel to upload data concurrently,
        improving throughput for large datasets. Each stream must be closed
        individually after writing data, and the session must be committed
        after all streams are closed.

        Parameters
        ----------
        session_id : str
            The write session identifier from create_write_session.
        stream_id : str or int
            Unique identifier for this stream within the session.
            Typically an integer (0, 1, 2, ...) for parallel streams.
        stream_version : int, default 0
            Version number for the stream. Increment if retrying
            after a failed upload to the same stream_id.

        Returns
        -------
        CreateWriteStreamResponse
            Response with data_schema, table_id, schema_version, and route_token.
            See :class:`CreateWriteStreamResponse`.

        Raises
        ------
        ValueError
            If called on an instance-based client.

        See Also
        --------
        create_write_session : Create the parent write session first.
        write_rows_stream : Write data to the created stream.
        close_write_stream : Close the stream after writing.
        get_write_stream : Get stream status.

        Notes
        -----
        Each stream can be written to independently, enabling parallel
        uploads from multiple workers. The stream_version allows retrying
        failed uploads without creating a new session. After a stream is
        closed, it cannot be reopened with the same stream_id and version.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import StorageApiClient
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        First create a write session, then create a single stream for
        sequential data upload:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> stream_resp = client.create_write_stream(session_id, stream_id=0)
        >>> print(f"Stream created, schema: {stream_resp.data_schema}")
        Stream created, schema: {'Columns': [...]}

        For parallel upload from multiple workers, create multiple streams
        with different stream_ids. Each worker processes one stream:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> # Worker 1 creates stream 0
        >>> stream_resp0 = client.create_write_stream(session_id, stream_id=0)
        >>> # Worker 2 creates stream 1
        >>> stream_resp1 = client.create_write_stream(session_id, stream_id=1)
        >>> # Worker 3 creates stream 2
        >>> stream_resp2 = client.create_write_stream(session_id, stream_id=2)
        >>> print(f"Created 3 parallel streams")

        Use stream_version to retry failed uploads. If a stream fails,
        increment the version to create a new stream with the same ID:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> stream_resp = client.create_write_stream(session_id, stream_id=0, stream_version=0)
        >>> # Attempt to write data (fails)
        >>> try:
        ...     writer = client.write_rows_stream(...)
        ...     writer.write(data)
        ... except Exception:
        ...     # Retry with incremented version
        ...     retry_resp = client.create_write_stream(
        ...         session_id, stream_id=0, stream_version=1
        ...     )
        ...     writer = client.write_rows_stream(...)
        ...     writer.write(data)

        Check the data_schema before writing to ensure your data matches
        the expected table schema:

        >>> stream_resp = client.create_write_stream(session_id, stream_id=0)
        >>> schema = stream_resp.data_schema
        >>> if schema:
        ...     columns = schema.get('Columns', [])
        ...     for col in columns:
        ...         print(f"Column: {col.get('Name')}, Type: {col.get('Type')}")
        Column: id, Type: bigint
        Column: name, Type: string
        """
        self._check_not_instance()
        request = CreateWriteStreamRequest(
            session_id=session_id,
            stream_id=stream_id,
            stream_version=stream_version,
        )

        body_dict = {
            "StreamId": request.stream_id,
            "StreamVersion": request.stream_version,
        }
        if request.exactly_once_mode:
            body_dict["ExactlyOnceMode"] = True
        extra_params = {"SessionId": request.session_id}

        resp_json, resp = self._request(
            "TableCreateWriteStream",
            extra_params=extra_params,
            body=json.dumps(body_dict),
            route_token=self._resolve_route_token(route_token),
        )

        response = CreateWriteStreamResponse()
        response.parse(resp, obj=response)
        route_token = resp.headers.get(ROUTE_TOKEN_HEADER)
        if route_token:
            response.route_token = route_token
            self._update_route_token(route_token)
        _update_request_id(response, resp)

        return response

    def get_write_stream(
        self,
        session_id: str,
        stream_id: str,
        stream_version: int,
        route_token=None,
        exactly_once_mode=False,
    ) -> "GetWriteStreamResponse":
        """
        Get the current status and metadata of a write stream.

        This method retrieves information about a specific write stream,
        including its current state, statistics, and any error information.
        Use this to monitor stream health during uploads or to verify
        stream state before closing.

        Parameters
        ----------
        session_id : str
            The unique identifier of the write session.
        stream_id : str
            The identifier of the stream to query (typically 0, 1, 2, ...).
        stream_version : int
            The version number of the stream.

        Returns
        -------
        GetWriteStreamResponse
            Response with status, record_count, byte_size, error_code,
            and error_message. See :class:`GetWriteStreamResponse`.

        Raises
        ------
        ValueError
            If any parameter is None or empty, or if called on an
            instance-based client.

        See Also
        --------
        create_write_stream : Create a stream before getting its status.
        close_write_stream : Close the stream after verifying status.
        get_write_session : Get all streams in a session.

        Notes
        -----
        Use get_write_stream to check if a stream has encountered errors
        before attempting to close it. If a stream is in ERROR state, it
        may need to be recreated with an incremented stream_version.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient
        ... )
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Create a write stream and check its status before writing data:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> stream_resp = client.create_write_stream(session_id, stream_id=0)
        >>> status = client.get_write_stream(session_id, stream_id="0", stream_version=0)
        >>> print(f"Stream status: {status.status}")
        Stream status: OPEN

        After writing data, check the stream statistics to verify upload
        progress before closing:

        >>> # Write data to the stream
        >>> writer = client.write_rows_stream(...)
        >>> writer.write(batch)
        >>> writer.finish()
        >>> # Check stream statistics
        >>> status = client.get_write_stream(session_id, stream_id="0", stream_version=0)
        >>> print(f"Uploaded {status.record_count or 0} records, {status.byte_size or 0} bytes")
        Uploaded 1000 records, 524288 bytes
        >>> # Close the stream after verifying the upload
        >>> client.close_write_stream(...)

        Monitor multiple parallel streams during upload to identify
        which streams have encountered errors:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> # Create 3 parallel streams and write data
        >>> for i in range(3):
        ...     stream_resp = client.create_write_stream(session_id, stream_id=i)
        >>> # Check status of all streams
        >>> for i in range(3):
        ...     status = client.get_write_stream(session_id, stream_id=str(i), stream_version=0)
        ...     if status.status == 'ERROR':
        ...         print(f"Stream {i} failed: {status.error_message}")
        ...     else:
        ...         print(f"Stream {i}: {status.status}, {status.record_count or 0} records")
        Stream 0: OPEN, 500 records
        Stream 1: OPEN, 300 records
        Stream 2: ERROR, Schema validation failed

        If a stream shows ERROR status, use get_write_stream to get
        detailed error information before deciding to retry:

        >>> status = client.get_write_stream(session_id, stream_id="0", stream_version=0)
        >>> if status.status == 'ERROR':
        ...     print(f"Stream error [{status.error_code}]: {status.error_message}")
        ...     # Decide whether to retry with new version or abort session
        ...     if status.error_code == 'SCHEMA_MISMATCH':
        ...         # Retry with corrected data
        ...         client.create_write_stream(session_id, stream_id=0, stream_version=1)
        ...     else:
        ...         # Abort the entire session
        ...         client.abort_write_session(session_id)
        """
        self._check_not_instance()
        url = self._get_v2_url()
        params = {
            "Action": "TableGetWriteStream",
            "Target": self._build_target(),
            "SessionId": session_id,
            "StreamId": stream_id,
            "StreamVersion": str(stream_version),
        }
        if exactly_once_mode:
            params["ExactlyOnceMode"] = "true"
        if self._quota_name:
            params["quotaName"] = self._quota_name
        headers = self._fill_common_headers(
            route_token=self._resolve_route_token(route_token)
        )

        resp = self.tunnel_rest.get(url, params=params, headers=headers)

        response = GetWriteStreamResponse()
        response.parse(resp, obj=response)
        _update_request_id(response, resp)
        route_token_val = resp.headers.get(ROUTE_TOKEN_HEADER)
        if route_token_val:
            response.route_token = route_token_val
            self._update_route_token(route_token_val)

        return response

    def write_rows_stream(
        self,
        session_id,
        stream_id,
        stream_version=0,
        record_count=0,
        compression=None,
        route_token=None,
        row_offset=-1,
        access_token=None,
    ) -> StreamWriter:
        """
        Write row data to a write stream via streaming upload.

        This method creates a streaming writer for uploading row data to a
        write stream. The writer accepts Arrow record batches or raw binary
        data and uploads it incrementally, enabling efficient handling of
        large datasets without loading all data into memory at once.

        Parameters
        ----------
        session_id : str
            The write session identifier.
        stream_id : str or int
            The stream identifier from create_write_stream.
        stream_version : int, default 0
            Version of the stream (should match create_write_stream).
        record_count : int, default 0
            Total number of records to be written. Set this to the
            expected count for validation, or 0 if unknown.
        compression : Compression, default None
            Compression algorithm for the uploaded data stream. None means
            Compression.UNCOMPRESSED.

        Returns
        -------
        StreamWriter
            Stream writer with write(), finish(), get_status(),
            get_request_id(), and writable() methods. See :class:`StreamWriter`.

        Raises
        ------
        ValueError
            If called on an instance-based client.

        See Also
        --------
        create_write_stream : Create a stream before writing.
        ArrowWriter : Convert Arrow batches to stream format.
        close_write_stream : Close the stream after finishing write.

        Notes
        -----
        The returned StreamWriter accepts binary data directly. For Arrow
        workflows, wrap it with ArrowWriter which converts record batches
        to Arrow IPC format. The writer must be finished before closing
        the stream. Call finish() after writing all batches.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient, ArrowWriter, Compression
        ... )
        >>> import pyarrow as pa
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Complete workflow using ArrowWriter to upload Arrow record batches:

        >>> # 1. Create write session
        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> # 2. Create write stream
        >>> stream_resp = client.create_write_stream(session_id, stream_id=0)
        >>> # 3. Write Arrow data
        >>> schema = pa.schema([
        ...     pa.field("id", pa.int64()),
        ...     pa.field("name", pa.string()),
        ...     pa.field("value", pa.float64()),
        ... ])
        >>> batch = pa.record_batch([
        ...     pa.array([1, 2, 3]),
        ...     pa.array(["Alice", "Bob", "Carol"]),
        ...     pa.array([100.0, 200.0, 150.0]),
        ... ], schema=schema)
        >>> stream_writer = client.write_rows_stream(session_id, stream_id=0, record_count=3)
        >>> arrow_writer = ArrowWriter(stream_writer, Compression.UNCOMPRESSED)
        >>> arrow_writer.write(batch)
        >>> commit_msg, success = arrow_writer.finish()
        >>> print(f"Upload successful: {success}")
        Upload successful: True
        >>> # 4. Close stream and commit session
        >>> client.close_write_stream(session_id, stream_id=0)
        >>> client.commit_write_session(session_id)
        """
        self._check_not_instance()
        request = WriteStreamRequest(
            session_id=session_id,
            stream_id=stream_id,
            stream_version=stream_version,
            record_count=record_count,
            compression=compression,
            row_offset=row_offset,
            access_token=access_token,
        )

        url = self._get_v2_url()
        params = {
            "Action": "TableWrite",
            "Target": self._build_target(),
            "SessionId": request.session_id,
            "StreamId": request.stream_id,
            "StreamVersion": str(request.stream_version),
            "Count": str(request.record_count),
        }
        if request.row_offset >= 0:
            params["RowOffset"] = str(request.row_offset)
        if self._quota_name:
            params["quotaName"] = self._quota_name

        headers = self._fill_common_headers(
            {
                "Content-Type": "application/octet-stream",
                "Transfer-Encoding": "chunked",
            },
            route_token=self._resolve_route_token(route_token),
        )
        if request.access_token:
            headers[WRITE_ACCESS_TOKEN_HEADER] = request.access_token

        def upload(data):
            return self.tunnel_rest.post(url, data=data, params=params, headers=headers)

        return StreamWriter(upload, on_route_token=self._update_route_token)

    def close_write_stream(
        self,
        session_id=None,
        stream_id=None,
        stream_version=0,
        route_token=None,
    ) -> CloseWriteStreamResponse:
        """
        Close a write stream to finalize the data upload for that stream.

        After finishing the write operation via the stream writer, call this
        method to formally close the stream. This signals to the server that
        no more data will be uploaded to this stream and marks the stream as
        ready for the session commit. All streams must be closed before the
        session can be committed.

        Parameters
        ----------
        session_id : str
            The write session identifier.
        stream_id : str or int
            The stream identifier to close.
        stream_version : int, default 0
            Version of the stream (should match create_write_stream).

        Returns
        -------
        CloseWriteStreamResponse
            Response with warning_message and request_id.
            See :class:`CloseWriteStreamResponse`.

        Raises
        ------
        ValueError
            If called on an instance-based client.

        See Also
        --------
        write_rows_stream : Finish writing before closing the stream.
        create_write_stream : Create the stream before writing.
        commit_write_session : Commit session after all streams closed.
        get_write_stream : Verify stream is closed.

        Notes
        -----
        The stream writer's finish() method and close_write_stream serve
        different purposes: finish() completes the data upload, while
        close_write_stream formally closes the stream on the server side.
        Both must be called in sequence: finish the writer, then close
        the stream. After closing, the stream cannot accept more data.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient, ArrowWriter, Compression
        ... )
        >>> import pyarrow as pa
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Complete workflow showing the relationship between writer finish()
        and stream close. First finish the writer, then close the stream:

        >>> # 1. Create write session and stream
        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> stream_resp = client.create_write_stream(session_id, stream_id=0)
        >>> # 2. Write Arrow data
        >>> batch = pa.record_batch([...], schema=schema)
        >>> stream_writer = client.write_rows_stream(session_id, stream_id=0, record_count=3)
        >>> arrow_writer = ArrowWriter(stream_writer, Compression.UNCOMPRESSED)
        >>> arrow_writer.write(batch)
        >>> # 3. Finish the writer to complete data upload
        >>> commit_msg, success = arrow_writer.finish()
        >>> print(f"Writer finished: {success}")
        Writer finished: True
        >>> # 4. Close the stream to mark it ready for commit
        >>> close_resp = client.close_write_stream(session_id, stream_id=0)
        >>> if close_resp.warning_message:
        ...     print(f"Warning: {close_resp.warning_message}")
        >>> # 5. Commit the session to make data visible
        >>> client.commit_write_session(session_id)

        Close multiple parallel streams after each finishes uploading.
        All streams must be closed before committing the session:

        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> # Create 3 parallel streams
        >>> for stream_id in range(3):
        ...     # Create stream, write data, finish writer
        ...     client.create_write_stream(session_id, stream_id=stream_id)
        ...     writer = ArrowWriter(
        ...         client.write_rows_stream(session_id, stream_id=stream_id),
        ...         Compression.UNCOMPRESSED
        ...     )
        ...     writer.write(batch)
        ...     writer.finish()
        >>> # Close all 3 streams
        >>> for stream_id in range(3):
        ...     close_resp = client.close_write_stream(session_id, stream_id=stream_id)
        ...     print(f"Stream {stream_id} closed")
        Stream 0 closed
        Stream 1 closed
        Stream 2 closed
        >>> # Now commit the session
        >>> client.commit_write_session(session_id)

        Check for warnings when closing streams to detect potential
        issues that may affect the commit:

        >>> close_resp = client.close_write_stream(session_id, stream_id=0)
        >>> if close_resp.warning_message:
        ...     print(f"Stream close warning: {close_resp.warning_message}")
        ...     # Decide whether to proceed with commit or abort
        ...     if "data_incomplete" in close_resp.warning_message:
        ...         client.abort_write_session(session_id)
        ...     else:
        ...         # Warning is informational, proceed with commit
        ...         client.commit_write_session(session_id)
        ... else:
        ...     client.commit_write_session(session_id)

        Use get_write_stream to verify a stream is properly closed
        before attempting to commit the session:

        >>> # Close stream
        >>> client.close_write_stream(session_id, stream_id=0)
        >>> # Verify it's closed
        >>> status = client.get_write_stream(session_id, stream_id="0", stream_version=0)
        >>> if status.status == 'CLOSED':
        ...     print("Stream properly closed, safe to commit")
        ...     client.commit_write_session(session_id)
        ... else:
        ...     print(f"Unexpected stream status: {status.status}")
        Stream properly closed, safe to commit

        Handle the case where close fails due to stream errors by
        checking the warning message and potentially aborting:

        >>> try:
        ...     close_resp = client.close_write_stream(session_id, stream_id=0)
        ...     if close_resp.warning_message and "error" in close_resp.warning_message.lower():
        ...         print(f"Stream had errors: {close_resp.warning_message}")
        ...         # Abort session instead of committing incomplete data
        ...         client.abort_write_session(session_id)
        ... except Exception as e:
        ...     print(f"Failed to close stream: {e}")
        ...     client.abort_write_session(session_id)
        """
        self._check_not_instance()
        request = CloseWriteStreamRequest(
            session_id=session_id,
            stream_id=stream_id,
            stream_version=stream_version,
        )

        body_dict = {
            "SessionId": request.session_id,
            "StreamId": request.stream_id,
            "StreamVersion": request.stream_version,
        }
        extra_params = {"SessionId": request.session_id}

        resp_json, resp = self._request(
            "TableCloseWriteStream",
            extra_params=extra_params,
            body=json.dumps(body_dict),
            route_token=self._resolve_route_token(route_token),
        )

        response = CloseWriteStreamResponse()
        response.parse(resp, obj=response)
        _update_request_id(response, resp)

        if response.warning_message:
            logger.warning(response.warning_message)

        return response

    # ---- Blob ----

    def write_blob_stream(
        self,
        session_id,
        stream_id,
        stream_version=0,
        partition_values=None,
        column_index=0,
        compression=None,
    ) -> BlobStreamWriter:
        """
        Upload a single blob via streaming upload.

        This method creates a streaming writer for uploading a single binary
        blob (such as an image, video, or large binary file) to a specific
        column in a MaxCompute table. The data is optionally compressed
        using the specified compression algorithm and verified with MD5
        checksum to ensure upload integrity.

        Parameters
        ----------
        session_id : str
            The write session identifier from create_write_session.
        stream_id : str or int
            Stream identifier for this upload.
        stream_version : int, default 0
            Version number for the stream.
        partition_values : list of str, optional
            Partition values for the blob location.
            Format: ['pt=20230101', 'region=us-west'].
        column_index : int, default 0
            Column index in the table schema where the blob will be stored.
        compression : Compression, default None
            Compression algorithm to use. None means Compression.UNCOMPRESSED.
            See :class:`Compression`.
            Supported values: Compression.ZSTD, Compression.LZ4_FRAME,
            Compression.UNCOMPRESSED.

        Returns
        -------
        BlobStreamWriter
            Blob stream writer with write(), finish(), get_status(),
            get_request_id(), and writable() methods. See :class:`BlobStreamWriter`.

        Raises
        ------
        ValueError
            If called on an instance-based client.
        errors.DependencyNotInstalledError
            If zstandard library is not installed (required for ZSTD compression).
        errors.ChecksumError
            If MD5 checksum verification fails after finish().

        See Also
        --------
        write_blob_batch : Upload multiple blobs in one request.
        read_blobs : Read uploaded blobs by reference.
        BlobWriteItem : Helper class for batch blob uploads.

        Notes
        -----
        Blob uploads default to no compression (Compression.UNCOMPRESSED).
        Compression algorithms like ZSTD or LZ4_FRAME can be enabled for
        efficient transfer. Set compression=Compression.ZSTD for zstd
        compression or compression=Compression.LZ4_FRAME for lz4.
        The writer computes MD5 checksum incrementally during write and
        verifies against server response when finished. If the checksum
        doesn't match, a ChecksumError is raised, indicating data corruption
        during upload.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient
        ... )
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Complete workflow for uploading a single image blob to a table
        with a binary column:

        >>> # 1. Create write session
        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> # 2. Create write stream
        >>> stream_resp = client.create_write_stream(session_id, stream_id=0)
        >>> # 3. Upload blob data
        >>> blob_writer = client.write_blob_stream(
        ...     session_id, stream_id=0,
        ...     partition_values=['pt=20230101'],
        ...     column_index=0  # First column is the blob column
        ... )
        >>> # Read image file and upload in chunks
        >>> with open('image.jpg', 'rb') as f:
        ...     while True:
        ...         chunk = f.read(8192)  # 8KB chunks
        ...         if not chunk:
        ...             break
        ...         blob_writer.write(chunk)
        >>> # 4. Finish and verify MD5
        >>> response = blob_writer.finish()
        >>> print(f"Blob reference: {response.blob_reference}")
        >>> print(f"Uploaded size: {response.size} bytes")
        Blob reference: blob_ref_abc123
        Uploaded size: 524288 bytes
        >>> # 5. Close stream and commit session
        >>> client.close_write_stream(session_id, stream_id=0)
        >>> client.commit_write_session(session_id)

        Upload blob data directly from memory without reading from a file,
        useful for dynamically generated binary data:

        >>> blob_writer = client.write_blob_stream(session_id, stream_id=0, column_index=0)
        >>> # Generate or prepare binary data
        >>> binary_data = b"generated binary content..."
        >>> blob_writer.write(binary_data)
        >>> # Finish upload
        >>> response = blob_writer.finish()
        >>> print(f"Blob uploaded: {response.blob_reference}")

        Upload a blob to a specific partition by providing partition values.
        The blob is stored in the specified partition's location:

        >>> blob_writer = client.write_blob_stream(
        ...     session_id, stream_id=0,
        ...     partition_values=['pt=20230101', 'region=us-west'],
        ...     column_index=0
        ... )
        >>> blob_writer.write(image_data)
        >>> response = blob_writer.finish()
        >>> # Blob is now in partition pt=20230101/region=us-west

        Handle checksum errors when the upload integrity check fails,
        indicating data corruption during transfer:

        >>> blob_writer = client.write_blob_stream(session_id, stream_id=0, column_index=0)
        >>> blob_writer.write(data)
        >>> try:
        ...     response = blob_writer.finish()
        ...     print("Upload successful")
        ... except errors.ChecksumError as e:
        ...     print(f"Checksum mismatch: {e}")
        ...     # Data was corrupted during upload
        ...     # Retry the upload or abort the session
        ...     client.abort_write_session(session_id)
        Checksum mismatch: MD5 value mismatch, expected: abc123, actual: def456

        Upload large blobs incrementally by reading and writing in chunks
        to avoid loading the entire blob into memory:

        >>> blob_writer = client.write_blob_stream(session_id, stream_id=0, column_index=0)
        >>> # Upload a large video file in 64KB chunks
        >>> chunk_size = 65536
        >>> with open('large_video.mp4', 'rb') as f:
        ...     while True:
        ...         chunk = f.read(chunk_size)
        ...         if not chunk:
        ...             break
        ...         success = blob_writer.write(chunk)
        ...         if not success:
        ...             print("Writer closed unexpectedly")
        ...             break
        >>> response = blob_writer.finish()
        >>> print(f"Large video uploaded: {response.size} bytes")
        Large video uploaded: 104857600 bytes

        Upload text data by converting strings to bytes. The writer
        automatically handles the conversion:

        >>> blob_writer = client.write_blob_stream(session_id, stream_id=0, column_index=0)
        >>> text_content = "This is text data to store as a blob"
        >>> blob_writer.write(text_content)  # Automatically converted to bytes
        >>> response = blob_writer.finish()

        Monitor writer status during long uploads to detect early failures:

        >>> blob_writer = client.write_blob_stream(session_id, stream_id=0, column_index=0)
        >>> for chunk in large_data_chunks:
        ...     success = blob_writer.write(chunk)
        ...     if not success or blob_writer.get_status() != Status.RUNNING:
        ...         print("Upload stopped unexpectedly")
        ...         break
        >>> response = blob_writer.finish()
        >>> request_id = blob_writer.get_request_id()
        """
        self._check_not_instance()
        if compression is None:
            compression = Compression.UNCOMPRESSED
        elif not isinstance(compression, Compression):
            raise ValueError(
                f"compression must be a Compression enum value, got {type(compression)}"
            )
        request = WriteBlobRequest(
            session_id=session_id,
            stream_id=stream_id,
            stream_version=stream_version,
            partition_values=partition_values,
            column_index=column_index,
        )

        url = self._get_v2_url()
        params = {
            "Action": "TableWriteBlob",
            "Target": self._build_target(),
            "SessionId": request.session_id,
            "StreamId": request.stream_id,
            "StreamVersion": str(request.stream_version),
            "PartitionValues": ",".join(request.partition_values)
            if request.partition_values
            else "",
            "ColumnIndex": str(request.column_index),
        }
        if self._quota_name:
            params["quotaName"] = self._quota_name

        headers = self._fill_common_headers(
            {
                "Content-Type": "application/octet-stream",
                "Transfer-Encoding": "chunked",
            }
        )
        compress_algo = self._compression_to_compress_algo(compression)
        if compress_algo is not None:
            encoding = compress_algo.get_encoding()
            if encoding:
                headers["Content-Encoding"] = encoding

        def upload(data):
            return self.tunnel_rest.post(url, data=data, params=params, headers=headers)

        return BlobStreamWriter(upload, compression=compression)

    def write_blob_batch(
        self,
        items: List[BlobWriteItem],
        session_id=None,
        stream_id=None,
        stream_version=0,
        partition_values=None,
        column_index=0,
    ) -> WriteBlobResponse:
        """
        Upload multiple blobs in a single batch request for efficiency.

        This method uploads multiple binary blobs (such as images, videos, or
        files) in one consolidated request, reducing network overhead compared
        to individual uploads. Each blob is packaged with metadata (partition
        location, column index, MIME type) and optional checksum verification.

        Parameters
        ----------
        items : list of BlobWriteItem
            List of blob items to upload. Each item contains data (bytes),
            partition_values (list of str), column_index (int),
            distribution_key (str), mime_type (str), and checksum_type
            (ChecksumType). See :class:`BlobWriteItem`.
        session_id : str, optional
            The write session identifier from create_write_session.
        stream_id : str or int, optional
            Stream identifier for this batch upload.
        stream_version : int, default 0
            Version number for the stream.
        partition_values : list of str, optional
            Default partition values for blobs (can be overridden per item).
            Format: ['pt=20230101', 'region=us-west'].
        column_index : int, default 0
            Default column index for blobs (can be overridden per item).

        Returns
        -------
        WriteBlobResponse
            Response with blob_reference, blob_references, warning_message,
            and size. See :class:`WriteBlobResponse`.

        Raises
        ------
        ValueError
            If called on an instance-based client.

        See Also
        --------
        write_blob_stream : Upload single blob via streaming.
        read_blobs : Read uploaded blobs using the references.
        BlobWriteItem : Class to construct blob items.

        Notes
        -----
        Batch upload is more efficient than individual uploads when uploading
        many small blobs. The method uses a custom wire format with header,
        data, and footer sections for each blob. Checksums (CRC32 or MD5)
        are computed and included in the footer for server-side verification.
        The order of blob_references matches the input items order.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import (
        ...     StorageApiClient, BlobWriteItem, ChecksumType
        ... )
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Upload multiple image blobs in a single batch request for efficient
        network usage:

        >>> # 1. Create write session and stream
        >>> session_resp = client.create_write_session()
        >>> session_id = session_resp.session_id
        >>> stream_resp = client.create_write_stream(session_id, stream_id=0)
        >>> # 2. Prepare multiple blob items
        >>> items = []
        >>> for image_path in ['img1.jpg', 'img2.jpg', 'img3.jpg']:
        ...     with open(image_path, 'rb') as f:
        ...         image_data = f.read()
        ...     item = BlobWriteItem(
        ...         data=image_data,
        ...         partition_values=['pt=20230101'],
        ...         column_index=0,
        ...         mime_type='image/jpeg',
        ...         checksum_type=ChecksumType.CRC32  # Verify integrity
        ...     )
        ...     items.append(item)
        >>> # 3. Upload all blobs in one batch
        >>> batch_resp = client.write_blob_batch(items, session_id, stream_id=0)
        >>> print(f"Uploaded {len(batch_resp.blob_references)} blobs")
        >>> print(f"Total size: {batch_resp.size} bytes")
        Uploaded 3 blobs
        Total size: 1572864 bytes
        >>> # 4. Store references for later reading
        >>> blob_refs = batch_resp.blob_references
        >>> # 5. Close stream and commit session
        >>> client.close_write_stream(session_id, stream_id=0)
        >>> client.commit_write_session(session_id)

        Upload blobs to different partitions and columns by customizing
        each BlobWriteItem's metadata:

        >>> items = [
        ...     BlobWriteItem(
        ...         data=blob1_data,
        ...         partition_values=['pt=20230101'],
        ...         column_index=0
        ...     ),
        ...     BlobWriteItem(
        ...         data=blob2_data,
        ...         partition_values=['pt=20230102'],
        ...         column_index=1
        ...     ),
        ...     BlobWriteItem(
        ...         data=blob3_data,
        ...         partition_values=['pt=20230103'],
        ...         column_index=0
        ...     ),
        ... ]
        >>> response = client.write_blob_batch(items, session_id, stream_id=0)
        >>> # Each blob is stored in its specified partition/column

        Use MD5 checksum for stronger integrity verification on critical
        data uploads. The checksum is computed and sent with the blob:

        >>> critical_data = read_critical_file()
        >>> item = BlobWriteItem(
        ...     data=critical_data,
        ...     column_index=0,
        ...     checksum_type=ChecksumType.MD5
        ... )
        >>> response = client.write_blob_batch([item], session_id, stream_id=0)
        >>> # Server verifies MD5 checksum matches computed value

        Add MIME type metadata to help applications understand blob content
        type without examining the binary data:

        >>> items = [
        ...     BlobWriteItem(
        ...         data=json_data_bytes,
        ...         mime_type='application/json',
        ...         column_index=0
        ...     ),
        ...     BlobWriteItem(
        ...         data=pdf_data_bytes,
        ...         mime_type='application/pdf',
        ...         column_index=1
        ...     ),
        ... ]
        >>> response = client.write_blob_batch(items, session_id, stream_id=0)
        >>> # MIME types stored with blobs for content type hints

        Use distribution_key for hash-based storage to ensure blobs are
        distributed across storage locations based on the key:

        >>> items = [
        ...     BlobWriteItem(
        ...         data=blob_data,
        ...         distribution_key='user123',
        ...         column_index=0
        ...     ),
        ... ]
        >>> response = client.write_blob_batch(items, session_id, stream_id=0)
        >>> # Blob stored at location determined by distribution key hash

        Upload a large number of small files efficiently by batching them
        instead of uploading individually:

        >>> # Prepare 100 small file blobs
        >>> items = []
        >>> for file_path in small_files:  # 100 small files
        ...     with open(file_path, 'rb') as f:
        ...         data = f.read()
        ...     items.append(BlobWriteItem(data=data, column_index=0))
        >>> # Upload all in one batch (much faster than 100 individual uploads)
        >>> response = client.write_blob_batch(items, session_id, stream_id=0)
        >>> print(f"Batch uploaded {len(response.blob_references)} blobs")
        Batch uploaded 100 blobs

        Check for warnings in the response to detect partial upload failures
        or other issues:

        >>> response = client.write_blob_batch(items, session_id, stream_id=0)
        >>> if response.warning_message:
        ...     print(f"Upload warning: {response.warning_message}")
        ...     # Some blobs may have had issues
        ...     # Check if all blob_references are present
        ...     if len(response.blob_references) < len(items):
        ...         print(f"Only {len(response.blob_references)} of {len(items)} uploaded")
        Upload warning: 2 blobs exceeded size limit
        Only 98 of 100 uploaded

        Use the returned blob_references to read the blobs later by calling
        read_blobs with the reference list:

        >>> batch_resp = client.write_blob_batch(items, session_id, stream_id=0)
        >>> blob_refs = batch_resp.blob_references
        >>> # Later, read the blobs back
        >>> blob_iterator = client.read_blobs(blob_references=blob_refs)
        >>> for data, mime_type in blob_iterator:
        ...     # Process each blob data
        ...     print(f"Read blob: {len(data)} bytes, type: {mime_type}")
        """
        self._check_not_instance()
        request = WriteBlobRequest(
            session_id=session_id,
            stream_id=stream_id,
            stream_version=stream_version,
            partition_values=partition_values,
            column_index=column_index,
        )

        url = self._get_v2_url()
        params = {
            "Action": "TableWriteBlob",
            "Target": self._build_target(),
            "SessionId": request.session_id,
            "StreamId": request.stream_id,
            "StreamVersion": str(request.stream_version),
            "Mode": "Batch",
        }
        if self._quota_name:
            params["quotaName"] = self._quota_name

        headers = self._fill_common_headers(
            {
                "Content-Type": "application/octet-stream",
                "Transfer-Encoding": "chunked",
            }
        )

        def upload(data):
            return self.tunnel_rest.post(url, data=data, params=params, headers=headers)

        req_io = RequestsIO(upload, chunk_size=options.chunk_size)
        req_io.start()
        for item in items:
            item.write_frame_to(req_io)
        resp = req_io.finish()

        response = WriteBlobResponse()
        response.parse(resp, obj=response)
        _update_request_id(response, resp)

        if response.warning_message:
            logger.warning(response.warning_message)

        return response

    def read_blobs(self, blob_references=None, compression=None, stream=False):
        """
        Download binary blobs by their blob references.

        This method retrieves blob data uploaded via write_blob_stream or
        write_blob_batch by providing the blob references returned during
        upload. The data is returned as an iterator that yields tuples of
        (data_bytes, mime_type), handling protocol framing and CRC
        stripping automatically.

        Parameters
        ----------
        blob_references : list of str or bytes, optional
            List of blob references obtained from previous
            upload operations (write_blob_stream or write_blob_batch).
            References may be UTF-8 strings or raw bytes and will
            be decoded automatically if needed.
        compression : Compression, default None
            Compression algorithm for the response data. None means
            Compression.UNCOMPRESSED.
        stream : bool, optional
            If True, return a :class:`BlobStreamReader` instead of
            :class:`BlobDataIterator`. The reader provides file-like
            ``read()`` access per blob, a ``mime_type`` property,
            and a ``next()`` method to advance to the next blob.
            Default is False.

        Returns
        -------
        BlobDataIterator or BlobStreamReader
            When *stream* is False (default), returns an iterator
            yielding ``(data_bytes, mime_type)`` tuples per blob.
            When *stream* is True, returns a :class:`BlobStreamReader`
            for incremental, file-like reading of each blob.

        See Also
        --------
        write_blob_batch : Upload blobs and get references.
        write_blob_stream : Upload single blob and get reference.
        BlobWriteItem : Helper for batch uploads.

        Notes
        -----
        The download stream uses a multi-layer protocol:
        1. CRC32C checksums are embedded every 4096 bytes (stripped automatically)
        2. Protocol framing wraps each blob (parsed automatically)

        For single blob downloads, the server may omit framing and send raw
        decompressed data directly. The iterator automatically detects this
        case and returns the entire payload as one blob.

        Examples
        --------
        >>> from odps import ODPS
        >>> from odps.apis.storage_api_v2 import StorageApiClient
        >>> odps = ODPS(
        ...     access_id="your_access_id",
        ...     secret_access_key="your_secret_access_key",
        ...     project="your_project",
        ...     endpoint="your_endpoint"
        ... )
        >>> table = odps.get_table("your_table")
        >>> client = StorageApiClient(odps, table)

        Read blobs using references obtained from previous uploads.
        First upload blobs, then read them back using the references:

        >>> # Upload blobs (from earlier write_blob_batch example)
        >>> blob_refs = ['blob_ref_001', 'blob_ref_002', 'blob_ref_003']
        >>> # Read the blobs back using references
        >>> blob_iterator = client.read_blobs(blob_references=blob_refs)
        >>> for data, mime_type in blob_iterator:
        ...     print(f"Blob size: {len(data)} bytes, MIME type: {mime_type}")
        ...     # Save blob to file
        ...     filename = f"blob_{blob_refs[i]}"
        ...     with open(filename, 'wb') as f:
        ...         f.write(data)
        Blob size: 1024 bytes, MIME type: image/jpeg
        Blob size: 2048 bytes, MIME type: image/png
        Blob size: 512 bytes, MIME type: application/json

        Download a single blob by passing a single-element reference list.
        The iterator yields one (data, mime_type) tuple:

        >>> single_ref = ['blob_ref_abc123']
        >>> blob_iterator = client.read_blobs(blob_references=single_ref)
        >>> data, mime_type = next(blob_iterator)
        >>> print(f"Downloaded {len(data)} bytes")
        Downloaded 524288 bytes
        >>> # Process the blob data
        >>> if mime_type == 'image/jpeg':
        ...     # Process as JPEG image
        ...     process_image(data)

        Download all blobs from a batch upload by storing references during
        upload and using them to read later:

        >>> # Earlier: upload batch
        >>> batch_resp = client.write_blob_batch(items, session_id, stream_id=0)
        >>> all_refs = batch_resp.blob_references  # Save these
        >>> # Later: read all blobs from the batch
        >>> blob_iterator = client.read_blobs(blob_references=all_refs)
        >>> downloaded_blobs = []
        >>> for data, mime_type in blob_iterator:
        ...     downloaded_blobs.append({
        ...         'data': data,
        ...         'mime_type': mime_type,
        ...         'size': len(data)
        ...     })
        >>> print(f"Downloaded {len(downloaded_blobs)} blobs")
        Downloaded 10 blobs

        Process blobs incrementally without loading all into memory by
        iterating and processing one at a time:

        >>> blob_iterator = client.read_blobs(blob_references=large_blob_refs)
        >>> for i, (data, mime_type) in enumerate(blob_iterator):
        ...     # Process each blob and discard data after processing
        ...     result = analyze_blob(data)
        ...     print(f"Blob {i}: {result}")
        ...     # data is freed after this iteration
        Blob 0: Analysis complete
        Blob 1: Analysis complete

        Check MIME type to determine how to handle blob content when the
        type was set during upload:

        >>> blob_iterator = client.read_blobs(blob_references=blob_refs)
        >>> for data, mime_type in blob_iterator:
        ...     if mime_type == 'application/json':
        ...         # Parse as JSON
        ...         import json
        ...         json_obj = json.loads(data.decode('utf-8'))
        ...         process_json(json_obj)
        ...     elif mime_type == 'image/jpeg':
        ...         # Process as image
        ...         display_image(data)
        ...     elif mime_type is None:
        ...         # No type hint, use generic binary handling
        ...         process_binary(data)
        ...     else:
        ...         print(f"Unknown MIME type: {mime_type}")

        Convert blob iterator to list for multiple iterations or indexing,
        though this loads all data into memory:

        >>> blob_iterator = client.read_blobs(blob_references=blob_refs)
        >>> all_blobs = list(blob_iterator)  # [(data1, mime1), (data2, mime2), ...]
        >>> # Now can access by index
        >>> first_blob_data = all_blobs[0][0]
        >>> first_blob_mime = all_blobs[0][1]
        >>> # Can iterate multiple times
        >>> for data, mime_type in all_blobs:
        ...     process_blob(data)
        >>> for data, mime_type in all_blobs:
        ...     validate_blob(data)

        Handle empty blob reference list by checking iterator length
        or attempting to iterate:

        >>> empty_refs = []
        >>> blob_iterator = client.read_blobs(blob_references=empty_refs)
        >>> count = 0
        >>> for _ in blob_iterator:
        ...     count += 1
        >>> print(f"Downloaded {count} blobs")
        Downloaded 0 blobs

        Use blob references from a table query to download specific blobs
        referenced in table rows:

        >>> # Query table to get blob references
        >>> instance = odps.execute_sql("SELECT blob_ref FROM your_table WHERE id=1")
        >>> with instance.open_reader() as reader:
        ...     for record in reader:
        ...         blob_ref = record[0]  # blob_ref column
        ...         # Download the referenced blob
        ...         blob_iterator = client.read_blobs(blob_references=[blob_ref])
        ...         data, mime_type = next(blob_iterator)
        ...         print(f"Downloaded blob {blob_ref}: {len(data)} bytes")

        Stream blobs incrementally using file-like read interface
        with ``stream=True``:

        >>> reader = client.read_blobs(blob_references=refs, stream=True)
        >>> while reader is not None:
        ...     print(f"MIME: {reader.mime_type}")
        ...     chunk = reader.read(4096)
        ...     while chunk:
        ...         process(chunk)
        ...         chunk = reader.read(4096)
        ...     reader = reader.next()
        """
        request = ReadBlobRequest(blob_references=blob_references)

        url = self._get_v2_url()
        params = {
            "Action": "BlobRead",
            "Target": "generic.blob",
        }
        if self._quota_name:
            params["quotaName"] = self._quota_name

        body_dict = {"BlobReferences": request.blob_references}
        extra_headers = {}
        if compression is not None:
            compress_algo = self._compression_to_compress_algo(compression)
            if compress_algo is not None:
                encoding = compress_algo.get_encoding()
                if encoding:
                    extra_headers["ACCEPT-ENCODING"] = encoding
        headers = self._fill_common_headers(extra_headers)

        resp = self.tunnel_rest.post(
            url, data=json.dumps(body_dict), params=params, headers=headers, stream=True
        )

        # Determine compression from the response Content-Encoding header
        # and wrap the raw stream with a decompressor when needed.
        content_encoding = resp.headers.get("Content-Encoding")
        raw_stream = resp.raw
        if content_encoding:
            compress_algo = CompressOption.CompressAlgorithm.from_encoding(
                content_encoding
            )
            compress_option = CompressOption(compress_algo=compress_algo)
            raw_stream = get_decompress_stream(resp, compress_option)

        iterator = BlobDataIterator(raw_stream)
        if stream:
            return BlobStreamReader(iterator)
        return iterator


class StorageApiArrowClient(StorageApiClient):
    """Arrow batch client for the Storage API V2.

    Extends :class:`StorageApiClient` with convenience methods that
    wrap raw stream I/O with :class:`ArrowReader` and :class:`ArrowWriter`,
    so you can work directly with PyArrow ``RecordBatch`` objects instead
    of raw bytes.

    Parameters
    ----------
    odps : ODPS
        ODPS entry object.
    table_or_instance : Table or Instance, optional
        MaxCompute table or SQL instance to operate on. If an Instance
        is provided, only read operations are available.
    rest_endpoint : str, optional
        Custom REST endpoint for the storage API.
    quota_name : str, optional
        Quota name for resource management.
    tags : str or list of str, optional
        Tags for request tracking.

    See Also
    --------
    StorageApiClient : Base client with raw stream I/O.

    Examples
    --------
    >>> from odps import ODPS
    >>> from odps.apis.storage_api_v2 import StorageApiArrowClient
    >>> odps = ODPS(
    ...     access_id="your_access_id",
    ...     secret_access_key="your_secret_access_key",
    ...     project="your_project",
    ...     endpoint="your_endpoint"
    ... )
    >>> table = odps.get_table("your_table")
    >>> client = StorageApiArrowClient(odps, table)

    Read data using ArrowReader:

    >>> read_resp = client.create_read_session()
    >>> reader = client.read_rows_arrow(read_resp.session_id, split_index=0)
    >>> while True:
    ...     batch = reader.read()
    ...     if batch is None:
    ...         break
    ...     df = batch.to_pandas()

    Write data using ArrowWriter:

    >>> write_resp = client.create_write_session()
    >>> writer = client.write_rows_arrow(
    ...     write_resp.session_id, stream_id="0", record_count=100
    ... )
    >>> writer.write(record_batch)
    >>> writer.finish()
    """

    def read_rows_arrow(
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
        route_token=None,
    ) -> ArrowReader:
        """Read one split of the read session as Arrow batches.

        This is the Arrow convenience wrapper for :meth:`read_rows_stream`.
        It reads raw bytes from the server and wraps them in an
        :class:`ArrowReader` that yields ``pyarrow.RecordBatch`` objects.

        Parameters
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
        max_batch_rows : int, default 4096
            Maximum rows per Arrow batch in the stream. Controls
            memory usage during reading.
        skip_row_num : int, default 0
            Number of rows to skip before reading.
        max_batch_raw_size : int, default 0
            Maximum raw byte size per batch. 0 means no limit.
        data_format : DataFormat, optional
            Format of returned data (Arrow V5 is default).
        data_columns : list of str, optional
            Specific columns to read. Must match session schema.
        compression : Compression, default None
            Compression algorithm for the stream data. None means
            Compression.UNCOMPRESSED.

        Returns
        -------
        ArrowReader
            Arrow batch reader with read() method yielding RecordBatch objects.
            See :class:`ArrowReader`.
        """
        return ArrowReader(
            self.read_rows_stream(
                session_id=session_id,
                split_index=split_index,
                row_offset=row_offset,
                row_count=row_count,
                max_batch_rows=max_batch_rows,
                skip_row_num=skip_row_num,
                max_batch_raw_size=max_batch_raw_size,
                data_format=data_format,
                data_columns=data_columns,
                compression=compression,
                route_token=route_token,
            )
        )

    def write_rows_arrow(
        self,
        session_id=None,
        stream_id=None,
        stream_version=0,
        record_count=0,
        compression=None,
        route_token=None,
        row_offset=-1,
        access_token=None,
    ) -> ArrowWriter:
        """Write Arrow batches to a write stream.

        This is the Arrow convenience wrapper for :meth:`write_rows_stream`.
        It creates an :class:`ArrowWriter` that serializes
        ``pyarrow.RecordBatch`` objects into Arrow IPC format and writes
        them to the underlying stream.

        Parameters
        ----------
        session_id : str
            The write session identifier.
        stream_id : str or int
            The stream identifier from create_write_stream.
        stream_version : int, default 0
            Version of the stream (should match create_write_stream).
        record_count : int, default 0
            Total number of records to be written. Set this to the
            expected count for validation, or 0 if unknown.
        compression : Compression, default None
            Compression algorithm for the uploaded data stream. None means
            Compression.UNCOMPRESSED.
        row_offset : int, default -1
            Row offset for Exactly-Once mode. -1 means not used.
        access_token : str, optional
            Access token for Exactly-Once mode.

        Returns
        -------
        ArrowWriter
            Arrow batch writer with write() and finish() methods.
            See :class:`ArrowWriter`.
        """
        stream_writer = self.write_rows_stream(
            session_id=session_id,
            stream_id=stream_id,
            stream_version=stream_version,
            record_count=record_count,
            compression=compression,
            route_token=route_token,
            row_offset=row_offset,
            access_token=access_token,
        )
        if compression is None:
            compression = Compression.UNCOMPRESSED
        return ArrowWriter(stream_writer, compression)

    def preview_table_arrow(
        self, limit=None, partition=None, columns=None
    ) -> ArrowReader:
        """Preview table data as Arrow batches.

        This is the Arrow convenience wrapper for :meth:`preview_table`.
        It reads a small sample of rows and returns them via an
        :class:`ArrowReader` that yields ``pyarrow.RecordBatch`` objects.

        Parameters
        ----------
        limit : int, optional
            Maximum number of rows to preview. If None, returns a
            small default sample (typically 100-1000 rows).
        partition : str, optional
            Partition specification to preview specific partition data.
            Format: 'pt=20230101' or 'pt=20230101,region=us-west'.
        columns : list of str, optional
            Specific columns to preview. If empty, all columns are returned.

        Returns
        -------
        ArrowReader
            Arrow batch reader with read() method yielding RecordBatch objects.
            See :class:`ArrowReader`.
        """
        return ArrowReader(
            self.preview_table(limit=limit, partition=partition, columns=columns)
        )
