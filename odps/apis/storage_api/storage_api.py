# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

"""Client for interacting with the MaxCompute Storage API."""

import collections
import json
import logging
from io import IOBase, BytesIO
from enum import Enum
from hashlib import md5
try:
    import pyarrow as pa
except ImportError:
    pa = None

from ... import ODPS, serializers
from ...lib.requests import codes
from ...models import Table
from ...models.core import JSONRemoteModel
from ...utils import to_binary

STORAGE_VERSION = "1"
URL_PREFIX = "/api/storage/v" + STORAGE_VERSION

logger = logging.getLogger(__name__)


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


class SplitOptions(JSONRemoteModel):
    class SplitMode(str, Enum):
        SIZE = "Size"
        PARALLELISM = "Parallelism"
        ROW_OFFSET = "RowOffset"
        BUCKET = "Bucket"

    split_mode = serializers.JSONNodeField('SplitMode', parse_callback=lambda s: SplitOptions.SplitMode(s))
    split_number = serializers.JSONNodeField("SplitNumber")
    cross_partition = serializers.JSONNodeField("CrossPartition")

    def __init__(self, **kwargs):
        super(SplitOptions, self).__init__(**kwargs)

        self.split_mode = self.split_mode or SplitOptions.SplitMode.SIZE
        self.split_number = self.split_number or 256*1024*1024
        self.cross_partition = self.cross_partition or True

    @classmethod
    def get_default_options(self, mode):
        options = SplitOptions()
        options.cross_partition = True
        if mode == SplitOptions.SplitMode.SIZE:
            options.split_mode = SplitOptions.SplitMode.SIZE
            options.split_number = 256*1024*1024
        elif mode == SplitOptions.SplitMode.PARALLELISM:
            options.split_mode = SplitOptions.SplitMode.PARALLELISM
            options.split_number = 32
        elif mode == SplitOptions.SplitMode.ROW_OFFSET:
            options.split_mode = SplitOptions.SplitMode.ROW_OFFSET
            options.split_number = 0
        elif mode == SplitOptions.SplitMode.BUCKET:
            options.split_mode = SplitOptions.SplitMode.BUCKET

        return options


class ArrowOptions(JSONRemoteModel):
    class TimestampUnit(str, Enum):
        SECOND = "second"
        MILLI = "milli"
        MICRO = "micro"
        NANO = "nano"

    timestamp_unit = serializers.JSONNodeField('TimestampUnit', parse_callback=lambda s: ArrowOptions.TimestampUnit(s))
    date_time_unit = serializers.JSONNodeField('DatetimeUnit', parse_callback=lambda s: ArrowOptions.TimestampUnit(s))

    def __init__(self, **kwargs):
        super(ArrowOptions, self).__init__(**kwargs)

        self.timestamp_unit = self.timestamp_unit or ArrowOptions.TimestampUnit.NANO
        self.date_time_unit = self.date_time_unit or ArrowOptions.TimestampUnit.MILLI


class Column(JSONRemoteModel):
    name = serializers.JSONNodeField("Name")
    type = serializers.JSONNodeField("Type")
    comment = serializers.JSONNodeField("Comment")
    nullable = serializers.JSONNodeField("Nullable")


class DataSchema(JSONRemoteModel):
    data_columns = serializers.JSONNodesReferencesField(Column, 'DataColumns')
    partition_columns = serializers.JSONNodesReferencesField(Column, 'PartitionColumns')


class DataFormat(JSONRemoteModel):
    type = serializers.JSONNodeField("Type")
    version = serializers.JSONNodeField("Version")


class DynamicPartitionOptions(JSONRemoteModel):
    invalid_strategy = serializers.JSONNodeField("InvalidStrategy")
    invalid_limit = serializers.JSONNodeField("InvalidLimit")
    dynamic_partition_limit = serializers.JSONNodeField("DynamicPartitionLimit")

    def __init__(self, **kwargs):
        super(DynamicPartitionOptions, self).__init__(**kwargs)

        self.invalid_strategy = self.invalid_strategy or "Exception"
        self.invalid_limit = self.invalid_limit or 1
        self.dynamic_partition_limit = self.dynamic_partition_limit or 512


class Order(JSONRemoteModel):
    name = serializers.JSONNodeField("Name")
    sort_direction = serializers.JSONNodeField("SortDirection")


class RequiredDistribution(JSONRemoteModel):
    type = serializers.JSONNodeField("Type")
    cluster_keys = serializers.JSONNodeField("ClusterKeys")
    buckets_number = serializers.JSONNodeField("BucketsNumber")


class Compression(Enum):
    UNCOMPRESSED = 0
    ZSTD = 1
    LZ4_FRAME = 2

    def to_string(self):
        if self.value == 0:
            return None
        elif self.value == 1:
            return "zstd"
        elif self.value == 2:
            return "lz4"
        else:
            return "unknown"


class TableBatchScanRequest(serializers.JSONSerializableModel):
    required_data_columns = serializers.JSONNodeField("RequiredDataColumns")
    required_partition_columns = serializers.JSONNodeField("RequiredPartitionColumns")
    required_partitions = serializers.JSONNodeField("RequiredPartitions")
    required_bucket_ids = serializers.JSONNodeField("RequiredBucketIds")
    split_options = serializers.JSONNodeReferenceField(SplitOptions, "SplitOptions")
    arrow_options = serializers.JSONNodeReferenceField(ArrowOptions, "ArrowOptions")
    filter_predicate = serializers.JSONNodeField("FilterPredicate")

    def __init__(self, **kwargs):
        super(TableBatchScanRequest, self).__init__(**kwargs)

        self.required_data_columns = self.required_data_columns or []
        self.required_partition_columns = self.required_partition_columns or []
        self.required_partitions = self.required_partitions or []
        self.required_bucket_ids = self.required_bucket_ids or []
        self.split_options = self.split_options or SplitOptions()
        self.arrow_options = self.arrow_options or ArrowOptions()
        self.filter_predicate = self.filter_predicate or ""


class TableBatchScanResponse(serializers.JSONSerializableModel):
    __slots__ = ['status', 'request_id']

    session_id = serializers.JSONNodeField("SessionId")
    session_type = serializers.JSONNodeField("SessionType")
    session_status = serializers.JSONNodeField('SessionStatus', parse_callback=lambda s: SessionStatus(s.upper()))
    expiration_time = serializers.JSONNodeField("ExpirationTime")
    split_count = serializers.JSONNodeField("SplitsCount")
    record_count = serializers.JSONNodeField("RecordCount")
    data_schema = serializers.JSONNodeReferenceField(DataSchema, 'DataSchema')
    supported_data_format = serializers.JSONNodesReferencesField(DataFormat, "SupportedDataFormat")

    def __init__(self):
        super(TableBatchScanResponse, self).__init__()

        self.status = Status.INIT
        self.request_id = ""


class SessionRequest(object):
    def __init__(self, session_id):
        self.session_id = session_id


class TableBatchWriteRequest(serializers.JSONSerializableModel):
    dynamic_partition_options = serializers.JSONNodeReferenceField(DynamicPartitionOptions, "DynamicPartitionOptions")
    arrow_options = serializers.JSONNodeReferenceField(ArrowOptions, "ArrowOptions")
    overwrite = serializers.JSONNodeField("Overwrite")
    partition_spec = serializers.JSONNodeField("PartitionSpec")
    support_write_cluster = serializers.JSONNodeField("SupportWriteCluster")

    def __init__(self, **kwargs):
        super(TableBatchWriteRequest, self).__init__(**kwargs)

        self.partition_spec = self.partition_spec or ""
        self.arrow_options = self.arrow_options or ArrowOptions()
        self.dynamic_partition_options = self.dynamic_partition_options or DynamicPartitionOptions()
        self.overwrite = self.overwrite or True
        self.support_write_cluster = self.support_write_cluster or False


class TableBatchWriteResponse(serializers.JSONSerializableModel):
    __slots__ = ['status', "request_id"]

    session_status = serializers.JSONNodeField('SessionStatus', parse_callback=lambda s: SessionStatus(s.upper()))
    expiration_time = serializers.JSONNodeField("ExpirationTime")
    session_id = serializers.JSONNodeField("SessionId")
    data_schema = serializers.JSONNodeReferenceField(DataSchema, "DataSchema")
    supported_data_format = serializers.JSONNodesReferencesField(DataFormat, "SupportedDataFormat")
    max_block_num = serializers.JSONNodeField("MaxBlockNumber")
    required_ordering = serializers.JSONNodesReferencesField(Order, "RequiredOrdering")
    required_distribution = serializers.JSONNodeReferenceField(RequiredDistribution, "RequiredDistribution")

    def __init__(self):
        super(TableBatchWriteResponse, self).__init__()

        self.status = Status.INIT
        self.request_id = ""


class ReadRowsRequest(object):
    def __init__(self, session_id, split_index=0,
                 row_index=0, row_count=0, max_batch_rows=4096,
                 compression=Compression.LZ4_FRAME,
                 data_format=DataFormat()):
        self.session_id = session_id
        self.split_index = split_index
        self.row_index = row_index
        self.row_count = row_count
        self.max_batch_rows = max_batch_rows
        self.compression = compression
        self.data_format = data_format


class ReadRowsResponse(object):
    def __init__(self):
        self.status = Status.INIT
        self.request_id = ""


class WriteRowsRequest(object):
    def __init__(self, session_id,
                 block_number=0, attempt_number=0, bucket_id=0,
                 compression=Compression.LZ4_FRAME,
                 data_format=DataFormat()):
        self.session_id = session_id
        self.block_number = block_number
        self.attempt_number = attempt_number
        self.bucket_id = bucket_id
        self.compression = compression
        self.data_format = data_format


class WriteRowsResponse(object):
    def __init__(self):
        self.status = Status.INIT
        self.request_id = ""
        self.commit_message = ""


def update_request_id(response, resp):
    if "x-odps-request-id" in resp.headers:
        response.request_id = resp.headers["x-odps-request-id"]


class StreamReader(IOBase):
    """Stream reader."""

    def __init__(self, download):
        self._stopped = False
        raw_reader = download()

        self._raw_reader = raw_reader
        # need to confirm read size
        self._chunk_size = 65536
        self._buffers = collections.deque()

    def readable(self):
        """Check whether this stream reader has been closed or not.

        Returns:
            Readable or not.
        """
        return not self._stopped

    def _read_chunk(self):
        buf = self._raw_reader.raw.read(self._chunk_size)
        return buf

    def _fill_next_buffer(self):
        data = self._read_chunk()
        if len(data) == 0:
            return

        self._buffers.append(BytesIO(data))

    def read(self, nbytes=None):
        """Read stream data from the server.

        Args:
            nbytes: The number of bytes to be read. All data will be read at once if set to None.

        Returns:
            Stream data. None means all the data has been read or there is error occurred.
        """
        if self._stopped:
            return b''

        total_size = 0
        bufs = []
        while nbytes is None or total_size < nbytes:
            if not self._buffers:
                self._fill_next_buffer()
                if not self._buffers:
                    break

            to_read = nbytes - total_size if nbytes is not None else None
            buf = self._buffers[0].read(to_read)

            if not buf:
                self._buffers.popleft()
            else:
                bufs.append(buf)
                total_size += len(buf)

        return b''.join(bufs)

    def get_status(self):
        """Get the status of the stream reader.

        Returns:
            Status.OK or Status.RUNNING.
        """
        if not self._stopped:
            return Status.RUNNING
        else:
            return Status.OK

    def get_request_id(self):
        """Get the request id.

        Returns:
            Request id.
        """
        if not self._stopped:
            logger.error("The reader is not closed yet, please wait")
            return None

        if self._raw_reader is not None and "x-odps-request-id" in self._raw_reader.headers:
            return self._raw_reader.headers["x-odps-request-id"]
        else:
            return None

    def close(self):
        """If there is no data can be read from server, it will be called to close the stream reader."""
        self._stopped = True


class StreamWriter(IOBase):
    """Stream writer."""

    def __init__(self, upload):
        self._writer = upload()
        self._writer.open()
        self._res = None
        self._stopped = False

    def writable(self):
        """Check whether this stream writer has been closed or not.

        Returns:
            Writable or not.
        """
        return not self._stopped

    def write(self, data):
        """Write data to the server.

        Returns:
            Success or not.
        """
        if self._stopped:
            return False

        self._writer.write(data)
        return True

    def finish(self):
        """The stream writer is not expected to write data if finish has been called.

        Returns:
            Commit message returned from the server. User should bring this message to do the write session commit.
            Success or not.
        """
        self._stopped = True
        self._writer.close()
        self._res = self._writer.result

        if self._res is not None and self._res.status_code == codes['ok']:
            resp_json = self._res.json()
            return resp_json["CommitMessage"], True
        else:
           return None, False

    def get_status(self):
        """Get the status of this stream writer.

        Returns:
            Status.OK or Status.RUNNING.
        """
        if not self._stopped:
            return Status.RUNNING
        else:
            return Status.OK

    def get_request_id(self):
        """Get the request id.

        Returns:
            Request id.
        """
        if not self._stopped:
            logger.error("The writer is not closed yet, please close first")
            return None

        if self._res is not None and "x-odps-request-id" in self._res.headers:
            return self._res.headers["x-odps-request-id"]
        else:
            return None


class ArrowReader(object):
    """Arrow batch reader."""

    def __init__(self, stream_reader):
        if pa is None:
            raise ValueError("To use arrow reader you need to install pyarrow")

        self._reader = stream_reader
        self._arrow_stream = None

    def _read_next_batch(self):
        if self._arrow_stream is None:
            self._arrow_stream = pa.ipc.open_stream(self._reader)

        try:
            batch = self._arrow_stream.read_next_batch()
            return batch
        except StopIteration:
            return None

    def read(self):
        """Read arrow batch from the server.

        Returns:
            Arrow record batch. None means all the data has been read or there is error occurred.
        """
        if not self._reader.readable():
            logger.error("Reader has been closed")
            return None

        batch = self._read_next_batch()
        if batch is None:
            self._reader.close()

        return batch

    def get_status(self):
        """Get the status of the arrow batch reader.

        Returns:
            Status.OK or Status.RUNNING.
        """
        return self._reader.get_status()

    def get_request_id(self):
        """Get the request id.

        Returns:
            Request id.
        """
        return self._reader.get_request_id()


class ArrowWriter(object):
    """Arrow batch writer."""

    def __init__(self, stream_writer, compression):
        self._arrow_writer = None
        self._compression = compression
        self._sink = stream_writer

    def write(self, record_batch):
        """Write one arrow batch to the server.

        Args:
            record_batch: The arrow batch to be written.
        Returns:
            Success or not.
        """
        if not self._sink.writable():
            logger.error("Writer has been closed")
            return False

        if self._arrow_writer is None:
            self._arrow_writer = pa.ipc.new_stream(
                self._sink,
                record_batch.schema,
                options=pa.ipc.IpcWriteOptions(compression=self._compression.to_string()),
            )

        self._arrow_writer.write_batch(record_batch)

        if not self._sink.writable():
            logger.error("Writer has been closed as exception occurred")
            return False

        return True

    def finish(self):
        """The arrow writer is not expected to write data if finish has been called.

        Returns:
            Commit message returned from the server. User should bring this message
            to do the write session commit.
            Success ot not.
        """
        if self._arrow_writer:
            self._arrow_writer.close()
        return self._sink.finish()

    def get_status(self):
        """Get the status of the arrow batch writer.

        Returns:
            Status.OK or Status.RUNNING.
        """
        return self._sink.get_status()

    def get_request_id(self):
        """Get the request id.

        Returns:
            Request id.
        """
        return self._sink.get_request_id()


class StorageApiClient(object):
    """Client to bundle configuration needed for API requests."""

    def __init__(self, odps: ODPS, table: Table, rest_endpoint: str = None):
        if isinstance(odps, ODPS) and isinstance(table, Table):
            self._odps = odps
            self._table = table
            self._rest_endpoint = rest_endpoint
            self._tunnel_rest = None
        else:
            raise ValueError("Please input odps configuration")

    @property
    def table(self):
        return self._table

    @property
    def tunnel_rest(self):
        if self._tunnel_rest is not None:
            return self._tunnel_rest

        from ...tunnel.tabletunnel import TableTunnel

        tunnel = TableTunnel(self._odps, endpoint=self._rest_endpoint)
        self._tunnel_rest = tunnel.tunnel_rest
        return self._tunnel_rest

    def _get_resource(self, *args) -> str:
        endpoint = self.tunnel_rest.endpoint + URL_PREFIX
        url = self._table.table_resource(endpoint=endpoint, force_schema=True)
        return "/".join([url] + list(args))

    def create_read_session(self, request: TableBatchScanRequest) -> TableBatchScanResponse:
        """Create a read session.

        Args:
            request: Table split parameters sent to the server.

        Returns:
            Read session response returned from the server.
        """
        if not isinstance(request, TableBatchScanRequest):
            raise ValueError("Use TableBatchScanRequest class to build request for create read session interface")

        json_str = request.serialize()

        url = self._get_resource("sessions")
        headers = {"Content-Type": "application/json"}
        if json_str != "":
            headers["Content-MD5"] = md5(to_binary(json_str)).hexdigest()
        params = {"session_type": "batch_read"}

        res = self.tunnel_rest.post(url, data=json_str, params=params, headers=headers)

        response = TableBatchScanResponse()
        response.parse(res, obj=response)
        response.status = Status.OK if res.status_code == codes['created'] else Status.WAIT
        update_request_id(response, res)

        return response

    def get_read_session(self, request: SessionRequest) -> TableBatchScanResponse:
        """Get the read session.

        Args:
            request: Read session parameters sent to the server.

        Returns:
            Read session response returned from the server.
        """
        if not isinstance(request, SessionRequest):
            raise ValueError("Use SessionRequest class to build request for get read session interface")

        url = self._get_resource("sessions", request.session_id)
        headers = {}
        params = {"session_type": "batch_read"}

        res = self.tunnel_rest.get(url, params=params, headers=headers)

        response = TableBatchScanResponse()
        response.parse(res, obj=response)
        response.status = Status.OK
        update_request_id(response, res)

        return response

    def read_rows_stream(self, request: ReadRowsRequest) -> StreamReader:
        """Read one split of the read session. Stream means the data read from server is serialized arrow record batch.

        Args:
            request: Batch split parameters sent to the server.

        Returns:
            Stream reader.
        """
        if not isinstance(request, ReadRowsRequest):
            raise ValueError("Use ReadRowsRequest class to build request for read rows interface")

        url = self._get_resource("data")
        headers = {
            "Connection": "Keep-Alive",
            "Accept-Encoding": request.compression.name if request.compression != Compression.UNCOMPRESSED else ""
        }
        params = {
            "session_id": request.session_id,
            "max_batch_rows": str(request.max_batch_rows),
            "split_index": str(request.split_index),
            "row_count": str(request.row_count),
            "row_index": str(request.row_index)
        }
        if request.data_format.type is not None:
            params["data_format_type"] = request.data_format.type
        if request.data_format.version is not None:
            params["data_format_version"] = request.data_format.version

        def download():
            return self.tunnel_rest.get(url, stream=True, params=params, headers=headers)

        return StreamReader(download)

    def create_write_session(self, request: TableBatchWriteRequest) -> TableBatchWriteResponse:
        """Create a write session.

        Args:
            request: Table write parameters sent to the server.

        Returns:
            Write session response returned from the server.
        """
        if not isinstance(request, TableBatchWriteRequest):
            raise ValueError("Use TableBatchWriteRequest class to build request for create write session interface")

        json_str = request.serialize()

        url = self._get_resource("sessions")
        headers = {"Content-Type": "application/json"}
        if json_str != "":
            headers["Content-MD5"] = md5(to_binary(json_str)).hexdigest()
        params = {"session_type": "batch_write"}

        res = self.tunnel_rest.post(url, data=json_str, params=params, headers=headers)

        response = TableBatchWriteResponse()
        response.parse(res, obj=response)
        response.status = Status.OK
        update_request_id(response, res)

        return response

    def get_write_session(self, request: SessionRequest) -> TableBatchWriteResponse:
        """Get a write session.

        Args:
            request: Write session parameters sent to the server.

        Returns:
            Write session response returned from the server.
        """
        if not isinstance(request, SessionRequest):
            raise ValueError("Use SessionRequest class to build request for get write session interface")

        url = self._get_resource("sessions", request.session_id)
        headers = {}
        params = {"session_type": "batch_write"}

        res = self.tunnel_rest.get(url, params=params, headers=headers)

        response = TableBatchWriteResponse()
        response.parse(res, obj=response)
        response.status = Status.OK
        update_request_id(response, res)

        return response

    def write_rows_stream(self, request: WriteRowsRequest) -> StreamWriter:
        """Write one block of data to the write session. Stream means the data written to server is serialized arrow record batch.

        Args:
            request: Batch write parameters sent to the server.

        Returns:
            Stream writer.
        """
        if not isinstance(request, WriteRowsRequest):
            raise ValueError("Use WriteRowsRequest class to build request for write rows interface")

        url = self._get_resource("sessions", request.session_id, "data")
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Encoding": "deflate",
            "Transfer-Encoding": "chunked"
        }

        params = {
            "attempt_number": str(request.attempt_number),
            "block_number": str(request.block_number)
        }
        if request.data_format.type != None:
            params["data_format_type"] = str(request.data_format.type)
        if request.data_format.version != None:
            params["data_format_version"] = str(request.data_format.version)

        def upload():
            return self.tunnel_rest.post(url, params=params, headers=headers, file_upload=True)

        return StreamWriter(upload)

    def commit_write_session(self, request: SessionRequest, commit_msg: list) -> TableBatchWriteResponse:
        """Commit the write session after write the last stream data.

        Args:
            request: Commit write session parameters sent to the server.
            commit_msg: Commit messages collected from the write_rows_stream().

        Returns:
            Write session response returned from the server.
        """
        if not isinstance(request, SessionRequest):
            raise ValueError("Use SessionRequest class to build request for commit write session interface")
        if not isinstance(commit_msg, list):
            raise ValueError("Use list for commit message")

        commit_message_dict = {"CommitMessages": commit_msg}
        json_str = json.dumps(commit_message_dict)

        url = self._get_resource("commit")
        headers = {"Content-Type": "application/json"}
        params = {"session_id": request.session_id}

        res = self.tunnel_rest.post(url, data=json_str, params=params, headers=headers)

        response = TableBatchWriteResponse()
        response.parse(res, obj=response)
        response.status = Status.OK if res.status_code == codes['created'] else Status.WAIT
        update_request_id(response, res)

        return response


class StorageApiArrowClient(StorageApiClient):
    """Arrow batch client to bundle configuration needed for API requests."""
    def read_rows_arrow(self, request: ReadRowsRequest) -> ArrowReader:
        """Read one split of the read session.

        Args:
            request: Arrow batch split parameters sent to the server.

        Returns:
            Arrow batch reader.
        """
        if not isinstance(request, ReadRowsRequest):
            raise ValueError("Use ReadRowsRequest class to build request for read rows interface")

        return ArrowReader(self.read_rows_stream(request))

    def write_rows_arrow(self, request: WriteRowsRequest) -> ArrowWriter:
        """Write one block of data to the write session.

        Args:
            request: Arrow batch write parameters sent to the server.

        Returns:
            Arrow batch writer.
        """
        if not isinstance(request, WriteRowsRequest):
            raise ValueError("Use WriteRowsRequest class to build request for write rows interface")

        return ArrowWriter(self.write_rows_stream(request), request.compression)
