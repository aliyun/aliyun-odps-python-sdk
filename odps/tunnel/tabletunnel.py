#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import logging
import random
import sys
import time

import requests

from .. import errors, options, serializers, types, utils
from ..compat import Enum, six
from ..lib.monotonic import monotonic
from ..models import Projects, Record, TableSchema
from ..types import Column
from .base import TUNNEL_VERSION, BaseTunnel
from .errors import TunnelError, TunnelWriteTimeout
from .io.reader import ArrowRecordReader, TunnelArrowReader, TunnelRecordReader
from .io.stream import CompressOption, get_decompress_stream
from .io.writer import (
    ArrowWriter,
    BufferedArrowWriter,
    BufferedRecordWriter,
    RecordWriter,
    StreamRecordWriter,
    Upsert,
)

try:
    import numpy as np
except ImportError:
    np = None
try:
    import pyarrow as pa
except ImportError:
    pa = None

logger = logging.getLogger(__name__)
TUNNEL_DATA_TRANSFORM_VERSION = "v1"
DEFAULT_UPSERT_COMMIT_TIMEOUT = 120


def _wrap_upload_call(request_id):
    def wrapper(func):
        @six.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except requests.ConnectionError as ex:
                ex_str = str(ex)
                if "timed out" in ex_str:
                    raise TunnelWriteTimeout(ex_str, request_id=request_id)
                else:
                    raise

        return wrapped

    return wrapper


class BaseTableTunnelSession(serializers.JSONSerializableModel):
    @staticmethod
    def get_common_headers(content_length=None, chunked=False, tags=None):
        header = {
            "odps-tunnel-date-transform": TUNNEL_DATA_TRANSFORM_VERSION,
            "odps-tunnel-sdk-support-schema-evolution": "true",
            "x-odps-tunnel-version": TUNNEL_VERSION,
        }
        if content_length is not None:
            header["Content-Length"] = content_length
        if chunked:
            header.update(
                {
                    "Transfer-Encoding": "chunked",
                    "Content-Type": "application/octet-stream",
                }
            )
        tags = tags or options.tunnel.tags
        if tags:
            if isinstance(tags, six.string_types):
                tags = tags.split(",")
            header["odps-tunnel-tags"] = ",".join(tags)
        return header

    @staticmethod
    def normalize_partition_spec(partition_spec):
        if isinstance(partition_spec, six.string_types):
            partition_spec = types.PartitionSpec(partition_spec)
        if isinstance(partition_spec, types.PartitionSpec):
            partition_spec = str(partition_spec).replace("'", "")
        return partition_spec

    def get_common_params(self, **kwargs):
        params = {k: str(v) for k, v in kwargs.items()}
        if getattr(self, "_quota_name", None):
            params["quotaName"] = self._quota_name
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params["partition"] = self._partition_spec
        return params

    def check_tunnel_response(self, resp):
        if not self._client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

    def new_record(self, values=None):
        return Record(
            schema=self.schema,
            values=values,
            max_field_size=getattr(self, "max_field_size", None),
        )


class TableDownloadSession(BaseTableTunnelSession):
    __slots__ = (
        "_client",
        "_table",
        "_partition_spec",
        "_compress_option",
        "_quota_name",
        "_tags",
    )

    class Status(Enum):
        Unknown = "UNKNOWN"
        Normal = "NORMAL"
        Closes = "CLOSES"
        Expired = "EXPIRED"
        Initiating = "INITIATING"

    id = serializers.JSONNodeField("DownloadID")
    status = serializers.JSONNodeField(
        "Status", parse_callback=lambda s: TableDownloadSession.Status(s.upper())
    )
    count = serializers.JSONNodeField("RecordCount")
    schema = serializers.JSONNodeReferenceField(TableSchema, "Schema")
    quota_name = serializers.JSONNodeField("QuotaName")

    def __init__(
        self,
        client,
        table,
        partition_spec,
        download_id=None,
        compress_option=None,
        async_mode=True,
        timeout=None,
        quota_name=None,
        tags=None,
        **kw
    ):
        super(TableDownloadSession, self).__init__()

        self._client = client
        self._table = table
        self._partition_spec = self.normalize_partition_spec(partition_spec)

        self._quota_name = quota_name

        if "async_" in kw:
            async_mode = kw.pop("async_")
        if kw:
            raise TypeError("Cannot accept arguments %s" % ", ".join(kw.keys()))

        self._tags = tags or options.tunnel.tags
        if isinstance(self._tags, six.string_types):
            self._tags = self._tags.split(",")

        if download_id is None:
            self._init(async_mode=async_mode, timeout=timeout)
        else:
            self.id = download_id
            self.reload()
        self._compress_option = compress_option

        logger.info("Tunnel session created: %r", self)
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return "<TableDownloadSession id=%s project=%s table=%s partition_spec=%s>" % (
            self.id,
            self._table.project.name,
            self._table.name,
            self._partition_spec,
        )

    def _init(self, async_mode, timeout):
        params = self.get_common_params(downloads="")
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        if async_mode:
            params["asyncmode"] = "true"

        url = self._table.table_resource()
        try:
            resp = self._client.post(
                url, {}, params=params, headers=headers, timeout=timeout
            )
        except requests.exceptions.ReadTimeout:
            if callable(options.tunnel_session_create_timeout_callback):
                options.tunnel_session_create_timeout_callback(*sys.exc_info())
            raise
        self.check_tunnel_response(resp)

        delay_time = 0.1
        self.parse(resp, obj=self)
        while self.status == self.Status.Initiating:
            time.sleep(delay_time)
            delay_time = min(delay_time * 2, 5)
            self.reload()
        if self.schema is not None:
            self.schema.build_snapshot()

    def reload(self):
        params = self.get_common_params(downloadid=self.id)
        headers = self.get_common_headers(content_length=0, tags=self._tags)

        url = self._table.table_resource()
        resp = self._client.get(url, params=params, headers=headers)
        self.check_tunnel_response(resp)

        self.parse(resp, obj=self)
        if self.schema is not None:
            self.schema.build_snapshot()

    def _build_input_stream(
        self, start, count, compress=False, columns=None, arrow=False
    ):
        compress_option = self._compress_option or CompressOption()

        actions = ["data"]
        params = self.get_common_params(downloadid=self.id)
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        if compress:
            encoding = compress_option.algorithm.get_encoding()
            if encoding:
                headers["Accept-Encoding"] = encoding

        params["rowrange"] = "(%s,%s)" % (start, count)
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params["columns"] = ",".join(col_name(col) for col in columns)

        if arrow:
            actions.append("arrow")

        url = self._table.table_resource()
        resp = self._client.get(
            url, stream=True, actions=actions, params=params, headers=headers
        )
        self.check_tunnel_response(resp)

        content_encoding = resp.headers.get("Content-Encoding")
        if content_encoding is not None:
            compress_algo = CompressOption.CompressAlgorithm.from_encoding(
                content_encoding
            )
            if compress_algo != compress_option.algorithm:
                compress_option = self._compress_option = CompressOption(
                    compress_algo, -1, 0
                )
            compress = True
        else:
            compress = False

        option = compress_option if compress else None
        return get_decompress_stream(resp, option)

    def _open_reader(
        self,
        start,
        count,
        compress=False,
        columns=None,
        arrow=False,
        reader_cls=None,
        **kw
    ):
        pt_cols = (
            set(types.PartitionSpec(self._partition_spec).keys())
            if self._partition_spec
            else set()
        )
        reader_cols = [c for c in columns if c not in pt_cols] if columns else columns
        stream_kw = dict(compress=compress, columns=reader_cols, arrow=arrow)

        def stream_creator(cursor):
            return self._build_input_stream(start + cursor, count - cursor, **stream_kw)

        return reader_cls(self.schema, stream_creator, columns=columns, **kw)

    def open_record_reader(
        self, start, count, compress=False, columns=None, append_partitions=True
    ):
        return self._open_reader(
            start,
            count,
            compress=compress,
            columns=columns,
            append_partitions=append_partitions,
            partition_spec=self._partition_spec,
            reader_cls=TunnelRecordReader,
        )

    def open_arrow_reader(
        self, start, count, compress=False, columns=None, append_partitions=False
    ):
        return self._open_reader(
            start,
            count,
            compress=compress,
            columns=columns,
            arrow=True,
            append_partitions=append_partitions,
            partition_spec=self._partition_spec,
            reader_cls=TunnelArrowReader,
        )


class TableUploadSession(BaseTableTunnelSession):
    __slots__ = (
        "_client",
        "_table",
        "_partition_spec",
        "_compress_option",
        "_overwrite",
        "_quota_name",
        "_tags",
    )

    class Status(Enum):
        Unknown = "UNKNOWN"
        Normal = "NORMAL"
        Closing = "CLOSING"
        Closed = "CLOSED"
        Canceled = "CANCELED"
        Expired = "EXPIRED"
        Critical = "CRITICAL"

    id = serializers.JSONNodeField("UploadID")
    status = serializers.JSONNodeField(
        "Status", parse_callback=lambda s: TableUploadSession.Status(s.upper())
    )
    blocks = serializers.JSONNodesField("UploadedBlockList", "BlockID")
    schema = serializers.JSONNodeReferenceField(TableSchema, "Schema")
    max_field_size = serializers.JSONNodeField("MaxFieldSize")
    quota_name = serializers.JSONNodeField("QuotaName")

    def __init__(
        self,
        client,
        table,
        partition_spec,
        upload_id=None,
        compress_option=None,
        overwrite=False,
        quota_name=None,
        tags=None,
    ):
        super(TableUploadSession, self).__init__()

        self._client = client
        self._table = table
        self._partition_spec = self.normalize_partition_spec(partition_spec)

        self._quota_name = quota_name
        self._overwrite = overwrite

        self._tags = tags or options.tunnel.tags
        if isinstance(self._tags, six.string_types):
            self._tags = self._tags.split(",")

        if upload_id is None:
            self._init()
        else:
            self.id = upload_id
            self.reload()
        self._compress_option = compress_option

        logger.info("Tunnel session created: %r", self)
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return "<TableUploadSession id=%s project=%s table=%s partition_spec=%s>" % (
            self.id,
            self._table.project.name,
            self._table.name,
            self._partition_spec,
        )

    def _create_or_reload_session(self, reload=False):
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        params = self.get_common_params(reload=reload)
        if not reload and self._overwrite:
            params["overwrite"] = "true"

        if reload:
            params["uploadid"] = self.id
        else:
            params["uploads"] = 1

        def _call_tunnel(func, *args, **kw):
            resp = func(*args, **kw)
            self.check_tunnel_response(resp)
            return resp

        url = self._table.table_resource()
        if reload:
            resp = utils.call_with_retry(
                _call_tunnel, self._client.get, url, params=params, headers=headers
            )
        else:
            resp = utils.call_with_retry(
                _call_tunnel, self._client.post, url, {}, params=params, headers=headers
            )

        self.parse(resp, obj=self)
        if self.schema is not None:
            self.schema.build_snapshot()

    def _init(self):
        self._create_or_reload_session(reload=False)

    def reload(self):
        self._create_or_reload_session(reload=True)

    @classmethod
    def _iter_data_in_batches(cls, data):
        pos = 0
        chunk_size = options.chunk_size
        while pos < len(data):
            yield data[pos : pos + chunk_size]
            pos += chunk_size

    def _open_writer(
        self,
        block_id=None,
        compress=False,
        buffer_size=None,
        writer_cls=None,
        initial_block_id=None,
        block_id_gen=None,
    ):
        compress_option = self._compress_option or CompressOption()

        params = self.get_common_params(uploadid=self.id)
        headers = self.get_common_headers(chunked=True, tags=self._tags)
        if compress:
            # special: rewrite LZ4 to ARROW_LZ4 for arrow tunnels
            if (
                writer_cls is not None
                and issubclass(writer_cls, ArrowWriter)
                and compress_option.algorithm
                == CompressOption.CompressAlgorithm.ODPS_LZ4
            ):
                compress_option.algorithm = (
                    CompressOption.CompressAlgorithm.ODPS_ARROW_LZ4
                )
            encoding = compress_option.algorithm.get_encoding()
            if encoding:
                headers["Content-Encoding"] = encoding

        url = self._table.table_resource()
        option = compress_option if compress else None

        if block_id is None:

            @_wrap_upload_call(self.id)
            def upload_block(blockid, data):
                params["blockid"] = blockid

                def upload_func():
                    if isinstance(data, (bytes, bytearray)):
                        to_upload = self._iter_data_in_batches(data)
                    else:
                        to_upload = data
                    return self._client.put(
                        url, data=to_upload, params=params, headers=headers
                    )

                return utils.call_with_retry(upload_func)

            if writer_cls is ArrowWriter:
                writer_cls = BufferedArrowWriter
                params["arrow"] = ""
            else:
                writer_cls = BufferedRecordWriter

            writer = writer_cls(
                self.schema,
                upload_block,
                compress_option=option,
                buffer_size=buffer_size,
                block_id=initial_block_id,
                block_id_gen=block_id_gen,
            )
        else:
            params["blockid"] = block_id

            @_wrap_upload_call(self.id)
            def upload(data):
                return self._client.put(url, data=data, params=params, headers=headers)

            if writer_cls is ArrowWriter:
                params["arrow"] = ""

            writer = writer_cls(self.schema, upload, compress_option=option)
        return writer

    def open_record_writer(
        self,
        block_id=None,
        compress=False,
        buffer_size=None,
        initial_block_id=None,
        block_id_gen=None,
    ):
        return self._open_writer(
            block_id=block_id,
            compress=compress,
            buffer_size=buffer_size,
            initial_block_id=initial_block_id,
            block_id_gen=block_id_gen,
            writer_cls=RecordWriter,
        )

    def open_arrow_writer(
        self,
        block_id=None,
        compress=False,
        buffer_size=None,
        initial_block_id=None,
        block_id_gen=None,
    ):
        return self._open_writer(
            block_id=block_id,
            compress=compress,
            buffer_size=buffer_size,
            initial_block_id=initial_block_id,
            block_id_gen=block_id_gen,
            writer_cls=ArrowWriter,
        )

    def get_block_list(self):
        self.reload()
        return self.blocks

    def commit(self, blocks):
        if blocks is None:
            raise ValueError("Invalid parameter: blocks.")
        if isinstance(blocks, six.integer_types):
            blocks = [blocks]

        server_block_map = dict(
            [(int(block_id), True) for block_id in self.get_block_list()]
        )
        client_block_map = dict([(int(block_id), True) for block_id in blocks])

        if len(server_block_map) != len(client_block_map):
            raise TunnelError(
                "Blocks not match, server: %s, tunnelServerClient: %s. "
                "Make sure all block writers closed or with-blocks exited."
                % (len(server_block_map), len(client_block_map))
            )

        for block_id in blocks:
            if block_id not in server_block_map:
                raise TunnelError(
                    "Block not exists on server, block id is %s" % (block_id,)
                )

        self._complete_upload()

    def _complete_upload(self):
        headers = self.get_common_headers()
        params = self.get_common_params(uploadid=self.id)
        url = self._table.table_resource()

        resp = utils.call_with_retry(
            self._client.post,
            url,
            "",
            params=params,
            headers=headers,
            exc_type=(
                requests.Timeout,
                requests.ConnectionError,
                errors.InternalServerError,
            ),
        )
        self.parse(resp, obj=self)


class Slot(object):
    def __init__(self, slot, server):
        self._slot = slot
        self._ip = None
        self._port = None
        self.set_server(server, True)

    @property
    def slot(self):
        return self._slot

    @property
    def ip(self):
        return self._ip

    @property
    def port(self):
        return self._port

    @property
    def server(self):
        return str(self._ip) + ":" + str(self._port)

    def set_server(self, server, check_empty=False):
        if len(server.split(":")) != 2:
            raise TunnelError("Invalid slot format: {}".format(server))

        ip, port = server.split(":")

        if check_empty:
            if (not ip) or (not port):
                raise TunnelError("Empty server ip or port")
        if ip:
            self._ip = ip
        if port:
            self._port = int(port)


class TableStreamUploadSession(BaseTableTunnelSession):
    __slots__ = (
        "_client",
        "_table",
        "_partition_spec",
        "_compress_option",
        "_quota_name",
        "_create_partition",
        "_zorder_columns",
        "_allow_schema_mismatch",
        "_schema_version_reloader",
        "_tags",
    )

    class Slots(object):
        def __init__(self, slot_elements):
            self._slots = []
            self._cur_index = -1
            for value in slot_elements:
                if len(value) != 2:
                    raise TunnelError("Invalid slot routes")
                self._slots.append(Slot(value[0], value[1]))

            if len(self._slots) > 0:
                self._cur_index = random.randint(0, len(self._slots))
            self._iter = iter(self)

        def __len__(self):
            return len(self._slots)

        def __next__(self):
            return next(self._iter)

        def __iter__(self):
            while True:
                if self._cur_index < 0:
                    yield None
                else:
                    self._cur_index += 1
                    if self._cur_index >= len(self._slots):
                        self._cur_index = 0
                    yield self._slots[self._cur_index]

    schema = serializers.JSONNodeReferenceField(TableSchema, "schema")
    id = serializers.JSONNodeField("session_name")
    status = serializers.JSONNodeField("status")
    slots = serializers.JSONNodeField(
        "slots", parse_callback=lambda val: TableStreamUploadSession.Slots(val)
    )
    quota_name = serializers.JSONNodeField("QuotaName")
    schema_version = serializers.JSONNodeField("schema_version")

    def __init__(
        self,
        client,
        table,
        partition_spec,
        compress_option=None,
        quota_name=None,
        create_partition=False,
        zorder_columns=None,
        schema_version=None,
        allow_schema_mismatch=True,
        upload_id=None,
        tags=None,
        schema_version_reloader=None,
    ):
        super(TableStreamUploadSession, self).__init__()

        self._client = client
        self._table = table
        self._partition_spec = self.normalize_partition_spec(partition_spec)

        self._quota_name = quota_name
        self._create_partition = create_partition
        self._zorder_columns = zorder_columns
        self._allow_schema_mismatch = allow_schema_mismatch
        self.schema_version = schema_version
        self._schema_version_reloader = schema_version_reloader

        self._tags = tags or options.tunnel.tags
        if isinstance(self._tags, six.string_types):
            self._tags = self._tags.split(",")

        if upload_id is None:
            if not allow_schema_mismatch and not schema_version:
                self._init_with_latest_schema()
            else:
                self._init()
        else:
            self.id = upload_id
            self.reload()
        self._compress_option = compress_option

        logger.info("Tunnel session created: %r", self)
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return (
            "<TableStreamUploadSession id=%s project=%s table=%s partition_spec=%s>"
            % (
                self.id,
                self._table.project.name,
                self._table.name,
                self._partition_spec,
            )
        )

    def _init(self):
        params = self.get_common_params()
        headers = self.get_common_headers(content_length=0, tags=self._tags)

        if self._create_partition:
            params["create_partition"] = ""
        if self.schema_version is not None:
            params["schema_version"] = str(self.schema_version)
        if self._zorder_columns:
            cols = self._zorder_columns
            if not isinstance(self._zorder_columns, six.string_types):
                cols = ",".join(self._zorder_columns)
            params["zorder_columns"] = cols
        params["check_latest_schema"] = str(not self._allow_schema_mismatch).lower()

        url = self._get_resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        self.check_tunnel_response(resp)

        self.parse(resp, obj=self)
        self._quota_name = self.quota_name
        if self.schema is not None:
            self.schema.build_snapshot()

    def _init_with_latest_schema(self):
        def init_with_table_version():
            self.schema_version = self._schema_version_reloader()
            self._init()

        return utils.call_with_retry(
            init_with_table_version, retry_times=None, exc_type=errors.NoSuchSchema
        )

    def _get_resource(self):
        return self._table.table_resource() + "/streams"

    def reload(self):
        params = self.get_common_params(uploadid=self.id)
        headers = self.get_common_headers(content_length=0, tags=self._tags)

        url = self._get_resource()
        resp = self._client.get(url, params=params, headers=headers)
        self.check_tunnel_response(resp)

        self.parse(resp, obj=self)
        self._quota_name = self.quota_name
        if self.schema is not None:
            self.schema.build_snapshot()

    def abort(self):
        params = self.get_common_params(uploadid=self.id)

        slot = next(iter(self.slots))
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        headers["odps-tunnel-routed-server"] = slot.server

        url = self._get_resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        self.check_tunnel_response(resp)

    def reload_slots(self, slot, server, slot_num):
        if len(self.slots) != slot_num:
            self.reload()
        else:
            slot.set_server(server)

    def _open_writer(self, compress=False):
        compress_option = self._compress_option or CompressOption()

        slot = next(iter(self.slots))

        headers = self.get_common_headers(chunked=True, tags=self._tags)
        headers.update(
            {
                "odps-tunnel-slot-num": str(len(self.slots)),
                "odps-tunnel-routed-server": slot.server,
            }
        )

        if compress:
            encoding = compress_option.algorithm.get_encoding()
            if encoding:
                headers["Content-Encoding"] = encoding

        params = self.get_common_params(uploadid=self.id, slotid=slot.slot)
        url = self._get_resource()
        option = compress_option if compress else None

        @_wrap_upload_call(self.id)
        def upload_block(data):
            return self._client.put(url, data=data, params=params, headers=headers)

        writer = StreamRecordWriter(
            self.schema, upload_block, session=self, slot=slot, compress_option=option
        )

        return writer

    def open_record_writer(self, compress=False):
        return self._open_writer(compress=compress)


class TableUpsertSession(BaseTableTunnelSession):
    __slots__ = (
        "_client",
        "_table",
        "_partition_spec",
        "_compress_option",
        "_slot_num",
        "_commit_timeout",
        "_quota_name",
        "_lifecycle",
        "_tags",
    )

    UPSERT_EXTRA_COL_NUM = 5
    UPSERT_VERSION_KEY = "__version"
    UPSERT_APP_VERSION_KEY = "__app_version"
    UPSERT_OPERATION_KEY = "__operation"
    UPSERT_KEY_COLS_KEY = "__key_cols"
    UPSERT_VALUE_COLS_KEY = "__value_cols"

    class Status(Enum):
        Normal = "NORMAL"
        Committing = "COMMITTING"
        Committed = "COMMITTED"
        Expired = "EXPIRED"
        Critical = "CRITICAL"
        Aborted = "ABORTED"

    class Slots(object):
        def __init__(self, slot_elements):
            self._slots = []
            self._buckets = {}
            for value in slot_elements:
                slot = Slot(value["slot_id"], value["worker_addr"])
                self._slots.append(slot)
                self._buckets.update({idx: slot for idx in value["buckets"]})

            for idx in self._buckets.keys():
                if idx > len(self._buckets):
                    raise TunnelError("Invalid bucket value: " + str(idx))

        @property
        def buckets(self):
            return self._buckets

        def __len__(self):
            return len(self._slots)

    schema = serializers.JSONNodeReferenceField(TableSchema, "schema")
    id = serializers.JSONNodeField("id")
    status = serializers.JSONNodeField(
        "status", parse_callback=lambda s: TableUpsertSession.Status(s.upper())
    )
    slots = serializers.JSONNodeField(
        "slots", parse_callback=lambda val: TableUpsertSession.Slots(val)
    )
    quota_name = serializers.JSONNodeField("quota_name")
    hash_keys = serializers.JSONNodeField("hash_key")
    hasher = serializers.JSONNodeField("hasher")

    def __init__(
        self,
        client,
        table,
        partition_spec,
        compress_option=None,
        slot_num=1,
        commit_timeout=DEFAULT_UPSERT_COMMIT_TIMEOUT,
        lifecycle=None,
        quota_name=None,
        upsert_id=None,
        tags=None,
    ):
        super(TableUpsertSession, self).__init__()

        self._client = client
        self._table = table
        self._partition_spec = self.normalize_partition_spec(partition_spec)
        self._lifecycle = lifecycle
        self._quota_name = quota_name

        self._slot_num = slot_num
        self._commit_timeout = commit_timeout

        self._tags = tags or options.tunnel.tags
        if isinstance(self._tags, six.string_types):
            self._tags = self._tags.split(",")

        if upsert_id is None:
            self._init()
        else:
            self.id = upsert_id
            self.reload()
        self._compress_option = compress_option

        logger.info("Upsert session created: %r", self)
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return "<TableUpsertSession id=%s project=%s table=%s partition_spec=%s>" % (
            self.id,
            self._table.project.name,
            self._table.name,
            self._partition_spec,
        )

    @property
    def endpoint(self):
        return self._client.endpoint

    @property
    def buckets(self):
        return self.slots.buckets

    def _get_resource(self):
        return self._table.table_resource() + "/upserts"

    def _patch_schema(self):
        if self.schema is None:
            return
        patch_schema = types.OdpsSchema(
            [
                Column(self.UPSERT_VERSION_KEY, "bigint"),
                Column(self.UPSERT_APP_VERSION_KEY, "bigint"),
                Column(self.UPSERT_OPERATION_KEY, "tinyint"),
                Column(self.UPSERT_KEY_COLS_KEY, "array<bigint>"),
                Column(self.UPSERT_VALUE_COLS_KEY, "array<bigint>"),
            ],
        )
        self.schema = self.schema.extend(patch_schema)
        self.schema.build_snapshot()

    def _init_or_reload(self, reload=False):
        params = self.get_common_params()
        headers = self.get_common_headers(content_length=0, tags=self._tags)

        if not reload:
            params["slotnum"] = str(self._slot_num)
        else:
            params["upsertid"] = self.id

        url = self._get_resource()
        if not reload:
            if self._lifecycle:
                params["lifecycle"] = self._lifecycle
            resp = self._client.post(url, {}, params=params, headers=headers)
        else:
            resp = self._client.get(url, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
            self._patch_schema()
        else:
            e = TunnelError.parse(resp)
            raise e

    def _init(self):
        self._init_or_reload()

    def new_record(self, values=None):
        if values:
            values = list(values) + [None] * 5
        return super(TableUpsertSession, self).new_record(values)

    def reload(self, init=False):
        self._init_or_reload(reload=True)

    def abort(self):
        params = self.get_common_params(upsertid=self.id)
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        headers["odps-tunnel-routed-server"] = self.slots.buckets[0].server

        url = self._get_resource()
        resp = self._client.delete(url, params=params, headers=headers)
        self.check_tunnel_response(resp)

    def open_upsert_stream(self, compress=False):
        params = self.get_common_params(upsertid=self.id)
        headers = self.get_common_headers(tags=self._tags)

        compress_option = self._compress_option or CompressOption()
        if not compress:
            compress_option = None
        else:
            encoding = compress_option.algorithm.get_encoding()
            if encoding:
                headers["Content-Encoding"] = encoding

        url = self._get_resource()

        @_wrap_upload_call(self.id)
        def upload_block(bucket, slot, record_count, data):
            req_params = params.copy()
            req_params.update(
                dict(
                    bucketid=bucket,
                    slotid=str(slot.slot),
                    record_count=str(record_count),
                )
            )
            req_headers = headers.copy()
            req_headers["odps-tunnel-routed-server"] = slot.server
            req_headers["Content-Length"] = len(data)
            return self._client.put(
                url, data=data, params=req_params, headers=req_headers
            )

        return Upsert(self.schema, upload_block, self, compress_option)

    def commit(self, async_=False):
        params = self.get_common_params(upsertid=self.id)
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        headers["odps-tunnel-routed-server"] = self.slots.buckets[0].server

        url = self._get_resource()
        resp = self._client.post(url, params=params, headers=headers)
        self.check_tunnel_response(resp)
        self.reload()

        if async_:
            return

        delay = 1
        start = monotonic()
        while self.status in (
            TableUpsertSession.Status.Committing,
            TableUpsertSession.Status.Normal,
        ):
            try:
                if monotonic() - start > self._commit_timeout:
                    raise TunnelError("Commit session timeout")
                time.sleep(delay)

                resp = self._client.post(url, params=params, headers=headers)
                self.check_tunnel_response(resp)
                self.reload()

                delay = min(8, delay * 2)
            except (errors.StreamSessionNotFound, errors.UpsertSessionNotFound):
                self.status = TableUpsertSession.Status.Committed
        if self.status != TableUpsertSession.Status.Committed:
            raise TunnelError("commit session failed, status: " + self.status.value)


class TableTunnel(BaseTunnel):
    def _get_tunnel_table(self, table, schema=None):
        project_odps = None
        try:
            project_odps = self._project.odps
            if isinstance(table, six.string_types):
                table = project_odps.get_table(table, project=self._project.name)
        except:
            pass

        project_name = self._project.name
        if not isinstance(table, six.string_types):
            project_name = table.project.name or project_name
            schema = schema or getattr(table.get_schema(), "name", None)
            table = table.name

        parent = Projects(client=self.tunnel_rest)[project_name]
        # tailor project for resource locating only
        parent._set_tunnel_defaults(odps_entry=project_odps)
        if schema is not None:
            parent = parent.schemas[schema]
        return parent.tables[table]

    @staticmethod
    def _build_compress_option(compress_algo=None, level=None, strategy=None):
        if compress_algo is None:
            return None
        return CompressOption(
            compress_algo=compress_algo, level=level, strategy=strategy
        )

    def create_download_session(
        self,
        table,
        async_mode=True,
        partition_spec=None,
        download_id=None,
        compress_option=None,
        compress_algo=None,
        compress_level=None,
        compress_strategy=None,
        schema=None,
        timeout=None,
        tags=None,
        **kw
    ):
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo,
            level=compress_level,
            strategy=compress_strategy,
        )
        if "async_" in kw:
            async_mode = kw.pop("async_")
        if kw:
            raise TypeError("Cannot accept arguments %s" % ", ".join(kw.keys()))
        return TableDownloadSession(
            self.tunnel_rest,
            table,
            partition_spec,
            download_id=download_id,
            compress_option=compress_option,
            async_mode=async_mode,
            timeout=timeout,
            quota_name=self._quota_name,
            tags=tags,
        )

    def create_upload_session(
        self,
        table,
        partition_spec=None,
        upload_id=None,
        compress_option=None,
        compress_algo=None,
        compress_level=None,
        compress_strategy=None,
        schema=None,
        overwrite=False,
        tags=None,
    ):
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo,
            level=compress_level,
            strategy=compress_strategy,
        )
        return TableUploadSession(
            self.tunnel_rest,
            table,
            partition_spec,
            upload_id=upload_id,
            compress_option=compress_option,
            overwrite=overwrite,
            quota_name=self._quota_name,
            tags=tags,
        )

    def create_stream_upload_session(
        self,
        table,
        partition_spec=None,
        compress_option=None,
        compress_algo=None,
        compress_level=None,
        compress_strategy=None,
        schema=None,
        schema_version=None,
        upload_id=None,
        tags=None,
        allow_schema_mismatch=True,
    ):
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo,
            level=compress_level,
            strategy=compress_strategy,
        )
        version_need_reloaded = [False]

        def schema_version_reloader():
            src_table = self._project.tables[table.name]
            if version_need_reloaded[0]:
                src_table.reload_extend_info()
            version_need_reloaded[0] = True
            return src_table.schema_version

        return TableStreamUploadSession(
            self.tunnel_rest,
            table,
            partition_spec,
            compress_option=compress_option,
            quota_name=self._quota_name,
            schema_version=schema_version,
            upload_id=upload_id,
            tags=tags,
            allow_schema_mismatch=allow_schema_mismatch,
            schema_version_reloader=schema_version_reloader,
        )

    def create_upsert_session(
        self,
        table,
        partition_spec=None,
        slot_num=1,
        commit_timeout=120,
        compress_option=None,
        compress_algo=None,
        compress_level=None,
        compress_strategy=None,
        schema=None,
        upsert_id=None,
        tags=None,
    ):
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo,
            level=compress_level,
            strategy=compress_strategy,
        )
        return TableUpsertSession(
            self.tunnel_rest,
            table,
            partition_spec,
            slot_num=slot_num,
            upsert_id=upsert_id,
            commit_timeout=commit_timeout,
            compress_option=compress_option,
            quota_name=self._quota_name,
            tags=tags,
        )

    def open_preview_reader(
        self,
        table,
        partition_spec=None,
        columns=None,
        limit=None,
        compress_option=None,
        compress_algo=None,
        compress_level=None,
        compress_strategy=None,
        arrow=True,
        timeout=None,
        make_compat=True,
        read_all=False,
        tags=None,
    ):
        if pa is None:
            raise ImportError("Need pyarrow to run open_preview_reader.")

        tunnel_table = self._get_tunnel_table(table)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo,
            level=compress_level,
            strategy=compress_strategy,
        )

        params = {"limit": str(limit) if limit else "-1"}
        partition_spec = BaseTableTunnelSession.normalize_partition_spec(partition_spec)
        if columns:
            col_set = set(columns)
            ordered_col = [c.name for c in table.table_schema if c.name in col_set]
            params["columns"] = ",".join(ordered_col)
        if partition_spec is not None and len(partition_spec) > 0:
            params["partition"] = partition_spec

        headers = BaseTableTunnelSession.get_common_headers(content_length=0, tags=tags)
        if compress_option:
            encoding = compress_option.algorithm.get_encoding(legacy=False)
            if encoding:
                headers["Accept-Encoding"] = encoding

        url = tunnel_table.table_resource(force_schema=True) + "/preview"
        resp = self.tunnel_rest.get(
            url, stream=True, params=params, headers=headers, timeout=timeout
        )
        if not self.tunnel_rest.is_ok(resp):  # pragma: no cover
            e = TunnelError.parse(resp)
            raise e

        input_stream = get_decompress_stream(resp)
        if input_stream.peek() is None:
            # stream is empty, replace with empty stream
            input_stream = None

        def stream_creator(pos):
            # part retry not supported currently
            assert pos == 0
            return input_stream

        reader = TunnelArrowReader(
            table.table_schema, stream_creator, columns=columns, use_ipc_stream=True
        )
        if not arrow:
            reader = ArrowRecordReader(
                reader, make_compat=make_compat, read_all=read_all
            )
        return reader
