#!/usr/bin/env python
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

import enum
import functools
import logging
import random
import sys
import threading
import time
import weakref
from datetime import datetime

import requests

from .. import errors, options, serializers, types, utils
from ..models import Projects, Record, TableSchema
from ..types import Column
from .base import TUNNEL_VERSION, BaseTunnel, TunnelRetryMixin
from .errors import TunnelError, TunnelReadTimeout, TunnelWriteTimeout
from .io.reader import (
    ArrowRecordReader,
    BufferedRecordReader,
    TunnelArrowReader,
    TunnelRecordReader,
)
from .io.stream import CompressOption, get_decompress_stream
from .io.writer import (
    ArrowWriter,
    BufferedArrowWriter,
    BufferedRecordWriter,
    RecordWriter,
    StreamRecordWriter,
    Upsert,
)
from .retry import BAD_GATEWAY, GATEWAY_TIMEOUT, OptionsRetryPolicy, TunnelRetryHandler

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
        @functools.wraps(func)
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


class BaseTableTunnelSession(serializers.JSONSerializableModel, TunnelRetryMixin):
    __slots__ = ("_retry_handler",)

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
            if isinstance(tags, str):
                tags = tags.split(",")
            header["odps-tunnel-tags"] = ",".join(tags)
        return header

    @staticmethod
    def normalize_partition_spec(partition_spec):
        if isinstance(partition_spec, str):
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

    @classmethod
    def _get_default_compress_option(cls):
        if not options.tunnel.compress.enabled:
            return None
        return CompressOption(
            compress_algo=options.tunnel.compress.algo,
            level=options.tunnel.compress.level,
            strategy=options.tunnel.compress.strategy,
        )

    def new_record(self, values=None):
        """
        Generate a record of the current upload session.

        :param values: the values of this records
        :type values: list
        :return: record
        :rtype: :class:`odps.models.Record`

        :Example:

        >>> session = TableTunnel(o).create_upload_session('test_table')
        >>> record = session.new_record()
        >>> record[0] = 'my_name'
        >>> record[1] = 'my_id'
        >>> record = session.new_record(['my_name', 'my_id'])

        .. seealso:: :class:`odps.models.Record`
        """
        return Record(
            schema=self.schema,
            values=values,
            max_field_size=getattr(self, "max_field_size", None),
        )


class TableDownloadSession(BaseTableTunnelSession):
    """
    Tunnel session for downloading data from tables. Instances of this class
    should be created by :meth:`TableTunnel.create_download_session`.
    """

    __slots__ = (
        "_client",
        "_table",
        "_partition_spec",
        "_compress_option",
        "_quota_name",
        "_tags",
    )

    class Status(enum.Enum):
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
    support_read_by_raw_size = serializers.JSONNodeField(
        "SupportReadByRawSize", default=False
    )

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
            raise TypeError(f"Cannot accept arguments {', '.join(kw.keys())}")

        self._tags = tags or options.tunnel.tags
        if isinstance(self._tags, str):
            self._tags = self._tags.split(",")

        if download_id is None:
            self._init(async_mode=async_mode, timeout=timeout)
        else:
            self.id = download_id
            self.reload()
        self._compress_option = compress_option or self._get_default_compress_option()

        logger.info("Tunnel session created: %r", self)
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return f"<TableDownloadSession id={self.id} project={self._table.project.name} table={self._table.name} partition_spec={self._partition_spec!r}>"

    def _init(self, async_mode, timeout):
        params = self.get_common_params(downloads="")
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        if async_mode:
            params["asyncmode"] = "true"

        url = self._table.table_resource()
        ts = time.monotonic()
        try:
            resp = self.retry_handler.execute_with_retry_headers(
                lambda stamped: self._client.post(
                    url, {}, params=params, headers=stamped, timeout=timeout
                ),
                headers,
            )
        except requests.exceptions.ReadTimeout:
            if callable(options.tunnel_session_create_timeout_callback):
                options.tunnel_session_create_timeout_callback(*sys.exc_info())
            raise
        self.check_tunnel_response(resp)

        delay_time = 0.1
        self.parse(resp, obj=self)
        while self.status == self.Status.Initiating:
            if timeout and time.monotonic() - ts > timeout:
                try:
                    raise TunnelReadTimeout(
                        f"Waiting for tunnel ready timed out. id={self.id}, table={self._table.name}"
                    )
                except TunnelReadTimeout:
                    if callable(options.tunnel_session_create_timeout_callback):
                        options.tunnel_session_create_timeout_callback(*sys.exc_info())
                    raise
            time.sleep(delay_time)
            delay_time = min(delay_time * 2, 5)
            self.reload()
        if self.schema is not None:
            self.schema.build_snapshot()

    def reload(self):
        params = self.get_common_params(downloadid=self.id)
        headers = self.get_common_headers(content_length=0, tags=self._tags)

        url = self._table.table_resource()
        resp = self.retry_handler.execute_with_retry_headers(
            lambda stamped: self._client.get(url, params=params, headers=stamped),
            headers,
        )
        self.check_tunnel_response(resp)

        self.parse(resp, obj=self)
        if self.schema is not None:
            self.schema.build_snapshot()

    def _build_input_stream(
        self, start, count, compress=False, columns=None, arrow=False, raw_size=None
    ):
        compress_option = self._compress_option or CompressOption()

        actions = ["data"]
        params = self.get_common_params(downloadid=self.id)
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        if compress:
            encoding = compress_option.algorithm.get_encoding()
            if encoding:
                headers["Accept-Encoding"] = encoding

        params["rowrange"] = f"({start},{count})"
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params["columns"] = ",".join(col_name(col) for col in columns)

        if arrow:
            actions.append("arrow")
        if raw_size:
            params["raw_size"] = str(raw_size)

        url = self._table.table_resource()
        resp = self.retry_handler.execute_with_retry_headers(
            lambda stamped: self._client.get(
                url, stream=True, actions=actions, params=params, headers=stamped
            ),
            headers,
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
        compress=None,
        columns=None,
        arrow=False,
        on_exception=None,
        reader_cls=None,
        **kw
    ):
        pt_cols = (
            set(types.PartitionSpec(self._partition_spec).keys())
            if self._partition_spec
            else set()
        )
        reader_cols = [c for c in columns if c not in pt_cols] if columns else columns

        if compress is None:
            compress = self._compress_option is not None

        stream_kw = dict(compress=compress, columns=reader_cols, arrow=arrow)

        def stream_creator(cursor, row_number=None, raw_size=None):
            if cursor >= count:
                return None

            if row_number is None:
                row_number = count - cursor
            else:
                row_number = min(row_number, count - cursor)
            return self._build_input_stream(
                start + cursor, row_number, raw_size=raw_size, **stream_kw
            )

        return reader_cls(
            self.schema,
            stream_creator,
            columns=columns,
            on_exception=on_exception,
            **kw
        )

    def open_record_reader(
        self,
        start,
        count,
        compress=False,
        columns=None,
        append_partitions=True,
        buffered=False,
        buffer_size=None,
        row_batch_size=None,
        on_exception=None,
    ):
        """
        Open a reader to read data as records from the tunnel.

        :param int start: start row index
        :param int count: number of rows to read
        :param bool compress: whether to compress data
        :param columns: list of column names to read
        :param append_partitions: whether to append partition values as columns
        :param bool buffered: whether to use buffered reader
        :param int buffer_size: download buffer size in bytes. Num of rows read
            in every batch will be limited by this parameter as well as
            `row_batch_size`.
        :param bool row_batch_size: number of rows to read per batch. Num of
            rows read in every batch will be limited by this parameter as well
            as `buffer_size`.
        :param on_exception: custom error handling function accepting
            an Exception instance as input. If return value is True,
            error will be raised. Otherwise retry will continue.

        :return: a record reader
        :rtype: :class:`TunnelRecordReader`
        """
        reader_cls = BufferedRecordReader if buffered else TunnelRecordReader
        kw = {}
        if buffer_size:
            kw["buffer_size"] = buffer_size
        if row_batch_size:
            kw["row_batch_size"] = row_batch_size
        if buffered:
            kw["session_with_byte_size_limit"] = self.support_read_by_raw_size
        return self._open_reader(
            start,
            count,
            compress=compress,
            columns=columns,
            append_partitions=append_partitions,
            partition_spec=self._partition_spec,
            on_exception=on_exception,
            reader_cls=reader_cls,
            **kw
        )

    def open_arrow_reader(
        self,
        start,
        count,
        compress=False,
        columns=None,
        append_partitions=False,
        on_exception=None,
        buffered=False,
    ):
        """
        Open a reader to read data as Arrow format from the tunnel.

        :param int start: start row index
        :param int count: number of rows to read
        :param bool compress: whether to compress data
        :param columns: list of column names to read
        :param append_partitions: whether to append partition values
            as columns
        :param on_exception: custom error handling function accepting
            an Exception instance as input. If return value is True,
            error will be raised. Otherwise retry will continue.

        :return: an Arrow reader
        :rtype: :class:`TunnelArrowReader`
        """
        assert not buffered, "Buffered mode is not supported for Arrow reader."
        return self._open_reader(
            start,
            count,
            compress=compress,
            columns=columns,
            arrow=True,
            append_partitions=append_partitions,
            partition_spec=self._partition_spec,
            on_exception=on_exception,
            reader_cls=TunnelArrowReader,
        )


class TableUploadSession(BaseTableTunnelSession):
    """
    Tunnel session for uploading data to tables. Instances of this class
    should be created by :meth:`TableTunnel.create_upload_session`.
    """

    __slots__ = (
        "_client",
        "_table",
        "_partition_spec",
        "_compress_option",
        "_create_partition",
        "_overwrite",
        "_quota_name",
        "_tags",
    )

    class Status(enum.Enum):
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
        create_partition=None,
        overwrite=False,
        quota_name=None,
        tags=None,
    ):
        super(TableUploadSession, self).__init__()

        self._client = client
        self._table = table
        self._partition_spec = self.normalize_partition_spec(partition_spec)
        self._create_partition = create_partition

        self._quota_name = quota_name
        self._overwrite = overwrite

        self._tags = tags or options.tunnel.tags
        if isinstance(self._tags, str):
            self._tags = self._tags.split(",")

        if upload_id is None:
            self._init()
        else:
            self.id = upload_id
            self.reload()
        self._compress_option = compress_option or self._get_default_compress_option()

        logger.info("Tunnel session created: %r", self)
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        repr_args = f"id={self.id} project={self._table.project.name} table={self._table.name} partition_spec={self._partition_spec!r}"
        if self._overwrite:
            repr_args += " overwrite=True"
        return f"<TableUploadSession {repr_args}>"

    def _create_or_reload_session(self, reload=False):
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        params = self.get_common_params(reload=reload)
        if self._create_partition:
            params["create_partition"] = "true"
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
            resp = self.retry_handler.execute_with_retry_headers(
                lambda stamped: _call_tunnel(
                    self._client.get, url, params=params, headers=stamped
                ),
                headers,
            )
        else:
            resp = self.retry_handler.execute_with_retry_headers(
                lambda stamped: _call_tunnel(
                    self._client.post, url, {}, params=params, headers=stamped
                ),
                headers,
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
        compress=None,
        buffer_size=None,
        writer_cls=None,
        initial_block_id=None,
        block_id_gen=None,
        on_exception=None,
    ):
        compress_option = self._compress_option or CompressOption()

        params = self.get_common_params(uploadid=self.id)
        headers = self.get_common_headers(chunked=True, tags=self._tags)

        if compress is None:
            compress = self._compress_option is not None

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

                def upload_func(stamped):
                    if isinstance(data, (bytes, bytearray)):
                        to_upload = self._iter_data_in_batches(data)
                    else:
                        to_upload = data
                    return self._client.put(
                        url, data=to_upload, params=params, headers=stamped
                    )

                return self.retry_handler.execute_with_retry_headers(
                    upload_func, headers, on_exception=on_exception
                )

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
                # data is a generator from RequestsIO.data_generator() —
                # consumed once, cannot be replayed on retry.
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
        on_exception=None,
        initial_block_id=None,
        block_id_gen=None,
    ):
        """
        Open a writer to write data in records to the tunnel.

        :param int block_id: id of the block to write to. If not specified,
            a :class:`BufferedRecordWriter` will be created.
        :param int buffer_size: size of the buffer to use for buffered writers.
        :param bool compress: whether to compress data
        :param on_exception: custom error handling function accepting
            an Exception instance as input. If return value is True,
            error will be raised. Otherwise retry will continue.

        :return: a record writer
        :rtype: :class:`RecordWriter` or :class:`BufferedRecordWriter`
        """
        return self._open_writer(
            block_id=block_id,
            compress=compress,
            buffer_size=buffer_size,
            on_exception=on_exception,
            initial_block_id=initial_block_id,
            block_id_gen=block_id_gen,
            writer_cls=RecordWriter,
        )

    def open_arrow_writer(
        self,
        block_id=None,
        compress=False,
        buffer_size=None,
        on_exception=None,
        initial_block_id=None,
        block_id_gen=None,
    ):
        """
        Open a writer to write data in Arrow format to the tunnel.

        :param int block_id: id of the block to write to. If not specified,
            a :class:`BufferedArrowWriter` will be created.
        :param int buffer_size: size of the buffer to use for buffered writers.
        :param bool compress: whether to compress data
        :param on_exception: custom error handling function accepting
            an Exception instance as input. If return value is True,
            error will be raised. Otherwise retry will continue.

        :return: an Arrow writer
        :rtype: :class:`ArrowWriter` or :class:`BufferedArrowWriter`
        """
        return self._open_writer(
            block_id=block_id,
            compress=compress,
            buffer_size=buffer_size,
            on_exception=on_exception,
            initial_block_id=initial_block_id,
            block_id_gen=block_id_gen,
            writer_cls=ArrowWriter,
        )

    def get_block_list(self):
        self.reload()
        return self.blocks

    def commit(self, blocks):
        """
        Commit written blocks to the tunnel. Can be called only once on a single session.

        :param list blocks: list of block ids to commit
        """
        if blocks is None:
            raise ValueError("Invalid parameter: blocks.")
        if isinstance(blocks, int):
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
                raise TunnelError(f"Block not exists on server, block id is {block_id}")

        self._complete_upload()

    def _complete_upload(self):
        headers = self.get_common_headers()
        params = self.get_common_params(uploadid=self.id)
        url = self._table.table_resource()

        resp = self.retry_handler.execute_with_retry_headers(
            lambda stamped: self._client.post(url, "", params=params, headers=stamped),
            headers,
        )
        self.parse(resp, obj=self)


class Slot:
    """A tunnel slot route: ``slot_id`` + ``ip:port`` worker address.

    The constructor validates that both slot id and server are
    non-empty. ``set_server`` parses ``ip:port``; an invalid format or
    empty ip raises ``TunnelError`` so a stale address is never
    silently retained.
    """

    def __init__(self, slot, server):
        if slot is None or slot == "" or not server:
            raise TunnelError("Slot or Routed server is empty")
        self._slot = slot
        self._ip = None
        self._port = None
        self.set_server(server, check_empty=True)

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
        """Parse ``server`` (``ip:port``) and update the address.

        With ``check_empty`` (used by the constructor) an empty ip is
        also rejected. Without it — the reload path — an empty ip still
        raises, while the port is always re-parsed so a
        stale value can never linger.
        """
        segs = server.split(":")
        if len(segs) != 2:
            raise TunnelError(f"Invalid slot format: {server}")

        ip, port = segs
        if check_empty and (not ip or not port):
            raise TunnelError(f"Empty server ip or port: {server}")
        if not ip:
            raise TunnelError(f"Empty server ip: {server}")
        if not port:
            raise TunnelError(f"Empty server port: {server}")
        self._ip = ip
        self._port = int(port)

    def __eq__(self, other):
        if not isinstance(other, Slot):
            return NotImplemented
        return (
            self._slot == other._slot
            and self._ip == other._ip
            and self._port == other._port
        )

    # Slot is mutable (set_server mutates ip/port); compare by value and
    # stay unhashable so equal-but-mutated instances cannot key a dict.
    __hash__ = None


class TableStreamUploadSession(BaseTableTunnelSession):
    """
    Tunnel session for uploading data in stream method to tables. Instances
    of this class should be created by :meth:`TableTunnel.create_stream_upload_session`.
    """

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
        "_slot_num",
        "_dynamic_partition",
        "_reloading",
        "_last_reload_time",
        "_reload_throttle",
    )

    class Slots:
        """Round-robin slot iterator with a random start offset.

        ``next()`` is synchronized so concurrent writers do not observe
        the same slot index mid-increment.
        """

        def __init__(self, slot_elements, slot_idx=None):
            self._slots = []
            for value in slot_elements:
                if len(value) != 2:
                    raise TunnelError("Invalid slot routes")
                self._slots.append(Slot(value[0], value[1]))

            if slot_idx is not None:
                self._idx = slot_idx
            elif self._slots:
                self._idx = random.randint(0, len(self._slots) - 1)
            else:
                self._idx = 0
            self._lock = threading.Lock()

        def __len__(self):
            return len(self._slots)

        def next(self):
            """Return the next slot in round-robin order (thread-safe)."""
            if not self._slots:
                return None
            with self._lock:
                slot = self._slots[self._idx % len(self._slots)]
                self._idx += 1
                return slot

        def __next__(self):
            return self.next()

        def current(self):
            if not self._slots:
                return None
            with self._lock:
                return self._slots[
                    (self._idx + len(self._slots) - 1) % len(self._slots)
                ]

    schema = serializers.JSONNodeReferenceField(TableSchema, "schema")
    id = serializers.JSONNodeField("session_name")
    status = serializers.JSONNodeField("status")
    slots = serializers.JSONNodeField(
        "slots", parse_callback=lambda val: TableStreamUploadSession.Slots(val)
    )
    quota_name = serializers.JSONNodeField("QuotaName")
    schema_version = serializers.JSONNodeField("schema_version")
    last_batch_id = serializers.JSONNodeField(
        "last_batch_id", parse_callback=utils.skip_na_call(int)
    )
    last_batch_commit_time = serializers.JSONNodeField(
        "last_batch_commit_time",
        parse_callback=utils.skip_na_call(
            lambda v: datetime.fromtimestamp(int(v) / 1000.0)
        ),
    )

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
        slot_num=0,
        dynamic_partition=False,
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
        self._slot_num = slot_num
        self._dynamic_partition = dynamic_partition

        # Slot-route reload state. ``_reloading`` acts as a non-reentrant
        # CAS flag (acquired via ``acquire(blocking=False)``) so concurrent
        # non-force reloads collapse to a single in-flight request.
        self._reloading = threading.Lock()
        self._last_reload_time = 0.0
        self._reload_throttle = options.tunnel.stream_reload_throttle

        self._tags = tags or options.tunnel.tags
        if isinstance(self._tags, str):
            self._tags = self._tags.split(",")

        if upload_id is None:
            if not allow_schema_mismatch and not schema_version:
                self._init_with_latest_schema()
            else:
                self._init()
        else:
            self.id = upload_id
            self.reload()
        self._compress_option = compress_option or self._get_default_compress_option()

        logger.info("Tunnel session created: %r", self)
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return (
            f"<TableStreamUploadSession id={self.id}"
            f" project={self._table.project.name}"
            f" table={self._table.name}"
            f" partition_spec={self._partition_spec}>"
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
            if not isinstance(self._zorder_columns, str):
                cols = ",".join(self._zorder_columns)
            params["zorder_columns"] = cols
        if self._dynamic_partition:
            params["dynamic_partition"] = "true"
        if self._slot_num > 0:
            headers["odps-slot-num"] = str(self._slot_num)
        params["check_latest_schema"] = str(not self._allow_schema_mismatch).lower()

        url = self._get_resource()
        resp = self.retry_handler.execute_with_retry_headers(
            lambda stamped: self._client.post(url, {}, params=params, headers=stamped),
            headers,
        )
        self.check_tunnel_response(resp)

        self.parse(resp, obj=self)
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

    def reload(self, force=False, readonly=False):
        """Refresh the session's slot routes from the server.

        Parameters
        ----------
        force : bool
            Bypass the 30s throttle and the in-flight de-duplication, and
            always issue a GET. Used after slot-count changes and on
            502/504 gateway errors.
        readonly : bool
            Send ``read_only=true`` so the server returns metadata
            (last_batch_id / last_batch_commit_time) without touching the
            write routing state. Only meaningful for non-force queries.
        """
        if not force:
            # Non-force reload: throttle to one request per
            # ``_reload_throttle`` seconds and collapse concurrent callers
            # to a single in-flight request (CAS via non-blocking acquire).
            if time.monotonic() - self._last_reload_time < self._reload_throttle:
                return
            if not self._reloading.acquire(blocking=False):
                # Another thread is already reloading; skip.
                return
        else:
            # Force reload: block until we hold the lock so the caller
            # observes the refreshed routes after returning.
            self._reloading.acquire()

        try:
            params = self.get_common_params(uploadid=self.id)
            headers = self.get_common_headers(content_length=0, tags=self._tags)
            if self.schema_version is not None:
                params["schema_version"] = str(self.schema_version)
            if readonly:
                params["read_only"] = "true"

            url = self._get_resource()
            resp = self.retry_handler.execute_with_retry_headers(
                lambda stamped: self._client.get(url, params=params, headers=stamped),
                headers,
            )
            self.check_tunnel_response(resp)

            slot_idx = self.slots._idx
            self.parse(resp, obj=self)
            self.slots._idx = slot_idx
            if self.schema is not None:
                self.schema.build_snapshot()

            self._last_reload_time = time.monotonic()
        finally:
            self._reloading.release()

    def abort(self):
        """
        Abort the upload session.
        """
        params = self.get_common_params(uploadid=self.id)

        slot = next(self.slots)
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        headers["odps-tunnel-routed-server"] = slot.server

        url = self._get_resource()
        resp = self.retry_handler.execute_with_retry_headers(
            lambda stamped: self._client.post(url, {}, params=params, headers=stamped),
            headers,
        )
        self.check_tunnel_response(resp)

    def reload_slots(self, slot, server, slot_num):
        """Refresh routing after a successful write.

        If the server-reported slot count differs from the local count,
        force a full reload. Otherwise, if only this slot's routed server
        changed, update it in place without a request.
        """
        if len(self.slots) != slot_num:
            self.reload(force=True)
        else:
            if slot.server != server:
                slot.set_server(server)

    def _readonly_reload(self):
        """Non-force, readonly reload returning refreshed metadata."""
        self.reload(force=False, readonly=True)
        return self

    def get_last_batch_id(self):
        """Return the last committed batch id.

        Uses a non-force readonly reload, which is throttled by
        ``options.tunnel.stream_reload_throttle`` (default 30s): a call
        within the throttle window skips the server reload and returns
        the cached value, so the result may be up to that window stale.
        This mirrors the Java SDK ``StreamUploadSession.getLastBatchId``
        semantics (eventual consistency for monitoring accessors).
        """
        return self._readonly_reload().last_batch_id

    def get_last_batch_commit_time(self):
        """Return the last batch commit time.

        Uses a non-force readonly reload, throttled by
        ``options.tunnel.stream_reload_throttle`` (default 30s); the
        result may be up to that window stale. Mirrors the Java SDK
        ``StreamUploadSession.getLastBatchCommitTime`` semantics.
        """
        return self._readonly_reload().last_batch_commit_time

    def _get_upload_params(self, slot, compress=False):
        compress_option = self._compress_option or CompressOption()

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
        if self.schema_version is not None:
            params["schema_version"] = str(self.schema_version)
        if self._zorder_columns:
            cols = self._zorder_columns
            if not isinstance(self._zorder_columns, str):
                cols = ",".join(self._zorder_columns)
            params["zorder_columns"] = cols
        if self._dynamic_partition:
            params["dynamic_partition"] = "true"
        params["check_latest_schema"] = str(not self._allow_schema_mismatch).lower()

        url = self._get_resource()
        return url, headers, params

    def _open_writer(self, compress=False):
        slot = self.slots.next()
        option = (self._compress_option or CompressOption()) if compress else None
        # Mutable per-writer state resolved fresh on every attempt so a
        # mid-flight reload is honoured by the next retry.
        state = {"slot": slot, "url": None, "headers": None, "params": None}
        state["url"], state["headers"], state["params"] = self._get_upload_params(
            slot, compress=compress
        )

        def upload_block(data):
            @_wrap_upload_call(self.id)
            def do_put(ctx):
                chunk_size = options.chunk_size
                cur_url = state["url"]
                cur_headers = dict(state["headers"])
                cur_params = state["params"]
                ctx.inject_headers(cur_headers)

                def gen():
                    offset = 0
                    while offset < len(data):
                        yield data[offset : offset + chunk_size]
                        offset += chunk_size

                return self._client.put(
                    cur_url, data=gen(), params=cur_params, headers=cur_headers
                )

            def on_error_status(status_code):
                # 502/504: gateway error — the route is likely stale, so
                # force a full reload and re-resolve the slot before retry.
                if status_code in (BAD_GATEWAY, GATEWAY_TIMEOUT):
                    try:
                        self.reload(force=True)
                    except TunnelError:
                        # Swallow reload errors so the retry handler can
                        # decide based on the original failure; the next
                        # attempt reuses the previous slot.
                        return
                    new_slot = self.slots.current()
                    state["slot"] = new_slot
                    (
                        state["url"],
                        state["headers"],
                        state["params"],
                    ) = self._get_upload_params(new_slot, compress=compress)

            return self.retry_handler.execute_with_retry_ctx(
                do_put, on_error_status=on_error_status
            )

        writer = StreamRecordWriter(
            self.schema, upload_block, session=self, slot=slot, compress_option=option
        )

        return writer

    def open_record_writer(self, compress=False):
        """
        Open a writer to write data in records to the tunnel.

        :param bool compress: whether to compress data

        :return: a record writer
        :rtype: :class:`RecordWriter`
        """
        return self._open_writer(compress=compress)


class TableUpsertSession(BaseTableTunnelSession):
    """
    Tunnel session for inserting or updating data to upsert tables. Instances
    of this class should be created by :meth:`TableTunnel.create_upsert_session`.
    """

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
        "_buckets_lock",
        "_keepalive_scheduler",
        "_keepalive_interval",
        "_keepalive_lock",
        "_keepalive_stopped",
    )

    UPSERT_EXTRA_COL_NUM = 5
    UPSERT_VERSION_KEY = "__version"
    UPSERT_APP_VERSION_KEY = "__app_version"
    UPSERT_OPERATION_KEY = "__operation"
    UPSERT_KEY_COLS_KEY = "__key_cols"
    UPSERT_VALUE_COLS_KEY = "__value_cols"

    class Status(enum.Enum):
        Normal = "NORMAL"
        Committing = "COMMITTING"
        Committed = "COMMITTED"
        Expired = "EXPIRED"
        Critical = "CRITICAL"
        Aborted = "ABORTED"

    class Slots:
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
    support_partial_update = serializers.JSONNodeField("enable_partial_update")

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
        if isinstance(self._tags, str):
            self._tags = self._tags.split(",")

        # Read-write lock guarding the buckets map. Reload replaces the
        # whole map (write); update_buckets / get_buckets read it.
        self._buckets_lock = threading.RLock()
        self._keepalive_interval = options.tunnel.upsert_keepalive_interval
        self._keepalive_scheduler = None
        self._keepalive_lock = threading.Lock()
        self._keepalive_stopped = False
        if upsert_id is None:
            self._init()
        else:
            self.id = upsert_id
            self.reload()
        self._compress_option = compress_option or self._get_default_compress_option()

        # Start the background keepalive once the session is loaded.
        self._start_keepalive()

        logger.info("Upsert session created: %r", self)
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return f"<TableUpsertSession id={self.id} project={self._table.project.name} table={self._table.name} partition_spec={self._partition_spec}>"

    @property
    def endpoint(self):
        return self._client.endpoint

    @property
    def buckets(self):
        with self._buckets_lock:
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
            if self._lifecycle and 0 < self._lifecycle <= 24:
                params["lifecycle"] = str(self._lifecycle)
            resp = self.retry_handler.execute_with_retry_headers(
                lambda stamped: self._client.post(
                    url, {}, params=params, headers=stamped
                ),
                headers,
            )
        else:
            resp = self.retry_handler.execute_with_retry_headers(
                lambda stamped: self._client.get(url, params=params, headers=stamped),
                headers,
            )
        if self._client.is_ok(resp):
            with self._buckets_lock:
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

    def get_buckets(self):
        """Return a snapshot copy of the current bucket→slot map."""
        with self._buckets_lock:
            return dict(self.slots.buckets)

    def update_buckets(self, bucket_id, new_slot_server):
        """Refresh routing for a single bucket after a 308 response.

        If ``new_slot_server`` is falsy, fall back to a full reload
        (GET session). Otherwise update only the slot serving
        ``bucket_id`` in place, without a request.
        """
        if not new_slot_server:
            self.reload()
            return
        with self._buckets_lock:
            slot = self.slots.buckets.get(bucket_id)
            if slot is not None and slot.server != new_slot_server:
                slot.set_server(new_slot_server)

    @staticmethod
    def _keepalive_tick(sess_ref):
        """One keepalive tick; no-op if the session has been GC'd."""
        sess = sess_ref()
        if sess is None:
            return
        try:
            sess.reload()
            if sess.status not in (
                TableUpsertSession.Status.Normal,
                TableUpsertSession.Status.Committing,
            ):
                sess._stop_keepalive()
                return
        except Exception:
            logger.debug("Upsert keepalive reload failed", exc_info=True)
        with sess._keepalive_lock:
            if sess._keepalive_stopped:
                return
            sess._keepalive_scheduler = sess._arm_keepalive(sess_ref)

    def _arm_keepalive(self, sess_ref):
        """Create and start the next keepalive timer (caller holds the lock)."""
        timer = threading.Timer(
            self._keepalive_interval,
            TableUpsertSession._keepalive_tick,
            args=(sess_ref,),
        )
        timer.daemon = True
        timer.start()
        return timer

    def _start_keepalive(self):
        """Start the background keepalive that refreshes buckets.

        Re-armed after each tick (``scheduleAtFixedRate`` semantics).
        Stops when status leaves NORMAL/COMMITTING.  The timer callback
        holds only a :class:`weakref.ref` to the session so a dead
        session is never kept alive.
        """
        with self._keepalive_lock:
            if self._keepalive_scheduler is not None or self._keepalive_stopped:
                return
            if self._keepalive_interval <= 0:
                # Non-positive interval disables keepalive; never arm a
                # Timer(0) that re-arms into an unthrottled reload loop.
                return
            self._keepalive_scheduler = self._arm_keepalive(weakref.ref(self))

    def _stop_keepalive(self):
        with self._keepalive_lock:
            self._keepalive_stopped = True
            timer = self._keepalive_scheduler
            self._keepalive_scheduler = None
        if timer is not None:
            timer.cancel()

    def close(self):
        """Stop the background keepalive scheduler."""
        self._stop_keepalive()

    def abort(self):
        """
        Abort the current session.
        """
        params = self.get_common_params(upsertid=self.id)
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        headers["odps-tunnel-routed-server"] = self.slots.buckets[0].server

        url = self._get_resource()
        resp = self.retry_handler.execute_with_retry_headers(
            lambda stamped: self._client.delete(url, params=params, headers=stamped),
            headers,
        )
        self.check_tunnel_response(resp)
        self._stop_keepalive()

    def open_upsert_stream(self, compress=False):
        """
        Open an upsert stream to insert or update data in records to the tunnel.

        :param bool compress: whether to compress data

        :return: an upsert stream
        :rtype: :class:`Upsert`
        """
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
            return self.retry_handler.execute_with_retry_headers(
                lambda stamped: self._client.put(
                    url, data=data, params=req_params, headers=stamped
                ),
                req_headers,
            )

        return Upsert(self.schema, upload_block, self, compress_option)

    def commit(self, async_=False):
        """
        Commit the current session. Can be called only once on a single session.
        """
        params = self.get_common_params(upsertid=self.id)
        headers = self.get_common_headers(content_length=0, tags=self._tags)
        headers["odps-tunnel-routed-server"] = self.slots.buckets[0].server

        url = self._get_resource()
        resp = self.retry_handler.execute_with_retry_headers(
            lambda stamped: self._client.post(url, params=params, headers=stamped),
            headers,
        )
        self.check_tunnel_response(resp)
        self.reload()

        if async_:
            return

        delay = 1
        start = time.monotonic()
        while self.status in (
            TableUpsertSession.Status.Committing,
            TableUpsertSession.Status.Normal,
        ):
            try:
                if time.monotonic() - start > self._commit_timeout:
                    raise TunnelError("Commit session timeout")
                time.sleep(delay)
                resp = self.retry_handler.execute_with_retry_headers(
                    lambda stamped: self._client.post(
                        url, params=params, headers=stamped
                    ),
                    headers,
                )
                self.check_tunnel_response(resp)
                self.reload()

                delay = min(8, delay * 2)
            except (errors.StreamSessionNotFound, errors.UpsertSessionNotFound):
                self.status = TableUpsertSession.Status.Committed
        if self.status != TableUpsertSession.Status.Committed:
            raise TunnelError("commit session failed, status: " + self.status.value)
        self._stop_keepalive()


class TableTunnel(BaseTunnel):
    """
    Table tunnel API Entry.

    :param odps: ODPS Entry object
    :param str project: project name
    :param str endpoint: tunnel endpoint
    :param str quota_name: name of tunnel quota
    """

    def _get_tunnel_table(self, table, schema=None):
        project_odps = None
        try:
            project_odps = self._project.odps
            if isinstance(table, str):
                table = project_odps.get_table(table, project=self._project.name)
        except Exception:
            pass

        project_name = self._project.name
        if not isinstance(table, str):
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
        """
        Create a download session for table.

        :param table: table object to read
        :type table: str | :class:`odps.models.Table`
        :param partition_spec: partition spec to read
        :type partition_spec: str | :class:`odps.types.PartitionSpec`
        :param str download_id: existing download id
        :param compress_option: compress option
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param str compress_algo: compress algorithm
        :param int compress_level: compress level
        :param str schema: name of schema of the table
        :param tags: tags of the upload session
        :type tags: str | list

        :return: :class:`TableDownloadSession`
        """
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo,
            level=compress_level,
            strategy=compress_strategy,
        )
        if "async_" in kw:
            async_mode = kw.pop("async_")
        if kw:
            raise TypeError(f"Cannot accept arguments {', '.join(kw.keys())}")
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
        create_partition=False,
        tags=None,
    ):
        """
        Create an upload session for table.

        :param table: table object to read
        :type table: str | :class:`odps.models.Table`
        :param partition_spec: partition spec
        :type partition_spec: str | :class:`odps.types.PartitionSpec`
        :param str upload_id: existing upload id
        :param compress_option: compress option
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param str compress_algo: compress algorithm
        :param int compress_level: compress level
        :param str schema: name of schema of the table
        :param bool overwrite: whether to overwrite the table
        :param bool create_partition: whether to create partition if not exist
        :param tags: tags of the upload session
        :type tags: str | list

        :return: :class:`TableUploadSession`
        """
        table = self._get_tunnel_table(table, schema)
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
            create_partition=create_partition,
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
        zorder_columns=None,
        upload_id=None,
        tags=None,
        allow_schema_mismatch=True,
        create_partition=False,
        slot_num=0,
        dynamic_partition=False,
    ):
        """
        Create a stream upload session for table.

        :param table: table object to read
        :type table: str | :class:`odps.models.Table`
        :param partition_spec: partition spec
        :type partition_spec: str | :class:`odps.types.PartitionSpec`
        :param str upload_id: existing upload id
        :param compress_option: compress option
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param str compress_algo: compress algorithm
        :param int compress_level: compress level
        :param str schema: name of schema of the table
        :param str schema_version: schema version of the upload
        :param zorder_columns: zorder columns for clustering
        :type zorder_columns: str | list
        :param tags: tags of the upload session
        :type tags: str | list
        :param bool allow_schema_mismatch: whether to allow table schema to be mismatched
        :param bool create_partition: whether to create partition if not exist
        :param int slot_num: number of slots for the session
        :param bool dynamic_partition: whether to enable dynamic partition

        :return: :class:`TableStreamUploadSession`
        """
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo,
            level=compress_level,
            strategy=compress_strategy,
        )
        version_need_reloaded = False

        def schema_version_reloader():
            nonlocal version_need_reloaded
            src_table = self._project.tables[table.name]
            if version_need_reloaded:
                src_table.reload_extend_info()
            version_need_reloaded = True
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
            create_partition=create_partition,
            zorder_columns=zorder_columns,
            slot_num=slot_num,
            dynamic_partition=dynamic_partition,
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
        lifecycle=None,
    ):
        """
        Create an upsert session for table.

        :param table: table object to read
        :type table: str | :class:`odps.models.Table`
        :param partition_spec: partition spec
        :type partition_spec: str | :class:`odps.types.PartitionSpec`
        :param str upsert_id: existing upsert id
        :param commit_timeout: timeout for commit
        :param compress_option: compress option
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param str compress_algo: compress algorithm
        :param int compress_level: compress level
        :param str schema: name of schema of the table
        :param tags: tags of the upload session
        :type tags: str | list
        :param int lifecycle: session lifecycle in hours, valid range 1-24

        :return: :class:`TableUpsertSession`
        """
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
            lifecycle=lifecycle,
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
        """
        Open a preview reader for table to read initial rows.

        :param table: table object to read
        :type table: str | :class:`odps.models.Table`
        :param partition_spec: partition spec to read
        :type partition_spec: str | :class:`odps.types.PartitionSpec`
        :param columns: columns to read
        :param int limit: number of rows to read, 10000 by default
        :param compress_option: compress option
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param str compress_algo: compress algorithm
        :param int compress_level: compress level
        :param str schema: name of schema of the table
        :param bool arrow: if True, return an Arrow reader, otherwise return a record reader
        :param tags: tags of the upload session
        :type tags: str | list
        """
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
        retry_handler = TunnelRetryHandler(default_retry_policy=OptionsRetryPolicy())
        resp = retry_handler.execute_with_retry_headers(
            lambda stamped: self.tunnel_rest.get(
                url, stream=True, params=params, headers=stamped, timeout=timeout
            ),
            headers,
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
            table.table_schema,
            stream_creator,
            columns=columns,
            use_ipc_stream=True,
            timestamp_as_struct=True,
        )
        if not arrow:
            reader = ArrowRecordReader(
                reader, make_compat=make_compat, read_all=read_all
            )
        return reader
