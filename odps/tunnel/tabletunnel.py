#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import logging
import random
import sys
import time

from .. import errors, serializers, types, options
from ..compat import Enum, six
from ..lib import requests
from ..models import Projects, Record, TableSchema, Tenant
from ..types import Column
from .base import BaseTunnel, TUNNEL_VERSION
from .errors import (
    MetaTransactionFailed,
    TunnelError,
    TunnelWriteTimeout,
)
from .io.reader import TunnelRecordReader, TunnelArrowReader, ArrowRecordReader
from .io.stream import CompressOption, get_decompress_stream
from .io.writer import (
    RecordWriter,
    BufferedRecordWriter,
    StreamRecordWriter,
    ArrowWriter,
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
                if 'timed out' in ex_str:
                    raise TunnelWriteTimeout(ex_str, request_id=request_id)
                else:
                    raise

        return wrapped

    return wrapper


class BaseTableTunnelSession(serializers.JSONSerializableModel):
    @staticmethod
    def get_common_headers(content_length=None, chunked=False):
        header = {
            "odps-tunnel-date-transform": TUNNEL_DATA_TRANSFORM_VERSION,
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
        return header

    @staticmethod
    def normalize_partition_spec(partition_spec):
        if isinstance(partition_spec, six.string_types):
            partition_spec = types.PartitionSpec(partition_spec)
        if isinstance(partition_spec, types.PartitionSpec):
            partition_spec = str(partition_spec).replace("'", '')
        return partition_spec

    def get_common_params(self, **kwargs):
        params = {k: str(v) for k, v in kwargs.items()}
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        return params

    def check_tunnel_response(self, resp):
        if not self._client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

    def new_record(self, values=None):
        return Record(
            schema=self.schema, values=values,
            max_field_size=getattr(self, "max_field_size", None),
        )


class TableDownloadSession(BaseTableTunnelSession):
    __slots__ = (
        '_client', '_table', '_partition_spec', '_compress_option', '_quota_name'
    )

    class Status(Enum):
        Unknown = 'UNKNOWN'
        Normal = 'NORMAL'
        Closes = 'CLOSES'
        Expired = 'EXPIRED'
        Initiating = 'INITIATING'

    id = serializers.JSONNodeField('DownloadID')
    status = serializers.JSONNodeField(
        'Status', parse_callback=lambda s: TableDownloadSession.Status(s.upper())
    )
    count = serializers.JSONNodeField('RecordCount')
    schema = serializers.JSONNodeReferenceField(TableSchema, 'Schema')
    quota_name = serializers.JSONNodeField('QuotaName')

    def __init__(self, client, table, partition_spec, download_id=None,
                 compress_option=None, async_mode=True, timeout=None,
                 quota_name=None, **kw):
        super(TableDownloadSession, self).__init__()

        self._client = client
        self._table = table
        self._partition_spec = self.normalize_partition_spec(partition_spec)

        self._quota_name = quota_name

        if "async_" in kw:
            async_mode = kw.pop("async_")
        if kw:
            raise TypeError("Cannot accept arguments %s" % ", ".join(kw.keys()))
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
        params = self.get_common_params(downloads='')
        headers = self.get_common_headers(content_length=0)
        if self._quota_name:
            params['quotaName'] = self._quota_name
        if async_mode:
            params['asyncmode'] = 'true'

        url = self._table.table_resource()
        try:
            resp = self._client.post(url, {}, params=params, headers=headers, timeout=timeout)
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
        headers = self.get_common_headers(content_length=0)

        url = self._table.table_resource()
        resp = self._client.get(url, params=params, headers=headers)
        self.check_tunnel_response(resp)

        self.parse(resp, obj=self)
        if self.schema is not None:
            self.schema.build_snapshot()

    def _open_reader(
        self, start, count, compress=False, columns=None, arrow=False, reader_cls=None, **kw
    ):
        compress_option = self._compress_option or CompressOption()

        actions = ["data"]
        params = self.get_common_params(downloadid=self.id)
        headers = self.get_common_headers(content_length=0)
        if compress:
            encoding = compress_option.algorithm.get_encoding()
            if encoding:
                headers['Accept-Encoding'] = encoding

        params['rowrange'] = '(%s,%s)' % (start, count)
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params['columns'] = ','.join(col_name(col) for col in columns)

        if arrow:
            actions.append("arrow")

        url = self._table.table_resource()
        resp = self._client.get(url, stream=True, actions=actions, params=params, headers=headers)
        self.check_tunnel_response(resp)

        content_encoding = resp.headers.get('Content-Encoding')
        if content_encoding is not None:
            compress_algo = CompressOption.CompressAlgorithm.from_encoding(content_encoding)
            if compress_algo != compress_option.algorithm:
                compress_option = self._compress_option = CompressOption(compress_algo, -1, 0)
            compress = True
        else:
            compress = False

        option = compress_option if compress else None
        input_stream = get_decompress_stream(resp, option)
        return reader_cls(self.schema, input_stream, columns=columns, **kw)

    def open_record_reader(self, start, count, compress=False, columns=None):
        return self._open_reader(
            start, count, compress=compress, columns=columns,
            reader_cls=TunnelRecordReader, partition_spec=self._partition_spec,
        )

    if np is not None:
        def open_pandas_reader(self, start, count, compress=False, columns=None):
            from .pdio.pdreader_c import TunnelPandasReader
            return self._open_reader(start, count, compress=compress, columns=columns,
                                     reader_cls=TunnelPandasReader)

    def open_arrow_reader(self, start, count, compress=False, columns=None):
        return self._open_reader(
            start, count, compress=compress, columns=columns,
            arrow=True, reader_cls=TunnelArrowReader
        )


class TableUploadSession(BaseTableTunnelSession):
    __slots__ = (
        '_client', '_table', '_partition_spec', '_compress_option',
        '_overwrite', '_quota_name',
    )

    class Status(Enum):
        Unknown = 'UNKNOWN'
        Normal = 'NORMAL'
        Closing = 'CLOSING'
        Closed = 'CLOSED'
        Canceled = 'CANCELED'
        Expired = 'EXPIRED'
        Critical = 'CRITICAL'

    id = serializers.JSONNodeField('UploadID')
    status = serializers.JSONNodeField(
        'Status', parse_callback=lambda s: TableUploadSession.Status(s.upper())
    )
    blocks = serializers.JSONNodesField('UploadedBlockList', 'BlockID')
    schema = serializers.JSONNodeReferenceField(TableSchema, 'Schema')
    max_field_size = serializers.JSONNodeField('MaxFieldSize')
    quota_name = serializers.JSONNodeField('QuotaName')

    def __init__(self, client, table, partition_spec, upload_id=None,
                 compress_option=None, overwrite=False, quota_name=None):
        super(TableUploadSession, self).__init__()

        self._client = client
        self._table = table
        self._partition_spec = self.normalize_partition_spec(partition_spec)

        self._quota_name = quota_name
        self._overwrite = overwrite

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
        headers = self.get_common_headers(content_length=0)
        params = self.get_common_params(reload=reload)
        if not reload and self._overwrite:
            params["overwrite"] = "true"
        if not reload and self._quota_name:
            params['quotaName'] = self._quota_name

        if reload:
            params['uploadid'] = self.id
        else:
            params['uploads'] = 1

        retry_num = 0
        while True:
            try:
                url = self._table.table_resource()
                if reload:
                    resp = self._client.get(url, params=params, headers=headers)
                else:
                    resp = self._client.post(url, {}, params=params, headers=headers)
                self.check_tunnel_response(resp)

                self.parse(resp, obj=self)
                if self.schema is not None:
                    self.schema.build_snapshot()
                break
            except MetaTransactionFailed:
                time.sleep(0.1)
                retry_num += 1
                if retry_num > options.retry_times:
                    raise
                continue

    def _init(self):
        self._create_or_reload_session(reload=False)

    def reload(self):
        self._create_or_reload_session(reload=True)

    def _open_writer(self, block_id=None, compress=False, buffer_size=None, writer_cls=None):
        compress_option = self._compress_option or CompressOption()

        params = self.get_common_params(uploadid=self.id)
        headers = self.get_common_headers(chunked=True)
        if compress:
            # special: rewrite LZ4 to ARROW_LZ4 for arrow tunnels
            if (
                writer_cls is not None
                and issubclass(writer_cls, ArrowWriter)
                and compress_option.algorithm == CompressOption.CompressAlgorithm.ODPS_LZ4
            ):
                compress_option.algorithm = CompressOption.CompressAlgorithm.ODPS_ARROW_LZ4
            encoding = compress_option.algorithm.get_encoding()
            if encoding:
                headers['Content-Encoding'] = encoding

        url = self._table.table_resource()
        option = compress_option if compress else None

        if block_id is None:
            @_wrap_upload_call(self.id)
            def upload_block(blockid, data):
                params['blockid'] = blockid
                return self._client.put(url, data=data, params=params, headers=headers)

            writer = BufferedRecordWriter(self.schema, upload_block, compress_option=option,
                                          buffer_size=buffer_size)
        else:
            params['blockid'] = block_id

            @_wrap_upload_call(self.id)
            def upload(chunk_size):
                return self._client.put(
                    url, params=params, headers=headers, file_upload=True, chunk_size=chunk_size
                )

            if writer_cls is ArrowWriter:
                params['arrow'] = ''

            writer = writer_cls(self.schema, upload, compress_option=option)
        return writer

    def open_record_writer(self, block_id=None, compress=False, buffer_size=None):
        return self._open_writer(block_id=block_id, compress=compress, buffer_size=buffer_size,
                                 writer_cls=RecordWriter)

    def open_arrow_writer(self, block_id=None, compress=False):
        return self._open_writer(block_id=block_id, compress=compress, writer_cls=ArrowWriter)

    if np is not None:
        def open_pandas_writer(self, block_id=None, compress=False, buffer_size=None):
            from .pdio import TunnelPandasWriter
            return self._open_writer(block_id=block_id, compress=compress, buffer_size=buffer_size,
                                     writer_cls=TunnelPandasWriter)

    def get_block_list(self):
        self.reload()
        return self.blocks

    def commit(self, blocks):
        if blocks is None:
            raise ValueError('Invalid parameter: blocks.')
        if isinstance(blocks, six.integer_types):
            blocks = [blocks, ]

        server_block_map = dict(
            [
                (int(block_id), True) for block_id in self.get_block_list()
            ]
        )
        client_block_map = dict([(int(block_id), True) for block_id in blocks])

        if len(server_block_map) != len(client_block_map):
            raise TunnelError(
                'Blocks not match, server: %s, tunnelServerClient: %s. '
                'Make sure all block writers closed or with-blocks exited.' % (
                    len(server_block_map), len(client_block_map)
                )
            )

        for block_id in blocks:
            if block_id not in server_block_map:
                raise TunnelError(
                    'Block not exists on server, block id is %s' % (block_id,)
                )

        self._complete_upload()

    def _complete_upload(self):
        headers = self.get_common_headers()
        params = self.get_common_params(uploadid=self.id)
        url = self._table.table_resource()

        retries = options.retry_times
        while True:
            try:
                resp = self._client.post(url, '', params=params, headers=headers)
                break
            except (requests.Timeout, requests.ConnectionError, errors.InternalServerError):
                if retries > 0:
                    retries -= 1
                else:
                    raise
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
        return str(self._ip) + ':' + str(self._port)

    def set_server(self, server, check_empty=False):
        if len(server.split(':')) != 2:
            raise TunnelError('Invalid slot format: {}'.format(server))

        ip, port = server.split(':')

        if check_empty:
            if (not ip) or (not port):
                raise TunnelError('Empty server ip or port')
        if ip:
            self._ip = ip
        if port:
            self._port = int(port)


class TableStreamUploadSession(BaseTableTunnelSession):
    __slots__ = (
        '_client', '_table', '_partition_spec', '_compress_option', '_quota_name'
    )

    class Slots(object):
        def __init__(self, slot_elements):
            self._slots = []
            self._cur_index = -1
            for value in slot_elements:
                if len(value) != 2:
                    raise TunnelError('Invalid slot routes')
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

    schema = serializers.JSONNodeReferenceField(TableSchema, 'schema')
    id = serializers.JSONNodeField('session_name')
    status = serializers.JSONNodeField('status')
    slots = serializers.JSONNodeField(
        'slots', parse_callback=lambda val: TableStreamUploadSession.Slots(val))
    quota_name = serializers.JSONNodeField('QuotaName')

    def __init__(
        self, client, table, partition_spec, compress_option=None, quota_name=None
    ):
        super(TableStreamUploadSession, self).__init__()

        self._client = client
        self._table = table
        self._partition_spec = self.normalize_partition_spec(partition_spec)

        self._quota_name = quota_name

        self._init()
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
        headers = self.get_common_headers(content_length=0)

        if self._quota_name:
            params['quotaName'] = self._quota_name

        url = self._get_resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        self.check_tunnel_response(resp)

        self.parse(resp, obj=self)
        if self.schema is not None:
            self.schema.build_snapshot()

    def _get_resource(self):
        return self._table.table_resource() + '/streams'

    def reload(self):
        params = self.get_common_params(uploadid=self.id)
        headers = self.get_common_headers(content_length=0)

        url = self._get_resource()
        resp = self._client.get(url, params=params, headers=headers)
        self.check_tunnel_response(resp)

        self.parse(resp, obj=self)
        if self.schema is not None:
            self.schema.build_snapshot()

    def abort(self):
        params = self.get_common_params(uploadid=self.id)

        slot = next(iter(self.slots))
        headers = self.get_common_headers(content_length=0)
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

        headers = self.get_common_headers(chunked=True)
        headers.update({
            "odps-tunnel-slot-num": str(len(self.slots)),
            "odps-tunnel-routed-server": slot.server,
        })

        if compress:
            encoding = compress_option.algorithm.get_encoding()
            if encoding:
                headers['Content-Encoding'] = encoding

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
        '_client', '_table', '_partition_spec', '_compress_option',
        '_slot_num', '_commit_timeout', '_quota_name'
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
                slot = Slot(value['slot_id'], value['worker_addr'])
                self._slots.append(slot)
                self._buckets.update({idx: slot for idx in value['buckets']})

            for idx in self._buckets.keys():
                if idx > len(self._buckets):
                    raise TunnelError("Invalid bucket value: " + str(idx))

        @property
        def buckets(self):
            return self._buckets

        def __len__(self):
            return len(self._slots)

    schema = serializers.JSONNodeReferenceField(TableSchema, 'schema')
    id = serializers.JSONNodeField('id')
    status = serializers.JSONNodeField(
        'status', parse_callback=lambda s: TableUpsertSession.Status(s.upper())
    )
    slots = serializers.JSONNodeField(
        'slots', parse_callback=lambda val: TableUpsertSession.Slots(val))
    quota_name = serializers.JSONNodeField('quota_name')
    hash_keys = serializers.JSONNodeField('hash_key')
    hasher = serializers.JSONNodeField('hasher')

    def __init__(
        self, client, table, partition_spec, compress_option=None, slot_num=1,
        commit_timeout=DEFAULT_UPSERT_COMMIT_TIMEOUT, quota_name=None
    ):
        super(TableUpsertSession, self).__init__()

        self._client = client
        self._table = table
        self._partition_spec = self.normalize_partition_spec(partition_spec)
        self._quota_name = quota_name

        self._slot_num = slot_num
        self._commit_timeout = commit_timeout

        self._init()
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
        return self._table.table_resource() + '/upserts'

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
        headers = self.get_common_headers(content_length=0)

        if self._quota_name is not None:
            params['quotaName'] = self._quota_name

        if not reload:
            params['slotnum'] = str(self._slot_num)
        else:
            params['upsertid'] = self.id

        url = self._get_resource()
        if not reload:
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
        headers = self.get_common_headers(content_length=0)
        headers["odps-tunnel-routed-server"] = self.slots.buckets[0].server

        if self._quota_name is not None:
            params['quotaName'] = self._quota_name

        url = self._get_resource()
        resp = self._client.delete(url, params=params, headers=headers)
        self.check_tunnel_response(resp)

    def open_upsert_stream(self, compress=False):
        params = self.get_common_params(upsertid=self.id)
        headers = self.get_common_headers()

        compress_option = self._compress_option or CompressOption()
        if compress:
            encoding = compress_option.algorithm.get_encoding()
            if encoding:
                headers['Content-Encoding'] = encoding

        url = self._get_resource()

        @_wrap_upload_call(self.id)
        def upload_block(bucket, slot, record_count, data):
            req_params = params.copy()
            req_params.update(
                dict(bucketid=bucket, slotid=str(slot.slot), record_count=str(record_count))
            )
            req_headers = headers.copy()
            req_headers["odps-tunnel-routed-server"] = slot.server
            req_headers["Content-Length"] = len(data)
            return self._client.put(url, data=data, params=req_params, headers=req_headers)

        return Upsert(self.schema, upload_block, self, compress_option)

    def commit(self, async_=False):
        params = self.get_common_params(upsertid=self.id)
        headers = self.get_common_headers(content_length=0)
        headers["odps-tunnel-routed-server"] = self.slots.buckets[0].server

        url = self._get_resource()
        resp = self._client.post(url, params=params, headers=headers)
        self.check_tunnel_response(resp)
        self.reload()

        if async_:
            return

        delay = 1
        start = time.time()
        while self.status in (TableUpsertSession.Status.Committing, TableUpsertSession.Status.Normal):
            try:
                if time.time() - start > self._commit_timeout:
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
        if not isinstance(table, six.string_types):
            schema = schema or getattr(table.get_schema(), "name", None)
            table = table.name
        parent = Projects(client=self.tunnel_rest)[self._project.name]
        # tunnel rest does not have tenant options, thus creating a default one
        parent.odps._default_tenant = Tenant(parameters={})
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

    def create_download_session(self, table, async_mode=False, partition_spec=None,
                                download_id=None, compress_option=None,
                                compress_algo=None, compress_level=None,
                                compress_strategy=None, schema=None, timeout=None, **kw):
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo, level=compress_level, strategy=compress_strategy
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
    ):
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo, level=compress_level, strategy=compress_strategy
        )
        return TableUploadSession(
            self.tunnel_rest,
            table,
            partition_spec,
            upload_id=upload_id,
            compress_option=compress_option,
            overwrite=overwrite,
            quota_name=self._quota_name,
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
    ):
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo, level=compress_level, strategy=compress_strategy
        )
        return TableStreamUploadSession(
            self.tunnel_rest,
            table,
            partition_spec,
            compress_option=compress_option,
            quota_name=self._quota_name,
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
    ):
        table = self._get_tunnel_table(table, schema)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo, level=compress_level, strategy=compress_strategy
        )
        return TableUpsertSession(
            self.tunnel_rest,
            table,
            partition_spec,
            slot_num=slot_num,
            commit_timeout=commit_timeout,
            compress_option=compress_option,
            quota_name=self._quota_name,
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
    ):
        if pa is None:
            raise ImportError("Need pyarrow to run open_preview_reader.")

        tunnel_table = self._get_tunnel_table(table)
        compress_option = compress_option or self._build_compress_option(
            compress_algo=compress_algo, level=compress_level, strategy=compress_strategy
        )

        params = {"limit": str(limit) if limit else "-1"}
        partition_spec = BaseTableTunnelSession.normalize_partition_spec(partition_spec)
        if partition_spec is not None and len(partition_spec) > 0:
            params['partition'] = partition_spec

        headers = BaseTableTunnelSession.get_common_headers(content_length=0)
        if compress_option:
            encoding = compress_option.algorithm.get_encoding(legacy=False)
            if encoding:
                headers['Accept-Encoding'] = encoding

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

        reader = TunnelArrowReader(
            table.table_schema, input_stream, columns=columns, use_ipc_stream=True
        )
        if not arrow:
            reader = ArrowRecordReader(
                reader, make_compat=make_compat, read_all=read_all
            )
        return reader
