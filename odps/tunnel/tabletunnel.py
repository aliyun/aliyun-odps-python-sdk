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

import time
import random

import requests

from .. import errors, serializers, types, options
from ..compat import Enum, six
from ..models import Projects, Record, Schema
from .base import BaseTunnel, TUNNEL_VERSION
from .io.writer import RecordWriter, BufferredRecordWriter, StreamRecordWriter, ArrowWriter
from .io.reader import TunnelRecordReader, TunnelArrowReader
from .io.stream import CompressOption, SnappyRequestsInputStream, RequestsInputStream
from .errors import TunnelError, TunnelWriteTimeout

try:
    import numpy as np
except ImportError:
    np = None

TUNNEL_DATA_TRANSFORM_VERSION = "v1"


def _wrap_upload_call(request_id):
    def wrapper(func):
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except requests.ConnectionError as ex:
                ex_str = str(ex)
                if 'timed out' in ex_str:
                    raise TunnelWriteTimeout(ex_str, request_id=request_id)
                else:
                    raise

        wrapped.__name__ = func.__name__
        return wrapped

    return wrapper


class TableDownloadSession(serializers.JSONSerializableModel):
    __slots__ = '_client', '_table', '_partition_spec', '_compress_option'

    class Status(Enum):
        Unknown = 'UNKNOWN'
        Normal = 'NORMAL'
        Closes = 'CLOSES'
        Expired = 'EXPIRED'
        Initiating = 'INITIATING'

    id = serializers.JSONNodeField('DownloadID')
    status = serializers.JSONNodeField(
        'Status', parse_callback=lambda s: TableDownloadSession.Status(s.upper()))
    count = serializers.JSONNodeField('RecordCount')
    schema = serializers.JSONNodeReferenceField(Schema, 'Schema')

    def __init__(self, client, table, partition_spec, download_id=None,
                 compress_option=None, async_=False, timeout=None):
        super(TableDownloadSession, self).__init__()

        self._client = client
        self._table = table

        if isinstance(partition_spec, six.string_types):
            partition_spec = types.PartitionSpec(partition_spec)
        if isinstance(partition_spec, types.PartitionSpec):
            partition_spec = str(partition_spec).replace("'", '')
        self._partition_spec = partition_spec

        if download_id is None:
            self._init(async_=async_, timeout=timeout)
        else:
            self.id = download_id
            self.reload()
        self._compress_option = compress_option
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return "<TableDownloadSession id=%s project=%s table=%s partition_spec=%s>" % (
            self.id,
            self._table.project.name,
            self._table.name,
            self._partition_spec,
        )

    def _init(self, async_, timeout):
        params = {'downloads': ''}
        headers = {'Content-Length': 0}
        if self._partition_spec is not None and \
                len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        if async_:
            params['asyncmode'] = 'true'

        url = self._table.resource()
        resp = self._client.post(url, {}, params=params, headers=headers, timeout=timeout)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
            while self.status == self.Status.Initiating:
                time.sleep(random.randint(0, 30) + 5)
                self.reload()
            if self.schema is not None:
                self.schema.build_snapshot()
        else:
            e = TunnelError.parse(resp)
            raise e

    def reload(self):
        params = {'downloadid': self.id}
        headers = {'Content-Length': 0}
        if self._partition_spec is not None and \
                        len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        url = self._table.resource()
        resp = self._client.get(url, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
            if self.schema is not None:
                self.schema.build_snapshot()
        else:
            e = TunnelError.parse(resp)
            raise e

    def _open_reader(
        self, start, count, compress=False, columns=None, arrow=False, reader_cls=None
    ):
        compress_option = self._compress_option or CompressOption()

        params = {}
        headers = {'Content-Length': 0, 'x-odps-tunnel-version': 4}
        if compress:
            if compress_option.algorithm == CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Accept-Encoding'] = 'deflate'
            elif compress_option.algorithm == CompressOption.CompressAlgorithm.ODPS_SNAPPY:
                headers['Accept-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')
        params['downloadid'] = self.id
        params['data'] = ''
        params['rowrange'] = '(%s,%s)' % (start, count)
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params['columns'] = ','.join(col_name(col) for col in columns)

        if arrow:
            params['arrow'] = ''

        url = self._table.resource()
        resp = self._client.get(url, stream=True, params=params, headers=headers)
        if not self._client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

        content_encoding = resp.headers.get('Content-Encoding')
        if content_encoding is not None:
            if content_encoding == 'deflate':
                self._compress_option = CompressOption(
                    CompressOption.CompressAlgorithm.ODPS_ZLIB, -1, 0)
            elif content_encoding == 'x-snappy-framed':
                self._compress_option = CompressOption(
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY, -1, 0)
            else:
                raise TunnelError('invalid content encoding')
            compress = True
        else:
            compress = False

        option = compress_option if compress else None
        if option is None:
            input_stream = RequestsInputStream(resp)  # create a file-like object from body
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_RAW:
            input_stream = RequestsInputStream(resp)  # create a file-like object from body
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_ZLIB:
            input_stream = RequestsInputStream(resp)  # Requests automatically decompress gzip data!
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_SNAPPY:
            input_stream = SnappyRequestsInputStream(resp)
        else:
            raise errors.InvalidArgument('Invalid compression algorithm.')

        return reader_cls(self.schema, input_stream, columns=columns)

    def open_record_reader(self, start, count, compress=False, columns=None):
        return self._open_reader(start, count, compress=compress, columns=columns,
                                 reader_cls=TunnelRecordReader)

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


class TableUploadSession(serializers.JSONSerializableModel):
    __slots__ = '_client', '_table', '_partition_spec', '_compress_option', '_overwrite'

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
        'Status', parse_callback=lambda s: TableUploadSession.Status(s.upper()))
    blocks = serializers.JSONNodesField('UploadedBlockList', 'BlockID')
    schema = serializers.JSONNodeReferenceField(Schema, 'Schema')

    def __init__(self, client, table, partition_spec,
                 upload_id=None, compress_option=None, overwrite=False):
        super(TableUploadSession, self).__init__()

        self._client = client
        self._table = table

        if isinstance(partition_spec, six.string_types):
            partition_spec = types.PartitionSpec(partition_spec)
        if isinstance(partition_spec, types.PartitionSpec):
            partition_spec = str(partition_spec).replace("'", '')
        self._partition_spec = partition_spec

        self._overwrite = overwrite

        if upload_id is None:
            self._init()
        else:
            self.id = upload_id
            self.reload()
        self._compress_option = compress_option
        if options.tunnel_session_create_callback:
            options.tunnel_session_create_callback(self)

    def __repr__(self):
        return "<TableUploadSession id=%s project=%s table=%s partition_spec=%s>" % (
            self.id,
            self._table.project.name,
            self._table.name,
            self._partition_spec,
        )

    def _init(self):
        params = {'uploads': 1}
        headers = {'Content-Length': 0}
        if self._partition_spec is not None and \
                len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        if self._overwrite:
            params["overwrite"] = "true"

        url = self._table.resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
            if self.schema is not None:
                self.schema.build_snapshot()
        else:
            e = TunnelError.parse(resp)
            raise e

    def reload(self):
        params = {'uploadid': self.id}
        headers = {'Content-Length': 0}
        if self._partition_spec is not None and \
                        len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        url = self._table.resource()
        resp = self._client.get(url, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
            if self.schema is not None:
                self.schema.build_snapshot()
        else:
            e = TunnelError.parse(resp)
            raise e

    def new_record(self, values=None):
        return Record(schema=self.schema, values=values)

    def _open_writer(self, block_id=None, compress=False, buffer_size=None, writer_cls=None):
        compress_option = self._compress_option or CompressOption()

        params = {}
        headers = {'Transfer-Encoding': 'chunked',
                   'Content-Type': 'application/octet-stream',
                   'x-odps-tunnel-version': 4}
        if compress:
            if compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Content-Encoding'] = 'deflate'
            elif compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY:
                headers['Content-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != \
                    CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')
        params['uploadid'] = self.id
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        url = self._table.resource()
        option = compress_option if compress else None

        if block_id is None:
            @_wrap_upload_call(self.id)
            def upload_block(blockid, data):
                params['blockid'] = blockid
                self._client.put(url, data=data, params=params, headers=headers)

            writer = BufferredRecordWriter(self.schema, upload_block, compress_option=option,
                                           buffer_size=buffer_size)
        else:
            params['blockid'] = block_id

            @_wrap_upload_call(self.id)
            def upload(data):
                self._client.put(url, data=data, params=params, headers=headers)

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
            raise TunnelError('Blocks not match, server: '+str(len(server_block_map))+
                              ', tunnelServerClient: '+str(len(client_block_map)))

        for block_id in blocks:
            if block_id not in server_block_map:
                raise TunnelError('Block not exists on server, block id is'+block_id)

        self._complete_upload()

    def _complete_upload(self):
        params = {'uploadid': self.id}
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        url = self._table.resource()

        retries = options.retry_times
        while True:
            try:
                resp = self._client.post(url, '', params=params)
                break
            except (requests.Timeout, requests.ConnectionError, errors.InternalServerError):
                if retries > 0:
                    retries -= 1
                else:
                    raise
        self.parse(resp, obj=self)


class TableStreamUploadSession(serializers.JSONSerializableModel):
    __slots__ = '_client', '_table', '_partition_spec', '_compress_option',

    schema = serializers.JSONNodeReferenceField(Schema, 'schema')
    id = serializers.JSONNodeField('session_name')
    status = serializers.JSONNodeField('status')
    slots = serializers.JSONNodeField(
        'slots', parse_callback=lambda val: TableStreamUploadSession.Slots(val))

    class Slots(object):
        def __init__(self, slot_elements):
            self._slots = []
            self._cur_index = -1
            for value in slot_elements:
                if len(value) != 2:
                    raise TunnelError('Invalid slot routes')
                self._slots.append(TableStreamUploadSession.Slot(value[0], value[1]))

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

    def __init__(self, client, table, partition_spec, compress_option=None):
        super(TableStreamUploadSession, self).__init__()

        self._client = client
        self._table = table

        if isinstance(partition_spec, six.string_types):
            partition_spec = types.PartitionSpec(partition_spec)
        if isinstance(partition_spec, types.PartitionSpec):
            partition_spec = str(partition_spec).replace("'", '')
        self._partition_spec = partition_spec

        self._init()
        self._compress_option = compress_option
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
        params = dict()
        headers = {
            "Content-Length": 0,
            "odps-tunnel-date-transform": TUNNEL_DATA_TRANSFORM_VERSION,
            "x-odps-tunnel-version": str(TUNNEL_VERSION),
        }
        if self._partition_spec is not None and \
                len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        url = self._get_resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
            if self.schema is not None:
                self.schema.build_snapshot()
        else:
            e = TunnelError.parse(resp)
            raise e

    def _get_resource(self):
        return self._table.resource() + '/' + 'streams'

    def reload(self):
        params = {'uploadid': self.id}
        if self._partition_spec is not None and \
                len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        headers = {
            "Content-Length": 0,
            "odps-tunnel-date-transform": TUNNEL_DATA_TRANSFORM_VERSION,
            "x-odps-tunnel-version": str(TUNNEL_VERSION),
        }

        url = self._get_resource()
        resp = self._client.get(url, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
            if self.schema is not None:
                self.schema.build_snapshot()
        else:
            e = TunnelError.parse(resp)
            raise e

    def abort(self):
        params = {'uploadid': self.id}
        if self._partition_spec is not None and \
                len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        slot = next(iter(self.slots))
        headers = {
            "Content-Length": 0,
            "odps-tunnel-date-transform": TUNNEL_DATA_TRANSFORM_VERSION,
            "x-odps-tunnel-version": str(TUNNEL_VERSION),
            "odps-tunnel-routed-server": slot.server,
        }

        url = self._get_resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        if not self._client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

    def reload_slots(self, slot, server, slot_num):
        if len(self.slots) != slot_num:
            self.reload()
        else:
            slot.set_server(server)

    def new_record(self, values=None):
        return Record(schema=self.schema, values=values)

    def _open_writer(self, compress=False):
        compress_option = self._compress_option or CompressOption()

        slot = next(iter(self.slots))

        headers = {
            'Transfer-Encoding': 'chunked',
            "Content-Type": "application/octet-stream",
            "x-odps-tunnel-version": str(TUNNEL_VERSION),
            "odps-tunnel-slot-num": str(len(self.slots)),
            "odps-tunnel-routed-server": slot.server,
        }

        if compress:
            if compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Content-Encoding'] = 'deflate'
            elif compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY:
                headers['Content-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != \
                    CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')

        params = dict(uploadid=self.id, slotid=slot.slot)
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        url = self._get_resource()
        option = compress_option if compress else None

        def upload(data):
            return self._client.put(url, data=data, params=params, headers=headers)

        writer = StreamRecordWriter(
            self.schema, upload, session=self, slot=slot, compress_option=option
        )

        return writer

    def open_record_writer(self, compress=False):
        return self._open_writer(compress=compress)


class TableTunnel(BaseTunnel):
    def create_download_session(self, table, async_=False, partition_spec=None,
                                download_id=None, compress_option=None,
                                compress_algo=None, compress_level=None,
                                compress_strategy=None, timeout=None):
        if not isinstance(table, six.string_types):
            table = table.name
        table = Projects(client=self.tunnel_rest)[self._project.name].tables[table]
        compress_option = compress_option
        if compress_option is None and compress_algo is not None:
            compress_option = CompressOption(
                compress_algo=compress_algo, level=compress_level, strategy=compress_strategy
            )

        return TableDownloadSession(
            self.tunnel_rest,
            table,
            partition_spec,
            download_id=download_id,
            compress_option=compress_option,
            async_=async_,
            timeout=timeout,
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
        overwrite=False,
    ):
        if not isinstance(table, six.string_types):
            table = table.name
        table = Projects(client=self.tunnel_rest)[self._project.name].tables[table]
        compress_option = compress_option
        if compress_option is None and compress_algo is not None:
            compress_option = CompressOption(
                compress_algo=compress_algo, level=compress_level, strategy=compress_strategy)

        return TableUploadSession(
            self.tunnel_rest,
            table,
            partition_spec,
            upload_id=upload_id,
            compress_option=compress_option,
            overwrite=overwrite,
        )

    def create_stream_upload_session(self, table, partition_spec=None, compress_option=None,
                                     compress_algo=None, compress_level=None, compress_strategy=None):
        if not isinstance(table, six.string_types):
            table = table.name
        table = Projects(client=self.tunnel_rest)[self._project.name].tables[table]
        compress_option = compress_option
        if compress_option is None and compress_algo is not None:
            compress_option = CompressOption(
                compress_algo=compress_algo, level=compress_level, strategy=compress_strategy)

        return TableStreamUploadSession(
            self.tunnel_rest, table, partition_spec, compress_option=compress_option
        )
