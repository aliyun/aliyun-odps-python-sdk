#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from .base import BaseTunnel
from .io.writer import RecordWriter, BufferredRecordWriter
from .io.reader import TunnelRecordReader
from .io.stream import CompressOption, SnappyRequestsInputStream, RequestsInputStream
from .errors import TunnelError
from .. import errors, serializers, types
from ..compat import Enum, six
from ..models import Projects, Record, Schema


class TableDownloadSession(serializers.JSONSerializableModel):
    __slots__ = '_client', '_table', '_partition_spec', '_compress_option'

    class Status(Enum):
        Unknown = 'UNKNOWN'
        Normal = 'NORMAL'
        Closes = 'CLOSES'
        Expired = 'EXPIRED'

    id = serializers.JSONNodeField('DownloadID')
    status = serializers.JSONNodeField(
        'Status', parse_callback=lambda s: TableDownloadSession.Status(s.upper()))
    count = serializers.JSONNodeField('RecordCount')
    schema = serializers.JSONNodeReferenceField(Schema, 'Schema')

    def __init__(self, client, table, partition_spec,
                 download_id=None, compress_option=None):
        super(TableDownloadSession, self).__init__()

        self._client = client
        self._table = table

        if isinstance(partition_spec, six.string_types):
            partition_spec = types.PartitionSpec(partition_spec)
        if isinstance(partition_spec, types.PartitionSpec):
            partition_spec = str(partition_spec).replace("'", '')
        self._partition_spec = partition_spec

        if download_id is None:
            self._init()
        else:
            self.id = download_id
            self.reload()
        self._compress_option = compress_option

    def _init(self):
        params = {'downloads': ''}
        headers = {'Content-Length': 0}
        if self._partition_spec is not None and \
                        len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

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

    def open_record_reader(self, start, count, compress=False, columns=None):
        compress_option = self._compress_option or CompressOption()

        params = {}
        headers = {'Content-Length': 0, 'x-odps-tunnel-version': 4}
        if compress:
            if compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Accept-Encoding'] = 'deflate'
            elif compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY:
                headers['Accept-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != \
                    CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')
        params['downloadid'] = self.id
        params['data'] = ''
        params['rowrange'] = '(%s,%s)' % (start, count)
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params['columns'] = ','.join(col_name(col) for col in columns)

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

        return TunnelRecordReader(self.schema, input_stream, columns=columns)


class TableUploadSession(serializers.JSONSerializableModel):
    __slots__ = '_client', '_table', '_partition_spec', '_compress_option'

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
                 upload_id=None, compress_option=None):
        super(TableUploadSession, self).__init__()

        self._client = client
        self._table = table

        if isinstance(partition_spec, six.string_types):
            partition_spec = types.PartitionSpec(partition_spec)
        if isinstance(partition_spec, types.PartitionSpec):
            partition_spec = str(partition_spec).replace("'", '')
        self._partition_spec = partition_spec

        if upload_id is None:
            self._init()
        else:
            self.id = upload_id
            self.reload()
        self._compress_option = compress_option

    def _init(self):
        params = {'uploads': 1}
        headers = {'Content-Length': 0}
        if self._partition_spec is not None and \
                        len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

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

    def open_record_writer(self, block_id=None, compress=False, buffer_size=None):
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
            def upload_block(blockid, data):
                params['blockid'] = blockid
                self._client.put(url, data=data, params=params, headers=headers)
            writer = BufferredRecordWriter(self.schema, upload_block, compress_option=option,
                                           buffer_size=buffer_size)
        else:
            params['blockid'] = block_id

            def upload(data):
                self._client.put(url, data=data, params=params, headers=headers)
            writer = RecordWriter(self.schema, upload, compress_option=option)
        return writer

    def get_block_list(self):
        self.reload()
        return self.blocks

    def commit(self, blocks):
        if blocks is None:
            raise ValueError('Invalid parameter: blocks.')
        if isinstance(blocks, six.integer_types):
            blocks = [blocks, ]

        server_block_map = dict([(int(block_id), True) for block_id \
                                 in self.get_block_list()])
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
        resp = self._client.post(url, '', params=params)
        self.parse(resp, obj=self)


class TableTunnel(BaseTunnel):
    def create_download_session(self, table, partition_spec=None,
                                download_id=None, compress_option=None,
                                compress_algo=None, compres_level=None, compress_strategy=None):
        if not isinstance(table, six.string_types):
            table = table.name
        table = Projects(client=self.tunnel_rest)[self._project.name].tables[table]
        compress_option = compress_option
        if compress_option is None and compress_algo is not None:
            compress_option = CompressOption(
                compress_algo=compress_algo, level=compres_level, strategy=compress_strategy)

        return TableDownloadSession(self.tunnel_rest, table, partition_spec,
                                    download_id=download_id,
                                    compress_option=compress_option)

    def create_upload_session(self, table, partition_spec=None,
                              upload_id=None, compress_option=None,
                              compress_algo=None, compres_level=None, compress_strategy=None):
        if not isinstance(table, six.string_types):
            table = table.name
        table = Projects(client=self.tunnel_rest)[self._project.name].tables[table]
        compress_option = compress_option
        if compress_option is None and compress_algo is not None:
            compress_option = CompressOption(
                compress_algo=compress_algo, level=compres_level, strategy=compress_strategy)

        return TableUploadSession(self.tunnel_rest, table, partition_spec,
                                  upload_id=upload_id,
                                  compress_option=compress_option)
