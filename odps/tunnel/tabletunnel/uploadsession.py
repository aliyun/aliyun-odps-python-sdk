#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from ..errors import TunnelError
from ..io import CompressOption
from ... import types, serializers
from ...models import Schema, Record
from ...compat import Enum, six
from .writer import RecordWriter, BufferredRecordWriter


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
        else:
            e = TunnelError.parse(resp)
            raise e

    def new_record(self, values=None):
        return Record(self.schema.columns, values=values)

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
