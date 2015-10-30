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


import json
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from odps.models import Schema
from odps.models import Record
from odps.tunnel.errors import TunnelError
from odps.tunnel.conf import CompressOption
from odps.tunnel.writer import ProtobufOutputWriter


class UploadSession(object):
    class Status(object):
        UNKNOWN, NORMAL, CLOSING, CLOSED, CANCELED, EXPIRED, CRITICAL = range(7)
        
    def __init__(self, conf, project_name, table_name, partition_spec,
                 upload_id=None, compress_option=None):
        self.conf = conf
        self.project_name = project_name
        self.table_name = table_name
        self.partition_spec = partition_spec
        if upload_id is None:
            self.init()
        else:
            self.id_ = upload_id
            self.reload()
        self.blocks = []
        self.compress_option = compress_option

    def init(self):
        params = {'uploads': 1}
        headers = {'Content-Length': 0}
        if self.partition_spec is not None and \
                len(self.partition_spec) > 0:
            params['partition'] = self.partition_spec

        client = self.get_resource()
        resp = client.post({}, params=params, headers=headers)
        if client.is_ok(resp):
            self._parse(resp)
        else:
            e = TunnelError.parse(resp)
            raise e
        
    def reload(self):
        params = {'uploadid': self.id_}
        headers = {'Content-Length': 0}
        if self.partition_spec is not None and \
                len(self.partition_spec) > 0:
            params['partition'] = self.partition_spec

        client = self.get_resource()
        resp = client.get(params=params, headers=headers)
        if client.is_ok(resp):
            self._parse(resp)
        else:
            e = TunnelError.parse(resp)
            raise e
            
    def _parse(self, xml):
        root = json.loads(xml.content)
        node = root.get('UploadID')
        if node is not None:
            self.id_ = node
        node = root.get('Status')
        if node is not None:
            self.status = getattr(UploadSession.Status, node.upper())
            
        node = root.get('UploadedBlockList')
        if node is not None:
            self.blocks = [nd['BlockID'] for nd in node]
        node = root.get('Schema')
        if node is not None:
            self.schema = Schema.parse(json.dumps(node))
            
    def get_resource(self):
        return self.conf.get_resource(self.project_name, self.table_name)
    
    def new_record(self):
        return Record(self.schema.columns)
            
    def open_record_writer(self, block_id, compress=False):
        """
        BlockId是由用户选取的0~19999之间的数值，标识本次上传数据块
        """
        compress_option = self.compress_option or self.conf.option

        params = {}
        headers = {'Transfer-Encoding': 'chunked',
                   'Content-Type': 'application/octet-stream',
                   'x-odps-tunnel-version': 4}
        if compress:
            if compress_option.algorithm == \
                    CompressOption.CompressionAlgorithm.ODPS_ZLIB:
                headers['Content-Encoding'] = 'deflate'
            elif compress_option.algorithm == \
                    CompressOption.CompressionAlgorithm.ODPS_SNAPPY:
                headers['Content-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != \
                    CompressOption.CompressionAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')
        params['uploadid'] = self.id_
        params['blockid'] = block_id
        if self.partition_spec is not None and len(self.partition_spec) > 0:
            params['partition'] = self.partition_spec
        
        fp = StringIO()
        client = self.get_resource()
        chunk_upload = lambda: client.chunk_upload(
            fp, params=params, headers=headers)
        option = compress_option if compress else None
        writer = ProtobufOutputWriter(self.schema, fp, chunk_upload,
                                      compress_opt=option)
        return writer
    
    def get_block_list(self):
        self.reload()
        return self.blocks
    
    def commit(self, blocks):
        if blocks is None:
            raise ValueError('Invalid parameter: blocks.')
        if isinstance(blocks, (int, long)):
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
            
        self.complete_upload()
    
    def complete_upload(self):
        params = {'uploadid': self.id_}
        if self.partition_spec is not None and len(self.partition_spec) > 0:
            params['partition'] = self.partition_spec
        client = self.get_resource()
        resp = client.post('', params=params)
        self._parse(resp)
