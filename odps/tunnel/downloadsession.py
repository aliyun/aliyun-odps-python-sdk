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
from odps.tunnel.errors import TunnelError
from odps.tunnel.conf import CompressOption
from odps.tunnel.reader import ProtobufInputReader


class DownloadSession(object):
    class Status(object):
        UNKNOWN, NORMAL, CLOSES, EXPIRED = range(4)
        
    def __init__(self, conf, project_name, table_name, 
                 partition_spec, download_id=None, compress_option=None):
        self.conf = conf
        self.project_name = project_name
        self.table_name = table_name
        self.partition_spec = partition_spec
        if download_id is None:
            self.init()
        else:
            self.id_ = download_id
            self.reload()
        self.compress_option = compress_option

    def init(self):
        params = {'downloads': ''}
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
        params = {'downloadid': self.id_}
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
        node = root.get('DownloadID')
        if node is not None:
            self.id_ = node
        node = root.get('Status')
        if node is not None:
            self.status = getattr(DownloadSession.Status, node.upper())
        node = root.get('RecordCount')
        if node is not None:
            self.count = int(node)
        node = root.get('Schema')
        if node is not None:
            self.schema = Schema.parse(json.dumps(node))
            
    def get_resource(self):
        return self.conf.get_resource(self.project_name, self.table_name)
            
    def open_record_reader(self, start, count, compress=False):
        compress_option = self.compress_option or self.conf.option

        params = {}
        headers = {'Content-Length': 0, 'x-odps-tunnel-version': 4}
        if compress:
            if compress_option.algorithm == \
                    CompressOption.CompressionAlgorithm.ODPS_ZLIB:
                headers['Accept-Encoding'] = 'deflate'
            elif compress_option.algorithm == \
                    CompressOption.CompressionAlgorithm.ODPS_SNAPPY:
                headers['Content-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != \
                    CompressOption.CompressionAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')
        params['downloadid'] = self.id_
        params['data'] = ''
        params['rowrange'] = '(%s,%s)' % (start, count)
        if self.partition_spec is not None and len(self.partition_spec) > 0:
            params['partition'] = self.partition_spec

        client = self.get_resource()
        resp = client.get(params=params, headers=headers)
        if not client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e
        
        content_encoding = resp.headers.get('Content-Encoding')
        if content_encoding is not None:
            if content_encoding == 'deflate':
                self.conf.option = CompressOption(
                    CompressOption.CompressionAlgorithm.ODPS_ZLIB, -1, 0)
            elif content_encoding == 'x-snappy-framed':
                self.conf.option = CompressOption(
                    CompressOption.CompressionAlgorithm.ODPS_SNAPPY, -1, 0)
            else:
                raise TunnelError('invalid content encoding')
            compress = True
        else:
            compress = False
        
        option = compress_option if compress else None
        return ProtobufInputReader(self.schema, StringIO(resp.content), option)
