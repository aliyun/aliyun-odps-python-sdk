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


import platform
from urlparse import urlparse

from odps import rest
from odps.tunnel.router import TunnelServerRouter


class CompressOption(object):
    class CompressionAlgorithm(object):
        ODPS_RAW, ODPS_ZLIB, ODPS_SNAPPY = range(3)
        
    def __init__(self, 
                 compress_algo=CompressionAlgorithm.ODPS_ZLIB, 
                 level=1, strategy=0):
        self.algorithm = compress_algo
        self.level = level
        self.strategy = strategy


class Conf(object):
    # 上传数据的默认块大小(单位字节)。默认值按照MTU－4设置，MTU值1500
    DEFAULT_CHUNK_SIZE = 1496

    # 底层网络连接的默认超时时间，180秒
    DEFAULT_SOCKET_CONNECT_TIMEOUT = 180

    # 底层网络默认超时时间，300秒
    DEFAULT_SOCKET_TIMEOUT = 300
   
    chunk_size = DEFAULT_CHUNK_SIZE
    socket_connect_timeout = DEFAULT_SOCKET_CONNECT_TIMEOUT
    socket_timeout = DEFAULT_SOCKET_TIMEOUT
    
    def __init__(self, odps):
        self.odps = odps
        self.client = odps.rest
        self.user_agent = 'TunnelSDK/0.12.0;%s' % \
            ' '.join((platform.system(), platform.release()))
        # self.client.set_user_agent(self.user_agent)
        self.router = TunnelServerRouter(self.client)
        self.option = CompressOption()
        # tunnel server endpoint
        self.endpoint = None

    def get_endpoint(self, project_name):
        if self.endpoint != None:
            return self.endpoint
        uri = urlparse(self.odps.endpoint)
        return self.router.get_tunnel_server(project_name, uri.scheme)
    
    def set_endpoint(self, endpoint):
        if endpoint != None:
            self.endpoint = endpoint

    def get_uri(self, project_name, table_name):
        endpoint = self.get_endpoint(project_name).geturl()
        endpoint = endpoint.rstrip('/')
        url = '%s/projects/%s/tables/%s' % (endpoint, project_name, table_name)
        return urlparse(url)
    
    def get_resource(self, project_name, table_name, shards=None):
        tunnel_endpoint = self.get_endpoint(project_name)
        tunnel_client = rest.RestClient(self.odps.account, tunnel_endpoint)
        res = tunnel_client.projects[project_name].tables[table_name]
        if shards is not None:
            res = res[shards]
        return res
