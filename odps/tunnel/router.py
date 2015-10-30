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


from urlparse import urlparse

from odps.tunnel.errors import TunnelError

class TunnelServerRouter(object):
    
    def __init__(self, client):
        self.client = client
        
    def get_tunnel_server(self, project_name, protocal):
        if protocal is None or protocal not in ('http', 'https'):
            raise TunnelError("Invalid protocal: "+protocal)
        
        params = {'service': None}
        resp = self.client.projects[project_name].tunnel.get(params=params)
        if self.client.is_ok(resp):
            addr = resp.content
            return urlparse('%s://%s' % (protocal, addr)).geturl()
        else:
            raise TunnelError("Can't get tunnel server address")
