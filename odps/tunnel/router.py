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

from odps.tunnel.errors import TunnelError
from odps.compat import urlparse


class TunnelServerRouter(object):
    
    def __init__(self, client):
        self.client = client
        
    def get_tunnel_server(self, project, protocol):
        if protocol is None or protocol not in ('http', 'https'):
            raise TunnelError("Invalid protocol: "+protocol)

        url = '/'.join([project.resource().rstrip('/'), 'tunnel'])
        params = {'service': ''}
        resp = self.client.get(url, params=params)

        if self.client.is_ok(resp):
            addr = resp.text
            return urlparse('%s://%s' % (protocol, addr)).geturl()
        else:
            raise TunnelError("Can't get tunnel server address")
