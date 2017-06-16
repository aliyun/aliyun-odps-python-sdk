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

from .errors import TunnelError
from ..rest import RestClient
from ..models import Projects
from ..compat import urlparse, six
from .. import options


class BaseTunnel(object):
    def __init__(self, odps=None, client=None, project=None, endpoint=None):
        self._client = odps.rest if odps is not None else client
        self._account = self._client.account
        if project is None and odps is None:
            raise AttributeError('%s requires project parameter.' % type(self).__name__)
        if isinstance(project, six.string_types):
            self._project = Projects(client=self._client)[project or odps.project]
        elif project is None:
            self._project = odps.get_project()
        else:
            self._project = project

        self._endpoint = endpoint or self._project._tunnel_endpoint or options.tunnel.endpoint
        self._tunnel_rest = None

    @property
    def endpoint(self):
        return self._endpoint

    def _get_tunnel_server(self, project, protocol):
        if protocol is None or protocol not in ('http', 'https'):
            raise TunnelError("Invalid protocol: " + protocol)

        url = '/'.join([project.resource().rstrip('/'), 'tunnel'])
        params = {'service': ''}
        resp = self._client.get(url, params=params)

        if self._client.is_ok(resp):
            addr = resp.text
            return urlparse('%s://%s' % (protocol, addr)).geturl()
        else:
            raise TunnelError("Can't get tunnel server address")

    @property
    def tunnel_rest(self):
        if self._tunnel_rest is not None:
            return self._tunnel_rest

        kw = dict()
        if options.data_proxy is not None:
            kw['proxy'] = options.data_proxy

        endpoint = self._endpoint
        if endpoint is None:
            scheme = urlparse(self._client.endpoint).scheme
            endpoint = self._get_tunnel_server(self._project, scheme)
        self._tunnel_rest = RestClient(self._account, endpoint, self._client.project, **kw)
        return self._tunnel_rest
