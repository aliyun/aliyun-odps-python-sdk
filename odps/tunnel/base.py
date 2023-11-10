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

from .. import options
from ..compat import six, urlparse
from ..models import Projects
from ..rest import RestClient
from .errors import TunnelError

TUNNEL_VERSION = 5

_endpoint_cache = dict()


class BaseTunnel(object):
    def __init__(self, odps=None, client=None, project=None, endpoint=None, quota_name=None):
        self._client = odps.rest if odps is not None else client
        self._account = self._client.account
        if project is None and odps is None:
            raise AttributeError('%s requires project parameter.' % type(self).__name__)
        if isinstance(project, six.string_types):
            if odps is not None:
                self._project = odps.get_project(project or odps.project)
            else:
                self._project = Projects(client=self._client)[project]
        elif project is None:
            self._project = odps.get_project()
        else:
            self._project = project

        self._quota_name = quota_name or options.tunnel.quota_name
        if quota_name is not None:
            self._endpoint = endpoint
        else:
            self._endpoint = endpoint or self._project._tunnel_endpoint or options.tunnel.endpoint
        self._tunnel_rest = None

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def quota_name(self):
        return self._quota_name

    def _get_tunnel_server(self, project):
        protocol = urlparse(self._client.endpoint).scheme
        if protocol is None or protocol not in ('http', 'https'):
            raise TunnelError("Invalid protocol: %s" % protocol)

        ep_cache_key = (self._client.endpoint, project.name, self._quota_name)
        if ep_cache_key in _endpoint_cache:
            return _endpoint_cache[ep_cache_key]

        url = '/'.join([project.resource().rstrip('/'), 'tunnel'])
        params = {}
        if self._quota_name:
            params["quotaName"] = self._quota_name
        resp = self._client.get(url, action='service', params=params)

        if self._client.is_ok(resp):
            addr = resp.text
            server_ep = _endpoint_cache[ep_cache_key] = urlparse(
                '%s://%s' % (protocol, addr)
            ).geturl()
            return server_ep
        else:
            raise TunnelError("Can't get tunnel server address")

    @property
    def tunnel_rest(self):
        if self._tunnel_rest is not None:
            return self._tunnel_rest

        kw = dict(tag="TUNNEL")
        if options.data_proxy is not None:
            kw['proxy'] = options.data_proxy
        if getattr(self._client, 'app_account', None) is not None:
            kw['app_account'] = self._client.app_account

        endpoint = self._endpoint
        if endpoint is None:
            endpoint = self._get_tunnel_server(self._project)
        self._tunnel_rest = RestClient(self._account, endpoint, self._client.project, **kw)
        return self._tunnel_rest
