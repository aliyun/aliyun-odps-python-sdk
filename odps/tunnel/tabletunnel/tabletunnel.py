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

from .uploadsession import TableUploadSession
from .downloadsession import TableDownloadSession
from ..io import CompressOption
from ..router import TunnelServerRouter
from ...rest import RestClient
from ... import options
from ...compat import urlparse, six
from ...models import Projects


class TableTunnel(object):
    def __init__(self, odps=None, client=None, project=None, endpoint=None):
        self._client = odps.rest if odps is not None else client
        self._account = self._client.account
        if project is None and odps is None:
            raise AttributeError('TableTunnel requires project parameter.')
        if isinstance(project, six.string_types):
            self._project = Projects(client=self._client)[project or odps.project]
        elif project is None:
            self._project = odps.get_project()
        else:
            self._project = project

        self._router = TunnelServerRouter(self._client)
        self._endpoint = endpoint or options.tunnel_endpoint

        self._tunnel_rest = None

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def tunnel_rest(self):
        if self._tunnel_rest is not None:
            return self._tunnel_rest

        endpoint = self._endpoint
        if endpoint is None:
            scheme = urlparse(self._client.endpoint).scheme
            endpoint = self._router.get_tunnel_server(self._project, scheme)
        self._tunnel_rest = RestClient(self._account, endpoint, self._client.project)
        return self._tunnel_rest

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
