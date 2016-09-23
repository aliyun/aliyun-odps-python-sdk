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
from ..base import BaseTunnel
from ..io import CompressOption
from ...compat import six
from ...models import Projects


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
