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


from odps.tunnel.uploadsession import UploadSession
from odps.tunnel.downloadsession import DownloadSession
from odps.tunnel.conf import Conf

class TableTunnel(object):
    def __init__(self, odps, endpoint=None):
        self.conf = Conf(odps)
        if endpoint is not None and len(endpoint) != 0:
            self.conf.set_endpoint(endpoint)

    # Deprecate it ? self.endpoint is sufficient.(note: set the tunnel server endpoint)
    def set_endpoint(self, endpoint):
        self.conf.set_endpoint(endpoint)
        
    def create_download_session(self, table_name, project_name=None,
                                partition_spec=None, download_id=None,
                                compresss_option=None):
        project_name = project_name or self.conf.odps.project
        return DownloadSession(self.conf, project_name, table_name,
                               partition_spec, download_id=download_id,
                               compress_option=compresss_option)

    def create_upload_session(self, table_name, project_name=None, 
                              partition_spec=None, upload_id=None,
                              compress_option=None):
        project_name = project_name or self.conf.odps.project
        return UploadSession(self.conf, project_name, table_name,
                             partition_spec, upload_id=upload_id,
                             compress_option=compress_option)
