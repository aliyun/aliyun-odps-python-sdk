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
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class TunnelError(RuntimeError):

    @classmethod
    def parse(cls, resp):
        try:
            root = ET.fromstring(resp.content)
            error = TunnelError()
            error.code = root.find('./Code').text
            error.msg = root.find('./Message').text
            
            request_id = root.find('./RequestId')
            if request_id:
                error.request_id = request_id.text
            else:
                error.request_id = resp.headers.get('x-odps-request-id')
        except:
            error.request_id = resp.headers['x-odps-request-id']
            obj = json.loads(resp.content)
            error.msg = obj['Message']
            error.code = obj['InvalidArgument']
            
        return error
