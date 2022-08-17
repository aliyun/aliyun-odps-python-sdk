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

import json

import requests

from ..compat import ElementTree as ET, TimeoutError
from ..errors import ODPSError


class TunnelError(ODPSError):

    @classmethod
    def parse(cls, resp):
        try:
            root = ET.fromstring(resp.content)
            code = root.find('./Code').text
            msg = root.find('./Message').text
            request_id = root.find('./RequestId')
            if request_id:
                request_id = request_id.text
            else:
                request_id = resp.headers.get('x-odps-request-id')

            error = TunnelError(msg, code=code, request_id=request_id)
        except:
            request_id = resp.headers['x-odps-request-id']
            obj = json.loads(resp.content)
            msg = obj['Message']
            code = obj['InvalidArgument']

            error = TunnelError(msg, code=code, request_id=request_id)

        return error


class TunnelReadTimeout(TunnelError, TimeoutError, requests.ReadTimeout):
    pass


class TunnelWriteTimeout(TunnelError, TimeoutError, requests.ConnectionError):
    pass
