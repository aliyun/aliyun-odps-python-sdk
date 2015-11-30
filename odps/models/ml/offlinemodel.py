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

from ..core import LazyLoad
from ... import serializers, utils


class OfflineModel(LazyLoad):

    name = serializers.XMLNodeField('Name')
    owner = serializers.XMLNodeField('Owner')
    creation_time = serializers.XMLNodeField(
        'CreationTime', parse_callback=utils.parse_rfc822)
    last_modified_time = serializers.XMLNodeField(
        'LastModifiedTime', parse_callback=utils.parse_rfc822)

    @property
    def project(self):
        return self.parent.parent

    def reload(self):
        resp = self._client.get(self.resource())
        self.parse(self._client, resp, obj=self)

    def get_model(self):
        url = self.resource()
        params = {'data': ''}
        resp = self._client.get(url, params=params)

        return resp.text

    def drop(self):
        self.parent.delete(self)
