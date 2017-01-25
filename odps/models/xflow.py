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

from .core import LazyLoad
from .. import serializers, utils


class XFlow(LazyLoad):
    __slots__ = 'xml_source',

    name = serializers.XMLNodeField('Name')
    owner = serializers.XMLNodeField('Owner')
    creation_time = serializers.XMLNodeField('CreationTime',
                                             parse_callback=utils.parse_rfc822)
    last_modified_time = serializers.XMLNodeField('LastModifiedTime',
                                                  parse_callback=utils.parse_rfc822)

    def reload(self):
        url = self.resource()
        resp = self._client.get(url)

        self.xml_source = resp.content
        self.owner = resp.headers.get('x-odps-owner')
        self.creation_time = utils.parse_rfc822(resp.headers.get('x-odps-creation-time'))
        self.last_modified_time = utils.parse_rfc822(resp.headers.get('Last-Modified'))

    def update(self):
        return self._parent.update(self)

    def drop(self):
        return self._parent.delete(self)