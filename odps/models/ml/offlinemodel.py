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

from ..core import LazyLoad
from ... import serializers, utils
from ...compat import urlparse


class OfflineModelInfo(serializers.XMLSerializableModel):
    _root = 'Offlinemodel'

    name = serializers.XMLNodeField('Name', default='')
    model_path = serializers.XMLNodeField('ModelPath', default='')
    role_arn = serializers.XMLNodeField('Rolearn')
    type = serializers.XMLNodeField('Type')
    version = serializers.XMLNodeField('Version')
    processor = serializers.XMLNodeField('Processor')
    configuration = serializers.XMLNodeField('Configuration')
    src_project = serializers.XMLNodeField('SrcProject')
    src_model = serializers.XMLNodeField('SrcModel')
    dest_project = serializers.XMLNodeField('DestProject')
    dest_model = serializers.XMLNodeField('DestModel')


class OfflineModel(LazyLoad):
    """
    Representing an ODPS offline model.
    """

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
        """
        Get PMML text of the current model. Note that model file obtained
        via this method might be incomplete due to size limitations.
        """
        url = self.resource()
        params = {'data': ''}
        resp = self._client.get(url, params=params)

        return resp.text

    def copy(self, new_name, new_project=None, async_=False):
        """
        Copy current model into a new location.

        :param new_name: name of the new model
        :param new_project: new project name. if absent, original project name will be used
        :param async_: if True, return the copy instance. otherwise return the newly-copied model
        """
        url = self.parent.resource()
        new_project = new_project or self.project.name

        info = OfflineModelInfo(src_model=self.name, src_project=self.project.name,
                                dest_model=new_name, dest_project=new_project)
        headers = {'Content-Type': 'application/xml'}
        resp = self._client.post(url, info.serialize(), headers=headers)

        inst_url = resp.headers['Location'].rstrip('/')
        inst_id = urlparse(inst_url).path.rsplit('/', 1)[-1]
        inst = self.project.instances[inst_id]

        if not async_:
            inst.wait_for_success()
            return self.parent[new_name]
        else:
            return inst

    def drop(self):
        self.parent.delete(self)
