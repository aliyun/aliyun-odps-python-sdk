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

from .core import Iterable
from .resource import Resource, FileResource, TableResource
from .. import serializers, errors
from ..compat import six


class Resources(Iterable):

    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems')
    resources = serializers.XMLNodesReferencesField(Resource, 'Resource')

    def _get(self, name):
        splitter = '/resources/'
        if splitter in name:
            project_name, name = tuple(name.split(splitter, 1))

            from .projects import Projects
            return Projects(client=self._client)[project_name].resources[name]
        return Resource(client=self._client, parent=self, name=name)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            try:
                resource = self._get(item)
            except errors.NoSuchObject:
                return False
        elif isinstance(item, Resource):
            resource = item
        else:
            return False

        try:
            resource.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self):
        params = {'expectmarker': 'true'}

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, params=params)

            r = Resources.parse(self._client, resp, obj=self)
            params['marker'] = r.marker

            return r.resources

        while True:
            resources = _it()
            if resources is None:
                break
            for resource in resources:
                yield resource

    def create(self, obj=None, **kwargs):
        if obj is None and 'type' not in kwargs:
            raise ValueError('Unknown resource type to create.')

        if 'temp' in kwargs:
            kwargs['is_temp_resource'] = kwargs.pop('temp')

        ctor_kw = kwargs.copy()
        ctor_kw.pop('file_obj', None)
        obj = obj or Resource(parent=self, client=self._client, **ctor_kw)

        if obj.type == Resource.Type.UNKOWN:
            raise ValueError('Unknown resource type to create.')
        if obj.parent is None:
            obj._parent = self
        if obj._client is None:
            obj._client = self._client
        return obj.create(overwrite=False, **kwargs)

    def update(self, obj, **kwargs):
        return obj.create(overwrite=True, **kwargs)

    def delete(self, name):
        if not isinstance(name, Resource):
            resource = Resource(name=name, parent=self, type=Resource.Type.UNKOWN)
        else:
            resource = name
            name = name.name
        del self[name]  # release cache

        url = resource.resource()
        self._client.delete(url)

    def _request(self, name, stream=False):
        if isinstance(name, FileResource):
            res = name
        else:
            res = Resource(name, parent=self, client=self._client)
        url = res.resource()

        headers = {'Content-Type': 'application/octet-stream'}
        resp = self._client.get(url, headers=headers, stream=stream)

        return resp

    def iter_resource_content(self, name, text_mode=False):
        resp = self._request(name, stream=True)

        return resp.iter_content(decode_unicode=text_mode)

    def read_resource(self, name, encoding='utf-8', text_mode=False):
        resp = self._request(name)

        content = resp.content
        if not text_mode:
            if isinstance(content, six.text_type):
                content = content.encode(encoding)  # read as bytes
            return six.BytesIO(content)
        else:
            if isinstance(content, six.binary_type):
                content = content.decode(encoding)
            return six.StringIO(content)
