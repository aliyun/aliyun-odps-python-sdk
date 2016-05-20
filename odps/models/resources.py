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
            if not resources:
                break
            for resource in resources:
                yield resource

    def _create_file(self, file_obj, resource, overwrite):
        """
        :param file_obj: file-like object
        :param resource:
        :param overwrite: update if True else create
        :return:
        """
        if file_obj is None:
            raise ValueError('File content cannot be None.')
        if resource.name is None or len(resource.name.strip()) == 0:
            raise errors.ODPSError('File Resource Name should not empty.')

        method = self._client.post if not overwrite else self._client.put
        url = self.resource() if not overwrite else resource.resource()

        headers = {'Content-Type': 'application/octet-stream',
                   'Content-Disposition': 'attachment;filename=%s'%resource.name,
                   'x-odps-resource-type': resource.type.value.lower(),
                   'x-odps-resource-name': resource.name}
        if resource._getattr('comment') is not None:
            headers['x-odps-comment'] = resource.comment
        if resource._getattr('is_temp_resource'):
            headers['x-odps-resource-istemp'] = 'true' if resource.is_temp_resource else 'false'

        if not isinstance(file_obj, six.string_types):
            file_obj.seek(0)
            content = file_obj.read()
        else:
            content = file_obj
        method(url, content, headers=headers)

        if overwrite:
            resource.reload()
        return resource

    def _create_table(self, resource, overwrite):
        if not isinstance(resource, TableResource):
            raise ValueError('cannot create table resource when resource type is %s' % resource.type)
        if resource.name is None or len(resource.name.strip()) == 0:
            raise errors.ODPSError('Table Resource Name should not empty.')

        method = self._client.post if not overwrite else self._client.put
        url = self.resource() if not overwrite else resource.resource()

        headers = {'Content-Type': 'text/plain',
                   'x-odps-resource-type': resource.type.value.lower(),
                   'x-odps-resource-name': resource.name,
                   'x-odps-copy-table-source': resource.source_table_name}
        if resource._getattr('comment') is not None:
            headers['x-odps-comment'] = resource._getattr('comment')

        method(url, '', headers=headers)

        if overwrite:
            del self[resource.name]
            return self[resource.name]
        return resource

    def create(self, obj=None, **kwargs):
        if obj is None and 'type' not in kwargs:
            raise ValueError('Unknown resource type to create.')
        fp = kwargs.pop('file_obj', None)
        obj = obj or Resource(parent=self, client=self._client, **kwargs)
        if obj.type == Resource.Type.UNKOWN:
            raise ValueError('Unknown resource type to create.')
        if obj.parent is None:
            obj._parent = self
        if obj._client is None:
            obj._client = self._client

        if isinstance(obj, FileResource):
            if fp is None:
                raise ValueError('parameter `file_obj` cannot be None, either string or file-like object')
            if isinstance(fp, six.text_type):
                fp = fp.encode('utf-8')
            if isinstance(fp, six.binary_type):
                fp = six.BytesIO(fp)
            return self._create_file(fp, obj, False)
        elif isinstance(obj, TableResource):
            return self._create_table(obj, False)
        else:
            raise NotImplementedError

    def update(self, obj, **kwargs):
        fp = kwargs.get('file_obj')
        if isinstance(obj, FileResource):
            return self._create_file(fp, obj, True)
        elif isinstance(obj, TableResource):
            return self._create_table(obj, True)
        else:
            raise NotImplementedError

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
