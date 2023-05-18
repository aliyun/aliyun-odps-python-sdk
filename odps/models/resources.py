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

from .core import Iterable
from .resource import Resource, FileResource
from .. import serializers, errors
from ..compat import six

_RESOURCE_SPLITTER = '/resources/'
_SCHEMA_SPLITTER = '/schemas/'

DEFAULT_RESOURCE_CHUNK_SIZE = 64 << 20


class Resources(Iterable):

    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems')
    resources = serializers.XMLNodesReferencesField(Resource, 'Resource')

    def get_typed(self, name, type):
        type_cls = Resource._get_cls(type)

        if _RESOURCE_SPLITTER in name:
            project_schema_name, name = name.split(_RESOURCE_SPLITTER, 1)

            if _SCHEMA_SPLITTER not in project_schema_name:
                project_name, schema_name = project_schema_name, None
            else:
                project_name, schema_name = project_schema_name.split(_SCHEMA_SPLITTER, 1)

            parent = self.parent.project.parent[project_name]
            if schema_name is not None:
                parent = parent.schemas[schema_name]
            return parent.resources[name]
        return type_cls(client=self._client, parent=self, name=name)

    def _get(self, name):
        return self.get_typed(name, None)

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

    def get(self, name, type=None):
        Resource._get_cls(type)

    def iterate(self, name=None, owner=None):
        params = {'expectmarker': 'true'}
        if name is not None:
            params['name'] = name
        if owner is not None:
            params['owner'] = owner
        schema_name = self._get_schema_name()
        if schema_name is not None:
            params['curr_schema'] = schema_name

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
        if 'part' in kwargs:
            kwargs['is_part_resource'] = kwargs.pop('part')

        ctor_kw = kwargs.copy()
        ctor_kw.pop('file_obj', None)
        ctor_kw.pop('fileobj', None)
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
        self._client.delete(url, curr_schema=self._get_schema_name())

    def _request(self, name, stream=False, offset=None, read_size=None):
        if isinstance(name, FileResource):
            res = name
        else:
            res = Resource(name, parent=self, client=self._client)
        url = res.resource()

        headers = {'Content-Type': 'application/octet-stream'}
        params = {}
        if offset is not None:
            params["rOffset"] = str(offset)
        if read_size is not None:
            params["rSize"] = str(read_size)

        resp = self._client.get(
            url, headers=headers, params=params, stream=stream, curr_schema=self._get_schema_name()
        )
        return resp

    def iter_resource_content(self, name, text_mode=False):
        resp = self._request(name, stream=True)

        return resp.iter_content(decode_unicode=text_mode)

    def read_resource(
        self, name, encoding='utf-8', text_mode=False, offset=None, read_size=None
    ):
        resp = self._request(name, offset=offset, read_size=read_size)

        content = resp.content
        if not text_mode:
            if isinstance(content, six.text_type):
                content = content.encode(encoding)  # read as bytes
            sio = six.BytesIO(content)
        else:
            if isinstance(content, six.binary_type):
                content = content.decode(encoding)
            sio = six.StringIO(content)

        has_remaining = resp.headers.get("x-odps-resource-has-remaining") or "false"
        sio.is_eof = has_remaining.lower() != "true"
        return sio

    def merge_part_files(self, resource, part_resources, md5_hex, overwrite=False):
        content = md5_hex + "|" + ",".join(res.name for res in part_resources)
        total_bytes = sum(res.size for res in part_resources)
        resource_args = resource.extract(

        )
        resource_args.update(
            {"parent": self, "client": self._client, "merge_total_bytes": total_bytes}
        )
        merge_res = Resource(**resource_args)
        merge_res.create(overwrite=overwrite, fileobj=content)
