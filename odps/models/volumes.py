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
import warnings

from .. import serializers, errors, utils
from ..compat import six, Enum
from ..errors import InternalServerError
from .cache import cache
from .core import Iterable, LazyLoad


class Volume(LazyLoad):
    """
    Volume is the file-accessing object provided by ODPS.
    """

    EXTERNAL_VOLUME_LOCATION_KEY = "external.location"
    EXTERNAL_VOLUME_ROLEARN_KEY = "odps.properties.rolearn"

    class Type(Enum):
        NEW = 'NEW'
        OLD = 'OLD'
        EXTERNAL = 'EXTERNAL'
        UNKNOWN = 'UNKNOWN'

    _root = 'Meta'
    _type_indicator = 'type'

    name = serializers.XMLNodeField('Name')
    owner = serializers.XMLNodeField('Owner')
    comment = serializers.XMLNodeField('Comment')
    type = serializers.XMLNodeField(
        'Type', parse_callback=lambda t: Volume.Type(t.upper()), serialize_callback=lambda t: t.value
    )
    length = serializers.XMLNodeField('Length', parse_callback=int)
    file_number = serializers.XMLNodeField('FileNumber', parse_callback=int)
    creation_time = serializers.XMLNodeField('CreationTime', parse_callback=utils.parse_rfc822)
    last_modified_time = serializers.XMLNodeField('LastModifiedTime', parse_callback=utils.parse_rfc822)
    properties = serializers.XMLNodeField(
        'Properties', parse_callback=json.loads, serialize_callback=json.dumps
    )

    @classmethod
    def _get_cls(cls, typo):
        if typo is None:
            return cls
        if isinstance(typo, six.string_types):
            typo = Volume.Type(typo.upper())

        if typo == Volume.Type.OLD:
            from . import PartedVolume
            return PartedVolume
        elif typo == Volume.Type.NEW:
            from . import FSVolume
            return FSVolume
        elif typo == Volume.Type.EXTERNAL:
            from . import ExternalVolume
            return ExternalVolume
        elif typo == Volume.Type.UNKNOWN:
            return Volume

    @staticmethod
    def _filter_cache(_, **kwargs):
        return kwargs.get('type') is not None and kwargs['type'] != Volume.Type.UNKNOWN

    @cache
    def __new__(cls, *args, **kwargs):
        typo = kwargs.get('type')
        if typo is not None or (cls != Volume and issubclass(cls, Volume)):
            return object.__new__(cls._get_cls(typo))

        kwargs['type'] = Volume.Type.UNKNOWN
        obj = Volume(**kwargs)
        try:
            obj.reload()
            return Volume(**obj.extract())
        except InternalServerError as ex:
            warnings.warn(
                "Cannot reload volume %s due to error %s" % (obj.name, str(ex))
            )
            return obj

    def __init__(self, **kwargs):
        typo = kwargs.get('type')

        properties = kwargs.get("properties") or {}
        location = kwargs.pop("location", None)
        rolearn = kwargs.pop("rolearn", None)
        if location:
            properties[self.EXTERNAL_VOLUME_LOCATION_KEY] = location
        if rolearn:
            properties[self.EXTERNAL_VOLUME_ROLEARN_KEY] = rolearn
        if properties:
            kwargs["properties"] = properties

        if isinstance(typo, six.string_types):
            kwargs['type'] = Volume.Type(typo.upper())
        super(Volume, self).__init__(**kwargs)

    def reload(self):
        params = {'meta': ''}
        schema_name = self._get_schema_name()
        if schema_name is not None:
            params['curr_schema'] = schema_name
        resp = self._client.get(self.resource(), params=params)
        self.parse(self._client, resp, obj=self)

    def drop(self):
        return self.parent.delete(self)


class Volumes(Iterable):

    marker = serializers.XMLNodeField('Marker')
    volumes = serializers.XMLNodesReferencesField(Volume, 'Volume')

    def _get(self, item):
        return Volume(client=self._client, parent=self, name=item)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            try:
                # as reload() will be done in constructor of Volume, we return directly.
                return self._get(item)
            except errors.NoSuchObject:
                return False
        elif isinstance(item, Volume):
            volume = item
            try:
                volume.reload()
                return True
            except errors.NoSuchObject:
                return False
        else:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self, name=None, owner=None):
        """

        :param name: the prefix of volume name name
        :param owner:
        :return:
        """
        schema_name = self._get_schema_name()
        params = {'expectmarker': 'true'}
        if name is not None:
            params['name'] = name
        if owner is not None:
            params['owner'] = owner
        if schema_name is not None:
            params['curr_schema'] = schema_name

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                    (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, params=params)

            v = Volumes.parse(self._client, resp, obj=self)
            params['marker'] = v.marker

            return v.volumes

        while True:
            volumes = _it()
            if volumes is None:
                break
            for volume in volumes:
                yield volume

    def _create(self, obj=None, **kwargs):
        volume = obj or Volume(parent=self, client=self._client, **kwargs)
        if volume.parent is None:
            volume._parent = self
        if volume._client is None:
            volume._client = self._client

        headers = {'Content-Type': 'application/xml'}
        data = volume.serialize()

        self._client.post(
            self.resource(), data, headers=headers, curr_schema=self._get_schema_name()
        )
        return self[volume.name]

    def create_parted(self, obj=None, **kwargs):
        return self._create(obj=obj, type='old', **kwargs)

    def create_fs(self, obj=None, **kwargs):
        return self._create(obj=obj, type='new', **kwargs)

    def create_external(self, obj=None, **kwargs):
        return self._create(obj=obj, type='external', **kwargs)

    def delete(self, name):
        if not isinstance(name, Volume):
            volume = Volume(name=name, parent=self, client=self._client)
        else:
            volume = name
            name = name.name
        del self[name]  # release cache

        url = volume.resource()

        self._client.delete(url, curr_schema=self._get_schema_name())
