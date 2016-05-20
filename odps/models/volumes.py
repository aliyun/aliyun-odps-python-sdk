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
from .volume import Volume
from .. import serializers, errors
from ..compat import six


class Volumes(Iterable):

    marker = serializers.XMLNodeField('Marker')
    volumes = serializers.XMLNodesReferencesField(Volume, 'Volume')

    def _get(self, item):
        return Volume(client=self._client, parent=self, name=item)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            volume = self._get(item)
        elif isinstance(item, Volume):
            volume = item
        else:
            return False

        try:
            volume.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self, name=None, owner=None):
        """

        :param name: the prefix of volume name name
        :param owner:
        :return:
        """

        params = {'expectmarker': 'true'}
        if name is not None:
            params['name'] = name
        if owner is not None:
            params['owner'] = owner

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
            if not volumes:
                break
            for volume in volumes:
                yield volume

    def create(self, obj=None, **kwargs):
        volume = obj or Volume(parent=self, client=self._client, **kwargs)
        if volume.parent is None:
            volume._parent = self
        if volume._client is None:
            volume._client = self._client

        headers = {'Content-Type': 'application/xml'}
        data = volume.serialize()

        self._client.post(self.resource(), data, headers=headers)

        volume.reload()
        return volume

    def delete(self, name):
        if not isinstance(name, Volume):
            volume = Volume(name=name, parent=self)
        else:
            volume = name
            name = name.name
        del self[name]  # release cache

        url = volume.resource()

        self._client.delete(url)
