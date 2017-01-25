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

from ..core import Iterable
from ... import serializers, errors
from ...compat import six
from .offlinemodel import OfflineModel


class OfflineModels(Iterable):

    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems', parse_callback=int)
    offline_models = serializers.XMLNodesReferencesField(OfflineModel, 'OfflineModel')

    def _get(self, item):
        return OfflineModel(client=self._client, parent=self, name=item)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            offline_model = self._get(item)
        elif isinstance(item, OfflineModel):
            offline_model = item
        else:
            return False

        try:
            offline_model.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self, name=None, owner=None):
        """

        :param name: the prefix of offline model name
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

            t = OfflineModels.parse(self._client, resp, obj=self)
            params['marker'] = t.marker

            return t.offline_models

        while True:
            offline_models = _it()
            if offline_models is None:
                break
            for offline_model in offline_models:
                yield offline_model

    def delete(self, name):
        if not isinstance(name, OfflineModel):
            offline_model = OfflineModel(name=name, parent=self)
        else:
            offline_model = name
            name = name.name

        del self[name]  # delete from cache

        url = offline_model.resource()

        self._client.delete(url)