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

import time

from ...compat import six
from ..core import Iterable
from .onlinemodel import OnlineModel
from ... import serializers, errors


class OnlineModels(Iterable):
    _root = 'Onlinemodels'

    marker = serializers.XMLNodeField
    online_models = serializers.XMLNodesReferencesField(OnlineModel, 'Onlinemodel')
    max_items = serializers.XMLNodeField(parse_callback=int)

    @property
    def project(self):
        return self.parent

    def _get(self, item):
        return OnlineModel(client=self._client, parent=self, name=item)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            online_model = self._get(item)
        elif isinstance(item, OnlineModel):
            online_model = item
        else:
            return False

        try:
            online_model.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self, name=None, owner=None):
        """

        :param name: the prefix of online model name
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

            v = OnlineModels.parse(self._client, resp, obj=self)
            params['marker'] = v.marker

            return v.online_models

        while True:
            volumes = _it()
            if volumes is None:
                break
            for volume in volumes:
                yield volume

    def create(self, obj=None, **kwargs):
        kwargs.setdefault('qos', 100)
        async = kwargs.pop('async', False)

        if callable(getattr(kwargs.get('predictor'), 'upload_files', None)):
            kwargs.get('predictor').upload_files(self.odps)

        online_model = obj or OnlineModel(parent=self, client=self._client, **kwargs)
        if online_model.parent is None:
            online_model._parent = self
        if online_model._client is None:
            online_model._client = self._client

        headers = {'Content-Type': 'application/xml'}
        data = online_model.serialize()

        self._client.post(self.resource(), data, headers=headers)

        online_model.reload()

        if async:
            return online_model
        online_model.wait_for_service()
        return online_model

    def delete(self, name, async=False):
        if not isinstance(name, OnlineModel):
            online_model = OnlineModel(name=name, client=self._client, parent=self)
        else:
            online_model = name
            name = name.name
        del self[name]  # release cache

        url = online_model.resource()

        self._client.delete(url)

        if async:
            return
        online_model.wait_for_deletion()
