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

from .core import Container
from .project import Project
from .. import serializers, errors
from ..compat import six


class Projects(Container):
    __slots__ = "_odps_ref",

    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems')
    projects = serializers.XMLNodesReferencesField(Project, 'Project')

    def __init__(self, *args, **kwargs):
        self._odps_ref = None
        super(Projects, self).__init__(*args, **kwargs)

    def _get(self, item):
        return Project(client=self._client, _parent=self, name=item)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            project = self._get(item)
        elif isinstance(item, Project):
            project = item
        else:
            return False

        try:
            project.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(
        self, owner=None, user=None, group=None, max_items=None, name=None,
        region_id=None, tenant_id=None, quota_nick_name=None, quota_type=None,
        quota_name=None,
    ):
        params = {
            'expectmarker': 'true',
            'name': name,
            'owner': owner,
            'user': user,
            'group': group,
            'maxitems': max_items,
            'region': region_id,
            'tenant': tenant_id,
            'quotanickname': quota_nick_name,
            'quota_name': quota_name,
            'quota_type': quota_type,
        }
        params = dict(
            (k, v) for k, v in params.items() if v is not None
        )

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                    (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, params=params)

            t = Projects.parse(self._client, resp, obj=self)
            params['marker'] = t.marker

            return t.projects

        while True:
            projects = _it()
            if projects is None:
                break
            for project in projects:
                yield project
