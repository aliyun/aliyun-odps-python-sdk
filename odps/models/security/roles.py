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

from ..core import Iterable, LazyLoad
from ...compat import six
from ... import serializers, errors


class Role(LazyLoad):
    __slots__ = '_policy_cache',
    name = serializers.XMLNodeField('Name')
    comment = serializers.XMLNodeField('Comment')

    def __init__(self, **kw):
        super(Role, self).__init__(**kw)
        self._policy_cache = None

    @property
    def project(self):
        return self.parent.parent

    def reload(self):
        self._policy_cache = None
        resp = self._client.get(self.resource())
        self.parse(self._client, resp, obj=self)

    @property
    def users(self):
        from .users import Users
        params = dict(users='')
        resp = self._client.get(self.resource(), params=params)
        users = Users.parse(self._client, resp, parent=self.project)
        users._iter_local = True
        return users

    @property
    def policy(self):
        if self._policy_cache is None:
            params = dict(policy='')
            resp = self._client.get(self.resource(), params=params)
            self._policy_cache = resp.content.decode() if six.PY3 else resp.content
        return json.loads(self._policy_cache)

    @policy.setter
    def policy(self, value):
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        self._policy_cache = value
        params = dict(policy='')
        self._client.put(self.resource(), data=value, params=params)

    def grant_to(self, name):
        from .users import User
        if isinstance(name, User):
            name = name.display_name
        self.project.run_security_query('grant %s to %s' % (self.name, name))

    def revoke_from(self, name):
        from .users import User
        if isinstance(name, User):
            name = name.display_name
        self.project.run_security_query('revoke %s from %s' % (self.name, name))


class Roles(Iterable):
    __slots__ = '_iter_local',
    roles = serializers.XMLNodesReferencesField(Role, 'Role')

    def __init__(self, **kw):
        self._iter_local = False
        super(Roles, self).__init__(**kw)

    def _get(self, item):
        return Role(client=self._client, parent=self, name=item)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            role = self._get(item)
        elif isinstance(item, Role):
            role = item
        else:
            return False

        if not self._iter_local:
            try:
                role.reload()
                return True
            except errors.NoSuchObject:
                return False
        else:
            return any(r.name == role.name for r in self.roles)

    def __iter__(self):
        return self.iterate()

    @property
    def project(self):
        return self.parent

    def create(self, name):
        self.project.run_security_query('create role %s' % name)
        return Role(client=self._client, parent=self, name=name)

    def iterate(self, name=None):
        """
        :return:
        """
        if not self._iter_local:
            params = dict()
            if name is not None:
                params['name'] = name

            url = self.resource()
            resp = self._client.get(url, params=params)

            Roles.parse(self._client, resp, obj=self)

        for role in self.roles:
            yield role

    def delete(self, name):
        if isinstance(name, Role):
            name = name.name

        del self[name]  # delete from cache
        self.project.run_security_query('drop role %s' % name)
