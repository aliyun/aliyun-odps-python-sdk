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

from ..core import Iterable, LazyLoad
from ...compat import six
from ... import serializers, errors


class User(LazyLoad):
    id = serializers.XMLNodeField('ID')
    display_name = serializers.XMLNodeField('DisplayName')
    comment = serializers.XMLNodeField('Comment')

    def _name(self):
        return self.id

    @property
    def project(self):
        return self.parent.parent

    def reload(self):
        if self._getattr('id') is None:
            resp = self._client.get(self.parent.resource() + '/' + self._encode(self.display_name),
                                    params=dict(type='displayname'))
            self.parse(self._client, resp, obj=self)
        else:
            resp = self._client.get(self.resource())
            self.parse(self._client, resp, obj=self)

    @property
    def roles(self):
        from .roles import Roles
        params = dict(roles='', type='displayname')
        resp = self._client.get(self.resource(), params=params)
        roles = Roles.parse(self._client, resp, parent=self.project)
        roles._iter_local = True
        return roles

    def grant_role(self, name):
        from .roles import Role
        if isinstance(name, Role):
            name = name.name
        self.project.run_security_query('grant %s to %s' % (name, self.display_name))

    def revoke_role(self, name):
        from .roles import Role
        if isinstance(name, Role):
            name = name.name
        self.project.run_security_query('revoke %s from %s' % (name, self.display_name))


class Users(Iterable):
    __slots__ = '_iter_local',
    users = serializers.XMLNodesReferencesField(User, 'User')

    def __init__(self, **kw):
        self._iter_local = False
        super(Users, self).__init__(**kw)

    def _get(self, item):
        return User(client=self._client, parent=self, display_name=item)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            user = self._get(item)
        elif isinstance(item, User):
            user = item
        else:
            return False

        if not self._iter_local:
            try:
                user.reload()
                return True
            except errors.NoSuchObject:
                return False
        else:
            return any(u.display_name == user.display_name for u in self.users)

    def __iter__(self):
        return self.iterate()

    @property
    def project(self):
        return self.parent

    def create(self, name):
        self.project.run_security_query('add user %s' % name)
        return User(client=self._client, parent=self, display_name=name)

    def iterate(self):
        """
        :return:
        """
        if not self._iter_local:
            params = dict()
            url = self.resource()
            resp = self._client.get(url, params=params)

            Users.parse(self._client, resp, obj=self)

        for user in self.users:
            yield user

    def delete(self, name):
        if isinstance(name, User):
            name = name.display_name

        self.project.run_security_query('remove user %s' % name)
