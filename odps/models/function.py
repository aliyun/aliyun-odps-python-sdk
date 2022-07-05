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

from .core import LazyLoad
from .resource import Resource
from .. import serializers, utils


class Function(LazyLoad):
    """
    Function can be used in UDF when user writes a SQL.
    """
    __slots__ = '_resources_objects', '_owner_changed'

    _root = 'Function'

    name = serializers.XMLNodeField('Alias')
    _owner = serializers.XMLNodeField('Owner')
    creation_time = serializers.XMLNodeField('CreationTime', parse_callback=utils.parse_rfc822)
    class_type = serializers.XMLNodeField('ClassType')
    _resources = serializers.XMLNodesField('Resources', 'ResourceName')

    def __init__(self, **kwargs):
        self._resources_objects = None
        self._owner_changed = False

        resources = kwargs.pop('resources', None)
        if 'owner' in kwargs:
            kwargs['_owner'] = kwargs.pop('owner')
        super(Function, self).__init__(**kwargs)
        if resources is not None:
            self.resources = resources

    @property
    def project(self):
        return self.parent.parent

    def reload(self):
        resp = self._client.get(self.resource())
        self.parse(self._client, resp, obj=self)

    @property
    def resources(self):
        """
        Return all the resources which this function refer to.

        :return: resources
        :rtype: list

        .. seealso:: :class:`odps.models.Resource`
        """

        if self._resources_objects is not None:
            return self._resources_objects

        resources = self.parent.parent.resources
        resources = [resources[name] for name in self._resources]
        self._resources_objects = resources
        return resources

    @resources.setter
    def resources(self, value):
        def get_resource_name(res):
            if isinstance(res, Resource):
                if res.project == self.project.name:
                    return res.name
                else:
                    return '%s/resources/%s' % (res.project, res.name)
            else:
                return res

        self._resources_objects = None
        self._resources = [get_resource_name(res) for res in value]

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        self._owner_changed = True
        self._owner = value

    def update(self):
        """
        Update this function.

        :return: None
        """
        if self._owner_changed:
            self.update_owner(self.owner)

        self._resources = [res.name for res in self.resources]
        return self.parent.update(self)

    def update_owner(self, new_owner):
        params = {
            'updateowner': ''
        }
        headers = {
            'x-odps-owner': new_owner
        }
        self._client.put(self.resource(), None, params=params, headers=headers)

    def drop(self):
        """
        Delete this Function.

        :return: None
        """

        return self.parent.delete(self)
