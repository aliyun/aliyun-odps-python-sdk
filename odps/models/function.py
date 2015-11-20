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

from .core import LazyLoad
from .resource import Resource
from .. import serializers, utils


class Function(LazyLoad):
    """
    Function can be used in UDF when user writes a SQL.
    """

    _root = 'Function'

    name = serializers.XMLNodeField('Alias')
    owner = serializers.XMLNodeField('Owner')
    creation_time = serializers.XMLNodeField('CreationTime', parse_callback=utils.parse_rfc822)
    class_type = serializers.XMLNodeField('ClassType')
    _resources = serializers.XMLNodesField('Resources', 'ResourceName')

    def __init__(self, **kwargs):
        resources = kwargs.pop('resources', None)
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

        resources = self.parent.parent.resources
        resources = [resources[name] for name in self._resources]
        return resources

    @resources.setter
    def resources(self, value):
        get_name = lambda res: res.name if isinstance(res, Resource) else res
        self._resources = [get_name(res) for res in value]

    def drop(self):
        """
        Delete this Function.

        :return: None
        """

        return self.parent.delete(self)
