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

from .core import LazyLoad, XMLRemoteModel
from .tables import Tables
from .instances import Instances
from .functions import Functions
from .resources import Resources
from .volumes import Volumes
from .xflows import XFlows
from .ml.offlinemodels import OfflineModels
from .. import serializers, utils


class Project(LazyLoad):
    """
    Project is the counterpart of **database** in a RDBMS.

    By get an object of Project, users can get the properties like ``name``, ``owner``,
    ``comment``, ``creation_time``, ``last_modified_time``, and so on.

    These properties will not load from remote ODPS service, unless users
    try to get them explicitly. If users want to check the newest status,
    try use ``reload`` method.

    :Example:

    >>> project = odps.get_project('my_project')
    >>> project.last_modified_time  # this property will be fetched from the remote ODPS service
    >>> project.last_modified_time  # Once loaded, the property will not bring remote call
    >>> project.owner  # so do the other properties, they are fetched together
    >>> project.reload()  # force to update each properties
    >>> project.last_modified_time  # already updated
    """

    __slots__ = 'extended_properties'

    class Cluster(XMLRemoteModel):

        name = serializers.XMLNodeField('Name')
        quota_id = serializers.XMLNodeField('QuotaID')

        def __init__(self, **kwargs):
            self.name, self.quota_id = None, None
            super(Project.Cluster, self).__init__(**kwargs)
            if not self.name or not self.quota_id:
                raise ValueError('Missing arguments: name or quotaID')

    class ExtendedProperties(XMLRemoteModel):
        extended_properties = serializers.XMLNodePropertiesField('ExtendedProperties', 'Property',
                                                                 key_tag='Name', value_tag='Value',
                                                                 set_to_parent=True)

    name = serializers.XMLNodeField('Name')
    owner = serializers.XMLNodeField('Owner')
    comment = serializers.XMLNodeField('Comment')
    creation_time = serializers.XMLNodeField('CreationTime',
                                             parse_callback=utils.parse_rfc822)
    last_modified_time = serializers.XMLNodeField('LastModifiedTime',
                                                  parse_callback=utils.parse_rfc822)
    project_group_name = serializers.XMLNodeField('ProjectGroupName')
    properties = serializers.XMLNodePropertiesField('Properties', 'Property',
                                                    key_tag='Name', value_tag='Value')
    extended_properties = serializers.XMLNodePropertiesField('ExtendedProperties', 'Property',
                                                             key_tag='Name', value_tag='Value')
    state = serializers.XMLNodeField('State')
    clusters = serializers.XMLNodesReferencesField(Cluster, 'Clusters', 'Cluster')

    def reload(self):
        url = self.resource()
        resp = self._client.get(url)

        self.parse(self._client, resp, obj=self)

        self.owner = resp.headers['x-odps-owner']
        self.creation_time = utils.parse_rfc822(resp.headers['x-odps-creation-time'])
        self.last_modified_time = utils.parse_rfc822(resp.headers['Last-Modified'])

        self._loaded = True

    def __getattribute__(self, attr):
        if attr == 'extended_properties':
            url = self.resource()
            params = {'extended': ''}

            resp = self._client.get(url, params=params)
            model = Project.ExtendedProperties.parse(self._client, resp, parent=self)

            return self._getattr('extended_properties')

        return super(Project, self).__getattribute__(attr)

    @property
    def tables(self):
        return Tables(client=self._client, parent=self)

    @property
    def instances(self):
        return Instances(client=self._client, parent=self)

    @property
    def functions(self):
        return Functions(client=self._client, parent=self)

    @property
    def resources(self):
        return Resources(client=self._client, parent=self)

    @property
    def volumes(self):
        return Volumes(client=self._client, parent=self)

    @property
    def xflows(self):
        return XFlows(client=self._client, parent=self)

    @property
    def offline_models(self):
        return OfflineModels(client=self._client, parent=self)
