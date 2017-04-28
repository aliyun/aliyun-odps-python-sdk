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

import json

from .core import LazyLoad, XMLRemoteModel
from .functions import Functions
from .instances import Instances, CachedInstances
from .ml import OfflineModels, OnlineModels
from .resources import Resources
from .tables import Tables
from .volumes import Volumes
from .xflows import XFlows
from .security.users import Users, User
from .security.roles import Roles
from .. import serializers, utils
from ..compat import six


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

    _user_cache = dict()
    __slots__ = '_policy_cache', '_tunnel_endpoint'

    class Cluster(XMLRemoteModel):

        name = serializers.XMLNodeField('Name')
        quota_id = serializers.XMLNodeField('QuotaID')

        @classmethod
        def deserial(cls, content, obj=None, **kw):
            ret = super(Project.Cluster, cls).deserial(content, obj=obj, **kw)
            if not getattr(ret, 'name', None) or not getattr(ret, 'quota_id', None):
                raise ValueError('Missing arguments: name or quotaID')
            return ret

    class ExtendedProperties(XMLRemoteModel):
        extended_properties = serializers.XMLNodePropertiesField('ExtendedProperties', 'Property',
                                                                 key_tag='Name', value_tag='Value',
                                                                 set_to_parent=True)

    class AuthQueryRequest(serializers.XMLSerializableModel):
        _root = 'Authorization'
        query = serializers.XMLNodeField('Query')
        use_json = serializers.XMLNodeField('ResponseInJsonFormat', type='bool')

    class AuthQueryResponse(serializers.XMLSerializableModel):
        _root = 'Authorization'
        result = serializers.XMLNodeField('Result')

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

    def __init__(self, *args, **kwargs):
        self._tunnel_endpoint = None
        super(Project, self).__init__(*args, **kwargs)

    def reload(self):
        self._policy_cache = None

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
            Project.ExtendedProperties.parse(self._client, resp, parent=self)

            return self._getattr('extended_properties')

        return super(Project, self).__getattribute__(attr)

    def __init__(self, **kw):
        super(Project, self).__init__(**kw)
        self._policy_cache = None

    @property
    def tables(self):
        return Tables(client=self._client, parent=self)

    @property
    def instances(self):
        return Instances(client=self._client, parent=self)

    @property
    def instance_queueing_infos(self):
        return CachedInstances(client=self._client, parent=self)

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

    @property
    def online_models(self):
        return OnlineModels(client=self._client, parent=self)

    @property
    def users(self):
        return Users(client=self._client, parent=self)

    @property
    def roles(self):
        return Roles(client=self._client, parent=self)

    @property
    def security_options(self):
        from .security import SecurityConfiguration
        return SecurityConfiguration(client=self._client, parent=self)

    @property
    def system_info(self):
        resp = self._client.get(self.resource() + '/system')
        return json.loads(resp.text if six.PY3 else resp.content)

    @property
    def policy(self):
        if self._policy_cache is None:
            params = dict(policy='')
            resp = self._client.get(self.resource(), params=params)
            self._policy_cache = resp.text if six.PY3 else resp.content
        if self._policy_cache:
            return json.loads(self._policy_cache)
        else:
            return None

    @policy.setter
    def policy(self, value):
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif value is None:
            value = ''
        self._policy_cache = value
        params = dict(policy='')
        self._client.put(self.resource(), data=value, params=params)

    @property
    def current_user(self):
        user_cache = type(self)._user_cache
        user_key = self._client.account.access_id + '##' + self.name
        if user_key not in user_cache:
            user = self.run_security_query('whoami')
            user_cache[user_key] = User(_client=self._client, parent=self.users,
                                        id=user['ID'], display_name=user['DisplayName'])
        return user_cache[user_key]

    def run_security_query(self, query, token=None):
        url = self.resource() + '/authorization'
        req_obj = self.AuthQueryRequest(query=query, use_json=True).serialize()
        headers = dict()
        if token:
            headers['odps-x-supervision-token'] = token

        resp = self.AuthQueryResponse.parse(self._client.post(url, headers=headers, data=req_obj))
        return json.loads(resp.result)
