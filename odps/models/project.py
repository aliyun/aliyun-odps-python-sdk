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
import time

from ..compat import Enum
from ..errors import SecurityQueryError
from .core import LazyLoad, XMLRemoteModel
from .functions import Functions
from .instances import Instances, CachedInstances
from .ml import OfflineModels
from .resources import Resources
from .tables import Tables
from .volumes import Volumes
from .xflows import XFlows
from .security.users import Users, User
from .security.roles import Roles
from .. import serializers, utils
from ..compat import six


_notset = object()


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
    __slots__ = (
        "_policy_cache",
        "_logview_host",
        "_tunnel_endpoint",
        "_all_props_loaded",
        "_extended_props_loaded",
    )

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
        _extended_properties = serializers.XMLNodePropertiesField('ExtendedProperties', 'Property',
                                                                 key_tag='Name', value_tag='Value',
                                                                 set_to_parent=True)

    class AuthQueryRequest(serializers.XMLSerializableModel):
        _root = 'Authorization'
        query = serializers.XMLNodeField('Query')
        use_json = serializers.XMLNodeField('ResponseInJsonFormat', type='bool')

    class AuthQueryResponse(serializers.XMLSerializableModel):
        _root = 'Authorization'
        result = serializers.XMLNodeField('Result')

    class AuthQueryStatus(Enum):
        TERMINATED = "TERMINATED"
        RUNNING = "RUNNING"
        FAILED = "FAILED"

    class AuthQueryStatusResponse(serializers.XMLSerializableModel):
        _root = "AuthorizationQuery"
        result = serializers.XMLNodeField('Result')
        status = serializers.XMLNodeField(
            'Status', parse_callback=lambda s: Project.AuthQueryStatus(s.upper())
        )

    class AuthQueryInstance(object):
        def __init__(self, project, instance_id, output_json=True):
            self.project = project
            self.instance_id = instance_id
            self.output_json = output_json

        def wait_for_completion(self, interval=1):
            while not self.is_terminated:
                time.sleep(interval)

        def wait_for_success(self, interval=1):
            self.wait_for_completion(interval=interval)
            status = self.query_status()
            if status.status == Project.AuthQueryStatus.TERMINATED:
                return json.loads(status.result) if self.output_json else status.result
            else:
                raise SecurityQueryError("Authorization query failed: " + status.result)

        def query_status(self):
            resource = self.project.resource() + "/authorization/" + self.instance_id
            resp = self.project._client.get(resource)
            return Project.AuthQueryStatusResponse.parse(resp)

        @property
        def is_terminated(self):
            status = self.query_status()
            return status.status != Project.AuthQueryStatus.RUNNING

        @property
        def is_successful(self):
            status = self.query_status()
            return status.status == Project.AuthQueryStatus.TERMINATED

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
    _extended_properties = serializers.XMLNodePropertiesField('ExtendedProperties', 'Property',
                                                             key_tag='Name', value_tag='Value')
    state = serializers.XMLNodeField('State')
    clusters = serializers.XMLNodesReferencesField(Cluster, 'Clusters', 'Cluster')

    def __init__(self, *args, **kwargs):
        self._tunnel_endpoint = None
        self._policy_cache = None
        self._all_props_loaded = False
        super(Project, self).__init__(*args, **kwargs)

    def reload(self, all_props=False):
        self._policy_cache = None

        url = self.resource()
        params = dict()
        if all_props:
            params["properties"] = "all"
        resp = self._client.get(url, params=params)

        self.parse(self._client, resp, obj=self)

        self.owner = resp.headers['x-odps-owner']
        self.creation_time = utils.parse_rfc822(resp.headers['x-odps-creation-time'])
        self.last_modified_time = utils.parse_rfc822(resp.headers['Last-Modified'])

        self._loaded = True
        self._all_props_loaded = all_props
        self._extended_props_loaded = False

    @property
    def extended_properties(self):
        if self._extended_props_loaded:
            return self._getattr("_extended_properties")

        url = self.resource()
        params = {"extended": ""}

        resp = self._client.get(url, params=params)
        Project.ExtendedProperties.parse(self._client, resp, parent=self)

        self._extended_props_loaded = True
        return self._getattr("_extended_properties")

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
        return json.loads(resp.content.decode() if six.PY3 else resp.content)

    @property
    def odps(self):
        from ..core import ODPS

        client = self._client

        return ODPS._from_account(
            client.account,
            client.project,
            endpoint=client.endpoint,
            tunnel_endpoint=self._tunnel_endpoint,
            logview_host=self._logview_host,
            app_account=getattr(client, 'app_account', None),
        )

    @property
    def policy(self):
        if self._policy_cache is None:
            params = dict(policy='')
            resp = self._client.get(self.resource(), params=params)
            self._policy_cache = resp.content.decode() if six.PY3 else resp.content
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

    def run_security_query(self, query, token=None, output_json=True):
        url = self.resource() + '/authorization'
        req_obj = self.AuthQueryRequest(query=query, use_json=output_json).serialize()
        headers = {'Content-Type': 'application/xml'}
        if token:
            headers['odps-x-supervision-token'] = token

        query_resp = self._client.post(url, headers=headers, data=req_obj)
        resp = self.AuthQueryResponse.parse(query_resp)

        if query_resp.status_code == 200:
            return json.loads(resp.result) if output_json else resp.result
        return Project.AuthQueryInstance(self, resp.result, output_json=output_json)

    def get_property(self, item, default=_notset):
        if not self._all_props_loaded:
            self.reload(all_props=True)
        if item in self.properties:
            return self.properties[item]
        try:
            return self.extended_properties[item]
        except KeyError:
            if default is _notset:
                raise
            return default
