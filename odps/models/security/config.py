#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from ..core import LazyLoad
from ... import serializers, utils


class SecurityConfiguration(LazyLoad):
    _root = 'SecurityConfiguration'

    class ProjectProtection(serializers.XMLSerializableModel):
        _root = 'ProjectProtection'
        protected = serializers.XMLNodeField('Protected')
        exception = serializers.XMLNodeField('Exceptions')

    check_permission_using_acl = serializers.XMLNodeField('CheckPermissionUsingAcl', type='bool')
    check_permission_using_policy = serializers.XMLNodeField('CheckPermissionUsingPolicy', type='bool')
    label_security = serializers.XMLNodeField('LabelSecurity', type='bool')
    object_creator_has_access_permission = serializers.XMLNodeField('ObjectCreatorHasAccessPermission', type='bool')
    object_creator_has_grant_permission = serializers.XMLNodeField('ObjectCreatorHasGrantPermission', type='bool')
    project_protection = serializers.XMLNodeReferenceField(ProjectProtection, 'ProjectProtection')
    check_permission_using_acl_v2 = serializers.XMLNodeField('CheckPermissionUsingAclV2', type='bool')
    check_permission_using_policy_v2 = serializers.XMLNodeField('CheckPermissionUsingPolicyV2', type='bool')
    support_acl = serializers.XMLNodeField('SupportACL', parse_callback=utils.str_to_bool, type='bool')
    support_policy = serializers.XMLNodeField('SupportPolicy', parse_callback=utils.str_to_bool, type='bool')
    support_package = serializers.XMLNodeField('SupportPackage', parse_callback=utils.str_to_bool, type='bool')
    support_acl_v2 = serializers.XMLNodeField('SupportACLV2', parse_callback=utils.str_to_bool, type='bool')
    support_package_v2 = serializers.XMLNodeField('SupportPackageV2', parse_callback=utils.str_to_bool, type='bool')
    check_permission_using_package = serializers.XMLNodeField('CheckPermissionUsingPackage', type='bool')
    create_package = serializers.XMLNodeField('CreatePackage', parse_callback=utils.str_to_bool, type='bool')
    create_package_v2 = serializers.XMLNodeField('CreatePackageV2', parse_callback=utils.str_to_bool, type='bool')

    @property
    def project(self):
        return self.parent

    def reload(self):
        resp = self._client.get(self.project.resource(), params=dict(security_configuration=''))
        self.parse(self._client, resp, obj=self)

    def update(self):
        content = self.serialize()
        self._client.put(self.project.resource(), params=dict(security_configuration=''), data=content)
