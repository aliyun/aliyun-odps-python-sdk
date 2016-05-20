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

from .core import Container
from .project import Project
from .. import serializers, errors
from ..compat import six


class Projects(Container):

    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems')
    projects = serializers.XMLNodesReferencesField(Project, 'Project')

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
