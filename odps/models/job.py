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

from .core import XMLRemoteModel
from .. import serializers, utils
from .tasks import Task


class Job(XMLRemoteModel):
    # define __slots__ to keep the property sequence when serializing into a xml
    __slots__ = 'name', 'comment', 'priority', 'running_clusters', 'tasks', 'run_mode'

    _root = 'Job'

    name = serializers.XMLNodeField('Name')
    comment = serializers.XMLNodeField("Comment")
    owner = serializers.XMLNodeField('Owner')
    creation_time = serializers.XMLNodeField('CreationTime',
                                             parse_callback=utils.parse_rfc822)
    last_modified_time = serializers.XMLNodeField('LastModifiedTime',
                                                  parse_callback=utils.parse_rfc822)
    priority = serializers.XMLNodeField('Priority',
                                        parse_callback=int, serialize_callback=int, default=9)
    running_cluster = serializers.XMLNodeField('RunningCluster')
    run_mode = serializers.XMLNodeField('DAG', 'RunMode', default='Sequence')
    tasks = serializers.XMLNodesReferencesField(Task, 'Tasks', '*')

    def add_task(self, task):
        if self.tasks is None:
            self.tasks = []
        self.tasks.append(task)