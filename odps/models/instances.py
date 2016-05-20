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

import uuid
from datetime import datetime

from .core import Iterable
from .instance import Instance
from .job import Job
from .. import serializers, errors, utils
from ..compat import six


class Instances(Iterable):

    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems')
    instances = serializers.XMLNodesReferencesField(Instance, 'Instance')

    def _get(self, name):
        return Instance(client=self._client, parent=self, name=name)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            instance = self._get(item)
        elif isinstance(item, Instance):
            instance = item
        else:
            return False

        try:
            instance.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self, from_time=None, end_time=None, status=None, only_owner=None):
        params = dict()
        if status is not None:
            params['status'] = status.value
        if from_time is not None or end_time is not None:
            daterange = six.StringIO()
            if from_time is not None:
                if isinstance(from_time, datetime):
                    daterange.write(str(utils.to_timestamp(from_time)))
                else:
                    daterange.write(str(int(from_time)))
            daterange.write(':')
            if end_time is not None:
                if isinstance(end_time, datetime):
                    daterange.write(str(utils.to_timestamp(end_time)))
                else:
                    daterange.write(str(int(end_time)))
            params['daterange'] = daterange.getvalue()
        if only_owner is not None:
            params['onlyowner'] = 'yes' if only_owner else 'no'

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, params=params)

            inst = Instances.parse(self._client, resp, obj=self)
            params['marker'] = inst.marker

            return inst.instances

        while True:
            instances = _it()
            if not instances:
                break
            for instance in instances:
                yield instance

    @classmethod
    def _create_job(cls, job=None, task=None, priority=None, running_cluster=None, uuid_=None):
        job = job or Job()
        if priority is not None:
            if priority < 0:
                raise errors.ODPSError('Priority must more than or equal to zero.')
            job.priority = priority
        if running_cluster is not None:
            job.running_cluster = running_cluster
        if task is not None:
            job.add_task(task)
        if job.tasks is None or len(job.tasks) == 0:
            raise ValueError('Job tasks are required')

        guid = uuid_ or str(uuid.uuid4())
        for t in job.tasks:
            t.set_property('uuid', guid)
            if t.name is None:
                raise errors.ODPSError('Task name is required')

        return job

    @classmethod
    def _get_submit_instance_content(cls, job):
        return Instance.AnonymousSubmitInstance(job=job).serialize()

    def create(self, xml=None, job=None, task=None, priority=None, running_cluster=None):
        if xml is None:
            job = self._create_job(job=job, task=task, priority=priority,
                                   running_cluster=running_cluster)

            xml = self._get_submit_instance_content(job)

        headers = {'Content-Type': 'application/xml'}
        url = self.resource()
        resp = self._client.post(url, xml, headers=headers)

        location = resp.headers.get('Location')
        if location is None or len(location) == 0:
            raise errors.ODPSError('Invalid response, Location header required.')

        instance_id = location.rsplit('/', 1)[1]

        body = resp.text
        if body:
            instance_result = Instance.InstanceResult.parse(self._client, resp)
            results = dict([(r.name, r.result) for r in instance_result.task_results])
        else:
            results = None

        instance = Instance(name=instance_id, task_results=results,
                            parent=self, client=self._client)
        return instance
