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

from .partition import Partition
from .core import Iterable
from .. import serializers, errors, types
from ..compat import six


class Partitions(Iterable):
    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems')
    partitions = serializers.XMLNodesReferencesField(Partition, 'Partition')

    def _name(self):
        return

    def _get(self, item):
        return Partition(client=self._client, parent=self, spec=item)

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            item = types.PartitionSpec(item)
            return self._get(item)
        elif isinstance(item, types.PartitionSpec):
            return self._get(item)
        return super(Partitions, self).__getitem__(item)

    def __contains__(self, item):
        if isinstance(item, (six.string_types, types.PartitionSpec)):
            if isinstance(item, six.string_types):
                item = types.PartitionSpec(item)
            partition = self._get(item)
        elif isinstance(item, Partition):
            partition = item
        else:
            return False

        try:
            partition.reload()
            return True
        except errors.NoSuchObject:
            return False

    @classmethod
    def _get_partition_spec(self, partition_spec):
        if isinstance(partition_spec, types.PartitionSpec):
            return partition_spec
        return types.PartitionSpec(partition_spec)

    def __iter__(self):
        return self.iterate_partitions()

    @property
    def project(self):
        return self.parent.project

    def iterate_partitions(self, spec=None):
        if spec is not None:
            spec = self._get_partition_spec(spec)

        params = {
            'partitions': '',
            'expectmarker': 'true'
        }
        if spec is not None and not spec.is_empty:
            params['partition'] = str(spec)

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, params=params)

            t = self.parse(self._client, resp, obj=self)
            params['marker'] = t.marker

            return t.partitions

        while True:
            partitions = _it()
            if partitions is None:
                break
            for partition in partitions:
                yield partition

    def create(self, partition_spec, if_not_exists=False, async=False):
        partition_spec = self._get_partition_spec(partition_spec)

        buf = six.StringIO()
        buf.write('ALTER TABLE %s.%s ADD ' % (self.project.name, self.parent.name))

        if if_not_exists:
            buf.write('IF NOT EXISTS ')

        buf.write('PARTITION (%s);' % partition_spec)

        from .tasks import SQLTask
        task = SQLTask(name='SQLAddPartitionTask', query=buf.getvalue())
        instance = self.project.instances.create(task=task)

        if not async:
            instance.wait_for_success()
            return self[partition_spec]
        else:
            return instance

    def delete(self, partition_spec, if_exists=False, async=False):
        if isinstance(partition_spec, Partition):
            partition_spec = partition_spec.partition_spec
        else:
            partition_spec = self._get_partition_spec(partition_spec)

        buf = six.StringIO()
        buf.write('ALTER TABLE %s.%s DROP ' % (self.project.name, self.parent.name))

        if if_exists:
            buf.write('IF EXISTS ')

        buf.write('PARTITION (%s);' % partition_spec)

        from .tasks import SQLTask
        task = SQLTask(name='SQLDropPartitionTask', query=buf.getvalue())
        instance = self.project.instances.create(task=task)

        if not async:
            instance.wait_for_success()
        else:
            return instance
