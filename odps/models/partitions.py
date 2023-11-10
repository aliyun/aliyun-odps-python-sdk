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

from collections import defaultdict, OrderedDict

from .. import serializers, errors, types
from ..compat import six
from ..utils import with_wait_argument
from .core import Iterable
from .partition import Partition


class PartitionSpecCondition(object):
    _predicates = OrderedDict([
        ("==", lambda a, b: a == b),
        (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b),
        ("<>", lambda a, b: a != b),
        ("!=", lambda a, b: a != b),
        (">", lambda a, b: a > b),
        ("<", lambda a, b: a < b),
        ("=", lambda a, b: a == b),
    ])

    def __init__(self, part_fields, condition=None):
        self._part_to_conditions = defaultdict(list)
        field_set = set(part_fields)
        condition = str(condition) if condition else None
        condition_splits = condition.split(",") if condition else []
        for split in condition_splits:
            for pred in self._predicates:
                if pred not in split:
                    continue

                parts = split.split(pred, 1)
                if len(parts) != 2:
                    raise ValueError("Invalid partition condition %r" % split)
                part = parts[0].strip()
                val = parts[1].strip().replace('"', '').replace("'", '')

                if part not in field_set:
                    raise ValueError("Invalid partition field %r" % part)

                self._part_to_conditions[part].append((pred, val))
                break
            else:
                raise ValueError("Invalid partition condition %r" % split)
        specs = []
        for field in part_fields:
            if (
                field not in self._part_to_conditions
                or len(self._part_to_conditions[field]) > 1
                or self._part_to_conditions[field][0][0] not in ("=", "==")
            ):
                break
            specs.append("%s=%s" % (field, self._part_to_conditions.pop(field)[0][1]))
        self.partition_spec = types.PartitionSpec(",".join(specs)) if specs else None

    def match(self, spec):
        for field, conditions in self._part_to_conditions.items():
            real_val = spec[field]
            for pred, val in conditions:
                if not self._predicates[pred](real_val, val):
                    return False
        return True


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
    def _get_partition_spec(cls, partition_spec):
        if isinstance(partition_spec, types.PartitionSpec):
            return partition_spec
        return types.PartitionSpec(partition_spec)

    def __iter__(self):
        return self.iterate_partitions()

    @property
    def project(self):
        return self.parent.project

    def iterate_partitions(self, spec=None, reverse=False):
        condition = PartitionSpecCondition(
            [pt.name for pt in self.parent.table_schema.partitions], spec
        )
        spec = condition.partition_spec

        actions = ['partitions']
        params = {'expectmarker': 'true'}
        if reverse:
            actions.append('reverse')
        if spec is not None and not spec.is_empty:
            params['partition'] = str(spec)
        schema_name = self._get_schema_name()
        if schema_name:
            params['curr_schema'] = schema_name

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, actions=actions, params=params)

            t = self.parse(self._client, resp, obj=self)
            params['marker'] = t.marker

            return t.partitions

        while True:
            partitions = _it()
            if partitions is None:
                break
            for partition in partitions:
                if condition.match(partition.partition_spec):
                    yield partition

    def get_max_partition(self, spec=None, skip_empty=True, reverse=False):
        table_parts = self.parent.table_schema.partitions

        if spec is not None:
            spec = self._get_partition_spec(spec)
            if len(spec) >= len(table_parts):
                raise ValueError(
                    "Size of prefix should not exceed number of partitions of the table"
                )

            for exist_pt, user_pt_name in zip(table_parts, spec.kv):
                if exist_pt.name != user_pt_name:
                    table_pt_str = ",".join(pt.name for pt in table_parts[:len(spec)])
                    prefix_pt_str = ",".join(spec.kv.keys())
                    raise ValueError(
                        "Partition prefix %s not agree with table partitions %s",
                        prefix_pt_str,
                        table_pt_str,
                    )

        part_values = [
            (part, tuple(part.partition_spec.values()))
            for part in self.iterate_partitions(spec)
        ]
        if not part_values:
            return None
        elif not skip_empty:
            return max(part_values, key=lambda tp: tp[1])[0]
        else:
            reversed_table_parts = sorted(part_values, key=lambda tp: tp[1], reverse=not reverse)
            return next(
                (
                    part
                    for part, _ in reversed_table_parts
                    if not skip_empty or part.physical_size > 0
                ),
                None,
            )

    @with_wait_argument
    def create(self, partition_spec, if_not_exists=False, async_=False, hints=None):
        if isinstance(partition_spec, Partition):
            partition_spec = partition_spec.partition_spec
        else:
            partition_spec = self._get_partition_spec(partition_spec)

        buf = six.StringIO()
        buf.write('ALTER TABLE %s ADD ' % self.parent.full_table_name)

        if if_not_exists:
            buf.write('IF NOT EXISTS ')

        buf.write('PARTITION (%s);' % partition_spec)

        from .tasks import SQLTask
        task = SQLTask(name='SQLAddPartitionTask', query=buf.getvalue())
        hints = hints or {}
        schema_name = self._get_schema_name()
        if schema_name is not None:
            hints["odps.sql.allow.namespace.schema"] = "true"
            hints["odps.namespace.schema"] = "true"
        task.update_sql_settings(hints)
        instance = self.project.parent[self._client.project].instances.create(task=task)

        if not async_:
            instance.wait_for_success()
            return self[partition_spec]
        else:
            return instance

    @with_wait_argument
    def delete(self, partition_spec, if_exists=False, async_=False, hints=None):
        if isinstance(partition_spec, Partition):
            partition_spec = partition_spec.partition_spec
        else:
            partition_spec = self._get_partition_spec(partition_spec)

        buf = six.StringIO()
        buf.write('ALTER TABLE %s DROP ' % self.parent.full_table_name)

        if if_exists:
            buf.write('IF EXISTS ')

        buf.write('PARTITION (%s);' % partition_spec)

        from .tasks import SQLTask
        task = SQLTask(name='SQLDropPartitionTask', query=buf.getvalue())

        hints = hints or {}
        hints['odps.sql.submit.mode'] = ''
        schema_name = self._get_schema_name()
        if schema_name is not None:
            hints["odps.sql.allow.namespace.schema"] = "true"
            hints["odps.namespace.schema"] = "true"
        task.update_sql_settings(hints)
        instance = self.project.parent[self._client.project].instances.create(task=task)

        if not async_:
            instance.wait_for_success()
        else:
            return instance
