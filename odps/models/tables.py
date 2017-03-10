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

from .core import Iterable
from .table import Table, TableSchema
from .. import serializers, errors
from ..compat import six, OrderedDict
from ..utils import to_str


class Tables(Iterable):

    marker = serializers.XMLNodeField('Marker')
    max_items = serializers.XMLNodeField('MaxItems', parse_callback=int)
    tables = serializers.XMLNodesReferencesField(Table, 'Table')

    def _get(self, item):
        return Table(client=self._client, parent=self, name=item)

    def __contains__(self, item):
        if isinstance(item, six.string_types):
            table = self._get(item)
        elif isinstance(item, Table):
            table = item
        else:
            return False

        try:
            table.reload()
            return True
        except errors.NoSuchObject:
            return False

    def __iter__(self):
        return self.iterate()

    def iterate(self, name=None, owner=None):
        """

        :param name: the prefix of table name
        :param owner:
        :return:
        """

        params = {'expectmarker': 'true'}
        if name is not None:
            params['name'] = name
        if owner is not None:
            params['owner'] = owner

        def _it():
            last_marker = params.get('marker')
            if 'marker' in params and \
                (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, params=params)

            t = Tables.parse(self._client, resp, obj=self)
            params['marker'] = t.marker

            return t.tables

        while True:
            tables = _it()
            if tables is None:
                break
            for table in tables:
                yield table

    def _gen_create_table_sql(self, table_name, table_schema, comment=None,
                              if_not_exists=False, lifecycle=None,
                              shard_num=None, hub_lifecycle=None):
        project_name = self._parent.name

        buf = six.StringIO()

        buf.write('CREATE TABLE ')
        if if_not_exists:
            buf.write('IF NOT EXISTS ')
        buf.write('%s.`%s` ' % (project_name, table_name))

        for i, arr in enumerate([table_schema.simple_columns,
                                 table_schema.partitions]):
            if i == 1 and not arr:
                continue
            if i == 1:
                buf.write(' PARTITIONED BY ')
            size = len(arr)
            buf.write('(')
            for idx, column in enumerate(arr):
                buf.write('`%s` %s' % (to_str(column.name), to_str(column.type)))
                if column.comment:
                    buf.write(" COMMENT '%s'" % to_str(column.comment))
                if idx < size - 1:
                    buf.write(',')
            buf.write(')')
            if i == 0 and comment is not None:
                buf.write(" COMMENT '%s'" % comment)

        if lifecycle is not None:
            buf.write(' LIFECYCLE %s' % lifecycle)
        if shard_num is not None:
            buf.write(' INTO %s SHARDS' % shard_num)
            if hub_lifecycle is not None:
                buf.write(' HUBLIFECYCLE %s' % hub_lifecycle)

        return buf.getvalue()

    def create(self, table_name, table_schema, comment=None, if_not_exists=False,
               lifecycle=None, shard_num=None, hub_lifecycle=None, async=False):

        def make_schema_dict(schema_repr):
            if schema_repr is None:
                return schema_repr
            if not isinstance(schema_repr, dict):
                if isinstance(schema_repr, six.string_types):
                    schema_array = []
                    for f in schema_repr.split(','):
                        splited = f.strip().rsplit(' ', 1)
                        splited[0] = splited[0].strip('`')
                        schema_array.append(tuple(splited))
                    schema_repr = schema_array
                schema_repr = OrderedDict(schema_repr)
            return schema_repr

        if not isinstance(table_schema, TableSchema):
            if isinstance(table_schema, tuple):
                table_schema, partition_schema = table_schema
            else:
                partition_schema = None
            table_schema = make_schema_dict(table_schema)
            partition_schema = make_schema_dict(partition_schema)

            table_schema = TableSchema.from_dict(table_schema, partition_schema)

        sql = self._gen_create_table_sql(table_name, table_schema, comment=comment,
                                         if_not_exists=if_not_exists, lifecycle=lifecycle,
                                         shard_num=shard_num, hub_lifecycle=hub_lifecycle)

        from .tasks import SQLTask
        task = SQLTask(name='SQLCreateTableTask', query=sql)
        instance = self._parent.instances.create(task=task)

        if not async:
            instance.wait_for_success()

            table = Table(parent=self, client=self._client,
                          name=table_name, schema=table_schema.to_ignorecase_schema())
            return table
        else:
            return instance

    def _gen_delete_table_sql(self, table_name, if_exists=False):
        project_name = self._parent.name

        buf = six.StringIO()

        buf.write('DROP TABLE ')
        if if_exists:
            buf.write('IF EXISTS ')
        buf.write('%s.`%s`' % (project_name, table_name))

        return buf.getvalue()

    def delete(self, table_name, if_exists=False, async=False):
        if isinstance(table_name, Table):
            table_name = table_name.name

        del self[table_name]  # release table in cache

        sql = self._gen_delete_table_sql(table_name, if_exists=if_exists)

        from .tasks import SQLTask
        task = SQLTask(name='SQLDropTableTask', query=sql)
        instance = self._parent.instances.create(task=task)

        if not async:
            instance.wait_for_success()
        else:
            return instance
