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

from .core import Iterable
from .table import Table
from .. import serializers, errors
from ..compat import six


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

    def create(self, table_name, table_schema, comment=None, if_not_exists=False,
               lifecycle=None, shard_num=None, hub_lifecycle=None, async_=False, **kw):
        async_ = kw.pop('async', async_)
        sql = Table.gen_create_table_sql(table_name, table_schema, comment=comment,
                                         if_not_exists=if_not_exists, lifecycle=lifecycle,
                                         shard_num=shard_num, hub_lifecycle=hub_lifecycle,
                                         project=self._parent.name, **kw)

        from .tasks import SQLTask
        task = SQLTask(name='SQLCreateTableTask', query=sql)
        task.update_sql_settings()
        instance = self._parent.instances.create(task=task)

        if not async_:
            instance.wait_for_success()

            return self[table_name]
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

    def delete(self, table_name, if_exists=False, async_=False, **kw):
        async_ = kw.get('async', async_)
        if isinstance(table_name, Table):
            table_name = table_name.name

        del self[table_name]  # release table in cache

        sql = self._gen_delete_table_sql(table_name, if_exists=if_exists)

        from .tasks import SQLTask
        task = SQLTask(name='SQLDropTableTask', query=sql)
        task.update_sql_settings({'odps.sql.submit.mode': ''})
        instance = self._parent.instances.create(task=task)

        if not async_:
            instance.wait_for_success()
        else:
            return instance
