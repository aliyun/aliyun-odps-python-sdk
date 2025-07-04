#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from .. import errors, serializers, utils
from ..compat import six
from .core import Iterable
from .table import Table


class Tables(Iterable):
    marker = serializers.XMLNodeField("Marker")
    max_items = serializers.XMLNodeField("MaxItems", parse_callback=int)
    tables = serializers.XMLNodesReferencesField(Table, "Table")

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

    def iterate(self, name=None, owner=None, type=None, extended=False):
        """
        :param name: the prefix of table name
        :param owner: owner of the table
        :param type: type of the table
        :param extended: load extended information for table
        :return:
        """
        actions = []
        params = {"expectmarker": "true"}
        if name is not None:
            params["name"] = name
        if owner is not None:
            params["owner"] = owner
        if type is not None:
            table_type = type.upper() if isinstance(type, str) else type
            table_type = Table.Type(table_type)
            params["type"] = table_type.value
        if extended:
            actions.append("extended")

        schema_name = self._get_schema_name()
        if schema_name is not None:
            params["curr_schema"] = schema_name

        def _it():
            last_marker = params.get("marker")
            if "marker" in params and (last_marker is None or len(last_marker) == 0):
                return

            url = self.resource()
            resp = self._client.get(url, actions=actions, params=params)

            t = Tables.parse(self._client, resp, obj=self)
            params["marker"] = t.marker

            return t.tables

        while True:
            tables = _it()
            if tables is None:
                break
            for table in tables:
                yield table

    def _run_table_sql(self, query, task_name=None, hints=None, wait=True, **inst_kw):
        from .tasks import SQLTask

        task = SQLTask(name=task_name, query=query)

        hints = hints or {}
        hints["odps.sql.submit.mode"] = ""
        schema_name = self._get_schema_name()
        if schema_name is not None:
            hints["odps.sql.allow.namespace.schema"] = "true"
            hints["odps.namespace.schema"] = "true"
        if self._parent.project.odps.quota_name:
            hints["odps.task.wlm.quota"] = self._parent.project.odps.quota_name
        task.update_sql_settings(hints)

        instance_project = inst_kw.pop("instance_project", inst_kw.pop("project", None))
        if instance_project is not None:
            proj = self._parent.project.parent[instance_project]
        else:
            proj = self._parent.project
        instance = proj.instances.create(task=task, **inst_kw)

        if wait:
            instance.wait_for_success()
        return instance

    @utils.with_wait_argument
    def create(
        self,
        table_name,
        table_schema,
        comment=None,
        if_not_exists=False,
        lifecycle=None,
        shard_num=None,
        hub_lifecycle=None,
        hints=None,
        transactional=False,
        storage_tier=None,
        async_=False,
        **kw
    ):
        project_name = self._parent.project.name
        schema_name = self._get_schema_name()
        sql = Table.gen_create_table_sql(
            table_name,
            table_schema,
            comment=comment,
            if_not_exists=if_not_exists,
            lifecycle=lifecycle,
            shard_num=shard_num,
            hub_lifecycle=hub_lifecycle,
            transactional=transactional,
            project=project_name,
            schema=schema_name,
            **kw
        )

        hints = hints or {}
        if storage_tier:
            hints["odps.tiered.storage.enable"] = "true"
        instance = self._run_table_sql(
            sql, task_name="SQLCreateTableTask", hints=hints, wait=not async_
        )
        if not async_:
            return self[table_name]
        else:
            return instance

    def _gen_delete_table_sql(self, table_name, if_exists=False, table_type=None):
        project_name = self._parent.project.name
        schema_name = self._get_schema_name()

        buf = six.StringIO()

        if table_type is not None and isinstance(table_type, six.string_types):
            table_type = Table.Type(table_type.upper())

        # override provided type if the object is already cached
        cached_table_type = self._get(table_name)._getattr("type")
        if cached_table_type is not None and (
            table_type is None or table_type == Table.Type.MANAGED_TABLE
        ):
            table_type = cached_table_type

        if table_type == Table.Type.VIRTUAL_VIEW:
            type_str = "VIEW"
        elif table_type == Table.Type.MATERIALIZED_VIEW:
            type_str = "MATERIALIZED VIEW"
        else:
            type_str = "TABLE"

        buf.write("DROP %s " % type_str)
        if if_exists:
            buf.write("IF EXISTS ")
        if schema_name is not None:
            buf.write(
                "%s.%s.%s"
                % (project_name, schema_name, utils.backquote_string(table_name))
            )
        else:
            buf.write("%s.%s" % (project_name, utils.backquote_string(table_name)))

        return buf.getvalue()

    @utils.with_wait_argument
    def delete(
        self,
        table_name,
        if_exists=False,
        async_=False,
        hints=None,
        table_type=None,
        **inst_kw
    ):
        if isinstance(table_name, Table):
            table_name = table_name.name

        sql = self._gen_delete_table_sql(
            table_name, if_exists=if_exists, table_type=table_type
        )

        del self[table_name]  # release table in cache
        return self._run_table_sql(
            sql, task_name="SQLDropTableTask", hints=hints, wait=not async_, **inst_kw
        )
