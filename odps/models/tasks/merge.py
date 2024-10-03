# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import random
import re
import time
import warnings
from collections import namedtuple

from ... import compat, serializers
from ...config import options
from .core import Task, build_execute_method

_ARCHIVE_TABLE_REGEX = re.compile(
    r"^alter\s+table\s+(?P<table>[^\s;]+)\s+(|partition\s*\((?P<partition>[^)]+)\s*\))\s*"
    r"archive[\s;]*$",
    re.I,
)
_FREEZE_COMMAND_REGEX = re.compile(
    r"^alter\s+table\s+(?P<table>[^\s;]+)\s+(|partition\s*\((?P<partition>[^)]+)\s*\))\s*"
    r"(?P<command>freeze|restore)[\s;]*$",
    re.I,
)
_MERGE_SMALL_FILES_REGEX = re.compile(
    r"^alter\s+table\s+(?P<table>[^\s;]+)\s+(|partition\s*\((?P<partition>[^)]+)\s*\))\s*"
    r"(merge\s+smallfiles|compact\s+(?P<compact_type>[^\s;]+)(|\s+(?P<compact_opts>[^;]+)))[\s;]*$",
    re.I,
)

_COMPACT_TYPES = {
    "major": "major_compact",
    "minor": "minor_compact",
}
_FREEZE_TYPES = {
    "freeze": "backup",
    "restore": "restore",
}


_MergeTaskTableProps = namedtuple(
    "_MergeTaskTableProps", "table, schema, task_table_name"
)


class MergeTask(Task):
    _root = "Merge"

    table = serializers.XMLNodeField("TableName")

    def __init__(self, name=None, **kwargs):
        name_prefix = kwargs.pop("name_prefix", None) or "merge_task"
        if name is None:
            name = "{0}_{1}_{2}".format(
                name_prefix, int(time.time()), random.randint(100000, 999999)
            )
        kwargs["name"] = name
        super(MergeTask, self).__init__(**kwargs)

    @staticmethod
    def _extract_table_props(odps, table, partition=None, schema=None, project=None):
        from ...core import ODPS

        schema = schema or odps.schema
        if not isinstance(table, compat.six.string_types):
            if table.get_schema():
                schema = table.get_schema().name
            table_name = table.full_table_name
        else:
            table_name = table
            table = odps.get_table(table, project=project, schema=schema)
            _, schema, _ = odps._split_object_dots(table_name)
        if partition:
            table_name += " partition(%s)" % (ODPS._parse_partition_string(partition))
        return _MergeTaskTableProps(table, schema, table_name.replace("`", ""))

    @staticmethod
    def _parse_compact_opts(force_mode, recent_hours, kwargs):
        compact_opts = kwargs.pop("compact_opts", None)
        if not compact_opts:
            return force_mode, recent_hours
        if force_mode is not None or recent_hours is not None:
            raise TypeError(
                "compact_opts and force_mode / recent_hours "
                "can not be specified at the same time"
            )
        compact_opts_list = compact_opts.lower().split()
        if "-f" in compact_opts_list:
            force_mode = True
        try:
            hours_index = compact_opts_list.index("-h")
        except ValueError:
            hours_index = None

        if hours_index is not None:
            if (
                hours_index + 1 >= len(compact_opts_list)
                or not compact_opts_list[hours_index + 1].isnumeric()
            ):
                raise ValueError("Need to specify hours after -h suffix")
            recent_hours = int(compact_opts_list[hours_index + 1])

        return force_mode, recent_hours

    @classmethod
    def _create_base_merge_task(
        cls,
        odps,
        table,
        partition=None,
        project=None,
        schema=None,
        hints=None,
        quota_name=None,
        name_prefix=None,
    ):
        props = cls._extract_table_props(
            odps, table, partition=partition, schema=schema, project=project
        )

        hints = hints or dict()
        if options.default_task_settings:
            hints.update(options.default_task_settings)

        if odps.is_schema_namespace_enabled(hints) or props.schema is not None:
            hints.update(
                {
                    "odps.sql.allow.namespace.schema": "true",
                    "odps.namespace.schema": "true",
                }
            )
        if props.schema is not None:
            hints["odps.default.schema"] = props.schema
        if quota_name or odps.quota_name:
            hints["odps.task.wlm.quota"] = quota_name or odps.quota_name

        task = cls(table=props.task_table_name, name_prefix=name_prefix)
        task.update_settings(hints)
        return task, props

    @classmethod
    def _submit_merge_task(
        cls,
        odps,
        task,
        project=None,
        priority=None,
        running_cluster=None,
        create_callback=None,
    ):
        priority = priority if priority is not None else options.priority
        if priority is None and options.get_priority is not None:
            priority = options.get_priority(odps)

        project = odps.get_project(name=project)
        return project.instances.create(
            task=task,
            running_cluster=running_cluster,
            priority=priority,
            create_callback=create_callback,
        )

    @classmethod
    def run_merge_files(
        cls,
        odps,
        table,
        partition=None,
        project=None,
        schema=None,
        hints=None,
        priority=None,
        running_cluster=None,
        compact_type=None,
        force_mode=None,
        recent_hours=None,
        quota_name=None,
        create_callback=None,
        **kwargs
    ):
        """
        Start running a task to merge multiple files in tables.

        :param table: name of the table to optimize
        :param partition: partition to optimize
        :param project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param hints: settings for merge task.
        :param priority: instance priority, 9 as default
        :param running_cluster: cluster to run this instance
        :param compact_type: compact option for transactional table, can be major or minor.
        :return: instance
        :rtype: :class:`odps.models.Instance`
        """
        force_mode, recent_hours = cls._parse_compact_opts(
            force_mode, recent_hours, kwargs
        )
        if kwargs:
            raise TypeError("Unsupported keyword arguments %s" % ", ".join(kwargs))

        prefix = "merge_task" if compact_type is None else "compact_task"
        task, props = cls._create_base_merge_task(
            odps,
            table,
            partition=partition,
            project=project,
            schema=schema,
            hints=hints,
            name_prefix=prefix,
            quota_name=quota_name,
        )

        hints = hints or dict()
        compact_type = _COMPACT_TYPES.get(compact_type)
        if compact_type:
            hints.update(
                {
                    "odps.merge.txn.table.compact": compact_type,
                    "odps.merge.restructure.action": "hardlink",
                }
            )
            if compact_type == "minor_compact":
                if (
                    recent_hours is not None
                    and recent_hours < props.table.acid_data_retain_hours
                    and not force_mode
                ):
                    warnings.warn(
                        "setting 'recentHoursThresholdForPartialCompact' below the data "
                        "retention period (%s hours) prevents past time travel. "
                        "It's now set to match the retention period. "
                        "Use -f to override." % props.table.acid_data_retain_hours
                    )
                    recent_hours = props.table.acid_data_retain_hours
                recent_hours = recent_hours or -1
                hints["odps.merge.txn.table.compact.txn.id"] = str(recent_hours)

        task.update_settings(hints)

        return cls._submit_merge_task(
            odps,
            task,
            project=project,
            priority=priority,
            running_cluster=running_cluster,
            create_callback=create_callback,
        )

    execute_merge_files = build_execute_method(
        run_merge_files,
        """
        Execute a task to merge multiple files in tables and wait for termination.
        """,
    )

    @classmethod
    def run_archive_table(
        cls,
        odps,
        table,
        partition=None,
        project=None,
        schema=None,
        hints=None,
        priority=None,
        running_cluster=None,
        quota_name=None,
        create_callback=None,
    ):
        """
        Start running a task to archive tables.

        :param table: name of the table to archive
        :param partition: partition to archive
        :param project: project name, if not provided, will be the default project
        :param hints: settings for table archive task.
        :param priority: instance priority, 9 as default
        :return: instance
        :rtype: :class:`odps.models.Instance`
        """
        task, props = cls._create_base_merge_task(
            odps,
            table,
            partition=partition,
            project=project,
            schema=schema,
            hints=hints,
            name_prefix="archive_task",
            quota_name=quota_name,
        )
        task._update_property_json("archiveSettings", {"odps.merge.archive.flag": True})
        return cls._submit_merge_task(
            odps,
            task,
            project=project,
            priority=priority,
            running_cluster=running_cluster,
            create_callback=create_callback,
        )

    execute_archive_table = build_execute_method(
        run_archive_table,
        """
        Execute a task to archive tables and wait for termination.
        """,
    )

    @classmethod
    def run_freeze_command(
        cls,
        odps,
        table,
        partition=None,
        command=None,
        project=None,
        schema=None,
        hints=None,
        priority=None,
        running_cluster=None,
        quota_name=None,
        create_callback=None,
    ):
        """
        Start running a task to freeze or restore tables.

        :param table: name of the table to archive
        :param partition: partition to archive
        :param command: freeze command to execute, can be freeze or restore
        :param project: project name, if not provided, will be the default project
        :param hints: settings for table archive task.
        :param priority: instance priority, 9 as default
        :return: instance
        :rtype: :class:`odps.models.Instance`
        """
        task, props = cls._create_base_merge_task(
            odps,
            table,
            partition=partition,
            project=project,
            schema=schema,
            hints=hints,
            quota_name=quota_name,
            name_prefix=command.lower() + "_task",
        )

        hints = hints or dict()
        hints["odps.merge.cold.storage.mode"] = _FREEZE_TYPES[command.lower()]
        task.update_settings(hints)

        return cls._submit_merge_task(
            odps,
            task,
            project=project,
            priority=priority,
            running_cluster=running_cluster,
            create_callback=create_callback,
        )

    execute_freeze_command = build_execute_method(
        run_freeze_command,
        """
        Execute a task to archive tables and wait for termination.
        """,
    )

    @classmethod
    def submit_alter_table_instance(
        cls,
        odps,
        sql,
        project=None,
        schema=None,
        priority=None,
        running_cluster=None,
        hints=None,
        quota_name=None,
        create_callback=None,
    ):
        command_to_call = [
            (_ARCHIVE_TABLE_REGEX, cls.run_archive_table),
            (_FREEZE_COMMAND_REGEX, cls.run_freeze_command),
            (_MERGE_SMALL_FILES_REGEX, cls.run_merge_files),
        ]
        for cmd_regex, run_cmd in command_to_call:
            cmd_regex_match = cmd_regex.match(sql)
            if cmd_regex_match:
                kwargs = cmd_regex_match.groupdict().copy()
                kwargs.update(
                    {
                        "project": project,
                        "schema": schema,
                        "hints": hints,
                        "running_cluster": running_cluster,
                        "priority": priority,
                        "quota_name": quota_name,
                        "create_callback": create_callback,
                    }
                )
                return run_cmd(odps, **kwargs)
        return None
