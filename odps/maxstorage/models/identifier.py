#
# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

"""Identifiers for storage API resource targeting.

``TableIdentifier`` / ``InstanceIdentifier`` build the ``Target`` query-param
value consumed by every storage API request.
"""


class TableIdentifier:
    """Identifies a table for the ``Target`` query param.

    ``projects.{project}.schemas.{schema}.tables.{table}`` when a schema is
    given, otherwise ``projects.{project}.tables.{table}``.
    """

    def __init__(self, project, table, schema=None):
        self.project = project
        self.table = table
        self.schema = schema

    @classmethod
    def from_table(cls, table):
        """Build from an :class:`odps.models.Table` instance."""
        project = (
            table.project.name
            if hasattr(table, "project") and hasattr(table.project, "name")
            else table.project
        )
        schema = (
            table._get_schema_name() if hasattr(table, "_get_schema_name") else None
        )
        return cls(project, table.name, schema)

    def to_target(self):
        # When schema is None the schema segment is omitted, producing
        # ``projects.{project}.tables.{table}``.  The server accepts both
        # this form and the fully-qualified
        # ``projects.{project}.schemas.{schema}.tables.{table}`` (used when
        # a schema is given); both target shapes are server-supported.
        if self.schema:
            return f"projects.{self.project}.schemas.{self.schema}.tables.{self.table}"
        return f"projects.{self.project}.tables.{self.table}"

    def __repr__(self):
        return f"TableIdentifier({self.to_target()})"


class InstanceIdentifier:
    """Identifies an instance for the ``Target`` query param.

    ``projects.{project}.instances.{instance}``.
    """

    def __init__(self, project, instance):
        self.project = project
        self.instance = instance

    @classmethod
    def from_instance(cls, instance):
        project = (
            instance.project.name
            if hasattr(instance, "project") and hasattr(instance.project, "name")
            else instance.project
        )
        return cls(project, instance.id)

    def to_target(self):
        return f"projects.{self.project}.instances.{self.instance}"

    def __repr__(self):
        return f"InstanceIdentifier({self.to_target()})"
