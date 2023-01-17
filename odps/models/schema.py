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

import datetime

from .. import serializers
from ..compat import Enum
from .core import JSONRemoteModel, LazyLoad
from .functions import Functions
from .resources import Resources
from .tables import Tables
from .volumes import Volumes


def _parse_schema_time(time_str):
    return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")


class SchemaType(Enum):
    MANAGED = 0
    EXTERNAL = 1


class SchemaDescription(JSONRemoteModel):
    name = serializers.JSONNodeField('name', set_to_parent=True)
    owner = serializers.JSONNodeField('owner', set_to_parent=True)
    description = serializers.JSONNodeField('description', set_to_parent=True)
    create_time = serializers.JSONNodeField(
        'createTime', parse_callback=_parse_schema_time, set_to_parent=True
    )
    modify_time = serializers.JSONNodeField(
        'modifyTime', parse_callback=_parse_schema_time, set_to_parent=True
    )
    type = serializers.JSONNodeField(
        'type', parse_callback=lambda x: getattr(SchemaType, x.upper())
    )


class Schema(LazyLoad):
    name = serializers.XMLNodeField('Name')
    owner = serializers.XMLNodeField('Owner')
    description = serializers.XMLNodeField('Description')
    create_time = serializers.XMLNodeField('CreateTime', parse_callback=_parse_schema_time)
    modify_time = serializers.XMLNodeField('ModifyTime', parse_callback=_parse_schema_time)
    type = serializers.XMLNodeField(
        'Type', parse_callback=lambda x: getattr(SchemaType, x.upper())
    )

    def reload(self):
        desc_instance = self.project.odps.execute_sql("DESC SCHEMA %s" % self.name)
        desc_result = desc_instance.get_task_results().get("AnonymousSQLTask")
        desc_obj = SchemaDescription(parent=self)
        desc_obj.parse(self._client, desc_result, obj=desc_obj)
        self._loaded = True

    def resource(self, client=None):
        return self.parent.resource(client)

    def drop(self, async_=False):
        self.parent.delete(self, async_=async_)

    @property
    def functions(self):
        return Functions(client=self._client, parent=self)

    @property
    def resources(self):
        return Resources(client=self._client, parent=self)

    @property
    def tables(self):
        return Tables(client=self._client, parent=self)

    @property
    def volumes(self):
        return Volumes(client=self._client, parent=self)
