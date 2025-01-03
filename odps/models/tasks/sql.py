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

import json
import os
import sys
import warnings
from collections import OrderedDict

from ... import serializers, utils
from ...compat import six
from ...config import options
from .core import Task, format_cdata


def collect_sql_settings(value, glob):
    from ... import __version__

    settings = OrderedDict()
    if options.default_task_settings:
        settings = options.default_task_settings

    if utils.str_to_bool(os.environ.get("PYODPS_SUBMIT_CLIENT_VERSIONS") or "true"):
        settings["PYODPS_VERSION"] = __version__
        settings["PYODPS_PYTHON_VERSION"] = sys.version

    if glob:
        if options.sql.use_odps2_extension:
            settings["odps.sql.type.system.odps2"] = True
        if options.local_timezone is not None:
            if not options.local_timezone:
                settings["odps.sql.timezone"] = "Etc/GMT"
            elif isinstance(options.local_timezone, bool):
                from ...lib import tzlocal

                zone = tzlocal.get_localzone()
                settings["odps.sql.timezone"] = utils.get_zone_name(zone)
            elif isinstance(options.local_timezone, six.string_types):
                settings["odps.sql.timezone"] = options.local_timezone
            else:
                zone = options.local_timezone
                zone_str = utils.get_zone_name(zone)
                if zone_str is None:
                    warnings.warn(
                        "Failed to get timezone string from options.local_timezone. "
                        "You need to deal with timezone in the return data yourself."
                    )
                else:
                    settings["odps.sql.timezone"] = zone_str
        if options.sql.settings:
            settings.update(options.sql.settings)
    if value:
        settings.update(value)
    return settings


class SQLTask(Task):
    __slots__ = ("_anonymous_sql_task_name",)

    _root = "SQL"
    _anonymous_sql_task_name = "AnonymousSQLTask"

    query = serializers.XMLNodeField(
        "Query", serialize_callback=lambda s: format_cdata(s, True)
    )

    def __init__(self, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = SQLTask._anonymous_sql_task_name
        super(SQLTask, self).__init__(**kwargs)

    def serial(self):
        if self.properties is None:
            self.properties = OrderedDict()

        key = "settings"
        if key not in self.properties:
            self.properties[key] = '{"odps.sql.udf.strict.mode": "true"}'

        return super(SQLTask, self).serial()

    def update_sql_settings(self, value=None, glob=True):
        settings = collect_sql_settings(value, glob)
        self.update_settings(settings)

    def update_aliases(self, value):
        self._update_property_json("aliases", value)

    @property
    def warnings(self):
        return json.loads(self.get_info("warnings")).get("warnings")


class SQLCostTask(Task):
    __slots__ = ("_anonymous_sql_cost_task_name",)

    _root = "SQLCost"
    _anonymous_sql_cost_task_name = "AnonymousSQLCostTask"

    query = serializers.XMLNodeField(
        "Query", serialize_callback=lambda s: format_cdata(s, True)
    )

    def __init__(self, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = self._anonymous_sql_cost_task_name
        super(SQLCostTask, self).__init__(**kwargs)

    def update_sql_cost_settings(self, value=None, glob=True):
        settings = collect_sql_settings(value, glob)
        self.update_settings(settings)


class SQLRTTask(Task):
    _root = "SQLRT"

    def update_sql_rt_settings(self, value=None, glob=True):
        settings = collect_sql_settings(value, glob)
        self.update_settings(settings)
