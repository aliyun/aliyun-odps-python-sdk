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

import json
from collections import OrderedDict

from ... import serializers
from ...compat import enum
from ...config import options
from .core import Task


class MaxFrameTask(Task):
    __slots__ = ("_output_format", "_major_version", "_service_endpoint")
    _root = "MaxFrame"
    _anonymous_task_name = "AnonymousMaxFrameTask"

    class CommandType(enum.Enum):
        CREATE_SESSION = "CREATE_SESSION"
        PYTHON_PACK = "PYTHON_PACK"
        RAY_CLUSTER_INIT = "RAY_CLUSTER_INIT"
        RAY_CLUSTER_FREE = "RAY_CLUSTER_FREE"

    command = serializers.XMLNodeField(
        "Command",
        default=CommandType.CREATE_SESSION,
        parse_callback=lambda t: MaxFrameTask.CommandType(t.upper()),
        serialize_callback=lambda t: t.value,
    )

    def __init__(self, **kwargs):
        kwargs["name"] = kwargs.get("name") or self._anonymous_task_name
        self._major_version = kwargs.pop("major_version", None)
        self._service_endpoint = kwargs.pop("service_endpoint", None)
        super(MaxFrameTask, self).__init__(**kwargs)

        if self.properties is None:
            self.properties = OrderedDict()
        self.properties["settings"] = "{}"

    def serial(self):
        if options.default_task_settings:
            settings = options.default_task_settings.copy()
        else:
            settings = OrderedDict()

        if self._major_version is not None:
            settings["odps.task.major.version"] = self._major_version
        if self._service_endpoint is not None:
            settings["odps.service.endpoint"] = self._service_endpoint

        if "settings" in self.properties:
            settings.update(json.loads(self.properties["settings"]))

        self.properties["settings"] = json.dumps(settings)
        return super(MaxFrameTask, self).serial()
