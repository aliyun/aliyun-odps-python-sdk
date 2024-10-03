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

from ... import serializers
from .core import Task, format_cdata


class CupidTask(Task):
    _root = "CUPID"

    plan = serializers.XMLNodeField("Plan", serialize_callback=format_cdata)

    def __init__(self, name=None, plan=None, hints=None, **kwargs):
        kwargs["name"] = name
        kwargs["plan"] = plan
        super(CupidTask, self).__init__(**kwargs)
        hints = hints or {}
        self.set_property("type", "cupid")
        if hints:
            self.set_property("settings", json.dumps(hints))
