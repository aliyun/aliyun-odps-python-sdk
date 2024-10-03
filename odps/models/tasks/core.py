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

import importlib
import itertools
import json
import textwrap
from collections import OrderedDict

from ... import errors, serializers
from ...compat import six
from ..core import AbstractXMLRemoteModel

_type_to_task_classes = dict()


class Task(AbstractXMLRemoteModel):
    __slots__ = ("name", "comment", "properties")

    _type_indicator = "type"

    name = serializers.XMLNodeField("Name")
    type = serializers.XMLTagField(".")
    comment = serializers.XMLNodeField("Comment")
    properties = serializers.XMLNodePropertiesField(
        "Config", "Property", key_tag="Name", value_tag="Value"
    )

    @classmethod
    def _load_task_classes(cls):
        if _type_to_task_classes:
            return
        mod = importlib.import_module("odps.models.tasks")
        for v in six.itervalues(mod.__dict__):
            if not isinstance(v, type) or not issubclass(v, Task) or v is Task:
                continue
            cls_type = getattr(v, "_root", v.__name__)
            _type_to_task_classes[cls_type] = v

    def __new__(cls, *args, **kwargs):
        typo = kwargs.get("type")

        if typo is not None:
            cls._load_task_classes()
            task_cls = _type_to_task_classes.get(typo, cls)
        else:
            task_cls = cls

        return object.__new__(task_cls)

    def set_property(self, key, value):
        if self.properties is None:
            self.properties = OrderedDict()
        self.properties[key] = value

    def _update_property_json(self, field, value):
        def update(kv, dest):
            if not kv:
                return
            for k, v in six.iteritems(kv):
                if isinstance(v, bool):
                    dest[k] = "true" if v else "false"
                else:
                    dest[k] = str(v)

        if self.properties is None:
            self.properties = OrderedDict()
        if field in self.properties:
            settings = json.loads(self.properties[field])
        else:
            settings = OrderedDict()
        update(value, settings)
        self.properties[field] = json.dumps(settings)

    def update_settings(self, value):
        self._update_property_json("settings", value)

    def serialize(self):
        if type(self) is Task:
            raise errors.ODPSError("Unknown task type")
        return super(Task, self).serialize()

    @property
    def instance(self):
        return self.parent.parent

    @property
    def progress(self):
        """
        Get progress of a task.
        """
        return self.instance.get_task_progress(self.name)

    @property
    def stages(self):
        """
        Get execution stages of a task.
        """
        return self.instance.get_task_progress(self.name).stages

    @property
    def result(self):
        """
        Get execution result of the task.
        """
        return self.instance.get_task_result(self.name)

    @property
    def summary(self):
        """
        Get execution summary of the task.
        """
        return self.instance.get_task_summary(self.name)

    @property
    def detail(self):
        """
        Get execution details of the task.
        """
        return self.instance.get_task_detail(self.name)

    @property
    def quota(self):
        """
        Get quota json of the task.
        """
        return self.instance.get_task_quota(self.name)

    @property
    def workers(self):
        """
        Get workers of the task.
        """
        return self.instance.get_task_workers(self.name)

    def get_info(self, key, raise_empty=False):
        """
        Get associated information of the task.
        """
        return self.instance.get_task_info(self.name, key, raise_empty=raise_empty)

    def put_info(self, key, value, raise_empty=False):
        """
        Put associated information of the task.
        """
        return self.instance.put_task_info(
            self.name, key, value, raise_empty=raise_empty
        )


def format_cdata(query, semicolon=False):
    stripped_query = query.strip()
    if semicolon and not stripped_query.endswith(";"):
        stripped_query += ";"
    return "<![CDATA[%s]]>" % stripped_query


def build_execute_method(func, head_docstr):
    ext_wrapper = None
    unwrap_func = func
    if isinstance(func, classmethod):
        unwrap_func = func.__func__
        ext_wrapper = classmethod

    @six.wraps(unwrap_func)
    def wrapped(cls, *args, **kw):
        inst = unwrap_func(cls, *args, **kw)
        inst.wait_for_success()
        return inst

    wrapped.__name__ = unwrap_func.__name__.replace("run_", "execute_")

    dent_count = min(
        len(list(itertools.takewhile(lambda c: c == " ", line)))
        for line in unwrap_func.__doc__.splitlines()
        if line.strip()
    )
    _, rest_doc = textwrap.dedent(unwrap_func.__doc__).split("\n\n", 1)
    doc = "\n" + head_docstr.strip() + "\n\n" + rest_doc
    wrapped.__doc__ = "\n".join(
        " " * dent_count + line if line else "" for line in doc.splitlines()
    )

    if ext_wrapper is not None:
        wrapped = ext_wrapper(wrapped)
    return wrapped
