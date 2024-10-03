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

import sys

from ..compat import six
from ..counters import Counters

PY2 = sys.version_info[0] == 2
if PY2:
    long_type = long
else:
    long_type = type("DummyLong", (object,), {})

_original_int = int
_annotated_classes = {}

__all__ = [
    "get_execution_context",
    "ExecutionContext",
    "BaseUDAF",
    "BaseUDTF",
    "int",
    "annotate",
    "get_annotation",
]


class ExecutionContext(object):
    counters = Counters()

    def get_counters(self):
        return self.counters

    def get_counters_as_json_string(self):
        return self.counters.to_json_string()


class BaseUDAF(object):
    pass


class BaseUDTF(object):
    def forward(self, *args):
        pass

    def close(self):
        pass


def get_execution_context():
    return ExecutionContext()


def int(v, silent=True):
    v = _original_int(v)
    try:
        if not PY2:
            # when in python 3, check long value by bytes conversion
            v.to_bytes(8, byteorder="little", signed=True)
        elif type(v) is long_type:
            raise OverflowError
    except OverflowError:
        if silent:
            return None
        else:
            six.raise_from(
                OverflowError("Python int too large to convert to bigint: %s" % v), None
            )
    return v


def annotate(prototype):
    def ann(clz):
        _annotated_classes[clz] = prototype
        return clz

    return ann


def get_annotation(clz):
    return _annotated_classes.get(clz)
