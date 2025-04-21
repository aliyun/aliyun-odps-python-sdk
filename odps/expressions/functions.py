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

import datetime

try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import pandas as pd
except ImportError:
    pd = None

from ..compat import datetime_utcnow, six
from ..utils import camel_to_underline

_name_to_funcs = {}


class ExprFunction(object):
    arg_count = None

    @classmethod
    def _load_name_to_funcs(cls):
        if not _name_to_funcs:
            for val in globals().values():
                if (
                    not isinstance(val, type)
                    or not issubclass(val, ExprFunction)
                    or val is ExprFunction
                ):
                    continue
                cls_name = getattr(val, "_func_name", camel_to_underline(val.__name__))
                _name_to_funcs[cls_name.lower()] = val
        return _name_to_funcs

    @classmethod
    def get_cls(cls, func_name):
        try:
            return ExprFunction._load_name_to_funcs()[func_name.lower()]
        except KeyError:
            six.raise_from(ValueError("%s function not found" % func_name), None)

    @classmethod
    def call(cls, *args):
        raise NotImplementedError

    @classmethod
    def to_str(cls, arg_strs):
        return "%s(%s)" % (cls._func_name, ", ".join(arg_strs))


_date_patterns = {
    "year": "%Y",
    "month": "%Y-%m",
    "day": "%Y-%m-%d",
    "hour": "%Y-%m-%d %H:00:00",
}


class TruncateTime(ExprFunction):
    _func_name = "trunc_time"
    arg_count = 2

    @classmethod
    def _call_single(cls, val, date_part):
        if not isinstance(val, datetime.datetime):
            val = datetime.datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
        return val.strftime(_date_patterns[date_part])

    @classmethod
    def call(cls, arg, date_part):
        assert isinstance(date_part, six.string_types)
        date_part = date_part.lower()
        if pa and isinstance(arg, (pa.Array, pa.ChunkedArray)):
            res = [cls._call_single(x, date_part) for x in arg.to_pandas()]
            return pa.array(res)
        elif pd and isinstance(arg, pd.Series):
            return arg.map(lambda x: cls._call_single(x, date_part))
        return cls._call_single(arg, date_part)


class CurrentTimestampNTZ(ExprFunction):
    _func_name = "current_timestamp_ntz"
    arg_count = 0

    @classmethod
    def call(cls):
        return datetime_utcnow()
