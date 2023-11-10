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

import ctypes
import calendar
import struct

from .. import types
from ..compat import PY27
from ..utils import to_binary, to_milliseconds

_int32_struct = struct.Struct("<l")
_int64_struct = struct.Struct("<q")
_float_struct = struct.Struct("<f")
_double_struct = struct.Struct("<d")

_ord_code = ord if PY27 else (lambda x: x)


class AbstractHasher(object):
    def hash_bigint(self, val):
        raise NotImplementedError

    def hash_float(self, val):
        raise NotImplementedError

    def hash_double(self, val):
        raise NotImplementedError

    def hash_bool(self, val):
        raise NotImplementedError

    def hash_string(self, val):
        raise NotImplementedError


class DefaultHasher(AbstractHasher):
    def hash_bigint(self, val):
        val = (~val) + ctypes.c_int64(val << 18).value
        val ^= (val >> 31)
        val = ctypes.c_int64(val * 21).value
        val ^= (val >> 11)
        val += ctypes.c_int64(val << 6).value
        val ^= (val >> 22)
        return ctypes.c_int32(val).value

    def hash_float(self, val):
        return self.hash_bigint(_int32_struct.unpack(_float_struct.pack(val))[0])

    def hash_double(self, val):
        return self.hash_bigint(_int64_struct.unpack(_double_struct.pack(val))[0])

    def hash_bool(self, val):
        # it is a magic number
        if val:
            return 0x172ba9c7
        else:
            return -0x3a59cb12

    def hash_string(self, val):
        val = to_binary(val)
        hash_val = 0
        for ch in val:
            hash_val += _ord_code(ch)
            hash_val += ctypes.c_int32(hash_val << 10).value
            hash_val ^= hash_val >> 6
        hash_val += ctypes.c_int32(hash_val << 3).value
        hash_val ^= hash_val >> 11
        hash_val += ctypes.c_int32(hash_val << 15).value
        return ctypes.c_int32(hash_val).value


class LegacyHasher(AbstractHasher):
    def hash_bigint(self, val):
        return ctypes.c_int32((val >> 32) ^ val).value

    def hash_float(self, val):
        return self.hash_bigint(_int32_struct.unpack(_float_struct.pack(val))[0])

    def hash_double(self, val):
        return self.hash_bigint(_int64_struct.unpack(_double_struct.pack(val))[0])

    def hash_bool(self, val):
        # it is a magic number
        if val:
            return 0x172ba9c7
        else:
            return -0x3a59cb12

    def hash_string(self, val):
        val = to_binary(val)
        hash_val = 0
        for ch in val:
            hash_val = ctypes.c_int32(hash_val * 31 + _ord_code(ch)).value
        return hash_val


def get_hasher(hasher_type):
    if hasher_type == "legacy":
        return LegacyHasher()
    elif hasher_type == "default":
        return DefaultHasher()
    else:
        raise ValueError("Hasher type %s not supported" % hasher_type)


def _hash_date(hasher, x):
    return hasher.hash_bigint(int(calendar.timegm(x.timetuple())))


def _hash_datetime(hasher, x):
    return hasher.hash_bigint(int(to_milliseconds(x)))


def _hash_timestamp(hasher, x):
    seconds = int(to_milliseconds(x.to_pydatetime()) / 1000)
    nanos = x.microsecond * 1000 + x.nanosecond
    return hasher.hash_bigint((seconds << 30) | nanos)


def _hash_timedelta(hasher, x):
    seconds = int(x.total_seconds())
    nanos = x.microseconds * 1000 + x.nanoseconds
    return hasher.hash_bigint((seconds << 30) | nanos)


_type_to_hash_fun = {
    types.tinyint: lambda hasher, x: hasher.hash_bigint(x),
    types.smallint: lambda hasher, x: hasher.hash_bigint(x),
    types.int_: lambda hasher, x: hasher.hash_bigint(x),
    types.bigint: lambda hasher, x: hasher.hash_bigint(x),
    types.boolean: lambda hasher, x: hasher.hash_bool(x),
    types.float_: lambda hasher, x: hasher.hash_float(x),
    types.double: lambda hasher, x: hasher.hash_double(x),
    types.date: _hash_date,
    types.datetime: _hash_datetime,
    types.timestamp: _hash_timestamp,
    types.interval_day_time: _hash_timedelta,
    types.binary: lambda hasher, x: hasher.hash_string(x),
    types.string: lambda hasher, x: hasher.hash_string(x),
}


class RecordHasher(object):
    def __init__(self, schema, hasher_type, hash_keys):
        self._schema = schema
        self._hasher = get_hasher(hasher_type)
        self._hash_keys = hash_keys

        self._column_hash_appenders = []
        for col_name in hash_keys:
            col = self._schema.get_column(col_name)
            if col.type in _type_to_hash_fun:
                self._column_hash_appenders.append(
                    _type_to_hash_fun[col.type]
                )
            elif isinstance(col.type, (types.Char, types.Varchar)):
                self._column_hash_appenders.append(
                    _type_to_hash_fun[types.string]
                )
            else:
                raise TypeError("Hash for type %s not supported" % col.type)

    def hash(self, record):
        hash_sum = 0
        for idx, key in enumerate(self._hash_keys):
            if record[key] is None:
                continue
            hash_sum += self._column_hash_appenders[idx](self._hasher, record[key])
        hash_sum = ctypes.c_int32(hash_sum).value
        return hash_sum ^ (hash_sum >> 8)


def hash_value(hasher_type, data_type, value):
    """Simple value hash function for test purpose"""
    hasher = get_hasher(hasher_type)
    data_type = types.validate_data_type(data_type)
    return _type_to_hash_fun[data_type](hasher, value)
