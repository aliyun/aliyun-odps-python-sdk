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

import calendar
import sys

from cpython.datetime cimport import_datetime
from libc.stdint cimport *

from .. import compat, types

from ..src.types_c cimport BaseRecord
from ..src.utils_c cimport CMillisecondsConverter

cdef:
    int64_t BOOL_TYPE_ID = types.boolean._type_id
    int64_t DATETIME_TYPE_ID = types.datetime._type_id
    int64_t DATE_TYPE_ID = types.date._type_id
    int64_t STRING_TYPE_ID = types.string._type_id
    int64_t FLOAT_TYPE_ID = types.float_._type_id
    int64_t DOUBLE_TYPE_ID = types.double._type_id
    int64_t BIGINT_TYPE_ID = types.bigint._type_id
    int64_t BINARY_TYPE_ID = types.binary._type_id
    int64_t TIMESTAMP_TYPE_ID = types.timestamp._type_id
    int64_t TIMESTAMP_NTZ_TYPE_ID = types.timestamp_ntz._type_id
    int64_t INTERVAL_DAY_TIME_TYPE_ID = types.interval_day_time._type_id
    int64_t INTERVAL_YEAR_MONTH_TYPE_ID = types.interval_year_month._type_id
    int64_t DECIMAL_TYPE_ID = types.Decimal._type_id
    int64_t JSON_TYPE_ID = types.Json._type_id

    bint _is_py2 = sys.version_info[0] == 2

import_datetime()


cdef class AbstractHasher:
    cdef int32_t c_hash_bigint(self, int64_t val) noexcept nogil:
        return 0

    def hash_bigint(self, int64_t val):
        return self.c_hash_bigint(val)

    cdef int32_t c_hash_float(self, float val) nogil:
        cdef int32_t *ptr = <int32_t *>&val
        return self.c_hash_bigint(ptr[0])

    def hash_float(self, float val):
        return self.c_hash_float(val)

    cdef int32_t c_hash_double(self, double val) nogil:
        cdef int64_t *ptr = <int64_t *>&val
        return self.c_hash_bigint(ptr[0])

    def hash_double(self, double val):
        return self.c_hash_double(val)

    cdef int32_t c_hash_bool(self, bint val) nogil:
        return 0

    def hash_bool(self, bint val):
        return self.c_hash_bool(val)

    cdef int32_t c_hash_string(self, char *ptr, size_t size) nogil:
        return 0

    def hash_string(self, val):
        cdef bytes bval
        if isinstance(val, unicode):
            bval = (<unicode>val).encode()
        else:
            bval = val
        return self.c_hash_string(bval, len(bval))


cdef class DefaultHasher(AbstractHasher):
    cdef int32_t c_hash_bigint(self, int64_t val) noexcept nogil:
        val = (~val) + (val << 18)
        val ^= val >> 31
        val *= <long>21
        val ^= val >> 11
        val += val << 6
        val ^= val >> 22
        return <int32_t>val

    cdef int32_t  c_hash_bool(self, bint val) nogil:
        # it is a magic number
        if val:
            return 0x172ba9c7
        else:
            return -0x3a59cb12

    cdef int32_t  c_hash_string(self, char *ptr, size_t size) nogil:
        cdef int32_t hash_val = 0
        for i in range(size):
            hash_val += ptr[i]
            hash_val += hash_val << 10
            hash_val ^= hash_val >> 6
        hash_val += hash_val << 3
        hash_val ^= hash_val >> 11
        hash_val += hash_val << 15
        return hash_val


cdef class LegacyHasher(AbstractHasher):
    cdef int32_t  c_hash_bigint(self, int64_t val) noexcept nogil:
        return (val >> 32) ^ val

    cdef int32_t  c_hash_bool(self, bint val) nogil:
        # it is a magic number
        if val:
            return 0x172ba9c7
        else:
            return -0x3a59cb12

    cdef int32_t  c_hash_string(self, char *ptr, size_t size) nogil:
        cdef int32_t hash_val = 0
        for i in range(size):
            hash_val = hash_val * 31 + ptr[i]
        return hash_val


cdef class FieldHasher:
    def __init__(self, AbstractHasher hasher):
        self._hasher = hasher

    cdef int32_t hash_object(self, object value) except? -1:
        raise NotImplementedError


cdef class BigintFieldHasher(FieldHasher):
    cdef int32_t hash_object(self, object value) except? -1:
        return self._hasher.c_hash_bigint(value)


cdef class FloatFieldHasher(FieldHasher):
    cdef int32_t hash_object(self, object value) except? -1:
        return self._hasher.c_hash_float(value)


cdef class DoubleFieldHasher(FieldHasher):
    cdef int32_t hash_object(self, object value) except? -1:
        return self._hasher.c_hash_double(value)


cdef class BoolFieldHasher(FieldHasher):
    cdef int32_t hash_object(self, object value) except? -1:
        return self._hasher.c_hash_bool(value)


cdef class StringFieldHasher(FieldHasher):
    cdef int32_t hash_object(self, object value) except? -1:
        cdef bytes bval
        if isinstance(value, unicode):
            bval = (<unicode>value).encode()
        else:
            bval = value
        return self._hasher.c_hash_string(bval, len(bval))


cdef class DateFieldHasher(FieldHasher):
    cdef int32_t hash_object(self, object value) except? -1:
        return self._hasher.c_hash_bigint(
            int(calendar.timegm(value.timetuple()))
        )


cdef class FieldHasherWithTZ(FieldHasher):
    cdef CMillisecondsConverter _mills_converter

    def __init__(
        self, AbstractHasher hasher, CMillisecondsConverter mills_converter
    ):
        super(FieldHasherWithTZ, self).__init__(hasher)
        self._mills_converter = mills_converter


cdef class DatetimeFieldHasher(FieldHasherWithTZ):
    cdef int32_t hash_object(self, object value) except? -1:
        return self._hasher.c_hash_bigint(
            self._mills_converter.to_milliseconds(value)
        )


cdef class TimestampFieldHasher(FieldHasherWithTZ):
    cdef int32_t hash_object(self, object value) except? -1:
        cdef int64_t seconds = int(
            self._mills_converter.to_milliseconds(value.to_pydatetime()) / 1000
        )
        cdef int64_t nanos = value.microsecond * 1000 + value.nanosecond
        return self._hasher.hash_bigint((seconds << 30) | nanos)


cdef class TimedeltaFieldHasher(FieldHasherWithTZ):
    cdef int32_t hash_object(self, object value) except? -1:
        cdef int64_t seconds = int(value.total_seconds())
        cdef int64_t nanos = value.microseconds * 1000 + value.nanoseconds
        return self._hasher.hash_bigint((seconds << 30) | nanos)


cdef class DecimalFieldHasher(FieldHasher):
    cdef int32_t _precision
    cdef int32_t _scale

    def __init__(
        self, AbstractHasher hasher, int32_t precision, int32_t scale
    ):
        super(DecimalFieldHasher, self).__init__(hasher)
        self._precision = precision
        self._scale = scale

    cdef int32_t hash_object(self, object value) except? -1:
        cdef:
            bytes x
            bytes num_without_exp
            int32_t exponent
            int32_t x_len
            char *ptr
            bint is_negative
            bint found_dot, found_exponent
            int32_t value_scale
            int32_t i
            object tmp_result

        x = str(value).encode()
        x_len = len(x)
        ptr = x
        is_negative = False

        if x_len > 0:
            if ptr[0] == ord("-"):
                is_negative = True
                ptr += 1
                x_len -= 1
            elif ptr[0] == ord("+"):
                is_negative = False
                ptr += 1
                x_len -= 1
        while x_len > 0 and ptr[0] == ord("0"):
            ptr += 1
            x_len -= 1

        value_scale = 0
        found_dot = found_exponent = False
        for i in range(x_len):
            c = ptr[i]
            if ord("0") <= c <= ord("9"):
                if found_dot:
                    value_scale += 1
            elif c == ord(".") and not found_dot:
                found_dot = True
            elif c in (ord("e"), ord("E")) and i + 1 < x_len:
                found_exponent = True
                exponent = int(ptr[i + 1 :])
                value_scale -= exponent
                x_len = i
                break
            else:
                raise ValueError("Invalid decimal format: " + x)

        num_without_exp = ptr[0 : x_len]
        if found_dot:
            num_without_exp = num_without_exp.replace(b".", b"")
        if not num_without_exp:
            tmp_result = 0
        else:
            if not _is_py2:
                tmp_result = int(num_without_exp)
            else:
                tmp_result = compat.long_type(num_without_exp)

            if value_scale > self._scale:
                tmp_result //= <object>int(10) ** (value_scale - self._scale)
                if num_without_exp[
                    len(num_without_exp) - (value_scale - self._scale)
                ] >= ord("5"):
                    tmp_result += 1
            elif value_scale < self._scale:
                tmp_result *= <object>int(10) ** (self._scale - value_scale)
            if is_negative:
                tmp_result *= -1

        if self._precision > 18:
            return self._hasher.c_hash_bigint(
                <int64_t>(tmp_result & 0xFFFFFFFFFFFFFFFF)
            ) + self._hasher.c_hash_bigint(<int64_t>(tmp_result >> 64))
        return self._hasher.c_hash_bigint(<int64_t>tmp_result)


cpdef AbstractHasher get_hasher(hasher_type):
    if hasher_type == "legacy":
        return LegacyHasher()
    elif hasher_type == "default":
        return DefaultHasher()
    else:
        raise ValueError("Hasher type %s not supported" % hasher_type)


def _hash_timestamp(hasher, x):
    seconds = int(x.timestamp())
    nanos = x.microsecond * 1000 + x.nanosecond
    return hasher.hash_bigint((seconds << 30) | nanos)


def _hash_timedelta(hasher, x):
    seconds = int(x.total_seconds())
    nanos = x.microsecond * 1000 + x.nanosecond
    return hasher.hash_bigint((seconds << 30) | nanos)


cdef class RecordHasher:
    def __init__(self, schema, hasher_type, hash_keys):
        cdef int type_id, i
        cdef int precision, scale
        cdef int key_count = len(hash_keys)
        cdef set hash_keys_set = set(hash_keys)

        self._schema_snapshot = schema.build_snapshot()
        self._hasher = get_hasher(hasher_type)
        self._mills_converter = CMillisecondsConverter()

        self._col_ids.reserve(key_count)
        self._idx_to_hash_fun = [None] * key_count
        for i in range(self._schema_snapshot._col_count):
            if self._schema_snapshot._columns[i].name not in hash_keys_set:
                continue

            self._col_ids.push_back(i)
            type_id = self._schema_snapshot._col_type_ids[i]
            if type_id == BIGINT_TYPE_ID:
                self._idx_to_hash_fun[i] = BigintFieldHasher(self._hasher)
            elif type_id == FLOAT_TYPE_ID:
                self._idx_to_hash_fun[i] = FloatFieldHasher(self._hasher)
            elif type_id == DOUBLE_TYPE_ID:
                self._idx_to_hash_fun[i] = DoubleFieldHasher(self._hasher)
            elif type_id == STRING_TYPE_ID or type_id == BINARY_TYPE_ID:
                self._idx_to_hash_fun[i] = StringFieldHasher(self._hasher)
            elif type_id == BOOL_TYPE_ID:
                self._idx_to_hash_fun[i] = BoolFieldHasher(self._hasher)
            elif type_id == DATE_TYPE_ID:
                self._idx_to_hash_fun[i] = DateFieldHasher(self._hasher)
            elif type_id == DATETIME_TYPE_ID:
                self._idx_to_hash_fun[i] = DatetimeFieldHasher(
                    self._hasher, self._mills_converter
                )
            elif type_id == TIMESTAMP_TYPE_ID:
                self._idx_to_hash_fun[i] = TimestampFieldHasher(
                    self._hasher, self._mills_converter
                )
            elif type_id == INTERVAL_DAY_TIME_TYPE_ID:
                self._idx_to_hash_fun[i] = TimedeltaFieldHasher(
                    self._hasher, self._mills_converter
                )
            elif type_id == DECIMAL_TYPE_ID:
                precision = (
                    self._schema_snapshot._col_types[i].precision
                    or types.Decimal._max_precision
                )
                scale = (
                    self._schema_snapshot._col_types[i].scale
                    or types.Decimal._max_scale
                )
                self._idx_to_hash_fun[i] = DecimalFieldHasher(
                    self._hasher, precision, scale
                )
            else:
                raise TypeError(
                    "Hash for type %s not supported"
                    % self._schema_snapshot._col_types[i]
                )

    cpdef int32_t hash(self, BaseRecord record):
        cdef int i
        cdef int32_t hash_sum = 0

        for i in range(self._col_ids.size()):
            if record._c_values[<int>self._col_ids[i]] is None:
                continue
            hash_sum += (<FieldHasher>self._idx_to_hash_fun[i]).hash_object(
                record._c_values[i]
            )
        return hash_sum ^ (hash_sum >> 8)


cpdef int32_t hash_value(hasher_type, data_type, value):
    """Simple hash function for test purpose"""
    cdef RecordHasher rec_hasher

    from ..models import Record
    from ..types import Column, OdpsSchema

    schema = OdpsSchema([Column("col", data_type)])
    record = Record(schema=schema, values=[value])

    rec_hasher = RecordHasher(schema, hasher_type, ["col"])
    return (<FieldHasher>rec_hasher._idx_to_hash_fun[0]).hash_object(value)
