# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from libc.stdint cimport *
from libc.string cimport *
from cpython.datetime cimport import_datetime, PyDateTime_Check

from datetime import datetime
from decimal import Decimal

from .. import types
from .. import utils
from ..compat import decimal

cdef int64_t bigint_min = types.bigint._bounds[0]
cdef int64_t bigint_max = types.bigint._bounds[1]
cdef int string_len_max = types.string._max_length
cdef int decimal_int_len_max = 36
cdef int decimal_scale_max = 18
cdef object to_scale = decimal.Decimal(str(10 ** -decimal_scale_max))

cdef:
    int64_t BOOL_TYPE_ID = types.boolean._type_id
    int64_t DATETIME_TYPE_ID = types.datetime._type_id
    int64_t STRING_TYPE_ID = types.string._type_id
    int64_t DOUBLE_TYPE_ID = types.double._type_id
    int64_t BIGINT_TYPE_ID = types.bigint._type_id
    int64_t DECIMAL_TYPE_ID = types.decimal._type_id


def init_module():
    import_datetime()


init_module()


cdef int64_t _validate_bigint(int64_t val):
    if bigint_min <= val <= bigint_max:
        return val
    raise ValueError('InvalidData: Bigint(%s) out of range' % val)


cdef unicode _validate_string(bytes val):
    if strlen(val) <= string_len_max:
        return val.decode('utf-8')
    raise ValueError(
        "InvalidData: Length of string(%s) is more than %sM.'" %
        (val, string_len_max / (1024 ** 2)))


cdef object _validate_datetime(object val):
    if PyDateTime_Check(val):
        return val
    if isinstance(val, (bytes, unicode)):
        return datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
    raise TypeError("Invalid data type: expect datetime, got %s" % type(val))


cdef object _validate_decimal(object val):
    cdef:
        object scaled_val
        int int_len

    if not isinstance(val, decimal.Decimal):
        val = decimal.Decimal(utils.to_str(val))

    scaled_val = val.quantize(to_scale, decimal.ROUND_HALF_UP)
    int_len = len(str(scaled_val)) - decimal_scale_max - 1
    if int_len > decimal_int_len_max:
        raise ValueError(
            'decimal value %s overflow, max integer digit number is %s.' %
            (val, decimal_int_len_max))

    return Decimal(str(val))

cdef double _validate_double(double val):
    return val


cdef bint _validate_boolean(bint val):
    return val


cpdef object validate_value(object val, object value_type):
    if val is None and value_type.nullable:
        return val

    cdef int64_t type_id = value_type._type_id

    try:
        if type_id == BIGINT_TYPE_ID:
            return _validate_bigint(val)
        elif type_id == DOUBLE_TYPE_ID:
            return _validate_double(val)
        elif type_id == STRING_TYPE_ID:
            return _validate_string(val)
        elif type_id == DATETIME_TYPE_ID:
            return _validate_datetime(val)
        elif type_id == BOOL_TYPE_ID:
            return _validate_boolean(val)
        elif type_id == DECIMAL_TYPE_ID:
            return _validate_decimal(val)
    except TypeError:
        pass

    return types.validate_value(val, value_type)


cdef _get_record_field_by_index(object record, int i):
    cpdef list values = record._values
    return values[i]


cdef _get_record_field_by_slice(object record, slice i):
    cpdef list values = record._values
    return values[i]


cdef _get_record_field_by_name(object record, object name):
    cdef dict name_indexes = record._name_indexes
    cdef int i
    if name in name_indexes:
        i = name_indexes[name]
        return _get_record_field_by_index(record, i)
    raise AttributeError('Record does not have field: %s' % name)

cdef _get_record_field(object record, object item):
    try:
        return _get_record_field_by_index(record, item)
    except TypeError:
        if isinstance(item, slice):
            return _get_record_field_by_slice(record, item)
        return _get_record_field_by_name(record, item)


cdef _getitem(object record, object item):
    try:
        return _get_record_field(record, item)
    except (TypeError, AttributeError) as e:
        if isinstance(item, (list, tuple)):
            return [_getitem(record, it) for it in item]
        raise e


class Record(types.Record):

    def _mode(self):
        return 'c'

    def _set(self, int i, object value):
        cdef object t = self._columns[i].type
        cdef list values = self._values
        try:
            value = validate_value(value, t)
        except TypeError:
            value = types.validate_value(value, t)

        values[i] = value

    set = _set

    def __getitem__(self, item):
        return _getitem(self, item)
