#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from libc.stdint cimport *
from libc.string cimport *

import time
from datetime import datetime
from decimal import Decimal

from . import types
from . import utils
from .compat import six, decimal

cdef int64_t bigint_min = types.bigint._bounds[0]
cdef int64_t bigint_max = types.bigint._bounds[1]
cdef int string_len_max = types.string._max_length
cdef double ticks_min = types.datetime._ticks_bound[0]
cdef double ticks_max = types.datetime._ticks_bound[1]
cdef int decimal_int_len_max = 36
cdef int decimal_scale_max = 18
cdef object to_scale = decimal.Decimal(str(10 ** -decimal_scale_max))



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


cdef double _to_timestamp(object val):
    return time.mktime(val.timetuple())


cdef object _validate_datetime(object val):
    if isinstance(val, (bytes, unicode)):
        val = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')

    cdef double ts = _to_timestamp(val)
    if ticks_min <= ts <= ticks_max:
        return val
    raise ValueError('InvalidData: Datetime(%s) out of range' % val)


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


cdef dict validates = {
    types.bigint: _validate_bigint,
    types.double: _validate_double,
    types.string: _validate_string,
    types.datetime: _validate_datetime,
    types.boolean: _validate_boolean,
    types.decimal: _validate_decimal,
}


cpdef object validate_value(object val, object value_type):
    if val is None and value_type.nullable:
        return val

    validate = validates.get(value_type)
    if validate is not None:
        try:
            return validate(val)
        except TypeError:
            pass

    return types.validate_value(val, value_type)


cdef _get_record_field_by_index(object record, int i):
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


