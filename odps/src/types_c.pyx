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
from cpython.bool cimport PyBool_Check
from cpython.datetime cimport import_datetime, PyDateTime_Check

from datetime import datetime
from decimal import Decimal

from .. import types, options, utils
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


cdef object _validate_bigint(object val):
    cdef int64_t i_val = val
    if bigint_min <= i_val <= bigint_max:
        return i_val
    raise ValueError('InvalidData: Bigint(%s) out of range' % val)


cdef object _validate_string(object val):
    cdef:
        bytes bytes_val
        unicode u_val
        size_t s_size

    if isinstance(val, bytes):
        bytes_val = val
        u_val = bytes_val.decode('utf-8')
    elif isinstance(val, unicode):
        u_val = val
        bytes_val = val.encode('utf-8')
    else:
        raise TypeError("Invalid data type: expect bytes or unicode, got %s" % type(val))

    s_size = len(bytes_val)
    if s_size <= string_len_max:
        return u_val
    raise ValueError(
        "InvalidData: Length of string(%s) is more than %sM.'" %
        (val, string_len_max / (1024 ** 2)))


cdef object _validate_binary(object val):
    cdef:
        bytes bytes_val
        unicode u_val
        size_t s_size

    if isinstance(val, bytes):
        bytes_val = val
    elif isinstance(val, unicode):
        u_val = val
        bytes_val = val.encode('utf-8')
    else:
        raise TypeError("Invalid data type: expect bytes or unicode, got %s" % type(val))

    s_size = len(bytes_val)
    if s_size <= string_len_max:
        return bytes_val
    raise ValueError(
        "InvalidData: Length of string(%s) is more than %sM.'" %
        (val, string_len_max / (1024 ** 2)))


py_strptime = datetime.strptime

cdef object _validate_datetime(object val):
    if PyDateTime_Check(val):
        return val
    if isinstance(val, (bytes, unicode)):
        return py_strptime(val, '%Y-%m-%d %H:%M:%S')
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

cdef object _validate_double(object val):
    cdef double dbl_val = val
    return dbl_val


cdef object _validate_boolean(object val):
    if PyBool_Check(val):
        return val
    raise TypeError("Invalid data type: expect bool, got %s" % type(val))


cdef object validate_value(object val, object value_type):
    if val is None and value_type.nullable:
        return val

    cdef int64_t type_id = value_type._type_id

    try:
        if type_id == BIGINT_TYPE_ID:
            return _validate_bigint(val)
        elif type_id == DOUBLE_TYPE_ID:
            return _validate_double(val)
        elif type_id == STRING_TYPE_ID:
            if options.tunnel.string_as_binary:
                return _validate_binary(val)
            else:
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


cdef class SchemaSnapshot:
    def __cinit__(self, schema):
        cdef int type_id, i
        self._col_count = len(schema)
        self._col_types = [c.type for c in schema.columns]
        self._col_type_ids = [t._type_id if t._type_id is not None else -1 for t in self._col_types]
        self._col_nullable = [t.nullable for t in self._col_types]
        if isinstance(schema, types.OdpsSchema):
            self._col_is_partition = [1 if schema.is_partition(c) else 0
                                      for c in schema.columns]
        else:
            self._col_is_partition = [0] * self._col_count

        self._col_validators.resize(self._col_count)
        for i in range(self._col_count):
            type_id = self._col_type_ids[i]
            if type_id == BIGINT_TYPE_ID:
                self._col_validators[i] = _validate_bigint
            elif type_id == DOUBLE_TYPE_ID:
                self._col_validators[i] = _validate_double
            elif type_id == STRING_TYPE_ID:
                if options.tunnel.string_as_binary:
                    self._col_validators[i] = _validate_binary
                else:
                    self._col_validators[i] = _validate_string
            elif type_id == DATETIME_TYPE_ID:
                self._col_validators[i] = _validate_datetime
            elif type_id == BOOL_TYPE_ID:
                self._col_validators[i] = _validate_boolean
            elif type_id == DECIMAL_TYPE_ID:
                self._col_validators[i] = _validate_decimal
            else:
                self._col_validators[i] = NULL

    cdef object validate_value(self, int i, object val):
        cdef _VALIDATE_FUNC vfun = self._col_validators[i]
        if val is None and self._col_nullable[i]:
            return val
        if vfun != NULL:
            try:
                return vfun(val)
            except TypeError:
                pass
        return types.validate_value(val, self._col_types[i])


cdef class BaseRecord:
    def __cinit__(self, columns=None, schema=None, values=None):
        self._c_schema_snapshot = getattr(schema, '_snapshot', None)
        if columns is not None:
            self._c_columns = columns
            self._c_name_indexes = dict((col.name, i) for i, col in enumerate(self._columns))
        else:
            self._c_columns = schema.columns
            self._c_name_indexes = schema._name_indexes

        if self._c_columns is None:
            raise ValueError('Either columns or schema should not be provided')

        self._c_values = [None, ] * len(self._c_columns)
        if values is not None:
            self._sets(values)

    @property
    def _columns(self):
        return self._c_columns

    @_columns.setter
    def _columns(self, value):
        self._c_columns = value

    @property
    def _values(self):
        return self._c_values

    @_values.setter
    def _values(self, value):
        self._c_values = value

    @property
    def _name_indexes(self):
        return self._c_name_indexes

    @_name_indexes.setter
    def _name_indexes(self, value):
        self._c_name_indexes = value

    def _mode(self):
        return 'c'

    def _exclude_partition_columns(self):
        return [col for col in self._c_columns if not isinstance(col, types.Partition)]

    cpdef object get_by_name(self, object name):
        cdef int idx = self._c_name_indexes[name]
        return self._c_values[idx]

    cpdef set_by_name(self, object name, object value):
        cdef int idx = self._c_name_indexes[name]
        self._set(idx, value)

    cpdef _set(self, int i, object value):
        cdef object t
        if self._c_schema_snapshot is not None:
            value = self._c_schema_snapshot.validate_value(i, value)
        else:
            t = self._c_columns[i].type
            try:
                value = validate_value(value, t)
            except TypeError:
                value = types.validate_value(value, t)

        self._c_values[i] = value

    cpdef object _get(self, int i):
        return self._c_values[i]

    get = _get  # to keep compatible
    set = _set  # to keep compatible

    cpdef _sets(self, object values):
        cdef int i
        cdef object value

        if len(values) != len(self._c_columns) and \
                        len(values) != len(self._exclude_partition_columns()):
            raise ValueError('The values set to records are against the schema, '
                             'expect len %s, got len %s' % (len(self._c_columns), len(values)))
        for i, value in enumerate(values):
            self._set(i, value)

    def __getitem__(self, item):
        if isinstance(item, (bytes, unicode)):
            return self.get_by_name(item)
        elif isinstance(item, (list, tuple)):
            return [self[it] for it in item]
        return self._values[item]

    def __setitem__(self, key, value):
        if isinstance(key, (bytes, unicode)):
            self.set_by_name(key, value)
        else:
            self._set(key, value)

    def __getattr__(self, item):
        if item == '_name_indexes':
            return self._c_name_indexes
        if item in self._c_name_indexes:
            i = self._c_name_indexes[item]
            return self._c_values[i]
        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        cdef int i
        if key in self._c_name_indexes:
            i = self._c_name_indexes[key]
            self._set(i, value)
        else:
            object.__setattr__(self, key, value)

    def __len__(self):
        return len(self._c_columns)

    def __contains__(self, item):
        return item in self._c_name_indexes

    def __iter__(self):
        cdef int i
        cdef object col
        for i, col in enumerate(self._c_columns):
            yield (col.name, self._get(i))

    @property
    def values(self):
        return self._c_values

    @property
    def n_columns(self):
        return len(self._c_columns)

    def get_columns_count(self):  # compatible
        return self.n_columns
