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

import sys
from collections import OrderedDict

from cpython.bool cimport PyBool_Check
from cpython.datetime cimport PyDate_Check, PyDateTime_Check, import_datetime
from libc.stdint cimport *
from libc.string cimport *

from datetime import date, datetime

from .. import options, types
from ..compat import DECIMAL_TYPES
from ..compat import decimal as _decimal


cdef int64_t bigint_min = types.bigint._bounds[0]
cdef int64_t bigint_max = types.bigint._bounds[1]
cdef int string_len_max = types.string._max_length
cdef object pd_na_type = types.pd_na_type
cdef bint is_py3 = sys.version_info[0] >= 3
cdef bint _has_other_decimal_type = types.Decimal._has_other_decimal_type
cdef object _decimal_type = _decimal.Decimal
cdef object _decimal_ROUND_HALF_UP = _decimal.ROUND_HALF_UP

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
    int64_t JSON_TYPE_ID = types.Json._type_id
    int64_t DECIMAL_TYPE_ID = types.Decimal._type_id
    int64_t ARRAY_TYPE_ID = types.Array._type_id
    int64_t MAP_TYPE_ID = types.Map._type_id
    int64_t STRUCT_TYPE_ID = types.Struct._type_id

import_datetime()


cdef class TypeValidator:
    cdef object validate(self, object val, int64_t max_field_size):
        pass


cdef class BigintValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        cdef int64_t i_val = val
        if bigint_min <= i_val <= bigint_max:
            return i_val
        raise ValueError("InvalidData: Bigint(%s) out of range" % val)


cdef class StringValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        cdef:
            size_t s_size
            unicode u_val

        if max_field_size == 0:
            max_field_size = string_len_max

        if type(val) is bytes or isinstance(val, bytes):
            s_size = len(<bytes> val)
            u_val = (<bytes> val).decode("utf-8")
        elif type(val) is unicode or isinstance(val, unicode):
            u_val = <unicode> val
            s_size = 4 * len(u_val)
            if s_size > max_field_size:
                # only encode when strings are long enough
                s_size = len(u_val.encode("utf-8"))
        else:
            raise TypeError("Invalid data type: expect bytes or unicode, got %s" % type(val))

        if s_size <= max_field_size:
            return u_val
        raise ValueError(
            "InvalidData: Length of string(%s) is more than %sM.'" %
            (val, max_field_size / (1024 ** 2))
        )


cdef class BinaryValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        cdef:
            size_t s_size
            bytes bytes_val

        if max_field_size == 0:
            max_field_size = string_len_max

        if type(val) is bytes or isinstance(val, bytes):
            bytes_val = <bytes> val
        elif type(val) is unicode or isinstance(val, unicode):
            bytes_val = (<unicode> val).encode("utf-8")
        else:
            raise TypeError("Invalid data type: expect bytes or unicode, got %s" % type(val))

        s_size = len(bytes_val)
        if s_size <= max_field_size:
            return bytes_val
        raise ValueError(
            "InvalidData: Length of string(%s) is more than %sM.'" %
            (val, max_field_size / (1024 ** 2)))


py_strptime = datetime.strptime


cdef class DatetimeValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        if PyDateTime_Check(val):
            return val
        if isinstance(val, (bytes, unicode)):
            return py_strptime(val, "%Y-%m-%d %H:%M:%S")
        raise TypeError("Invalid data type: expect datetime, got %s" % type(val))


cdef class DateValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        if PyDate_Check(val):
            return val
        if PyDateTime_Check(val):
            return val.date()
        if isinstance(val, (bytes, unicode)):
            return py_strptime(val, "%Y-%m-%d").date()
        raise TypeError("Invalid data type: expect date, got %s" % type(val))


pd_ts = None
pd_ts_strptime = None


cdef class TimestampValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        global pd_ts, pd_ts_strptime
        if pd_ts is None:
            try:
                import pandas as pd
                pd_ts = pd.Timestamp
                pd_ts_strptime = pd_ts.strptime
            except (ImportError, ValueError):
                raise ImportError("To use TIMESTAMP in pyodps, you need to install pandas.")

        if isinstance(val, pd_ts):
            return val
        if isinstance(val, (bytes, unicode)):
            return pd_ts_strptime(val, "%Y-%m-%d %H:%M:%S")
        raise TypeError("Invalid data type: expect timestamp, got %s" % type(val))


cdef class JsonValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        if not isinstance(val, (list, dict, unicode, bytes, float, int, long)):
            raise ValueError("Invalid data type: cannot accept %r for json type" % val)

        if is_py3 and type(val) is bytes:
            val = (<bytes> val).decode("utf-8")
        elif not is_py3 and type(val) is unicode:
            val = (<unicode> val).encode("utf-8")
        return val


cdef class FloatValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        cdef float flt_val = val
        return flt_val


cdef class DoubleValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        cdef double dbl_val = val
        return dbl_val


cdef class BoolValidator(TypeValidator):
    cdef object validate(self, object val, int64_t max_field_size):
        if PyBool_Check(val):
            return val
        raise TypeError("Invalid data type: expect bool, got %s" % type(val))


cdef class DecimalValidator(TypeValidator):
    cdef int precision, scale
    cdef object _decimal_ctx, _scale_decimal

    def __init__(self, dec_type):
        self.precision = dec_type.precision or dec_type._max_precision
        self.scale = dec_type.scale or dec_type._max_scale
        self._scale_decimal = dec_type._scale_decimal
        self._decimal_ctx = dec_type._decimal_ctx

    cdef object validate(self, object val, int64_t max_field_size):
        cdef int int_len
        cdef object scaled_val

        if (
            _has_other_decimal_type
            and not isinstance(val, _decimal_type)
            and isinstance(val, DECIMAL_TYPES)
        ):
            val = _decimal_type(str(val))
        elif type(val) is not _decimal_type and not isinstance(val, _decimal_type):
            if is_py3 and isinstance(val, bytes):
                val = (<bytes>val).decode("utf-8")
            val = _decimal_type(val)

        scaled_val = val.quantize(
            self._scale_decimal, _decimal_ROUND_HALF_UP, self._decimal_ctx
        )
        int_len = len(<str> str(scaled_val)) - self.scale - 1
        if int_len > self.precision:
            raise ValueError(
                "decimal value %s overflow, max integer digit number is %s."
                % (val, self.precision)
            )
        return val


cdef class ArrayValidator(TypeValidator):
    cdef object _value_type
    cdef TypeValidator _value_validator

    def __init__(self, array_type):
        self._value_type = array_type.value_type
        try:
            self._value_validator = _build_type_validator(
                self._value_type._type_id, self._value_type
            )
        except TypeError:
            self._value_validator = None

    cdef object validate(self, object val, int64_t max_field_size):
        if type(val) is not list:
            if not isinstance(val, list):
                raise ValueError("Array data type requires `list`, instead of %s" % val)
            val = list(val)

        cdef int idx
        cdef list results = [None] * len(<list>val)
        cdef object x
        if self._value_validator is not None:
            for idx, x in enumerate(<list>val):
                results[idx] = self._value_validator.validate(x, max_field_size)
        else:
            for idx, x in enumerate(<list>val):
                results[idx] = types.validate_value(
                    x, self._value_type, max_field_size=max_field_size
                )
        return results


cdef class MapValidator(TypeValidator):
    cdef object _key_type
    cdef TypeValidator _key_validator
    cdef object _value_type
    cdef TypeValidator _value_validator
    cdef bint _use_ordered_dict

    def __init__(self, map_type):
        self._use_ordered_dict = map_type._use_ordered_dict
        self._key_type = map_type.key_type
        self._value_type = map_type.value_type

        try:
            self._key_validator = _build_type_validator(
                self._key_type._type_id, self._key_type
            )
        except TypeError:
            self._key_validator = None

        try:
            self._value_validator = _build_type_validator(
                self._value_type._type_id, self._value_type
            )
        except TypeError:
            self._value_validator = None

    cdef inline int _validate_kv(
        self, object k, object v, dict dict_ret, object obj_ret, int64_t max_field_size
    ) except? -1:
        if self._key_validator is not None:
            k = self._key_validator.validate(k, max_field_size)
        else:
            k = types.validate_value(
                k, self._key_type, max_field_size=max_field_size
            )

        if self._value_validator is not None:
            v = self._value_validator.validate(v, max_field_size)
        else:
            v = types.validate_value(
                v, self._value_type, max_field_size=max_field_size
            )

        if not self._use_ordered_dict:
            dict_ret[k] = v
        else:
            obj_ret[k] = v
        return 0

    cdef object validate(self, object val, int64_t max_field_size):
        cdef object k, v, obj_ret
        cdef dict dict_ret

        if not isinstance(val, dict):
            raise ValueError("Map data type requires `dict`, instead of %s" % val)

        if self._use_ordered_dict:
            dict_ret = None
            obj_ret = OrderedDict()
        else:
            dict_ret = dict()
            obj_ret = dict_ret

        for k, v in val.items():
            self._validate_kv(k, v, dict_ret, obj_ret, max_field_size)
        return obj_ret


cdef class StructValidator(TypeValidator):
    cdef dict _attr_to_validator
    cdef list _validators, _type_list
    cdef object _field_types, _namedtuple_type
    cdef bint _struct_as_dict

    def __init__(self, struct_type):
        cdef int idx
        cdef object validator

        self._field_types = struct_type.field_types
        self._namedtuple_type = struct_type.namedtuple_type
        self._struct_as_dict = struct_type._struct_as_dict

        self._attr_to_validator = dict()
        self._validators = [None] * len(self._field_types)
        self._type_list = list(self._field_types.values())

        for idx, (k, v) in enumerate(self._field_types.items()):
            try:
                validator = _build_type_validator(v._type_id, v)
            except TypeError:
                validator = None
            self._validators[idx] = validator
            self._attr_to_validator[k] = validator

    cdef inline object _validate_by_key(
        self, object key, object val, int64_t max_field_size
    ):
        cdef TypeValidator field_validator

        field_validator = self._attr_to_validator[key]
        if field_validator is not None:
            return field_validator.validate(val, max_field_size)
        else:
            return types.validate_value(
                val, self._field_types[key], max_field_size=max_field_size
            )

    cdef inline object _validate_by_index(
        self, int index, object val, int64_t max_field_size
    ):
        cdef TypeValidator field_validator

        field_validator = self._validators[index]
        if field_validator is not None:
            return field_validator.validate(val, max_field_size)
        else:
            return types.validate_value(
                val, self._type_list[index], max_field_size=max_field_size
            )

    cdef object validate(self, object val, int64_t max_field_size):
        cdef list ret_list
        cdef int idx
        cdef object ret

        if self._struct_as_dict:
            if isinstance(val, tuple):
                fields = getattr(val, "_fields", None) or self._field_types.keys()
                val = OrderedDict(zip(fields, val))
            if isinstance(val, dict):
                ret = OrderedDict()
                for k, v in val.items():
                    ret[k] = self._validate_by_key(k, v, max_field_size)
                return ret
        else:
            if isinstance(val, tuple):
                if type(val) is tuple:
                    ret_list = [None] * len(<tuple>val)
                    for idx, v in enumerate(<tuple>val):
                        ret_list[idx] = self._validate_by_index(idx, v, max_field_size)
                else:
                    ret_list = [None] * len(val)
                    for idx, v in enumerate(val):
                        ret_list[idx] = self._validate_by_index(idx, v, max_field_size)
                return self._namedtuple_type(*tuple(ret_list))
            elif isinstance(val, dict):
                ret_list = [None] * len(val)
                for idx, k in enumerate(self._field_types.keys()):
                    ret_list[idx] = self._validate_by_key(k, val.get(k), max_field_size)
                return self._namedtuple_type(*tuple(ret_list))
        raise ValueError(
            "Struct data type requires `tuple` or `dict`, instead of %s" % type(val)
        )


cdef object _build_type_validator(int type_id, object data_type):
    if type_id == BIGINT_TYPE_ID:
        return BigintValidator()
    elif type_id == FLOAT_TYPE_ID:
        return FloatValidator()
    elif type_id == DOUBLE_TYPE_ID:
        return DoubleValidator()
    elif type_id == STRING_TYPE_ID:
        if options.tunnel.string_as_binary:
            return BinaryValidator()
        else:
            return StringValidator()
    elif type_id == DATETIME_TYPE_ID:
        return DatetimeValidator()
    elif type_id == DATE_TYPE_ID:
        return DateValidator()
    elif type_id == BOOL_TYPE_ID:
        return BoolValidator()
    elif type_id == BINARY_TYPE_ID:
        return BinaryValidator()
    elif type_id == TIMESTAMP_TYPE_ID:
        return TimestampValidator()
    elif type_id == JSON_TYPE_ID:
        return JsonValidator()
    elif type_id == DECIMAL_TYPE_ID:
        return DecimalValidator(data_type)
    elif type_id == ARRAY_TYPE_ID:
        return ArrayValidator(data_type)
    elif type_id == MAP_TYPE_ID:
        return MapValidator(data_type)
    elif type_id == STRUCT_TYPE_ID:
        return StructValidator(data_type)
    return None


cdef class SchemaSnapshot:
    def __cinit__(self, schema):
        cdef int i
        self._columns = schema.columns
        self._col_count = len(schema)
        self._col_types = [c.type for c in schema.columns]
        self._col_type_ids = [
            t._type_id if t._type_id is not None else -1 for t in self._col_types
        ]
        self._col_nullable = [t.nullable for t in self._col_types]
        if isinstance(schema, types.OdpsSchema):
            self._col_is_partition = [
                1 if schema.is_partition(c) else 0
                for c in schema.columns
            ]
            self._partition_col_count = sum(self._col_is_partition)
        else:
            self._col_is_partition = [0] * self._col_count
            self._partition_col_count = 0

        import_datetime()

        self._col_validators = [None] * self._col_count
        for i in range(self._col_count):
            self._col_validators[i] = _build_type_validator(
                self._col_type_ids[i], self._col_types[i]
            )

    cdef object validate_value(self, int i, object val, int64_t max_field_size):
        cdef TypeValidator validator = self._col_validators[i]
        if val is None and self._col_nullable[i]:
            return val
        if validator is not None:
            try:
                return validator.validate(val, max_field_size)
            except TypeError:
                pass
        return types.validate_value(
            val, self._col_types[i], max_field_size=max_field_size
        )


cdef class BaseRecord:
    def __cinit__(self, columns=None, schema=None, values=None, max_field_size=None):
        self._c_schema_snapshot = getattr(schema, "_snapshot", None)
        if columns is not None:
            self._c_columns = columns
            self._c_name_indexes = {
                col.name: i for i, col in enumerate(self._c_columns)
            }
        else:
            self._c_columns = (
                schema.columns
                if self._c_schema_snapshot is None
                else self._c_schema_snapshot._columns
            )
            self._c_name_indexes = schema._name_indexes

        self._max_field_size = max_field_size or 0

        if self._c_columns is None:
            raise ValueError("Either columns or schema should not be provided")

        self._c_values = [None] * len(self._c_columns)
        if values is not None:
            self._sets(values)

    def __reduce__(self):
        return type(self), (self._c_columns, None, self._c_values)

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
        return "c"

    cdef size_t _get_non_partition_col_count(self):
        if self._c_schema_snapshot is not None:
            return (
                self._c_schema_snapshot._col_count
                - self._c_schema_snapshot._partition_col_count
            )
        else:
            return len(
                [col for col in self._c_columns if not isinstance(col, types.Partition)]
            )

    cpdef object get_by_name(self, object name):
        cdef int idx = self._c_name_indexes[name]
        return self._c_values[idx]

    cpdef set_by_name(self, object name, object value):
        cdef int idx = self._c_name_indexes[name]
        self._set(idx, value)

    cpdef _set(self, int i, object value):
        cdef object t
        if self._c_schema_snapshot is not None:
            value = self._c_schema_snapshot.validate_value(i, value, self._max_field_size)
        else:
            value = types.validate_value(
                value, self._c_columns[i].type, max_field_size=self._max_field_size
            )

        self._c_values[i] = value

    cpdef object _get(self, int i):
        return self._c_values[i]

    get = _get  # to keep compatible
    set = _set  # to keep compatible

    cpdef _sets(self, object values):
        cdef int i
        cdef object value
        cdef size_t n_values

        if type(values) is list:
            n_values = len(<list>values)
        else:
            n_values = len(values)

        if (
            n_values != len(self._c_columns) and
            n_values != self._get_non_partition_col_count()
        ):
            raise ValueError(
                "The values set to records are against the schema, "
                "expect len %s, got len %s" % (len(self._c_columns), n_values)
            )

        if type(values) is list:
            for i, value in enumerate(<list>values):
                self._set(i, value)
        else:
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
        if item == "_name_indexes":
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
        if self._c_schema_snapshot is not None:
            return self._c_schema_snapshot._col_count
        return len(self._c_columns)

    def get_columns_count(self):  # compatible
        return self.n_columns
