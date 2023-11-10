# -*- coding: utf-8 -*-
# Copyright 1999-2022 Alibaba Group Holding Ltd.
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
from libcpp.vector cimport vector

ctypedef object (*_VALIDATE_FUNC)(object val, int64_t max_field_size)

cdef class SchemaSnapshot:
    cdef list _columns
    cdef int _col_count
    cdef int _partition_col_count
    cdef list _col_types
    cdef vector[int] _col_type_ids
    cdef vector[int] _col_is_partition
    cdef vector[int] _col_nullable
    cdef vector[_VALIDATE_FUNC] _col_validators

    cdef object validate_value(self, int i, object val, int64_t max_field_size)

cdef class BaseRecord:
    cdef list _c_columns, _c_values
    cdef dict _c_name_indexes
    cdef SchemaSnapshot _c_schema_snapshot
    cdef int64_t _max_field_size

    cpdef object get_by_name(self, object name)
    cpdef set_by_name(self, object name, object value)

    cdef size_t _get_non_partition_col_count(self)

    cpdef object _get(self, int i)
    cpdef _set(self, int i, object value)
    cpdef _sets(self, object values)
