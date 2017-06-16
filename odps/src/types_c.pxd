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
from libcpp.vector cimport vector

ctypedef object (*_VALIDATE_FUNC)(object val)

cdef class SchemaSnapshot:
    cdef int _col_count
    cdef list _col_types
    cdef vector[int] _col_type_ids
    cdef vector[int] _col_is_partition
    cdef vector[int] _col_nullable
    cdef vector[_VALIDATE_FUNC] _col_validators

    cdef object validate_value(self, int i, object val)

cdef class BaseRecord:
    cdef list _c_columns, _c_values
    cdef dict _c_name_indexes
    cdef SchemaSnapshot _c_schema_snapshot

    cpdef object get_by_name(self, object name)
    cpdef set_by_name(self, object name, object value)

    cpdef object _get(self, int i)
    cpdef _set(self, int i, object value)
    cpdef _sets(self, object values)
