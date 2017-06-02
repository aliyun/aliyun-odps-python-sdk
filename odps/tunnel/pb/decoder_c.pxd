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

from util_c cimport *

cdef class Decoder:
    cdef int _pos
    cdef object _stream

    cpdef int position(self)

    cpdef add_offset(self, int n)

    cpdef int32_t read_field_number(self)

    cpdef read_field_number_and_wire_type(self)

    cpdef int32_t read_sint32(self)

    cpdef uint32_t read_uint32(self)

    cpdef int64_t read_sint64(self)

    cpdef uint64_t read_uint64(self)

    cpdef bint read_bool(self)

    cpdef double read_double(self)

    cpdef float read_float(self)

    cpdef bytes read_string(self)