# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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


cdef class CDecoder:
    cdef size_t _pos
    cdef object _stream
    cdef bytes _buffer
    cdef char* _begin
    cdef char* _end
    cdef bint _is_source_eof

    cdef bint _record_network_time
    cdef public long long _network_wall_time_ns

    cdef size_t position(self) nogil
    cdef int32_t read_field_number(self, int32_t * p_wire_type) except? -1 nogil
    cdef int32_t read_sint32(self) except? -1 nogil
    cdef uint32_t read_uint32(self) except? 0xffffffff nogil
    cdef int64_t read_sint64(self) except? -1 nogil
    cdef uint64_t read_uint64(self) except? 0xffffffff nogil
    cdef bint read_bool(self) except? False nogil
    cdef double read_double(self) except? -1.0 nogil
    cdef float read_float(self) except? -1.0 nogil
    cdef bytes read_string(self)
    cdef int _load_next_buffer(self) except -1 with gil


cdef class Decoder:
    cdef CDecoder _decoder
