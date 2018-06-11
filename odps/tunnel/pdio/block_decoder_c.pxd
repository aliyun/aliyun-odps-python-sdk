# -*- coding: utf-8 -*-
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
from libcpp.string cimport string


cdef struct FieldParam:
    int field_number
    int wire_type


cdef class Decoder:
    cdef int _pos
    cdef int _buf_len
    cdef int _last_error
    cdef char *_buf_ptr

    cdef init(self, char *buf_ptr, int buf_len)

    cdef int get_last_error(self) nogil
    cdef void set_last_error(self, int errno) nogil

    cdef int position(self) nogil
    cdef void add_offset(self, int n) nogil
    cdef int32_t read_field_number(self) nogil
    cdef FieldParam read_field_number_and_wire_type(self) nogil
    cdef int32_t read_sint32(self) nogil
    cdef uint32_t read_uint32(self) nogil
    cdef int64_t read_sint64(self) nogil
    cdef uint64_t read_uint64(self) nogil
    cdef bint read_bool(self) nogil
    cdef double read_double(self) nogil
    cdef float read_float(self) nogil
    cdef string read_string(self) nogil

    cdef int _read_input_byte(self) nogil
    cdef int32_t _get_varint32(self) nogil
    cdef int64_t _get_varint64(self) nogil
    cdef int32_t _get_signed_varint32(self) nogil
    cdef int64_t _get_signed_varint64(self) nogil
