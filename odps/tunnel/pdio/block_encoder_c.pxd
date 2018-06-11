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


cdef class Encoder:
    cdef int _pos
    cdef int _buf_len
    cdef int _last_error
    cdef char *_buf_ptr

    cdef init(self, char *buf_ptr, int buf_len)

    cdef int position(self) nogil
    cdef int get_last_error(self) nogil
    cdef void set_last_error(self, int errno) nogil

    cdef int append_tag(self, int field_num, int wire_type) nogil
    cdef int append_sint32(self, int32_t value) nogil
    cdef int append_uint32(self, uint32_t value) nogil
    cdef int append_sint64(self, int64_t value) nogil
    cdef int append_uint64(self, uint64_t value) nogil
    cdef int append_bool(self, bint value) nogil
    cdef int append_float(self, float value) nogil
    cdef int append_double(self, double value) nogil
    cdef int append_string(self, const char *ptr, int value_len) nogil

    cdef int _set_varint32(self, int32_t varint) nogil
    cdef int _set_varint64(self, int64_t varint) nogil
    cdef int _set_signed_varint32(self, int32_t varint) nogil
    cdef int _set_signed_varint64(self, int64_t varint) nogil
