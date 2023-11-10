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

include "util_c.pxi"

from libc.stdint cimport *
from libc.string cimport *
from libcpp.string cimport string
from ...src.stringstream cimport stringstream


cdef class CEncoder:
    def __cinit__(self, size_t reserve):
        cdef string* prealloc_string = new string()
        prealloc_string.reserve(reserve)
        self._buffer = new stringstream(prealloc_string[0])
        del prealloc_string

    def __dealloc__(self):
        del self._buffer

    cdef size_t position(self) nogil:
        return self._buffer.tellp()

    cpdef bytes tostring(self):
        return bytes(self._buffer.to_string())

    cdef int append_tag(self, int field_num, int wire_type) except + nogil:
        cdef int key
        key = (field_num << 3) | wire_type
        cdef int size = set_varint64(key, self._buffer[0])
        return size

    cdef int append_sint32(self, int32_t value) except + nogil:
        return set_signed_varint32(value, self._buffer[0])

    cdef int append_uint32(self, uint32_t value) except + nogil:
        return set_varint32(value, self._buffer[0])

    cdef int append_sint64(self, int64_t value) except + nogil:
        return set_signed_varint64(value, self._buffer[0])

    cdef int append_uint64(self, uint64_t value) except + nogil:
        return set_varint64(value, self._buffer[0])

    cdef int append_bool(self, bint value) except + nogil:
        return set_varint32(value, self._buffer[0])

    cdef int append_float(self, float value) except + nogil:
        self._buffer.write(<const char *>&value, sizeof(float))
        return sizeof(float)

    cdef int append_double(self, double value) except + nogil:
        self._buffer.write(<const char*>&value, sizeof(double))
        return sizeof(double)

    cdef int append_string(self, const char *ptr, size_t value_len) except + nogil:
        cdef int size = set_varint32(value_len, self._buffer[0])
        self._buffer.write(ptr, value_len)
        return size + value_len


cdef class Encoder:
    def __init__(self):
        self._encoder = CEncoder(4096)

    def __len__(self):
        return self._encoder.position()

    def position(self):
        return self._encoder.position()

    def tostring(self):
        return self._encoder.tostring()

    def append_tag(self, int field_num, int wire_type):
        return self._encoder.append_tag(field_num, wire_type)

    def append_sint32(self, int32_t value):
        return self._encoder.append_sint32(value)

    def append_uint32(self, uint32_t value):
        return self._encoder.append_uint32(value)

    def append_sint64(self, int64_t value):
        return self._encoder.append_sint64(value)

    def append_uint64(self, uint64_t value):
        return self._encoder.append_uint64(value)

    def append_bool(self, bint value):
        return self._encoder.append_bool(value)

    def append_float(self, float value):
        self._encoder.append_float(value)

    def append_double(self, double value):
        self._encoder.append_double(value)

    def append_string(self, bytes value):
        return self._encoder.append_string(value, len(value))
