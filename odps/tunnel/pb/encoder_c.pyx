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

from util_c cimport *

cdef class Encoder:
    def __cinit__(self):
        self._buffer = bytearray()

    def __dealloc__(self):
        pass

    def __len__(self):
        return len(self._buffer)

    cpdef bytes tostring(self):
        return bytes(self._buffer)

    cpdef int append_tag(self, int field_num, int wire_type):
        cdef int key
        key = (field_num << 3) | wire_type
        cdef int size = set_varint64(key, self._buffer)
        return size

    cpdef int append_sint32(self, int32_t value):
        return set_signed_varint32(value, self._buffer)

    cpdef int append_uint32(self, uint32_t value):
        return set_varint32(value, self._buffer)

    cpdef int append_sint64(self, int64_t value):
        return set_signed_varint64(value, self._buffer)

    cpdef int append_uint64(self, uint64_t value):
        return set_varint64(value, self._buffer)

    cpdef int append_bool(self, bint value):
        return set_varint32(value, self._buffer)

    cpdef int append_float(self, float value):
        self._buffer += (<unsigned char *>&value)[:sizeof(float)]
        return sizeof(float)

    cpdef int append_double(self, double value):
        self._buffer += (<unsigned char *>&value)[:sizeof(double)]
        return sizeof(double)

    cpdef int append_string(self, bytes value):
        cdef int value_len = len(value)
        cdef int size = set_varint32(value_len, self._buffer)
        self._buffer += value
        return size + value_len

