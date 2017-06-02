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

import struct
from wire_format import TAG_TYPE_BITS as PY_TAG_TYPE_BITS, _TAG_TYPE_MASK as _PY_TAG_TYPE_MASK

cdef:
    int TAG_TYPE_BITS = PY_TAG_TYPE_BITS
    int _TAG_TYPE_MASK = _PY_TAG_TYPE_MASK

cdef class Decoder:

    def __cinit__(self, stream):
        self._pos = 0
        self._stream = stream

    def __dealloc__(self):
        pass

    def read(self, n):
        return self._stream.read(n)

    cpdef add_offset(self, int n):
        self._pos += n

    cpdef int position(self):
        return self._pos

    cpdef int32_t read_field_number(self):
        cdef int32_t tag_and_type
        tag_and_type = self.read_uint32()
        return tag_and_type >> TAG_TYPE_BITS

    cpdef read_field_number_and_wire_type(self):
        cdef int32_t tag_and_type
        tag_and_type = self.read_uint32()
        return (tag_and_type >> TAG_TYPE_BITS), (tag_and_type & _TAG_TYPE_MASK)

    def __len__(self):
        return self._pos

    cpdef int32_t read_sint32(self):
        return get_signed_varint32(self)

    cpdef uint32_t read_uint32(self):
        return get_varint32(self)

    cpdef int64_t read_sint64(self):
        return get_signed_varint64(self)

    cpdef uint64_t read_uint64(self):
        return get_varint64(self)

    cpdef bint read_bool(self):
        return get_varint32(self)

    cpdef double read_double(self):
        cdef char * read_bytes
        cdef double retval

        cdef bytes input_bytes = self._stream.read(sizeof(double))

        read_bytes = input_bytes
        self._pos += sizeof(double)
        memcpy(&retval, read_bytes, sizeof(double))
        return retval

    cpdef float read_float(self):
        cdef char * read_bytes
        cdef float retval

        cdef bytes input_bytes = self._stream.read(sizeof(float))

        read_bytes = input_bytes
        self._pos += sizeof(float)
        memcpy(&retval, read_bytes, sizeof(float))
        return retval

    cpdef bytes read_string(self):
        cdef int size
        cdef bytes read_bytes

        size = self.read_uint32()
        read_bytes = self._stream.read(size)
        self._pos += size
        return read_bytes