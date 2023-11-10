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
from libc.string cimport *
from ...src.stringstream cimport stringstream


cdef class CEncoder:
    cdef stringstream *_buffer

    cdef size_t position(self) nogil
    cpdef bytes tostring(self)
    cdef int append_tag(self, int field_num, int wire_type) except + nogil
    cdef int append_sint32(self, int32_t value) except + nogil
    cdef int append_uint32(self, uint32_t value) except + nogil
    cdef int append_sint64(self, int64_t value) except + nogil
    cdef int append_uint64(self, uint64_t value) except + nogil
    cdef int append_bool(self, bint value) except + nogil
    cdef int append_double(self, double value) except + nogil
    cdef int append_float(self, float value) except + nogil
    cdef int append_string(self, const char *ptr, size_t value_len) except + nogil


cdef class Encoder:
    cdef CEncoder _encoder
