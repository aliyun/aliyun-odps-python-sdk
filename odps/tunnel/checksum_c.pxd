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

cdef class Checksum:

    cdef uint32_t _checksum
    cdef int use_c

    cdef void c_update_bool(self, bint val) nogil
    cdef void c_update_int(self, int32_t val) nogil
    cdef void c_update_long(self, int64_t val) nogil
    cdef void c_update_float(self, float val) nogil
    cdef void c_update_double(self, double val) nogil
    cdef void c_update(self, char *ptr, size_t length) nogil
    cdef uint32_t c_getvalue(self) nogil
    cdef uint32_t c_setvalue(self, uint32_t val) nogil
    cdef void c_reset(self) nogil

    cpdef update_bool(self, bint val)
    cpdef update_int(self, int32_t val)
    cpdef update_long(self, int64_t val)
    cpdef update_float(self, float val)
    cpdef update_double(self, double val)
    cpdef update(self, bytes b)
    cpdef uint32_t getvalue(self)
    cpdef uint32_t setvalue(self, uint32_t val)
    cpdef reset(self)