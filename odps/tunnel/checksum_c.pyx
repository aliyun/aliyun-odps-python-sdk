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


cdef extern from "../src/crc32.c":
    uint32_t crc32(uint32_t crc, const void *buf, size_t length) nogil
    uint32_t crc32c(uint32_t crc, const void *buf, size_t length) nogil


cdef class Checksum:

    def __cinit__(self, method='crc32c'):
        self._checksum = 0
        if method == 'crc32c':
            self.use_c = 1
        else:
            self.use_c = 0

    cdef void c_update_bool(self, bint val) nogil:
        cdef char retval
        retval = 1 if val else 0
        self.c_update(<char *>&retval, 1)

    cpdef update_bool(self, bint val):
        self.c_update_bool(val)

    cdef void c_update_int(self, int32_t val) nogil:
        self.c_update(<char *>&val, sizeof(int32_t))

    cpdef update_int(self, int32_t val):
        self.c_update(<char *>&val, sizeof(int32_t))

    cdef void c_update_long(self, int64_t val) nogil:
        self.c_update(<char *>&val, sizeof(int64_t))

    cpdef update_long(self, int64_t val):
        self.c_update(<char *>&val, sizeof(int64_t))

    cdef void c_update_float(self, float val) nogil:
        self.c_update(<char *>&val, sizeof(float))

    cpdef update_float(self, float val):
        self.c_update(<char *>&val, sizeof(float))

    cdef void c_update_double(self, double val) nogil:
        self.c_update(<char *>&val, sizeof(double))

    cpdef update_double(self, double val):
        self.c_update(<char *>&val, sizeof(double))

    cdef void c_update(self, const char *ptr, size_t length) nogil:
        if self.use_c:
            self._checksum = crc32c(self._checksum, ptr, length)
        else:
            self._checksum = crc32(self._checksum, ptr, length)

    cpdef update(self, bytes b):
        buf = bytearray(b)
        if self.use_c:
            self._checksum = crc32c(self._checksum, <const void*>(<char *>buf), len(buf))
        else:
            self._checksum = crc32(self._checksum, <const void*>(<char *>buf), len(buf))

    cdef uint32_t c_getvalue(self) nogil:
        return self._checksum

    cpdef uint32_t getvalue(self):
        return self._checksum

    cdef uint32_t c_setvalue(self, uint32_t val) nogil:
        self._checksum = val

    cpdef uint32_t setvalue(self, uint32_t val):
        self._checksum = val

    cdef void c_reset(self) nogil:
        self._checksum = 0

    cpdef reset(self):
        self._checksum = 0
