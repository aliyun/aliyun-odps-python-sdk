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

from ..crc import Crc32

cdef extern from "../src/crc32c/crc32c.c":
    uint32_t crc32c(uint32_t crc, const void *buf, size_t length)

cdef class Checksum:

    def __cinit__(self, method='crc32c'):
        if method == 'crc32c':
            self._crc32c = 0
            self.use_c = 1
        else:
            self.crc = Crc32()
            self.use_c = 0

    cdef void c_update_bool(self, bint val):
        cdef char retval
        retval = 1 if val else 0
        self.c_update(<char *>&retval, 1)

    cpdef update_bool(self, bint val):
        self.c_update_bool(val)

    cdef void c_update_int(self, int32_t val):
        self.c_update(<char *>&val, sizeof(int32_t))

    cpdef update_int(self, int32_t val):
        self.c_update(<char *>&val, sizeof(int32_t))

    cdef void c_update_long(self, int64_t val):
        self.c_update(<char *>&val, sizeof(int64_t))

    cpdef update_long(self, int64_t val):
        self.c_update(<char *>&val, sizeof(int64_t))

    cdef void c_update_float(self, double val):
        self.c_update(<char *>&val, sizeof(double))

    cpdef update_float(self, double val):
        self.c_update(<char *>&val, sizeof(double))

    cdef void c_update(self, char *ptr, size_t length):
        if self.use_c:
            self._crc32c = crc32c(self._crc32c, ptr, length)
        else:
            self.crc.update(bytearray(ptr[:length]))

    cpdef update(self, bytes b):
        if self.use_c:
            buf = bytearray(b)
            self._crc32c = crc32c(self._crc32c, <const void*>(<char *>buf), len(buf))
        else:
            self.crc.update(bytearray(b))

    cpdef uint32_t getvalue(self):
        if self.use_c:
            return self._crc32c
        else:
            return self.crc.getvalue()

    cpdef reset(self):
        if self.use_c:
            self._crc32c = 0
        else:
            self.crc.reset()