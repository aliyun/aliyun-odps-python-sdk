# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from libc.stdint cimport *
from libc.string cimport *

from ..crc32c_c import Crc32c
from ..crc import Crc32

cdef class Checksum:

    def __cinit__(self, method='crc32c'):
        self.crc = Crc32c() if method == 'crc32c' else Crc32()

    cpdef update_bool(self, bint val):
        cdef char retval
        retval = 1 if val else 0
        self.update((<char *>&retval)[:1])

    cpdef update_int(self, int32_t val):
        self.update((<char *>&val)[:sizeof(int32_t)])

    cpdef update_long(self, int64_t val):
        self.update((<char *>&val)[:sizeof(int64_t)])

    cpdef update_float(self, double val):
        self.update((<char *>&val)[:sizeof(double)])

    cpdef update(self, bytes b):
        self.crc.update(bytearray(b))

    cpdef uint32_t getvalue(self):
        return self.crc.getvalue()

    cpdef reset(self):
        return self.crc.reset()