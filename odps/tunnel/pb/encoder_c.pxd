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

from util_c cimport *

cdef class Encoder:
    cdef bytearray _buffer

    cpdef bytes tostring(self)

    cpdef int append_tag(self, int field_num, int wire_type)

    cpdef int append_sint32(self, int32_t value)

    cpdef int append_uint32(self, uint32_t value)

    cpdef int append_sint64(self, int64_t value)

    cpdef int append_uint64(self, uint64_t value)

    cpdef int append_bool(self, bint value)

    cpdef int append_double(self, double value)

    cpdef int append_float(self, float value)

    cpdef int append_string(self, bytes value)
