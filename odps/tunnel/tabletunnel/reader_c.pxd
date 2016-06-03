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

from ..pb.decoder_c cimport Decoder
from ..checksum_c cimport Checksum


cdef class BaseTableTunnelReader:

    cdef object _schema
    cdef object _columns
    cdef Decoder _reader
    cdef Checksum _crc
    cdef Checksum _crccrc
    cdef int _curr_cusor

    cdef list _read_array(self, object value_type)
    cdef bytes _read_string(self)
    cdef double _read_double(self)
    cdef bint _read_bool(self)
    cdef int64_t _read_bigint(self)
    cdef object _read_datetime(self)
    cdef _set_string(self, object record, int i)
    cdef _set_double(self, object record, int i)
    cdef _set_bool(self, object record, int i)
    cdef _set_bigint(self, object record, int i)
    cdef _set_datetime(self, object record, int i)
    cdef _set_decimal(self, object record, int i)
    cdef dict _get_read_functions(self)
    cpdef read(self)