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
from libcpp.vector cimport vector

from ...src.types_c cimport SchemaSnapshot
from ..pb.decoder_c cimport Decoder
from ..checksum_c cimport Checksum


ctypedef void (*_SET_FUNCTION)(BaseTunnelRecordReader self, list record, int i)


cdef class BaseTunnelRecordReader:

    cdef object _schema
    cdef object _columns
    cdef object _to_datetime
    cdef Decoder _reader
    cdef Checksum _crc
    cdef Checksum _crccrc
    cdef int _curr_cursor
    cdef int _n_columns
    cdef int _read_limit
    cdef vector[_SET_FUNCTION] _column_setters
    cdef SchemaSnapshot _schema_snapshot

    cdef list _read_array(self, object value_type)
    cdef bytes _read_string(self)
    cdef double _read_double(self)
    cdef bint _read_bool(self)
    cdef int64_t _read_bigint(self)
    cdef object _read_datetime(self)
    cdef void _set_record_list_value(self, list record, int i, object value)
    cdef void _set_string(self, list record, int i)
    cdef void _set_double(self, list record, int i)
    cdef void _set_bool(self, list record, int i)
    cdef void _set_bigint(self, list record, int i)
    cdef void _set_datetime(self, list record, int i)
    cdef void _set_decimal(self, list record, int i)
    cpdef read(self)
