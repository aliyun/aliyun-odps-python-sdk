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

from ...src.types_c cimport BaseRecord, SchemaSnapshot
from ..checksum_c cimport Checksum
from ..pb.encoder_c cimport Encoder
from ...src.utils_c cimport TO_MILLISECONDS_FUN

cdef class ProtobufRecordWriter:

    cdef int DEFAULT_BUFFER_SIZE
    cdef Encoder _encoder
    cdef object _output
    cdef int _buffer_size
    cdef int _n_total
    cdef int _n_columns

    cpdef _re_init(self, output)

    cpdef flush(self)

    cpdef close(self)

    cpdef flush_all(self)

    cpdef _refresh_buffer(self)

    cpdef _write_tag(self, int field_num, int wire_type)

    cpdef _write_raw_long(self, int64_t val)

    cpdef _write_raw_uint(self, uint32_t val)

    cpdef _write_raw_bool(self, bint val)

    cpdef _write_raw_double(self, double val)

    cpdef _write_raw_string(self, bytes val)


cdef class BaseRecordWriter(ProtobufRecordWriter):
    cdef size_t _curr_cursor_c
    cdef object _to_milliseconds
    cdef TO_MILLISECONDS_FUN _c_to_milliseconds
    cdef object _reader_schema
    cdef Checksum _crc_c
    cdef Checksum _crccrc_c
    cdef SchemaSnapshot _schema_snapshot

    cpdef write(self, BaseRecord record)

    cpdef _write_bool(self, bint data)

    cpdef _write_long(self, int64_t data)

    cpdef _write_double(self, double data)

    cpdef _write_string(self, object data)

    cpdef _write_primitive(self, object data, object data_type)

    cpdef _write_array(self, object data, object data_type)

    cpdef close(self)