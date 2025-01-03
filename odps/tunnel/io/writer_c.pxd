# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from ...src.types_c cimport BaseRecord, SchemaSnapshot
from ..checksum_c cimport Checksum
from ..pb.encoder_c cimport CEncoder


cdef class ProtobufRecordWriter:

    cdef int DEFAULT_BUFFER_SIZE
    cdef CEncoder _encoder
    cdef object _output
    cdef int _buffer_size
    cdef int _n_total
    cdef int _n_columns
    cdef int _last_flush_time

    cpdef _re_init(self, output)
    cpdef flush(self)
    cpdef close(self)
    cpdef flush_all(self)
    cpdef int _refresh_buffer(self) except -1
    cdef public int _write_tag(self, int field_num, int wire_type) except -1 nogil
    cdef public int _write_raw_long(self, int64_t val) except -1 nogil
    cdef public int _write_raw_int(self, int32_t val) except -1 nogil
    cdef public int _write_raw_uint(self, uint32_t val) except -1 nogil
    cdef public int _write_raw_bool(self, bint val) except -1 nogil
    cdef public int _write_raw_float(self, float val) except -1 nogil
    cdef public int _write_raw_double(self, double val) except -1 nogil
    cdef public int _write_raw_string(self, const char *ptr, uint32_t size) except -1 nogil


cdef class BaseRecordWriter(ProtobufRecordWriter):
    cdef object _encoding
    cdef public bint _is_utf8
    cdef object _schema
    cdef object _columns
    cdef size_t _curr_cursor_c
    cdef object _reader_schema
    cdef public Checksum _crc_c
    cdef Checksum _crccrc_c
    cdef SchemaSnapshot _schema_snapshot
    cdef list _field_writers

    cdef bint _c_enable_client_metrics  # to avoid conflict with child classes
    cdef long _c_local_wall_time_ms

    cpdef write(self, BaseRecord record)
    cpdef _write_finish_tags(self)
    cpdef close(self)
