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

from ...src.types_c cimport BaseRecord, SchemaSnapshot
from ...src.utils_c cimport CMillisecondsConverter
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
    cdef int _write_tag(self, int field_num, int wire_type) except + nogil
    cdef int _write_raw_long(self, int64_t val) except + nogil
    cdef int _write_raw_int(self, int32_t val) except + nogil
    cdef int _write_raw_uint(self, uint32_t val) except + nogil
    cdef int _write_raw_bool(self, bint val) except + nogil
    cdef int _write_raw_float(self, float val) except + nogil
    cdef int _write_raw_double(self, double val) except + nogil
    cdef int _write_raw_string(self, const char *ptr, uint32_t size) except + nogil


cdef class BaseRecordWriter(ProtobufRecordWriter):
    cdef object _encoding
    cdef bint _is_utf8
    cdef object _schema
    cdef object _columns
    cdef size_t _curr_cursor_c
    cdef object _to_days
    cdef object _reader_schema
    cdef CMillisecondsConverter _mills_converter
    cdef CMillisecondsConverter _mills_converter_utc
    cdef Checksum _crc_c
    cdef Checksum _crccrc_c
    cdef SchemaSnapshot _schema_snapshot

    cpdef write(self, BaseRecord record)
    cdef void _write_bool(self, bint data) except + nogil
    cdef void _write_long(self, int64_t data) except + nogil
    cdef void _write_float(self, float data) except + nogil
    cdef void _write_double(self, double data) except + nogil
    cdef _write_string(self, object data)
    cdef _write_timestamp_base(self, object data, bint ntz)
    cdef _write_timestamp(self, object data)
    cdef _write_timestamp_ntz(self, object data)
    cdef _write_interval_day_time(self, object data)
    cdef _write_field(self, object val, int data_type_id, object data_type)
    cdef _write_array(self, object data, object data_type)
    cdef _write_struct(self, object data, object data_type)
    cpdef _write_finish_tags(self)
    cpdef close(self)