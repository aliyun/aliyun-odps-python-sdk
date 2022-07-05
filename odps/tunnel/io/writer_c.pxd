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
from ..pb.encoder_c cimport Encoder

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
    cpdef int _refresh_buffer(self) except -1
    cpdef int _write_tag(self, int field_num, int wire_type) except -1
    cpdef int _write_raw_long(self, int64_t val) except -1
    cpdef int _write_raw_int(self, int32_t val) except -1
    cpdef int _write_raw_uint(self, uint32_t val) except -1
    cpdef int _write_raw_bool(self, bint val) except -1
    cpdef int _write_raw_float(self, float val) except -1
    cpdef int _write_raw_double(self, double val) except -1
    cpdef int _write_raw_string(self, bytes val) except -1


cdef class BaseRecordWriter(ProtobufRecordWriter):
    cdef object _encoding
    cdef object _schema
    cdef object _columns
    cdef size_t _curr_cursor_c
    cdef object _to_days
    cdef object _reader_schema
    cdef CMillisecondsConverter _mills_converter
    cdef Checksum _crc_c
    cdef Checksum _crccrc_c
    cdef SchemaSnapshot _schema_snapshot

    cpdef write(self, BaseRecord record)
    cpdef _write_bool(self, bint data)
    cpdef _write_long(self, int64_t data)
    cpdef _write_float(self, float data)
    cpdef _write_double(self, double data)
    cpdef _write_string(self, object data)
    cpdef _write_timestamp(self, object data)
    cpdef _write_interval_day_time(self, object data)
    cpdef _write_field(self, object val, int data_type_id, object data_type)
    cpdef _write_array(self, object data, object data_type)
    cpdef _write_struct(self, object data, object data_type)
    cpdef close(self)