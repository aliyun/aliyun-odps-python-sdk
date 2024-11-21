# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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
from libcpp.vector cimport vector

from ...src.types_c cimport SchemaSnapshot
from ...src.utils_c cimport CMillisecondsConverter
from ..checksum_c cimport Checksum
from ..pb.decoder_c cimport CDecoder

ctypedef int (*_SET_FUNCTION)(BaseTunnelRecordReader self, list record, int i) except? -1


cdef class BaseTunnelRecordReader:
    cdef public object _schema
    cdef object _columns
    cdef object _stream_creator
    cdef CMillisecondsConverter _mills_converter
    cdef CMillisecondsConverter _mills_converter_utc
    cdef object _to_date
    cdef CDecoder _reader
    cdef Checksum _crc
    cdef Checksum _crccrc
    cdef int _curr_cursor
    cdef int _attempt_row_count
    cdef int _last_n_bytes
    cdef int _n_columns
    cdef int _read_limit
    cdef bint _overflow_date_as_none
    cdef bint _struct_as_dict
    cdef vector[_SET_FUNCTION] _column_setters
    cdef SchemaSnapshot _schema_snapshot
    cdef list _partition_vals
    cdef bint _append_partitions

    cdef public bytes _server_metrics_string
    cdef bint _enable_client_metrics
    cdef long _c_local_wall_time_ms
    cdef long _c_acc_network_time_ms

    cdef int _n_injected_error_cursor
    cdef object _injected_error_exc

    cdef object _read_struct(self, object value_type)
    cdef object _read_element(self, int data_type_id, object data_type)
    cdef list _read_array(self, object value_type)
    cdef bytes _read_string(self)
    cdef float _read_float(self) except? -1.0 nogil
    cdef double _read_double(self) except? -1.0 nogil
    cdef bint _read_bool(self) except? False nogil
    cdef int64_t _read_bigint(self) except? -1 nogil
    cdef object _read_datetime(self)
    cdef object _read_date(self)
    cdef object _read_timestamp_base(self, bint ntz)
    cdef object _read_timestamp(self)
    cdef object _read_timestamp_ntz(self)
    cdef object _read_interval_day_time(self)
    cdef int _set_record_list_value(self, list record, int i, object value) except? -1
    cdef int _set_string(self, list record, int i) except? -1
    cdef int _set_float(self, list record, int i) except? -1
    cdef int _set_double(self, list record, int i) except? -1
    cdef int _set_bool(self, list record, int i) except? -1
    cdef int _set_bigint(self, list record, int i) except? -1
    cdef int _set_datetime(self, list record, int i) except? -1
    cdef int _set_date(self, list record, int i) except? -1
    cdef int _set_decimal(self, list record, int i) except? -1
    cdef int _set_timestamp(self, list record, int i) except? -1
    cdef int _set_timestamp_ntz(self, list record, int i) except? -1
    cdef int _set_interval_day_time(self, list record, int i) except? -1
    cdef int _set_interval_year_month(self, list record, int i) except? -1
    cdef int _set_json(self, list record, int i) except? -1
    cdef _read(self)
    cpdef read(self)
