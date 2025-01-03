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

from ...src.types_c cimport SchemaSnapshot
from ..checksum_c cimport Checksum
from ..pb.decoder_c cimport CDecoder


cdef class BaseTunnelRecordReader:
    cdef public object _schema
    cdef object _columns
    cdef object _stream_creator
    cdef public CDecoder _reader
    cdef public Checksum _crc
    cdef Checksum _crccrc
    cdef int _curr_cursor
    cdef int _attempt_row_count
    cdef int _last_n_bytes
    cdef int _n_columns
    cdef int _read_limit
    cdef list _field_readers
    cdef SchemaSnapshot _schema_snapshot
    cdef list _partition_vals
    cdef bint _append_partitions

    cdef public bytes _server_metrics_string
    cdef bint _enable_client_metrics
    cdef long _c_local_wall_time_ms
    cdef long _c_acc_network_time_ms

    cdef int _n_injected_error_cursor
    cdef object _injected_error_exc

    cdef public int _set_record_list_value(
        self, list record, int i, object value
    ) except? -1
    cdef _read(self)
    cpdef read(self)
