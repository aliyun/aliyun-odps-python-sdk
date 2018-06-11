# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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
from ..checksum_c cimport Checksum
from .util_c cimport *
from .block_decoder_c cimport Decoder


ctypedef void (*_NOGIL_READER)(TunnelPandasReader self, ArrayVariantPtrs &record, int row) nogil


cdef class TunnelPandasReader:
    cdef object _schema
    cdef object _columns
    cdef object _stream
    cdef Checksum _crc
    cdef Checksum _crccrc
    cdef int _n_columns
    cdef int _read_limit
    cdef object _reader_schema
    cdef SchemaSnapshot _schema_snapshot
    cdef Decoder _decoder

    cdef int _cur_cursor
    cdef int _mem_cache_size
    cdef int _mem_cache_bound
    cdef int _row_cache_size

    cdef vector[_NOGIL_READER] _nogil_readers
    cdef object _mem_cache
    cdef int _row_mem_ptr
    cdef uint32_t _row_checksum
    cdef int _row_ptr
    cdef int _use_no_gil

    cdef void _read_bool(self, ArrayVariantPtrs &aptr, int row) nogil
    cdef void _read_int64(self, ArrayVariantPtrs &aptr, int ptr) nogil
    cdef void _read_float(self, ArrayVariantPtrs &aptr, int idx) nogil
    cdef void _read_double(self, ArrayVariantPtrs &aptr, int row) nogil
    cdef int _fill_ndarrays_nogil(self, vector[ArrayVariantPtrs] &col_ptrs, int start_row, int limit) nogil

    cdef void _scan_schema(self)
    cpdef refill_cache(self)
    cpdef reset_positions(self, object cache, int cache_size)

    cpdef int readinto(self, object buffers, object columns=*, int limit=*) except? -1
