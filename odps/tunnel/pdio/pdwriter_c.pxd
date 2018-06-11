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
from .block_encoder_c cimport Encoder


ctypedef void (*_NOGIL_WRITER)(BasePandasWriter self, ArrayVariantPtrs &record, int row) nogil


cdef class BasePandasWriter:
    cdef object _schema
    cdef object _columns
    cdef object _stream
    cdef Checksum _crc
    cdef Checksum _crccrc
    cdef int _n_columns
    cdef int _read_limit
    cdef Encoder _encoder

    cdef object _mem_cache
    cdef object _mem_cache_view
    cdef int _mem_cache_size

    cdef int _count
    cdef uint32_t _row_pos

    cdef vector[_NOGIL_WRITER] _nogil_writers

    cdef void _write_long_val(self, long val) nogil

    cdef void _write_long(self, ArrayVariantPtrs &aptr, int index) nogil

    cdef void _write_bool(self, ArrayVariantPtrs &aptr, int index) nogil

    cdef void _write_float(self, ArrayVariantPtrs &aptr, int index) nogil

    cdef void _write_double(self, ArrayVariantPtrs &aptr, int index) nogil

    cdef int _write_single_ndarray_nogil(
            self, ArrayVariantPtrs &col_ptr, vector[int] &dims, vector[int] &col_to_dim,
            long start_pos, long limit, vector[long] &dim_offsets) nogil
    cdef int _write_dims_nogil(
            self, vector[ArrayVariantPtrs] &col_ptrs, vector[int] &col_to_dim,
            long start_row, long limit) nogil

    cpdef init_cache(self)
    cpdef reset_positions(self)
    cpdef write_stream(self, object data, int length)

    cpdef _write_single_array(self, object data, object columns, long limit, object dim_offsets)
    cpdef _write_dims(self, object data, object columns, long limit)
    cpdef write(self, object data, object columns=*, long limit=*, object dim_offsets=*)

    cpdef flush(self)
    cpdef close(self)
