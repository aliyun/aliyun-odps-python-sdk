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

import numpy as np
cimport numpy as np
try:
    from scipy import sparse as sps
except ImportError:
    sps = None

from ..checksum_c cimport Checksum
from .block_encoder_c cimport Encoder
from . import errno
from ..pb import wire_format
from ..wireconstants import ProtoWireConstants
from ... import types, options

try:
    import pandas as pd
except ImportError:
    pd = None

cdef:
    uint32_t WIRETYPE_VARINT = wire_format.WIRETYPE_VARINT
    uint32_t WIRETYPE_FIXED32 = wire_format.WIRETYPE_FIXED32
    uint32_t WIRETYPE_FIXED64 = wire_format.WIRETYPE_FIXED64
    uint32_t WIRETYPE_LENGTH_DELIMITED = wire_format.WIRETYPE_LENGTH_DELIMITED

    uint32_t WIRE_TUNNEL_META_COUNT = ProtoWireConstants.TUNNEL_META_COUNT
    uint32_t WIRE_TUNNEL_META_CHECKSUM = ProtoWireConstants.TUNNEL_META_CHECKSUM
    uint32_t WIRE_TUNNEL_END_RECORD = ProtoWireConstants.TUNNEL_END_RECORD

    int BD_SUCCESS = errno.BD_SUCCESS
    int BD_BUFFER_EXHAUSTED = errno.BD_BUFFER_EXHAUSTED
    int BD_CHECKSUM_INVALID = errno.BD_CHECKSUM_INVALID
    int BD_COUNT_NOT_MATCH = errno.BD_COUNT_NOT_MATCH
    int BD_INVALID_STREAM_DATA = errno.BD_INVALID_STREAM_DATA
    int BD_INVALID_PB_TAG = errno.BD_INVALID_PB_TAG


cdef class BasePandasWriter:
    def __init__(self, object schema, object output_stream, columns=None):
        self._schema = schema
        if columns is None:
            self._columns = self._schema.columns
        else:
            self._columns = [self._schema[c] for c in columns]
        self._n_columns = len(self._columns)

        self._mem_cache_size = options.tunnel.pd_mem_cache_size

        self._stream = output_stream

        self._count = 0
        self._row_pos = 0
        self._crc = Checksum()
        self._crccrc = Checksum()

    cdef void _write_long_val(self, long val) nogil:
        self._crc.c_update_long(val)
        self._encoder.append_sint32(val)

    cdef void _write_long(self, ArrayVariantPtrs &aptr, int index) nogil:
        cdef long val = aptr.v.pl_array[index]
        self._crc.c_update_long(val)
        self._encoder.append_sint32(val)

    cdef void _write_bool(self, ArrayVariantPtrs &aptr, int index) nogil:
        cdef np.npy_bool val = aptr.v.pbool_array[index]
        self._crc.c_update_bool(val)
        self._encoder.append_bool(val)

    cdef void _write_float(self, ArrayVariantPtrs &aptr, int index) nogil:
        cdef float val = aptr.v.pflt_array[index]
        self._crc.c_update_float(val)
        self._encoder.append_float(val)

    cdef void _write_double(self, ArrayVariantPtrs &aptr, int index) nogil:
        cdef double val = aptr.v.pdbl_array[index]
        self._crc.c_update_double(val)
        self._encoder.append_double(val)

    cpdef reset_positions(self):
        self._row_pos = 0
        self._encoder.init(self._mem_cache, self._mem_cache_size)

    cpdef init_cache(self):
        self._mem_cache = bytearray(self._mem_cache_size)
        self._mem_cache_view = memoryview(self._mem_cache)
        self._encoder = Encoder()
        self.reset_positions()

    cpdef write_stream(self, object data, int length):
        self._stream.write(self._mem_cache_view[:length])

    cdef int _write_single_ndarray_nogil(self, ArrayVariantPtrs &col_ptr, vector[int] &dims,
                                         vector[int] &col_to_dim, long start_pos, long limit,
                                         vector[long] &dim_offsets) nogil:
        cdef:
            int i, j, dim_id
            long flat_pos
            long max_pos = 1
            long rest_pos
            vector[int] array_pos

        array_pos.resize(dims.size())
        rest_pos = start_pos
        for i in reversed(range(dims.size())):
            max_pos *= dims[i]
            array_pos[i] = rest_pos % dims[i]
            rest_pos /= dims[i]

        if limit > 0:
            max_pos = min(limit, max_pos)

        self._crc.c_setvalue(0)
        for i in range(start_pos, max_pos):
            self._crc.c_setvalue(0)

            for j in range(col_to_dim.size()):
                if col_to_dim[j] < 0:
                    continue
                self._crc.c_update_int(j + 1)
                if col_to_dim[j] == 0:
                    self._encoder.append_tag(j + 1, col_ptr.wire_type)
                    if self._encoder.get_last_error() != BD_SUCCESS:
                        return i - start_pos

                    self._nogil_writers[0](self, col_ptr, i)
                    if self._encoder.get_last_error() != BD_SUCCESS:
                        return i - start_pos
                else:
                    dim_id = col_to_dim[j] - 1
                    self._encoder.append_tag(j + 1, WIRETYPE_VARINT)
                    if self._encoder.get_last_error() != BD_SUCCESS:
                        return i - start_pos

                    self._write_long_val(array_pos[dim_id] + dim_offsets[dim_id])
                    if self._encoder.get_last_error() != BD_SUCCESS:
                        return i - start_pos

            checksum = <int32_t>self._crc.c_getvalue()

            self._encoder.append_tag(WIRE_TUNNEL_END_RECORD, WIRETYPE_VARINT)
            if self._encoder.get_last_error() != BD_SUCCESS:
                return i - start_pos
            self._encoder.append_uint32(checksum)
            if self._encoder.get_last_error() != BD_SUCCESS:
                return i - start_pos
            self._crccrc.c_update_int(checksum)

            self._row_pos = self._encoder.position()

            array_pos[array_pos.size() - 1] += 1
            for j in reversed(range(1, array_pos.size())):
                if array_pos[j] >= dims[j]:
                    array_pos[j - 1] += 1
                    array_pos[j] = 0
                else:
                    break

        return max_pos - start_pos

    cdef int _write_dims_nogil(self, vector[ArrayVariantPtrs] &col_ptrs, vector[int] &col_to_dim,
                               long start_row, long limit) nogil:
        cdef:
            int i
            int dim_id
            int row_id

        for row_id in range(start_row, limit):
            self._crc.c_setvalue(0)

            for i in range(col_to_dim.size()):
                if col_to_dim[i] < 0:
                    continue
                self._crc.c_update_int(i + 1)
                dim_id = col_to_dim[i] - 1

                self._encoder.append_tag(i + 1, col_ptrs[dim_id].wire_type)
                if self._encoder.get_last_error() != BD_SUCCESS:
                    return row_id - start_row

                self._nogil_writers[dim_id](self, col_ptrs[dim_id], row_id)
                if self._encoder.get_last_error() != BD_SUCCESS:
                    return row_id - start_row

            checksum = <int32_t>self._crc.c_getvalue()

            self._encoder.append_tag(WIRE_TUNNEL_END_RECORD, WIRETYPE_VARINT)
            if self._encoder.get_last_error() != BD_SUCCESS:
                return row_id - start_row
            self._encoder.append_uint32(checksum)
            if self._encoder.get_last_error() != BD_SUCCESS:
                return row_id - start_row
            self._crccrc.c_update_int(checksum)

            self._row_pos = self._encoder.position()

        return limit - start_row

    cpdef _write_single_array(self, object data, object columns, long limit, object dim_offsets):
        cdef:
            int i
            long start_pos
            long count_delta
            long total_size
            vector[int] dims
            vector[long] dim_offset_vct
            vector[int] col_to_dim

            ArrayVariantPtrs col_ptr
            Py_buffer buf

            dict col_ids
            object array_type
            object val_column

        total_size = 1
        dims.resize(len(data.shape))
        for i in range(dims.size()):
            dims[i] = data.shape[i]
            total_size *= dims[i]

        dim_offset_vct.resize(len(data.shape))
        if dim_offsets is None:
            for i in range(dim_offset_vct.size()):
                dim_offset_vct[i] = 0
        else:
            for i in range(dim_offset_vct.size()):
                dim_offset_vct[i] = dim_offsets[i]

        col_to_dim.resize(data.ndim + 1)
        if columns is None:
            if self._n_columns != col_to_dim.size():
                raise ValueError('Column number not consistent with array shape: num of '
                                 'columns should be 1 + ndim')
            for i in range(col_to_dim.size()):
                col_to_dim[i] = i + 1
            col_to_dim[col_to_dim.size() - 1] = 0
            val_column = self._schema[-1]
        else:
            col_ids = dict()
            for idx, col_name in enumerate(columns):
                col_ids[col_name] = idx + 1
            col_ids[columns[-1]] = 0

            for idx, col in enumerate(self._schema):
                if col.name in col_ids:
                    i = idx
                    col_to_dim[i] = col_ids[col.name]
                    if col_to_dim[i] == 0:
                        val_column = col
                else:
                    col_to_dim[i] = -1

        self._nogil_writers.resize(1)
        if val_column.type == types.float_:
            col_ptr.wire_type = WIRETYPE_FIXED32
            data = data.astype(np.float_) if data.dtype != np.float_ else data
            col_ptr.v.pflt_array = <float *>np.PyArray_DATA(data)
            self._nogil_writers[0] = self._write_float
        elif val_column.type == types.double:
            col_ptr.wire_type = WIRETYPE_FIXED64
            data = data.astype(np.double) if data.dtype != np.double else data
            col_ptr.v.pdbl_array = <double *>np.PyArray_DATA(data)
            self._nogil_writers[0] = self._write_double
        elif val_column.type in types.integer_types:
            col_ptr.wire_type = WIRETYPE_VARINT
            data = data.astype(np.int64) if data.dtype != np.int64 else data
            col_ptr.v.pl_array = <int64_t *>np.PyArray_DATA(data)
            self._nogil_writers[0] = self._write_long
        elif val_column.type == types.boolean:
            col_ptr.wire_type = WIRETYPE_VARINT
            data = data.astype(np.bool_) if data.dtype != np.bool_ else data
            col_ptr.v.pbool_array = <np.npy_bool *>np.PyArray_DATA(data)
            self._nogil_writers[0] = self._write_bool

        if limit <= 0:
            limit = total_size

        start_pos = 0
        self.init_cache()
        while start_pos < limit:
            self.reset_positions()
            count_delta = self._write_single_ndarray_nogil(
                col_ptr, dims, col_to_dim, start_pos, limit, dim_offset_vct)
            self._count += count_delta
            start_pos += count_delta
            self.write_stream(self._mem_cache, self._row_pos)

    cpdef _write_dims(self, object data, object columns, long limit):
        cdef:
            int i
            long start_pos
            long count_delta
            long total_size
            vector[int] dims
            vector[int] col_to_dim

            vector[ArrayVariantPtrs] col_ptrs
            dict col_dict
            dict col_idx

            object col_data
            int64_t[:] int_mmap
            float[:] flt_mmap
            double[:] dbl_mmap
            np.ndarray[np.npy_bool, ndim=1, cast=True] bool_array

        col_dict = dict()
        col_idx = dict()

        if isinstance(data, dict):
            col_dict = data
        elif pd and isinstance(data, pd.DataFrame):
            for col_name in data:
                col_dict[col_name] = data[col_name].values
        elif columns is not None:
            for col_name, buf in zip(columns, data):
                col_dict[col_name] = buf
        else:
            for col, buf in zip(self._columns, data):
                col_dict[col.name] = buf

        if limit < 0:
            limit = 0x7fffffff
            for buf in col_dict.values():
                limit = min(limit, len(buf))

        col_ptrs.resize(self._n_columns)
        col_to_dim.resize(self._n_columns)

        i = 0
        self._nogil_writers.resize(len(col_dict))
        for col_name, col_data in col_dict.items():
            data_type = self._schema[col_name].type
            if data_type == types.float_:
                col_ptrs[i].wire_type = WIRETYPE_FIXED32
                flt_mmap = col_data.astype(np.float_) if col_data.dtype != np.float_ else col_data
                col_ptrs[i].v.pflt_array = &flt_mmap[0]
                self._nogil_writers[i] = self._write_float
            elif data_type == types.double:
                col_ptrs[i].wire_type = WIRETYPE_FIXED64
                dbl_mmap = col_data.astype(np.double) if col_data.dtype != np.double else col_data
                col_ptrs[i].v.pdbl_array = &dbl_mmap[0]
                self._nogil_writers[i] = self._write_double
            elif data_type in types.integer_types:
                col_ptrs[i].wire_type = WIRETYPE_VARINT
                int_mmap  = col_data.astype(np.int64) if col_data.dtype != np.int64 else col_data
                col_ptrs[i].v.pl_array = &int_mmap[0]
                self._nogil_writers[i] = self._write_long
            elif data_type == types.boolean:
                col_ptrs[i].wire_type = WIRETYPE_VARINT
                bool_array  = col_data.astype(np.bool_) if col_data.dtype != np.bool_ else col_data
                col_ptrs[i].v.pbool_array = <np.npy_bool *>bool_array.data
                self._nogil_writers[i] = self._write_bool
            col_idx[col_name] = i
            i += 1

        for i in range(self._n_columns):
            col_name = self._columns[i].name

            if col_name not in col_idx:
                col_to_dim[i] = -1
                continue
            else:
                col_to_dim[i] = col_idx[col_name] + 1

        start_pos = 0
        self.init_cache()
        while start_pos < limit:
            self.reset_positions()
            count_delta = self._write_dims_nogil(col_ptrs, col_to_dim, start_pos, limit)
            self._count += count_delta
            start_pos += count_delta
            self._stream.write(self._mem_cache_view[:self._row_pos])

    cpdef write(self, object data, object columns=None, long limit=-1, object dim_offsets=None):
        """
        Write a numpy array, a pandas DataFrame or a dict of column names and columns into a table.
        When writing a numpy array, the indices and value of every element is written. The indices
        are written before values. The argument ``columns`` determines the subset and order of data.
        
        Currently only ``bigint``, ``float``, ``double`` and ``boolean`` are supported.
        
        :param data: data to write, can be numpy array, dict or pandas DataFrame
        :param columns: column names to write
        :param limit: total number of records to write
        :param dim_offsets: offsets for every dimensions, only applicable for arrays
        :return: number of records written
        """
        if isinstance(data, np.ndarray):
            return self._write_single_array(data, columns, limit, dim_offsets)

        if sps:
            if isinstance(data, sps.csr_matrix):
                data = data.tocoo()
            if isinstance(data, sps.coo_matrix):
                row = data.row
                col = data.col
                data_col = data.data
                del data
                if dim_offsets is not None:
                    row += dim_offsets[0]
                    col += dim_offsets[1]
                return self._write_dims([row, col, data_col], columns, limit)

        return self._write_dims(data, columns, limit)

    cpdef close(self):
        self.reset_positions()
        self._encoder.append_tag(WIRE_TUNNEL_META_COUNT, WIRETYPE_VARINT)
        self._encoder.append_sint64(self._count)
        self._encoder.append_tag(WIRE_TUNNEL_META_CHECKSUM, WIRETYPE_VARINT)
        self._encoder.append_uint32(<uint32_t>self._crccrc.getvalue())
        self._stream.write(self._mem_cache_view[:self._encoder.position()])
        self.flush()

    cpdef flush(self):
        self._stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if an error occurs inside the with block, we do not commit
        if exc_val is not None:
            return
        self.close()
