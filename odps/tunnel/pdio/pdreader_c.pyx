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

import ctypes
import numpy as np
cimport numpy as np

from ..checksum_c cimport Checksum
from .block_decoder_c cimport Decoder
from .util_c cimport *
from . import errno
from ..wireconstants import ProtoWireConstants
from ... import types, options

try:
    import pandas as pd
except ImportError:
    pd = None

cdef:
    uint32_t WIRE_TUNNEL_META_COUNT = ProtoWireConstants.TUNNEL_META_COUNT
    uint32_t WIRE_TUNNEL_META_CHECKSUM = ProtoWireConstants.TUNNEL_META_CHECKSUM
    uint32_t WIRE_TUNNEL_END_RECORD = ProtoWireConstants.TUNNEL_END_RECORD

    int BD_SUCCESS = errno.BD_SUCCESS
    int BD_BUFFER_EXHAUSTED = errno.BD_BUFFER_EXHAUSTED
    int BD_CHECKSUM_INVALID = errno.BD_CHECKSUM_INVALID
    int BD_COUNT_NOT_MATCH = errno.BD_COUNT_NOT_MATCH
    int BD_INVALID_STREAM_DATA = errno.BD_INVALID_STREAM_DATA
    int BD_INVALID_PB_TAG = errno.BD_INVALID_PB_TAG


cdef class TunnelPandasReader:
    def __init__(self, object schema, object input_stream, columns=None):
        self._schema = schema
        if columns is None:
            self._columns = self._schema.columns
        else:
            self._columns = [self._schema[c] for c in columns]
        self._reader_schema = types.OdpsSchema(columns=self._columns)
        self._schema_snapshot = self._reader_schema.build_snapshot()
        self._n_columns = len(self._columns)

        self._mem_cache_size = options.tunnel.pd_mem_cache_size

        self._stream = input_stream

        self._row_mem_ptr = 0
        self._row_checksum = 0
        self._cur_cursor = 0
        self._mem_cache_bound = 0
        self._mem_cache = None
        self._crc = Checksum()
        self._crccrc = Checksum()

        self._scan_schema()

    cdef void _scan_schema(self):
        self._nogil_readers.resize(self._n_columns)
        self._use_no_gil = 1
        for i in range(self._n_columns):
            self._nogil_readers[i] = NULL
            data_type = self._schema_snapshot._col_types[i]
            if data_type == types.boolean:
                self._nogil_readers[i] = self._read_bool
            elif data_type == types.datetime:
                self._use_no_gil = 0
            elif data_type == types.string:
                self._use_no_gil = 0
            elif data_type == types.float_:
                self._nogil_readers[i] = self._read_float
            elif data_type == types.double:
                self._nogil_readers[i] = self._read_double
            elif data_type in types.integer_types:
                self._nogil_readers[i] = self._read_int64
            elif data_type == types.binary:
                self._use_no_gil = 0
            elif data_type == types.timestamp:
                self._use_no_gil = 0
            elif data_type == types.interval_day_time:
                self._use_no_gil = 0
            elif data_type == types.interval_year_month:
                self._use_no_gil = 0
            elif isinstance(data_type, types.Decimal):
                self._use_no_gil = 0
            elif isinstance(data_type, (types.Char, types.Varchar)):
                self._use_no_gil = 0
            else:
                self._use_no_gil = 0

    @property
    def row_mem_ptr(self):
        return self._row_mem_ptr

    @property
    def mem_cache_bound(self):
        return self._mem_cache_bound

    cpdef reset_positions(self, object cache, int cache_size):
        cdef char *cache_ptr
        cdef uint64_t cache_ptr_int

        self._mem_cache_bound = cache_size
        self._row_mem_ptr = 0
        self._row_ptr = 0
        self._crc.c_setvalue(self._row_checksum)
        self._mem_cache = cache

        if isinstance(cache, ctypes.Array):
            cache_ptr_int = ctypes.addressof(cache)
            cache_ptr = <char *>cache_ptr_int
        else:
            cache_ptr = cache
        self._decoder = Decoder()
        self._decoder.init(cache_ptr, self._mem_cache_bound)

    cpdef refill_cache(self):
        cdef bytearray new_mem_cache = bytearray(self._mem_cache_size)
        cdef object new_mem_view = memoryview(new_mem_cache)
        cdef object old_mem_view
        cdef int left_size = 0
        cdef int read_size = 0

        if self._mem_cache is not None:
            old_mem_view = memoryview(self._mem_cache)
            left_size = self._mem_cache_bound - self._row_mem_ptr
            if left_size:
                new_mem_view[0:left_size] = old_mem_view[self._row_mem_ptr:]

        read_size = self._stream.readinto(new_mem_view[left_size:])
        self.reset_positions(new_mem_cache, left_size + read_size)
        return read_size

    cdef void _read_bool(self, ArrayVariantPtrs &aptr, int row) nogil:
        cdef bint val
        val = self._decoder.read_bool()
        if self._decoder.get_last_error():
            return
        aptr.v.pbool_array[row] = val
        self._crc.c_update_bool(val)

    cdef void _read_int64(self, ArrayVariantPtrs &aptr, int row) nogil:
        cdef int64_t val
        val = self._decoder.read_sint64()
        if self._decoder.get_last_error():
            return
        aptr.v.pl_array[row] = val
        self._crc.c_update_long(val)

    cdef void _read_float(self, ArrayVariantPtrs &aptr, int row) nogil:
        cdef float val
        val = self._decoder.read_float()
        if self._decoder.get_last_error():
            return
        aptr.v.pflt_array[row] = val
        self._crc.c_update_float(val)

    cdef void _read_double(self, ArrayVariantPtrs &aptr, int row) nogil:
        cdef double val
        val = self._decoder.read_double()
        if self._decoder.get_last_error():
            return
        aptr.v.pdbl_array[row] = val
        self._crc.c_update_double(val)

    cdef int _fill_ndarrays_nogil(self, vector[ArrayVariantPtrs] &col_ptrs, int start_row, int limit) nogil:
        cdef int idx
        cdef int field_num
        cdef int idx_of_checksum
        cdef int rowid = start_row
        cdef int32_t checksum

        while rowid < limit:
            while True:
                field_num = self._decoder.read_field_number()
                if self._decoder.get_last_error() == BD_BUFFER_EXHAUSTED:
                    return rowid - start_row

                if field_num == 0:
                    continue
                if field_num == WIRE_TUNNEL_END_RECORD:
                    checksum = <int32_t>self._crc.c_getvalue()
                    if self._decoder.read_uint32() != <uint32_t>checksum:
                        if self._decoder.get_last_error() == BD_BUFFER_EXHAUSTED:
                            return rowid - start_row
                        self._decoder.set_last_error(BD_CHECKSUM_INVALID)
                        return rowid - start_row
                    self._crc.c_reset()
                    self._crccrc.c_update_int(checksum)
                    break
                if field_num == WIRE_TUNNEL_META_COUNT:
                    if self._cur_cursor != self._decoder.read_sint64():
                        if self._decoder.get_last_error() == BD_BUFFER_EXHAUSTED:
                            return rowid - start_row
                        self._decoder.set_last_error(BD_COUNT_NOT_MATCH)
                        return rowid - start_row
                    idx_of_checksum = self._decoder.read_field_number()
                    if self._decoder.get_last_error() == BD_BUFFER_EXHAUSTED:
                        return rowid - start_row
                    if WIRE_TUNNEL_META_CHECKSUM != idx_of_checksum:
                        self._decoder.set_last_error(BD_INVALID_STREAM_DATA)
                        return rowid - start_row
                    if self._crccrc.c_getvalue() != self._decoder.read_uint32():
                        if self._decoder.get_last_error() == BD_BUFFER_EXHAUSTED:
                            return rowid - start_row
                        self._decoder.set_last_error(BD_CHECKSUM_INVALID)
                        return rowid - start_row
                    self._row_mem_ptr = self._decoder.position()
                    return rowid - start_row

                if field_num > self._n_columns:
                    self._decoder.set_last_error(BD_INVALID_PB_TAG)
                    return rowid - start_row

                self._crc.c_update_int(field_num)

                idx = field_num - 1
                self._nogil_readers[idx](self, col_ptrs[idx], rowid)
                if self._decoder.get_last_error() == BD_BUFFER_EXHAUSTED:
                    return rowid - start_row
            self._row_mem_ptr = self._decoder.position()
            self._row_checksum = self._crc.c_getvalue()
            rowid += 1
            self._cur_cursor += 1

        return rowid - start_row

    cpdef int readinto(self, object buffers, object columns=None, int limit=-1) except? -1:
        """
        Read data into an existing buffer. The ``buffers`` variable can be a list or a dict
        of numpy arrays, or a Pandas DataFrame. The argument ``columns`` determines the subset
        and order of data.
        
        Currently only ``bigint``, ``float``, ``double`` and ``boolean`` are supported.
        
        :param buffers: data buffer to read, can be numpy array, dict or pandas DataFrame
        :param columns: column names to read
        :param limit: total number of records to read
        :return: number of records read
        """
        cdef:
            int i
            int filled
            int filled_total = 0
            int fetch_count = 0

            vector[ArrayVariantPtrs] col_ptrs
            int64_t[:] int_mmap
            float[:] flt_mmap
            double[:] dbl_mmap
            np.ndarray[np.npy_bool, ndim=1, cast=True] bool_array

            dict col_dict

        if not self._use_no_gil:
            raise NotImplementedError('Currently complex types are not supported.')

        col_dict = dict()
        if isinstance(buffers, dict):
            col_dict = buffers
        elif pd and isinstance(buffers, pd.DataFrame):
            for col_name in buffers:
                col_dict[col_name] = buffers[col_name].values
        elif columns is not None:
            for col_name, buf in zip(columns, buffers):
                col_dict[col_name] = buf
        else:
            for col, buf in zip(self._columns, buffers):
                col_dict[col.name] = buf

        if limit < 0:
            limit = 0x7fffffff
            for buf in col_dict.values():
                limit = min(limit, len(buf))

        col_ptrs.resize(self._n_columns)
        for i in range(self._n_columns):
            col_name = self._columns[i].name
            data_type = self._schema_snapshot._col_types[i]
            if data_type == types.float_:
                flt_mmap = col_dict[col_name]
                col_ptrs[i].v.pflt_array = &flt_mmap[0]
            elif data_type == types.double:
                dbl_mmap = col_dict[col_name]
                col_ptrs[i].v.pdbl_array = &dbl_mmap[0]
            elif data_type in types.integer_types:
                int_mmap = col_dict[col_name]
                col_ptrs[i].v.pl_array = &int_mmap[0]
            elif data_type == types.boolean:
                bool_array = col_dict[col_name]
                col_ptrs[i].v.pbool_array = <np.npy_bool *>bool_array.data

        while filled_total < limit:
            fetch_count = self.refill_cache()
            if fetch_count == 0:
                break
            self._decoder.set_last_error(BD_SUCCESS)
            filled = self._fill_ndarrays_nogil(col_ptrs, filled_total, limit)
            if self._decoder.get_last_error() != BD_SUCCESS:
                if self._decoder.get_last_error() == BD_CHECKSUM_INVALID:
                    raise IOError('Checksum invalid')
                elif self._decoder.get_last_error() == BD_INVALID_STREAM_DATA:
                    raise IOError('Invalid stream data')
                elif self._decoder.get_last_error() == BD_COUNT_NOT_MATCH:
                    raise IOError('Count not match')
                elif self._decoder.get_last_error() == BD_INVALID_PB_TAG:
                    raise IOError('Invalid protobuf tag. Perhaps the datastream '
                                  'from server is crushed.')
            filled_total += filled

        return filled_total

    def read(self, columns=None, limit=-1):
        """
        Read data into a pandas DataFrame. If pandas is not installed, a dict will be returned instead.
        The argument ``columns`` determines the subset and order of data.

        Currently only ``bigint``, ``float``, ``double`` and ``boolean`` are supported.

        :param columns: column names to read
        :param limit: total number of records to read
        :return: number of records read
        """
        if not columns:
            columns = self._schema.names

        buf = dict()
        results = dict()
        buf_len = options.tunnel.pd_row_cache_size

        for col in columns:
            results[col] = []
            col_type = self._schema[col].type
            if col_type == types.float_:
                buf[col] = np.empty((buf_len,), dtype=np.float32)
            elif col_type == types.double:
                buf[col] = np.empty((buf_len,), dtype=np.float64)
            elif col_type in types.integer_types:
                buf[col] = np.empty((buf_len,), dtype=np.int64)
            elif col_type == types.boolean:
                buf[col] = np.empty((buf_len,), dtype=np.bool_)

        read_all = limit < 0
        while read_all or limit > 0:
            read_cols = self.readinto(buf, columns, limit)
            if read_all:
                if read_cols == 0:
                    break
            else:
                limit -= read_cols
            for col in buf:
                results[col].append(np.copy(buf[col]))

        merged = dict()
        for col in results:
            if len(results[col]) > 1:
                merged[col] = np.concatenate(results[col])
            else:
                merged[col] = results[col][0]

        if pd:
            return pd.DataFrame(merged, columns=columns)
        else:
            return merged

    def close(self):
        if hasattr(self._schema, 'close'):
            self._schema.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
