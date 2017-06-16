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

from ...src.types_c cimport BaseRecord
from ..pb.decoder_c cimport Decoder
from ..checksum_c cimport Checksum

import warnings
from ..wireconstants import ProtoWireConstants
from ... import utils, types, compat, options
from ...models import Record
from ...readers import AbstractRecordReader


cdef:
    uint32_t WIRE_TUNNEL_META_COUNT = ProtoWireConstants.TUNNEL_META_COUNT
    uint32_t WIRE_TUNNEL_META_CHECKSUM = ProtoWireConstants.TUNNEL_META_CHECKSUM
    uint32_t WIRE_TUNNEL_END_RECORD = ProtoWireConstants.TUNNEL_END_RECORD


cdef class BaseTunnelRecordReader:
    def __init__(self, object schema, object input_stream, columns=None):
        self._schema = schema
        if columns is None:
            self._columns = self._schema.columns
        else:
            self._columns = [self._schema[c] for c in columns]
        self._reader_schema = types.OdpsSchema(columns=self._columns)
        self._schema_snapshot = self._reader_schema.build_snapshot()
        self._n_columns = len(self._columns)

        self._column_setters.resize(self._n_columns)
        for i in range(self._n_columns):
            data_type = self._schema_snapshot._col_types[i]
            if data_type == types.boolean:
                self._column_setters[i] = self._set_bool
            elif data_type == types.datetime:
                self._column_setters[i] = self._set_datetime
            elif data_type == types.string:
                self._column_setters[i] = self._set_string
            elif data_type == types.double:
                self._column_setters[i] = self._set_double
            elif data_type == types.bigint:
                self._column_setters[i] = self._set_bigint
            elif data_type == types.decimal:
                self._column_setters[i] = self._set_decimal
            else:
                self._column_setters[i] = NULL

        self._reader = Decoder(input_stream)
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._curr_cursor = 0
        self._read_limit = -1 if options.table_read_limit is None else options.table_read_limit
        self._to_datetime = utils.build_to_datetime()

    def _mode(self):
        return 'c'

    @property
    def count(self):
        return self._curr_cursor

    cdef list _read_array(self, object value_type):
        cdef:
            uint32_t size
            object val
            list res = []

        size = self._reader.read_uint32()
        for _ in range(size):
            if self._reader.read_bool():
                res.append(None)
            else:
                if value_type == types.string:
                    val = self._read_string()
                    val = val.decode('utf-8')
                elif value_type == types.bigint:
                    val = self._read_bigint()
                elif value_type == types.double:
                    val = self._read_double()
                elif value_type == types.boolean:
                    val = self._read_bool()
                else:
                    raise IOError('Unsupport array type. type: %s' % value_type)
                res.append(val)

        return res

    cdef bytes _read_string(self):
        cdef bytes val

        val = self._reader.read_string()
        self._crc.update(val)

        return val

    cdef double _read_double(self):
        cdef double val

        val = self._reader.read_double()
        self._crc.c_update_float(val)

        return val

    cdef bint _read_bool(self):
        cdef bint val

        val = self._reader.read_bool()
        self._crc.c_update_bool(val)

        return val

    cdef int64_t _read_bigint(self):
        cdef int64_t val

        val = self._reader.read_sint64()
        self._crc.c_update_long(val)

        return val

    cdef object _read_datetime(self):
        cdef int64_t val

        val = self._reader.read_sint64()
        self._crc.c_update_long(val)
        return self._to_datetime(val)

    cdef void _set_record_list_value(self, list record, int i, object value):
        record[i] = self._schema_snapshot.validate_value(i, value)

    cdef void _set_string(self, list record, int i):
        cdef object val = self._read_string()
        self._set_record_list_value(record, i, val)

    cdef void _set_double(self, list record, int i):
        cdef double val = self._read_double()
        self._set_record_list_value(record, i, val)

    cdef void _set_bool(self, list record, int i):
        cdef bint val = self._read_bool()
        self._set_record_list_value(record, i, val)

    cdef void _set_bigint(self, list record, int i):
        cdef int64_t val = self._read_bigint()
        self._set_record_list_value(record, i, val)

    cdef void _set_datetime(self, list record, int i):
        cdef object val = self._read_datetime()
        self._set_record_list_value(record, i, val)

    cdef void _set_decimal(self, list record, int i):
        cdef bytes val

        val = self._reader.read_string()
        self._crc.update(val)
        self._set_record_list_value(record, i, val)

    cpdef read(self):
        cdef:
            int index
            int checksum
            int idx_of_checksum
            int i
            int data_type_id
            object data_type
            BaseRecord record
            list rec_list

        if self._curr_cursor >= self._read_limit > 0:
            warnings.warn('Number of lines read via tunnel already reaches the limitation.')
            return None

        record = Record(schema=self._reader_schema)
        rec_list = record._c_values

        while True:
            index = self._reader.read_field_number()

            if index == 0:
                continue
            if index == WIRE_TUNNEL_END_RECORD:
                checksum = <int32_t>self._crc.getvalue()
                if self._reader.read_uint32() != <uint32_t>checksum:
                    raise IOError('Checksum invalid')
                self._crc.reset()
                self._crccrc.c_update_int(checksum)
                break

            if index == WIRE_TUNNEL_META_COUNT:
                if self._curr_cursor != self._reader.read_sint64():
                    raise IOError('count does not match')
                idx_of_checksum = self._reader.read_field_number()
                if WIRE_TUNNEL_META_CHECKSUM != idx_of_checksum:
                    raise IOError('Invalid stream data.')
                if self._crccrc.getvalue() != self._reader.read_uint32():
                    raise IOError('Checksum invalid.')
                # if not self._reader.at_end():
                #     raise IOError('Expect at the end of stream, but not.')

                return

            if index > self._n_columns:
                raise IOError('Invalid protobuf tag. Perhaps the datastream '
                              'from server is crushed.')

            self._crc.update_int(index)

            i = index - 1
            if self._column_setters[i] != NULL:
                self._column_setters[i](self, rec_list, i)
            else:
                data_type = self._schema_snapshot._col_types[i]
                if isinstance(data_type, types.Array):
                    val = self._read_array(data_type.value_type)
                    rec_list[i] = val
                elif isinstance(data_type, types.Map):
                    keys = self._read_array(data_type.key_type)
                    values = self._read_array(data_type.value_type)
                    val = compat.OrderedDict(zip(keys, values))
                    rec_list[i] = val
                else:
                    raise IOError('Unsupported type %s' % data_type)

        self._curr_cursor += 1
        return record

    def reads(self):
        return self.__iter__()

    @property
    def n_bytes(self):
        return self._reader.position()

    def get_total_bytes(self):
        return self.n_bytes

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if hasattr(self._schema, 'close'):
            self._schema.close()


class TunnelRecordReader(BaseTunnelRecordReader, AbstractRecordReader):
    def __next__(self):
        record = self.read()
        if record is None:
            raise StopIteration
        return record

    next = __next__
