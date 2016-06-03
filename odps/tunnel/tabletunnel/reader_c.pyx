# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from libc.stdint cimport *
from libc.string cimport *

from ..pb.decoder_c cimport Decoder
from ..checksum_c cimport Checksum

from ..wireconstants import ProtoWireConstants
from ... import utils, types, compat
from ...models import Record
from ...readers import AbstractRecordReader


cdef class BaseTableTunnelReader:
    def __init__(self, object schema, object input_stream, columns=None):
        self._schema = schema
        if columns is None:
            self._columns = self._schema.columns
        else:
            self._columns = [self._schema[c] for c in columns]
        self._reader = Decoder(input_stream)
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._curr_cusor = 0

    def _mode(self):
        return 'c'

    @property
    def count(self):
        return self._curr_cusor

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
        self._crc.update_float(val)

        return val

    cdef bint _read_bool(self):
        cdef bint val

        val = self._reader.read_bool()
        self._crc.update_bool(val)

        return val

    cdef int64_t _read_bigint(self):
        cdef int64_t val

        val = self._reader.read_sint64()
        self._crc.update_long(val)

        return val

    cdef object _read_datetime(self):
        cdef int64_t val

        val = self._reader.read_sint64()
        self._crc.update_long(val)
        return utils.to_datetime(val)

        return val

    cdef _set_string(self, object record, int i):
        cdef bytes val = self._read_string()
        record[i] = val

    cdef _set_double(self, object record, int i):
        cdef double val = self._read_double()
        record[i] = val

    cdef _set_bool(self, object record, int i):
        cdef bint val = self._read_bool()
        record[i] = val

    cdef _set_bigint(self, object record, int i):
        cdef int64_t val = self._read_bigint()
        record[i] = val

    cdef _set_datetime(self, object record, int i):
        cdef object val = self._read_datetime()
        record[i] = val

    cdef _set_decimal(self, object record, int i):
        cdef bytes val

        val = self._reader.read_string()
        self._crc.update(val)
        record[i] = val

    cdef dict _get_read_functions(self):
        return {
            types.boolean: self._set_bool,
            types.datetime: self._set_datetime,
            types.string: self._set_string,
            types.double: self._set_double,
            types.bigint: self._set_bigint,
            types.decimal: self._set_decimal
        }

    cpdef read(self):
        cdef:
            int index
            int checksum
            int idx_of_checksum
            int i
            object data_type
            object record

        record = Record(self._columns)

        while True:
            index, _ = self._reader.read_field_number_and_wire_type()

            if index == 0:
                continue
            if index == ProtoWireConstants.TUNNEL_END_RECORD:
                checksum = utils.long_to_int(self._crc.getvalue())
                if int(self._reader.read_uint32()) != utils.int_to_uint(checksum):
                    raise IOError('Checksum invalid')
                self._crc.reset()
                self._crccrc.update_int(checksum)
                break

            if index == ProtoWireConstants.TUNNEL_META_COUNT:
                if self.count != self._reader.read_sint64():
                    raise IOError('count does not match')
                idx_of_checksum, _ = self._reader.read_field_number_and_wire_type()
                if ProtoWireConstants.TUNNEL_META_CHECKSUM != idx_of_checksum:
                    raise IOError('Invalid stream data.')
                if int(self._crccrc.getvalue()) != self._reader.read_uint32():
                    raise IOError('Checksum invalid.')
                # if not self._reader.at_end():
                #     raise IOError('Expect at the end of stream, but not.')

                return

            if index > len(self._columns):
                raise IOError('Invalid protobuf tag. Perhaps the datastream '
                              'from server is crushed.')

            self._crc.update_int(index)

            read_functions = self._get_read_functions()

            i = index - 1
            data_type = self._columns[i].type
            if data_type in read_functions:
                read_functions[data_type](self, record, i)
            elif isinstance(data_type, types.Array):
                val = self._read_array(data_type.value_type)
                record[i] = val
            elif isinstance(data_type, types.Map):
                keys = self._read_array(data_type.key_type)
                values = self._read_array(data_type.value_type)
                val = compat.OrderedDict(zip(keys, values))
                record[i] = val
            else:
                raise IOError('Unsupported type %s' % data_type)

        self._curr_cusor += 1
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


class TableTunnelReader(BaseTableTunnelReader, AbstractRecordReader):
    def __next__(self):
        record = self.read()
        if record is None:
            raise StopIteration
        return record

    next = __next__