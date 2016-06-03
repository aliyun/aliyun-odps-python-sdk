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

from ..checksum_c cimport Checksum
from ..pb.encoder_c cimport Encoder

from ..pb.wire_format import WIRETYPE_VARINT, WIRETYPE_FIXED64, WIRETYPE_LENGTH_DELIMITED
from ..wireconstants import ProtoWireConstants
from ... import types, compat, utils, errors
from ...compat import six


cdef class ProtobufRecordWriter:

    def __init__(self, object output, buffer_size=None):
        self.DEFAULT_BUFFER_SIZE = 4096
        self._encoder = Encoder()
        self._output = output
        self._n_total = 0
        self._buffer_size = buffer_size or self.DEFAULT_BUFFER_SIZE

    cpdef _re_init(self, output):
        self._encoder = Encoder()
        self._output = output
        self._n_total = 0

    def _mode(self):
        return 'c'

    cpdef flush(self):
        if len(self._encoder) > 0:
            data = self._encoder.tostring()
            self._output.write(data)
            self._n_total += len(self._encoder)
            self._encoder = Encoder()

    cpdef close(self):
        self.flush_all()

    cpdef flush_all(self):
        self.flush()
        self._output.flush()

    cpdef _refresh_buffer(self):
        """Control the buffer size of _encoder. Flush if necessary"""
        if len(self._encoder) > self._buffer_size:
            self.flush()

    @property
    def n_bytes(self):
        return self._n_total + len(self._encoder)

    def __len__(self):
        return self.n_bytes

    cpdef _write_tag(self, int field_num, int wire_type):
        self._encoder.append_tag(field_num, wire_type)
        self._refresh_buffer()

    cpdef _write_raw_long(self, int64_t val):
        self._encoder.append_sint64(val)
        self._refresh_buffer()

    cpdef _write_raw_uint(self, uint32_t val):
        self._encoder.append_uint32(val)
        self._refresh_buffer()

    cpdef _write_raw_bool(self, bint val):
        self._encoder.append_bool(val)
        self._refresh_buffer()

    cpdef _write_raw_double(self, double val):
        self._encoder.append_double(val)
        self._refresh_buffer()

    cpdef _write_raw_string(self, bytes val):
        self._encoder.append_string(val)
        self._refresh_buffer()


cdef class BaseRecordWriter(ProtobufRecordWriter):

    def __init__(self, object schema, object out, encoding='utf-8'):
        self._encoding = encoding
        self._schema = schema
        self._columns = self._schema.columns
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._curr_cursor = 0

        super(BaseRecordWriter, self).__init__(out)

    def _get_write_functions(self):
        def write_boolean(pb_index, val):
            self._write_tag(pb_index, WIRETYPE_VARINT)
            self._write_bool(val)

        def write_datetime(pb_index, val):
            val = utils.to_milliseconds(val)
            self._write_tag(pb_index, WIRETYPE_VARINT)
            self._write_long(val)

        def write_string(pb_index, val):
            self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
            self._write_string(val)

        def write_double(pb_index, val):
            self._write_tag(pb_index, WIRETYPE_FIXED64)
            self._write_double(val)

        def write_bigint(pb_index, val):
            self._write_tag(pb_index, WIRETYPE_VARINT)
            self._write_long(val)

        def write_decimal(pb_index, val):
            self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
            self._write_string(str(val))

        return {
            types.boolean: write_boolean,
            types.datetime: write_datetime,
            types.string: write_string,
            types.double: write_double,
            types.bigint: write_bigint,
            types.decimal: write_decimal
        }

    cpdef write(self, object record):
        cdef:
            int n_record_fields
            int n_columns
            int pb_index
            int i
            int checksum
            dict write_functions

        n_record_fields = len(record)
        n_columns = len(self._columns)

        if n_record_fields > n_columns:
            raise IOError('record fields count is more than schema.')

        write_functions = self._get_write_functions()

        for i in range(min(n_record_fields, n_columns)):
            if self._schema.is_partition(self._columns[i]):
                continue

            val = record[i]
            if val is None:
                continue

            pb_index = i + 1
            self._crc.update_int(pb_index)

            data_type = self._columns[i].type
            if data_type in write_functions:
                write_functions[data_type](pb_index, val)
            elif isinstance(data_type, types.Array):
                self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                self._write_raw_uint(len(val))
                self._write_array(val, data_type.value_type)
            elif isinstance(data_type, types.Map):
                self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                self._write_raw_uint(len(val))
                self._write_array(compat.lkeys(val), data_type.key_type)
                self._write_raw_uint(len(val))
                self._write_array(compat.lvalues(val), data_type.value_type)
            else:
                raise IOError('Invalid data type: %s' % data_type)

        checksum = utils.long_to_int(self._crc.getvalue())
        self._write_tag(ProtoWireConstants.TUNNEL_END_RECORD, WIRETYPE_VARINT)
        self._write_raw_uint(utils.long_to_uint(checksum))
        self._crc.reset()
        self._crccrc.update_int(checksum)
        self._curr_cursor += 1

    cpdef _write_bool(self, bint data):
        self._crc.update_bool(data)
        self._write_raw_bool(data)

    cpdef _write_long(self, int64_t data):
        self._crc.update_long(data)
        self._write_raw_long(data)

    cpdef _write_double(self, double data):
        self._crc.update_float(data)
        self._write_raw_double(data)

    cpdef _write_string(self, object data):
        if isinstance(data, six.text_type):
            data = data.encode(self._encoding)
        self._crc.update(data)
        self._write_raw_string(data)

    cpdef _write_primitive(self, object data, object data_type):
        if data_type == types.string:
            self._write_string(data)
        elif data_type == types.bigint:
            self._write_long(data)
        elif data_type == types.double:
            self._write_double(data)
        elif data_type == types.boolean:
            self._write_bool(data)
        else:
            raise IOError('Not a primitive type in array. type: %s' % data_type)

    cpdef _write_array(self, object data, object data_type):
        for value in data:
            if value is None:
                self._write_raw_bool(True)
            else:
                self._write_raw_bool(False)
                self._write_primitive(value, data_type)

    @property
    def count(self):
        return self._curr_cursor

    cpdef close(self):
        self._write_tag(ProtoWireConstants.TUNNEL_META_COUNT, WIRETYPE_VARINT)
        self._write_raw_long(self.count)
        self._write_tag(ProtoWireConstants.TUNNEL_META_CHECKSUM, WIRETYPE_VARINT)
        self._write_raw_uint(utils.long_to_uint(self._crccrc.getvalue()))
        super(BaseRecordWriter, self).close()
        self._curr_cursor = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()