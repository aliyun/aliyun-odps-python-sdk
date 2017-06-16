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

from ...src.types_c cimport BaseRecord, SchemaSnapshot
from ..checksum_c cimport Checksum
from ..pb.encoder_c cimport Encoder
from ...src.utils_c cimport get_to_milliseconds_fun_ptr

from ..pb.wire_format import WIRETYPE_VARINT as PY_WIRETYPE_VARINT,\
    WIRETYPE_FIXED64 as PY_WIRETYPE_FIXED64, WIRETYPE_LENGTH_DELIMITED as PY_WIRETYPE_LENGTH_DELIMITED
from ..wireconstants import ProtoWireConstants
from ... import types, compat, utils, errors
from ...compat import six

cdef:
    uint32_t WIRETYPE_VARINT = PY_WIRETYPE_VARINT
    uint32_t WIRETYPE_FIXED64 = PY_WIRETYPE_FIXED64
    uint32_t WIRETYPE_LENGTH_DELIMITED = PY_WIRETYPE_LENGTH_DELIMITED

cdef:
    uint32_t WIRE_TUNNEL_META_COUNT = ProtoWireConstants.TUNNEL_META_COUNT
    uint32_t WIRE_TUNNEL_META_CHECKSUM = ProtoWireConstants.TUNNEL_META_CHECKSUM
    uint32_t WIRE_TUNNEL_END_RECORD = ProtoWireConstants.TUNNEL_END_RECORD

cdef:
    int64_t BOOL_TYPE_ID = types.boolean._type_id
    int64_t DATETIME_TYPE_ID = types.datetime._type_id
    int64_t STRING_TYPE_ID = types.string._type_id
    int64_t DOUBLE_TYPE_ID = types.double._type_id
    int64_t BIGINT_TYPE_ID = types.bigint._type_id
    int64_t DECIMAL_TYPE_ID = types.decimal._type_id


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
        self._schema_snapshot = self._schema.build_snapshot()

        self._crc_c = Checksum()
        self._crccrc_c = Checksum()
        self._curr_cursor_c = 0
        self._n_columns = len(self._columns)
        self._to_milliseconds = utils.build_to_milliseconds()
        self._c_to_milliseconds = get_to_milliseconds_fun_ptr(self._to_milliseconds)

        super(BaseRecordWriter, self).__init__(out)

    @property
    def _crc(self):
        return self._crc_c

    @property
    def _crccrc(self):
        return self._crccrc_c

    cpdef write(self, BaseRecord record):
        cdef:
            int n_record_fields
            int pb_index
            int i
            int data_type_id
            int checksum
            object val
            uint64_t l_val

        n_record_fields = len(record)

        if n_record_fields > self._n_columns:
            raise IOError('record fields count is more than schema.')

        for i in range(min(n_record_fields, self._n_columns)):
            if self._schema_snapshot._col_is_partition[i]:
                continue

            val = record._get(i)
            if val is None:
                continue

            pb_index = i + 1
            self._crc_c.update_int(pb_index)

            data_type_id = self._schema_snapshot._col_type_ids[i]
            if data_type_id == BOOL_TYPE_ID:
                self._write_tag(pb_index, WIRETYPE_VARINT)
                self._write_bool(val)
            elif data_type_id == DATETIME_TYPE_ID:
                if self._c_to_milliseconds != NULL:
                    l_val = self._c_to_milliseconds(val)
                else:
                    l_val = self._to_milliseconds(val)
                self._write_tag(pb_index, WIRETYPE_VARINT)
                self._write_long(l_val)
            elif data_type_id == STRING_TYPE_ID:
                self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                self._write_string(val)
            elif data_type_id == DOUBLE_TYPE_ID:
                self._write_tag(pb_index, WIRETYPE_FIXED64)
                self._write_double(val)
            elif data_type_id == BIGINT_TYPE_ID:
                self._write_tag(pb_index, WIRETYPE_VARINT)
                self._write_long(val)
            elif data_type_id == DECIMAL_TYPE_ID:
                self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                self._write_string(str(val))
            else:
                data_type = self._schema_snapshot._col_types[i]
                if isinstance(data_type, types.Array):
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

        checksum = <int32_t>self._crc_c.getvalue()
        self._write_tag(WIRE_TUNNEL_END_RECORD, WIRETYPE_VARINT)
        self._write_raw_uint(<uint32_t>checksum)
        self._crc_c.reset()
        self._crccrc_c.c_update_int(checksum)
        self._curr_cursor_c += 1

    cpdef _write_bool(self, bint data):
        self._crc_c.c_update_bool(data)
        self._write_raw_bool(data)

    cpdef _write_long(self, int64_t data):
        self._crc_c.c_update_long(data)
        self._write_raw_long(data)

    cpdef _write_double(self, double data):
        self._crc_c.c_update_float(data)
        self._write_raw_double(data)

    cpdef _write_string(self, object data):
        if isinstance(data, six.text_type):
            data = data.encode(self._encoding)
        self._crc_c.update(data)
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
        return self._curr_cursor_c

    @property
    def _curr_cursor(self):
        return self._curr_cursor_c

    @_curr_cursor.setter
    def _curr_cursor(self, value):
        self._curr_cursor_c = value

    cpdef close(self):
        self._write_tag(WIRE_TUNNEL_META_COUNT, WIRETYPE_VARINT)
        self._write_raw_long(self.count)
        self._write_tag(WIRE_TUNNEL_META_CHECKSUM, WIRETYPE_VARINT)
        self._write_raw_uint(<uint32_t>self._crccrc_c.getvalue())
        super(BaseRecordWriter, self).close()
        self._curr_cursor_c = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()