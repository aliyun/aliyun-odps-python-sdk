# -*- coding: utf-8 -*-
# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import time

import cython
from cpython.datetime cimport import_datetime
from libc.stdint cimport *
from libc.string cimport *
from libcpp.vector cimport vector

from ...src.types_c cimport BaseRecord, SchemaSnapshot
from ..checksum_c cimport Checksum
from ..pb.encoder_c cimport CEncoder

from ... import types, compat, utils
from ...compat import six
from ...src.utils_c cimport CMillisecondsConverter
from ..pb.wire_format import WIRETYPE_VARINT as PY_WIRETYPE_VARINT, \
    WIRETYPE_FIXED32 as PY_WIRETYPE_FIXED32,\
    WIRETYPE_FIXED64 as PY_WIRETYPE_FIXED64,\
    WIRETYPE_LENGTH_DELIMITED as PY_WIRETYPE_LENGTH_DELIMITED
from ..wireconstants import ProtoWireConstants

cdef:
    uint32_t WIRETYPE_VARINT = PY_WIRETYPE_VARINT
    uint32_t WIRETYPE_FIXED32 = PY_WIRETYPE_FIXED32
    uint32_t WIRETYPE_FIXED64 = PY_WIRETYPE_FIXED64
    uint32_t WIRETYPE_LENGTH_DELIMITED = PY_WIRETYPE_LENGTH_DELIMITED

cdef:
    uint32_t WIRE_TUNNEL_META_COUNT = ProtoWireConstants.TUNNEL_META_COUNT
    uint32_t WIRE_TUNNEL_META_CHECKSUM = ProtoWireConstants.TUNNEL_META_CHECKSUM
    uint32_t WIRE_TUNNEL_END_RECORD = ProtoWireConstants.TUNNEL_END_RECORD

cdef:
    int64_t BOOL_TYPE_ID = types.boolean._type_id
    int64_t DATETIME_TYPE_ID = types.datetime._type_id
    int64_t DATE_TYPE_ID = types.date._type_id
    int64_t STRING_TYPE_ID = types.string._type_id
    int64_t FLOAT_TYPE_ID = types.float_._type_id
    int64_t DOUBLE_TYPE_ID = types.double._type_id
    int64_t BIGINT_TYPE_ID = types.bigint._type_id
    int64_t BINARY_TYPE_ID = types.binary._type_id
    int64_t TIMESTAMP_TYPE_ID = types.timestamp._type_id
    int64_t INTERVAL_DAY_TIME_TYPE_ID = types.interval_day_time._type_id
    int64_t INTERVAL_YEAR_MONTH_TYPE_ID = types.interval_year_month._type_id
    int64_t DECIMAL_TYPE_ID = types.Decimal._type_id

import_datetime()

cdef vector[int] data_type_to_wired_type

data_type_to_wired_type.resize(128, -1)
data_type_to_wired_type[BOOL_TYPE_ID] = WIRETYPE_VARINT
data_type_to_wired_type[DATETIME_TYPE_ID] = WIRETYPE_VARINT
data_type_to_wired_type[DATE_TYPE_ID] = WIRETYPE_VARINT
data_type_to_wired_type[BIGINT_TYPE_ID] = WIRETYPE_VARINT
data_type_to_wired_type[INTERVAL_YEAR_MONTH_TYPE_ID] = WIRETYPE_VARINT
data_type_to_wired_type[FLOAT_TYPE_ID] = WIRETYPE_FIXED32
data_type_to_wired_type[DOUBLE_TYPE_ID] = WIRETYPE_FIXED64
data_type_to_wired_type[STRING_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED
data_type_to_wired_type[BINARY_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED
data_type_to_wired_type[DECIMAL_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED
data_type_to_wired_type[TIMESTAMP_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED
data_type_to_wired_type[INTERVAL_DAY_TIME_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED


cdef class ProtobufRecordWriter:

    def __init__(self, object output, buffer_size=None):
        self.DEFAULT_BUFFER_SIZE = 4096
        self._output = output
        self._n_total = 0
        self._buffer_size = buffer_size or self.DEFAULT_BUFFER_SIZE
        self._encoder = CEncoder(self._buffer_size)
        self._last_flush_time = int(time.time())

    cpdef _re_init(self, output):
        self._encoder = CEncoder(self._buffer_size)
        self._output = output
        self._n_total = 0
        self._last_flush_time = int(time.time())

    def _mode(self):
        return 'c'

    @property
    def last_flush_time(self):
        return self._last_flush_time

    cpdef flush(self):
        if self._encoder.position() > 0:
            data = self._encoder.tostring()
            self._output.write(data)
            self._n_total += self._encoder.position()
            self._encoder = CEncoder(self._buffer_size)
            self._last_flush_time = int(time.time())

    cpdef close(self):
        self.flush_all()

    cpdef flush_all(self):
        self.flush()
        self._output.flush()

    cpdef int _refresh_buffer(self) except -1:
        """Control the buffer size of _encoder. Flush if necessary"""
        if self._encoder.position() > self._buffer_size:
            self.flush()

    @property
    def n_bytes(self):
        return self._n_total + self._encoder.position()

    def __len__(self):
        return self.n_bytes

    cdef int _write_tag(self, int field_num, int wire_type) nogil except +:
        return self._encoder.append_tag(field_num, wire_type)

    cdef int _write_raw_long(self, int64_t val) nogil except +:
        return self._encoder.append_sint64(val)

    cdef int _write_raw_int(self, int32_t val) nogil except +:
        return self._encoder.append_sint32(val)

    cdef int _write_raw_uint(self, uint32_t val) nogil except +:
        return self._encoder.append_uint32(val)

    cdef int _write_raw_bool(self, bint val) nogil except +:
        return self._encoder.append_bool(val)

    cdef int _write_raw_float(self, float val) nogil except +:
        return self._encoder.append_float(val)

    cdef int _write_raw_double(self, double val) nogil except +:
        return self._encoder.append_double(val)

    cdef int _write_raw_string(self, const char *ptr, uint32_t size) nogil except +:
        return self._encoder.append_string(ptr, size)


cdef class BaseRecordWriter(ProtobufRecordWriter):

    def __init__(self, object schema, object out, encoding='utf-8'):
        self._encoding = encoding
        self._is_utf8 = encoding == "utf-8"
        self._schema = schema
        self._columns = self._schema.columns
        self._schema_snapshot = self._schema.build_snapshot()

        self._crc_c = Checksum()
        self._crccrc_c = Checksum()
        self._curr_cursor_c = 0
        self._n_columns = len(self._columns)
        self._mills_converter = CMillisecondsConverter()
        self._to_days = utils.to_days

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
            object data_type

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
            self._crc_c.c_update_int(pb_index)

            data_type_id = self._schema_snapshot._col_type_ids[i]
            data_type = None
            if data_type_id >= 0 and data_type_to_wired_type[data_type_id] != -1:
                self._write_tag(pb_index, data_type_to_wired_type[data_type_id])
            else:
                data_type = self._schema_snapshot._col_types[i]
                if isinstance(data_type, (types.Array, types.Map, types.Struct)):
                    self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                else:
                    raise IOError('Invalid data type: %s' % data_type)

            self._write_field(val, data_type_id, data_type)

        self._refresh_buffer()

        checksum = <int32_t>self._crc_c.getvalue()
        self._write_tag(WIRE_TUNNEL_END_RECORD, WIRETYPE_VARINT)
        self._write_raw_uint(<uint32_t>checksum)
        self._crc_c.c_reset()
        self._crccrc_c.c_update_int(checksum)
        self._curr_cursor_c += 1

    cdef void _write_bool(self, bint data) nogil except +:
        self._crc_c.c_update_bool(data)
        self._write_raw_bool(data)

    cdef void _write_long(self, int64_t data) nogil except +:
        self._crc_c.c_update_long(data)
        self._write_raw_long(data)

    cdef void _write_float(self, float data) nogil except +:
        self._crc_c.c_update_float(data)
        self._write_raw_float(data)

    cdef void _write_double(self, double data) nogil except +:
        self._crc_c.c_update_double(data)
        self._write_raw_double(data)

    @cython.nonecheck(False)
    cdef _write_string(self, object data):
        cdef bytes bdata
        if type(data) is bytes:
            bdata = data
        elif self._is_utf8 and type(data) is unicode:
            bdata = (<unicode> data).encode("utf-8")
        elif isinstance(data, unicode):
            bdata = data.encode(self._encoding)
        else:
            bdata = bytes(data)
        self._crc_c.c_update(bdata, len(bdata))
        self._write_raw_string(bdata, len(bdata))

    cdef _write_timestamp(self, object data):
        cdef:
            object py_datetime = data.to_pydatetime(warn=False)
            long l_val
            int nanosecs

        l_val = self._mills_converter.to_milliseconds(py_datetime) // 1000
        nanosecs = data.microsecond * 1000 + data.nanosecond
        self._crc_c.c_update_long(l_val)
        self._write_raw_long(l_val)
        self._crc_c.c_update_int(nanosecs)
        self._write_raw_int(nanosecs)

    cdef _write_interval_day_time(self, object data):
        cdef:
            long l_val
            int nanosecs

        l_val = data.days * 24 * 3600 + data.seconds
        nanosecs = data.microseconds * 1000 + data.nanoseconds
        self._crc_c.c_update_long(l_val)
        self._write_raw_long(l_val)
        self._crc_c.c_update_int(nanosecs)
        self._write_raw_int(nanosecs)

    cdef _write_field(self, object val, int data_type_id, object data_type):
        cdef int64_t l_val

        if data_type_id == BIGINT_TYPE_ID:
            self._write_long(val)
        elif data_type_id == STRING_TYPE_ID:
            self._write_string(val)
        elif data_type_id == DOUBLE_TYPE_ID:
            self._write_double(val)
        elif data_type_id == FLOAT_TYPE_ID:
            self._write_float(val)
        elif data_type_id == DECIMAL_TYPE_ID:
            self._write_string(str(val))
        elif data_type_id == BINARY_TYPE_ID:
            self._write_string(val)
        elif data_type_id == BOOL_TYPE_ID:
            self._write_bool(val)
        elif data_type_id == DATETIME_TYPE_ID:
            l_val = self._mills_converter.to_milliseconds(val)
            self._write_long(l_val)
        elif data_type_id == DATE_TYPE_ID:
            self._write_long(self._to_days(val))
        elif data_type_id == TIMESTAMP_TYPE_ID:
            self._write_timestamp(val)
        elif data_type_id == INTERVAL_DAY_TIME_TYPE_ID:
            self._write_interval_day_time(val)
        elif data_type_id == INTERVAL_YEAR_MONTH_TYPE_ID:
            self._write_long(val.total_months())
        else:
            if isinstance(data_type, types.Array):
                self._write_raw_uint(len(val))
                self._write_array(val, data_type.value_type)
            elif isinstance(data_type, types.Map):
                self._write_raw_uint(len(val))
                self._write_array(compat.lkeys(val), data_type.key_type)
                self._write_raw_uint(len(val))
                self._write_array(compat.lvalues(val), data_type.value_type)
            elif isinstance(data_type, types.Struct):
                self._write_struct(val, data_type)
            else:
                raise IOError('Invalid data type: %s' % data_type)

    cdef _write_array(self, object data, object data_type):
        cdef int data_type_id = data_type._type_id
        for value in data:
            if value is None:
                self._write_raw_bool(True)
            else:
                self._write_raw_bool(False)
                self._write_field(value, data_type_id, data_type)

    cdef _write_struct(self, object data, object data_type):
        for key, value in six.iteritems(data):
            if value is None:
                self._write_raw_bool(True)
            else:
                self._write_raw_bool(False)
                field_type = data_type.field_types[key]
                self._write_field(value, field_type._type_id, field_type)

    @property
    def count(self):
        return self._curr_cursor_c

    @property
    def _curr_cursor(self):
        return self._curr_cursor_c

    @_curr_cursor.setter
    def _curr_cursor(self, value):
        self._curr_cursor_c = value

    cpdef _write_finish_tags(self):
        self._write_tag(WIRE_TUNNEL_META_COUNT, WIRETYPE_VARINT)
        self._write_raw_long(self._curr_cursor_c)
        self._write_tag(WIRE_TUNNEL_META_CHECKSUM, WIRETYPE_VARINT)
        self._write_raw_uint(<uint32_t>self._crccrc_c.c_getvalue())

    cpdef close(self):
        self._write_finish_tags()
        super(BaseRecordWriter, self).close()
        self._curr_cursor_c = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if an error occurs inside the with block, we do not commit
        if exc_val is not None:
            return
        self.close()
