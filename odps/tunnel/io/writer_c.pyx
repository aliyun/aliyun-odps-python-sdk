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

import json
import time

import cython

from cpython.datetime cimport import_datetime
from libc.stdint cimport *
from libc.string cimport *
from libcpp.vector cimport vector

from ...src.types_c cimport BaseRecord, SchemaSnapshot
from ..checksum_c cimport Checksum
from ..pb.encoder_c cimport CEncoder

from ... import compat, types, utils
from ...config import options
from ...lib.monotonic import monotonic
from ...src.utils_c cimport CMillisecondsConverter, to_days
from ..pb.wire_format import WIRETYPE_FIXED32 as PY_WIRETYPE_FIXED32
from ..pb.wire_format import WIRETYPE_FIXED64 as PY_WIRETYPE_FIXED64
from ..pb.wire_format import WIRETYPE_LENGTH_DELIMITED as PY_WIRETYPE_LENGTH_DELIMITED
from ..pb.wire_format import WIRETYPE_VARINT as PY_WIRETYPE_VARINT
from ..wireconstants import ProtoWireConstants

DEF MICRO_SEC_PER_SEC = 1_000_000L

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
    int64_t JSON_TYPE_ID = types.Json._type_id
    int64_t TIMESTAMP_NTZ_TYPE_ID = types.timestamp_ntz._type_id
    int64_t ARRAY_TYPE_ID = types.Array._type_id
    int64_t MAP_TYPE_ID = types.Map._type_id
    int64_t STRUCT_TYPE_ID = types.Struct._type_id

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
data_type_to_wired_type[JSON_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED
data_type_to_wired_type[TIMESTAMP_NTZ_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED
data_type_to_wired_type[ARRAY_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED
data_type_to_wired_type[MAP_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED
data_type_to_wired_type[STRUCT_TYPE_ID] = WIRETYPE_LENGTH_DELIMITED


cdef class ProtobufRecordWriter:

    def __init__(self, object output, buffer_size=None):
        self.DEFAULT_BUFFER_SIZE = 4096
        self._output = output
        self._n_total = 0
        self._buffer_size = buffer_size or self.DEFAULT_BUFFER_SIZE
        self._encoder = CEncoder(self._buffer_size)

    cpdef _re_init(self, output):
        self._encoder = CEncoder(self._buffer_size)
        self._output = output
        self._n_total = 0

    def _mode(self):
        return "c"

    cpdef flush(self):
        if self._encoder.position() > 0:
            data = self._encoder.tostring()
            self._output.write(data)
            self._n_total += self._encoder.position()
            self._encoder = CEncoder(self._buffer_size)

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

    cdef int _write_tag(self, int field_num, int wire_type) except -1 nogil:
        return self._encoder.append_tag(field_num, wire_type)

    cdef int _write_raw_long(self, int64_t val) except -1 nogil:
        return self._encoder.append_sint64(val)

    cdef int _write_raw_int(self, int32_t val) except -1 nogil:
        return self._encoder.append_sint32(val)

    cdef int _write_raw_uint(self, uint32_t val) except -1 nogil:
        return self._encoder.append_uint32(val)

    cdef int _write_raw_bool(self, bint val) except -1 nogil:
        return self._encoder.append_bool(val)

    cdef int _write_raw_float(self, float val) except -1 nogil:
        return self._encoder.append_float(val)

    cdef int _write_raw_double(self, double val) except -1 nogil:
        return self._encoder.append_double(val)

    cdef int _write_raw_string(self, const char *ptr, uint32_t size) except -1 nogil:
        return self._encoder.append_string(ptr, size)


cdef class BaseRecordWriter(ProtobufRecordWriter):

    def __init__(self, object schema, object out, encoding="utf-8"):
        cdef double ts
        cdef int idx

        self._c_enable_client_metrics = options.tunnel.enable_client_metrics
        self._c_local_wall_time_ms = 0

        if self._c_enable_client_metrics:
            ts = monotonic()

        self._encoding = encoding
        self._is_utf8 = encoding == "utf-8"
        self._schema = schema
        self._columns = self._schema.columns
        self._schema_snapshot = self._schema.build_snapshot()

        self._crc_c = Checksum()
        self._crccrc_c = Checksum()
        self._curr_cursor_c = 0
        self._n_columns = len(self._columns)

        super(BaseRecordWriter, self).__init__(out)

        if self._c_enable_client_metrics:
            self._c_local_wall_time_ms += <long>(
                MICRO_SEC_PER_SEC * (<double>monotonic() - ts)
            )

        import_datetime()

        self._field_writers = [None] * self._schema_snapshot._col_count
        for idx, col_type in enumerate(self._schema_snapshot._col_types):
            self._field_writers[idx] = _build_field_writer(self, col_type)

    @property
    def _crc(self):
        return self._crc_c

    @property
    def _crccrc(self):
        return self._crccrc_c

    @property
    def _local_wall_time_ms(self):
        return self._c_local_wall_time_ms

    @_local_wall_time_ms.setter
    def _local_wall_time_ms(self, val):
        self._c_local_wall_time_ms = val

    cpdef write(self, BaseRecord record):
        cdef:
            int n_record_fields
            int pb_index
            int i
            int data_type_id
            int checksum
            object val
            object data_type
            double ts

        if self._c_enable_client_metrics:
            ts = monotonic()

        n_record_fields = len(record)

        if n_record_fields > self._n_columns:
            raise IOError("record fields count is more than schema.")

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
                raise IOError("Invalid data type: %s" % data_type)

            (<AbstractFieldWriter>self._field_writers[i]).write(val)

        self._refresh_buffer()

        checksum = <int32_t>self._crc_c.getvalue()
        self._write_tag(WIRE_TUNNEL_END_RECORD, WIRETYPE_VARINT)
        self._write_raw_uint(<uint32_t>checksum)
        self._crc_c.c_reset()
        self._crccrc_c.c_update_int(checksum)
        self._curr_cursor_c += 1

        if self._c_enable_client_metrics:
            self._c_local_wall_time_ms += <long>(
                MICRO_SEC_PER_SEC * (<double>monotonic() - ts)
            )

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


def _build_field_writer(BaseRecordWriter record_writer, object field_type):
    cdef int data_type_id = field_type._type_id

    if data_type_id == BIGINT_TYPE_ID:
        return LongFieldWriter(record_writer)
    elif data_type_id == STRING_TYPE_ID:
        return StringFieldWriter(record_writer)
    elif data_type_id == DOUBLE_TYPE_ID:
        return DoubleFieldWriter(record_writer)
    elif data_type_id == FLOAT_TYPE_ID:
        return FloatFieldWriter(record_writer)
    elif data_type_id == DECIMAL_TYPE_ID:
        return DecimalFieldWriter(record_writer)
    elif data_type_id == BINARY_TYPE_ID:
        return StringFieldWriter(record_writer)
    elif data_type_id == BOOL_TYPE_ID:
        return BoolFieldWriter(record_writer)
    elif data_type_id == DATETIME_TYPE_ID:
        return DatetimeFieldWriter(record_writer)
    elif data_type_id == DATE_TYPE_ID:
        return DateFieldWriter(record_writer)
    elif data_type_id == TIMESTAMP_TYPE_ID:
        return TimestampFieldWriter(record_writer)
    elif data_type_id == TIMESTAMP_NTZ_TYPE_ID:
        return TimestampNTZFieldWriter(record_writer)
    elif data_type_id == INTERVAL_DAY_TIME_TYPE_ID:
        return IntervalDayTimeFieldWriter(record_writer)
    elif data_type_id == INTERVAL_YEAR_MONTH_TYPE_ID:
        return IntervalYearMonthFieldWriter(record_writer)
    elif data_type_id == JSON_TYPE_ID:
        return JsonFieldWriter(record_writer)
    elif data_type_id == ARRAY_TYPE_ID:
        return ArrayFieldWriter(record_writer, field_type)
    elif data_type_id == MAP_TYPE_ID:
        return MapFieldWriter(record_writer, field_type)
    elif data_type_id == STRUCT_TYPE_ID:
        return StructFieldWriter(record_writer, field_type)
    else:
        raise IOError("Invalid data type: %s" % field_type)


cdef class AbstractFieldWriter:
    cdef BaseRecordWriter _record_writer

    def __init__(self, BaseRecordWriter record_writer):
        self._record_writer = record_writer

    cdef int write(self, object val) except -1:
        raise NotImplementedError


cdef class BoolFieldWriter(AbstractFieldWriter):
    cdef int write(self, object val) except -1:
        cdef bint bval = val
        self._record_writer._crc_c.c_update_bool(bval)
        self._record_writer._write_raw_bool(bval)
        return 0


cdef class LongFieldWriter(AbstractFieldWriter):
    cdef inline int _write(self, int64_t ival) except -1 nogil:
        self._record_writer._crc_c.c_update_long(ival)
        self._record_writer._write_raw_long(ival)
        return 0

    cdef int write(self, object val) except -1:
        cdef int64_t ival = val
        return self._write(ival)


cdef class FloatFieldWriter(AbstractFieldWriter):
    cdef int write(self, object val) except -1:
        cdef float fval = val
        self._record_writer._crc_c.c_update_float(fval)
        self._record_writer._write_raw_float(fval)
        return 0


cdef class DoubleFieldWriter(AbstractFieldWriter):
    cdef int write(self, object val) except -1:
        cdef double dblval = val
        self._record_writer._crc_c.c_update_double(dblval)
        self._record_writer._write_raw_double(dblval)
        return 0


cdef class StringFieldWriter(AbstractFieldWriter):
    @cython.nonecheck(False)
    cdef int write(self, object val) except -1:
        cdef bytes bdata
        cdef const char *pbytes
        cdef int32_t data_size

        if type(val) is bytes:
            pbytes = val
            data_size = len(<bytes>val)
        elif self._record_writer._is_utf8 and type(val) is unicode:
            bdata = (<unicode> val).encode("utf-8")
            pbytes = bdata
            data_size = len(bdata)
        elif isinstance(val, unicode):
            bdata = val.encode(self._record_writer._encoding)
            pbytes = bdata
            data_size = len(bdata)
        else:
            bdata = bytes(val)
            pbytes = bdata
            data_size = len(bdata)

        self._record_writer._crc_c.c_update(pbytes, data_size)
        self._record_writer._write_raw_string(pbytes, data_size)
        return 0


cdef class DecimalFieldWriter(AbstractFieldWriter):
    cdef StringFieldWriter _string_writer

    def __init__(self, BaseRecordWriter record_writer):
        super(DecimalFieldWriter, self).__init__(record_writer)
        self._string_writer = StringFieldWriter(record_writer)

    cdef int write(self, object val) except -1:
        return self._string_writer.write(str(val))


cdef class JsonFieldWriter(AbstractFieldWriter):
    cdef StringFieldWriter _string_writer

    def __init__(self, BaseRecordWriter record_writer):
        super(JsonFieldWriter, self).__init__(record_writer)
        self._string_writer = StringFieldWriter(record_writer)

    cdef int write(self, object val) except -1:
        return self._string_writer.write(json.dumps(val))


cdef class DatetimeFieldWriter(AbstractFieldWriter):
    cdef:
        LongFieldWriter _long_writer
        CMillisecondsConverter _mills_converter

    def __init__(self, BaseRecordWriter record_writer):
        super(DatetimeFieldWriter, self).__init__(record_writer)
        self._long_writer = LongFieldWriter(record_writer)
        self._mills_converter = CMillisecondsConverter()

    cdef int write(self, object val) except -1:
        cdef int64_t l_val = self._mills_converter.to_milliseconds(val)
        return self._long_writer._write(l_val)


cdef class DateFieldWriter(AbstractFieldWriter):
    cdef:
        LongFieldWriter _long_writer

    def __init__(self, BaseRecordWriter record_writer):
        super(DateFieldWriter, self).__init__(record_writer)
        self._long_writer = LongFieldWriter(record_writer)

    cdef int write(self, object val) except -1:
        cdef int64_t l_val = to_days(val)
        return self._long_writer._write(l_val)


cdef class BaseTimestampFieldWriter(AbstractFieldWriter):
    cdef CMillisecondsConverter _mills_converter

    def __init__(self, BaseRecordWriter record_writer):
        super(BaseTimestampFieldWriter, self).__init__(record_writer)
        self._mills_converter = self._build_mills_converter()

    def _build_mills_converter(self):
        raise NotImplementedError

    cdef int _write(self, object val, bint ntz) except -1:
        cdef:
            object py_datetime = val.to_pydatetime(warn=False)
            long l_val
            int nanosecs

        l_val = self._mills_converter.to_milliseconds(py_datetime) // 1000
        nanosecs = val.microsecond * 1000 + val.nanosecond
        self._record_writer._crc_c.c_update_long(l_val)
        self._record_writer._write_raw_long(l_val)
        self._record_writer._crc_c.c_update_int(nanosecs)
        self._record_writer._write_raw_int(nanosecs)
        return 0


cdef class TimestampFieldWriter(BaseTimestampFieldWriter):
    def _build_mills_converter(self):
        return CMillisecondsConverter()

    cdef int write(self, object val) except -1:
        return self._write(val, False)


cdef class TimestampNTZFieldWriter(BaseTimestampFieldWriter):
    def _build_mills_converter(self):
        return CMillisecondsConverter(local_tz=False)

    cdef int write(self, object val) except -1:
        return self._write(val, True)


cdef class IntervalDayTimeFieldWriter(AbstractFieldWriter):
    cdef int write(self, object val) except -1:
        cdef:
            long l_val
            int nanosecs

        l_val = val.days * 24 * 3600 + val.seconds
        nanosecs = val.microseconds * 1000 + val.nanoseconds
        self._record_writer._crc_c.c_update_long(l_val)
        self._record_writer._write_raw_long(l_val)
        self._record_writer._crc_c.c_update_int(nanosecs)
        self._record_writer._write_raw_int(nanosecs)
        return 0


cdef class IntervalYearMonthFieldWriter(AbstractFieldWriter):
    cdef LongFieldWriter _long_writer

    def __init__(self, BaseRecordWriter record_writer):
        super(IntervalYearMonthFieldWriter, self).__init__(record_writer)
        self._long_writer = LongFieldWriter(record_writer)

    cdef int write(self, object val) except -1:
        cdef int64_t l_val = val.total_months()
        return self._long_writer._write(l_val)


cdef class ArrayFieldWriter(AbstractFieldWriter):
    cdef AbstractFieldWriter _element_writer

    def __init__(self, BaseRecordWriter record_writer, object data_type):
        super(ArrayFieldWriter, self).__init__(record_writer)
        self._element_writer = _build_field_writer(record_writer, data_type.value_type)

    cdef int write(self, object val) except -1:
        cdef object elem

        if type(val) is not list:  # not likely
            val = list(val)
        self._record_writer._write_raw_uint(len(<list>val))
        for elem in <list>val:
            if elem is None:
                self._record_writer._write_raw_bool(True)
            else:
                self._record_writer._write_raw_bool(False)
                self._element_writer.write(elem)
        return 0


cdef class MapFieldWriter(AbstractFieldWriter):
    cdef ArrayFieldWriter _keys_writer, _values_writer

    def __init__(self, BaseRecordWriter record_writer, object data_type):
        super(MapFieldWriter, self).__init__(record_writer)
        self._keys_writer = ArrayFieldWriter(record_writer, types.Array(data_type.key_type))
        self._values_writer = ArrayFieldWriter(record_writer, types.Array(data_type.value_type))

    cdef int write(self, object val) except -1:
        cdef object keys, values

        if type(val) is dict:
            keys = (<dict>val).keys()
            values = (<dict>val).values()
        else:
            keys = val.keys()
            values = val.values()
        self._keys_writer.write(keys)
        self._values_writer.write(values)
        return 0


cdef class StructFieldWriter(AbstractFieldWriter):
    cdef:
        list _field_keys
        list _field_types
        list _field_writers

    def __init__(self, BaseRecordWriter record_writer, object data_type):
        cdef int idx, field_count
        super(StructFieldWriter, self).__init__(record_writer)

        field_count = len(data_type.field_types)
        self._field_keys = [None] * field_count
        self._field_types = [None] * field_count
        self._field_writers = [None] * field_count
        for idx, (field_key, field_type) in enumerate(data_type.field_types.items()):
            self._field_keys[idx] = field_key
            self._field_types[idx] = field_type
            self._field_writers[idx] = _build_field_writer(record_writer, field_type)

    cdef int write(self, object val) except -1:
        cdef:
            int idx
            tuple tp_val
            list list_vals
            object elem

        if type(val) is tuple:
            tp_val = <tuple>val
        elif isinstance(val, tuple):
            tp_val = tuple(val)
        elif type(val) is dict:
            list_vals = [None] * len(<dict>val)
            for idx, key in enumerate(self._field_keys):
                list_vals[idx] = (<dict>val)[key]
            tp_val = tuple(list_vals)
        elif isinstance(val, dict):
            list_vals = [None] * len(val)
            for idx, key in enumerate(self._field_keys):
                list_vals[idx] = val[key]
            tp_val = tuple(list_vals)
        else:
            raise TypeError("Cannot write %s as struct", type(val))

        for idx, elem in enumerate(tp_val):
            if elem is None:
                self._record_writer._write_raw_bool(True)
            else:
                self._record_writer._write_raw_bool(False)
                (<AbstractFieldWriter>self._field_writers[idx]).write(elem)

        return 0
