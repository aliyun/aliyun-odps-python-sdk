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

import sys
import warnings
from libc.stdint cimport *
from libc.string cimport *

from ...src.types_c cimport BaseRecord
from ...src.utils_c cimport CMillisecondsConverter
from ..pb.decoder_c cimport Decoder
from ..checksum_c cimport Checksum

from ... import utils, types, compat, options
from ...compat import six
from ...models import Record
from ...readers import AbstractRecordReader
from ..wireconstants import ProtoWireConstants


cdef:
    uint32_t WIRE_TUNNEL_META_COUNT = ProtoWireConstants.TUNNEL_META_COUNT
    uint32_t WIRE_TUNNEL_META_CHECKSUM = ProtoWireConstants.TUNNEL_META_CHECKSUM
    uint32_t WIRE_TUNNEL_END_RECORD = ProtoWireConstants.TUNNEL_END_RECORD

cdef:
    object pd_timestamp = None
    object pd_timedelta = None


cdef class BaseTunnelRecordReader:
    def __init__(self, object schema, object input_stream, columns=None):
        self._last_error = None
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
            elif data_type == types.date:
                self._column_setters[i] = self._set_date
            elif data_type == types.string:
                self._column_setters[i] = self._set_string
            elif data_type == types.float_:
                self._column_setters[i] = self._set_float
            elif data_type == types.double:
                self._column_setters[i] = self._set_double
            elif data_type in types.integer_types:
                self._column_setters[i] = self._set_bigint
            elif data_type == types.binary:
                self._column_setters[i] = self._set_string
            elif data_type == types.timestamp:
                self._column_setters[i] = self._set_timestamp
            elif data_type == types.interval_day_time:
                self._column_setters[i] = self._set_interval_day_time
            elif data_type == types.interval_year_month:
                self._column_setters[i] = self._set_bigint
            elif isinstance(data_type, types.Decimal):
                self._column_setters[i] = self._set_decimal
            elif isinstance(data_type, (types.Char, types.Varchar)):
                self._column_setters[i] = self._set_string
            else:
                self._column_setters[i] = NULL

        self._reader = Decoder(input_stream)
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._curr_cursor = 0
        self._read_limit = -1 if options.table_read_limit is None else options.table_read_limit
        self._mills_converter = CMillisecondsConverter()
        self._to_date = utils.to_date

    def _mode(self):
        return 'c'

    @property
    def count(self):
        return self._curr_cursor

    cdef object _read_struct(self, object value_type):
        res = compat.OrderedDict()
        for k in value_type.field_types:
            if self._reader.read_bool():
                res[k] = None
            else:
                res[k] = self._read_element(value_type.field_types[k])
        return res

    cdef object _read_element(self, object data_type):
        if data_type == types.float_:
            val = self._read_float()
        elif data_type == types.double:
            val = self._read_double()
        elif data_type == types.boolean:
            val = self._read_bool()
        elif data_type in types.integer_types:
            val = self._read_bigint()
        elif data_type == types.string:
            val = self._read_string()
        elif data_type == types.datetime:
            val = self._read_datetime()
        elif data_type == types.binary:
            val = self._read_string()
        elif data_type == types.timestamp:
            val = self._read_timestamp()
        elif data_type == types.interval_day_time:
            val = self._read_interval_day_time()
        elif data_type == types.interval_year_month:
            val = compat.Monthdelta(self._read_bigint())
        elif isinstance(data_type, (types.Char, types.Varchar)):
            val = self._read_string()
        elif isinstance(data_type, types.Decimal):
            val = self._read_string()
            self._crc.update(val)
        elif isinstance(data_type, types.Array):
            val = self._read_array(data_type.value_type)
        elif isinstance(data_type, types.Map):
            keys = self._read_array(data_type.key_type)
            values = self._read_array(data_type.value_type)
            val = compat.OrderedDict(zip(keys, values))
        elif isinstance(data_type, types.Struct):
            val = self._read_struct(data_type)
        else:
            raise IOError('Unsupported type %s' % data_type)
        return val

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
                res.append(self._read_element(value_type))
        return res

    cdef bytes _read_string(self):
        cdef bytes val
        try:
            val = self._reader.read_string()
        except:
            self._last_error = sys.exc_info()
            raise
        self._crc.update(val)

        return val

    cdef float _read_float(self):
        cdef float val
        try:
            val = self._reader.read_float()
        except:
            self._last_error = sys.exc_info()
            raise
        self._crc.c_update_float(val)

        return val

    cdef double _read_double(self):
        cdef double val
        try:
            val = self._reader.read_double()
        except:
            self._last_error = sys.exc_info()
            raise
        self._crc.c_update_double(val)

        return val

    cdef bint _read_bool(self):
        cdef bint val
        try:
            val = self._reader.read_bool()
        except:
            self._last_error = sys.exc_info()
            raise
        self._crc.c_update_bool(val)

        return val

    cdef int64_t _read_bigint(self):
        cdef int64_t val
        try:
            val = self._reader.read_sint64()
        except:
            self._last_error = sys.exc_info()
            raise
        self._crc.c_update_long(val)

        return val

    cdef object _read_datetime(self):
        cdef int64_t val
        try:
            val = self._reader.read_sint64()
        except:
            self._last_error = sys.exc_info()
            raise
        self._crc.c_update_long(val)
        return self._mills_converter.from_milliseconds(val)

    cdef object _read_date(self):
        cdef int64_t val
        try:
            val = self._reader.read_sint64()
        except:
            self._last_error = sys.exc_info()
            raise
        self._crc.c_update_long(val)
        return self._to_date(val)

    cdef object _read_timestamp(self):
        cdef:
            int64_t val
            int32_t nano_secs
        global pd_timestamp, pd_timedelta

        if pd_timestamp is None:
            try:
                import pandas as pd
                pd_timestamp = pd.Timestamp
                pd_timedelta = pd.Timedelta
            except ImportError:
                self._last_error = sys.exc_info()
                raise
        try:
            val = self._reader.read_sint64()
            self._crc.c_update_long(val)
            nano_secs = self._reader.read_sint32()
            self._crc.c_update_int(nano_secs)
        except:
            self._last_error = sys.exc_info()
            raise
        return pd_timestamp(self._mills_converter.from_milliseconds(val * 1000)) + pd_timedelta(nanoseconds=nano_secs)

    cdef object _read_interval_day_time(self):
        cdef:
            int64_t val
            int32_t nano_secs
        global pd_timestamp, pd_timedelta

        if pd_timedelta is None:
            try:
                import pandas as pd
                pd_timestamp = pd.Timestamp
                pd_timedelta = pd.Timedelta
            except ImportError:
                self._last_error = sys.exc_info()
                raise
        try:
            val = self._reader.read_sint64()
            self._crc.c_update_long(val)
            nano_secs = self._reader.read_sint32()
            self._crc.c_update_int(nano_secs)
        except:
            self._last_error = sys.exc_info()
            raise
        return pd_timedelta(seconds=val, nanoseconds=nano_secs)

    cdef void _set_record_list_value(self, list record, int i, object value):
        record[i] = self._schema_snapshot.validate_value(i, value)

    cdef void _set_string(self, list record, int i):
        cdef object val = self._read_string()
        self._set_record_list_value(record, i, val)

    cdef void _set_float(self, list record, int i):
        cdef float val = self._read_float()
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

    cdef void _set_date(self, list record, int i):
        cdef object val = self._read_date()
        self._set_record_list_value(record, i, val)

    cdef void _set_decimal(self, list record, int i):
        cdef bytes val
        val = self._reader.read_string()
        self._crc.update(val)
        self._set_record_list_value(record, i, val)

    cdef void _set_timestamp(self, list record, int i):
        cdef object val = self._read_timestamp()
        self._set_record_list_value(record, i, val)

    cdef void _set_interval_day_time(self, list record, int i):
        cdef object val = self._read_interval_day_time()
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

            self._crc.c_update_int(index)

            i = index - 1
            if self._column_setters[i] != NULL:
                self._column_setters[i](self, rec_list, i)
            else:
                data_type = self._schema_snapshot._col_types[i]
                if isinstance(data_type, types.Array):
                    val = self._read_array(data_type.value_type)
                    rec_list[i] = self._schema_snapshot.validate_value(i, val)
                elif isinstance(data_type, types.Map):
                    keys = self._read_array(data_type.key_type)
                    values = self._read_array(data_type.value_type)
                    val = compat.OrderedDict(zip(keys, values))
                    rec_list[i] = self._schema_snapshot.validate_value(i, val)
                elif isinstance(data_type, types.Struct):
                    val = self._read_struct(data_type)
                    rec_list[i] = self._schema_snapshot.validate_value(i, val)
                else:
                    raise IOError('Unsupported type %s' % data_type)

            if self._last_error is not None:
                six.reraise(*self._last_error)

        self._curr_cursor += 1
        return record

    def reads(self):
        return self.__iter__()

    @property
    def n_bytes(self):
        return self._reader.position()

    def get_total_bytes(self):
        return self.n_bytes

    def close(self):
        if hasattr(self._schema, 'close'):
            self._schema.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class TunnelRecordReader(BaseTunnelRecordReader, AbstractRecordReader):
    def __next__(self):
        record = self.read()
        if record is None:
            raise StopIteration
        return record

    next = __next__
