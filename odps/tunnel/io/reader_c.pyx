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

import json
import warnings
from collections import OrderedDict
from cpython.datetime cimport import_datetime
from libc.stdint cimport *
from libc.string cimport *

from ...src.types_c cimport BaseRecord
from ...src.utils_c cimport CMillisecondsConverter
from ..pb.decoder_c cimport CDecoder
from ..checksum_c cimport Checksum

from ... import utils, types, compat, options
from ...errors import DatetimeOverflowError
from ...models import Record
from ...readers import AbstractRecordReader
from ...types import PartitionSpec
from ..wireconstants import ProtoWireConstants

cdef int64_t MAX_READ_SIZE_LIMIT = (1 << 63) - 1


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

cdef:
    object pd_timestamp = None
    object pd_timedelta = None

import_datetime()

cdef class BaseTunnelRecordReader:
    def __init__(self, object schema, object input_stream, columns=None, partition_spec=None):
        self._schema = schema
        if columns is None:
            self._columns = self._schema.columns
        else:
            self._columns = [self._schema[c] for c in columns]
        self._reader_schema = types.OdpsSchema(columns=self._columns)
        self._schema_snapshot = self._reader_schema.build_snapshot()
        self._n_columns = len(self._columns)
        self._overflow_date_as_none = options.tunnel.overflow_date_as_none
        self._struct_as_dict = options.struct_as_dict
        self._partition_vals = []

        partition_spec = PartitionSpec(partition_spec) if partition_spec is not None else None

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
            elif data_type == types.timestamp_ntz:
                self._column_setters[i] = self._set_timestamp_ntz
            elif data_type == types.interval_day_time:
                self._column_setters[i] = self._set_interval_day_time
            elif data_type == types.interval_year_month:
                self._column_setters[i] = self._set_interval_year_month
            elif data_type == types.json:
                self._column_setters[i] = self._set_json
            elif isinstance(data_type, types.Decimal):
                self._column_setters[i] = self._set_decimal
            elif isinstance(data_type, (types.Char, types.Varchar)):
                self._column_setters[i] = self._set_string
            else:
                self._column_setters[i] = NULL
            if partition_spec is not None and self._columns[i].name in partition_spec:
                self._partition_vals.append((i, partition_spec[self._columns[i].name]))

        self._reader = CDecoder(input_stream)
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._curr_cursor = 0
        self._read_limit = -1 if options.table_read_limit is None else options.table_read_limit
        self._mills_converter = CMillisecondsConverter()
        self._mills_converter_utc = CMillisecondsConverter(local_tz=False)
        self._to_date = utils.to_date

    def _mode(self):
        return 'c'

    @property
    def count(self):
        return self._curr_cursor

    cdef object _read_struct(self, object value_type):
        cdef:
            list res_list = [None] * len(value_type.field_types)
            int idx
        for idx, field_type in enumerate(value_type.field_types.values()):
            if not self._reader.read_bool():
                res_list[idx] = self._read_element(field_type._type_id, field_type)
        if self._struct_as_dict:
            return OrderedDict(zip(value_type.field_types.keys(), res_list))
        else:
            return value_type.namedtuple_type(*res_list)

    cdef object _read_element(self, int data_type_id, object data_type):
        if data_type_id == FLOAT_TYPE_ID:
            val = self._read_float()
        elif data_type_id == BIGINT_TYPE_ID:
            val = self._read_bigint()
        elif data_type_id == DOUBLE_TYPE_ID:
            val = self._read_double()
        elif data_type_id == STRING_TYPE_ID:
            val = self._read_string()
        elif data_type_id == BOOL_TYPE_ID:
            val = self._read_bool()
        elif data_type_id == DATETIME_TYPE_ID:
            val = self._read_datetime()
        elif data_type_id == BINARY_TYPE_ID:
            val = self._read_string()
        elif data_type_id == TIMESTAMP_TYPE_ID:
            val = self._read_timestamp()
        elif data_type_id == TIMESTAMP_NTZ_TYPE_ID:
            val = self._read_timestamp_ntz()
        elif data_type_id == INTERVAL_DAY_TIME_TYPE_ID:
            val = self._read_interval_day_time()
        elif data_type_id == INTERVAL_YEAR_MONTH_TYPE_ID:
            val = compat.Monthdelta(self._read_bigint())
        elif data_type_id == JSON_TYPE_ID:
            val = json.loads(self._read_string())
        elif data_type_id == DECIMAL_TYPE_ID:
            val = self._read_string()
        elif isinstance(data_type, (types.Char, types.Varchar)):
            val = self._read_string()
        elif isinstance(data_type, types.Array):
            val = self._read_array(data_type.value_type)
        elif isinstance(data_type, types.Map):
            keys = self._read_array(data_type.key_type)
            values = self._read_array(data_type.value_type)
            val = OrderedDict(zip(keys, values))
        elif isinstance(data_type, types.Struct):
            val = self._read_struct(data_type)
        else:
            raise IOError('Unsupported type %s' % data_type)
        return val

    cdef list _read_array(self, object value_type):
        cdef:
            uint32_t size
            int value_type_id = value_type._type_id
            object val

        size = self._reader.read_uint32()

        cdef list res = [None] * size
        for idx in range(size):
            if not self._reader.read_bool():
                res[idx] = self._read_element(value_type_id, value_type)
        return res

    cdef bytes _read_string(self):
        cdef bytes val = self._reader.read_string()
        self._crc.c_update(val, len(val))
        return val

    cdef float _read_float(self) except? -1.0 nogil:
        cdef float val = self._reader.read_float()
        self._crc.c_update_float(val)
        return val

    cdef double _read_double(self) except? -1.0 nogil:
        cdef double val = self._reader.read_double()
        self._crc.c_update_double(val)
        return val

    cdef bint _read_bool(self) except? False nogil:
        cdef bint val = self._reader.read_bool()
        self._crc.c_update_bool(val)
        return val

    cdef int64_t _read_bigint(self) except? -1 nogil:
        cdef int64_t val = self._reader.read_sint64()
        self._crc.c_update_long(val)
        return val

    cdef object _read_datetime(self):
        cdef int64_t val = self._reader.read_sint64()
        self._crc.c_update_long(val)
        try:
            return self._mills_converter.from_milliseconds(val)
        except DatetimeOverflowError:
            if not self._overflow_date_as_none:
                raise
            return None

    cdef object _read_date(self):
        cdef int64_t val = self._reader.read_sint64()
        self._crc.c_update_long(val)
        return self._to_date(val)

    cdef object _read_timestamp_base(self, bint ntz):
        cdef:
            int64_t val
            int32_t nano_secs
            CMillisecondsConverter converter

        global pd_timestamp, pd_timedelta

        if pd_timestamp is None:
            import pandas as pd
            pd_timestamp = pd.Timestamp
            pd_timedelta = pd.Timedelta

        val = self._reader.read_sint64()
        self._crc.c_update_long(val)
        nano_secs = self._reader.read_sint32()
        self._crc.c_update_int(nano_secs)

        if ntz:
            converter = self._mills_converter_utc
        else:
            converter = self._mills_converter

        try:
            return (
                pd_timestamp(converter.from_milliseconds(val * 1000))
                + pd_timedelta(nanoseconds=nano_secs)
            )
        except DatetimeOverflowError:
            if not self._overflow_date_as_none:
                raise
            return None

    cdef object _read_timestamp(self):
        return self._read_timestamp_base(False)

    cdef object _read_timestamp_ntz(self):
        return self._read_timestamp_base(True)

    cdef object _read_interval_day_time(self):
        cdef:
            int64_t val
            int32_t nano_secs
        global pd_timestamp, pd_timedelta

        if pd_timedelta is None:
            import pandas as pd
            pd_timestamp = pd.Timestamp
            pd_timedelta = pd.Timedelta
        val = self._reader.read_sint64()
        self._crc.c_update_long(val)
        nano_secs = self._reader.read_sint32()
        self._crc.c_update_int(nano_secs)
        return pd_timedelta(seconds=val, nanoseconds=nano_secs)

    cdef int _set_record_list_value(self, list record, int i, object value) except? -1:
        record[i] = self._schema_snapshot.validate_value(i, value, MAX_READ_SIZE_LIMIT)
        return 0

    cdef int _set_string(self, list record, int i) except? -1:
        cdef object val = self._read_string()
        self._set_record_list_value(record, i, val)
        return 0

    cdef int _set_float(self, list record, int i) except? -1:
        record[i] = self._read_float()
        return 0

    cdef int _set_double(self, list record, int i) except? -1:
        record[i] = self._read_double()
        return 0

    cdef int _set_bool(self, list record, int i) except? -1:
        record[i] = self._read_bool()
        return 0

    cdef int _set_bigint(self, list record, int i) except? -1:
        record[i] = self._read_bigint()
        return 0

    cdef int _set_datetime(self, list record, int i) except? -1:
        record[i] = self._read_datetime()
        return 0

    cdef int _set_date(self, list record, int i) except? -1:
        record[i] = self._read_date()
        return 0

    cdef int _set_decimal(self, list record, int i) except? -1:
        cdef bytes val = self._reader.read_string()
        self._crc.c_update(val, len(val))
        self._set_record_list_value(record, i, val)
        return 0

    cdef int _set_timestamp(self, list record, int i) except? -1:
        record[i] = self._read_timestamp()
        return 0

    cdef int _set_timestamp_ntz(self, list record, int i) except? -1:
        record[i] = self._read_timestamp_ntz()
        return 0

    cdef int _set_interval_day_time(self, list record, int i) except? -1:
        record[i] = self._read_interval_day_time()
        return 0

    cdef int _set_interval_year_month(self, list record, int i) except? -1:
        cdef int64_t val = self._read_bigint()
        self._set_record_list_value(record, i, compat.Monthdelta(val))
        return 0

    cdef int _set_json(self, list record, int i) except? -1:
        cdef bytes val = self._read_string()
        self._set_record_list_value(record, i, json.loads(val))
        return 0

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
            warnings.warn(
                'Number of lines read via tunnel already reaches the limitation.',
                RuntimeWarning,
            )
            return None

        record = Record(schema=self._reader_schema, max_field_size=MAX_READ_SIZE_LIMIT)
        rec_list = record._c_values

        while True:
            index = self._reader.read_field_number()

            if index == 0:
                continue
            if index == WIRE_TUNNEL_END_RECORD:
                checksum = <int32_t>self._crc.c_getvalue()
                if self._reader.read_uint32() != <uint32_t>checksum:
                    raise IOError('Checksum invalid')
                self._crc.c_reset()
                self._crccrc.c_update_int(checksum)
                break

            if index == WIRE_TUNNEL_META_COUNT:
                if self._curr_cursor != self._reader.read_sint64():
                    raise IOError('count does not match')
                idx_of_checksum = self._reader.read_field_number()
                if WIRE_TUNNEL_META_CHECKSUM != idx_of_checksum:
                    raise IOError('Invalid stream data.')
                if self._crccrc.c_getvalue() != self._reader.read_uint32():
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
                    rec_list[i] = self._schema_snapshot.validate_value(i, val, MAX_READ_SIZE_LIMIT)
                elif isinstance(data_type, types.Map):
                    keys = self._read_array(data_type.key_type)
                    values = self._read_array(data_type.value_type)
                    val = OrderedDict(zip(keys, values))
                    rec_list[i] = self._schema_snapshot.validate_value(i, val, MAX_READ_SIZE_LIMIT)
                elif isinstance(data_type, types.Struct):
                    val = self._read_struct(data_type)
                    rec_list[i] = self._schema_snapshot.validate_value(i, val, MAX_READ_SIZE_LIMIT)
                else:
                    raise IOError('Unsupported type %s' % data_type)

        for idx, val in self._partition_vals:
            rec_list[idx] = val

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

    @property
    def schema(self):
        return self._schema
