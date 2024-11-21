# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

cimport cython

import decimal
import json
import warnings
from collections import OrderedDict

from cpython.datetime cimport import_datetime
from libc.stdint cimport *
from libc.string cimport *

from ...lib.monotonic import monotonic

from ...src.types_c cimport BaseRecord
from ...src.utils_c cimport CMillisecondsConverter
from ..checksum_c cimport Checksum
from ..pb.decoder_c cimport CDecoder

from ... import compat, options, types, utils
from ...errors import ChecksumError, DatetimeOverflowError
from ...models import Record
from ...readers import AbstractRecordReader  # noqa
from ...types import PartitionSpec
from ..pb import wire_format
from ..wireconstants import ProtoWireConstants

DEF MAX_READ_SIZE_LIMIT = (1 << 63) - 1
DEF MICRO_SEC_PER_SEC = 1_000_000L


cdef:
    uint32_t WIRE_TUNNEL_META_COUNT = ProtoWireConstants.TUNNEL_META_COUNT
    uint32_t WIRE_TUNNEL_META_CHECKSUM = ProtoWireConstants.TUNNEL_META_CHECKSUM
    uint32_t WIRE_TUNNEL_END_RECORD = ProtoWireConstants.TUNNEL_END_RECORD
    uint32_t WIRE_TUNNEL_END_METRICS = ProtoWireConstants.TUNNEL_END_METRICS

    uint32_t WIRETYPE_LENGTH_DELIMITED = wire_format.WIRETYPE_LENGTH_DELIMITED

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
    def __init__(
        self,
        object schema,
        object stream_creator,
        object columns=None,
        object partition_spec=None,
        bint append_partitions=True,
    ):
        cdef double ts

        self._enable_client_metrics = options.tunnel.enable_client_metrics
        self._server_metrics_string = None
        self._c_local_wall_time_ms = 0
        self._c_acc_network_time_ms = 0

        if self._enable_client_metrics:
            ts = monotonic()

        self._schema = schema
        if columns is None:
            self._columns = (
                self._schema.columns
                if append_partitions
                else self._schema.simple_columns
            )
        else:
            self._columns = [self._schema[c] for c in columns]
        self._reader_schema = types.OdpsSchema(columns=self._columns)
        self._schema_snapshot = self._reader_schema.build_snapshot()
        self._n_columns = len(self._columns)
        self._overflow_date_as_none = options.tunnel.overflow_date_as_none
        self._struct_as_dict = options.struct_as_dict
        self._partition_vals = []
        self._append_partitions = append_partitions

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

        if self._enable_client_metrics:
            self._c_local_wall_time_ms += <long>(
                MICRO_SEC_PER_SEC * (<double>monotonic() - ts)
            )

        self._curr_cursor = 0
        self._stream_creator = stream_creator
        self._reader = None
        self._reopen_reader()

        if self._enable_client_metrics:
            ts = monotonic()

        self._read_limit = -1 if options.table_read_limit is None else options.table_read_limit
        self._mills_converter = CMillisecondsConverter()
        self._mills_converter_utc = CMillisecondsConverter(local_tz=False)
        self._to_date = utils.to_date

        if self._enable_client_metrics:
            self._c_local_wall_time_ms += <long>(
                MICRO_SEC_PER_SEC * (<double>monotonic() - ts)
            )

        self._n_injected_error_cursor = -1
        self._injected_error_exc = None

    @cython.cdivision(True)
    def _reopen_reader(self):
        cdef object stream
        cdef double ts

        if self._enable_client_metrics:
            ts = monotonic()

        stream = self._stream_creator(self._curr_cursor)
        if self._enable_client_metrics:
            self._c_acc_network_time_ms += <long>(
                MICRO_SEC_PER_SEC * (<double>monotonic() - ts)
            )
            if self._reader is not None:
                self._c_acc_network_time_ms += (
                    self._reader._network_wall_time_ns // 1000
                )

        self._reader = CDecoder(stream, record_network_time=self._enable_client_metrics)
        self._last_n_bytes = self._reader.position() if self._curr_cursor != 0 else 0
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._attempt_row_count = 0

        if self._enable_client_metrics:
            self._c_local_wall_time_ms += <long>(
                MICRO_SEC_PER_SEC * (<double>monotonic() - ts)
            )

    def _inject_error(self, cursor, exc):
        self._n_injected_error_cursor = cursor
        self._injected_error_exc = exc

    def _mode(self):
        return "c"

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
            raise IOError("Unsupported type %s" % data_type)
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

    cdef _read(self):
        cdef:
            int index
            int checksum
            int idx_of_checksum
            int i
            int data_type_id
            int32_t wire_type
            object data_type
            BaseRecord record
            list rec_list

        if self._n_injected_error_cursor == self._curr_cursor:
            self._n_injected_error_cursor = -1
            raise self._injected_error_exc

        if self._curr_cursor >= self._read_limit > 0:
            warnings.warn(
                "Number of lines read via tunnel already reaches the limitation.",
                RuntimeWarning,
            )
            return None

        record = Record(schema=self._reader_schema, max_field_size=MAX_READ_SIZE_LIMIT)
        rec_list = record._c_values

        while True:
            index = self._reader.read_field_number(NULL)

            if index == 0:
                continue
            if index == WIRE_TUNNEL_END_RECORD:
                checksum = <int32_t>self._crc.c_getvalue()
                if self._reader.read_uint32() != <uint32_t>checksum:
                    raise ChecksumError("Checksum invalid")
                self._crc.c_reset()
                self._crccrc.c_update_int(checksum)
                break

            if index == WIRE_TUNNEL_META_COUNT:
                if self._attempt_row_count != self._reader.read_sint64():
                    raise IOError("count does not match")
                idx_of_checksum = self._reader.read_field_number(&wire_type)

                if WIRE_TUNNEL_META_CHECKSUM != idx_of_checksum:
                    if wire_type != WIRETYPE_LENGTH_DELIMITED:
                        raise IOError("Invalid stream data.")
                    self._crc.c_update_int(idx_of_checksum)

                    self._server_metrics_string = self._reader.read_string()
                    self._crc.c_update(
                        self._server_metrics_string, len(self._server_metrics_string)
                    )

                    idx_of_checksum = self._reader.read_field_number(NULL)
                    if idx_of_checksum != WIRE_TUNNEL_END_METRICS:
                        raise IOError("Invalid stream data.")
                    checksum = <int32_t>self._crc.c_getvalue()
                    if <uint32_t>checksum != self._reader.read_uint32():
                        raise ChecksumError("Checksum invalid.")
                    self._crc.reset()

                    idx_of_checksum = self._reader.read_field_number(NULL)
                if WIRE_TUNNEL_META_CHECKSUM != idx_of_checksum:
                    raise IOError("Invalid stream data.")
                if self._crccrc.c_getvalue() != self._reader.read_uint32():
                    raise ChecksumError("Checksum invalid.")

                return

            if index > self._n_columns:
                raise IOError(
                    "Invalid protobuf tag. Perhaps the datastream "
                    "from server is crushed."
                )

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
                    raise IOError("Unsupported type %s" % data_type)

        if self._append_partitions:
            for idx, val in self._partition_vals:
                rec_list[idx] = val

        self._attempt_row_count += 1
        self._curr_cursor += 1
        return record

    cpdef read(self):
        cdef:
            int retry_num = 0
            double ts
            object result

        if self._enable_client_metrics:
            ts = monotonic()
        while True:
            try:
                result = self._read()
                if self._enable_client_metrics:
                    self._c_local_wall_time_ms += <long>(
                        MICRO_SEC_PER_SEC * (<double>monotonic() - ts)
                    )
                return result
            except:
                retry_num += 1
                if retry_num > options.retry_times:
                    raise
                self._reopen_reader()

    def reads(self):
        return self.__iter__()

    @property
    def n_bytes(self):
        return self._last_n_bytes + self._reader.position()

    def get_total_bytes(self):
        return self.n_bytes

    @property
    def _local_wall_time_ms(self):
        return self._c_local_wall_time_ms

    @property
    @cython.cdivision(True)
    def _network_wall_time_ms(self):
        return self._reader._network_wall_time_ns // 1000 + self._c_acc_network_time_ms


cdef int DECIMAL_FRAC_CNT = 2
cdef int DECIMAL_INTG_CNT = 4
cdef int DECIMAL_PREC_CNT = DECIMAL_INTG_CNT + DECIMAL_FRAC_CNT
cdef int DECIMAL_DIG_NUMS = 9
cdef int DECIMAL_FRAC_DIGS = DECIMAL_DIG_NUMS * DECIMAL_FRAC_CNT
cdef int DECIMAL_INTG_DIGS = DECIMAL_DIG_NUMS * DECIMAL_INTG_CNT
cdef int DECIMAL_PREC_DIGS = DECIMAL_DIG_NUMS * DECIMAL_PREC_CNT


@cython.cdivision(True)
cdef inline int32_t decimal_print_dig(
    char* buf, const int32_t* val, int count, bint tail = False
) nogil:
    cdef char* src = buf - count * DECIMAL_DIG_NUMS if tail else buf + 1
    cdef char* ret = src
    cdef char* ptr
    cdef int32_t i, data, r
    for i in range(count):
        ptr = buf
        data = val[i]
        while data != 0:
            r = data // 10
            ptr[0] = data - r * 10 + ord("0")
            if ptr[0] != ord("0") and (not tail or ret[0] == ord("0")):
                ret = ptr
            data = r
            ptr -= 1
        buf -= DECIMAL_DIG_NUMS
    return src - ret if src >= ret else ret - src


cpdef convert_legacy_decimal_bytes(bytes value, int32_t frac = 0):
    """
    Legacy decimal memory layout:
        int8_t  mNull;
        int8_t  mSign;
        int8_t  mIntg;
        int8_t  mFrac; only 0, 1, 2
        int32_t mData[6];
        int8_t mPadding[4]; //For Memory Align
    """
    if value is None:
        return None

    cdef const char *src_ptr = <const char *>value
    cdef bint is_null = src_ptr[0]
    cdef bint sign = src_ptr[1]
    cdef int mintg = src_ptr[2]
    cdef int mfrac = src_ptr[3]
    cdef const char *data = src_ptr + 4
    cdef int32_t dec_cnt

    cdef char buf[9 * (2 + 4) + 4]
    cdef char *buf_ptr = buf
    memset(buf_ptr, ord("0"), sizeof(buf))

    if is_null:  # pragma: no cover
        return None
    if mintg + mfrac == 0:  # IsZero
        buf[1] = ord(".")
        dec_cnt = 20  # "0.000000000000000000"
        if frac > 0:
            dec_cnt = dec_cnt if frac + 2 > dec_cnt else frac + 2
        else:
            dec_cnt = 1
        return decimal.Decimal(buf[0:dec_cnt].decode())

    cdef int32_t icnt = decimal_print_dig(
        buf_ptr + DECIMAL_INTG_DIGS, <const int32_t *>data + DECIMAL_FRAC_CNT, mintg
    )
    cdef char *start = buf_ptr + DECIMAL_INTG_DIGS + 1 - icnt if icnt > 0 else buf_ptr + DECIMAL_INTG_DIGS

    if sign:
        start -= 1
        start[0] = ord("-")

    cdef int32_t fcnt = decimal_print_dig(
        buf_ptr + DECIMAL_PREC_DIGS + 1, <const int32_t *>data, DECIMAL_FRAC_CNT, True
    )
    if frac <= DECIMAL_FRAC_DIGS:
        frac = frac if frac > 0 else 0
    else:
        frac = DECIMAL_FRAC_DIGS
    fcnt = max(fcnt, frac)
    buf[DECIMAL_INTG_DIGS + 1] = ord(".")

    dec_cnt = buf_ptr + DECIMAL_INTG_DIGS + 1 - start + (fcnt + 1 if fcnt > 0 else 0)
    return decimal.Decimal(start[0:dec_cnt].decode())
