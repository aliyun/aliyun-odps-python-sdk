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

cimport cython

import decimal
import json
import sys
import warnings
from collections import OrderedDict

from cpython.datetime cimport import_datetime
from libc.stdint cimport *
from libc.string cimport *

from ...lib.monotonic import monotonic

from ...src.types_c cimport BaseRecord
from ...src.utils_c cimport CMillisecondsConverter, to_date
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
    int64_t ARRAY_TYPE_ID = types.Array._type_id
    int64_t MAP_TYPE_ID = types.Map._type_id
    int64_t STRUCT_TYPE_ID = types.Struct._type_id

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
        self._partition_vals = []
        self._append_partitions = append_partitions

        partition_spec = PartitionSpec(partition_spec) if partition_spec is not None else None

        self._field_readers = [None] * self._schema_snapshot._col_count
        for idx, col_type in enumerate(self._schema_snapshot._col_types):
            self._field_readers[idx] = _build_field_reader(self, col_type)

        for i in range(self._n_columns):
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

    cdef int _set_record_list_value(self, list record, int i, object value) except? -1:
        record[i] = self._schema_snapshot.validate_value(i, value, MAX_READ_SIZE_LIMIT)
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
            (<AbstractFieldReader>self._field_readers[i]).read(rec_list, i)

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


cdef _build_field_reader(BaseTunnelRecordReader record_reader, object data_type):
    cdef int data_type_id = data_type._type_id

    import_datetime()

    if data_type_id == FLOAT_TYPE_ID:
        return FloatFieldReader(record_reader)
    elif data_type_id == BIGINT_TYPE_ID:
        return BigintFieldReader(record_reader)
    elif data_type_id == DOUBLE_TYPE_ID:
        return DoubleFieldReader(record_reader)
    elif data_type_id == STRING_TYPE_ID:
        return StringFieldReader(record_reader)
    elif data_type_id == BOOL_TYPE_ID:
        return BoolFieldReader(record_reader)
    elif data_type_id == DATETIME_TYPE_ID:
        return DatetimeFieldReader(record_reader)
    elif data_type_id == BINARY_TYPE_ID:
        return StringFieldReader(record_reader)
    elif data_type_id == TIMESTAMP_TYPE_ID:
        return TimestampFieldReader(record_reader)
    elif data_type_id == TIMESTAMP_NTZ_TYPE_ID:
        return TimestampNTZFieldReader(record_reader)
    elif data_type_id == DATE_TYPE_ID:
        return DateFieldReader(record_reader)
    elif data_type_id == INTERVAL_DAY_TIME_TYPE_ID:
        return IntervalDayTimeFieldReader(record_reader)
    elif data_type_id == INTERVAL_YEAR_MONTH_TYPE_ID:
        return IntervalYearMonthFieldReader(record_reader)
    elif data_type_id == JSON_TYPE_ID:
        return JsonFieldReader(record_reader)
    elif data_type_id == DECIMAL_TYPE_ID:
        return DecimalFieldReader(record_reader)
    elif data_type_id == ARRAY_TYPE_ID:
        return ArrayFieldReader(record_reader, data_type)
    elif data_type_id == MAP_TYPE_ID:
        return MapFieldReader(record_reader, data_type)
    elif data_type_id == STRUCT_TYPE_ID:
        return StructFieldReader(record_reader, data_type)
    elif isinstance(data_type, (types.Char, types.Varchar)):
        return StringFieldReader(record_reader)
    else:
        raise IOError("Unsupported type %s" % data_type)


cdef class AbstractFieldReader:
    need_validate = None

    cdef BaseTunnelRecordReader _record_reader
    cdef bint _need_validate

    def __init__(self, BaseTunnelRecordReader record_reader):
        self._record_reader = record_reader
        self._need_validate = self.need_validate

    cdef object _read_raw(self):
        raise NotImplementedError

    cdef inline int read(self, list dest, int idx) except? -1:
        if not self._need_validate:
            dest[idx] = self._read_raw()
        else:
            self._record_reader._set_record_list_value(
                dest, idx, self._read_raw()
            )
        return 0


cdef class BigintFieldReader(AbstractFieldReader):
    need_validate = False

    cdef object _read_raw(self):
        cdef int64_t val = self._record_reader._reader.read_sint64()
        self._record_reader._crc.c_update_long(val)
        return val


cdef class FloatFieldReader(AbstractFieldReader):
    need_validate = False

    cdef object _read_raw(self):
        cdef float val = self._record_reader._reader.read_float()
        self._record_reader._crc.c_update_float(val)
        return val


cdef class DoubleFieldReader(AbstractFieldReader):
    need_validate = False

    cdef object _read_raw(self):
        cdef double val = self._record_reader._reader.read_double()
        self._record_reader._crc.c_update_double(val)
        return val


cdef class BoolFieldReader(AbstractFieldReader):
    need_validate = False

    cdef object _read_raw(self):
        cdef bint val = self._record_reader._reader.read_bool()
        self._record_reader._crc.c_update_bool(val)
        return val


cdef class StringFieldReader(AbstractFieldReader):
    need_validate = True

    cdef object _read_raw(self):
        cdef bytes val = self._record_reader._reader.read_string()
        self._record_reader._crc.c_update(val, len(val))
        return val


cdef class DecimalFieldReader(AbstractFieldReader):
    need_validate = True

    cdef object _read_raw(self):
        cdef bytes val = self._record_reader._reader.read_string()
        self._record_reader._crc.c_update(val, len(val))
        return val


cdef class JsonFieldReader(AbstractFieldReader):
    need_validate = False

    cdef object _read_raw(self):
        cdef bytes val = self._record_reader._reader.read_string()
        self._record_reader._crc.c_update(val, len(val))
        return json.loads(val)


cdef class DatetimeFieldReader(AbstractFieldReader):
    need_validate = False

    cdef CMillisecondsConverter _mills_converter
    cdef bint _overflow_date_as_none

    def __init__(self, BaseTunnelRecordReader record_reader):
        super(DatetimeFieldReader, self).__init__(record_reader)
        self._mills_converter = CMillisecondsConverter()
        self._overflow_date_as_none = options.tunnel.overflow_date_as_none

    cdef object _read_raw(self):
        cdef int64_t val = self._record_reader._reader.read_sint64()
        self._record_reader._crc.c_update_long(val)
        try:
            return self._mills_converter.from_milliseconds(val)
        except DatetimeOverflowError:
            if not self._overflow_date_as_none:
                raise
            return None


cdef class DateFieldReader(AbstractFieldReader):
    need_validate = False

    cdef object _read_raw(self):
        cdef int64_t val = self._record_reader._reader.read_sint64()
        self._record_reader._crc.c_update_long(val)
        return to_date(val)


cdef class BaseTimestampFieldReader(AbstractFieldReader):
    need_validate = False
    _ntz = None

    cdef CMillisecondsConverter _mills_converter
    cdef bint _overflow_date_as_none

    def __init__(self, BaseTunnelRecordReader record_reader):
        super(BaseTimestampFieldReader, self).__init__(record_reader)
        self._overflow_date_as_none = options.tunnel.overflow_date_as_none
        if self._ntz:
            self._mills_converter = CMillisecondsConverter(local_tz=False)
        else:
            self._mills_converter = CMillisecondsConverter()

    cdef object _read_raw(self):
        cdef:
            int64_t val
            int32_t nano_secs

        global pd_timestamp, pd_timedelta

        if pd_timestamp is None:
            import pandas as pd
            pd_timestamp = pd.Timestamp
            pd_timedelta = pd.Timedelta

        val = self._record_reader._reader.read_sint64()
        self._record_reader._crc.c_update_long(val)
        nano_secs = self._record_reader._reader.read_sint32()
        self._record_reader._crc.c_update_int(nano_secs)

        try:
            return (
                pd_timestamp(self._mills_converter.from_milliseconds(val * 1000))
                + pd_timedelta(nanoseconds=nano_secs)
            )
        except DatetimeOverflowError:
            if not self._overflow_date_as_none:
                raise
            return None


cdef class TimestampFieldReader(BaseTimestampFieldReader):
    _ntz = False


cdef class TimestampNTZFieldReader(BaseTimestampFieldReader):
    _ntz = True


cdef class IntervalDayTimeFieldReader(AbstractFieldReader):
    need_validate = False

    cdef _read_raw(self):
        cdef:
            int64_t val
            int32_t nano_secs

        global pd_timestamp, pd_timedelta

        if pd_timedelta is None:
            import pandas as pd

            pd_timestamp = pd.Timestamp
            pd_timedelta = pd.Timedelta

        val = self._record_reader._reader.read_sint64()
        self._record_reader._crc.c_update_long(val)
        nano_secs = self._record_reader._reader.read_sint32()
        self._record_reader._crc.c_update_int(nano_secs)
        return pd_timedelta(seconds=val, nanoseconds=nano_secs)


cdef class IntervalYearMonthFieldReader(AbstractFieldReader):
    need_validate = False

    cdef _read_raw(self):
        cdef int64_t val = self._record_reader._reader.read_sint64()
        self._record_reader._crc.c_update_long(val)
        return compat.Monthdelta(val)


cdef class ArrayFieldReader(AbstractFieldReader):
    need_validate = True

    cdef AbstractFieldReader _element_reader

    def __init__(
        self, BaseTunnelRecordReader record_reader, object data_type
    ):
        super(ArrayFieldReader, self).__init__(record_reader)
        self._element_reader = _build_field_reader(record_reader, data_type.value_type)

    cdef _read_raw(self):
        cdef:
            uint32_t idx, size
            object val

        size = self._record_reader._reader.read_uint32()

        cdef list res = [None] * size
        for idx in range(size):
            if not self._record_reader._reader.read_bool():
                res[idx] = self._element_reader._read_raw()
        return res


cdef class MapFieldReader(AbstractFieldReader):
    need_validate = True

    cdef AbstractFieldReader _keys_reader, _values_reader
    cdef bint _use_ordered_dict

    def __init__(
        self, BaseTunnelRecordReader record_reader, object data_type
    ):
        super(MapFieldReader, self).__init__(record_reader)
        self._keys_reader = ArrayFieldReader(
            record_reader, types.Array(data_type.key_type)
        )
        self._values_reader = ArrayFieldReader(
            record_reader, types.Array(data_type.value_type)
        )
        if options.map_as_ordered_dict is None:
            self._use_ordered_dict = sys.version_info[:2] <= (3, 6)
        else:
            self._use_ordered_dict = options.map_as_ordered_dict

    cdef _read_raw(self):
        cdef list keys, values
        keys = self._keys_reader._read_raw()
        values = self._values_reader._read_raw()
        if self._use_ordered_dict:
            return OrderedDict(zip(keys, values))
        else:
            return dict(zip(keys, values))


cdef class StructFieldReader(AbstractFieldReader):
    need_validate = True

    cdef:
        bint _struct_as_dict
        list _field_readers
        list _field_keys
        list _field_types
        object _nt_type

    def __init__(
        self, BaseTunnelRecordReader record_reader, object data_type
    ):
        cdef int idx, field_count
        super(StructFieldReader, self).__init__(record_reader)

        self._struct_as_dict = options.struct_as_dict
        self._nt_type = data_type.namedtuple_type

        field_count = len(data_type.field_types)
        self._field_keys = [None] * field_count
        self._field_types = [None] * field_count
        self._field_readers = [None] * field_count
        for idx, (field_key, field_type) in enumerate(data_type.field_types.items()):
            self._field_keys[idx] = field_key
            self._field_types[idx] = field_type
            self._field_readers[idx] = _build_field_reader(record_reader, field_type)

    cdef _read_raw(self):
        cdef:
            list res_list = [None] * len(self._field_types)
            int idx
        for idx, field_type in enumerate(self._field_types):
            if not self._record_reader._reader.read_bool():
                res_list[idx] = (<AbstractFieldReader>self._field_readers[idx])._read_raw()
        if self._struct_as_dict:
            return OrderedDict(zip(self._field_keys, res_list))
        else:
            return self._nt_type(*res_list)


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
