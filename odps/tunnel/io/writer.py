#!/usr/bin/env python
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

import json
import struct

try:
    import pyarrow as pa
except (AttributeError, ImportError):
    pa = None
try:
    import pyarrow.compute as pac
except (AttributeError, ImportError):
    pac = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except (ImportError, ValueError):
    pd = None

from ... import compat, options, types, utils
from ...compat import Enum, futures, six
from ...lib.monotonic import monotonic
from ..base import TunnelMetrics
from ..checksum import Checksum
from ..errors import TunnelError
from ..pb.encoder import Encoder
from ..pb.wire_format import (
    WIRETYPE_FIXED32,
    WIRETYPE_FIXED64,
    WIRETYPE_LENGTH_DELIMITED,
    WIRETYPE_VARINT,
)
from ..wireconstants import ProtoWireConstants
from .stream import RequestsIO, get_compress_stream
from .types import odps_schema_to_arrow_schema

try:
    if not options.force_py:
        from ..hasher_c import RecordHasher
        from .writer_c import BaseRecordWriter
    else:
        from ..hasher import RecordHasher

        BaseRecordWriter = None
except ImportError as e:
    if options.force_c:
        raise e

    from ..hasher import RecordHasher

    BaseRecordWriter = None

MICRO_SEC_PER_SEC = 1000000

varint_tag_types = types.integer_types + (
    types.boolean,
    types.datetime,
    types.date,
    types.interval_year_month,
)
length_delim_tag_types = (
    types.string,
    types.binary,
    types.timestamp,
    types.timestamp_ntz,
    types.interval_day_time,
    types.json,
)


if BaseRecordWriter is None:

    class ProtobufWriter(object):
        """
        ProtobufWriter is a stream-interface wrapper around encoder_c.Encoder(c)
        and encoder.Encoder(py)
        """

        DEFAULT_BUFFER_SIZE = 4096

        def __init__(self, output, buffer_size=None):
            self._encoder = Encoder()
            self._output = output
            self._buffer_size = buffer_size or self.DEFAULT_BUFFER_SIZE
            self._n_total = 0

        def _re_init(self, output):
            self._encoder = Encoder()
            self._output = output
            self._n_total = 0

        def _mode(self):
            return "py"

        def flush(self):
            if len(self._encoder) > 0:
                data = self._encoder.tostring()
                self._output.write(data)
                self._n_total += len(self._encoder)
                self._encoder = Encoder()

        def close(self):
            self.flush_all()

        def flush_all(self):
            self.flush()
            self._output.flush()

        def _refresh_buffer(self):
            """Control the buffer size of _encoder. Flush if necessary"""
            if len(self._encoder) > self._buffer_size:
                self.flush()

        @property
        def n_bytes(self):
            return self._n_total + len(self._encoder)

        def __len__(self):
            return self.n_bytes

        def _write_tag(self, field_num, wire_type):
            self._encoder.append_tag(field_num, wire_type)
            self._refresh_buffer()

        def _write_raw_long(self, val):
            self._encoder.append_sint64(val)
            self._refresh_buffer()

        def _write_raw_int(self, val):
            self._encoder.append_sint32(val)
            self._refresh_buffer()

        def _write_raw_uint(self, val):
            self._encoder.append_uint32(val)
            self._refresh_buffer()

        def _write_raw_bool(self, val):
            self._encoder.append_bool(val)
            self._refresh_buffer()

        def _write_raw_float(self, val):
            self._encoder.append_float(val)
            self._refresh_buffer()

        def _write_raw_double(self, val):
            self._encoder.append_double(val)
            self._refresh_buffer()

        def _write_raw_string(self, val):
            self._encoder.append_string(val)
            self._refresh_buffer()

    class BaseRecordWriter(ProtobufWriter):
        def __init__(self, schema, out, encoding="utf-8"):
            self._encoding = encoding
            self._schema = schema
            self._columns = self._schema.columns
            self._crc = Checksum()
            self._crccrc = Checksum()
            self._curr_cursor = 0
            self._to_milliseconds = utils.MillisecondsConverter().to_milliseconds
            self._to_milliseconds_utc = utils.MillisecondsConverter(
                local_tz=False
            ).to_milliseconds
            self._to_days = utils.to_days

            self._enable_client_metrics = options.tunnel.enable_client_metrics
            self._local_wall_time_ms = 0

            super(BaseRecordWriter, self).__init__(out)

        def write(self, record):
            n_record_fields = len(record)
            n_columns = len(self._columns)

            if self._enable_client_metrics:
                ts = monotonic()

            if n_record_fields > n_columns:
                raise IOError("record fields count is more than schema.")

            for i in range(min(n_record_fields, n_columns)):
                if self._schema.is_partition(self._columns[i]):
                    continue

                val = record[i]
                if val is None:
                    continue

                pb_index = i + 1
                self._crc.update_int(pb_index)

                data_type = self._columns[i].type
                if data_type in varint_tag_types:
                    self._write_tag(pb_index, WIRETYPE_VARINT)
                elif data_type == types.float_:
                    self._write_tag(pb_index, WIRETYPE_FIXED32)
                elif data_type == types.double:
                    self._write_tag(pb_index, WIRETYPE_FIXED64)
                elif data_type in length_delim_tag_types:
                    self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                elif isinstance(
                    data_type,
                    (
                        types.Char,
                        types.Varchar,
                        types.Decimal,
                        types.Array,
                        types.Map,
                        types.Struct,
                    ),
                ):
                    self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                else:
                    raise IOError("Invalid data type: %s" % data_type)
                self._write_field(val, data_type)

            checksum = utils.long_to_int(self._crc.getvalue())
            self._write_tag(ProtoWireConstants.TUNNEL_END_RECORD, WIRETYPE_VARINT)
            self._write_raw_uint(utils.long_to_uint(checksum))
            self._crc.reset()
            self._crccrc.update_int(checksum)
            self._curr_cursor += 1

            if self._enable_client_metrics:
                self._local_wall_time_ms += int(MICRO_SEC_PER_SEC * (monotonic() - ts))

        def _write_bool(self, data):
            self._crc.update_bool(data)
            self._write_raw_bool(data)

        def _write_long(self, data):
            self._crc.update_long(data)
            self._write_raw_long(data)

        def _write_float(self, data):
            self._crc.update_float(data)
            self._write_raw_float(data)

        def _write_double(self, data):
            self._crc.update_double(data)
            self._write_raw_double(data)

        def _write_string(self, data):
            if isinstance(data, six.text_type):
                data = data.encode(self._encoding)
            self._crc.update(data)
            self._write_raw_string(data)

        def _write_timestamp(self, data, ntz=False):
            to_mills = self._to_milliseconds_utc if ntz else self._to_milliseconds
            t_val = int(to_mills(data.to_pydatetime(warn=False)) / 1000)
            nano_val = data.microsecond * 1000 + data.nanosecond
            self._crc.update_long(t_val)
            self._write_raw_long(t_val)
            self._crc.update_int(nano_val)
            self._write_raw_int(nano_val)

        def _write_interval_day_time(self, data):
            t_val = data.days * 3600 * 24 + data.seconds
            nano_val = data.microseconds * 1000 + data.nanoseconds
            self._crc.update_long(t_val)
            self._write_raw_long(t_val)
            self._crc.update_int(nano_val)
            self._write_raw_int(nano_val)

        def _write_array(self, data, data_type):
            for value in data:
                if value is None:
                    self._write_raw_bool(True)
                else:
                    self._write_raw_bool(False)
                    self._write_field(value, data_type)

        def _write_struct(self, data, data_type):
            if isinstance(data, dict):
                vals = [None] * len(data)
                for idx, key in enumerate(data_type.field_types.keys()):
                    vals[idx] = data[key]
                data = tuple(vals)
            for value, typ in zip(data, data_type.field_types.values()):
                if value is None:
                    self._write_raw_bool(True)
                else:
                    self._write_raw_bool(False)
                    self._write_field(value, typ)

        def _write_field(self, val, data_type):
            if data_type == types.boolean:
                self._write_bool(val)
            elif data_type == types.datetime:
                val = self._to_milliseconds(val)
                self._write_long(val)
            elif data_type == types.date:
                val = self._to_days(val)
                self._write_long(val)
            elif data_type == types.float_:
                self._write_float(val)
            elif data_type == types.double:
                self._write_double(val)
            elif data_type in types.integer_types:
                self._write_long(val)
            elif data_type == types.string:
                self._write_string(val)
            elif data_type == types.binary:
                self._write_string(val)
            elif data_type == types.timestamp or data_type == types.timestamp_ntz:
                self._write_timestamp(val, ntz=data_type == types.timestamp_ntz)
            elif data_type == types.interval_day_time:
                self._write_interval_day_time(val)
            elif data_type == types.interval_year_month:
                self._write_long(val.total_months())
            elif isinstance(data_type, (types.Char, types.Varchar)):
                self._write_string(val)
            elif isinstance(data_type, types.Decimal):
                self._write_string(str(val))
            elif isinstance(data_type, types.Json):
                self._write_string(json.dumps(val))
            elif isinstance(data_type, types.Array):
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
                raise IOError("Invalid data type: %s" % data_type)

        @property
        def count(self):
            return self._curr_cursor

        def _write_finish_tags(self):
            self._write_tag(ProtoWireConstants.TUNNEL_META_COUNT, WIRETYPE_VARINT)
            self._write_raw_long(self.count)
            self._write_tag(ProtoWireConstants.TUNNEL_META_CHECKSUM, WIRETYPE_VARINT)
            self._write_raw_uint(utils.long_to_uint(self._crccrc.getvalue()))

        def close(self):
            self._write_finish_tags()
            super(BaseRecordWriter, self).close()
            self._curr_cursor = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # if an error occurs inside the with block, we do not commit
            if exc_val is not None:
                return
            self.close()


class RecordWriter(BaseRecordWriter):
    """
    This writer uploads the output of serializer asynchronously within a long-lived http connection.
    """

    def __init__(
        self, schema, request_callback, compress_option=None, encoding="utf-8"
    ):
        self._enable_client_metrics = options.tunnel.enable_client_metrics
        self._server_metrics_string = None
        if self._enable_client_metrics:
            ts = monotonic()

        self._req_io = RequestsIO(
            request_callback,
            chunk_size=options.chunk_size,
            record_io_time=self._enable_client_metrics,
        )

        out = get_compress_stream(self._req_io, compress_option)
        super(RecordWriter, self).__init__(schema, out, encoding=encoding)
        self._req_io.start()

        if self._enable_client_metrics:
            self._local_wall_time_ms += int(MICRO_SEC_PER_SEC * (monotonic() - ts))

    @property
    def metrics(self):
        if self._server_metrics_string is None:
            return None
        return TunnelMetrics.from_server_json(
            type(self).__name__,
            self._server_metrics_string,
            self._local_wall_time_ms,
            self._req_io.io_time_ms,
        )

    def write(self, record):
        if self._req_io._async_err:
            ex_type, ex_value, tb = self._req_io._async_err
            six.reraise(ex_type, ex_value, tb)
        super(RecordWriter, self).write(record)

    def close(self):
        if self._enable_client_metrics:
            ts = monotonic()

        super(RecordWriter, self).close()
        resp = self._req_io.finish()

        if self._enable_client_metrics:
            self._local_wall_time_ms += int(MICRO_SEC_PER_SEC * (monotonic() - ts))

        self._server_metrics_string = resp.headers.get("odps-tunnel-metrics")

    def get_total_bytes(self):
        return self.n_bytes


class BufferedRecordWriter(BaseRecordWriter):
    """
    This writer buffers the output of serializer. When the buffer exceeds a fixed-size of limit
     (default 20 MiB), it uploads the buffered output within one http connection.
    """

    def __init__(
        self,
        schema,
        request_callback,
        compress_option=None,
        encoding="utf-8",
        buffer_size=None,
        block_id=None,
        block_id_gen=None,
    ):
        self._request_callback = request_callback
        self._block_id = block_id or 0
        self._blocks_written = []
        self._buffer = compat.BytesIO()
        self._n_bytes_written = 0
        self._compress_option = compress_option
        self._block_id_gen = block_id_gen

        self._enable_client_metrics = options.tunnel.enable_client_metrics
        self._server_metrics_string = None
        self._network_wall_time_ms = 0
        if not self._enable_client_metrics:
            self._accumulated_metrics = None
        else:
            self._accumulated_metrics = TunnelMetrics(type(self).__name__)
            ts = monotonic()

        out = get_compress_stream(self._buffer, compress_option)
        super(BufferedRecordWriter, self).__init__(schema, out, encoding=encoding)

        # make sure block buffer size is applied correctly here
        self._buffer_size = buffer_size or options.tunnel.block_buffer_size

        if self._enable_client_metrics:
            self._local_wall_time_ms += int(MICRO_SEC_PER_SEC * (monotonic() - ts))

    @property
    def cur_block_id(self):
        return self._block_id

    def _get_next_block_id(self):
        if callable(self._block_id_gen):
            return self._block_id_gen()
        return self._block_id + 1

    def write(self, record):
        super(BufferedRecordWriter, self).write(record)
        if 0 < self._buffer_size < self._n_raw_bytes:
            self._flush()

    def close(self):
        if self._n_raw_bytes > 0:
            self._flush()
        self.flush_all()
        self._buffer.close()

    def _collect_metrics(self):
        if self._enable_client_metrics:
            if self._server_metrics_string is not None:
                self._accumulated_metrics += TunnelMetrics.from_server_json(
                    type(self).__name__,
                    self._server_metrics_string,
                    self._local_wall_time_ms,
                    self._network_wall_time_ms,
                )
            self._server_metrics_string = None
            self._local_wall_time_ms = 0
            self._network_wall_time_ms = 0

    def _reset_writer(self, write_response):
        self._collect_metrics()
        if self._enable_client_metrics:
            ts = monotonic()

        self._buffer = compat.BytesIO()

        out = get_compress_stream(self._buffer, self._compress_option)
        self._re_init(out)
        self._curr_cursor = 0
        self._crccrc.reset()
        self._crc.reset()

        if self._enable_client_metrics:
            self._local_wall_time_ms += int(MICRO_SEC_PER_SEC * (monotonic() - ts))

    def _send_buffer(self):
        if self._enable_client_metrics:
            ts = monotonic()
        resp = self._request_callback(self._block_id, self._buffer.getvalue())
        if self._enable_client_metrics:
            self._network_wall_time_ms += int(MICRO_SEC_PER_SEC * (monotonic() - ts))
        return resp

    def _flush(self):
        if self._enable_client_metrics:
            ts = monotonic()

        self._write_finish_tags()
        self._n_bytes_written += self._n_raw_bytes
        self.flush_all()

        resp = self._send_buffer()
        self._server_metrics_string = resp.headers.get("odps-tunnel-metrics")
        self._blocks_written.append(self._block_id)
        self._block_id = self._get_next_block_id()

        if self._enable_client_metrics:
            self._local_wall_time_ms += int(MICRO_SEC_PER_SEC * (monotonic() - ts))

        self._reset_writer(resp)

    @property
    def metrics(self):
        return self._accumulated_metrics

    @property
    def _n_raw_bytes(self):
        return super(BufferedRecordWriter, self).n_bytes

    @property
    def n_bytes(self):
        return self._n_bytes_written + self._n_raw_bytes

    def get_total_bytes(self):
        return self.n_bytes

    def get_blocks_written(self):
        return self._blocks_written


# make sure original typo class also referable
BufferredRecordWriter = BufferedRecordWriter


class StreamRecordWriter(BufferedRecordWriter):
    def __init__(
        self,
        schema,
        request_callback,
        session,
        slot,
        compress_option=None,
        encoding="utf-8",
        buffer_size=None,
    ):
        self.session = session
        self.slot = slot
        self._record_count = 0

        super(StreamRecordWriter, self).__init__(
            schema,
            request_callback,
            compress_option=compress_option,
            encoding=encoding,
            buffer_size=buffer_size,
        )

    def write(self, record):
        super(StreamRecordWriter, self).write(record)
        self._record_count += 1

    def _reset_writer(self, write_response):
        self._record_count = 0
        slot_server = write_response.headers["odps-tunnel-routed-server"]
        slot_num = int(write_response.headers["odps-tunnel-slot-num"])
        self.session.reload_slots(self.slot, slot_server, slot_num)
        super(StreamRecordWriter, self)._reset_writer(write_response)

    def _send_buffer(self):
        def gen():  # synchronize chunk upload
            data = self._buffer.getvalue()
            chunk_size = options.chunk_size
            while data:
                to_send = data[:chunk_size]
                data = data[chunk_size:]
                yield to_send

        return self._request_callback(gen())


class BaseArrowWriter(object):
    def __init__(self, schema, out=None, chunk_size=None):
        if pa is None:
            raise ValueError("To use arrow writer you need to install pyarrow")

        self._schema = schema
        self._arrow_schema = odps_schema_to_arrow_schema(schema)
        self._chunk_size = chunk_size or options.chunk_size
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._cur_chunk_size = 0

        self._output = out
        self._chunk_size_written = False

    def _re_init(self, output):
        self._output = output
        self._chunk_size_written = False
        self._cur_chunk_size = 0

    def _write_chunk_size(self):
        self._write_uint32(self._chunk_size)

    def _write_uint32(self, val):
        data = struct.pack("!I", utils.long_to_uint(val))
        self._output.write(data)

    def _write_chunk(self, buf):
        if not self._chunk_size_written:
            self._write_chunk_size()
            self._chunk_size_written = True
        self._output.write(buf)
        self._crc.update(buf)
        self._crccrc.update(buf)
        self._cur_chunk_size += len(buf)
        if self._cur_chunk_size >= self._chunk_size:
            checksum = self._crc.getvalue()
            self._write_uint32(checksum)
            self._crc.reset()
            self._cur_chunk_size = 0

    @classmethod
    def _localize_timezone(cls, col, tz=None):
        from ...lib import tzlocal

        if tz is None:
            if options.local_timezone is True or options.local_timezone is None:
                tz = str(tzlocal.get_localzone())
            elif options.local_timezone is False:
                tz = "UTC"
            else:
                tz = str(options.local_timezone)

        if col.type.tz is not None:
            return col
        if hasattr(pac, "assume_timezone") and isinstance(tz, str):
            # pyarrow.compute.assume_timezone only accepts
            # string-represented zones
            col = pac.assume_timezone(col, timezone=tz)
            return col
        else:
            pd_col = col.to_pandas().dt.tz_localize(tz)
            return pa.Array.from_pandas(pd_col)

    def write(self, data):
        if isinstance(data, pd.DataFrame):
            arrow_data = pa.RecordBatch.from_pandas(data)
        elif isinstance(data, (pa.Table, pa.RecordBatch)):
            arrow_data = data
        else:
            raise TypeError("Cannot support writing data type %s", type(data))

        assert isinstance(arrow_data, (pa.RecordBatch, pa.Table))

        if arrow_data.schema != self._arrow_schema or any(
            isinstance(tp, pa.TimestampType) for tp in arrow_data.schema.types
        ):
            type_dict = dict(zip(arrow_data.schema.names, arrow_data.schema.types))
            column_dict = dict(zip(arrow_data.schema.names, arrow_data.columns))
            arrays = []
            for name, tp in zip(self._arrow_schema.names, self._arrow_schema.types):
                if name not in column_dict:
                    raise ValueError(
                        "Input record batch does not contain column %s" % name
                    )

                if isinstance(tp, pa.TimestampType):
                    if self._schema[name].type == types.timestamp_ntz:
                        col = self._localize_timezone(column_dict[name], "UTC")
                    else:
                        col = self._localize_timezone(column_dict[name])
                    column_dict[name] = col.cast(pa.timestamp(tp.unit, col.type.tz))

                if tp == type_dict[name]:
                    arrays.append(column_dict[name])
                else:
                    try:
                        arrays.append(column_dict[name].cast(tp, safe=False))
                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                        raise ValueError(
                            "Failed to cast column %s to type %s" % (name, tp)
                        )
            pa_type = type(arrow_data)
            arrow_data = pa_type.from_arrays(arrays, names=self._arrow_schema.names)

        if isinstance(arrow_data, pa.RecordBatch):
            batches = [arrow_data]
        else:  # pa.Table
            batches = arrow_data.to_batches()

        for batch in batches:
            data = batch.serialize().to_pybytes()
            written_bytes = 0
            while written_bytes < len(data):
                length = min(
                    self._chunk_size - self._cur_chunk_size, len(data) - written_bytes
                )
                chunk_data = data[written_bytes : written_bytes + length]
                self._write_chunk(chunk_data)
                written_bytes += length

    def _write_finish_tags(self):
        checksum = self._crccrc.getvalue()
        self._write_uint32(checksum)
        self._crccrc.reset()

    def flush(self):
        self._output.flush()

    def _finish(self):
        self._write_finish_tags()
        self._output.flush()

    def close(self):
        self._finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if an error occurs inside the with block, we do not commit
        if exc_val is not None:
            return
        self.close()


class ArrowWriter(BaseArrowWriter):
    def __init__(self, schema, request_callback, compress_option=None, chunk_size=None):
        self._req_io = RequestsIO(request_callback, chunk_size=chunk_size)

        out = get_compress_stream(self._req_io, compress_option)
        super(ArrowWriter, self).__init__(schema, out, chunk_size)
        self._req_io.start()

    def _finish(self):
        super(ArrowWriter, self)._finish()
        self._req_io.finish()


class BufferedArrowWriter(BaseArrowWriter):
    def __init__(
        self,
        schema,
        request_callback,
        compress_option=None,
        buffer_size=None,
        chunk_size=None,
        block_id=None,
        block_id_gen=None,
    ):
        self._buffer_size = buffer_size or options.tunnel.block_buffer_size
        self._request_callback = request_callback
        self._block_id = block_id or 0
        self._blocks_written = []
        self._buffer = compat.BytesIO()
        self._compress_option = compress_option
        self._n_bytes_written = 0
        self._block_id_gen = block_id_gen

        out = get_compress_stream(self._buffer, compress_option)
        super(BufferedArrowWriter, self).__init__(schema, out, chunk_size=chunk_size)

    @property
    def cur_block_id(self):
        return self._block_id

    def _get_next_block_id(self):
        if callable(self._block_id_gen):
            return self._block_id_gen()
        return self._block_id + 1

    def write(self, data):
        super(BufferedArrowWriter, self).write(data)
        if 0 < self._buffer_size < self._n_raw_bytes:
            self._flush()

    def close(self):
        if self._n_raw_bytes > 0:
            self._flush()
        self._finish()
        self._buffer.close()

    def _reset_writer(self):
        self._buffer = compat.BytesIO()

        out = get_compress_stream(self._buffer, self._compress_option)
        self._re_init(out)
        self._crccrc.reset()
        self._crc.reset()

    def _send_buffer(self):
        return self._request_callback(self._block_id, self._buffer.getvalue())

    def _flush(self):
        self._write_finish_tags()
        self._n_bytes_written += self._n_raw_bytes

        self._send_buffer()
        self._blocks_written.append(self._block_id)
        self._block_id = self._get_next_block_id()

        self._reset_writer()

    @property
    def _n_raw_bytes(self):
        return self._buffer.tell()

    @property
    def n_bytes(self):
        return self._n_bytes_written + self._n_raw_bytes

    def get_total_bytes(self):
        return self.n_bytes

    def get_blocks_written(self):
        return self._blocks_written


class Upsert(object):
    DEFAULT_MAX_BUFFER_SIZE = 64 * 1024**2
    DEFAULT_SLOT_BUFFER_SIZE = 1024**2

    class Operation(Enum):
        UPSERT = "UPSERT"
        DELETE = "DELETE"

    class Status(Enum):
        NORMAL = "NORMAL"
        ERROR = "ERROR"
        CLOSED = "CLOSED"

    def __init__(
        self,
        schema,
        request_callback,
        session,
        compress_option=None,
        encoding="utf-8",
        max_buffer_size=None,
        slot_buffer_size=None,
    ):
        self._schema = schema
        self._request_callback = request_callback
        self._session = session
        self._compress_option = compress_option

        self._max_buffer_size = max_buffer_size or self.DEFAULT_MAX_BUFFER_SIZE
        self._slot_buffer_size = slot_buffer_size or self.DEFAULT_SLOT_BUFFER_SIZE
        self._total_n_bytes = 0

        self._status = Upsert.Status.NORMAL

        self._schema = session.schema
        self._encoding = encoding
        self._hash_keys = self._session.hash_keys
        self._hasher = RecordHasher(schema, self._session.hasher, self._hash_keys)
        self._buckets = self._session.buckets.copy()
        self._bucket_buffers = {}
        self._bucket_writers = {}

        for slot in session.buckets.keys():
            self._build_bucket_writer(slot)

    @property
    def status(self):
        return self._status

    @property
    def n_bytes(self):
        return self._total_n_bytes

    def upsert(self, record):
        return self._write(record, Upsert.Operation.UPSERT)

    def delete(self, record):
        return self._write(record, Upsert.Operation.DELETE)

    def flush(self, flush_all=True):
        if len(self._session.buckets) != len(self._bucket_writers):
            raise TunnelError("session slot map is changed")
        else:
            self._buckets = self._session.buckets.copy()

        bucket_written = dict()
        bucket_to_count = dict()

        def write_bucket(bucket_id):
            slot = self._buckets[bucket_id]
            sio = self._bucket_buffers[bucket_id]
            rec_count = bucket_to_count[bucket_id]

            self._request_callback(bucket_id, slot, rec_count, sio.getvalue())
            self._build_bucket_writer(bucket_id)
            bucket_written[bucket_id] = True

        retry = 0
        while True:
            futs = []
            pool = futures.ThreadPoolExecutor(len(self._bucket_writers))
            try:
                self._check_status()
                for bucket, writer in self._bucket_writers.items():
                    if writer.n_bytes == 0 or bucket_written.get(bucket):
                        continue
                    if not flush_all and writer.n_bytes <= self._slot_buffer_size:
                        continue

                    bucket_to_count[bucket] = writer.count
                    writer.close()
                    futs.append(pool.submit(write_bucket, bucket))
                for fut in futs:
                    fut.result()
                break
            except KeyboardInterrupt:
                raise TunnelError("flush interrupted")
            except:
                retry += 1
                if retry == 3:
                    raise
            finally:
                pool.shutdown()

    def close(self):
        if self.status == Upsert.Status.NORMAL:
            self.flush()
            self._status = Upsert.Status.CLOSED

    def _build_bucket_writer(self, slot):
        self._bucket_buffers[slot] = compat.BytesIO()
        self._bucket_writers[slot] = BaseRecordWriter(
            self._schema,
            get_compress_stream(self._bucket_buffers[slot], self._compress_option),
            encoding=self._encoding,
        )

    def _check_status(self):
        if self._status == Upsert.Status.CLOSED:
            raise TunnelError("Stream is closed!")
        elif self._status == Upsert.Status.ERROR:
            raise TunnelError("Stream has error!")

    def _write(self, record, op, valid_columns=None):
        self._check_status()

        bucket = self._hasher.hash(record) % len(self._bucket_writers)
        if bucket not in self._bucket_writers:
            raise TunnelError(
                "Tunnel internal error! Do not have bucket for hash key " + bucket
            )
        record[self._session.UPSERT_OPERATION_KEY] = ord(
            b"U" if op == Upsert.Operation.UPSERT else b"D"
        )
        if valid_columns is None:
            record[self._session.UPSERT_VALUE_COLS_KEY] = []
        else:
            valid_cols_set = set(valid_columns)
            col_idxes = [
                idx for idx, col in self._schema.columns if col in valid_cols_set
            ]
            record[self._session.UPSERT_VALUE_COLS_KEY] = col_idxes

        writer = self._bucket_writers[bucket]
        prev_written_size = writer.n_bytes
        writer.write(record)
        written_size = writer.n_bytes
        self._total_n_bytes += written_size - prev_written_size

        if writer.n_bytes > self._slot_buffer_size:
            self.flush(False)
        elif self._total_n_bytes > self._max_buffer_size:
            self.flush(True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if an error occurs inside the with block, we do not commit
        if exc_val is not None:
            return
        self.close()
