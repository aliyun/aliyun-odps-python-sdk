#!/usr/bin/env python
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

from ...config import options
try:
    if not options.force_py:
        from .writer_c import BaseRecordWriter
    else:
        BaseRecordWriter = None
except ImportError as e:
    if options.force_c:
        raise e
    BaseRecordWriter = None
from ..pb.encoder import Encoder
from ..pb.wire_format import WIRETYPE_VARINT, WIRETYPE_FIXED64, WIRETYPE_LENGTH_DELIMITED
from ..checksum import Checksum
from ..wireconstants import ProtoWireConstants
from .stream import CompressOption, SnappyOutputStream, DeflateOutputStream, RequestsIO
from ... import types, compat, utils, errors, options
from ...compat import six, raise_exc


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
            return 'py'

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

        def _write_raw_uint(self, val):
            self._encoder.append_uint32(val)
            self._refresh_buffer()

        def _write_raw_bool(self, val):
            self._encoder.append_bool(val)
            self._refresh_buffer()

        def _write_raw_double(self, val):
            self._encoder.append_double(val)
            self._refresh_buffer()

        def _write_raw_string(self, val):
            self._encoder.append_string(val)
            self._refresh_buffer()


    class BaseRecordWriter(ProtobufWriter):
        def __init__(self, schema, out, encoding='utf-8'):
            self._encoding = encoding
            self._schema = schema
            self._columns = self._schema.columns
            self._crc = Checksum()
            self._crccrc = Checksum()
            self._curr_cursor = 0
            self._to_milliseconds = utils.build_to_milliseconds()

            super(BaseRecordWriter, self).__init__(out)

        def write(self, record):
            n_record_fields = len(record)
            n_columns = len(self._columns)

            if n_record_fields > n_columns:
                raise IOError('record fields count is more than schema.')

            for i in range(min(n_record_fields, n_columns)):
                if self._schema.is_partition(self._columns[i]):
                    continue

                val = record[i]
                if val is None:
                    continue

                pb_index = i + 1
                self._crc.update_int(pb_index)

                data_type = self._columns[i].type
                if data_type == types.boolean:
                    self._write_tag(pb_index, WIRETYPE_VARINT)
                    self._write_bool(val)
                elif data_type == types.datetime:
                    val = self._to_milliseconds(val)
                    self._write_tag(pb_index, WIRETYPE_VARINT)
                    self._write_long(val)
                elif data_type == types.string:
                    self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                    self._write_string(val)
                elif data_type == types.double:
                    self._write_tag(pb_index, WIRETYPE_FIXED64)
                    self._write_double(val)
                elif data_type == types.bigint:
                    self._write_tag(pb_index, WIRETYPE_VARINT)
                    self._write_long(val)
                elif data_type == types.decimal:
                    self._write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                    self._write_string(str(val))
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

        def _write_bool(self, data):
            self._crc.update_bool(data)
            self._write_raw_bool(data)

        def _write_long(self, data):
            self._crc.update_long(data)
            self._write_raw_long(data)

        def _write_double(self, data):
            self._crc.update_float(data)
            self._write_raw_double(data)

        def _write_string(self, data):
            if isinstance(data, six.text_type):
                data = data.encode(self._encoding)
            self._crc.update(data)
            self._write_raw_string(data)

        def _write_primitive(self, data, data_type):
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

        def _write_array(self, data, data_type):
            for value in data:
                if value is None:
                    self._write_raw_bool(True)
                else:
                    self._write_raw_bool(False)
                    self._write_primitive(value, data_type)

        @property
        def count(self):
            return self._curr_cursor

        def close(self):
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


class RecordWriter(BaseRecordWriter):
    """
    This writer uploads the output of serializer asynchronously within a long-lived http connection.
    """

    def __init__(self, schema, request_callback, compress_option=None, encoding='utf-8'):
        super(RecordWriter, self).__init__(schema, encoding)
        self._req_io = RequestsIO(request_callback, chunk_size=options.chunk_size)

        if compress_option is None:
            out = self._req_io
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_RAW:
            out = self._req_io
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_ZLIB:
            out = DeflateOutputStream(self._req_io)
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_SNAPPY:
            out = SnappyOutputStream(self._req_io)
        else:
            raise errors.InvalidArgument('Invalid compression algorithm.')
        super(RecordWriter, self).__init__(schema, out)
        self._upload_started = False

    def _start_upload(self):
        if self._upload_started:
            return
        self._req_io.start()
        self._upload_started = True

    def write(self, record):
        self._start_upload()
        if self._req_io._async_err:
            ex_type, ex_value, tb = self._req_io._async_err
            raise_exc(ex_type, ex_value, tb)
        super(RecordWriter, self).write(record)

    def close(self):
        super(RecordWriter, self).close()
        self._req_io.finish()

    def get_total_bytes(self):
        return self.n_bytes


class BufferredRecordWriter(BaseRecordWriter):
    """
    This writer buffers the output of serializer. When the buffer exceeds a fixed-size of limit
     (default 10 MiB), it uploads the buffered output within one http connection.
    """

    BUFFER_SIZE = 10485760

    def __init__(self, schema, request_callback, compress_option=None,
                 encoding='utf-8', buffer_size=None):
        super(BufferredRecordWriter, self).__init__(schema, encoding)
        self._buffer_size = buffer_size or self.BUFFER_SIZE
        self._request_callback = request_callback
        self._block_id = 0
        self._blocks_written = []
        self._buffer = compat.BytesIO()
        self._n_bytes_written = 0
        self._compress_option = compress_option

        if self._compress_option is None:
            out = self._buffer
        elif self._compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_RAW:
            out = self._buffer
        elif self._compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_ZLIB:
            out = DeflateOutputStream(self._buffer)
        elif self._compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_SNAPPY:
            out = SnappyOutputStream(self._buffer)
        else:
            raise errors.InvalidArgument('Invalid compression algorithm.')

        super(BufferredRecordWriter, self).__init__(schema, out)

    def write(self, record):
        super(BufferredRecordWriter, self).write(record)
        if self._n_raw_bytes > self._buffer_size:
            self._flush()

    def close(self):
        if self._n_raw_bytes > 0:
            self._flush()
        self.flush_all()
        self._buffer.close()

    def _flush(self):
        self._write_tag(ProtoWireConstants.TUNNEL_META_COUNT, WIRETYPE_VARINT)
        self._write_raw_long(self.count)
        self._write_tag(ProtoWireConstants.TUNNEL_META_CHECKSUM, WIRETYPE_VARINT)
        self._write_raw_uint(utils.long_to_uint(self._crccrc.getvalue()))

        self._n_bytes_written += self._n_raw_bytes
        self.flush_all()

        def gen():  # synchronize chunk upload
            data = self._buffer.getvalue()
            while data:
                to_send = data[:options.chunk_size]
                data = data[options.chunk_size:]
                yield to_send

        self._request_callback(self._block_id, gen())
        self._blocks_written.append(self._block_id)
        self._block_id += 1
        self._buffer = compat.BytesIO()

        if self._compress_option is None:
            out = self._buffer
        elif self._compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_RAW:
            out = self._buffer
        elif self._compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_ZLIB:
            out = DeflateOutputStream(self._buffer)
        elif self._compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_SNAPPY:
            out = SnappyOutputStream(self._buffer)
        else:
            raise errors.InvalidArgument('Invalid compression algorithm.')

        self._re_init(out)
        self._curr_cursor = 0
        self._crccrc.reset()
        self._crc.reset()

    @property
    def _n_raw_bytes(self):
        return super(BufferredRecordWriter, self).n_bytes

    @property
    def n_bytes(self):
        return self._n_bytes_written + self._n_raw_bytes

    def get_total_bytes(self):
        return self.n_bytes

    def get_blocks_written(self):
        return self._blocks_written

