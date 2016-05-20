#!/usr/bin/env python
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


from .. import io
from ..pb.writer import ProtobufWriter
from ..pb.wire_format import WIRETYPE_VARINT, WIRETYPE_FIXED64, WIRETYPE_LENGTH_DELIMITED
from ..checksum import Checksum
from ..errors import TunnelError
from ..io import CompressOption, SnappyOutputStream, DeflateOutputStream, RequestsIO
from ..wireconstants import ProtoWireConstants
from ... import utils, types, compat, options, serializers, errors
from ...models import Schema, Record
from ...compat import Enum, six


class TableUploadSession(serializers.JSONSerializableModel):
    __slots__ = '_client', '_table', '_partition_spec', '_compress_option'

    class Status(Enum):
        Unknown = 'UNKNOWN'
        Normal = 'NORMAL'
        Closing = 'CLOSING'
        Closed = 'CLOSED'
        Canceled = 'CANCELED'
        Expired = 'EXPIRED'
        Critical = 'CRITICAL'

    id = serializers.JSONNodeField('UploadID')
    status = serializers.JSONNodeField(
        'Status', parse_callback=lambda s: TableUploadSession.Status(s.upper()))
    blocks = serializers.JSONNodesField('UploadedBlockList', 'BlockID')
    schema = serializers.JSONNodeReferenceField(Schema, 'Schema')
        
    def __init__(self, client, table, partition_spec,
                 upload_id=None, compress_option=None):
        super(TableUploadSession, self).__init__()

        self._client = client
        self._table = table

        if isinstance(partition_spec, six.string_types):
            partition_spec = types.PartitionSpec(partition_spec)
        if isinstance(partition_spec, types.PartitionSpec):
            partition_spec = str(partition_spec).replace("'", '')
        self._partition_spec = partition_spec

        if upload_id is None:
            self._init()
        else:
            self.id = upload_id
            self.reload()
        self._compress_option = compress_option

    def _init(self):
        params = {'uploads': 1}
        headers = {'Content-Length': 0}
        if self._partition_spec is not None and \
                len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        url = self._table.resource()
        resp = self._client.post(url, {}, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
        else:
            e = TunnelError.parse(resp)
            raise e

    def reload(self):
        params = {'uploadid': self.id}
        headers = {'Content-Length': 0}
        if self._partition_spec is not None and \
                len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        url = self._table.resource()
        resp = self._client.get(url, params=params, headers=headers)
        if self._client.is_ok(resp):
            self.parse(resp, obj=self)
        else:
            e = TunnelError.parse(resp)
            raise e

    def new_record(self, values=None):
        return Record(self.schema.columns, values=values)

    def open_record_writer(self, block_id=None, compress=False, buffer_size=None):
        compress_option = self._compress_option or CompressOption()

        params = {}
        headers = {'Transfer-Encoding': 'chunked',
                   'Content-Type': 'application/octet-stream',
                   'x-odps-tunnel-version': 4}
        if compress:
            if compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Content-Encoding'] = 'deflate'
            elif compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY:
                headers['Content-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != \
                    CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')
        params['uploadid'] = self.id
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        url = self._table.resource()
        option = compress_option if compress else None

        if block_id is None:
            def upload_block(blockid, data):
                params['blockid'] = blockid
                self._client.put(url, data=data, params=params, headers=headers)
            writer = BufferredRecordWriter(self.schema, upload_block, compress_option=option,
                                           buffer_size=buffer_size)
        else:
            params['blockid'] = block_id

            def upload(data):
                self._client.put(url, data=data, params=params, headers=headers)
            writer = RecordWriter(self.schema, upload, compress_option=option)
        return writer

    def get_block_list(self):
        self.reload()
        return self.blocks

    def commit(self, blocks):
        if blocks is None:
            raise ValueError('Invalid parameter: blocks.')
        if isinstance(blocks, six.integer_types):
            blocks = [blocks, ]

        server_block_map = dict([(int(block_id), True) for block_id \
                                 in self.get_block_list()])
        client_block_map = dict([(int(block_id), True) for block_id in blocks])

        if len(server_block_map) != len(client_block_map):
            raise TunnelError('Blocks not match, server: '+str(len(server_block_map))+
                              ', tunnelServerClient: '+str(len(client_block_map)))

        for block_id in blocks:
            if block_id not in server_block_map:
                raise TunnelError('Block not exists on server, block id is'+block_id)

        self._complete_upload()

    def _complete_upload(self):
        params = {'uploadid': self.id}
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        url = self._table.resource()
        resp = self._client.post(url, '', params=params)
        self.parse(resp, obj=self)


class BaseRecordWriter(object):
    def __init__(self, schema, encoding='utf-8'):
        self._encoding = encoding
        self._schema = schema
        self._columns = self._schema.columns
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._curr_cursor = 0
        self._writer = None

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
                self._writer.write_tag(pb_index, WIRETYPE_VARINT)
                self._write_bool(val)
            elif data_type == types.datetime:
                val = utils.to_milliseconds(val)
                self._writer.write_tag(pb_index, WIRETYPE_VARINT)
                self._write_long(val)
            elif data_type == types.string:
                self._writer.write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                self._write_string(val)
            elif data_type == types.double:
                self._writer.write_tag(pb_index, WIRETYPE_FIXED64)
                self._write_double(val)
            elif data_type == types.bigint:
                self._writer.write_tag(pb_index, WIRETYPE_VARINT)
                self._write_long(val)
            elif data_type == types.decimal:
                self._writer.write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                self._write_string(str(val))
            elif isinstance(data_type, types.Array):
                self._writer.write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                self._writer.write_uint(len(val))
                self._write_array(val, data_type.value_type)
            elif isinstance(data_type, types.Map):
                self._writer.write_tag(pb_index, WIRETYPE_LENGTH_DELIMITED)
                self._writer.write_uint(len(val))
                self._write_array(compat.lkeys(val), data_type.key_type)
                self._writer.write_uint(len(val))
                self._write_array(compat.lvalues(val), data_type.value_type)
            else:
                raise IOError('Invalid data type: %s' % data_type)

        checksum = utils.long_to_int(self._crc.getvalue())
        self._writer.write_tag(ProtoWireConstants.TUNNEL_END_RECORD, WIRETYPE_VARINT)
        self._writer.write_uint(utils.long_to_uint(checksum))
        self._crc.reset()
        self._crccrc.update_int(checksum)
        self._curr_cursor += 1

    def _write_bool(self, data):
        self._crc.update_bool(data)
        self._writer.write_bool(data)

    def _write_long(self, data):
        self._crc.update_long(data)
        self._writer.write_long(data)

    def _write_double(self, data):
        self._crc.update_float(data)
        self._writer.write_double(data)

    def _write_string(self, data):
        if isinstance(data, six.text_type):
            data = data.encode(self._encoding)
        self._crc.update(data)
        self._writer.write_string(data)

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
                self._writer.write_bool(True)
            else:
                self._writer.write_bool(False)
                self._write_primitive(value, data_type)

    @property
    def count(self):
        return self._curr_cursor

    def close(self):
        self._writer.write_tag(ProtoWireConstants.TUNNEL_META_COUNT, WIRETYPE_VARINT)
        self._writer.write_long(self.count)
        self._writer.write_tag(ProtoWireConstants.TUNNEL_META_CHECKSUM, WIRETYPE_VARINT)
        self._writer.write_uint(utils.long_to_uint(self._crccrc.getvalue()))
        self._writer.close()
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
        self._writer = ProtobufWriter(out)
        self._upload_started = False

    def _start_upload(self):
        if self._upload_started:
            return
        self._req_io.start()
        self._upload_started = True

    def write(self, record):
        self._start_upload()
        if self._req_io.ASYNC_ERR:
            raise IOError("writer abort since an async error occurs")
        super(RecordWriter, self).write(record)

    def close(self):
        super(RecordWriter, self).close()
        self._req_io.finish()

    @property
    def n_bytes(self):
        return self._writer.n_bytes

    def get_total_bytes(self):
        return self.n_bytes


class BufferredRecordWriter(BaseRecordWriter):
    """
    This writer buffers the output of serializer. When the buffer exceeds a fixed-size of limit
     (default 10 MiB), it uploads the buffered output within one http connection.
    """

    BUFFER_SIZE = 10485760

    def __init__(self, schema, request_callback, compress_option=None, encoding='utf-8', buffer_size=None):
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

        self._writer = ProtobufWriter(out)

    def write(self, record):
        super(BufferredRecordWriter, self).write(record)
        if self._writer.n_bytes > self._buffer_size:
            self._flush()

    def close(self):
        if self._writer.n_bytes > 0:
            self._flush()
        self._writer.close()
        self._buffer.close()

    def _flush(self):
        self._writer.write_tag(ProtoWireConstants.TUNNEL_META_COUNT, WIRETYPE_VARINT)
        self._writer.write_long(self.count)
        self._writer.write_tag(ProtoWireConstants.TUNNEL_META_CHECKSUM, WIRETYPE_VARINT)
        self._writer.write_uint(utils.long_to_uint(self._crccrc.getvalue()))

        self._n_bytes_written += self._writer.n_bytes
        self._writer.close()

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

        self._writer = ProtobufWriter(out)
        self._curr_cursor = 0
        self._crccrc.reset()
        self._crc.reset()

    @property
    def n_bytes(self):
        return self._n_bytes_written + self._writer.n_bytes

    def get_total_bytes(self):
        return self.n_bytes

    def get_blocks_written(self):
        return self._blocks_written
