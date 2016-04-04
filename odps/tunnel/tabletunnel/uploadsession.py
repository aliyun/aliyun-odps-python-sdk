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


from enum import Enum
import six

from .. import io
from ..errors import TunnelError
from ..checksum import Checksum
from ..wireconstants import ProtoWireConstants
from ... import utils, types, compat, options, serializers
from ...models import Schema, Record


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

    def open_record_writer(self, block_id, compress=False):
        """
        BlockId是由用户选取的0~19999之间的数值，标识本次上传数据块
        """
        compress_option = self._compress_option or io.CompressOption()

        params = {}
        headers = {'Transfer-Encoding': 'chunked',
                   'Content-Type': 'application/octet-stream',
                   'x-odps-tunnel-version': 4}
        if compress:
            if compress_option.algorithm == \
                    io.CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Content-Encoding'] = 'deflate'
            elif compress_option.algorithm == \
                    io.CompressOption.CompressAlgorithm.ODPS_SNAPPY:
                headers['Content-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != \
                    io.CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')
        params['uploadid'] = self.id
        params['blockid'] = block_id
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec

        url = self._table.resource()
        chunk_upload = lambda data: self._client.put(
            url, data=data, params=params, headers=headers)
        option = compress_option if compress else None
        writer = TableTunnelWriter(self.schema, chunk_upload, compress_option=option)

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


class TableTunnelWriter(object):
    def __init__(self, schema, do_upload, compress_option=None,
                 compress_algo=None, compres_level=None, compress_strategy=None,
                 encoding='utf-8'):
        self._compress_option = compress_option
        if self._compress_option is None and compress_algo is not None:
            self._compress_option = io.CompressOption(
                compress_algo=compress_algo, level=compres_level, strategy=compress_strategy)
        self._encoding = encoding

        self._schema = schema
        self._columns = self._schema.columns
        self._req_io = io.RequestsIO(do_upload)

        self._writer = io.ProtobufWriter(self._req_io, compress_option=self._compress_option,
                                         buffer_size=options.chunk_size, encoding=encoding)

        self._crc = Checksum()
        self._crccrc = Checksum()
        self._curr_cursor = 0

        self._upload_started = False

    def _start_upload(self):
        if self._upload_started:
            return

        self._req_io.start()
        self._upload_started = True

    def _write_bool(self, pb_index, data):
        self._crc.update_bool(data)
        self._writer.write_bool(pb_index, data)

    def _write_long(self, pb_index, data):
        self._crc.update_long(data)
        self._writer.write_long(pb_index, data)

    def _write_double(self, pb_index, data):
        self._crc.update_float(data)
        self._writer.write_double(pb_index, data)

    def _write_string(self, pb_index, data):
        if isinstance(data, six.text_type):
            data = data.encode(self._encoding)

        self._crc.update(data)
        self._writer.write_string(pb_index, data)

    def _write_primitive(self, data, data_type):
        if data_type == types.string:
            if isinstance(data, six.text_type):
                data = data.encode(self._encoding)

            self._writer.write_raw_varint32(len(data))
            self._writer.write_raw_bytes(data)
            self._crc.update(data)
        elif data_type == types.bigint:
            self._writer.write_long_no_tag(data)
            self._crc.update_long(data)
        elif data_type == types.double:
            self._writer.write_double_no_tag(data)
            self._crc.update_float(data)
        elif data_type == types.boolean:
            self._writer.write_bool_no_tag(data)
            self._crc.update_bool(data)
        else:
            raise IOError('Not a primitive type in array. type: %s' % data_type)

    def _write_array(self, pb_index, data, data_type):
        self._writer.write_raw_varint32(len(data))
        for value in data:
            if value is None:
                self._writer.write_bool_no_tag(True)
            else:
                self._writer.write_bool_no_tag(False)
                self._write_primitive(value, data_type)

    def _write_map(self, pb_index, data, key_type, value_type):
        self._write_array(pb_index, compat.lkeys(data), key_type)
        self._write_array(pb_index, compat.lvalues(data), value_type)

    def write(self, record):
        self._start_upload()

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
                self._write_bool(pb_index, val)
            elif data_type == types.datetime:
                val = utils.to_milliseconds(val)
                self._write_long(pb_index, val)
            elif data_type == types.string:
                self._write_string(pb_index, val)
            elif data_type == types.double:
                self._write_double(pb_index, val)
            elif data_type == types.bigint:
                self._write_long(pb_index, val)
            elif data_type == types.decimal:
                self._write_string(pb_index, str(val))
            elif isinstance(data_type, types.Array):
                self._writer.write_length_delimited_tag(pb_index)
                self._write_array(pb_index, val, data_type.value_type)
            elif isinstance(data_type, types.Map):
                self._writer.write_length_delimited_tag(pb_index)
                self._write_map(pb_index, val, data_type.key_type, data_type.value_type)
            else:
                raise IOError('Invalid data type: %s' % data_type)

        checksum = utils.long_to_int(self._crc.getvalue())
        self._writer.write_uint32(
            ProtoWireConstants.TUNNEL_END_RECORD, utils.int_to_uint(checksum))

        self._crc.reset()
        self._crccrc.update_int(checksum)

        self._curr_cursor += 1

    @property
    def count(self):
        return self._curr_cursor

    def close(self):
        self._writer.write_long(ProtoWireConstants.TUNNEL_META_COUNT, self.count)
        self._writer.write_uint32(ProtoWireConstants.TUNNEL_META_CHECKSUM,
                                  utils.long_to_uint(self._crccrc.getvalue()))
        self._writer.close()
        self._req_io.finish()
        self._curr_cursor = 0

    @property
    def n_bytes(self):
        return self._writer.n_bytes

    def get_total_bytes(self):
        return self.n_bytes

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
