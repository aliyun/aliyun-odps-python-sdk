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

import math

from .. import io
from ..pb import decoder
from ..io import CompressOption, SnappyInputStream
from ..checksum import Checksum
from ..errors import TunnelError
from ..wireconstants import ProtoWireConstants
from ... import serializers, utils, types, compat
from ...compat import Enum, six
from ...models import Schema, Record
from ...readers import AbstractRecordReader


class TableDownloadSession(serializers.JSONSerializableModel):
    __slots__ = '_client', '_table', '_partition_spec', '_compress_option'

    class Status(Enum):
        Unknown = 'UNKNOWN'
        Normal = 'NORMAL'
        Closes = 'CLOSES'
        Expired = 'EXPIRED'

    id = serializers.JSONNodeField('DownloadID')
    status = serializers.JSONNodeField(
        'Status', parse_callback=lambda s: TableDownloadSession.Status(s.upper()))
    count = serializers.JSONNodeField('RecordCount')
    schema = serializers.JSONNodeReferenceField(Schema, 'Schema')

    def __init__(self, client, table, partition_spec,
                 download_id=None, compress_option=None):
        super(TableDownloadSession, self).__init__()

        self._client = client
        self._table = table

        if isinstance(partition_spec, six.string_types):
            partition_spec = types.PartitionSpec(partition_spec)
        if isinstance(partition_spec, types.PartitionSpec):
            partition_spec = str(partition_spec).replace("'", '')
        self._partition_spec = partition_spec

        if download_id is None:
            self._init()
        else:
            self.id = download_id
            self.reload()
        self._compress_option = compress_option

    def _init(self):
        params = {'downloads': ''}
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
        params = {'downloadid': self.id}
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
            
    def open_record_reader(self, start, count, compress=False, columns=None):
        compress_option = self._compress_option or CompressOption()

        params = {}
        headers = {'Content-Length': 0, 'x-odps-tunnel-version': 4}
        if compress:
            if compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_ZLIB:
                headers['Accept-Encoding'] = 'deflate'
            elif compress_option.algorithm == \
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY:
                headers['Accept-Encoding'] = 'x-snappy-framed'
            elif compress_option.algorithm != \
                    CompressOption.CompressAlgorithm.ODPS_RAW:
                raise TunnelError('invalid compression option')
        params['downloadid'] = self.id
        params['data'] = ''
        params['rowrange'] = '(%s,%s)' % (start, count)
        if self._partition_spec is not None and len(self._partition_spec) > 0:
            params['partition'] = self._partition_spec
        if columns is not None and len(columns) > 0:
            col_name = lambda col: col.name if isinstance(col, types.Column) else col
            params['columns'] = ','.join(col_name(col) for col in columns)

        url = self._table.resource()
        resp = self._client.get(url, stream=True, params=params, headers=headers)
        if not self._client.is_ok(resp):
            e = TunnelError.parse(resp)
            raise e

        content_encoding = resp.headers.get('Content-Encoding')
        if content_encoding is not None:
            if content_encoding == 'deflate':
                self._compress_option = CompressOption(
                    CompressOption.CompressAlgorithm.ODPS_ZLIB, -1, 0)
            elif content_encoding == 'x-snappy-framed':
                self._compress_option = CompressOption(
                    CompressOption.CompressAlgorithm.ODPS_SNAPPY, -1, 0)
            else:
                raise TunnelError('invalid content encoding')
            compress = True
        else:
            compress = False

        option = compress_option if compress else None
        if option is None:
            input_stream = compat.BytesIO(resp.content)  # create a file-like object from body
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_RAW:
            input_stream = compat.BytesIO(resp.content)  # create a file-like object from body
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_ZLIB:
            input_stream = compat.BytesIO(resp.content)  # Requests automatically decompress gzip data!
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_SNAPPY:
            input_stream = SnappyInputStream(compat.BytesIO(resp.content))
        else:
            raise errors.InvalidArgument('Invalid compression algorithm.')

        return TableTunnelReader(self.schema, input_stream, columns=columns)


class TableTunnelReader(AbstractRecordReader):
    def __init__(self, schema, input_stream, columns=None):
        self._schema = schema
        if columns is None:
            self._columns = self._schema.columns
        else:
            self._columns = [self._schema[c] for c in columns]
        self._reader = decoder.Decoder(input_stream)
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._curr_cusor = 0

    @property
    def count(self):
        return self._curr_cusor

    def _read_array(self, value_type):
        res = []

        size = self._reader.read_uint32()
        for _ in range(size):
            if self._reader.read_bool():
                res.append(None)
            else:
                if value_type == types.string:
                    val = utils.to_text(self._reader.read_string())
                    self._crc.update(val)
                elif value_type == types.bigint:
                    val = self._reader.read_sint64()
                    self._crc.update_long(val)
                elif value_type == types.double:
                    val = self._reader.read_double()
                    self._crc.update_float(val)
                elif value_type == types.boolean:
                    val = self._reader.read_bool()
                    self._crc.update_bool(val)
                else:
                    raise IOError('Unsupport array type. type: %s' % value_type)
                res.append(val)

        return res

    def read(self):
        record = Record(self._columns)

        while True:
            index, _ = self._reader.read_field_number_and_wire_type()

            if index == 0:
                continue
            if index == ProtoWireConstants.TUNNEL_END_RECORD:
                checksum = utils.long_to_int(self._crc.getvalue())
                if int(self._reader.read_uint32()) != utils.int_to_uint(checksum):
                    raise IOError('Checksum invalid')
                self._crc.reset()
                self._crccrc.update_int(checksum)
                break

            if index == ProtoWireConstants.TUNNEL_META_COUNT:
                if self.count != self._reader.read_sint64():
                    raise IOError('count does not match')
                idx_of_checksum, _ = self._reader.read_field_number_and_wire_type()
                if ProtoWireConstants.TUNNEL_META_CHECKSUM != idx_of_checksum:
                    raise IOError('Invalid stream data.')
                if int(self._crccrc.getvalue()) != self._reader.read_uint32():
                    raise IOError('Checksum invalid.')
                # if not self._reader.at_end():
                #     raise IOError('Expect at the end of stream, but not.')

                return

            if index > len(self._columns):
                raise IOError('Invalid protobuf tag. Perhaps the datastream '
                              'from server is crushed.')

            self._crc.update_int(index)

            i = index - 1
            data_type = self._columns[i].type
            if data_type == types.double:
                val = self._reader.read_double()
                self._crc.update_float(val)
                record[i] = val
            elif data_type == types.boolean:
                val = self._reader.read_bool()
                self._crc.update_bool(val)
                record[i] = val
            elif data_type == types.bigint:
                val = self._reader.read_sint64()
                self._crc.update_long(val)
                record[i] = val
            elif data_type == types.string:
                val = utils.to_text(self._reader.read_string())
                self._crc.update(val)
                record[i] = val
            elif data_type == types.datetime:
                val = self._reader.read_sint64()
                self._crc.update_long(val)
                record[i] = utils.to_datetime(val)
            elif data_type == types.decimal:
                val = self._reader.read_string()
                self._crc.update(val)
                record[i] = val
            elif isinstance(data_type, types.Array):
                val = self._read_array(data_type.value_type)
                record[i] = val
            elif isinstance(data_type, types.Map):
                keys = self._read_array(data_type.key_type)
                values = self._read_array(data_type.value_type)
                val = compat.OrderedDict(zip(keys, values))
                record[i] = val
            else:
                raise IOError('Unsupported type %s' % data_type)

        self._curr_cusor += 1
        return record

    def __next__(self):
        record = self.read()
        if record is None:
            raise StopIteration
        return record

    next = __next__

    def reads(self):
        return self.__iter__()

    @property
    def n_bytes(self):
        return self._reader.position()

    def get_total_bytes(self):
        return self.n_bytes

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if hasattr(self._schema, 'close'):
            self._schema.close()
