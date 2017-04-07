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

import warnings

from ..pb.decoder import Decoder
from ..checksum import Checksum
from ..wireconstants import ProtoWireConstants
from ... import utils, types, compat
from ...models import Record
from ...readers import AbstractRecordReader
from ...config import options

try:
    if not options.force_py:
        from .reader_c import TunnelRecordReader
    else:
        TunnelRecordReader = None
except ImportError as e:
    if options.force_c:
        raise e
    TunnelRecordReader = None

if TunnelRecordReader is None:
    class TunnelRecordReader(AbstractRecordReader):

        def __init__(self, schema, input_stream, columns=None):
            self._schema = schema
            if columns is None:
                self._columns = self._schema.columns
            else:
                self._columns = [self._schema[c] for c in columns]
            self._reader = Decoder(input_stream)
            self._crc = Checksum()
            self._crccrc = Checksum()
            self._curr_cursor = 0
            self._read_limit = options.table_read_limit
            self._to_datetime = utils.build_to_datetime()

        def _mode(self):
            return 'py'

        @property
        def count(self):
            return self._curr_cursor

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
            if self._read_limit is not None and self.count >= self._read_limit:
                warnings.warn('Number of lines read via tunnel already reaches the limitation.')
                return None

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
                    record[i] = self._to_datetime(val)
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

            self._curr_cursor += 1
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
