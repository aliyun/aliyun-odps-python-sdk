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

import struct
import warnings
from io import BytesIO

try:
    import pyarrow as pa
except ImportError:
    pa = None


from ..pb.decoder import Decoder, wire_format
from ..pb.errors import DecodeError
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
            self._to_date = utils.to_date

        def _mode(self):
            return 'py'

        @property
        def count(self):
            return self._curr_cursor

        def _read_field(self, data_type):
            if data_type == types.float_:
                val = self._reader.read_float()
                self._crc.update_float(val)
            elif data_type == types.double:
                val = self._reader.read_double()
                self._crc.update_double(val)
            elif data_type == types.boolean:
                val = self._reader.read_bool()
                self._crc.update_bool(val)
            elif data_type in types.integer_types:
                val = self._reader.read_sint64()
                self._crc.update_long(val)
            elif data_type == types.string:
                val = self._reader.read_string()
                self._crc.update(val)
            elif data_type == types.binary:
                val = self._reader.read_string()
                self._crc.update(val)
            elif data_type == types.datetime:
                val = self._reader.read_sint64()
                self._crc.update_long(val)
                val = self._to_datetime(val)
            elif data_type == types.date:
                val = self._reader.read_sint64()
                self._crc.update_long(val)
                val = self._to_date(val)
            elif data_type == types.timestamp:
                l_val = self._reader.read_sint64()
                self._crc.update_long(l_val)
                nano_secs = self._reader.read_sint32()
                self._crc.update_int(nano_secs)
                try:
                    import pandas as pd
                except ImportError:
                    raise ImportError('To use TIMESTAMP in pyodps, you need to install pandas.')
                val = pd.Timestamp(self._to_datetime(l_val * 1000)) + pd.Timedelta(nanoseconds=nano_secs)
            elif data_type == types.interval_day_time:
                l_val = self._reader.read_sint64()
                self._crc.update_long(l_val)
                nano_secs = self._reader.read_sint32()
                self._crc.update_int(nano_secs)
                try:
                    import pandas as pd
                except ImportError:
                    raise ImportError('To use INTERVAL_DAY_TIME in pyodps, you need to install pandas.')
                val = pd.Timedelta(seconds=l_val, nanoseconds=nano_secs)
            elif data_type == types.interval_year_month:
                l_val = self._reader.read_sint64()
                self._crc.update_long(l_val)
                return compat.Monthdelta(l_val)
            elif isinstance(data_type, (types.Char, types.Varchar)):
                val = self._reader.read_string()
                self._crc.update(val)
            elif isinstance(data_type, types.Decimal):
                val = self._reader.read_string()
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

        def _read_array(self, value_type):
            res = []

            size = self._reader.read_uint32()
            for _ in range(size):
                if self._reader.read_bool():
                    res.append(None)
                else:
                    res.append(self._read_field(value_type))

            return res

        def _read_struct(self, value_type):
            res = compat.OrderedDict()
            for k in value_type.field_types:
                if self._reader.read_bool():
                    res[k] = None
                else:
                    res[k] = self._read_field(value_type.field_types[k])
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
                record[i] = self._read_field(self._columns[i].type)

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

        def close(self):
            if hasattr(self._schema, 'close'):
                self._schema.close()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()


class TunnelArrowReader(object):
    def __init__(self, schema, input_stream, columns=None):
        self._schema = schema
        arrow_schema = odps_type_to_arrow_type(schema)
        if columns is None:
            self._arrow_schema = arrow_schema
        else:
            self._arrow_schema = pa.schema([s for s in arrow_schema if s.name in columns])

        self._reader = input_stream
        self._crc = Checksum()
        self._crccrc = Checksum()
        self._pos = 0
        self._buffer = BytesIO()
        self._chunk_size = None
        self._read_limit = options.table_read_limit

    @property
    def schema(self):
        return self._schema

    @staticmethod
    def _read_unint32(b):
        return struct.unpack('!I', b)

    def _read_chunk_size(self):
        try:
            i = self._read_unint32(self._reader.read(4))
            self._pos += 4
            return i[0]  # unpack() result is a 1-element tuple.
        except struct.error as e:
            raise DecodeError(e)

    def _read_chunk(self):
        read_size = self._chunk_size + 4
        b = self._reader.read(read_size)
        if 0 < len(b) < 4:
            raise IOError('Checksum invalid')
        self._pos += len(b)
        self._crc.update(b[:-4])
        self._crccrc.update(b[:-4])
        return b

    def read(self):
        if self._read_limit is not None and self._pos >= self._read_limit:
            warnings.warn('Number of lines read via tunnel already reaches the limitation.')
            return None

        if self._chunk_size is None:
            self._chunk_size = self._read_chunk_size()

        while True:
            b = self._read_chunk()
            data = b[:-4]
            crc_data = b[-4:]
            if len(b) == 0:
                break
            if len(b) < self._chunk_size + 4:
                # is last chunk
                read_checksum = self._read_unint32(crc_data)[0]
                checksum = int(self._crccrc.getvalue())
                if checksum != read_checksum:
                    raise IOError('Checksum invalid')
                self._pos += len(data) + 4
                self._buffer.write(data)
                self._crccrc.reset()
                break
            else:
                checksum = int(self._crc.getvalue())
                read_checksum = self._read_unint32(crc_data)[0]
                if checksum != read_checksum:
                    raise IOError('Checksum invalid')
                self._crc.reset()
                self._buffer.write(data)

        b = self._buffer.getvalue()
        if len(b) == 0:
            # empty table
            return self._arrow_schema.empty_table()
        else:
            complete_bytes = self._arrow_schema.serialize().to_pybytes() + b
            r = pa.ipc.open_stream(complete_bytes).read_all()
            array_dict = dict((arr._name, arr) for arr in r)
            new_dict = {}
            for name, arr in array_dict.items():
                if arr.type == pa.timestamp('ms'):
                    datetime = [utils.build_to_datetime()(v) for v in arr.cast('int64').to_numpy()]
                    new_dict[name] = datetime
                else:
                    new_dict[name] = arr
            return pa.Table.from_pydict(new_dict)

    @property
    def n_bytes(self):
        return self._pos

    def get_total_bytes(self):
        return self.n_bytes

    def close(self):
        if hasattr(self._reader, 'close'):
            self._reader.close()

    def to_pandas(self):
        record_batch = self.read()
        return record_batch.to_pandas()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def odps_type_to_arrow_type(odps_schema):
    from ... import types
    TYPE_MAPPING = {
        types.string: pa.string(),
        types.binary: pa.binary(),
        types.tinyint: pa.int8(),
        types.smallint: pa.int16(),
        types.int_: pa.int32(),
        types.bigint: pa.int64(),
        types.boolean: pa.bool_(),
        types.float_: pa.float32(),
        types.double: pa.float64(),
        types.date: pa.date32(),
        types.datetime: pa.timestamp('ms'),
        types.timestamp: pa.timestamp('ns')
    }

    arrow_schema = []
    for schema in odps_schema.simple_columns:
        col_name = schema.name
        if schema.type in TYPE_MAPPING:
            col_type = TYPE_MAPPING[schema.type]
        else:
            if isinstance(schema.type, types.Array):
                col_type = pa.list_(TYPE_MAPPING[schema.type.value_type])
            elif isinstance(schema.type, types.Decimal):
                col_type = pa.decimal128(schema.type.precision,
                                         schema.type.scale)
            else:
                raise TypeError('Unsupported type: {}'.format(schema.type))

        arrow_schema.append(pa.field(col_name, col_type))

    return pa.schema(arrow_schema)
