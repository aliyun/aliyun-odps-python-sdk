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


import zlib
import struct
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
from datetime import datetime

from google.protobuf.internal.decoder import _DecodeSignedVarint32
from google.protobuf.internal.decoder import *

from odps.utils import long_to_int, int_to_uint
from odps.models import Record
from odps.errors import DependencyNotInstalledError
from odps.tunnel.conf import CompressOption
from odps.tunnel.checksum import Checksum
from odps.tunnel.wireconstants import ProtoWireConstants


class ProtobufInputStream(object):
    def __init__(self, fp):
        self.total_bytes = 0L
        self.buf = fp.read()
        self.buf_size = len(self.buf)

    def read_field_number(self):
        tag, self.total_bytes = _DecodeSignedVarint32(self.buf, self.total_bytes)
        return wire_format.UnpackTag(tag)[0]

    def read_boolean(self):
        return _DecodeSignedVarint32(self.buf, self.total_bytes)[0] != 0

    def _decode_value(self, decoder, new_default=None, message=None):
        decode_key = '__decode_key'
        decode_dict = {}
        decode = decoder(None, False, False, decode_key, new_default)
        self.total_bytes = \
            decode(self.buf, self.total_bytes, self.buf_size, message, decode_dict)
        return decode_dict[decode_key]
    
    def read_sint32(self):
        return self._decode_value(SInt32Decoder)
    
    def read_uint32(self):
        return self._decode_value(UInt32Decoder)
    
    def read_long(self):
        return self._decode_value(SInt64Decoder)
    
    def read_double(self):
        return self._decode_value(DoubleDecoder)
        
    def read_raw_bytes(self):
        return bytearray(self._decode_value(BytesDecoder, bytearray, ''))
    
    def read_string(self):
        return self.read_raw_bytes().decode('utf-8')
    
    def close(self):
        self.fp.close()

    def read(self):
        if self.total_bytes < self.buf_size - 1:
            byte = self.buf[self.total_bytes]
            self.total_bytes += 1
            return struct.unpack('<b', byte)[0]
        return -1


class ProtobufInputReader(object):
    def __init__(self, table_schema, fp, compress_opt=None):
        self.columns = table_schema.columns
        
        if compress_opt is not None:
            if compress_opt.algorithm == \
                    CompressOption.CompressionAlgorithm.ODPS_ZLIB:
                # requests will do the `zlib.decompress`
                pass
            elif compress_opt.algorithm == \
                    CompressOption.CompressionAlgorithm.ODPS_SNAPPY:
                try:
                    import snappy
                except ImportError:
                    raise DependencyNotInstalledError(
                        'python-snappy library is required for snappy support')
                fp = StringIO(snappy.decompress(fp.read()))
            elif compress_opt.algorithm != \
                    CompressOption.CompressionAlgorithm.ODPS_RAW:
                raise IOError('invalid compression option.')
        self.stream = ProtobufInputStream(fp)
        
        self.crc = Checksum()
        self.crccrc = Checksum()
        self.count = 0
        
    def read(self):
        record = Record(self.columns)
        while True:
            checksum = 0
            i = self.stream.read_field_number()
            if i == 0:
                continue
            if i == ProtoWireConstants.TUNNEL_END_RECORD:
                checksum = long_to_int(self.crc.get_value())
                if int(self.stream.read_uint32()) != int_to_uint(checksum):
                    raise IOError('Checksum invalid')
                self.crc.reset()
                self.crccrc.update_(checksum, force='int')
                break

            if i == ProtoWireConstants.TUNNEL_META_COUNT:
                if self.count != self.stream.read_long():
                    raise IOError('count does not match')

                if ProtoWireConstants.TUNNEL_META_CHECKSUM != \
                    self.stream.read_field_number():
                    raise IOError('Invalid stream.')

                if int(self.crccrc.get_value()) != \
                    self.stream.read_uint32():
                    raise IOError('Checksum invalid.')

                if self.stream.read() >= 0:
                    raise IOError('Expect at the end of stream, but not.')

                return

            if i > len(self.columns):
                raise IOError('Invalid protobuf tag. Perhaps the datastream '
                              'from server is crushed.')

            self.crc.update_(i, force='int')

            t = self.columns[i-1].type.upper()
            if t == 'DOUBLE':
                v = self.stream.read_double()
                self.crc.update_(float(v), force='float')
                record.set(i-1, v)
            elif t == 'BOOLEAN':
                v = self.stream.read_boolean()
                self.crc.update_(bool(v), force='boolean')
                record.set(i-1, v)
            elif t == 'BIGINT':
                v = self.stream.read_long()
                self.crc.update_(long(v), force='long')
                record.set(i-1, v)
            elif t == 'STRING':
                bytes_ = self.stream.read_raw_bytes()
                self.crc.update(bytes_, 0, len(bytes_))
                record.set(i-1, bytes_)
            elif t == 'DATETIME':
                v = self.stream.read_long()
                self.crc.update_(long(v), force='long')
                v = float(v) / 1000
                record.set(i-1, datetime.fromtimestamp(v))
        
        self.count += 1
        return record
    
    def reads(self):
        while True:
            record = self.read()
            if record is None:
                break
            yield record
    
    def close(self):
        self.stream.close()
        
    def get_total_bytes(self):
        return self.stream.total_bytes
