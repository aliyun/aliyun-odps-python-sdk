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
import time

from google.protobuf.internal.encoder import _EncodeSignedVarint
from google.protobuf.internal.encoder import *
from google.protobuf.internal import wire_format

from odps.utils import int_to_uint, long_to_int, long_to_uint
from odps.errors import DependencyNotInstalledError
from odps.tunnel.conf import CompressOption
from odps.tunnel.checksum import Checksum
from odps.tunnel.wireconstants import ProtoWireConstants


class ProtobufOutputStream(object):
    BUFFER_SIZE = 8192
    
    def __init__(self, fp):
        self.fp = fp
        self.count = 0
        self.total_bytes = 0L
        
    def _flush(self):
        if self.count > 0:
            self.fp.flush()
            self.total_bytes += self.count
            self.count = 0
            
    def write(self, b):
        '''
        param b: byte, int between 0 ~ 255
        '''
        if self.count >= self.BUFFER_SIZE:
            self._flush()
        self.fp.write(chr(b))
        self.count += 1
        
    def write_(self, b, off=0, length=None):
        if length is None:
            length = len(b)

        if length >= self.BUFFER_SIZE:
            self._flush()
            self.fp.write(str(b[off:off+length]))
            self.total_bytes += length
            return
        
        if length > self.BUFFER_SIZE - self.count:
            self._flush()
        self.fp.write(str(b[off:off+length]))
        self.count += length

    def flush(self):
        self.fp.flush()
        
    def close(self):
        self.fp.seek(0)
        self.bytes = self.fp.read()
        self.fp.close()

    def get_bytes(self):
        return self.bytes
        
    def write_boolean(self, id_, v):
        BoolEncoder(id_, False, False)(self.write_, v)

    def write_int32(self, field_number, value):
        Int32Encoder(field_number, False, False)(self.write_, value)

    def write_sint32(self, field_number, value):
        SInt32Encoder(field_number, False, False)(self.write_, value)

    def write_uint32(self, field_number, value):
        UInt32Encoder(field_number, False, False)(self.write_, value)

    def write_long(self, id_, v):
        SInt64Encoder(id_, False, False)(self.write_, v)

    def write_double(self, id_, v):
        DoubleEncoder(id_, False, False)(self.write_, v)

    def write_raw_bytes(self, field_number, value):
        self.write_(TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED))
        _EncodeSignedVarint(self.write_, len(value))
        self.write_(value)

    def write_string(self, field_number, value):
        StringEncoder(field_number, False, False)(self.write_, value)


class ZlibFileObjectWriteWrapper(object):
    def __init__(self, fp, compress_obj):
        self.fp = fp
        self.compress_obj = compress_obj
        
    def write(self, obj):
        self.fp.write(self.compress_obj.compress(obj))
        
    def flush(self):
        self.fp.write(self.compress_obj.flush())
    
    def __getattr__(self, attr):
        return getattr(self.fp, attr)


class SnappyFileObjectWriteWrapper(object):
    def __init__(self, fp):
        self.fp = fp
        try:
            import snappy
        except ImportError:
            raise DependencyNotInstalledError(
                "python-snappy library is required for snappy support")
        self.stream = snappy.StreamCompressor()

    def write(self, obj):
        self.fp.write(self.stream.compress(obj))

    def __getattr__(self, attr):
        return getattr(self.fp, attr)


class ProtobufOutputWriter(object):
    def __init__(self, table_schema, fp, chunk_upload=None, compress_opt=None):
        self.columns = table_schema.columns
        if compress_opt is not None:
            if compress_opt.algorithm == \
                    CompressOption.CompressionAlgorithm.ODPS_ZLIB:
                compress_obj = zlib.compressobj(compress_opt.level,
                                                zlib.DEFLATED,
                                                zlib.MAX_WBITS,
                                                zlib.DEF_MEM_LEVEL,
                                                compress_opt.strategy)

                fp = ZlibFileObjectWriteWrapper(fp, compress_obj)
                self.stream = ProtobufOutputStream(fp)
            elif compress_opt.algorithm == \
                    CompressOption.CompressionAlgorithm.ODPS_SNAPPY:
                self.stream = ProtobufOutputStream(SnappyFileObjectWriteWrapper(fp))
            else:
                raise IOError('Invalid compression option.')
        else:
            self.stream = ProtobufOutputStream(fp)

        self.chunk_upload = chunk_upload
        
        self.crc = Checksum()
        self.crccrc = Checksum()
        self.count = 0L

    def write(self, record):
        record_values_length = record.get_columns_count()
        column_count = len(self.columns)
        if record_values_length > column_count:
            raise IOError('record values more than schema.')

        for i in range(min(record_values_length, column_count)):
            v = record.get(i)
            if v is None:
                continue

            pb_idx = i + 1
            self.crc.update_(pb_idx, force='int')

            t = self.columns[i].type.upper()
            if t == 'BOOLEAN':
                self.crc.update_(bool(v), force='boolean')
                self.stream.write_boolean(pb_idx, v)
            elif t == 'DATETIME':
                v = time.mktime(v.timetuple()) * 1000
                self.crc.update_(long(v), force='long')
                self.stream.write_long(pb_idx, long(v))
            elif t == 'STRING':
                if isinstance(v, unicode):
                    v = v.encode('utf-8')
                bytes_ = bytearray(v)
                self.crc.update(bytes_, 0, len(bytes_))
                self.stream.write_raw_bytes(pb_idx, bytes_)
            elif t == 'DOUBLE':
                self.crc.update_(float(v), force='float')
                self.stream.write_double(pb_idx, v)
            elif t == 'BIGINT':
                self.crc.update_(long(v), force='long')
                self.stream.write_long(pb_idx, v)
            else:
                raise IOError('Invalid data type: '+t)

        checksum = long_to_int(self.crc.get_value())
        self.stream.write_uint32(ProtoWireConstants.TUNNEL_END_RECORD, int_to_uint(checksum))

        self.crc.reset()
        self.crccrc.update_(checksum)

        self.count += 1

    def close(self):
        self.stream.write_long(ProtoWireConstants.TUNNEL_META_COUNT, self.count)
        self.stream.write_uint32(ProtoWireConstants.TUNNEL_META_CHECKSUM,
                                 long_to_uint(self.crccrc.get_value()))

        self.stream.flush()
        if self.chunk_upload is not None:
            self.chunk_upload()

        self.stream.close()
        self.count = 0
        
    def get_total_bytes(self):
        return self.stream.total_bytes

    def get_bytes(self):
        return self.stream.get_bytes()
    
    def __enter__(self):
        return self
    
    def __exit__(self, type_, value, traceback):
        self.close()
