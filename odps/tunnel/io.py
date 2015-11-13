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

from enum import Enum
import six
from six.moves import queue as Queue
from google.protobuf.internal.encoder import _EncodeSignedVarint, _EncodeVarint
from google.protobuf.internal.encoder import *
from google.protobuf.internal.decoder import _DecodeSignedVarint32
from google.protobuf.internal.decoder import *
from google.protobuf.internal import wire_format

from odps import errors


class CompressOption(object):

    class CompressAlgorithm(Enum):
        ODPS_RAW = 'RAW'
        ODPS_ZLIB = 'ZLIB'
        ODPS_SNAPPY = 'SNAPPY'

    def __init__(self, compress_algo=CompressAlgorithm.ODPS_ZLIB,
                 level=None, strategy=None):
        if isinstance(compress_algo, CompressOption.CompressAlgorithm):
            self.algorithm = compress_algo
        else:
            self.algorithm = \
                CompressOption.CompressAlgorithm(compress_algo.upper())
        self.level = level or 1
        self.strategy = strategy or 0


class ProtobufWriter(object):

    BUFFER_SIZE = 4096

    class ZlibBuffer(six.BytesIO):
        def __init__(self, compressobj, *args, **kwargs):
            self._compressobj = compressobj
            six.BytesIO.__init__(self, *args, **kwargs)

        def write(self, data):
            if isinstance(data, bytearray):
                data = six.binary_type(data)
            six.BytesIO.write(self, self._compressobj.compress(data))

        def flush(self):
            six.BytesIO.write(self, self._compressobj.flush())

    class SnappyBuffer(six.BytesIO):
        def __init__(self, *args, **kwargs):
            try:
                import snappy
            except ImportError:
                raise errors.DependencyNotInstalledError(
                    "python-snappy library is required for snappy support")
            self._compressobj = snappy.StreamCompressor()
            six.BytesIO.__init__(self, *args, **kwargs)

        def write(self, data):
            if isinstance(data, bytearray):
                data = six.binary_type(data)
            content = self._compressobj.compress(data)
            six.BytesIO.write(self, content)

    def __init__(self, compress_option=None, buffer_size=None, encoding='utf-8'):
        self._buffer = ProtobufWriter._get_buffer(compress_option)
        self._queue = Queue.Queue()

        self._buffer_size = buffer_size or self.BUFFER_SIZE
        self._curr_cursor = 0
        self._n_total = 0

        self._encoding = encoding

    @classmethod
    def _get_buffer(cls, compress_option=None):
        if compress_option is None or \
                        compress_option == CompressOption.CompressAlgorithm.ODPS_RAW:
            return six.BytesIO()
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_ZLIB:
            compress_obj = zlib.compressobj(
                compress_option.level, zlib.DEFLATED,
                zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL,
                compress_option.strategy)
            return ProtobufWriter.ZlibBuffer(compress_obj)
        elif compress_option.algorithm == \
                CompressOption.CompressAlgorithm.ODPS_SNAPPY:
            return ProtobufWriter.SnappyBuffer()
        else:
            raise IOError('Invalid compression option.')

    def flush(self):
        self._buffer.flush()
        if self._curr_cursor > 0:
            self._queue.put(self._buffer.getvalue())
            self._buffer = six.BytesIO()

            self._curr_cursor = 0

    def close(self):
        self.flush()
        self._queue.put(None)  # put None to remind the receiver that it is closed

        self._buffer.close()

    @property
    def n_bytes(self):
        return self._n_total

    def __len__(self):
        return self.n_bytes

    def _write(self, b, off=None, length=None):
        if isinstance(b, six.text_type):
            b = b.encode(self._encoding)

        off = off or 0
        rest = len(b) - off
        length = length or len(b)-off
        length = min(length, rest)

        self._buffer.write(b[off: off+length])

        self._curr_cursor += length
        self._n_total += length

        if self._curr_cursor >= self._buffer_size:
            self.flush()

    def write_bool(self, field_num, val):
        BoolEncoder(field_num, False, False)(self._write, val)

    def write_int32(self, field_num, val):
        Int32Encoder(field_num, False, False)(self._write, val)

    def write_sint32(self, field_num, val):
        SInt32Encoder(field_num, False, False)(self._write, val)

    def write_uint32(self, field_num, val):
        UInt32Encoder(field_num, False, False)(self._write, val)

    def write_long(self, field_num, val):
        SInt64Encoder(field_num, False, False)(self._write, val)

    def write_double(self, field_num, val):
        DoubleEncoder(field_num, False, False)(self._write, val)

    def write_string(self, field_num, val):
        if isinstance(val, six.text_type):
            val = val.encode(self._encoding)

        self.write_length_delimited_tag(field_num)
        self.write_raw_varint32(len(val))  # write length
        self._write(val)

    write_raw_bytes = _write

    def write_length_delimited_tag(self, field_num):
        self.write_tag(field_num, wire_format.WIRETYPE_LENGTH_DELIMITED)

    def write_tag(self, field_num, tag):
        self._write(TagBytes(field_num, tag))

    def write_raw_varint32(self, val):
        _EncodeSignedVarint(self._write, val)

    def write_long_no_tag(self, val):
        _EncodeVarint(self._write, wire_format.ZigZagEncode(val))

    def write_double_no_tag(self, val):
        self._write(struct.pack('<d', val))

    def write_bool_no_tag(self, val):
        val = bytearray([1]) if val else bytearray([0])
        self._write(val)

    def __next__(self):
        """
        make it iterable to support chunk upload,
        remember to do the iteration in a separate thread,
        or the queue will block current thread
        """
        val = self._queue.get()
        if val is not None:
            return val
        else:
            raise StopIteration

    next = __next__

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ProtobufReader(object):
    def __init__(self, stream, compress_option=None, encoding='utf-8'):
        self._encoding = encoding
        self._data = ProtobufReader._get_data(
            stream, compress_option=compress_option, encoding=encoding)

        self._curr_cursor = 0
        self._n_totals = len(self._data)

    @classmethod
    def _get_data(cls, stream, encoding='utf-8', compress_option=None):
        if isinstance(stream, six.text_type):
            data = stream.encode(encoding)
        elif isinstance(stream, six.binary_type):
            data = stream
        else:
            data = stream.read()  # due to the restriction of protobuf api, just read the data all
            stream.close()  # directly close the stream
            if isinstance(data, six.text_type):
                data = data.encode(encoding)

        if compress_option is None or \
                compress_option.algorithm == CompressOption.CompressAlgorithm.ODPS_RAW:
            return data
        elif compress_option.algorithm == CompressOption.CompressAlgorithm.ODPS_ZLIB:
            return data  # because requests do the unzip automatically, thanks to them O.O
        elif compress_option.algorithm == CompressOption.CompressAlgorithm.ODPS_SNAPPY:
            try:
                import snappy
            except ImportError:
                raise errors.DependencyNotInstalledError(
                    'python-snappy library is required for snappy support')
            data = snappy.decompress(data)
            return data
        else:
            raise IOError('invalid compression option.')

    @property
    def n_bytes(self):
        return self._n_totals

    def __len__(self):
        return self.n_bytes

    def _decode_value(self, decoder, new_default=None, msg=None):
        # tricky operations due to the hidden protobuf api which we need to hack into
        decode_key = '__decode_key'
        decode_dict = {}
        decode = decoder(None, False, False, decode_key, new_default)
        self._curr_cursor = \
            decode(self._data, self._curr_cursor, self._n_totals, msg, decode_dict)
        return decode_dict[decode_key]

    def read_field_num(self):
        tag, self._curr_cursor = _DecodeSignedVarint32(self._data, self._curr_cursor)
        return wire_format.UnpackTag(tag)[0]

    def read_bool(self):
        val, self._curr_cursor = _DecodeSignedVarint32(self._data, self._curr_cursor)
        return val != 0

    def read_sint32(self):
        return self._decode_value(SInt32Decoder)

    def read_uint32(self):
        return self._decode_value(UInt32Decoder)

    def read_long(self):
        return self._decode_value(SInt64Decoder)

    def read_double(self):
        return self._decode_value(DoubleDecoder)

    def read_string(self):
        return bytearray(self._decode_value(BytesDecoder, bytearray, ''))

    def at_end(self):
        return self._curr_cursor >= self._n_totals