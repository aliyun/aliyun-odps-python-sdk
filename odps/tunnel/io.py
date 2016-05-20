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
import math

import threading
from odps import errors, compat, options
from odps import errors
from odps import compat
from odps.compat import six
from odps.compat import Enum


class BaseRequestsIO(object):
    ASYNC_ERR = None
    CHUNK_SIZE = 1024

    def __init__(self, post_call, chunk_size=None):
        self._queue = None
        self._resp = None
        self._chunk_size = chunk_size or self.CHUNK_SIZE

        def data_generator():
            while True:
                data = self.get()
                if data is not None:
                    while data:
                        to_send = data[:self._chunk_size]
                        data = data[self._chunk_size:]
                        yield to_send
                else:
                    break

        def async_func():
            try:
                self._resp = post_call(data_generator())
            except Exception as e:
                self.ASYNC_ERR = e
                raise e

        self._async_func = async_func
        self._wait_obj = None

    def start(self):
        pass

    def get(self):
        return self._queue.get()

    def put(self, data):
        self._queue.put(data)

    def write(self, data):
        self._queue.put(data)

    def flush(self):
        pass

    def finish(self):
        self._queue.put(None)
        if self._wait_obj:
            self._wait_obj.join()
        return self._resp


class ThreadRequestsIO(BaseRequestsIO):
    def __init__(self, post_call, chunk_size=None):
        super(ThreadRequestsIO, self).__init__(post_call, chunk_size)
        from odps.compat import Queue
        self._queue = Queue()
        self._wait_obj = threading.Thread(target=self._async_func)

    def start(self):
        self._wait_obj.start()


class GreenletRequestsIO(BaseRequestsIO):
    def __init__(self, post_call, chunk_size=None):
        super(GreenletRequestsIO, self).__init__(post_call, chunk_size)
        import gevent
        from gevent.queue import Queue
        self._queue = Queue()
        self._wait_obj = gevent.spawn(self._async_func)
        self._gevent_mod = gevent

    def put(self, data):
        super(GreenletRequestsIO, self).put(data)
        # handover control
        self._gevent_mod.sleep(0)

RequestIO = ThreadRequestsIO


def reload_default_io():
    global RequestsIO
    try:
        import gevent
        RequestsIO = GreenletRequestsIO
    except ImportError:
        RequestsIO = ThreadRequestsIO

reload_default_io()


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


class DeflateOutputStream(object):

    def __init__(self, output, level=1):
        self._compressor = zlib.compressobj(level)
        self._output = output

    def write(self, data):
        if self._compressor:
            compressed_data = self._compressor.compress(data)
            if compressed_data:
                self._output.write(compressed_data)
            else:
                pass  # buffering
        else:
            self._output.write(data)

    def flush(self):
        if self._compressor:
            remaining = self._compressor.flush()
            if remaining:
                self._output.write(remaining)


class SnappyOutputStream(object):

    def __init__(self, output):
        try:
            import snappy
        except ImportError:
            raise errors.DependencyNotInstalledError(
                "python-snappy library is required for snappy support")
        self._compressor = snappy.StreamCompressor()
        self._output = output

    def write(self, data):
        if self._compressor:
            compressed_data = self._compressor.compress(data)
            if compressed_data:
                self._output.write(compressed_data)
            else:
                pass  # buffering
        else:
            self._output.write(data)

    def flush(self):
        if self._compressor:
            remaining = self._compressor.flush()
            if remaining:
                self._output.write(remaining)


class DeflateInputStream(object):

    READ_BLOCK_SIZE = 1024

    def __init__(self, input):
        self._decompressor = zlib.decompressobj(zlib.MAX_WBITS)
        self._input = input
        self._internal_buffer = compat.BytesIO()
        self._cursor = 0

    def read(self, limit):
        if self._internal_buffer.tell() == self._cursor:
            self._refill_buffer()
        b = self._internal_buffer.read(limit)
        return b

    def _refill_buffer(self):
        while True:
            data = self._input.read(self.READ_BLOCK_SIZE)
            if data:
                decompressed = self._decompressor.decompress(data)
                if decompressed:
                    self._internal_buffer.write(decompressed)
                    self._cursor += len(decompressed)
                    self._internal_buffer.seek(0)
                    break
                else:
                    pass  # buffering
            else:
                remainder = self._decompressor.flush()
                if remainder:
                    self._internal_buffer.write(remainder)
                    self._cursor += len(remainder)
                    self._internal_buffer.seek(0)
                else:
                    break


class SnappyInputStream(object):

    READ_BLOCK_SIZE = 1024

    def __init__(self, input):
        try:
            import snappy
        except ImportError:
            raise errors.DependencyNotInstalledError(
                "python-snappy library is required for snappy support")
        self._decompressor = snappy.StreamDecompressor()
        self._input = input
        self._internal_buffer = compat.BytesIO()
        self._cursor = 0

    def read(self, limit):
        if self._internal_buffer.tell() == self._cursor:
            self._refill_buffer()
        b = self._internal_buffer.read(limit)
        return b

    def _refill_buffer(self):
        while True:
            data = self._input.read(self.READ_BLOCK_SIZE)
            if data:
                decompressed = self._decompressor.decompress(data)
                if decompressed:
                    self._internal_buffer.write(decompressed)
                    self._cursor += len(decompressed)
                    self._internal_buffer.seek(0)
                    break
                else:
                    pass  # buffering
            else:
                remainder = self._decompressor.flush()
                if remainder:
                    self._internal_buffer.write(remainder)
                    self._cursor += len(remainder)
                    self._internal_buffer.seek(0)
                else:
                    break
