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

import sys
import zlib
import threading

from ... import errors, compat
from ...compat import Enum, raise_exc

# used for test case to force thread io
_FORCE_THREAD = False


class RequestsIO(object):
    _async_err = None
    CHUNK_SIZE = 1024

    def __new__(cls, *args, **kwargs):
        if cls is RequestsIO:
            if not isinstance(threading.current_thread(), threading._MainThread) or _FORCE_THREAD:
                return object.__new__(ThreadRequestsIO)
            elif GreenletRequestsIO is not None:
                return object.__new__(GreenletRequestsIO)
            else:
                return object.__new__(ThreadRequestsIO)
        else:
            return object.__new__(cls)

    def __init__(self, post_call, chunk_size=None):
        self._queue = None
        self._resp = None
        self._async_err = None
        self._chunk_size = chunk_size or self.CHUNK_SIZE

        def async_func():
            try:
                self._resp = post_call(self.data_generator())
            except:
                self._async_err = sys.exc_info()
            self._wait_obj = None

        self._async_func = async_func
        self._wait_obj = None

    def data_generator(self):
        while True:
            data = self.get()
            if data is not None:
                while data:
                    to_send = data[:self._chunk_size]
                    data = data[self._chunk_size:]
                    yield to_send
            else:
                break

    def start(self):
        pass

    def get(self):
        return self._queue.get()

    def put(self, data):
        self._queue.put(data)

    def write(self, data):
        if self._async_err is not None:
            ex_type, ex_value, tb = self._async_err
            raise_exc(ex_type, ex_value, tb)
        self.put(data)

    def flush(self):
        pass

    def finish(self):
        self.put(None)
        if self._wait_obj and self._wait_obj.is_alive():
            self._wait_obj.join()
        if self._async_err is None:
            return self._resp
        else:
            ex_type, ex_value, tb = self._async_err
            raise_exc(ex_type, ex_value, tb)


class ThreadRequestsIO(RequestsIO):
    def __init__(self, post_call, chunk_size=None):
        super(ThreadRequestsIO, self).__init__(post_call, chunk_size)
        from ...compat import Queue
        self._queue = Queue()
        self._wait_obj = threading.Thread(target=self._async_func)
        self._wait_obj.daemon = True

    def start(self):
        self._wait_obj.start()


try:
    from greenlet import greenlet

    class GreenletRequestsIO(RequestsIO):
        def __init__(self, post_call, chunk_size=None):
            super(GreenletRequestsIO, self).__init__(post_call, chunk_size)
            self._cur_greenlet = greenlet.getcurrent()
            self._writer_greenlet = greenlet(self._async_func)
            self._last_data = None
            self._writer_greenlet.switch()

        def get(self):
            self._cur_greenlet.switch()
            return self._last_data

        def put(self, data):
            self._last_data = data
            # handover control
            self._writer_greenlet.switch()

except ImportError:
    GreenletRequestsIO = None


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


class SimpleInputStream(object):

    READ_BLOCK_SIZE = 1024 * 10

    def __init__(self, input):
        self._input = input
        self._internal_buffer = compat.BytesIO()
        self._buffered_len = 0

    def read(self, limit):
        if limit <= self._buffered_len - self._internal_buffer.tell():
            return self._internal_buffer.read(limit)

        bufs = list()
        size_left = limit
        while size_left > 0:
            content = self._internal_read(size_left)
            if not content:
                break
            bufs.append(content)
            size_left -= len(content)
        return bytes().join(bufs)

    def _internal_read(self, limit):
        if self._internal_buffer.tell() == self._buffered_len:
            self._refill_buffer()
        b = self._internal_buffer.read(limit)
        return b

    def _refill_buffer(self):
        self._buffered_len = 0

        buffer = []
        left = self.READ_BLOCK_SIZE
        while True:
            content = self._buffer_next_chunk()
            if content is None:
                break
            if content:
                length = len(content)
                left -= length
                self._buffered_len += length
                buffer.append(content)
                break

        self._internal_buffer = compat.BytesIO(bytes().join(buffer))

    def _read_block(self):
        content = self._input.read(self.READ_BLOCK_SIZE)
        return content if content else None

    def _buffer_next_chunk(self):
        return self._read_block()


class RequestsInputStream(SimpleInputStream):
    def _read_block(self):
        content = self._input.raw.read(self.READ_BLOCK_SIZE, decode_content=True)
        return content if content else None


class DeflateInputStream(SimpleInputStream):
    def __init__(self, input):
        super(DeflateInputStream, self).__init__(input)
        self._decompressor = zlib.decompressobj(zlib.MAX_WBITS)

    def _buffer_next_chunk(self):
        data = self._read_block()
        if data is None:
            return None
        if data:
            return self._decompressor.decompress(data)
        else:
            return self._decompressor.flush()


class SnappyInputStream(SimpleInputStream):

    READ_BLOCK_SIZE = 1024

    def __init__(self, input):
        super(SnappyInputStream, self).__init__(input)
        try:
            import snappy
        except ImportError:
            raise errors.DependencyNotInstalledError(
                "python-snappy library is required for snappy support")
        self._decompressor = snappy.StreamDecompressor()

    def _buffer_next_chunk(self):
        data = self._read_block()
        if data is None:
            return None
        if data:
            return self._decompressor.decompress(data)
        else:
            return self._decompressor.flush()


class SnappyRequestsInputStream(SnappyInputStream):
    def _read_block(self):
        content = self._input.raw.read(self.READ_BLOCK_SIZE, decode_content=False)
        return content if content else None
