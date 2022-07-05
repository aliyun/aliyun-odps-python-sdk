# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import ctypes
import threading
import multiprocessing
import logging

from odps import options
from odps.tunnel.pdio import TunnelPandasReader, BasePandasWriter

from cupid.errors import SubprocessStreamEOFError

import numpy as np
del np

logger = logging.getLogger(__name__)


class CupidPandasReader(TunnelPandasReader):
    def __init__(self, schema, input_stream, columns=None):
        if isinstance(input_stream, tuple):
            self._refill_data = input_stream
            input_stream = None
        else:
            self._refill_data = None

        super(CupidPandasReader, self).__init__(schema, input_stream, columns=columns)

        self._input_stream = input_stream
        self._table_schema = schema
        self._input_columns = columns
        self._stream_eof = False
        self._closed = False

    def to_forkable(self):
        return type(self)(self._table_schema, self._build_refill_data(), self._input_columns)

    def __repr__(self):
        cls = type(self)
        if self._refill_data is not None:
            return '<%s.%s (slave) at 0x%x>' % (cls.__module__, cls.__name__, id(self))
        else:
            return '<%s.%s at 0x%x>' % (cls.__module__, cls.__name__, id(self))

    def _build_refill_data(self):
        if self._input_stream is None:
            return self._refill_data
        if self._refill_data is not None:
            return self._refill_data

        from multiprocessing.sharedctypes import RawArray
        req_queue = multiprocessing.Queue()
        rep_queue = multiprocessing.Queue()
        buf = RawArray(ctypes.c_char, options.cupid.mp_buffer_size)

        def _mp_thread():
            try:
                while True:
                    req_body = req_queue.get(timeout=60)
                    if req_body is None:
                        return
                    left_size, bound = req_body
                    try:
                        buf[:left_size] = buf[bound - left_size:bound]
                        read_size = self._input_stream.readinto(buf, left_size)
                    except SubprocessStreamEOFError:
                        return
                    rep_queue.put(read_size)
            finally:
                rep_queue.put(-1)
                self.close()

        stream_thread = threading.Thread(target=_mp_thread)
        stream_thread.daemon = True
        stream_thread.start()
        self._refill_data = (buf, req_queue, rep_queue)
        return self._refill_data

    def refill_cache(self):
        if self._refill_data is None:
            return super(CupidPandasReader, self).refill_cache()
        if self._stream_eof or self._closed:
            return 0

        buf, req_queue, rep_queue = self._refill_data
        left_size = self.mem_cache_bound - self.row_mem_ptr
        req_queue.put((left_size, self.mem_cache_bound))
        read_size = rep_queue.get(timeout=60)
        if read_size <= 0:
            self._stream_eof = True
            self.close()
            return 0
        self.reset_positions(buf, read_size + left_size)
        return read_size

    def close(self):
        super(CupidPandasReader, self).close()
        if self._input_stream is None and self._refill_data:
            buf, req_queue, rep_queue = self._refill_data
            req_queue.put(None)
        self._closed = True


class CupidPandasWriter(BasePandasWriter):
    def __init__(self, schema, output_stream):
        super(CupidPandasWriter, self).__init__(schema, output_stream)
        self._stream = output_stream
        self._block_id = None
        self._partition_spec = None
        self._table_schema = schema

    @property
    def block_id(self):
        return self._block_id

    @property
    def partition_spec(self):
        return self._partition_spec

    def write_stream(self, data, length):
        self._stream.write(data, length)

    def close(self):
        super(CupidPandasWriter, self).close()

        # sync by get result
        result = self._stream.result()
        logger.debug('Result fetched on writer close: %s', result)

        self._stream.close()
