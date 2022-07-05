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
from odps.compat import six, BytesIO
from odps.tunnel.io.reader import TunnelRecordReader
from odps.tunnel.io.writer import BaseRecordWriter

from cupid.errors import SubprocessStreamEOFError

logger = logging.getLogger(__name__)


class CupidRecordReader(TunnelRecordReader):
    def __init__(self, schema, input_stream, columns=None):
        super(CupidRecordReader, self).__init__(schema, input_stream, columns=columns)
        self._stream = input_stream
        self._table_schema = schema
        self._input_columns = columns
        self._mp_stream = None

    def to_forkable(self):
        return type(self)(self._table_schema, self._create_mp_stream(), columns=self._input_columns)

    def _create_mp_stream(self):
        if isinstance(self._stream, CupidMPInputStream):
            return self._stream
        if self._mp_stream is not None:
            return self._mp_stream

        from multiprocessing.sharedctypes import RawArray
        req_queue = multiprocessing.Queue()
        rep_queue = multiprocessing.Queue()
        buf = RawArray(ctypes.c_char, options.cupid.mp_buffer_size)

        def _mp_thread():
            try:
                while True:
                    read_size = req_queue.get()
                    if read_size < 0:
                        return
                    try:
                        read_size = self._stream.readinto(buf)
                    except SubprocessStreamEOFError:
                        rep_queue.put(-1)
                        return
                    rep_queue.put(read_size)
            finally:
                self.close()

        stream_thread = threading.Thread(target=_mp_thread)
        stream_thread.daemon = True
        stream_thread.start()
        self._mp_stream = CupidMPInputStream(buf, req_queue, rep_queue)
        return self._mp_stream

    def read(self):
        try:
            return super(CupidRecordReader, self).read()
        except SubprocessStreamEOFError:
            return None

    def close(self):
        super(CupidRecordReader, self).close()
        self._stream.close()


class CupidMPInputStream(object):
    def __init__(self, share_buf=None, req_queue=None, rep_queue=None):
        self._buf = share_buf
        self._buf_len = 0
        self._pos = 0
        self._q_req = req_queue
        self._q_rep = rep_queue
        self._eof = False

    def read(self, size):
        if self._eof:
            raise SubprocessStreamEOFError
        if self._pos + size <= self._buf_len:
            buf = bytes(self._buf[self._pos:self._pos + size])
            self._pos += size
            return buf
        else:
            size_left = size
            bio = BytesIO()
            while size_left:
                c = self._buf[self._pos:min(self._pos + size_left, self._buf_len)]
                bio.write(c)
                size_left -= len(c)
                self._pos += len(c)
                if size_left <= 0:
                    break
                self._q_req.put(options.cupid.mp_buffer_size)
                recv_size = self._q_rep.get()
                if recv_size < 0:
                    self._eof = True
                    break
                self._buf_len = recv_size
                self._pos = 0
            return bio.getvalue()

    def close(self):
        self._q_req.put(-1)


class CupidRecordWriter(BaseRecordWriter):
    def __init__(self, schema, output_stream, encoding='utf-8'):
        super(CupidRecordWriter, self).__init__(schema, output_stream, encoding=encoding)
        self._stream = output_stream
        self._block_id = None
        self._partition_spec = None
        self._table_schema = schema
        self._mp_stream = None

    def __getstate__(self):
        return self._table_schema, self._create_mp_stream(), self._block_id, self._partition_spec

    def __setstate__(self, state):
        self.__init__(*state[:2])
        self._block_id, self._partition_spec = state[2:]

    def _create_mp_stream(self):
        if isinstance(self._stream, CupidMPInputStream):
            return self._mp_stream
        if self._mp_stream is not None:
            return self._mp_stream

        from multiprocessing.sharedctypes import RawArray
        req_queue = multiprocessing.Queue()
        rep_queue = multiprocessing.Queue()
        buf = RawArray(ctypes.c_char, options.cupid.mp_buffer_size)

        def _mp_thread():
            try:
                while True:
                    size = req_queue.get()
                    if size < 0:
                        break
                    self.write(buf[:size])
                    rep_queue.put(size)
            finally:
                self.close()

        stream_thread = threading.Thread(target=_mp_thread)
        stream_thread.daemon = True
        stream_thread.start()
        self._mp_stream = CupidMPOutputStream(buf, req_queue, rep_queue)
        return self._mp_stream

    @property
    def block_id(self):
        return self._block_id

    @property
    def partition_spec(self):
        return self._partition_spec

    def close(self):
        super(CupidRecordWriter, self).close()

        # sync by get result
        result = self._stream.result()
        logger.debug('Result fetched on writer close: %s', result)

        self._stream.close()


class CupidMPOutputStream(object):
    def __init__(self, share_buf=None, req_queue=None, rep_queue=None):
        self._buf = share_buf
        self._buf_len = len(share_buf)
        self._pos = 0
        self._q_req = req_queue
        self._q_rep = rep_queue

    def write(self, data):
        data_pos = 0
        data_len = len(data)
        while data_pos < data_len:
            buffer_bound = min(self._buf_len, self._pos + data_len - data_pos)
            data_bound = data_pos + buffer_bound - self._pos
            self._buf[self._pos:buffer_bound] = data[data_pos:data_bound]
            data_pos = data_bound
            self._pos = buffer_bound

            if self._pos == self._buf_len:
                self.flush()
        self.flush()

    def flush(self):
        if self._pos == 0:
            return
        self._q_req.put(self._pos)
        self._q_rep.get()
        self._pos = 0

    def close(self):
        self._q_req.put(-1)
