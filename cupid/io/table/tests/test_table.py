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

import random
import sys
import threading
from unittest import TestCase
from cupid.io.table.record import CupidMPInputStream, CupidMPOutputStream, SubprocessStreamEOFError
from cupid.config import options

if sys.version_info[0] < 3:
    from Queue import Queue
    from cStringIO import StringIO as BytesIO
else:
    from queue import Queue
    from io import BytesIO

TEST_BYTES = list('abcdefghijklnmopqrstuvwxyz1234567890' * 4096)
random.shuffle(TEST_BYTES)
TEST_BYTES = ''.join(TEST_BYTES).encode()


class Test(TestCase):
    def testMPInputStream(self):
        req_queue = Queue()
        rep_queue = Queue()

        array_buf = bytearray(options.cupid.mp_buffer_size)

        def _supply_thread():
            pos = 0
            while True:
                size = req_queue.get()
                if size < 0:
                    break
                chunk = TEST_BYTES[pos:pos + size]
                array_buf[:len(chunk)] = chunk
                pos += len(chunk)
                if len(chunk) == 0:
                    rep_queue.put(-1)
                else:
                    rep_queue.put(len(chunk))

        th = threading.Thread(target=_supply_thread)
        th.daemon = True
        th.start()

        stream = CupidMPInputStream(array_buf, req_queue, rep_queue)
        bio = BytesIO()
        while True:
            try:
                buf = stream.read(1000)
            except SubprocessStreamEOFError:
                break
            bio.write(buf)
        self.assertEqual(bio.getvalue(), TEST_BYTES)

    def testMPOutputStream(self):
        req_queue = Queue()
        rep_queue = Queue()

        array_buf = bytearray(options.cupid.mp_buffer_size)

        bio = BytesIO()

        def _consume_thread():
            while True:
                size = req_queue.get()
                if size < 0:
                    break
                bio.write(array_buf[:size])
                rep_queue.put(size)

        th = threading.Thread(target=_consume_thread)
        th.daemon = True
        th.start()

        stream = CupidMPOutputStream(array_buf, req_queue, rep_queue)
        pos = 0
        while pos < len(TEST_BYTES):
            stream.write(TEST_BYTES[pos:pos + 10])
            pos += 10
        self.assertEqual(bio.getvalue(), TEST_BYTES)
