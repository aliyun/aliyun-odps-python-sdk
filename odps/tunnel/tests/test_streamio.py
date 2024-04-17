#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import io
import itertools
import random
import threading
import time
import traceback
try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters  # noqa: F401

import mock
import pytest

from ..io import stream as io_stream


TEXT = u"""
    上善若水。水善利万物而不争，处众人之所恶，故几於道。居善地，心善渊，与善仁，言善信，正善

治，事善能，动善时。夫唯不争，故无尤。

    绝学无忧，唯之与阿，相去几何？善之与恶，相去若何？人之所畏，不可不畏。荒兮其未央哉！众人

熙熙如享太牢、如春登台。我独泊兮其未兆，如婴儿之未孩；儡儡兮若无所归。众人皆有馀，而

我独若遗。我愚人之心也哉！沌沌兮。俗人昭昭，我独昏昏；俗人察察，我独闷闷。众人皆有以，而我独

顽且鄙。我独异於人，而贵食母。

     宠辱若惊，贵大患若身。何谓宠辱若惊？宠为下。得之若惊失之若惊是谓宠辱若惊。何谓贵大患若身

？吾所以有大患者，为吾有身，及吾无身，吾有何患。故贵以身为天下，若可寄天下。爱以身为天下，若

可托天下。"""


@pytest.fixture(autouse=True)
def fix_config():
    try:
        yield
    finally:
        io_stream._FORCE_THREAD = False


@pytest.mark.parametrize("compress_algo, package", [
    (io_stream.CompressOption.CompressAlgorithm.ODPS_RAW, None),
    (io_stream.CompressOption.CompressAlgorithm.ODPS_ZLIB, None),
    (io_stream.CompressOption.CompressAlgorithm.ODPS_SNAPPY, "snappy"),
    (io_stream.CompressOption.CompressAlgorithm.ODPS_ZSTD, "zstandard"),
    (io_stream.CompressOption.CompressAlgorithm.ODPS_LZ4, "lz4.frame"),
])
def test_compress_and_decompress(compress_algo, package):
    if package is not None:
        pytest.importorskip(package)

    tube = io.BytesIO()
    option = io_stream.CompressOption(compress_algo)

    data_bytes = TEXT.encode('utf-8')

    outstream = io_stream.get_compress_stream(tube, option)
    for pos in range(0, len(data_bytes), 128):
        outstream.write(data_bytes[pos : pos + 128])
    outstream.flush()

    tube.seek(0)
    instream = io_stream.get_decompress_stream(tube, option, requests=False)

    b = bytearray()
    while True:
        part = instream.read(1)
        if not part:
            break
        b += part

    assert TEXT.encode('utf8') == b

    tube.seek(0)
    instream = io_stream.get_decompress_stream(tube, option, requests=False)

    b = bytearray(len(TEXT.encode('utf8')))
    mv = memoryview(b)
    pos = 0
    while True:
        incr = instream.readinto(mv[pos:pos + 1])
        if not incr:
            break
        pos += incr

    assert TEXT.encode('utf8') == b


def test_class():
    io_stream._FORCE_THREAD = False

    req_io = io_stream.RequestsIO(lambda c: None)
    if io_stream.GreenletRequestsIO is None:
        assert isinstance(req_io, io_stream.ThreadRequestsIO)
    else:
        assert isinstance(req_io, io_stream.GreenletRequestsIO)
    assert isinstance(io_stream.ThreadRequestsIO(lambda c: None), io_stream.ThreadRequestsIO)
    if io_stream.GreenletRequestsIO is not None:
        assert isinstance(io_stream.GreenletRequestsIO(lambda c: None), io_stream.GreenletRequestsIO)

    io_stream._FORCE_THREAD = True

    req_io = io_stream.RequestsIO(lambda c: None)
    assert isinstance(req_io, io_stream.ThreadRequestsIO)
    assert isinstance(io_stream.ThreadRequestsIO(lambda c: None), io_stream.ThreadRequestsIO)
    if io_stream.GreenletRequestsIO is not None:
        assert isinstance(io_stream.GreenletRequestsIO(lambda c: None), io_stream.GreenletRequestsIO)


@pytest.fixture
def semaphore_random_delay(request):
    try:
        original_acquire = threading._Semaphore.acquire
        method_name = "threading._Semaphore.acquire"
    except AttributeError:
        original_acquire = threading.Semaphore.acquire
        method_name = "threading.Semaphore.acquire"

    if not request.param:
        yield
    else:
        def new_acquire(self, *args, **kw):
            time.sleep(random.random() / 4.0)
            ret = original_acquire(self, *args, **kw)
            time.sleep(random.random() / 4.0)
            return ret

        with mock.patch(method_name, new=new_acquire):
            yield


@pytest.mark.parametrize(
    "force_thread, semaphore_random_delay",
    itertools.product([False, True], [False, True]),
    indirect=["semaphore_random_delay"],
)
def test_raises(force_thread, semaphore_random_delay):
    exc_trace = [None]
    io_stream._FORCE_THREAD = force_thread

    def raise_poster(it):
        exc_trace[0] = None
        next(it)
        try:
            raise AttributeError
        except:
            tb = traceback.format_exc().splitlines()
            exc_trace[0] = '\n'.join(tb[-3:])
            raise

    req_io = io_stream.RequestsIO(raise_poster, chunk_size=5)
    req_io.start()
    try:
        req_io.write(b'TEST_DATA')
        req_io.write(b'ANOTHER_PIECE')
        req_io.write(b'THIS_SHALL_RAISE')
        assert False, "Statement above not raised"
    except AttributeError:
        tb = traceback.format_exc().splitlines()
        assert '\n'.join(tb[-3:]) == exc_trace[0]


@pytest.mark.parametrize(
    "force_thread, semaphore_random_delay",
    itertools.product([False, True], [False, True]),
    indirect=["semaphore_random_delay"],
)
def test_large_writes(force_thread, semaphore_random_delay):
    chunk_size = 103
    repeats = 7
    recv_chunks = []

    io_stream._FORCE_THREAD = force_thread

    def check_poster(it):
        for chunk in it:
            assert len(chunk) <= chunk_size
            recv_chunks.append(bytes(bytearray(chunk)))

    req_io = io_stream.RequestsIO(check_poster, chunk_size=chunk_size)
    req_io.start()
    for _ in range(repeats):
        req_io.write(TEXT.encode('utf-8'))
    req_io.finish()

    assert b"".join(recv_chunks) == TEXT.encode('utf-8') * repeats
