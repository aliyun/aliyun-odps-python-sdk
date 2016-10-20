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

from odps.tests.core import TestBase, snappy_case
from odps.compat import unittest, BytesIO
from odps.tunnel import io as tio

import io
import traceback

try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters


class Test(TestBase):
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

    def tearDown(self):
        super(Test, self).tearDown()
        tio._FORCE_THREAD = False

    def testCompressAndDecompressDeflate(self):
        tube = io.BytesIO()

        outstream = tio.DeflateOutputStream(tube)
        outstream.write(self.TEXT.encode('utf8'))
        outstream.flush()

        tube.seek(0)
        instream = tio.DeflateInputStream(tube)

        b = bytearray()
        while True:
            part = instream.read(1)
            if not part:
                break
            b += part

        self.assertEquals(self.TEXT.encode('utf8'), b)

    @snappy_case
    def testCompressAndDecompressSnappy(self):
        tube = io.BytesIO()

        outstream = tio.SnappyOutputStream(tube)
        outstream.write(self.TEXT.encode('utf8'))
        outstream.flush()

        tube.seek(0)
        instream = tio.SnappyInputStream(tube)

        b = bytearray()
        while True:
            part = instream.read(1)
            if not part:
                break
            b += part

        self.assertEquals(self.TEXT.encode('utf8'), b)

    def testClass(self):
        tio._FORCE_THREAD = False

        req_io = tio.RequestsIO(lambda c: None)
        if tio.GreenletRequestsIO is None:
            self.assertIsInstance(req_io, tio.ThreadRequestsIO)
        else:
            self.assertIsInstance(req_io, tio.GreenletRequestsIO)
        self.assertIsInstance(tio.ThreadRequestsIO(lambda c: None), tio.ThreadRequestsIO)
        if tio.GreenletRequestsIO is not None:
            self.assertIsInstance(tio.GreenletRequestsIO(lambda c: None), tio.GreenletRequestsIO)

        tio._FORCE_THREAD = True

        req_io = tio.RequestsIO(lambda c: None)
        self.assertIsInstance(req_io, tio.ThreadRequestsIO)
        self.assertIsInstance(tio.ThreadRequestsIO(lambda c: None), tio.ThreadRequestsIO)
        if tio.GreenletRequestsIO is not None:
            self.assertIsInstance(tio.GreenletRequestsIO(lambda c: None), tio.GreenletRequestsIO)

    def testRaises(self):
        exc_trace = [None]

        def raise_poster(it):
            exc_trace[0] = None
            next(it)
            try:
                raise AttributeError
            except:
                tb = traceback.format_exc().splitlines()
                exc_trace[0] = '\n'.join(tb[-3:])
                raise

        tio._FORCE_THREAD = True

        req_io = tio.ThreadRequestsIO(raise_poster)
        req_io.start()
        try:
            req_io.write('TEST_DATA')
            req_io.write('ANOTHER_PIECE')
            req_io.finish()
        except AttributeError:
            tb = traceback.format_exc().splitlines()
            self.assertEqual('\n'.join(tb[-3:]), exc_trace[0])

        if tio.GreenletRequestsIO is None:
            return

        req_io = tio.GreenletRequestsIO(raise_poster)
        req_io.start()
        try:
            req_io.write('TEST_DATA')
            req_io.write('ANOTHER_PIECE')
            req_io.finish()
        except AttributeError:
            tb = traceback.format_exc().splitlines()
            self.assertEqual('\n'.join(tb[-3:]), exc_trace[0])


if __name__ == '__main__':
    unittest.main()
