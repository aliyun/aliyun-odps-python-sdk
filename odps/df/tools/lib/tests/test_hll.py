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

from odps.compat import unittest, irange as xrange
from odps.df.tools.lib import HyperLogLog
from odps.tests.core import TestBase

class Test(TestBase):
    def testHLL(self):
        hll = HyperLogLog(0.05)
        buf = hll.buffer()

        for i in xrange(10000):
            hll(buf, str(i))

        self.assertAlmostEqual(hll.getvalue(buf) / float(10000), 1, delta=0.1)

        for i in xrange(100000, 200000):
            hll(buf, str(i))

        self.assertAlmostEqual(hll.getvalue(buf) / 110000, 1, delta=0.2)

        buf2 = hll.buffer()

        for i in xrange(10000):
            hll(buf2, str(i))

        hll.merge(buf, buf2)

        self.assertAlmostEqual(hll.getvalue(buf) / 110000, 1, delta=0.2)


if __name__ == '__main__':
    unittest.main()
