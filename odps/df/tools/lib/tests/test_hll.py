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

import pytest

from .....compat import irange as xrange
from .. import HyperLogLog


def test_hll():
    hll = HyperLogLog(0.05)
    buf = hll.buffer()

    for i in xrange(10000):
        hll(buf, str(i))

    assert pytest.approx(hll.getvalue(buf) / float(10000), 0.1) == 1

    for i in xrange(100000, 200000):
        hll(buf, str(i))

    assert pytest.approx(hll.getvalue(buf) / 110000, 0.2) == 1

    buf2 = hll.buffer()

    for i in xrange(10000):
        hll(buf2, str(i))

    hll.merge(buf, buf2)

    assert pytest.approx(hll.getvalue(buf) / 110000, 0.2) == 1
