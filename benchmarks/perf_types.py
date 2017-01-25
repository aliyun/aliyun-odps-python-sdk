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

import cProfile
from datetime import datetime
from decimal import Decimal
from pstats import Stats

from odps.compat import unittest
from odps.tests.core import TestBase
from odps.models import Schema
from odps.types import Record


class Test(TestBase):
    COMPRESS_DATA = True
    BUFFER_SIZE = 1024*1024
    DATA_AMOUNT = 100000
    STRING_LITERAL = "Soft kitty, warm kitty, little ball of fur; happy kitty, sleepy kitty, purr, purr"

    def setUp(self):
        TestBase.setUp(self)
        self.pr = cProfile.Profile()
        self.pr.enable()
        fields = ['bigint', 'double', 'datetime', 'boolean', 'string', 'decimal']
        types = ['bigint', 'double', 'datetime', 'boolean', 'string', 'decimal']
        self.SCHEMA = Schema.from_lists(fields, types)

    def tearDown(self):
        p = Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('cumtime')
        p.print_stats(40)
        TestBase.teardown(self)

    def testSetRecordFieldBigint(self):
        r = Record(schema=self.SCHEMA)
        for i in range(10**6):
            r['bigint'] = 2**63-1

    def testSetRecordFieldDouble(self):
        r = Record(schema=self.SCHEMA)
        for i in range(10**6):
            r['double'] = 0.0001

    def testSetRecordFieldBoolean(self):
        r = Record(schema=self.SCHEMA)
        for i in range(10**6):
            r['boolean'] = False

    def testSetRecordFieldString(self):
        r = Record(schema=self.SCHEMA)
        for i in range(10**6):
            r['string'] = self.STRING_LITERAL

    def testWriteSetRecordFieldDatetime(self):
        r = Record(schema=self.SCHEMA)
        for i in range(10**6):
            r['datetime'] = datetime(2016, 1, 1)

    def testSetRecordFieldDecimal(self):
        r = Record(schema=self.SCHEMA)
        for i in range(10**6):
            r['decimal'] = Decimal('1.111111')

if __name__ == '__main__':
    unittest.main()
