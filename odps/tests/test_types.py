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


from odps.types import *
from odps.models import Schema
from odps.tests.core import TestBase
from odps.compat import unittest


class Test(TestBase):

    def testNullableRecord(self):
        s = Schema.from_lists(
            ['col%s'%i for i in range(8)],
            ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal',
             'array<string>', 'map<string,bigint>'])
        r = Record(schema=s, values=[None]*8)
        self.assertSequenceEqual(r.values, [None]*8)

    def testRecordSetField(self):
        s = Schema.from_lists(['col1'], ['string',])
        r = Record(schema=s)
        r.col1 = 'a'
        self.assertEqual(r.col1, 'a')

        r['col1'] = 'b'
        self.assertEqual(r['col1'], 'b')

        r[0] = 'c'
        self.assertEqual(r[0], 'c')
        self.assertEqual(r['col1'], 'c')

if __name__ == '__main__':
    unittest.main()