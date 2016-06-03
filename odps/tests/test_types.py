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

import decimal as _decimal

from odps.types import *
from odps.models import Schema, Record
from odps.tests.core import TestBase
from odps.compat import unittest, OrderedDict
from datetime import datetime


class Test(TestBase):

    def testNullableRecord(self):
        s = Schema.from_lists(
            ['col%s'%i for i in range(8)],
            ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal',
             'array<string>', 'map<string,bigint>'])
        r = Record(schema=s, values=[None]*8)
        self.assertSequenceEqual(r.values, [None]*8)

    def testRecordSetAndGetByIndex(self):
        s = Schema.from_lists(
            ['col%s'%i for i in range(8)],
            ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal',
             'array<string>', 'map<string,bigint>'])
        r = Record(schema=s)
        r[0] = 1
        r[1] = 1.2
        r[2] = 'abc'
        r[3] = datetime(2016, 1, 1)
        r[4] = True
        r[5] = _decimal.Decimal('1.111')
        r[6] = ['a', 'b']
        r[7] = OrderedDict({'a': 1})
        self.assertSequenceEqual(r.values, [1, 1.2, 'abc', datetime(2016, 1, 1), True,
                                            _decimal.Decimal('1.111'), ['a', 'b'], OrderedDict({'a': 1})])
        self.assertEquals(1, r[0])
        self.assertEquals(1.2, r[1])
        self.assertEquals('abc', r[2])
        self.assertEquals(datetime(2016, 1, 1), r[3])
        self.assertEquals(True, r[4])
        self.assertEquals(_decimal.Decimal('1.111'), r[5])
        self.assertEquals(['a', 'b'], r[6])
        self.assertEquals( OrderedDict({'a': 1}), r[7])

    def testRecordSetAndGetByName(self):
        s = Schema.from_lists(
            ['col%s'%i for i in range(8)],
            ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal',
             'array<string>', 'map<string,bigint>'])
        r = Record(schema=s)
        r['col0'] = 1
        r['col1'] = 1.2
        r['col2'] = 'abc'
        r['col3'] = datetime(2016, 1, 1)
        r['col4'] = True
        r['col5'] = _decimal.Decimal('1.111')
        r['col6'] = ['a', 'b']
        r['col7'] = OrderedDict({'a': 1})
        self.assertSequenceEqual(r.values, [1, 1.2, 'abc', datetime(2016, 1, 1), True,
                                            _decimal.Decimal('1.111'), ['a', 'b'], OrderedDict({'a': 1})])
        self.assertEquals(1, r['col0'])
        self.assertEquals(1.2, r['col1'])
        self.assertEquals('abc', r['col2'])
        self.assertEquals(datetime(2016, 1, 1), r['col3'])
        self.assertEquals(True, r['col4'])
        self.assertEquals(_decimal.Decimal('1.111'), r['col5'])
        self.assertEquals(['a', 'b'], r['col6'])
        self.assertEquals( OrderedDict({'a': 1}), r['col7'])

    def testInplicitCast(self):
        bigint = Bigint()
        double = Double()
        datetime = Datetime()
        bool = Boolean()
        decimal = Decimal()
        string = String()

        self.assertTrue(double.can_implicit_cast(bigint))
        self.assertTrue(string.can_implicit_cast(bigint))
        self.assertTrue(decimal.can_implicit_cast(bigint))
        self.assertFalse(bool.can_implicit_cast(bigint))
        self.assertFalse(datetime.can_implicit_cast(bigint))

        self.assertTrue(bigint.can_implicit_cast(double))
        self.assertTrue(string.can_implicit_cast(double))
        self.assertTrue(decimal.can_implicit_cast(double))
        self.assertFalse(bool.can_implicit_cast(double))
        self.assertFalse(datetime.can_implicit_cast(double))

    def testSetWithCast(self):
        s = Schema.from_lists(
            ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal'],
            ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal'])
        r = Record(schema=s)
        r['double'] = 1
        self.assertEquals(1.0, r['double'])
        r['double'] = '1.33'
        self.assertEquals(1.33, r['double'])
        r['bigint'] = 1.1
        self.assertEquals(1, r['bigint'])
        r['datetime'] = '2016-01-01 0:0:0'
        self.assertEquals(datetime(2016, 1, 1), r['datetime'])

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

    def testDuplicateNames(self):
        self.assertRaises(ValueError, lambda: Schema.from_lists(['col1', 'col1'], ['string', 'string']))
        try:
            Schema.from_lists(['col1', 'col1'], ['string', 'string'])
        except ValueError as e:
            self.assertTrue('col1' in str(e))

    def testChineseSchema(self):
        s = Schema.from_lists([u'用户'], ['string'], ['分区'], ['bigint'])
        self.assertIn('用户', s)
        self.assertEqual(s.get_column('用户').type.name, 'string')
        self.assertEqual(s.get_partition(u'分区').type.name, 'bigint')
        self.assertEqual(s['用户'].type.name, 'string')
        self.assertEqual(s[u'分区'].type.name, 'bigint')

        s2 = Schema.from_lists(['用户'], ['string'], [u'分区'], ['bigint'])
        self.assertEqual(s, s2)

    def testRecordMultiFields(self):
        s = Schema.from_lists(['col1', 'col2'], ['string', 'bigint'])
        r = Record(values=[1, 2], schema=s)

        self.assertEqual(r['col1', 'col2'], ['1', 2])

        self.assertRaises(AttributeError, lambda: r['col3'])
        self.assertRaises(AttributeError, lambda: r['col3', ])

if __name__ == '__main__':
    unittest.main()