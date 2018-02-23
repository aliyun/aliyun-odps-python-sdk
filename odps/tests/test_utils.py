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

import calendar
import datetime
import os
import time
from collections import namedtuple

from odps import options, utils
from odps.tests.core import TestBase, module_depend_case
from odps.compat import unittest, long_type, reload_module

mytimetuple = namedtuple(
    'TimeTuple',
    [s for s in dir(datetime.datetime.now().timetuple()) if s.startswith('tm_')]
)


def bothPyAndC(func):
    def inner(self, *args, **kwargs):
        try:
            import cython
            ts = 'py', 'c'
        except ImportError:
            ts = 'py',
            import warnings
            warnings.warn('No c code tests for table tunnel')
        for t in ts:
            old_config = getattr(options, 'force_{0}'.format(t))
            setattr(options, 'force_{0}'.format(t), True)
            try:
                reload_module(utils)
                func(self, *args, **kwargs)
            finally:
                setattr(options, 'force_{0}'.format(t), old_config)

    return inner


class Test(TestBase):
    def testReplaceSqlParameters(self):
        cases = [
        {
            'ns': {
                'dss': ['20180101', '20180102', '20180101'],
            },
            'sql': 'select * from dual where id in :dss',
            'expected': ["select * from dual where id in ('20180101', '20180102', '20180101')"]
        },
        {
            'ns': {
                'dss': set(['20180101', '20180102', '20180101']),
            },
            'sql': 'select * from dual where id in :dss',
            'expected': [
                "select * from dual where id in ('20180101', '20180102')",
                "select * from dual where id in ('20180102', '20180101')"
            ]
        },
        {
            'ns': {'test1': 'new_test1', 'test3': 'new_\'test3\''},
            'sql': 'select :test1 from dual where :test2 > 0 and f=:test3.abc',
            'expected': ["select 'new_test1' from dual where :test2 > 0 and f='new_\\'test3\\''.abc"]
        },
        {
            'ns': {
                'dss': ('20180101', '20180102', 20180101),
            },
            'sql': 'select * from dual where id in :dss',
            'expected': ["select * from dual where id in ('20180101', '20180102', 20180101)"]
        },
        {
            'ns': {
                'ds': '20180101',
                'dss': ('20180101', 20180101),
                'id': 21312,
                'price': 6.4,
                'prices': (123, '123', 6.4)
            },
            'sql': 'select * from dual where ds = :ds or ds in :dss and id = :id and price > :price or price in :prices',
            'expected': ["select * from dual where ds = '20180101' or ds in ('20180101', 20180101) and id = 21312 and price > 6.4 or price in (123, '123', 6.4)"]
        }
        ]
        for case in cases:
            self.assertIn(utils.replace_sql_parameters(case['sql'], case['ns']), case['expected'])

    def testExperimental(self):
        @utils.experimental('Experimental method')
        def fun():
            pass

        try:
            os.environ['PYODPS_EXPERIMENTAL'] = 'false'
            self.assertRaises(utils.ExperimentalNotAllowed, fun)
        finally:
            del os.environ['PYODPS_EXPERIMENTAL']

    @bothPyAndC
    def testTimeConvertNative(self):
        class GMT8(datetime.tzinfo):
            def utcoffset(self, dt):
                return datetime.timedelta(hours=8)

            def dst(self, date_time):
                return datetime.timedelta(0)

            def tzname(self, dt):
                return "GMT +8"

        class UTC(datetime.tzinfo):
            def utcoffset(self, dt):
                return datetime.timedelta(hours=0)

            def dst(self, date_time):
                return datetime.timedelta(0)

            def tzname(self, dt):
                return "UTC"

        to_milliseconds = utils.to_milliseconds
        to_datetime = utils.to_datetime

        base_time = datetime.datetime.now().replace(microsecond=0)
        base_time_utc = datetime.datetime.utcfromtimestamp(time.mktime(base_time.timetuple()))
        milliseconds = long_type(time.mktime(base_time.timetuple())) * 1000

        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz=True))
        self.assertEqual(milliseconds, to_milliseconds(base_time_utc, local_tz=False))

        self.assertEqual(to_datetime(milliseconds, local_tz=True), base_time)
        self.assertEqual(to_datetime(milliseconds, local_tz=False), base_time_utc)

        base_time = datetime.datetime.now(tz=GMT8()).replace(microsecond=0)
        milliseconds = long_type(calendar.timegm(base_time.astimezone(UTC()).timetuple())) * 1000

        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz=True))
        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz=False))
        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz=UTC()))

        self.assertEqual(to_datetime(milliseconds, local_tz=GMT8()), base_time)

        base_time = base_time.replace(tzinfo=None)

        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz=GMT8()))

    @module_depend_case('pytz')
    @bothPyAndC
    def testTimeConvertPytz(self):
        import pytz

        to_milliseconds = utils.to_milliseconds
        to_datetime = utils.to_datetime

        base_time = datetime.datetime.now(tz=pytz.timezone('Etc/GMT-8')).replace(microsecond=0)
        milliseconds = long_type(calendar.timegm(base_time.astimezone(pytz.utc).timetuple())) * 1000

        self.assertEqual(to_datetime(milliseconds, local_tz='Etc/GMT-8'), base_time)

        base_time = base_time.replace(tzinfo=None)

        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz='Etc/GMT-8'))
        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz=pytz.timezone('Etc/GMT-8')))

        base_time = datetime.datetime.now(tz=pytz.timezone('Etc/GMT-8')).replace(microsecond=0)
        milliseconds = time.mktime(base_time.timetuple()) * 1000

        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz=True))
        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz=False))
        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz='Etc/GMT-1'))
        self.assertEqual(milliseconds, to_milliseconds(base_time, local_tz=pytz.timezone('Etc/GMT-1')))

    def testThreadLocalAttribute(self):
        class TestClass(object):
            _no_defaults = utils.thread_local_attribute('test_thread_local')
            _defaults = utils.thread_local_attribute('test_thread_local', lambda: 'TestValue')

        inst = TestClass()
        self.assertRaises(AttributeError, lambda: inst._no_defaults)
        self.assertEqual(inst._defaults, 'TestValue')

        inst._no_defaults = 'TestManualValue1'
        self.assertEqual(inst._no_defaults, 'TestManualValue1')
        inst._defaults = 'TestManualValue2'
        self.assertEqual(inst._defaults, 'TestManualValue2')

        from odps.compat import futures
        executor = futures.ThreadPoolExecutor(1)

        def test_fn():
            self.assertRaises(AttributeError, lambda: inst._no_defaults)
            self.assertEqual(inst._defaults, 'TestValue')

            inst._no_defaults = 'TestManualValue1'
            self.assertEqual(inst._no_defaults, 'TestManualValue1')
            inst._defaults = 'TestManualValue2'
            self.assertEqual(inst._defaults, 'TestManualValue2')

        executor.submit(test_fn).result()


if __name__ == '__main__':
    unittest.main()
