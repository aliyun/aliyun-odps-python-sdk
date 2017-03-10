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

from odps.tests.core import TestBase
from odps.compat import unittest, long_type
from odps.utils import replace_sql_parameters, experimental, ExperimentalNotAllowed, \
    to_milliseconds, to_datetime

try:
    import pytz
except ImportError:
    pytz = None

mytimetuple = namedtuple(
    'TimeTuple',
    [s for s in dir(datetime.datetime.now().timetuple()) if s.startswith('tm_')]
)


class Test(TestBase):
    def testReplaceSqlParameters(self):
        ns = {'test1': 'new_test1', 'test3': 'new_test3'}

        sql = 'select :test1 from dual where :test2 > 0 and f=:test3.abc'
        replaced_sql = replace_sql_parameters(sql, ns)

        expected = 'select new_test1 from dual where :test2 > 0 and f=new_test3.abc'
        self.assertEqual(expected, replaced_sql)

    def testExperimental(self):
        @experimental('Experimental method')
        def fun():
            pass

        try:
            os.environ['PYODPS_EXPERIMENTAL'] = 'false'
            self.assertRaises(ExperimentalNotAllowed, fun)
        finally:
            del os.environ['PYODPS_EXPERIMENTAL']

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

    @unittest.skipIf(pytz is None, 'Skipped because pytz is not installed.')
    def testTimeConvertPytz(self):
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


if __name__ == '__main__':
    unittest.main()
