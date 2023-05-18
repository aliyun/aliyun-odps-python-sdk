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

import calendar
import datetime
import functools
import os
import time
from collections import namedtuple

import pytest

from .. import utils
from ..compat import long_type
from .core import module_depend_case
try:
    from ..src.utils_c import CMillisecondsConverter
except ImportError:
    CMillisecondsConverter = None

mytimetuple = namedtuple(
    'TimeTuple',
    [s for s in dir(datetime.datetime.now().timetuple()) if s.startswith('tm_')]
)


def test_replace_sql_parameters():
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
            'expected': [r"select 'new_test1' from dual where :test2 > 0 and f='new_\'test3\''.abc"]
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
            'sql': 'select * from dual where ds = :ds or ds in :dss and id = :id '
                   'and price > :price or price in :prices',
            'expected': ["select * from dual where ds = '20180101' or ds in ('20180101', 20180101) "
                         "and id = 21312 and price > {0!r} or price in (123, '123', {0!r})".format(6.4)]
        }
    ]
    for case in cases:
        assert utils.replace_sql_parameters(case['sql'], case['ns']) in case['expected']


def test_experimental():
    @utils.experimental('Experimental method')
    def fun():
        pass

    try:
        os.environ['PYODPS_EXPERIMENTAL'] = 'false'
        pytest.raises(utils.ExperimentalNotAllowed, fun)
    finally:
        del os.environ['PYODPS_EXPERIMENTAL']


@pytest.mark.parametrize("force_py", (False, True) if CMillisecondsConverter else (True,))
def test_time_convert_native(force_py):
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

    to_milliseconds = functools.partial(utils.to_milliseconds, force_py=force_py)
    to_datetime = functools.partial(utils.to_datetime, force_py=force_py)

    base_time = datetime.datetime.now().replace(microsecond=0)
    base_time_utc = datetime.datetime.utcfromtimestamp(time.mktime(base_time.timetuple()))
    milliseconds = long_type(time.mktime(base_time.timetuple())) * 1000

    assert milliseconds == to_milliseconds(base_time, local_tz=True)
    assert milliseconds == to_milliseconds(base_time_utc, local_tz=False)

    assert to_datetime(milliseconds, local_tz=True) == base_time
    assert to_datetime(milliseconds, local_tz=False) == base_time_utc

    base_time = datetime.datetime.now(tz=GMT8()).replace(microsecond=0)
    milliseconds = long_type(calendar.timegm(base_time.astimezone(UTC()).timetuple())) * 1000

    assert milliseconds == to_milliseconds(base_time, local_tz=True)
    assert milliseconds == to_milliseconds(base_time, local_tz=False)
    assert milliseconds == to_milliseconds(base_time, local_tz=UTC())

    assert to_datetime(milliseconds, local_tz=GMT8()) == base_time

    base_time = base_time.replace(tzinfo=None)

    assert milliseconds == to_milliseconds(base_time, local_tz=GMT8())

    base_time = datetime.datetime(1969, 12, 30, 21, 42, 23, 211000)
    assert base_time == to_datetime(to_milliseconds(base_time))


@module_depend_case('pytz')
@pytest.mark.parametrize('force_py', [False, True] if CMillisecondsConverter else [True])
def test_time_convert_pytz(force_py):
    import pytz

    to_milliseconds = functools.partial(utils.to_milliseconds, force_py=force_py)
    to_datetime = functools.partial(utils.to_datetime, force_py=force_py)

    base_time = datetime.datetime.now(tz=pytz.timezone('Etc/GMT-8')).replace(microsecond=0)
    milliseconds = long_type(calendar.timegm(base_time.astimezone(pytz.utc).timetuple())) * 1000

    assert to_datetime(milliseconds, local_tz='Etc/GMT-8') == base_time

    base_time = base_time.replace(tzinfo=None)

    assert milliseconds == to_milliseconds(base_time, local_tz='Etc/GMT-8')
    assert milliseconds == to_milliseconds(base_time, local_tz=pytz.timezone('Etc/GMT-8'))

    base_time = datetime.datetime.now(tz=pytz.timezone('Etc/GMT-8')).replace(microsecond=0)
    milliseconds = time.mktime(base_time.timetuple()) * 1000

    assert milliseconds == to_milliseconds(base_time, local_tz=True)
    assert milliseconds == to_milliseconds(base_time, local_tz=False)
    assert milliseconds == to_milliseconds(base_time, local_tz='Etc/GMT-1')
    assert milliseconds == to_milliseconds(base_time, local_tz=pytz.timezone('Etc/GMT-1'))


def test_thread_local_attribute():
    class TestClass(object):
        _no_defaults = utils.thread_local_attribute('test_thread_local')
        _defaults = utils.thread_local_attribute('test_thread_local', lambda: 'TestValue')

    inst = TestClass()
    pytest.raises(AttributeError, lambda: inst._no_defaults)
    assert inst._defaults == 'TestValue'

    inst._no_defaults = 'TestManualValue1'
    assert inst._no_defaults == 'TestManualValue1'
    inst._defaults = 'TestManualValue2'
    assert inst._defaults == 'TestManualValue2'

    from ..compat import futures
    executor = futures.ThreadPoolExecutor(1)

    def test_fn():
        pytest.raises(AttributeError, lambda: inst._no_defaults)
        assert inst._defaults == 'TestValue'

        inst._no_defaults = 'TestManualValue1'
        assert inst._no_defaults == 'TestManualValue1'
        inst._defaults = 'TestManualValue2'
        assert inst._defaults == 'TestManualValue2'

    executor.submit(test_fn).result()


def test_wait_function_compatibility():
    @utils.with_wait_argument
    def arg_holder(arg, async_=False, kwa=False):
        return async_

    @utils.with_wait_argument
    def arg_holder_kw(arg, async_=False, kwa=False, **kw):
        return async_

    with pytest.warns(DeprecationWarning):
        assert arg_holder(0, True)

    with pytest.warns(Warning) as warn_info:
        assert arg_holder(0, async_=True, kwa=True, no_exist_arg=True)
    assert len(warn_info) == 1
    assert "no_exist_arg" in str(warn_info[0].message)
    assert "kwa" not in str(warn_info[0].message)

    with pytest.warns(None) as warn_info:
        assert arg_holder_kw(0, async_=True, kwa=True, no_exist_arg=True)
    assert len(warn_info) == 0

    assert arg_holder(0, **{"async": True})
    assert arg_holder(0, async_=True)
    assert arg_holder(0, wait=False)
