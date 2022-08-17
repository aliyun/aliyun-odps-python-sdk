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
import time
cimport cython
from cpython.datetime cimport (
    PyDateTime_DateTime,
    datetime,
    datetime_year,
    datetime_month,
    datetime_day,
    datetime_hour,
    datetime_minute,
    datetime_second,
    datetime_microsecond,
    datetime_new,
    import_datetime,
)
from libc.stdint cimport int64_t
from libc.time cimport time_t, tm, mktime, localtime, gmtime
from datetime import datetime

try:
    import pytz
except ImportError:
    pytz = None

from ..compat import utc
from ..config import options

cdef extern from "timegm.c":
    time_t timegm(tm* t) nogil

import_datetime()

cdef long _antique_mills

try:
    _antique_mills = time.mktime(
        datetime(1928, 1, 1).timetuple()
    ) * 1000
except OverflowError:
    _antique_mills = 0
_antique_errmsg = 'Date older than 1928/01/01 and may contain errors. ' \
                  'Ignore this error by configuring `options.allow_antique_date` to True.'


cdef inline bint datetime_hastzinfo(object o):
    return (<PyDateTime_DateTime*>o).hastzinfo


cdef class CMillisecondsConverter:
    @staticmethod
    def _get_tz(tz):
        if type(tz) is unicode or type(tz) is bytes:
            if pytz is None:
                raise ImportError('Package `pytz` is needed when specifying string-format time zone.')
            else:
                return pytz.timezone(tz)
        else:
            return tz

    def __init__(self, local_tz=None, is_dst=False):
        self._local_tz = local_tz if local_tz is not None else options.local_timezone
        if self._local_tz is None:
            self._local_tz = True
        self._use_default_tz = type(self._local_tz) is bool
        self._default_tz_local = self._use_default_tz and self._local_tz

        self._allow_antique = options.allow_antique_date or _antique_mills is None
        self._is_dst = is_dst

        if self._local_tz:
            self._mktime = time.mktime
            self._fromtimestamp = datetime.fromtimestamp
        else:
            self._mktime = calendar.timegm
            self._fromtimestamp = datetime.utcfromtimestamp

        self._tz = self._get_tz(self._local_tz) if not self._use_default_tz else None
        self._tz_has_localize = hasattr(self._tz, 'localize')

    cdef int _build_tm_struct(self, datetime dt, tm *p_tm) except? -1:
        p_tm.tm_year = datetime_year(dt) - 1900
        p_tm.tm_mon = datetime_month(dt) - 1
        p_tm.tm_mday = datetime_day(dt)
        p_tm.tm_yday = 0
        p_tm.tm_hour = datetime_hour(dt)
        p_tm.tm_min = datetime_minute(dt)
        p_tm.tm_sec = datetime_second(dt)
        p_tm.tm_isdst = -1

    @cython.cdivision(True)
    cpdef int64_t to_milliseconds(self, datetime dt) except? -1:
        cdef int64_t mills
        cdef tm tm_result
        cdef time_t unix_ts
        cdef bint with_tzinfo = datetime_hastzinfo(dt)
        cdef bint assume_local_tz = self._default_tz_local

        if not self._use_default_tz and not with_tzinfo:
            if self._tz_has_localize:
                dt = self._tz.localize(dt, is_dst=self._is_dst)
            else:
                dt = dt.replace(tzinfo=self._tz)
            with_tzinfo = True

        if with_tzinfo:
            dt = dt.astimezone(utc)
            assume_local_tz = False

        self._build_tm_struct(dt, &tm_result)
        if assume_local_tz:
            unix_ts = mktime(&tm_result)
        else:
            unix_ts = timegm(&tm_result)

        mills = unix_ts * 1000 + datetime_microsecond(dt) / 1000

        if not self._allow_antique and mills < _antique_mills:
            raise OverflowError(_antique_errmsg)
        return mills

    @cython.cdivision(True)
    cpdef datetime from_milliseconds(self, int64_t milliseconds):
        cdef time_t seconds
        cdef int64_t microseconds
        cdef tm* p_tm

        if not self._allow_antique and milliseconds < _antique_mills:
            raise OverflowError(_antique_errmsg)

        seconds = milliseconds / 1000
        microseconds = milliseconds % 1000 * 1000
        if self._use_default_tz:
            if self._default_tz_local:
                p_tm = localtime(&seconds)
            else:
                p_tm = gmtime(&seconds)

            return datetime_new(
                p_tm.tm_year + 1900,
                p_tm.tm_mon + 1,
                p_tm.tm_mday,
                p_tm.tm_hour,
                p_tm.tm_min,
                p_tm.tm_sec,
                microseconds,
                None
            )
        else:
            return datetime.utcfromtimestamp(seconds) \
                .replace(microsecond=microseconds, tzinfo=utc) \
                .astimezone(self._tz)
