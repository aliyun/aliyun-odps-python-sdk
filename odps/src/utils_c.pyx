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

cimport cython
import sys
import threading
import time
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
    timedelta_new,
    import_datetime,
)
from datetime import datetime
from libc.stdint cimport int64_t
from libc.time cimport time_t, tm, mktime, localtime, gmtime

try:
    import pytz
except ImportError:
    pytz = None

from ..compat import utc
from ..config import options

cdef extern from "timegm.c":
    time_t timegm(tm* t) nogil

cdef bint _is_windows = sys.platform.lower().startswith("win")
cdef bint _is_py3 = sys.version_info[0] == 3
cdef bint _datetime_inited = False
cdef object _time_init_lock = threading.Lock()

import_datetime()

cdef int64_t _antique_mills, _min_datetime_mills
cdef object _py_local_epoch

try:
    _py_local_epoch = datetime.fromtimestamp(0)
except OSError:  # workaround for bpo-29097  # pragma: no cover
    _py_local_epoch = datetime(*time.localtime(0)[:6])

try:
    _antique_mills = time.mktime(
        datetime(1928, 1, 1).timetuple()
    ) * 1000
except OverflowError:
    _antique_mills = int(
        (datetime(1928, 1, 1) - datetime.utcfromtimestamp(0)).total_seconds()
    ) * 1000
_min_datetime_mills = int(
    (datetime.min - datetime.utcfromtimestamp(0)).total_seconds() * 1000
)
_antique_errmsg = 'Date older than 1928/01/01 and may contain errors. ' \
                  'Ignore this error by configuring `options.allow_antique_date` to True.'
_min_datetime_errmsg = 'Date exceed range Python can handle. If you are reading data with tunnel, read '\
                       'the value as None by setting options.tunnel.overflow_date_as_none to True, ' \
                       'or convert the value into strings with SQL before processing them with Python.'

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
        global _datetime_inited

        if not _is_py3 and not _datetime_inited:
            # due to cython implementations, make sure the line below is always invoked
            with _time_init_lock:
                if not _datetime_inited:
                    import_datetime()
                    _datetime_inited = True

        self._local_tz = local_tz if local_tz is not None else options.local_timezone
        if self._local_tz is None:
            self._local_tz = True
        self._use_default_tz = type(self._local_tz) is bool
        self._default_tz_local = self._use_default_tz and self._local_tz

        self._allow_antique = options.allow_antique_date or _antique_mills is None
        self._is_dst = is_dst

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
            if not _is_windows or tm_result.tm_year > 1970:
                unix_ts = mktime(&tm_result)
            else:
                py_datetime = datetime_new(
                    tm_result.tm_year + 1900,
                    tm_result.tm_mon + 1,
                    tm_result.tm_mday,
                    tm_result.tm_hour,
                    tm_result.tm_min,
                    tm_result.tm_sec,
                    0,
                    None,
                )
                unix_ts = int((py_datetime - _py_local_epoch).total_seconds())
        else:
            unix_ts = timegm(&tm_result)

        mills = unix_ts * 1000 + datetime_microsecond(dt) / 1000

        if not self._allow_antique and mills < _antique_mills:
            from ..errors import DatetimeOverflowError

            raise DatetimeOverflowError(_antique_errmsg)
        return mills

    @cython.cdivision(True)
    cpdef datetime from_milliseconds(self, int64_t milliseconds):
        cdef time_t seconds, zero
        cdef int64_t microseconds
        cdef tm* p_tm

        if not self._allow_antique and milliseconds < _antique_mills:
            from ..errors import DatetimeOverflowError

            raise DatetimeOverflowError(_antique_errmsg)
        if milliseconds < _min_datetime_mills:
            from ..errors import DatetimeOverflowError

            raise DatetimeOverflowError(_min_datetime_errmsg)

        if milliseconds >= 0:
            seconds = milliseconds / 1000
            microseconds = milliseconds % 1000 * 1000
        else:
            seconds = milliseconds / 1000
            microseconds = milliseconds % 1000 * 1000
            if microseconds < 0:
                microseconds += 1000000
                seconds -= 1

        if self._use_default_tz:
            if self._default_tz_local and _is_windows and milliseconds < 0:
                # special logic for negative timestamp under Windows
                zero = 0
                p_tm = localtime(&zero)
                return datetime_new(
                    p_tm.tm_year + 1900,
                    p_tm.tm_mon + 1,
                    p_tm.tm_mday,
                    p_tm.tm_hour,
                    p_tm.tm_min,
                    p_tm.tm_sec,
                    microseconds,
                    None,
                ) + timedelta_new(0, seconds, 0)

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
                None,
            )
        else:
            return datetime.utcfromtimestamp(seconds) \
                .replace(microsecond=microseconds, tzinfo=utc) \
                .astimezone(self._tz)
