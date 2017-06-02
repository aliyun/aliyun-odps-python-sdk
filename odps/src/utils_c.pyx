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

from libc.stdint cimport *
from libc.time cimport time_t, tm, mktime
from cpython.datetime cimport *

cdef extern from "utils.h":
    time_t timegm(tm *)

from ..compat import utc


cdef void _datetime_to_tm(object dt, tm *ptm):
    ptm.tm_year = datetime_year(dt) - <int>1900
    ptm.tm_mon = datetime_month(dt) - <int>1
    ptm.tm_mday = datetime_day(dt)
    ptm.tm_hour = datetime_hour(dt)
    ptm.tm_min = datetime_minute(dt)
    ptm.tm_sec = datetime_second(dt)
    ptm.tm_isdst = -1


cdef uint64_t c_datetime_to_local_milliseconds(object dt):
    cdef tm tm_date
    cdef uint64_t local_time
    if (<PyDateTime_DateTime*>dt).hastzinfo:
        dt = dt.astimezone(utc)
        _datetime_to_tm(dt, &tm_date)
        local_time = timegm(&tm_date)
    else:
        _datetime_to_tm(dt, &tm_date)
        local_time = mktime(&tm_date)
    return local_time * <uint64_t>1000 + datetime_microsecond(dt) / <uint64_t>1000


cpdef uint64_t datetime_to_local_milliseconds(object dt):
    return c_datetime_to_local_milliseconds(dt)


cdef uint64_t c_datetime_to_gmt_milliseconds(object dt):
    cdef tm tm_date
    cdef uint64_t local_time
    if (<PyDateTime_DateTime*>dt).hastzinfo:
        dt = dt.astimezone(utc)
    _datetime_to_tm(dt, &tm_date)
    local_time = timegm(&tm_date)
    return local_time * <uint64_t>1000 + datetime_microsecond(dt) / <uint64_t>1000

cpdef uint64_t datetime_to_gmt_milliseconds(object dt):
    return c_datetime_to_gmt_milliseconds(dt)


cdef TO_MILLISECONDS_FUN get_to_milliseconds_fun_ptr(object fun):
    if fun is datetime_to_local_milliseconds:
        return c_datetime_to_local_milliseconds
    elif fun is datetime_to_gmt_milliseconds:
        return c_datetime_to_gmt_milliseconds
    else:
        return NULL
