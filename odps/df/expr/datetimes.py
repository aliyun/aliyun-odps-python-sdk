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

from .arithmetic import UnaryOp
from .expressions import *
from . import utils
from .. import types


class DatetimeOp(UnaryOp):
    __slots__ = ()

    def accept(self, visitor):
        return visitor.visit_datetime_op(self)


class Date(DatetimeOp):
    __slots__ = ()


class Time(DatetimeOp):
    __slots__ = ()


class Year(DatetimeOp):
    __slots__ = ()


class Month(DatetimeOp):
    __slots__ = ()


class Day(DatetimeOp):
    __slots__ = ()


class Hour(DatetimeOp):
    __slots__ = ()


class Minute(DatetimeOp):
    __slots__ = ()


class Second(DatetimeOp):
    __slots__ = ()


class MilliSecond(DatetimeOp):
    __slots__ = ()


class MicroSecond(DatetimeOp):
    __slots__ = ()


class Week(DatetimeOp):
    __slots__ = ()


class WeekOfYear(DatetimeOp):
    __slots__ = ()


class WeekDay(DatetimeOp):
    __slots__ = ()


class UnixTimestamp(DatetimeOp):
    __slots__ = ()


class DayOfYear(DatetimeOp):
    __slots__ = ()


class IsMonthStart(DatetimeOp):
    __slots__ = ()


class IsMonthEnd(DatetimeOp):
    __slots__ = ()


class IsYearStart(DatetimeOp):
    __slots__ = ()


class IsYearEnd(DatetimeOp):
    __slots__ = ()


class DTScalar(Int64Scalar):
    __slots__ = ()


class YearScalar(DTScalar):
    __slots__ = ()


class MonthScalar(DTScalar):
    __slots__ = ()


class DayScalar(DTScalar):
    __slots__ = ()


class HourScalar(DTScalar):
    __slots__ = ()


class MinuteScalar(DTScalar):
    __slots__ = ()


class SecondScalar(DTScalar):
    __slots__ = ()


class MilliSecondScalar(DTScalar):
    __slots__ = ()


class MicroSecondScalar(DTScalar):
    __slots__ = ()


class Strftime(DatetimeOp):
    _args = '_input', '_date_format'
    _add_args_slots = False

    def _init(self, *args, **kwargs):
        self._init_attr('_date_format', None)

        super(Strftime, self)._init(*args, **kwargs)

        if self._date_format is not None and not isinstance(self._date_format, Scalar):
            self._date_format = Scalar(_value=self._date_format)

    @property
    def date_format(self):
        return self._date_format.value


def datetime_op(expr, output_expr_cls, output_type=None, **kwargs):
    output_type = output_type or types.datetime

    if isinstance(expr, (DatetimeSequenceExpr, DatetimeScalar)):
        is_sequence = isinstance(expr, DatetimeSequenceExpr)

        if is_sequence:
            return output_expr_cls(_data_type=output_type, _input=expr, **kwargs)
        else:
            return output_expr_cls(_value_type=output_type, _input=expr, **kwargs)


@property
def _date(expr):
    return datetime_op(expr, Date)


@property
def _time(expr):
    return datetime_op(expr, Time)


@property
def _year(expr):
    return datetime_op(expr, Year, output_type=types.int64)


@property
def _month(expr):
    return datetime_op(expr, Month, output_type=types.int64)


@property
def _day(expr):
    return datetime_op(expr, Day, output_type=types.int64)


@property
def _hour(expr):
    return datetime_op(expr, Hour, output_type=types.int64)


@property
def _minute(expr):
    return datetime_op(expr, Minute, output_type=types.int64)


@property
def _second(expr):
    return datetime_op(expr, Second, output_type=types.int64)


@property
def _millisecond(expr):
    return datetime_op(expr, MilliSecond, output_type=types.int64)


@property
def _microsecond(expr):
    return datetime_op(expr, MicroSecond, output_type=types.int64)


@property
def _week(expr):
    return datetime_op(expr, Week, output_type=types.int64)


@property
def _weekofyear(expr):
    return datetime_op(expr, WeekOfYear, output_type=types.int64)


@property
def _weekday(expr):
    return datetime_op(expr, WeekDay, output_type=types.int64)


@property
def _dayofyear(expr):
    return datetime_op(expr, DayOfYear, output_type=types.int64)


@property
def _unix_timestamp(expr):
    return datetime_op(expr, UnixTimestamp, output_type=types.int64)


@property
def _is_month_start(expr):
    return datetime_op(expr, IsMonthStart, output_type=types.boolean)


@property
def _is_month_end(expr):
    return datetime_op(expr, IsMonthEnd, output_type=types.boolean)


@property
def _is_year_start(expr):
    return datetime_op(expr, IsYearStart, output_type=types.boolean)


@property
def _is_year_end(expr):
    return datetime_op(expr, IsYearEnd, output_type=types.boolean)


def _strftime(expr, date_format):
    """
    Return formatted strings specified by date_format,
    which supports the same string format as the python standard library.
    Details of the string format can be found in python string format doc

    :param expr:
    :param date_format: date format string (e.g. “%Y-%m-%d”)
    :type date_format: str
    :return:
    """

    return datetime_op(expr, Strftime, output_type=types.string,
                       _date_format=date_format)


def year(val):
    return YearScalar(_value=val, _value_type=types.int64)


def month(val):
    return MonthScalar(_value=val, _value_type=types.int64)


def day(val):
    return DayScalar(_value=val, _value_type=types.int64)


def hour(val):
    return HourScalar(_value=val, _value_type=types.int64)


def minute(val):
    return MinuteScalar(_value=val, _value_type=types.int64)


def second(val):
    return SecondScalar(_value=val, _value_type=types.int64)


def millisecond(val):
    return MilliSecondScalar(_value=val, _value_type=types.int64)


def microsecond(val):
    return MicroSecondScalar(_value=val, _value_type=types.int64)


_date_methods = dict(
    year=_year,
    month=_month,
    day=_day,
    week=_week,
    weekofyear=_weekofyear,
    dayofweek=_weekday,
    weekday=_weekday,
    dayofyear=_dayofyear,
    is_month_start=_is_month_start,
    is_month_end=_is_month_end,
    is_year_start=_is_year_start,
    is_year_end=_is_year_end,
)
_datetime_methods = dict(
    date=_date,
    time=_time,
    hour=_hour,
    minute=_minute,
    second=_second,
    microsecond=_microsecond,
    unix_timestamp=_unix_timestamp,
    strftime=_strftime
)
_datetime_methods.update(_date_methods)

utils.add_method(DateSequenceExpr, _date_methods)
utils.add_method(DatetimeSequenceExpr, _datetime_methods)
utils.add_method(DateScalar, _date_methods)
utils.add_method(DatetimeScalar, _datetime_methods)
