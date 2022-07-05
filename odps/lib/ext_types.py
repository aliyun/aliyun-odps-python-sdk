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

import sys
import re

if sys.version_info[0] <= 2:
    string_types = basestring
else:
    string_types = str

_MD_PATTERNS = [
    re.compile(r'^(?P<month>(\d+|-\d+))$'),
    re.compile(r'^(?P<year>(\d+|-\d+))\s*-\s*(?P<month>(\d+|-\d+))$'),
    re.compile(r'^(?P<year>(\d+|-\d+))\s*(years|year)\s*(?P<month>(\d+|-\d+))\s*(months|month)$'),
    re.compile(r'^(?P<year>(\d+|-\d+))\s*(years|year)$'),
    re.compile(r'^(?P<month>(\d+|-\d+))\s*(months|month)$'),
]


class Monthdelta(object):
    __slots__ = '_total_months',

    def __init__(self, years=None, months=None):
        if isinstance(years, string_types):
            years, months = self._parse(years)
        elif months is None:
            years, months = months, years
        years = years or 0
        months = months or 0
        self._total_months = int(years * 12 + months)

    @classmethod
    def _parse(cls, s):
        s = s.strip()
        for p in _MD_PATTERNS:
            match = p.match(s)
            if not match:
                continue
            match = match.groupdict()
            return int(match.get('year', 0)), int(match.get('month', 0))
        raise ValueError("Cannot parse '%s' for %s." % (s, cls.__name__))

    def total_months(self):
        return self._total_months

    @property
    def months(self):
        return self._total_months % 12

    @property
    def years(self):
        return self._total_months // 12

    def __getstate__(self):
        return dict(_total_months=self._total_months)

    def __setstate__(self, state):
        self._total_months = state['_total_months']

    def __int__(self):
        return self.total_months()

    if sys.version_info[0] <= 2:
        def __long__(self):
            return self.total_months()

    def __str__(self):
        years = self.years
        months = self.months
        m = lambda v: 's' if abs(v) > 1 else ''
        if years == 0 and months == 0:
            return '0'
        elif years == 0:
            return '%d month%s' % (months, m(months))
        elif months == 0:
            return '%d year%s' % (years, m(years))
        else:
            return '%d year%s %d month%s' % (years, m(years), months, m(months))

    def __repr__(self):
        return "%s('%s')" % (type(self).__name__, str(self))

    def __eq__(self, other):
        if not isinstance(other, Monthdelta):
            return False
        return self._total_months == other._total_months

    def __ne__(self, other):
        return not (self == other)

    def __gt__(self, other):
        try:
            return self._total_months > other._total_months
        except AttributeError:
            raise TypeError("unsupported operand type(s) for >: '%s' and '%s'" %
                            (type(self).__name__, type(other).__name__))

    def __ge__(self, other):
        try:
            return self._total_months >= other._total_months
        except AttributeError:
            raise TypeError("unsupported operand type(s) for >=: '%s' and '%s'" %
                            (type(self).__name__, type(other).__name__))

    def __lt__(self, other):
        try:
            return self._total_months < other._total_months
        except AttributeError:
            raise TypeError("unsupported operand type(s) for <: '%s' and '%s'" %
                            (type(self).__name__, type(other).__name__))

    def __le__(self, other):
        try:
            return self._total_months <= other._total_months
        except AttributeError:
            raise TypeError("unsupported operand type(s) for <=: '%s' and '%s'" %
                            (type(self).__name__, type(other).__name__))

    def __abs__(self):
        return Monthdelta(months=abs(self._total_months))

    def __neg__(self):
        return Monthdelta(months=-self._total_months)

    def _add(self, other):
        if isinstance(other, Monthdelta):
            return Monthdelta(months=self._total_months + other._total_months)

        new_year = other.year + self.years
        new_month = other.month + self.months
        if new_month < 0:
            new_month += 12
            new_year -= 1
        new_year += new_month // 12
        new_month = new_month % 12
        return other.replace(year=new_year, month=new_month)

    def __add__(self, other):
        try:
            return self._add(other)
        except AttributeError:
            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" %
                            (type(self).__name__, type(other).__name__))

    def __radd__(self, other):
        try:
            return self._add(other)
        except AttributeError:
            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" %
                            (type(other).__name__, type(self).__name__))

    def __sub__(self, other):
        if isinstance(other, Monthdelta):
            return Monthdelta(months=self._total_months - other._total_months)
        else:
            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" %
                            (type(other).__name__, type(self).__name__))

    def __rsub__(self, other):
        try:
            return (-self)._add(other)
        except AttributeError:
            raise TypeError("unsupported operand type(s) for -: '%s' and '%s'" %
                            (type(other).__name__, type(self).__name__))
