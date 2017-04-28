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

import sys
import logging.config
import itertools
import platform
import warnings
try:
    import xml.etree.cElementTree as ElementTree
except ImportError:
    import xml.etree.ElementTree as ElementTree
try:
    ElementTreeParseError = getattr(ElementTree, 'ParseError')
except AttributeError:
    ElementTreeParseError = getattr(ElementTree, 'XMLParserError')
from unicodedata import east_asian_width

from .lib import six

PY26 = six.PY2 and sys.version_info[1] == 6
PY27 = six.PY2 and sys.version_info[1] == 7
LESS_PY26 = sys.version_info[:2] < (2, 6)
LESS_PY32 = sys.version_info[:2] < (3, 2)
LESS_PY33 = sys.version_info[:2] < (3, 3)
LESS_PY34 = sys.version_info[:2] < (3, 4)
LESS_PY35 = sys.version_info[:2] < (3, 5)
PYPY = platform.python_implementation().lower() == 'pypy'

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

# Definition of East Asian Width
# http://unicode.org/reports/tr11/
# Ambiguous width can be changed by option
_EAW_MAP = {'Na': 1, 'N': 1, 'W': 2, 'F': 2, 'H': 1}

import decimal
DECIMAL_TYPES = [decimal.Decimal, ]

import json  # don't remove

if six.PY3:
    lrange = lambda *x: list(range(*x))
    lzip = lambda *x: list(zip(*x))
    lkeys = lambda x: list(x.keys())
    lvalues = lambda x: list(x.values())
    litems = lambda x: list(x.items())

    irange = range
    izip = zip

    long_type = int

    import io
    StringIO = io.StringIO
    BytesIO = io.BytesIO

    if LESS_PY34:
        from .lib import enum
    else:
        import enum

    if LESS_PY33:
        try:
            import cdecimal as decimal

            DECIMAL_TYPES.append(decimal.Decimal)
        except:
            import decimal
    else:
        import decimal

    import unittest
    from collections import OrderedDict

    def u(s):
        return s

    def strlen(data, encoding=None):
        # encoding is for compat with PY2
        return len(data)

    def east_asian_len(data, encoding=None, ambiguous_width=1):
        """
        Calculate display width considering unicode East Asian Width
        """
        if isinstance(data, six.text_type):
            return sum([_EAW_MAP.get(east_asian_width(c), ambiguous_width) for c in data])
        else:
            return len(data)

    dictconfig = lambda config: logging.config.dictConfig(config)

    import builtins
    from concurrent import futures  # don't remove

    from datetime import timedelta
    total_seconds = timedelta.total_seconds
else:
    lrange = range
    lzip = zip
    lkeys = lambda x: x.keys()
    lvalues = lambda x: x.values()
    litems = lambda x: x.items()

    irange = xrange
    izip = itertools.izip

    long_type = long

    from .lib import enum

    try:
        import cdecimal as decimal
        DECIMAL_TYPES.append(decimal.Decimal)
    except ImportError:
        import decimal

    try:
        import cStringIO as StringIO
    except ImportError:
        import StringIO
    StringIO = BytesIO = StringIO.StringIO

    def u(s):
        return unicode(s, "unicode_escape")

    def strlen(data, encoding=None):
        try:
            data = data.decode(encoding)
        except UnicodeError:
            pass
        return len(data)

    def east_asian_len(data, encoding=None, ambiguous_width=1):
        """
        Calculate display width considering unicode East Asian Width
        """
        if isinstance(data, six.text_type):
            try:
                data = data.decode(encoding)
            except UnicodeError:
                pass
            return sum([_EAW_MAP.get(east_asian_width(c), ambiguous_width) for c in data])
        else:
            return len(data)

    if PY26:
        warnings.warn('Python 2.6 is no longer supported by the Python core team. A future version of PyODPS ' +
                      'will drop support for this version.')

        try:
            import unittest2 as unittest
        except ImportError:
            pass

        try:
            from ordereddict import OrderedDict
        except ImportError:
            raise

        def total_seconds(self):
            return self.days * 86400.0 + self.seconds + self.microseconds * 1.0e-6
    else:
        import unittest
        from collections import OrderedDict

        dictconfig = lambda config: logging.config.dictConfig(config)

        from datetime import timedelta
        total_seconds = timedelta.total_seconds

    import __builtin__ as builtins  # don't remove
    from .lib import futures  # don't remove

if PY26:
    try:
        import simplejson as json
    except ImportError:
        pass
if PY26 or LESS_PY32:
    try:
        from .tests.dictconfig import dictConfig
        dictconfig = lambda config: dictConfig(config)
    except ImportError:
        pass

if six.PY3:
    from contextlib import suppress
else:
    from contextlib import contextmanager

    @contextmanager
    def suppress(*exceptions):
        try:
            yield
        except exceptions:
            pass

Enum = enum.Enum
DECIMAL_TYPES = tuple(DECIMAL_TYPES)
Decimal = decimal.Decimal

try:
    import pandas as pd
    if not hasattr(pd.DataFrame, 'sort_values'):
        pd.DataFrame.sort_values = pd.DataFrame.sort
except ImportError:
    pass

from .lib.lib_utils import isvalidattr, dir2, raise_exc, getargspec, getfullargspec

from .lib.six.moves import reduce
from .lib.six.moves import reload_module
from .lib.six.moves.queue import Queue, Empty
from .lib.six.moves.urllib.request import urlretrieve
from .lib.six.moves import cPickle as pickle
from .lib.six.moves.urllib.parse import urlencode, urlparse, unquote, quote, quote_plus, parse_qsl
from .lib.six.moves import configparser as ConfigParser


try:
    import pytz
    utc = pytz.utc
    FixedOffset = pytz._FixedOffset
except ImportError:
    import datetime
    _ZERO_TIMEDELTA = datetime.timedelta(0)

    # A class building tzinfo objects for fixed-offset time zones.
    # Note that FixedOffset(0, "UTC") is a different way to build a
    # UTC tzinfo object.

    class FixedOffset(datetime.tzinfo):
        """Fixed offset in minutes east from UTC."""

        def __init__(self, offset, name=None):
            self.__offset = datetime.timedelta(minutes=offset)
            self.__name = name

        def utcoffset(self, dt):
            return self.__offset

        def tzname(self, dt):
            return self.__name

        def dst(self, dt):
            return _ZERO_TIMEDELTA


    utc = FixedOffset(0, 'UTC')


__all__ = ['sys', 'builtins', 'logging.config', 'unittest', 'OrderedDict', 'dictconfig', 'suppress',
           'reduce', 'reload_module', 'Queue', 'Empty', 'ElementTree', 'ElementTreeParseError',
           'urlretrieve', 'pickle', 'urlencode', 'urlparse', 'unquote', 'quote', 'quote_plus', 'parse_qsl',
           'Enum', 'ConfigParser', 'decimal', 'Decimal', 'DECIMAL_TYPES', 'FixedOffset', 'utc']
