# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import itertools
import logging.config
import os
import platform
import sys
import warnings

try:
    if sys.version_info[:2] < (3, 3):
        import xml.etree.cElementTree as ElementTree
    else:
        import xml.etree.ElementTree as ElementTree
except ImportError:
    import xml.etree.ElementTree as ElementTree
try:
    ElementTreeParseError = getattr(ElementTree, "ParseError")
except AttributeError:
    ElementTreeParseError = getattr(ElementTree, "XMLParserError")
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from unicodedata import east_asian_width

from .lib import six

PY27 = six.PY2 and sys.version_info[1] == 7
LESS_PY32 = sys.version_info[:2] < (3, 2)
LESS_PY33 = sys.version_info[:2] < (3, 3)
LESS_PY34 = sys.version_info[:2] < (3, 4)
LESS_PY35 = sys.version_info[:2] < (3, 5)
LESS_PY36 = sys.version_info[:2] < (3, 6)
PYPY = platform.python_implementation().lower() == "pypy"

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

# Definition of East Asian Width
# http://unicode.org/reports/tr11/
# Ambiguous width can be changed by option
_EAW_MAP = {"Na": 1, "N": 1, "W": 2, "F": 2, "H": 1}

import decimal

DECIMAL_TYPES = [
    decimal.Decimal,
]

import json  # don't remove

try:
    TimeoutError = TimeoutError
except NameError:
    TimeoutError = type("TimeoutError", (RuntimeError,), {})


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

    if LESS_PY36:
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
            return sum(
                [_EAW_MAP.get(east_asian_width(c), ambiguous_width) for c in data]
            )
        else:
            return len(data)

    dictconfig = lambda config: logging.config.dictConfig(config)

    import builtins
    from concurrent import futures  # don't remove

    from .lib import cgi_compat as cgi

    UnsupportedOperation = io.UnsupportedOperation

    try:
        from .lib.version import Version
    except BaseException:
        from distutils.version import LooseVersion as Version

    from threading import Semaphore
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
            return sum(
                [_EAW_MAP.get(east_asian_width(c), ambiguous_width) for c in data]
            )
        else:
            return len(data)

    dictconfig = lambda config: logging.config.dictConfig(config)

    import cgi

    import __builtin__ as builtins  # don't remove

    from .lib import futures  # don't remove

    UnsupportedOperation = type("UnsupportedOperation", (OSError, ValueError), {})

    from distutils.version import LooseVersion as Version
    from threading import _Semaphore as _PySemaphore

    from .lib.monotonic import monotonic

    class Semaphore(_PySemaphore):
        def acquire(self, blocking=True, timeout=None):
            if not blocking and timeout is not None:
                raise ValueError("can't specify timeout for non-blocking acquire")
            rc = False
            endtime = None
            with self.__cond:
                while self.__value == 0:
                    if not blocking:
                        break
                    if timeout is not None:
                        if endtime is None:
                            endtime = monotonic() + timeout
                        else:
                            timeout = endtime - monotonic()
                            if timeout <= 0:
                                break
                    self.__cond.wait(timeout)
                else:
                    self.__value -= 1
                    rc = True
            return rc


if LESS_PY32:
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

    if not hasattr(pd.DataFrame, "sort_values"):
        pd.DataFrame.sort_values = pd.DataFrame.sort

    from pandas.core.internals import blocks as pd_blocks

    if not hasattr(pd_blocks, "new_block"):
        pd_blocks.new_block = pd_blocks.make_block

    if not hasattr(pd.RangeIndex, "start"):
        pd.RangeIndex.start = property(fget=lambda x: x._start)
        pd.RangeIndex.stop = property(fget=lambda x: x._stop)
        pd.RangeIndex.step = property(fget=lambda x: x._step)
except (ImportError, ValueError) as ex:
    if not isinstance(ex, ImportError):
        warnings.warn("Import of pandas skipped. Reasons: %s" % str(ex))

try:
    import numpy as np

    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64
except ImportError:
    pass

if sys.version_info[0] > 2:
    # workaround for polluted sys.path due to some packages
    try:
        import http.client
    except ImportError:
        sys.modules.pop("http", None)
        old_path = list(sys.path)
        sys.path = [os.path.dirname(os.__file__)] + sys.path
        import http.client

        sys.path = old_path

import datetime

from .lib.ext_types import Monthdelta
from .lib.lib_utils import dir2, getargspec, getfullargspec, isvalidattr
from .lib.six.moves import configparser as ConfigParser
from .lib.six.moves import cPickle as pickle
from .lib.six.moves import reduce, reload_module
from .lib.six.moves.queue import Empty, Queue
from .lib.six.moves.urllib.parse import (
    parse_qsl,
    quote,
    quote_plus,
    unquote,
    urlencode,
    urlparse,
)
from .lib.six.moves.urllib.request import urlretrieve


class _FixedOffset(datetime.tzinfo):
    """
    A class building tzinfo objects for fixed-offset time zones.
    Note that FixedOffset(0, "UTC") is a different way to build a
    UTC tzinfo object.
    """

    def __init__(self, offset, name=None):
        self.__offset = datetime.timedelta(minutes=offset)
        self.__name = name

    def utcoffset(self, dt):
        return self.__offset

    def tzname(self, dt):
        return self.__name

    def dst(self, dt):
        return _ZERO_TIMEDELTA


try:
    import zoneinfo

    utc = zoneinfo.ZoneInfo("UTC")
    FixedOffset = _FixedOffset
except ImportError:
    try:
        import pytz

        utc = pytz.utc
        FixedOffset = pytz._FixedOffset
    except ImportError:
        _ZERO_TIMEDELTA = datetime.timedelta(0)
        FixedOffset = _FixedOffset
        utc = FixedOffset(0, "UTC")


try:
    from email.utils import parsedate_to_datetime
except ImportError:
    import datetime
    from email.utils import parsedate_tz

    def parsedate_to_datetime(data):
        dt_tuple_with_tz = parsedate_tz(data)
        dtuple = dt_tuple_with_tz[:-1]
        tz = dt_tuple_with_tz[-1]
        if tz is None:
            return datetime.datetime(*dtuple[:6])
        return datetime.datetime(*dtuple[:6], tzinfo=FixedOffset(tz / 60.0))


__all__ = [
    "sys",
    "builtins",
    "logging.config",
    "dictconfig",
    "suppress",
    "reduce",
    "reload_module",
    "Queue",
    "Empty",
    "ElementTree",
    "ElementTreeParseError",
    "urlretrieve",
    "pickle",
    "urlencode",
    "urlparse",
    "unquote",
    "quote",
    "quote_plus",
    "parse_qsl",
    "Enum",
    "ConfigParser",
    "decimal",
    "Decimal",
    "DECIMAL_TYPES",
    "FixedOffset",
    "utc",
    "Monthdelta",
    "Iterable",
    "TimeoutError",
    "cgi",
    "parsedate_to_datetime",
    "Version",
    "Semaphore",
]
