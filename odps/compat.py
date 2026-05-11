# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import io
import json  # don't remove
import os
import platform
import sys
import warnings
import xml.etree.ElementTree as ElementTree
from unicodedata import east_asian_width

try:
    ElementTreeParseError = getattr(ElementTree, "ParseError")
except AttributeError:
    ElementTreeParseError = getattr(ElementTree, "XMLParserError")

PYPY = platform.python_implementation().lower() == "pypy"

# Definition of East Asian Width
# http://unicode.org/reports/tr11/
# Ambiguous width can be changed by option
_EAW_MAP = {"Na": 1, "N": 1, "W": 2, "F": 2, "H": 1}


def east_asian_len(data, encoding=None, ambiguous_width=1):
    """
    Calculate display width considering unicode East Asian Width
    """
    if isinstance(data, str):
        return sum([_EAW_MAP.get(east_asian_width(c), ambiguous_width) for c in data])
    else:
        return len(data)


from .lib import cgi_compat as cgi

UnsupportedOperation = io.UnsupportedOperation

try:
    from .lib.version import Version
except BaseException:
    from distutils.version import LooseVersion as Version

try:
    from html import unescape as html_unescape
except ImportError:  # pragma: no cover
    from html.parser import HTMLParser

    def html_unescape(s):
        return HTMLParser().unescape(s)


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
        warnings.warn(f"Import of pandas skipped. Reasons: {ex}")

try:
    import numpy as np

    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64
except ImportError:
    pass

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
    utc = datetime.timezone.utc
    FixedOffset = _FixedOffset
except AttributeError:
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


if not hasattr(datetime, "UTC"):
    datetime_utcnow = datetime.datetime.utcnow
else:

    def datetime_utcnow():
        return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


_legacy_items = {
    "BytesIO": lambda: io.BytesIO,
    "DECIMAL_TYPES": lambda: (__import__("decimal").Decimal,),
    "ConfigParser": lambda: __import__("configparser"),
    "Decimal": lambda: __import__("decimal").Decimal,
    "Empty": lambda: __import__("queue").Empty,
    "Enum": lambda: __import__("enum").Enum,
    "HTMLParser": lambda: __import__("html.parser", fromlist=["HTMLParser"]).HTMLParser,
    "Iterable": lambda: __import__("collections.abc", fromlist=["Iterable"]).Iterable,
    "Queue": lambda: __import__("queue").Queue,
    "Semaphore": lambda: __import__("threading").Semaphore,
    "StringIO": lambda: io.StringIO,
    "TimeoutError": lambda: TimeoutError,
    "UnsupportedOperation": lambda: io.UnsupportedOperation,
    "builtins": lambda: __import__("builtins"),
    "decimal": lambda: __import__("decimal"),
    "dictconfig": lambda: __import__(
        "logging.config", fromlist=["dictConfig"]
    ).dictConfig,
    "enum": lambda: __import__("enum"),
    "futures": lambda: __import__("concurrent.futures", fromlist=["futures"]),
    "irange": lambda: range,
    "izip": lambda: zip,
    "long_type": lambda: int,
    "lrange": lambda: (lambda *args: list(range(*args))),
    "lzip": lambda: (lambda *args: list(zip(*args))),
    "lkeys": lambda: (lambda d: list(d.keys())),
    "lvalues": lambda: (lambda d: list(d.values())),
    "litems": lambda: (lambda d: list(d.items())),
    "monotonic": lambda: __import__("time").monotonic,
    "parse_qsl": lambda: __import__("urllib.parse", fromlist=["parse_qsl"]).parse_qsl,
    "pickle": lambda: __import__("pickle"),
    "quote": lambda: __import__("urllib.parse", fromlist=["quote"]).quote,
    "quote_plus": lambda: __import__(
        "urllib.parse", fromlist=["quote_plus"]
    ).quote_plus,
    "reduce": lambda: __import__("functools").reduce,
    "reload_module": lambda: __import__("importlib").reload,
    "six": lambda: __import__("six"),
    "suppress": lambda: __import__("contextlib").suppress,
    "sys": lambda: sys,
    "unquote": lambda: __import__("urllib.parse", fromlist=["unquote"]).unquote,
    "urlencode": lambda: __import__("urllib.parse", fromlist=["urlencode"]).urlencode,
    "urlparse": lambda: __import__("urllib.parse", fromlist=["urlparse"]).urlparse,
    "urlretrieve": lambda: __import__(
        "urllib.request", fromlist=["urlretrieve"]
    ).urlretrieve,
}


def __getattr__(name):
    if name in _legacy_items:
        warnings.warn(
            f"Importing {name} from odps.compat is deprecated as support for Python<=3.6"
            " is dropped, please import it from original modules instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = _legacy_items[name]()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ElementTree",
    "ElementTreeParseError",
    "FixedOffset",
    "Monthdelta",
    "Version",
    "cgi",
    "datetime_utcnow",
    "html_unescape",
    "parsedate_to_datetime",
    "utc",
]
