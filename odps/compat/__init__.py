# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import sys
import logging.config
import itertools
try:
    import xml.etree.cElementTree as ElementTree
except ImportError:
    import xml.etree.ElementTree as ElementTree
from unicodedata import east_asian_width

from . import six

PY26 = six.PY2 and sys.version_info[1] == 6
LESS_PY26 = six.PY2 and sys.version_info[1] < 6
LESS_PY32 = six.PY3 and sys.version_info[1] < 2
LESS_PY33 = six.PY3 and sys.version_info[1] < 3
LESS_PY34 = six.PY3 and sys.version_info[1] < 4

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

# Definition of East Asian Width
# http://unicode.org/reports/tr11/
# Ambiguous width can be changed by option
_EAW_MAP = {'Na': 1, 'N': 1, 'W': 2, 'F': 2, 'H': 1}

import decimal
DECIMAL_TYPES = [decimal.Decimal, ]

if six.PY3:
    lrange = lambda *x: list(range(*x))
    lzip = lambda *x: list(zip(*x))
    lkeys = lambda x: list(x.keys())
    lvalues = lambda x: list(x.values())
    litems = lambda x: list(x.items())

    irange = range
    izip = zip

    import io
    StringIO = io.StringIO
    BytesIO = io.BytesIO

    if LESS_PY34:
        from ..lib import enum
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
else:
    lrange = range
    lzip = zip
    lkeys = lambda x: x.keys()
    lvalues = lambda x: x.values()
    litems = lambda x: x.items()

    irange = xrange
    izip = itertools.izip

    from ..lib import enum

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
        import warnings
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

    else:
        import unittest
        from collections import OrderedDict

        dictconfig = lambda config: logging.config.dictConfig(config)

    import __builtin__ as builtins
if PY26 or LESS_PY32:
    try:
        from ..tests.dictconfig import dictConfig
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

from .six.moves import reduce
from .six.moves import reload_module
from .six.moves.queue import Queue, Empty
from .six.moves.urllib.request import urlretrieve
from .six.moves import cPickle as pickle
from .six.moves.urllib.parse import urlparse, unquote, quote, quote_plus, parse_qsl
from .six.moves import configparser as ConfigParser

__all__ = ['sys', 'builtins', 'logging.config', 'unittest', 'OrderedDict', 'dictconfig', 'suppress',
           'reduce', 'reload_module', 'Queue', 'Empty',
           'urlretrieve', 'pickle', 'urlparse', 'unquote', 'quote', 'quote_plus', 'parse_qsl',
           'Enum', 'ConfigParser', 'decimal', 'Decimal', 'DECIMAL_TYPES']
