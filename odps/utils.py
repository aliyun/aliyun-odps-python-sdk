#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from __future__ import absolute_import, print_function

import base64
import bisect
import calendar
import codecs
import copy
import glob
import hmac
import logging
import math
import multiprocessing
import os
import random
import re
import shutil
import string
import struct
import sys
import threading
import time
import traceback
import types
import uuid
import warnings
import xml.dom.minidom
from base64 import b64encode
from datetime import date, datetime, timedelta
from email.utils import formatdate, parsedate_tz
from hashlib import md5, sha1

try:
    from collections.abc import Hashable, Iterable, Mapping
except ImportError:
    from collections import Hashable, Mapping, Iterable

from . import compat, options
from .compat import FixedOffset, getargspec, parsedate_to_datetime, six, utc
from .lib.monotonic import monotonic

try:
    import zoneinfo
except ImportError:
    zoneinfo = None
try:
    import pytz
except ImportError:
    pytz = None

try:
    from .src.utils_c import (
        CMillisecondsConverter,
        to_binary,
        to_lower_str,
        to_str,
        to_text,
    )
except ImportError:
    CMillisecondsConverter = to_str = to_text = to_binary = to_lower_str = None

TEMP_TABLE_PREFIX = "tmp_pyodps_"
if six.PY3:  # make flake8 happy
    unicode = str

_IS_WINDOWS = sys.platform.lower().startswith("win")

logger = logging.getLogger(__name__)

notset = object()


def deprecated(msg, cond=None):
    def _decorator(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emmitted
        when the function is used."""

        @six.wraps(func)
        def _new_func(*args, **kwargs):
            warn_msg = "Call to deprecated function %s." % func.__name__
            if isinstance(msg, six.string_types):
                warn_msg += " " + msg
            if cond is None or cond():
                warnings.warn(warn_msg, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return _new_func

    if isinstance(msg, (types.FunctionType, types.MethodType)):
        return _decorator(msg)
    return _decorator


class ExperimentalNotAllowed(Exception):
    pass


def experimental(msg, cond=None):
    warn_cache = set()

    real_cond = cond
    if callable(cond):
        cond_spec = getargspec(cond)
        if not cond_spec.args and not cond_spec.varargs:
            real_cond = lambda *_, **__: cond()

    def _decorator(func):
        @six.wraps(func)
        def _new_func(*args, **kwargs):
            if real_cond is None or real_cond(*args, **kwargs):
                if not str_to_bool(os.environ.get("PYODPS_EXPERIMENTAL", "true")):
                    err_msg = (
                        "Calling to experimental method %s is denied." % func.__name__
                    )
                    if isinstance(msg, six.string_types):
                        err_msg += " " + msg
                    raise ExperimentalNotAllowed(err_msg)

                if func not in warn_cache:
                    warn_msg = "Call to experimental function %s." % func.__name__
                    if isinstance(msg, six.string_types):
                        warn_msg += " " + msg

                    warnings.warn(warn_msg, category=FutureWarning, stacklevel=2)
                    warn_cache.add(func)
            return func(*args, **kwargs)

        # intentionally eliminate __doc__ for Volume 2
        _new_func.__doc__ = None
        return _new_func

    if isinstance(msg, (types.FunctionType, types.MethodType)):
        return _decorator(msg)
    return _decorator


def fixed_writexml(self, writer, indent="", addindent="", newl=""):
    # indent = current indentation
    # addindent = indentation to add to higher levels
    # newl = newline string
    writer.write(indent + "<" + self.tagName)

    attrs = self._get_attributes()
    a_names = compat.lkeys(attrs)
    a_names.sort()

    for a_name in a_names:
        writer.write(" %s=\"" % a_name)
        xml.dom.minidom._write_data(writer, attrs[a_name].value)
        writer.write("\"")
    if self.childNodes:
        if (
            len(self.childNodes) == 1
            and self.childNodes[0].nodeType == xml.dom.minidom.Node.TEXT_NODE
        ):
            writer.write(">")
            self.childNodes[0].writexml(writer, "", "", "")
            writer.write("</%s>%s" % (self.tagName, newl))
            return
        writer.write(">%s" % (newl))
        for node in self.childNodes:
            node.writexml(writer, indent + addindent, addindent, newl)
        writer.write("%s</%s>%s" % (indent, self.tagName, newl))
    else:
        writer.write("/>%s" % (newl))


# replace minidom's function with ours
xml.dom.minidom.Element.writexml = fixed_writexml
xml_fixed = lambda: None


def hmac_sha1(secret, data):
    return b64encode(hmac.new(secret, data, sha1).digest())


def md5_hexdigest(data):
    if data is None:
        return None
    return md5(to_binary(data)).hexdigest()


def rshift(val, n):
    return val >> n if val >= 0 else (val + 0x100000000) >> n


def long_bits_to_double(bits):
    """
    @type  bits: long
    @param bits: the bit pattern in IEEE 754 layout

    @rtype:  float
    @return: the double-precision floating-point value corresponding
             to the given bit pattern C{bits}.
    """
    return struct.unpack("d", struct.pack("Q", bits))[0]


def double_to_raw_long_bits(value):
    """
    @type  value: float
    @param value: a Python (double-precision) float value

    @rtype: long
    @return: the IEEE 754 bit representation (64 bits as a long integer)
             of the given double-precision floating-point value.
    """
    # pack double into 64 bits, then unpack as long int
    return struct.unpack("Q", struct.pack("d", float(value)))[0]


def camel_to_underline(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def underline_to_capitalized(name):
    return "".join([s[0].upper() + s[1 : len(s)] for s in name.strip("_").split("_")])


def underline_to_camel(name):
    parts = name.split("_")
    return parts[0] + "".join(v.title() for v in parts[1:])


def long_to_int(value):
    if value & 0x80000000:
        return int(-((value ^ 0xFFFFFFFF) + 1))
    else:
        return int(value)


def int_to_uint(v):
    if v < 0:
        return int(v + 2**32)
    return v


def long_to_uint(value):
    v = long_to_int(value)
    return int_to_uint(v)


def stringify_expt():
    lines = traceback.format_exception(*sys.exc_info())
    return "\n".join(lines)


def str_to_printable(field_name, auto_quote=True):
    if not field_name:
        return field_name

    escapes = {
        "\\": "\\\\",
        '\'': '\\\'',
        '"': '\\"',
        "\a": "\\a",
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\v": "\\v",
        " ": " ",
    }

    def _escape_char(c):
        if c in escapes:
            return escapes[c]
        elif c < " ":
            return "\\x%02x" % ord(c)
        else:
            return c

    need_escape = lambda c: c <= " " or c in escapes
    if any(need_escape(c) for c in field_name):
        ret = "".join(_escape_char(ch) for ch in field_name)
        return '"' + ret + '"' if auto_quote else ret
    return field_name


def indent(text, n_spaces):
    if n_spaces <= 0:
        return text
    block = " " * n_spaces
    return "\n".join((block + it) if len(it) > 0 else it for it in text.split("\n"))


def parse_rfc822(s, use_legacy_parsedate=None):
    if s is None:
        return None

    use_legacy_parsedate = (
        use_legacy_parsedate
        if use_legacy_parsedate is not None
        else options.use_legacy_parsedate
    )
    if use_legacy_parsedate:
        date_tuple = parsedate_tz(s)
        return datetime(*date_tuple[:6])

    time_obj = parsedate_to_datetime(s)
    if time_obj.tzinfo is None:
        return time_obj
    time_obj = time_obj.astimezone(utc)
    gmt_ts = calendar.timegm(time_obj.timetuple())
    return datetime.fromtimestamp(gmt_ts)


def gen_rfc822(dt=None, localtime=False, usegmt=False):
    if dt is not None:
        t = time.mktime(dt.timetuple())
    else:
        t = None
    return formatdate(t, localtime=localtime, usegmt=usegmt)


try:
    _antique_mills = time.mktime(datetime(1928, 1, 1).timetuple()) * 1000
except OverflowError:
    _antique_mills = (
        int((datetime(1928, 1, 1) - datetime.utcfromtimestamp(0)).total_seconds())
        * 1000
    )
_min_datetime_mills = int(
    (datetime.min - datetime.utcfromtimestamp(0)).total_seconds() * 1000
)
_antique_errmsg = (
    "Date older than 1928-01-01 and may contain errors. "
    "Ignore this error by configuring `options.allow_antique_date` to True."
)
_min_datetime_errmsg = (
    "Date exceed range Python can handle. If you are reading data with tunnel, read "
    "the value as None by setting options.tunnel.overflow_date_as_none to True, "
    "or convert the value into strings with SQL before processing them with Python."
)


def to_timestamp(dt, local_tz=None, is_dst=False):
    return int(to_milliseconds(dt, local_tz=local_tz, is_dst=is_dst) / 1000.0)


class MillisecondsConverter(object):
    _inst_cache = dict()

    @classmethod
    def _get_tz(cls, tz):
        if isinstance(tz, six.string_types):
            if pytz is None and zoneinfo is None:
                raise ImportError(
                    "Package `pytz` is needed when specifying string-format time zone."
                )
            else:
                return get_zone_from_name(tz)
        else:
            return tz

    def __new__(cls, local_tz=None, is_dst=False):
        cache_key = (cls, local_tz, is_dst)
        if cache_key in cls._inst_cache:
            return cls._inst_cache[cache_key]
        o = super(MillisecondsConverter, cls).__new__(cls)
        o.__init__(local_tz, is_dst)
        cls._inst_cache[cache_key] = o
        return o

    def _windows_mktime(self, timetuple):
        if self._local_tz:
            fromtimestamp = datetime.fromtimestamp
            mktime = time.mktime
        else:
            fromtimestamp = datetime.utcfromtimestamp
            mktime = calendar.timegm

        if timetuple[0] > 1970:
            return mktime(timetuple)
        dt = datetime(*timetuple[:6])
        epoch = fromtimestamp(0)
        return int((dt - epoch).total_seconds())

    def _windows_fromtimestamp(self, seconds):
        fromtimestamp = (
            datetime.fromtimestamp if self._local_tz else datetime.utcfromtimestamp
        )
        if seconds >= 0:
            return fromtimestamp(seconds)
        epoch = fromtimestamp(0)
        return epoch + timedelta(seconds=seconds)

    def __init__(self, local_tz=None, is_dst=False):
        self._local_tz = local_tz if local_tz is not None else options.local_timezone
        if self._local_tz is None:
            self._local_tz = True
        self._use_default_tz = type(self._local_tz) is bool

        self._allow_antique = options.allow_antique_date or _antique_mills is None
        self._is_dst = is_dst

        if self._local_tz:
            self._mktime = time.mktime
            self._fromtimestamp = datetime.fromtimestamp
        else:
            self._mktime = calendar.timegm
            self._fromtimestamp = datetime.utcfromtimestamp

        if _IS_WINDOWS:
            # special logic for negative timestamp under Windows
            self._mktime = self._windows_mktime
            self._fromtimestamp = self._windows_fromtimestamp

        self._tz = self._get_tz(self._local_tz) if not self._use_default_tz else None
        if hasattr(self._tz, "localize"):
            self._localize = lambda dt: self._tz.localize(dt, is_dst=is_dst)
        else:
            self._localize = lambda dt: dt.replace(tzinfo=self._tz)

    def to_milliseconds(self, dt):
        from .errors import DatetimeOverflowError

        if not self._use_default_tz and dt.tzinfo is None:
            dt = self._localize(dt)

        if dt.tzinfo is not None:
            mills = int(
                (
                    calendar.timegm(dt.astimezone(compat.utc).timetuple())
                    + dt.microsecond / 1000000.0
                )
                * 1000
            )
        else:
            mills = int(
                (self._mktime(dt.timetuple()) + dt.microsecond / 1000000.0) * 1000
            )

        if not self._allow_antique and mills < _antique_mills:
            raise DatetimeOverflowError(_antique_errmsg)
        return mills

    def from_milliseconds(self, milliseconds):
        from .errors import DatetimeOverflowError

        if not self._allow_antique and milliseconds < _antique_mills:
            raise DatetimeOverflowError(_antique_errmsg)
        if milliseconds < _min_datetime_mills:
            raise DatetimeOverflowError(_min_datetime_errmsg)

        seconds = compat.long_type(math.floor(milliseconds / 1000))
        microseconds = compat.long_type(milliseconds) % 1000 * 1000
        if self._use_default_tz:
            return self._fromtimestamp(seconds).replace(microsecond=microseconds)
        else:
            return (
                datetime.utcfromtimestamp(seconds)
                .replace(microsecond=microseconds, tzinfo=compat.utc)
                .astimezone(self._tz)
            )


def to_milliseconds(dt, local_tz=None, is_dst=False, force_py=False):
    cls = CMillisecondsConverter
    if force_py or cls is None:
        cls = MillisecondsConverter
    f = cls(local_tz, is_dst=is_dst)
    return f.to_milliseconds(dt)


def to_days(dt):
    start_day = date(1970, 1, 1)
    return (dt - start_day).days


def to_date(delta_day):
    start_day = date(1970, 1, 1)
    return start_day + timedelta(delta_day)


def to_datetime(milliseconds, local_tz=None, force_py=False):
    cls = CMillisecondsConverter
    if force_py or cls is None:
        cls = MillisecondsConverter
    f = cls(local_tz)
    return f.from_milliseconds(milliseconds)


def strptime_with_tz(dt, format="%Y-%m-%d %H:%M:%S"):
    try:
        return datetime.strptime(dt, format)
    except ValueError:
        naive_date_str, _, offset_str = dt.rpartition(" ")
        naive_dt = datetime.strptime(naive_date_str, format)
        offset = int(offset_str[-4:-2]) * 60 + int(offset_str[-2:])
        if offset_str[0] == "-":
            offset = -offset
        return naive_dt.replace(tzinfo=FixedOffset(offset))


if to_binary is None or to_text is None or to_str is None or to_lower_str is None:

    def to_binary(text, encoding="utf-8"):
        if text is None:
            return text
        if isinstance(text, six.text_type):
            return text.encode(encoding)
        elif isinstance(text, (six.binary_type, bytearray)):
            return bytes(text)
        else:
            return str(text).encode(encoding) if six.PY3 else str(text)

    def to_text(binary, encoding="utf-8"):
        if binary is None:
            return binary
        if isinstance(binary, (six.binary_type, bytearray)):
            return binary.decode(encoding)
        elif isinstance(binary, six.text_type):
            return binary
        else:
            return str(binary) if six.PY3 else str(binary).decode(encoding)

    def to_str(text, encoding="utf-8"):
        return (
            to_text(text, encoding=encoding)
            if six.PY3
            else to_binary(text, encoding=encoding)
        )

    def to_lower_str(s, encoding="utf-8"):
        if s is None:
            return None
        return to_str(s, encoding).lower()


def get_zone_from_name(tzname):
    return zoneinfo.ZoneInfo(tzname) if zoneinfo else pytz.timezone(tzname)


def get_zone_name(tz):
    return getattr(tz, "key", None) or getattr(tz, "zone", None)


# fix encoding conversion problem under windows
if sys.platform == "win32":

    def _replace_default_encoding(func):
        def _fun(s, encoding=None):
            return func(s, encoding=encoding or options.display.encoding)

        _fun.__name__ = func.__name__
        _fun.__doc__ = func.__doc__
        return _fun

    to_binary = _replace_default_encoding(to_binary)
    to_text = _replace_default_encoding(to_text)
    to_str = _replace_default_encoding(to_str)


def is_lambda(f):
    lam = lambda: 0
    return isinstance(f, type(lam)) and f.__name__ == lam.__name__


def str_to_kv(string, typ=None):
    d = dict()
    for pair in string.split(","):
        k, v = pair.split(":", 1)
        if typ:
            v = typ(v)
        d[k] = v
    return d


def interval_select(val, intervals, targets):
    return targets[bisect.bisect_left(intervals, val)]


def is_namedtuple(obj):
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def str_to_bool(s):
    if isinstance(s, bool) or s is None:
        return s
    s = s.lower().strip()
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        raise ValueError(s)


def bool_to_str(s):
    return str(s).lower()


def get_root_dir():
    return os.path.dirname(sys.modules[__name__].__file__)


def load_text_file(path):
    file_path = get_root_dir() + path
    if not os.path.exists(file_path):
        return None
    with codecs.open(file_path, encoding="utf-8") as f:
        inp_file = f.read()
        f.close()
    return inp_file


def load_file_paths(pattern):
    file_path = os.path.normpath(
        os.path.dirname(sys.modules[__name__].__file__) + pattern
    )
    return glob.glob(file_path)


def load_static_file_paths(path):
    return load_file_paths("/static/" + path)


def load_text_files(pattern, func=None):
    file_path = os.path.normpath(
        os.path.dirname(sys.modules[__name__].__file__) + pattern
    )
    content_dict = dict()
    for file_path in glob.glob(file_path):
        _, fn = os.path.split(file_path)
        if func and not func(fn):
            continue
        with codecs.open(file_path, encoding="utf-8") as f:
            content_dict[fn] = f.read()
            f.close()
    return content_dict


def load_static_text_file(path):
    return load_text_file("/static/" + path)


def load_internal_static_text_file(path):
    return load_text_file("/internal/static/" + path)


def load_static_text_files(pattern, func=None):
    return load_text_files("/static/" + pattern, func)


def init_progress_bar(val=1, use_console=True):
    try:
        from traitlets import TraitError

        ipython = True
    except ImportError:
        try:
            from IPython.utils.traitlets import TraitError

            ipython = True
        except ImportError:
            ipython = False

    from .console import ProgressBar, is_widgets_available

    if not ipython:
        bar = ProgressBar(val) if use_console else None
    else:
        try:
            if is_widgets_available():
                bar = ProgressBar(val, True)
            else:
                bar = ProgressBar(val) if use_console else None
        except TraitError:
            bar = ProgressBar(val) if use_console else None

    return bar


def init_progress_ui(val=1, lock=False, use_console=True, mock=False):
    from .ui import ProgressGroupUI, html_notify

    progress_group = None
    bar = None
    if not mock and is_main_thread():
        bar = init_progress_bar(val=val, use_console=use_console)
        if bar and bar._ipython_widget:
            try:
                progress_group = ProgressGroupUI(bar._ipython_widget)
            except:
                pass

    _lock = threading.Lock() if lock else None

    def ui_method(func):
        def inner(*args, **kwargs):
            if mock:
                return
            if _lock:
                with _lock:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return inner

    class ProgressUI(object):
        @ui_method
        def update(self, value=None):
            if bar:
                bar.update(value=value)

        @ui_method
        def current_progress(self):
            if bar and hasattr(bar, "_current_value"):
                return bar._current_value

        @ui_method
        def inc(self, value):
            if bar and hasattr(bar, "_current_value"):
                current_val = bar._current_value
                bar.update(current_val + value)

        @ui_method
        def status(self, prefix, suffix="", clear_keys=False):
            if progress_group:
                if clear_keys:
                    progress_group.clear_keys()
                progress_group.prefix = prefix
                progress_group.suffix = suffix

        @ui_method
        def add_keys(self, keys):
            if progress_group:
                progress_group.add_keys(keys)

        @ui_method
        def remove_keys(self, keys):
            if progress_group:
                progress_group.remove_keys(keys)

        @ui_method
        def update_group(self):
            if progress_group:
                progress_group.update()

        @ui_method
        def notify(self, msg):
            html_notify(msg)

        @ui_method
        def close(self):
            if bar:
                bar.close()
            if progress_group:
                progress_group.close()

    return ProgressUI()


def escape_odps_string(src):
    trans_dict = {
        "\b": r"\b",
        "\t": r"\t",
        "\n": r"\n",
        "\r": r"\r",
        "'": r"\'",
        '"': r"\"",
        "\\": r"\\",
        "\0": r"\0",
    }
    return "".join(trans_dict[ch] if ch in trans_dict else ch for ch in src)


def to_odps_scalar(s):
    try:
        from pandas import Timestamp as pd_Timestamp
    except ImportError:
        pd_Timestamp = type("DummyType", (object,), {})

    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "NULL"
    if isinstance(s, six.string_types):
        return "'%s'" % escape_odps_string(s)
    elif isinstance(s, (six.binary_type, bytearray)):
        return "unbase64('%s')" % base64.b64encode(s).decode()
    elif isinstance(s, (datetime, pd_Timestamp)):
        microsec = s.microsecond
        nanosec = getattr(s, "nanosecond", 0)
        if microsec or nanosec:
            s = s.strftime("%Y-%m-%d %H:%M:%S.%f") + ("%03d" % nanosec)
            out_type = "TIMESTAMP"
        else:
            s = s.strftime("%Y-%m-%d %H:%M:%S")
            out_type = "DATETIME"
        return "CAST('%s' AS %s)" % (escape_odps_string(s), out_type)
    return str(s)


def replace_sql_parameters(sql, ns):
    param_re = re.compile(r":([a-zA-Z_][a-zA-Z0-9_]*)")

    def is_numeric(val):
        return isinstance(val, (six.integer_types, float))

    def is_sequence(val):
        return isinstance(val, (tuple, set, list))

    def format_string(val):
        return "'{0}'".format(escape_odps_string(str(val)))

    def format_numeric(val):
        return repr(val)

    def format_sequence(val):
        escaped = [
            format_numeric(v) if is_numeric(v) else format_string(v) for v in val
        ]
        return "({0})".format(", ".join(escaped))

    def replace(matched):
        name = matched.group(1)
        val = ns.get(name)
        if val is None:
            return matched.group(0)
        elif is_numeric(val):
            return format_numeric(val)
        elif is_sequence(val):
            return format_sequence(val)
        else:
            return format_string(val)

    return param_re.sub(replace, sql)


def is_main_process():
    return "main" in multiprocessing.current_process().name.lower()


survey_calls = dict()


def survey(func):
    @six.wraps(func)
    def wrapped(*args, **kwargs):
        arg_spec = getargspec(func)

        if "self" in arg_spec.args:
            func_cls = args[0].__class__
        else:
            func_cls = None

        if func_cls:
            func_sig = ".".join([func_cls.__module__, func_cls.__name__, func.__name__])
        else:
            func_sig = ".".join([func.__module__, func.__name__])

        add_survey_call(func_sig)
        return func(*args, **kwargs)

    return wrapped


def add_survey_call(group):
    if any(r.search(group) is not None for r in options.skipped_survey_regexes):
        return
    if group not in survey_calls:
        survey_calls[group] = 1
    else:
        survey_calls[group] += 1


def get_survey_calls():
    return copy.copy(survey_calls)


def clear_survey_calls():
    survey_calls.clear()


def require_package(pack_name):
    def _decorator(func):
        try:
            __import__(pack_name, fromlist=[""])
            return func
        except ImportError:
            return None

    return _decorator


def gen_repr_object(**kwargs):
    obj = type("ReprObject", (), {})
    text = kwargs.pop("text", None)
    if six.PY2 and isinstance(text, unicode):
        text = text.encode("utf-8")
    if text:
        setattr(obj, "text", text)
        setattr(obj, "__repr__", lambda self: text)
    for k, v in six.iteritems(kwargs):
        setattr(obj, k, v)
        setattr(obj, "_repr_{0}_".format(k), lambda self: v)
    if "gv" in kwargs:
        try:
            from graphviz import Source

            setattr(
                obj,
                "_repr_svg_",
                lambda self: Source(self._repr_gv_(), encoding="utf-8")._repr_svg_(),
            )
        except ImportError:
            pass
    return obj()


def build_pyodps_dir(*args):
    default_dir = os.path.join(os.path.expanduser("~"), ".pyodps")
    if sys.platform == "win32" and "APPDATA" in os.environ:
        win_default_dir = os.path.join(os.environ["APPDATA"], "pyodps")
        if os.path.exists(default_dir):
            shutil.move(default_dir, win_default_dir)
        default_dir = win_default_dir
    home_dir = os.environ.get("PYODPS_DIR") or default_dir
    return os.path.join(home_dir, *args)


def object_getattr(obj, attr, default=None):
    try:
        return object.__getattribute__(obj, attr)
    except AttributeError:
        return default


def attach_internal(cls):
    cls_path = cls.__module__ + "." + cls.__name__
    try:
        from .internal.core import MIXIN_TARGETS

        mixin_cls = MIXIN_TARGETS[cls_path]
        for method_name in dir(mixin_cls):
            if method_name.startswith("_"):
                continue
            att = getattr(mixin_cls, method_name)
            if six.PY2 and type(att).__name__ in ("instancemethod", "method"):
                att = att.__func__
            setattr(cls, method_name, att)
        return cls
    except ImportError:
        return cls


def is_main_thread():
    if hasattr(threading, "main_thread"):
        return threading.current_thread() is threading.main_thread()
    return threading.current_thread().__class__.__name__ == "_MainThread"


def write_log(msg):
    """Legacy method to keep compatibility"""
    logger.info(msg)


def split_quoted(s, delimiter=",", maxsplit=0):
    pattern = r"""((?:[^""" + delimiter + r""""']|"[^"]*"|'[^']*')+)"""
    return re.split(pattern, s, maxsplit=maxsplit)[1::2]


def gen_temp_table():
    return "%s%s" % (TEMP_TABLE_PREFIX, str(uuid.uuid4()).replace("-", "_"))


def hashable(obj):
    if isinstance(obj, Hashable):
        items = obj
    elif isinstance(obj, Mapping):
        items = type(obj)((k, hashable(v)) for k, v in six.iteritems(obj))
    elif isinstance(obj, Iterable):
        items = tuple(hashable(item) for item in obj)
    else:
        raise TypeError(type(obj))

    return items


def thread_local_attribute(thread_local_name, default_value=None):
    attr_name = "_local_attr_%d" % random.randint(0, 99999999)

    def _get_thread_local(self):
        thread_local = getattr(self, thread_local_name, None)
        if thread_local is None:
            setattr(self, thread_local_name, threading.local())
            thread_local = getattr(self, thread_local_name)
        return thread_local

    def _getter(self):
        thread_local = _get_thread_local(self)
        if not hasattr(thread_local, attr_name) and callable(default_value):
            setattr(thread_local, attr_name, default_value())
        return getattr(thread_local, attr_name)

    def _setter(self, value):
        thread_local = _get_thread_local(self)
        setattr(thread_local, attr_name, value)

    return property(fget=_getter, fset=_setter)


def call_with_retry(func, *args, **kwargs):
    retry_num = 0
    retry_times = kwargs.pop("retry_times", options.retry_times)
    retry_timeout = kwargs.pop("retry_timeout", None)
    delay = kwargs.pop("delay", options.retry_delay)
    reset_func = kwargs.pop("reset_func", None)
    exc_type = kwargs.pop("exc_type", BaseException)
    allow_interrupt = kwargs.pop("allow_interrupt", True)
    no_raise = kwargs.pop("no_raise", False)

    start_time = monotonic() if retry_timeout is not None else None
    while True:
        try:
            return func(*args, **kwargs)
        except exc_type as ex:
            retry_num += 1
            time.sleep(delay)
            if allow_interrupt and isinstance(ex, KeyboardInterrupt):
                raise
            if (retry_times is not None and retry_num > retry_times) or (
                retry_timeout is not None
                and start_time is not None
                and monotonic() - start_time > retry_timeout
            ):
                if no_raise:
                    return sys.exc_info()
                raise
            if callable(reset_func):
                reset_func()


def get_id(n):
    if hasattr(n, "_node_id"):
        return n._node_id

    return id(n)


def strip_if_str(s):
    if isinstance(s, six.binary_type):
        s = to_str(s)
    if isinstance(s, six.string_types):
        return s.strip()
    return s


def with_wait_argument(func):
    func_spec = (
        compat.getfullargspec(func)
        if compat.getfullargspec
        else compat.getargspec(func)
    )
    args_set = set(func_spec.args)
    if hasattr(func_spec, "kwonlyargs"):
        args_set |= set(func_spec.kwonlyargs or [])
    has_varkw = (
        getattr(func_spec, "varkw", None) or getattr(func_spec, "keywords", None)
    ) is not None

    try:
        async_index = func_spec.args.index("async_")
    except ValueError:
        async_index = None

    @six.wraps(func)
    def wrapped(*args, **kwargs):
        if async_index is not None and len(args) >= async_index + 1:
            warnings.warn(
                "Please use async_ as a keyword argument, like obj.func(async_=True)",
                DeprecationWarning,
                stacklevel=2,
            )
            add_survey_call(".".join([func.__module__, func.__name__, "async_"]))
        elif "wait" in kwargs:
            kwargs["async_"] = not kwargs.pop("wait")
        elif "async" in kwargs:
            kwargs["async_"] = kwargs.pop("async")

        if not has_varkw and kwargs:
            no_args_match = [key for key in kwargs if key not in args_set]
            if no_args_match:
                warnings.warn(
                    "Arguments %s not supported, ignored by default. "
                    "Please check argument spellings." % (", ".join(no_args_match)),
                    stacklevel=2,
                )
            for arg in no_args_match:
                kwargs.pop(arg, None)
        return func(*args, **kwargs)

    return wrapped


def split_sql_by_semicolon(sql_statement):
    """
    Splits a SQL statement into multiple statements based on semicolons (;).
    The function removes standalone comments but keeps inline comments
    within SQL statements.

    :param str sql_statement: A string representing the SQL statement to be split.
    :return: A list of cleaned SQL statements.

    :Example:

    >>> split_sql_by_semicolon('SELECT * FROM table1; SELECT * FROM table2')
    ['SELECT * FROM table1;', 'SELECT * FROM table2']
    >>> split_sql_by_semicolon('-- Comment before\\nSELECT col1 /* Inline comment */ FROM table; -- Comment after')
    ['SELECT col1 /* Inline comment */ FROM table;']
    """
    sql_statement = sql_statement.replace("\r\n", "\n").replace("\r", "\n")
    left_brackets = {"}": "{", "]": "[", ")": "("}

    def cut_statement(stmt_start, stmt_end=None):
        stmt_end = stmt_end or len(sql_statement)
        stmt_writer = compat.StringIO()

        left = stmt_start
        has_statement = False
        for comm_start, comm_end in comment_blocks:
            if comm_end <= stmt_start:
                continue
            if comm_start > stmt_end:
                break

            segment = sql_statement[left:comm_start]
            stmt_writer.write(segment)
            if segment.strip():
                has_statement = True

            if has_statement:
                stmt_writer.write(sql_statement[comm_start:comm_end])
            left = comm_end

        stmt_writer.write(sql_statement[left:stmt_end])
        return "\n".join(
            line.rstrip() for line in stmt_writer.getvalue().splitlines()
        ).strip()

    start, pos = 0, 0
    statements = []
    comment_sign = None
    comment_pos = None
    comment_blocks = []
    quote_sign = None
    bracket_stack = []
    while pos < len(sql_statement):
        ch = sql_statement[pos]
        dch = sql_statement[pos : pos + 2] if pos + 1 < len(sql_statement) else None
        if quote_sign is None and comment_sign is None:
            if ch in ("{", "[", "("):
                # start of brackets
                bracket_stack.append(ch)
                pos += 1
            elif ch in ("}", "]", ")"):
                # end of brackets
                assert bracket_stack[-1] == left_brackets[ch]
                bracket_stack.pop()
                pos += 1
            elif ch in ('"', "'", "`"):
                # start of quote
                quote_sign = ch
                pos += 1
            elif dch in ("--", "/*"):
                # start of line or block comments
                comment_sign = dch
                comment_pos = pos
                pos += 2
            elif ch == ";" and not bracket_stack:
                # semicolon without brackets, quotes and comments
                part_statement = cut_statement(start, pos + 1)
                if part_statement and part_statement != ";":
                    statements.append(part_statement)
                pos += 1
                start = pos
            else:
                pos += 1
        elif quote_sign is not None and ch == quote_sign:
            quote_sign = None
            pos += 1
        elif quote_sign is not None and ch == "\\":
            # skip escape char
            pos += 2
        elif comment_sign == "--" and ch == "\n":
            # line comment ends
            comment_sign = None
            comment_blocks.append((comment_pos, pos))
            pos += 1
        elif comment_sign == "/*" and dch == "*/":
            # block comment ends
            comment_sign = None
            comment_blocks.append((comment_pos, pos + 2))
            pos += 2
        else:
            pos += 1

    if comment_sign == "--":
        # standalone comment without line ending
        comment_sign = None
        comment_blocks.append((comment_pos, pos))

    part_statement = cut_statement(start)
    if part_statement and part_statement != ";":
        statements.append(part_statement)
    return statements


def strip_backquotes(name_str):
    name_str = name_str.strip()
    if not name_str.startswith("`") or not name_str.endswith("`"):
        return name_str
    return name_str[1:-1].replace("``", "`")


def split_backquoted(s, sep=None, maxsplit=-1):
    if sep:
        if "`" in sep:
            raise ValueError("Separator cannot contain backquote")
        seps = (sep,)
    else:
        seps = (" ", "\t", "\n", "\r", "\f")

    results = []
    pos = 0
    cur_token_sio = compat.StringIO()
    quoted = False
    sep_len = len(sep) if sep else 1
    while pos < len(s):
        ch = s[pos]
        if ch == "`":
            quoted = not quoted
            cur_token_sio.write(ch)
            pos += 1
        elif (
            not quoted
            and (maxsplit < 0 or len(results) < maxsplit)
            and s[pos : pos + sep_len] in seps
        ):
            results.append(cur_token_sio.getvalue())
            cur_token_sio = compat.StringIO()
            pos += sep_len
        else:
            cur_token_sio.write(ch)
            pos += 1
    results.append(cur_token_sio.getvalue())
    return results


def backquote_string(s):
    return "`%s`" % s.replace("`", "``")


_convert_host_hash = (
    "6fe9b6c02efc24159c09863c1aadffb5",
    "9b75728355160c10b5eb75e4bf105a76",
)
_default_host_hash = "c7f4116fbf820f99284dcc89f340b372"


def get_default_logview_endpoint(default_endpoint, odps_endpoint):
    try:
        if odps_endpoint is None:
            return default_endpoint

        parsed_host = compat.urlparse(odps_endpoint).hostname
        if parsed_host is None:
            return default_endpoint

        default_host = compat.urlparse(default_endpoint).hostname
        hashed_host = md5_hexdigest(to_binary(parsed_host))
        hashed_default_host = md5_hexdigest(to_binary(default_host))

        if (
            hashed_default_host == _default_host_hash
            and hashed_host in _convert_host_hash
        ):
            suffix = r"\1" + string.ascii_letters[1::-1] * 2 + parsed_host[-8:]
            return re.sub(r"([a-z])[a-z]{3}\.[a-z]{3}$", suffix, default_endpoint)
        return default_endpoint
    except:
        return default_endpoint


_host_end_hashes = (
    "0a5ee4b73431f59ea08959e39fadd567",
    "a667fa51e0b92db2378e04ae0282e2d6",
)


def is_job_insight_released(odps_endpoint):
    try:
        if odps_endpoint is None:
            return False

        parsed_host = compat.urlparse(odps_endpoint).hostname
        if parsed_host is None:
            return False
        hashed_host = md5_hexdigest(to_binary(parsed_host))
        host_tail = ".".join(parsed_host.split(".")[-2:])
        if md5_hexdigest(host_tail) in _host_end_hashes:
            return hashed_host not in _convert_host_hash
        elif re.match(r"[^v]\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", parsed_host):
            return False
        else:
            return True
    except:
        return True


def _import_version_function():
    try:
        try:
            from importlib.metadata import version
        except ImportError:
            from importlib_metadata import version
        return version
    except ImportError:
        try:
            import pkg_resources
        except ImportError:
            return None
        return lambda x: pkg_resources.get_distribution(x).version


def get_package_version(pack_name, import_name=None):
    version_func = _import_version_function()
    if version_func is not None:
        try:
            return version_func(pack_name)
        except:
            pass
    import_name = pack_name if import_name is True else import_name
    if import_name:
        mod = __import__(import_name)
        return mod.__version__
    return None


def show_versions():  # pragma: no cover
    import locale
    import platform

    uname_result = platform.uname()
    language_code, encoding = locale.getlocale()

    results = {
        "python": ".".join([str(i) for i in sys.version_info]),
        "python-bits": struct.calcsize("P") * 8,
        "OS": uname_result.system,
        "OS-release": uname_result.release,
        "Version": uname_result.version,
        "machine": uname_result.machine,
        "processor": uname_result.processor,
        "byteorder": sys.byteorder,
        "LC_ALL": os.environ.get("LC_ALL"),
        "LANG": os.environ.get("LANG"),
        "LOCALE": {"language-code": language_code, "encoding": encoding},
    }

    try:
        from .src import crc32c_c  # noqa: F401

        results["USE_CLIB"] = True
    except ImportError:
        results["USE_CLIB"] = False

    try:
        from . import internal  # noqa: F401

        results["HAS_INTERNAL"] = True
    except ImportError:
        results["HAS_INTERNAL"] = False

    packages = {
        "pyodps": "odps",
        "requests": "requests",
        "urllib3": "urllib3",
        "charset_normalizer": "charset_normalizer",
        "chardet": "chardet",
        "idna": "idna",
        "certifi": "certifi",
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "sqlalchemy": "sqlalchemy",
        "pytz": "pytz",
        "python-dateutil": "dateutil",
        "IPython": "IPython",
        "maxframe": "maxframe",
    }
    for pack_name, pack_imp in packages.items():
        try:
            results[pack_name] = get_package_version(pack_name, pack_imp)
        except (ImportError, AttributeError):
            pass
    key_size = 1 + max(len(key) for key in results.keys())
    for key, val in results.items():
        print(key + " " * (key_size - len(key)) + ": " + str(val))


def get_supported_python_tag(align=None):
    if align is None:
        align = options.align_supported_python_tag
    if align:
        if sys.version_info[:2] >= (3, 11):
            return "cp311"
        elif sys.version_info[:2] >= (3, 6):
            return "cp37"
        else:
            return "cp27"
    else:
        return "cp" + str(sys.version_info[0]) + str(sys.version_info[1])


def chain_generator(*iterable):
    for itr in iterable:
        for item in itr:
            yield item
