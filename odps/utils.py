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

from __future__ import absolute_import, print_function

import bisect
import calendar
import codecs
import copy
import glob
import hmac
import multiprocessing
import os
import re
import shutil
import string
import struct
import sys
import threading
import time
import traceback
import types
import warnings
import uuid
import xml.dom.minidom
import collections
from hashlib import sha1, md5
from base64 import b64encode
from datetime import datetime
from email.utils import parsedate_tz, formatdate

from . import compat, options
from .compat import six, getargspec, FixedOffset

try:
    import pytz
except ImportError:
    pytz = None

TEMP_TABLE_PREFIX = 'tmp_pyodps_'


def deprecated(msg, cond=None):
    def _decorator(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emmitted
        when the function is used."""
        def _new_func(*args, **kwargs):
            warn_msg = "Call to deprecated function %s." % func.__name__
            if isinstance(msg, six.string_types):
                warn_msg += ' ' + msg
            if cond is None or cond():
                warnings.warn(warn_msg, category=DeprecationWarning)
            return func(*args, **kwargs)
        _new_func.__name__ = func.__name__
        _new_func.__doc__ = func.__doc__
        _new_func.__dict__.update(func.__dict__)
        return _new_func

    if isinstance(msg, (types.FunctionType, types.MethodType)):
        return _decorator(msg)
    return _decorator


class ExperimentalNotAllowed(Exception):
    pass


def experimental(msg, cond=None):
    def _decorator(func):
        def _new_func(*args, **kwargs):
            warn_msg = "Call to experimental function %s." % func.__name__
            if isinstance(msg, six.string_types):
                warn_msg += ' ' + msg

            if not str_to_bool(os.environ.get('PYODPS_EXPERIMENTAL', 'true')):
                err_msg = "Calling to experimental method %s is denied." % func.__name__
                if isinstance(msg, six.string_types):
                    err_msg += ' ' + msg
                raise ExperimentalNotAllowed(err_msg)

            if cond is None or cond():
                warnings.warn(warn_msg, category=FutureWarning)
            return func(*args, **kwargs)
        _new_func.__name__ = func.__name__

        # intentionally eliminate __doc__ for Volume 2
        _new_func.__doc__ = None

        _new_func.__dict__.update(func.__dict__)
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
        if len(self.childNodes) == 1 \
          and self.childNodes[0].nodeType == xml.dom.minidom.Node.TEXT_NODE:
            writer.write(">")
            self.childNodes[0].writexml(writer, "", "", "")
            writer.write("</%s>%s" % (self.tagName, newl))
            return
        writer.write(">%s"%(newl))
        for node in self.childNodes:
            node.writexml(writer,indent+addindent,addindent,newl)
        writer.write("%s</%s>%s" % (indent,self.tagName,newl))
    else:
        writer.write("/>%s"%(newl))
# replace minidom's function with ours
xml.dom.minidom.Element.writexml = fixed_writexml
xml_fixed = lambda: None


def hmac_sha1(secret, data):
    return b64encode(hmac.new(secret, data, sha1).digest())


def md5_hexdigest(data):
    return md5(data).hexdigest()


def rshift(val, n):
    return val >> n if val >= 0  else \
        (val+0x100000000) >> n


def long_bits_to_double(bits):
    """
    @type  bits: long
    @param bits: the bit pattern in IEEE 754 layout

    @rtype:  float
    @return: the double-precision floating-point value corresponding
             to the given bit pattern C{bits}.
    """
    return struct.unpack('d', struct.pack('Q', bits))[0]


def double_to_raw_long_bits(value):
    """
    @type  value: float
    @param value: a Python (double-precision) float value

    @rtype: long
    @return: the IEEE 754 bit representation (64 bits as a long integer)
             of the given double-precision floating-point value.
    """
    # pack double into 64 bits, then unpack as long int
    return struct.unpack('Q', struct.pack('d', float(value)))[0]


timetuple_to_datetime = lambda t: datetime(*t[:6])


def camel_to_underline(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def underline_to_capitalized(name):
    return "".join([s[0].upper() + s[1:len(s)] for s in name.strip('_').split('_')])


def underline_to_camel(name):
    parts = name.split('_')
    return parts[0] + ''.join(v.title() for v in parts[1:])


def camel_to_underscore(chars):
    ret = []
    for c in chars:
        if c in string.uppercase:
            ret.append('_')
            ret.append(c.lower())
        else:
            ret.append(c)
    return ''.join(ret)


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
    return '\n'.join(lines)


def str_to_printable(field_name, auto_quote=True):
    if not field_name:
        return field_name

    escapes = {'\\': '\\\\', '\'': '\\\'', '"': '\\"', '\a': '\\a', '\b': '\\b', '\f': '\\f',
               '\n': '\\n', '\r': '\\r', '\t': '\\t', '\v': '\\v', ' ': ' '}

    def _escape_char(c):
        if c in escapes:
            return escapes[c]
        elif c < ' ':
            return '\\x%02x' % ord(c)
        else:
            return c

    need_escape = lambda c: c <= ' ' or c in escapes
    if any(need_escape(c) for c in field_name):
        ret = ''.join(_escape_char(ch) for ch in field_name)
        return '"' + ret + '"' if auto_quote else ret
    return field_name


def indent(text, n_spaces):
    if n_spaces <= 0:
        return text
    block = ' ' * n_spaces
    return '\n'.join((block + it) if len(it) > 0 else it
                     for it in text.split('\n'))


def parse_rfc822(s):
    return timetuple_to_datetime(parsedate_tz(s))


def gen_rfc822(dt=None, localtime=False, usegmt=False):
    if dt is not None:
        t = time.mktime(dt.timetuple())
    else:
        t = None
    return formatdate(t, localtime=localtime, usegmt=usegmt)


def to_timestamp(dt, local_tz=None, is_dst=False):
    return int(to_milliseconds(dt, local_tz=local_tz, is_dst=is_dst) / 1000.0)


def build_to_milliseconds(local_tz=None, is_dst=False):
    from . import options
    utc = compat.utc
    local_tz = local_tz if local_tz is not None else options.local_timezone
    if isinstance(local_tz, bool):

        try:
            if options.force_py:
                raise ImportError
            if local_tz:
                from .src.utils_c import datetime_to_local_milliseconds
                to_milliseconds = datetime_to_local_milliseconds
            else:
                from .src.utils_c import datetime_to_gmt_milliseconds
                to_milliseconds = datetime_to_gmt_milliseconds
        except ImportError:
            if options.force_c:
                raise

            if local_tz:
                _mktime = time.mktime
            else:
                _mktime = calendar.timegm

            def to_milliseconds(dt):
                if dt.tzinfo is not None:
                    return int((calendar.timegm(dt.astimezone(utc).timetuple()) + dt.microsecond / 1000000.0) * 1000)
                else:
                    return int((_mktime(dt.timetuple()) + dt.microsecond / 1000000.0) * 1000)

        return to_milliseconds
    else:
        if isinstance(local_tz, six.string_types):
            if pytz is None:
                raise RuntimeError('Package `pytz` is needed when specifying string-format time zone.')
            tz = pytz.timezone(local_tz)
        else:
            tz = local_tz

        if hasattr(tz, 'localize'):
            localize = lambda dt: tz.localize(dt, is_dst=is_dst)
        else:
            localize = lambda dt: dt.replace(tzinfo=tz)

        def to_milliseconds(dt):
            if dt.tzinfo is None:
                dt = localize(dt)
            return int((calendar.timegm(dt.astimezone(utc).timetuple()) + dt.microsecond / 1000000.0) * 1000)

        return to_milliseconds


def to_milliseconds(dt, local_tz=None, is_dst=False):
    f = build_to_milliseconds(local_tz, is_dst=is_dst)
    return f(dt)


def build_to_datetime(local_tz=None):
    utc = compat.utc
    long_type = compat.long_type
    local_tz = local_tz if local_tz is not None else options.local_timezone
    if isinstance(local_tz, bool):
        if local_tz:
            _fromtimestamp = datetime.fromtimestamp
        else:
            _fromtimestamp = datetime.utcfromtimestamp

        def to_datetime(milliseconds):
            seconds = long_type(milliseconds / 1000)
            microseconds = long_type(milliseconds) % 1000 * 1000
            return _fromtimestamp(seconds).replace(microsecond=microseconds)

        return to_datetime
    else:
        if isinstance(local_tz, six.string_types):
            if pytz is None:
                raise RuntimeError('Package `pytz` is needed when specifying string-format time zone.')
            tz = pytz.timezone(local_tz)
        else:
            tz = local_tz

        def to_datetime(milliseconds):
            seconds = int(milliseconds / 1000)
            microseconds = milliseconds % 1000 * 1000
            return datetime.utcfromtimestamp(seconds).replace(microsecond=microseconds, tzinfo=utc).astimezone(tz)

        return to_datetime


def to_datetime(milliseconds, local_tz=None):
    f = build_to_datetime(local_tz)
    return f(milliseconds)


def strptime_with_tz(dt, format='%Y-%m-%d %H:%M:%S'):
    try:
        return datetime.strptime(dt, format)
    except ValueError:
        naive_date_str, _, offset_str = dt.rpartition(' ')
        naive_dt = datetime.strptime(naive_date_str, format)
        offset = int(offset_str[-4:-2]) * 60 + int(offset_str[-2:])
        if offset_str[0] == "-":
            offset = -offset
        return naive_dt.replace(tzinfo=FixedOffset(offset))


def to_binary(text, encoding='utf-8'):
    if text is None:
        return text
    if isinstance(text, six.text_type):
        return text.encode(encoding)
    elif isinstance(text, (six.binary_type, bytearray)):
        return bytes(text)
    else:
        return str(text).encode(encoding) if six.PY3 else str(text)


def to_text(binary, encoding='utf-8'):
    if binary is None:
        return binary
    if isinstance(binary, (six.binary_type, bytearray)):
        return binary.decode(encoding)
    elif isinstance(binary, six.text_type):
        return binary
    else:
        return str(binary) if six.PY3 else str(binary).decode(encoding)


def to_str(text, encoding='utf-8'):
    return to_text(text, encoding=encoding) if six.PY3 else to_binary(text, encoding=encoding)


# fix encoding conversion problem under windows
if sys.platform == 'win32':
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
    for pair in string.split(','):
        k, v = pair.split(':', 1)
        if typ:
            v = typ(v)
        d[k] = v
    return d


def interval_select(val, intervals, targets):
    return targets[bisect.bisect_left(intervals, val)]


def is_namedtuple(obj):
    return isinstance(obj, tuple) and hasattr(obj, '_fields')


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    s = s.lower().strip()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError


def bool_to_str(s):
    return str(s).lower()


def get_root_dir():
    return os.path.dirname(sys.modules[__name__].__file__)


def load_text_file(path):
    file_path = get_root_dir() + path
    if not os.path.exists(file_path):
        return None
    with codecs.open(file_path, encoding='utf-8') as f:
        inp_file = f.read()
        f.close()
    return inp_file


def load_file_paths(pattern):
    file_path = os.path.normpath(os.path.dirname(sys.modules[__name__].__file__) + pattern)
    return glob.glob(file_path)


def load_static_file_paths(path):
    return load_file_paths('/static/' + path)


def load_text_files(pattern, func=None):
    file_path = os.path.normpath(os.path.dirname(sys.modules[__name__].__file__) + pattern)
    content_dict = dict()
    for file_path in glob.glob(file_path):
        _, fn = os.path.split(file_path)
        if func and not func(fn):
            continue
        with codecs.open(file_path, encoding='utf-8') as f:
            content_dict[fn] = f.read()
            f.close()
    return content_dict


def load_static_text_file(path):
    return load_text_file('/static/' + path)


def load_internal_static_text_file(path):
    return load_text_file('/internal/static/' + path)


def load_static_text_files(pattern, func=None):
    return load_text_files('/static/' + pattern, func)


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
            if bar and hasattr(bar, '_current_value'):
                return bar._current_value

        @ui_method
        def inc(self, value):
            if bar and hasattr(bar, '_current_value'):
                current_val = bar._current_value
                bar.update(current_val + value)

        @ui_method
        def status(self, prefix, suffix='', clear_keys=False):
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
    trans_dict = {'\b': r'\b', '\t': r'\t', '\n': r'\n', '\r': r'\r', '\'': r'\'', '\"': r'\"', '\\': r'\\',
                  '\;': r'\;', '\Z': r'\Z', '\0': r'\0'}
    return ''.join(trans_dict[ch] if ch in trans_dict else ch for ch in src)


def replace_sql_parameters(sql, ns):
    param_re = re.compile(r':([a-zA-Z_][a-zA-Z0-9_]*)')

    def replace(matched):
        name = matched.group(1)
        val = ns.get(name)
        if val is None:
            return matched.group(0)
        elif isinstance(val, (six.integer_types, float)):
            return repr(val)
        else:
            return "'{0}'".format(escape_odps_string(str(val)))

    return param_re.sub(replace, sql)


def is_main_process():
    return 'main' in multiprocessing.current_process().name.lower()


survey_calls = dict()


def survey(func):
    def _decorator(*args, **kwargs):
        arg_spec = getargspec(func)

        if 'self' in arg_spec.args:
            func_cls = args[0].__class__
        else:
            func_cls = None

        if func_cls:
            func_sig = '.'.join([func_cls.__module__, func_cls.__name__, func.__name__])
        else:
            func_sig = '.'.join([func.__module__, func.__name__])

        if func_sig not in survey_calls:
            survey_calls[func_sig] = 1
        else:
            survey_calls[func_sig] += 1

        return func(*args, **kwargs)

    _decorator.__name__ = func.__name__
    _decorator.__doc__ = func.__doc__
    return _decorator


def get_survey_calls():
    return copy.copy(survey_calls)


def clear_survey_calls():
    survey_calls.clear()


def require_package(pack_name):
    def _decorator(func):
        try:
            __import__(pack_name, fromlist=[''])
            return func
        except ImportError:
            return None

    return _decorator


def gen_repr_object(**kwargs):
    obj = type('ReprObject', (), {})
    text = kwargs.pop('text', None)
    if six.PY2 and isinstance(text, unicode):
        text = text.encode('utf-8')
    if text:
        setattr(obj, 'text', text)
        setattr(obj, '__repr__', lambda self: text)
    for k, v in six.iteritems(kwargs):
        setattr(obj, k, v)
        setattr(obj, '_repr_{0}_'.format(k), lambda self: v)
    if 'gv' in kwargs:
        try:
            from graphviz import Source
            setattr(obj, '_repr_svg_', lambda self: Source(self._repr_gv_(), encoding='utf-8')._repr_svg_())
        except ImportError:
            pass
    return obj()


def build_pyodps_dir(*args):
    default_dir = os.path.join(os.path.expanduser('~'), '.pyodps')
    if sys.platform == 'win32' and 'APPDATA' in os.environ:
        win_default_dir = os.path.join(os.environ['APPDATA'], 'pyodps')
        if os.path.exists(default_dir):
            shutil.move(default_dir, win_default_dir)
        default_dir = win_default_dir
    home_dir = os.environ.get('PYODPS_DIR') or default_dir
    return os.path.join(home_dir, *args)


def object_getattr(obj, attr, default=None):
    try:
        return object.__getattribute__(obj, attr)
    except AttributeError as e:
        return default


def attach_internal(cls):
    cls_path = cls.__module__ + '.' + cls.__name__
    try:
        from .internal.core import MIXIN_TARGETS
        mixin_cls = MIXIN_TARGETS[cls_path]
        for method_name in dir(mixin_cls):
            if method_name.startswith('_'):
                continue
            att = getattr(mixin_cls, method_name)
            if six.PY2 and type(att).__name__ in ('instancemethod', 'method'):
                att = att.__func__
            setattr(cls, method_name, att)
        return cls
    except ImportError:
        return cls


_main_thread_local = threading.local()
_main_thread_local.is_main = True


def is_main_thread():
    return hasattr(_main_thread_local, 'is_main') and _main_thread_local.is_main


def write_log(msg):
    from . import options
    if options.verbose:
        (options.verbose_log or print)(msg)


def split_quoted(s, delimiter=',', maxsplit=0):
    pattern = r"""((?:[^""" + delimiter + r""""']|"[^"]*"|'[^']*')+)"""
    return re.split(pattern, s, maxsplit=maxsplit)[1::2]


def gen_temp_table():
    return '%s%s' % (TEMP_TABLE_PREFIX, str(uuid.uuid4()).replace('-', '_'))


def hashable(obj):
    if isinstance(obj, collections.Hashable):
        items = obj
    elif isinstance(obj, collections.Mapping):
        items = type(obj)((k, hashable(v)) for k, v in six.iteritems(obj))
    elif isinstance(obj, collections.Iterable):
        items = tuple(hashable(item) for item in obj)
    else:
        raise TypeError(type(obj))

    return items
