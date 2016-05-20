#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from __future__ import absolute_import

import bisect
import codecs
import copy
import glob
import hmac
import inspect
import multiprocessing
import os
import re
import string
import struct
import sys
import time
import traceback
import types
import warnings
import xml.dom.minidom
from hashlib import sha1, md5
from base64 import b64encode
from datetime import datetime
from email.utils import parsedate_tz, formatdate

from . import compat
from .compat import six

TEMP_TABLE_PREFIX = 'tmp_pyodps_'


def deprecated(msg):
    def _decorator(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emmitted
        when the function is used."""
        def _new_func(*args, **kwargs):
            warn_msg = "Call to deprecated function %s." % func.__name__
            if isinstance(msg, six.string_types):
                warn_msg += ' ' + msg
            warnings.warn(msg, category=DeprecationWarning)
            return func(*args, **kwargs)
        _new_func.__name__ = func.__name__
        _new_func.__doc__ = func.__doc__
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
    return "".join([s[0].upper() + s[1:len(s)] for s in name.split('_')])


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


def to_timestamp(dt):
    return int(time.mktime(dt.timetuple()))


def to_milliseconds(dt):
    return int((time.mktime(dt.timetuple()) + dt.microsecond/1000000.0) * 1000)


def to_datetime(milliseconds):
    seconds = int(milliseconds / 1000)
    microseconds = milliseconds % 1000 * 1000
    return datetime.fromtimestamp(seconds).replace(microsecond=microseconds)


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


def is_lambda(f):
    lam = lambda: 0
    return isinstance(f, type(lam)) and f.__name__ == lam.__name__


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


def get_root_dir():
    return os.path.dirname(sys.modules[__name__].__file__)


def load_text_file(path):
    file_path = get_root_dir() + path
    if not os.path.exists(file_path):
        return None
    with codecs.open(file_path, encoding='utf-8') as f:
        inp_file = f.read()
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
    return content_dict


def load_static_text_file(path):
    return load_text_file('/static/' + path)


def load_internal_static_text_file(path):
    return load_text_file('/internal/static/' + path)


def load_static_text_files(pattern, func=None):
    return load_text_files('/static/' + pattern, func)


def init_progress_bar(val=1):
    try:
        from traitlets import TraitError
        ipython = True
    except ImportError:
        try:
            from IPython.utils.traitlets import TraitError
            ipython = True
        except ImportError:
            ipython = False

    from odps.console import ProgressBar

    if not ipython:
        bar = ProgressBar(val)
    else:
        try:
            bar = ProgressBar(val, True)
        except TraitError:
            bar = ProgressBar(val)

    return bar


def init_progress_ui(val=1):
    from odps.ui import ProgressGroupUI, html_notify

    bar = init_progress_bar(val=val)
    if bar._ipython_widget:
        try:
            progress_group = ProgressGroupUI(bar._ipython_widget)
        except:
            progress_group = None
    else:
        progress_group = None

    class ProgressUI(object):
        def update(self, value=None):
            bar.update(value=value)

        def status(self, text):
            if progress_group:
                progress_group.text = text

        def add_keys(self, keys):
            if progress_group:
                progress_group.add_keys(keys)

        def remove_keys(self, keys):
            if progress_group:
                progress_group.remove_keys(keys)

        def update_group(self):
            if progress_group:
                progress_group.update()

        def notify(self, msg):
            html_notify(msg)

        def close(self):
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
        else:
            return val

    return param_re.sub(replace, sql)


def is_main_process():
    return 'main' in multiprocessing.current_process().name.lower()


survey_calls = dict()


def survey(func):
    def _decorator(*args, **kwargs):
        arg_spec = inspect.getargspec(func)

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
    home_dir = os.environ.get('PYODPS_DIR') or os.path.join(os.path.expanduser('~'), '.pyodps')
    return os.path.join(home_dir, *args)
