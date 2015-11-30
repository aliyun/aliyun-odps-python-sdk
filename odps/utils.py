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


import sys
import hmac
from hashlib import sha1, md5
from base64 import b64encode
import struct
import string
from datetime import datetime
import re
import xml.dom.minidom
import traceback
import time
from email.utils import parsedate_tz, formatdate

import six

from . import compat


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
    return to_text(text) if six.PY3 else to_binary(text)