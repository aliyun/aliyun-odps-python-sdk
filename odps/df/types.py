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

from datetime import datetime as _datetime
from decimal import Decimal as _Decimal

import six

from ..types import DataType
from ..models import Schema  # noqa
from ..compat import OrderedDict


class Primitive(DataType):
    __slots__ = ()

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        return value


class Integer(Primitive):
    __slots__ = ()

    def can_implicit_cast(self, other):
        if isinstance(other, Integer) and other._n_bytes <= self._n_bytes:
            return True
        return False

    def validate_value(self, val):
        if val is None and self.nullable:
            return True
        return self._bounds[0] <= val <= self._bounds[1]


class Float(Primitive):
    __slots__ = ()

    def can_implicit_cast(self, other):
        if isinstance(other, (Integer, Float)):
            return True
        return False

    def cast_value(self, value, data_type):
        self._can_cast_or_throw(value, data_type)

        return float(value)


class Int8(Integer):
    __slots__ = ()

    _n_bytes = 1
    _bounds = (-128, 127)


class Int16(Integer):
    __slots__ = ()

    _n_bytes = 2
    _bounds = (-32768, 32767)


class Int32(Integer):
    __slots__ = ()

    _n_bytes = 4
    _bounds = (-2147483648, 2147483647)


class Int64(Integer):
    __slots__ = ()

    _n_bytes = 8
    _bounds = (-9223372036854775808, 9223372036854775807)


class Float32(Float):
    __slots__ = ()

    _n_bytes = 4


class Float64(Float):
    __slots__ = ()

    _n_bytes = 8


class Datetime(Primitive):
    __slots__ = ()


class Boolean(Primitive):
    __slots__ = ()


class Decimal(Primitive):
    __slots__ = ()

    def can_implicit_cast(self, other):
        if isinstance(other, Integer):
            return True
        return False


class String(Primitive):
    __slots__ = ()


int8 = Int8()
int16 = Int16()
int32 = Int32()
int64 = Int64()
float32 = Float32()
float64 = Float64()
boolean = Boolean()
string = String()
decimal = Decimal()
datetime = Datetime()


_data_types = dict(
    (t.name, t) for t in
    (int8, int16, int32, int64, float32, float64,
     boolean, string, decimal, datetime)
)


def validate_data_type(data_type):
    if isinstance(data_type, DataType):
        return data_type

    if isinstance(data_type, six.string_types):
        data_type = data_type.lower()
        if data_type == 'int':
            data_type = 'int64'
        elif data_type == 'float':
            data_type = 'float64'
        if data_type in _data_types:
            return _data_types[data_type]

    raise ValueError('Invalid data type: %s' % repr(data_type))


def validate_value_type(value, data_type=None):
    if data_type is not None:
        data_type.validate_value(value)
        return data_type

    inferred_value_type = None

    if isinstance(value, bool):
        inferred_value_type = boolean
    elif isinstance(value, six.integer_types):
        for t in (int8, int16, int32, int64):
            if t.validate_value(value):
                inferred_value_type = t
                break
        if inferred_value_type is None:
            raise ValueError('Integer value too large: %s' % value)
    elif isinstance(value, float):
        inferred_value_type = float64
    elif isinstance(value, six.string_types):
        inferred_value_type = string
    elif isinstance(value, _Decimal):
        inferred_value_type = decimal
    elif isinstance(value, _datetime):
        inferred_value_type = datetime
    else:
        raise ValueError('Unknown value: %s, type: %s' % (value, type(value)))

    return inferred_value_type


_number_types = OrderedDict.fromkeys([
    int8, int16, int32, int64, float32, float64, decimal])


def number_types():
    return _number_types.keys()


def is_number(data_type):
    if not isinstance(data_type, DataType):
        data_type = validate_data_type(data_type)

    if data_type in _number_types:
        return True

    return False