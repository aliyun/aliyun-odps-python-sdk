#!/usr/bin/env python
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

import copy
import decimal as _decimal
from collections import OrderedDict  # noqa: F401

import pytest

try:
    import pandas as pd
except ImportError:
    pd = None

from ..types import *
from ..tests.core import py_and_c

from datetime import datetime


def _reloader():
    global TableSchema, Record

    from odps import models

    TableSchema, Record = models.TableSchema, models.Record


py_and_c_deco = py_and_c(["odps.models.record", "odps.models"], reloader=_reloader)


@py_and_c_deco
def test_nullable_record():
    col_types = ['tinyint', 'smallint', 'int', 'bigint', 'float', 'double',
         'string', 'datetime', 'boolean', 'decimal', 'binary', 'decimal(10, 2)',
         'interval_year_month', 'json', 'char(20)', 'varchar(20)',
         'array<string>', 'map<string,bigint>', 'struct<a:string,b:array<int>>']
    if pd is not None:
        col_types.extend(['interval_day_time', 'timestamp', 'timestamp_ntz'])

    s = TableSchema.from_lists(
        ['col%s' % i for i in range(len(col_types))],
        col_types,
    )
    r = Record(schema=s, values=[None] * len(col_types))
    assert list(r.values) == [None] * len(col_types)


@py_and_c_deco
def test_record_max_field_size():
    s = TableSchema.from_lists(["col"], ["string"])
    r = Record(schema=s, max_field_size=1024)

    r["col"] = "e" * 1024
    with pytest.raises(ValueError):
        r["col"] = "e" * 1025

    r = Record(schema=s)
    r["col"] = "e" * String._max_length
    with pytest.raises(ValueError):
        r["col"] = "e" * (String._max_length + 1)


@py_and_c_deco
def test_record_set_and_get_by_index():
    s = TableSchema.from_lists(
        ['col%s' % i for i in range(9)],
        ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal', 'json',
         'array<string>', 'map<string,bigint>'])
    s.build_snapshot()
    if options.force_py:
        assert s._snapshot is None
    else:
        assert s._snapshot is not None

    r = Record(schema=s)
    r[0] = 1
    r[1] = 1.2
    r[2] = 'abc'
    r[3] = datetime(2016, 1, 1)
    r[4] = True
    r[5] = _decimal.Decimal('1.111')
    r[6] = {"root": {"key": "value"}}
    r[7] = ['a', 'b']
    r[8] = OrderedDict({'a': 1})
    assert list(r.values) == [
        1, 1.2, 'abc', datetime(2016, 1, 1), True, _decimal.Decimal('1.111'),
        {"root": {"key": "value"}}, ['a', 'b'], OrderedDict({'a': 1}),
    ]
    assert 1 == r[0]
    assert 1.2 == r[1]
    assert 'abc' == r[2]
    assert datetime(2016, 1, 1) == r[3]
    assert r[4] is True
    assert _decimal.Decimal('1.111') == r[5]
    assert {"root": {"key": "value"}} == r[6]
    assert ['a', 'b'] == r[7]
    assert OrderedDict({'a': 1}) == r[8]
    assert [1, 1.2] == r[:2]


@py_and_c_deco
def test_record_set_and_get_by_name():
    s = TableSchema.from_lists(
        ['col%s' % i for i in range(9)],
        ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal', 'json',
         'array<string>', 'map<string,bigint>'])
    r = Record(schema=s)
    r['col0'] = 1
    r['col1'] = 1.2
    r['col2'] = 'abc'
    r['col3'] = datetime(2016, 1, 1)
    r['col4'] = True
    r['col5'] = _decimal.Decimal('1.111')
    r['col6'] = {"root": {"key": "value"}}
    r['col7'] = ['a', 'b']
    r['col8'] = OrderedDict({'a': 1})
    assert list(r.values) == [
        1, 1.2, 'abc', datetime(2016, 1, 1), True, _decimal.Decimal('1.111'),
        {"root": {"key": "value"}}, ['a', 'b'], OrderedDict({'a': 1})
    ]
    assert 1 == r['col0']
    assert 1.2 == r['col1']
    assert 'abc' == r['col2']
    assert datetime(2016, 1, 1) == r['col3']
    assert r['col4'] is True
    assert _decimal.Decimal('1.111') == r['col5']
    assert {"root": {"key": "value"}} == r['col6']
    assert ['a', 'b'] == r['col7']
    assert OrderedDict({'a': 1}) == r['col8']


def test_implicit_cast():
    tinyint = Tinyint()
    smallint = Smallint()
    int_ = Int()
    bigint = Bigint()
    float = Float()
    double = Double()
    datetime = Datetime()
    bool = Boolean()
    decimal = Decimal()
    string = String()
    json = Json()

    assert double.can_implicit_cast(bigint)
    assert string.can_implicit_cast(bigint)
    assert decimal.can_implicit_cast(bigint)
    assert not bool.can_implicit_cast(bigint)
    assert not datetime.can_implicit_cast(bigint)

    assert bigint.can_implicit_cast(double)
    assert string.can_implicit_cast(double)
    assert decimal.can_implicit_cast(double)
    assert not bool.can_implicit_cast(double)
    assert not datetime.can_implicit_cast(double)

    assert smallint.can_implicit_cast(tinyint)
    assert int_.can_implicit_cast(tinyint)
    assert bigint.can_implicit_cast(tinyint)
    assert int_.can_implicit_cast(smallint)
    assert bigint.can_implicit_cast(smallint)
    assert bigint.can_implicit_cast(int_)

    assert not tinyint.can_implicit_cast(smallint)
    assert not tinyint.can_implicit_cast(int_)
    assert not tinyint.can_implicit_cast(bigint)
    assert not smallint.can_implicit_cast(int_)
    assert not smallint.can_implicit_cast(bigint)
    assert not int_.can_implicit_cast(bigint)

    assert double.can_implicit_cast(float)
    assert not float.can_implicit_cast(double)

    assert json.can_implicit_cast(string)
    assert string.can_implicit_cast(json)


def test_composite_types():
    comp_type = validate_data_type('decimal')
    assert isinstance(comp_type, Decimal)

    comp_type = validate_data_type('decimal(10)')
    assert isinstance(comp_type, Decimal)
    assert comp_type.precision == 10

    comp_type = validate_data_type('decimal(10, 2)')
    assert isinstance(comp_type, Decimal)
    assert comp_type.precision == 10
    assert comp_type.scale == 2

    comp_type = validate_data_type('varchar(10)')
    assert isinstance(comp_type, Varchar)
    assert comp_type.size_limit == 10

    comp_type = validate_data_type('char(20)')
    assert isinstance(comp_type, Char)
    assert comp_type.size_limit == 20

    with pytest.raises(ValueError) as ex_info:
        validate_data_type('array')
    assert 'ARRAY' in str(ex_info.value)

    comp_type = validate_data_type('array<bigint>')
    assert isinstance(comp_type, Array)
    assert isinstance(comp_type.value_type, Bigint)

    with pytest.raises(ValueError) as ex_info:
        validate_data_type('map')
    assert 'MAP' in str(ex_info.value)

    comp_type = validate_data_type('map<bigint, string>')
    assert isinstance(comp_type, Map)
    assert isinstance(comp_type.key_type, Bigint)
    assert isinstance(comp_type.value_type, String)

    comp_type = validate_data_type('struct<abc:int, def:string>')
    assert isinstance(comp_type, Struct)
    assert len(comp_type.field_types) == 2
    assert isinstance(comp_type.field_types['abc'], Int)
    assert isinstance(comp_type.field_types['def'], String)

    comp_type = validate_data_type('struct<abc:int, def:map<bigint, string>, ghi:string>')
    assert isinstance(comp_type, Struct)
    assert len(comp_type.field_types) == 3
    assert isinstance(comp_type.field_types['abc'], Int)
    assert isinstance(comp_type.field_types['def'], Map)
    assert isinstance(comp_type.field_types['def'].key_type, Bigint)
    assert isinstance(comp_type.field_types['def'].value_type, String)
    assert isinstance(comp_type.field_types['ghi'], String)


@py_and_c_deco
def test_set_with_cast():
    s = TableSchema.from_lists(
        ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal'],
        ['bigint', 'double', 'string', 'datetime', 'boolean', 'decimal'])
    r = Record(schema=s)
    r['double'] = 1
    assert 1.0 == r['double']
    r['double'] = '1.33'
    assert 1.33 == r['double']
    r['bigint'] = 1.1
    assert 1 == r['bigint']
    r['datetime'] = '2016-01-01 0:0:0'
    assert datetime(2016, 1, 1) == r['datetime']


@py_and_c_deco
def test_record_copy():
    s = TableSchema.from_lists(['col1'], ['string'])
    r = Record(schema=s)
    r.col1 = 'a'

    cr = copy.copy(r)
    assert cr.col1 == r.col1


@py_and_c_deco
def test_record_set_field():
    s = TableSchema.from_lists(['col1'], ['string'])
    r = Record(schema=s)
    r.col1 = 'a'
    assert r.col1 == 'a'

    r['col1'] = 'b'
    assert r['col1'] == 'b'

    r[0] = 'c'
    assert r[0] == 'c'
    assert r['col1'] == 'c'


@py_and_c_deco
def test_duplicate_names():
    pytest.raises(ValueError, lambda: TableSchema.from_lists(['col1', 'col1'], ['string', 'string']))
    try:
        TableSchema.from_lists(['col1', 'col1'], ['string', 'string'])
    except ValueError as e:
        assert 'col1' in str(e)


@py_and_c_deco
def test_chinese_schema():
    s = TableSchema.from_lists([u'用户'], ['string'], ['分区'], ['bigint'])
    assert '用户' in s
    assert s.get_column('用户').type.name == 'string'
    assert s.get_partition(u'分区').type.name == 'bigint'
    assert s['用户'].type.name == 'string'
    assert s[u'分区'].type.name == 'bigint'

    s2 = TableSchema.from_lists(['用户'], ['string'], [u'分区'], ['bigint'])
    assert s == s2


@py_and_c_deco
def test_record_multi_fields():
    s = TableSchema.from_lists(['col1', 'col2'], ['string', 'bigint'])
    r = Record(values=[1, 2], schema=s)

    assert r['col1', 'col2'] == ['1', 2]

    pytest.raises(KeyError, lambda: r['col3'])
    pytest.raises(KeyError, lambda: r['col3', ])


@py_and_c_deco
def test_bizarre_repr():
    s = TableSchema.from_lists(['逗比 " \t'], ['string'], ['正常'], ['bigint'])
    s_repr = repr(s)
    assert '"逗比 \\" \\t"' in s_repr
    assert '"正常"' not in s_repr


@py_and_c_deco
def test_string_as_binary():
    try:
        options.tunnel.string_as_binary = True
        s = TableSchema.from_lists(['col1', 'col2'], ['string', 'bigint'])
        r = Record(values=[1, 2], schema=s)
        assert r['col1', 'col2'] == [b'1', 2]
        assert isinstance(r[0], bytes)

        r[0] = u'junk'
        assert r[0] == b'junk'
        assert isinstance(r[0], bytes)

        r[0] = b'junk'
        assert r[0] == b'junk'
        assert isinstance(r[0], bytes)
    finally:
        options.tunnel.string_as_binary = False


def test_validate_struct():
    try:
        options.struct_as_dict = True
        struct_type = validate_data_type('struct<abc: int, def: string>')
        assert validate_value(None, struct_type) is None

        vl = validate_value((10, "uvwxyz"), struct_type)
        assert isinstance(vl, dict)
        assert vl["abc"] == 10
        assert vl["def"] == "uvwxyz"

        vl = validate_value({"abc": 10, "def": "uvwxyz"}, struct_type)
        assert isinstance(vl, dict)
        assert vl["abc"] == 10
        assert vl["def"] == "uvwxyz"

        with pytest.raises(ValueError):
            validate_value({"abcd", "efgh"}, struct_type)

        options.struct_as_dict = False
        struct_type = validate_data_type('struct<abc: int, def: string>')
        vl = validate_value((10, "uvwxyz"), struct_type)
        assert isinstance(vl, tuple)
        assert vl == (10, "uvwxyz")

        vl = validate_value({"def": "uvwxyz", "abc": 10}, struct_type)
        assert isinstance(vl, tuple)
        assert vl == (10, "uvwxyz")

        with pytest.raises(ValueError):
            validate_value({"abcd", "efgh"}, struct_type)
    finally:
        options.struct_as_dict = False
