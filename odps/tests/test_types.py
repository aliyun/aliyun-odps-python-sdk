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

import copy
import datetime
import decimal as _decimal
from collections import OrderedDict  # noqa: F401

import pytest

try:
    import pandas as pd
except ImportError:
    pd = None

from .. import options
from .. import types as odps_types
from .. import utils
from ..tests.core import pandas_case, py_and_c


def _reloader():
    global TableSchema, Record

    from odps import models

    TableSchema, Record = models.TableSchema, models.Record


py_and_c_deco = py_and_c(
    ["odps.models.record", "odps.models", "odps.utils"], reloader=_reloader
)


@py_and_c_deco
def test_nullable_record():
    col_types = [
        "tinyint",
        "smallint",
        "int",
        "bigint",
        "float",
        "double",
        "string",
        "datetime",
        "boolean",
        "decimal",
        "binary",
        "decimal(10, 2)",
        "date",
        "interval_year_month",
        "json",
        "char(20)",
        "varchar(20)",
        "array<string>",
        "map<string,bigint>",
        "struct<a:string,b:array<int>>",
    ]
    if pd is not None:
        col_types.extend(["interval_day_time", "timestamp", "timestamp_ntz"])

    s = TableSchema.from_lists(
        ["col%s" % i for i in range(len(col_types))],
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
    r["col"] = "e" * odps_types.String._max_length
    with pytest.raises(ValueError):
        r["col"] = "e" * (odps_types.String._max_length + 1)

    r = Record(schema=s, max_field_size=0)
    r["col"] = "e" * odps_types.String._max_length
    with pytest.raises(ValueError):
        r["col"] = "e" * (odps_types.String._max_length + 1)


@py_and_c_deco
def test_record_set_and_get_by_index():
    s = TableSchema.from_lists(
        ["col%s" % i for i in range(10)],
        [
            "bigint",
            "double",
            "string",
            "datetime",
            "boolean",
            "decimal",
            "json",
            "date",
            "array<string>",
            "map<string,bigint>",
        ],
    )
    s.build_snapshot()
    if options.force_py:
        assert s._snapshot is None
    else:
        assert s._snapshot is not None

    r = Record(schema=s)
    r[0] = 1
    r[1] = 1.2
    r[2] = "abc"
    r[3] = datetime.datetime(2016, 1, 1, 12, 30, 11)
    r[4] = True
    r[5] = _decimal.Decimal("1.111")
    r[6] = {"root": {"key": "value"}}
    r[7] = datetime.date(2016, 1, 1)
    r[8] = ["a", "b"]
    r[9] = OrderedDict({"a": 1})
    assert list(r.values) == [
        1,
        1.2,
        "abc",
        datetime.datetime(2016, 1, 1, 12, 30, 11),
        True,
        _decimal.Decimal("1.111"),
        {"root": {"key": "value"}},
        datetime.date(2016, 1, 1),
        ["a", "b"],
        OrderedDict({"a": 1}),
    ]
    assert 1 == r[0]
    assert 1.2 == r[1]
    assert "abc" == r[2]
    assert datetime.datetime(2016, 1, 1, 12, 30, 11) == r[3]
    assert r[4] is True
    assert _decimal.Decimal("1.111") == r[5]
    assert {"root": {"key": "value"}} == r[6]
    assert datetime.date(2016, 1, 1) == r[7]
    assert ["a", "b"] == r[8]
    assert OrderedDict({"a": 1}) == r[9]
    assert [1, 1.2] == r[:2]


@py_and_c_deco
def test_record_set_and_get_by_name():
    s = TableSchema.from_lists(
        ["col%s" % i for i in range(9)],
        [
            "bigint",
            "double",
            "string",
            "datetime",
            "boolean",
            "decimal",
            "json",
            "array<string>",
            "map<string,bigint>",
        ],
    )
    r = Record(schema=s)
    r["col0"] = 1
    r["col1"] = 1.2
    r["col2"] = "abc"
    r["col3"] = datetime.datetime(2016, 1, 1)
    r["col4"] = True
    r["col5"] = _decimal.Decimal("1.111")
    r["col6"] = {"root": {"key": "value"}}
    r["col7"] = ["a", "b"]
    r["col8"] = OrderedDict({"a": 1})
    assert list(r.values) == [
        1,
        1.2,
        "abc",
        datetime.datetime(2016, 1, 1),
        True,
        _decimal.Decimal("1.111"),
        {"root": {"key": "value"}},
        ["a", "b"],
        OrderedDict({"a": 1}),
    ]
    assert 1 == r["col0"]
    assert 1.2 == r["col1"]
    assert "abc" == r["col2"]
    assert datetime.datetime(2016, 1, 1) == r["col3"]
    assert r["col4"] is True
    assert _decimal.Decimal("1.111") == r["col5"]
    assert {"root": {"key": "value"}} == r["col6"]
    assert ["a", "b"] == r["col7"]
    assert OrderedDict({"a": 1}) == r["col8"]


@py_and_c_deco
def test_record_set_and_get_ignore_cases():
    s = TableSchema.from_lists(["COL1", "col2"], ["bigint", "double"])
    r = Record(schema=s)
    r["col1"] = 1
    r["COL2"] = 2.0
    assert 1 == r["Col1"]
    assert 2.0 == r["Col2"]


def test_implicit_cast():
    tinyint = odps_types.Tinyint()
    smallint = odps_types.Smallint()
    int_ = odps_types.Int()
    bigint = odps_types.Bigint()
    float = odps_types.Float()
    double = odps_types.Double()
    datetime_ = odps_types.Datetime()
    bool = odps_types.Boolean()
    decimal = odps_types.Decimal()
    date = odps_types.Date()
    string = odps_types.String()
    json = odps_types.Json()

    assert double.can_implicit_cast(bigint)
    assert string.can_implicit_cast(bigint)
    assert decimal.can_implicit_cast(bigint)
    assert not bool.can_implicit_cast(bigint)
    assert not datetime_.can_implicit_cast(bigint)

    assert bigint.can_implicit_cast(double)
    assert string.can_implicit_cast(double)
    assert decimal.can_implicit_cast(double)
    assert not bool.can_implicit_cast(double)
    assert not datetime_.can_implicit_cast(double)

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

    assert date.can_implicit_cast(datetime_)
    assert date.can_implicit_cast(string)
    assert not date.can_implicit_cast(bigint)

    assert json.can_implicit_cast(string)
    assert string.can_implicit_cast(json)


def test_composite_types():
    comp_type = odps_types.validate_data_type("decimal")
    assert isinstance(comp_type, odps_types.Decimal)

    comp_type = odps_types.validate_data_type("decimal(10)")
    assert isinstance(comp_type, odps_types.Decimal)
    assert comp_type.precision == 10
    assert comp_type == "decimal(10)"

    comp_type = odps_types.validate_data_type("decimal(10, 2)")
    assert isinstance(comp_type, odps_types.Decimal)
    assert comp_type.precision == 10
    assert comp_type.scale == 2
    assert comp_type == "decimal(10,2)"

    comp_type = odps_types.validate_data_type("varchar(10)")
    assert isinstance(comp_type, odps_types.Varchar)
    assert comp_type.size_limit == 10
    assert comp_type == "varchar(10)"

    comp_type = odps_types.validate_data_type("char(20)")
    assert isinstance(comp_type, odps_types.Char)
    assert comp_type.size_limit == 20
    assert comp_type == "char(20)"

    with pytest.raises(ValueError) as ex_info:
        odps_types.validate_data_type("array")
    assert "ARRAY" in str(ex_info.value)

    comp_type = odps_types.validate_data_type("array<bigint>")
    assert isinstance(comp_type, odps_types.Array)
    assert isinstance(comp_type.value_type, odps_types.Bigint)
    assert comp_type == "array<bigint>"

    with pytest.raises(ValueError) as ex_info:
        odps_types.validate_data_type("map")
    assert "MAP" in str(ex_info.value)

    comp_type = odps_types.validate_data_type("map<bigint, string>")
    assert isinstance(comp_type, odps_types.Map)
    assert isinstance(comp_type.key_type, odps_types.Bigint)
    assert isinstance(comp_type.value_type, odps_types.String)
    assert comp_type == "map<bigint, string>"

    comp_type = odps_types.validate_data_type("struct<abc:int, def:string>")
    assert isinstance(comp_type, odps_types.Struct)
    assert len(comp_type.field_types) == 2
    assert isinstance(comp_type.field_types["abc"], odps_types.Int)
    assert isinstance(comp_type.field_types["def"], odps_types.String)
    assert comp_type == "struct<abc:int, def:string>"
    assert comp_type != "struct<abc:int>"
    assert comp_type != "struct<abc:int, uvw:string>"

    comp_type = odps_types.validate_data_type(
        "struct<abc:int, def:map<bigint, string>, ghi:string>"
    )
    assert isinstance(comp_type, odps_types.Struct)
    assert len(comp_type.field_types) == 3
    assert isinstance(comp_type.field_types["abc"], odps_types.Int)
    assert isinstance(comp_type.field_types["def"], odps_types.Map)
    assert isinstance(comp_type.field_types["def"].key_type, odps_types.Bigint)
    assert isinstance(comp_type.field_types["def"].value_type, odps_types.String)
    assert isinstance(comp_type.field_types["ghi"], odps_types.String)


@py_and_c_deco
def test_set_with_cast():
    s = TableSchema.from_lists(
        ["bigint", "double", "string", "datetime", "date", "boolean", "decimal"],
        ["bigint", "double", "string", "datetime", "date", "boolean", "decimal"],
    )
    r = Record(schema=s)
    r["double"] = 1
    assert 1.0 == r["double"]
    r["double"] = "1.33"
    assert 1.33 == r["double"]
    r["bigint"] = 1.1
    assert 1 == r["bigint"]
    r["datetime"] = "2016-01-01 0:0:0"
    assert datetime.datetime(2016, 1, 1) == r["datetime"]
    r["date"] = "2016-01-01"
    assert datetime.date(2016, 1, 1) == r["date"]
    r["decimal"] = "13.5641"
    assert _decimal.Decimal("13.5641") == r["decimal"]


@py_and_c_deco
def test_record_copy():
    s = TableSchema.from_lists(["col1"], ["string"])
    r = Record(s)
    r.col1 = "a"

    cr = copy.copy(r)
    assert cr == r
    assert cr.col1 == r.col1

    cr = copy.deepcopy(r)
    assert cr == r
    assert cr.col1 == r.col1


@py_and_c_deco
def test_record_set_field():
    s = TableSchema.from_lists(["col1"], ["string"])
    r = Record(schema=s)
    r.col1 = "a"
    assert r.col1 == "a"

    r["col1"] = "b"
    assert r["col1"] == "b"

    r[0] = "c"
    assert r[0] == "c"
    assert r["col1"] == "c"


@py_and_c_deco
def test_record_multi_fields():
    s = TableSchema.from_lists(["col1", "col2"], ["string", "bigint"])
    r = Record(values=[1, 2], schema=s)

    assert r["col1", "col2"] == ["1", 2]

    pytest.raises(KeyError, lambda: r["col3"])
    pytest.raises(KeyError, lambda: r["col3"])


@py_and_c_deco
def test_duplicate_names():
    with pytest.raises(ValueError):
        TableSchema.from_lists(["col1", "col1"], ["string", "string"])
    with pytest.raises(ValueError):
        TableSchema.from_lists(["COL1", "col1"], ["string", "string"])

    try:
        TableSchema.from_lists(["col1", "col1"], ["string", "string"])
    except ValueError as e:
        assert "col1" in str(e)


@py_and_c_deco
def test_schema_cases():
    schema = TableSchema.from_lists(
        ["col1", "COL2"], ["bigint", "double"], ["pt1", "PT2"], ["string", "string"]
    )
    assert schema.get_column("COL1").name == "col1"
    assert schema.get_column("col2").name == "COL2"
    assert schema.get_partition("PT1").name == "pt1"
    assert schema.get_partition("pt2").name == "PT2"
    assert schema.get_type("COL1").name == "bigint"
    assert schema.get_type("col2").name == "double"
    assert schema.get_type("PT1").name == "string"
    assert schema.get_type("pt2").name == "string"
    assert not schema.is_partition("COL1")
    assert not schema.is_partition("col2")
    assert schema.is_partition("PT1")
    assert schema.is_partition("pt2")


@py_and_c_deco
def test_chinese_schema():
    s = TableSchema.from_lists([u"用户"], ["string"], ["分区"], ["bigint"])
    assert "用户" in s
    assert s.get_column("用户").type.name == "string"
    assert s.get_partition(u"分区").type.name == "bigint"
    assert s["用户"].type.name == "string"
    assert s[u"分区"].type.name == "bigint"

    s2 = TableSchema.from_lists(["用户"], ["string"], [u"分区"], ["bigint"])
    assert s == s2


@py_and_c_deco
def test_bizarre_repr():
    s = TableSchema.from_lists(['不正常 " \t'], ["string"], ["正常"], ["bigint"])
    s_repr = repr(s)
    assert '"不正常 \\" \\t"' in s_repr
    assert '"正常"' not in s_repr


@py_and_c_deco
def test_string_as_binary():
    try:
        options.tunnel.string_as_binary = True
        s = TableSchema.from_lists(["col1", "col2"], ["string", "bigint"])
        r = Record(values=[1, 2], schema=s)
        assert r["col1", "col2"] == [b"1", 2]
        assert isinstance(r[0], bytes)

        r[0] = u"junk"
        assert r[0] == b"junk"
        assert isinstance(r[0], bytes)

        r[0] = b"junk"
        assert r[0] == b"junk"
        assert isinstance(r[0], bytes)
    finally:
        options.tunnel.string_as_binary = False


@py_and_c_deco
@pytest.mark.parametrize("map_as_ordered_dict", [False, True])
def test_validate_nested_types(map_as_ordered_dict):
    orig_map_as_ordered_dict = options.map_as_ordered_dict
    try:
        options.map_as_ordered_dict = map_as_ordered_dict

        s = TableSchema.from_lists(
            ["col1"], ["array<map<string,struct<abc: int, def: string>>>"]
        )
        r = Record(schema=s)
        r[0] = [{"abcd": (123, "uvw")}]
        assert r[0] == [{"abcd": (123, "uvw")}]
        map_type = OrderedDict if map_as_ordered_dict else dict
        assert type(r[0][0]) is map_type

        s = TableSchema.from_lists(
            ["col1"], ["struct<abc: int, def: map<string, int>>"]
        )
        r = Record(schema=s)
        r[0] = (123, {"uvw": 123})
        assert r[0] == (123, {"uvw": 123})
    finally:
        options.map_as_ordered_dict = orig_map_as_ordered_dict


@py_and_c_deco
def test_validate_struct():
    try:
        options.struct_as_dict = True
        struct_type = odps_types.validate_data_type("struct<abc: int, def: string>")
        assert odps_types.validate_value(None, struct_type) is None

        vl = odps_types.validate_value((10, "uvwxyz"), struct_type)
        assert isinstance(vl, dict)
        assert vl["abc"] == 10
        assert vl["def"] == "uvwxyz"

        vl = odps_types.validate_value({"abc": 10, "def": "uvwxyz"}, struct_type)
        assert isinstance(vl, dict)
        assert vl["abc"] == 10
        assert vl["def"] == "uvwxyz"

        with pytest.raises(ValueError):
            odps_types.validate_value({"abcd", "efgh"}, struct_type)

        options.struct_as_dict = False
        struct_type = odps_types.validate_data_type("struct<abc: int, def: string>")
        vl = odps_types.validate_value((10, "uvwxyz"), struct_type)
        assert isinstance(vl, tuple)
        assert vl == (10, "uvwxyz")

        vl = odps_types.validate_value({"def": "uvwxyz", "abc": 10}, struct_type)
        assert isinstance(vl, tuple)
        assert vl == (10, "uvwxyz")

        with pytest.raises(ValueError):
            odps_types.validate_value({"abcd", "efgh"}, struct_type)
    finally:
        options.struct_as_dict = False


@py_and_c_deco
@pytest.mark.parametrize("use_binary", [False, True])
def test_varchar_size_limit(use_binary):
    def _c(s):
        if use_binary:
            return utils.to_binary(s)
        return utils.to_text(s)

    s = TableSchema.from_lists(["col1"], ["varchar(3)"])
    r = Record(schema=s)
    r[0] = _c("123")
    r[0] = _c("测试字")
    with pytest.raises(ValueError):
        r[0] = _c("1234")
    with pytest.raises(ValueError):
        r[0] = _c("测试字符")


@py_and_c_deco
@pytest.mark.parametrize("use_binary", [False, True])
def test_field_size_limit(use_binary):
    def _c(s):
        if use_binary:
            return utils.to_binary(s)
        return utils.to_text(s)

    s = TableSchema.from_lists(["str_col", "bin_col"], ["string", "binary"])
    r = Record(schema=s, max_field_size=1024)
    r[0] = _c("1" * 1024)
    r[1] = _c("1" * 1024)
    with pytest.raises(ValueError):
        r[0] = _c("1" * 1023 + "测")
    with pytest.raises(ValueError):
        r[1] = _c("1" * 1023 + "测")


def test_validate_decimal():
    with pytest.raises(ValueError):
        odps_types.Decimal(32, 60)
    with pytest.raises(ValueError):
        odps_types.Decimal(None, 10)

    assert repr(odps_types.Decimal()) == "decimal"
    assert repr(odps_types.Decimal(20)) == "decimal(20)"
    assert repr(odps_types.Decimal(20, 10)) == "decimal(20,10)"

    assert odps_types.Decimal(20, 5) == odps_types.Decimal(20, 5)
    assert odps_types.Decimal(10) == "decimal(10)"

    decimal_type = odps_types.Decimal(10, 5)
    decimal_type.validate_value(None)
    decimal_type.validate_value(_decimal.Decimal("123456789.1"))
    decimal_type.validate_value(_decimal.Decimal("123456789.12345"))
    with pytest.raises(ValueError):
        decimal_type.validate_value(_decimal.Decimal("12345678901.12"))


@pandas_case
def test_validate_timestamp():
    with pytest.raises(ValueError):
        odps_types.validate_value("abcdef", odps_types.timestamp)

    vl = odps_types.validate_value("2023-12-19 14:24:31", odps_types.timestamp)
    assert vl == pd.Timestamp("2023-12-19 14:24:31")

    ts = pd.Timestamp("2023-12-19 14:24:31")
    vl = odps_types.validate_value(ts, odps_types.timestamp)
    assert vl == ts
