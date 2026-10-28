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

from collections import OrderedDict

import pytest

from odps import types as odps_types
from odps.models import TableSchema
from odps.readers import CsvRecordReader, parse_complex_value

try:
    import numpy as np
except ImportError:
    np = None


def _make_columns(type_list):
    return TableSchema.from_lists(
        [f"c{i}" for i in range(len(type_list))], type_list
    ).columns


def _assert_vector_result(result, expected):
    """Compare a vector parse result, accounting for float32 precision."""
    if np is not None and isinstance(result, np.ndarray):
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "value,data_type,expected",
    [
        ("[a,b,c]", odps_types.Array(odps_types.string), ["a", "b", "c"]),
        ("[1,2,3]", odps_types.Array(odps_types.bigint), [1, 2, 3]),
        ("[]", odps_types.Array(odps_types.string), []),
        ("[a,\\N,c]", odps_types.Array(odps_types.string), ["a", None, "c"]),
        (
            "[[a,b],[c,d]]",
            odps_types.Array(odps_types.Array(odps_types.string)),
            [["a", "b"], ["c", "d"]],
        ),
        (
            "[{k1:v1},{k2:v2}]",
            odps_types.Array(odps_types.Map(odps_types.string, odps_types.string)),
            [OrderedDict([("k1", "v1")]), OrderedDict([("k2", "v2")])],
        ),
        (
            "{k1:v1,k2:v2}",
            odps_types.Map(odps_types.string, odps_types.string),
            OrderedDict([("k1", "v1"), ("k2", "v2")]),
        ),
        (
            "{k1:1,k2:2}",
            odps_types.Map(odps_types.string, odps_types.bigint),
            OrderedDict([("k1", 1), ("k2", 2)]),
        ),
        (
            "{}",
            odps_types.Map(odps_types.string, odps_types.string),
            OrderedDict(),
        ),
        (
            "{k1:{k2:v2},k3:{k4:v4}}",
            odps_types.Map(
                odps_types.string, odps_types.Map(odps_types.string, odps_types.string)
            ),
            OrderedDict(
                [
                    ("k1", OrderedDict([("k2", "v2")])),
                    ("k3", OrderedDict([("k4", "v4")])),
                ]
            ),
        ),
        (
            "{k1:[a,b],k2:[c]}",
            odps_types.Map(odps_types.string, odps_types.Array(odps_types.string)),
            OrderedDict([("k1", ["a", "b"]), ("k2", ["c"])]),
        ),
        (
            "{k1:v1:v2}",
            odps_types.Map(odps_types.string, odps_types.string),
            OrderedDict([("k1", "v1:v2")]),
        ),
        (
            '{k1:"a,b",k2:c}',
            odps_types.Map(odps_types.string, odps_types.string),
            OrderedDict([("k1", "a,b"), ("k2", "c")]),
        ),
        (
            r"{k1:a\"b,k2:c}",
            odps_types.Map(odps_types.string, odps_types.string),
            OrderedDict([("k1", 'a"b'), ("k2", "c")]),
        ),
        (
            "[{k1:[1,2]},{k2:[3]}]",
            odps_types.Array(
                odps_types.Map(odps_types.string, odps_types.Array(odps_types.bigint))
            ),
            [OrderedDict([("k1", [1, 2])]), OrderedDict([("k2", [3])])],
        ),
    ],
)
def test_parse_complex_value(value, data_type, expected):
    assert parse_complex_value(value, data_type) == expected


@pytest.mark.parametrize(
    "value,struct_type,expected",
    [
        (
            "{a:hello,b:42}",
            odps_types.Struct([("a", odps_types.string), ("b", odps_types.bigint)]),
            ("hello", 42),
        ),
        (
            "{name:foo,attrs:{x:1,y:2}}",
            odps_types.Struct(
                [
                    ("name", odps_types.string),
                    ("attrs", odps_types.Map(odps_types.string, odps_types.bigint)),
                ]
            ),
            ("foo", OrderedDict([("x", 1), ("y", 2)])),
        ),
        (
            "{name:foo,tags:[a,b,c]}",
            odps_types.Struct(
                [
                    ("name", odps_types.string),
                    ("tags", odps_types.Array(odps_types.string)),
                ]
            ),
            ("foo", ["a", "b", "c"]),
        ),
    ],
)
def test_parse_complex_value_struct(value, struct_type, expected):
    result = parse_complex_value(value, struct_type)
    assert result == struct_type.namedtuple_type(*expected)


@pytest.mark.parametrize(
    "value,data_type,error_match",
    [
        ("not_an_array", odps_types.Array(odps_types.string), "Array format error"),
        (
            "not_a_map",
            odps_types.Map(odps_types.string, odps_types.string),
            "Map format error",
        ),
        (
            "not_a_vector",
            odps_types.Vector(odps_types.float_, 2),
            "Vector format error",
        ),
        ("[]", odps_types.Vector(odps_types.float_, 2), "at least one element"),
        (
            "[1.0, 2.0]",
            odps_types.Vector(odps_types.float_, 3),
            "Vector dimension mismatch",
        ),
        ("[nan, 1.0]", odps_types.Vector(odps_types.float_, 2), "NaN or Infinity"),
        ("[inf, 1.0]", odps_types.Vector(odps_types.float_, 2), "NaN or Infinity"),
        (
            "[-inf, 1.0]",
            odps_types.Vector(odps_types.float_, 2),
            "NaN or Infinity",
        ),
        (
            "[3.5e38, \\N]",
            odps_types.Vector(odps_types.float_, 2),
            "must not be null",
        ),
        pytest.param(
            "[3.5e38, 1.0]",
            odps_types.Vector(odps_types.float_, 2),
            "overflow",
            marks=pytest.mark.skipif(np is None, reason="numpy unavailable"),
        ),
    ],
)
def test_parse_complex_value_error(value, data_type, error_match):
    with pytest.raises((ValueError, FloatingPointError), match=error_match):
        parse_complex_value(value, data_type)


@pytest.mark.parametrize(
    "col_type,value,expected",
    [
        (
            odps_types.Map(
                odps_types.string,
                odps_types.Map(odps_types.string, odps_types.string),
            ),
            "{k1:{k2:v2},k3:{k4:v4}}",
            OrderedDict(
                [
                    ("k1", OrderedDict([("k2", "v2")])),
                    ("k3", OrderedDict([("k4", "v4")])),
                ]
            ),
        ),
        (
            odps_types.Array(odps_types.string),
            "[a,b,c]",
            ["a", "b", "c"],
        ),
        (
            odps_types.Struct([("x", odps_types.bigint), ("y", odps_types.string)]),
            "{x:42,y:hello}",
            None,  # checked separately via namedtuple
        ),
        (
            odps_types.Vector(odps_types.float_, 3),
            "[1.0, 2.0, 3.0]",
            [1.0, 2.0, 3.0],
        ),
        (
            odps_types.Vector(odps_types.double, 2),
            "[0.5, -0.5]",
            [0.5, -0.5],
        ),
    ],
)
def test_get_caster_complex_types(col_type, value, expected):
    cols = _make_columns([col_type])
    caster = CsvRecordReader._get_caster(cols[0].type)
    assert caster is not None
    result = caster(value)
    if expected is None:
        assert result == col_type.namedtuple_type(42, "hello")
    elif isinstance(col_type, odps_types.Vector):
        _assert_vector_result(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "col_type",
    [
        odps_types.Binary(),
        odps_types.Blob(),
        odps_types.Array(odps_types.Binary()),
        odps_types.Map(odps_types.string, odps_types.Blob()),
        odps_types.Array(odps_types.Map(odps_types.string, odps_types.Binary())),
    ],
)
def test_get_caster_unsupported_returns_none(col_type):
    assert CsvRecordReader._get_caster(col_type) is None


@pytest.mark.parametrize(
    "value,element_type,dimension,expected",
    [
        ("[1.0, 2.5, -3.14]", odps_types.float_, 3, [1.0, 2.5, -3.14]),
        ("[1.0, 2.5, -3.14]", odps_types.double, 3, [1.0, 2.5, -3.14]),
        ("[0.0]", odps_types.float_, 1, [0.0]),
        ("[1.0, 0.0, 1.0, 0.0]", odps_types.float_, 4, [1.0, 0.0, 1.0, 0.0]),
    ],
)
def test_parse_complex_value_vector(value, element_type, dimension, expected):
    result = parse_complex_value(value, odps_types.Vector(element_type, dimension))
    _assert_vector_result(result, expected)


def test_cast_value_legacy_vector():
    """Legacy CSV casting must parse Vector columns like other composites."""
    cols = _make_columns([odps_types.Vector(odps_types.float_, 2)])
    reader = CsvRecordReader.__new__(CsvRecordReader)  # skip __init__ (no stream)
    reader._csv_columns = cols

    result = reader._cast_value_legacy(0, "[1.0, 2.0]")
    _assert_vector_result(result, [1.0, 2.0])
