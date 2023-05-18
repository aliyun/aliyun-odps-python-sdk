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

import textwrap

import pytest

from ....compat import six, LESS_PY35
from ..reduction import *
from ... import types


@pytest.fixture
def src_expr():
    schema = TableSchema.from_lists(types._data_types.keys(), types._data_types.values())
    return CollectionExpr(_source_data=None, _schema=schema)


def test_min(src_expr):
    min_ = src_expr.string.min()
    assert isinstance(min_, StringScalar)

    min_ = src_expr.boolean.min()
    assert isinstance(min_, BooleanScalar)

    min_ = src_expr.int16.min()
    assert isinstance(min_, Int16Scalar)

    expr = src_expr.min()
    assert isinstance(expr, Summary)
    assert len(expr.fields) == len(types._data_types)
    assert all(isinstance(node, Min) for node in expr.fields) is True


def test_max(src_expr):
    max_ = src_expr.string.max()
    assert isinstance(max_, StringScalar)

    max_ = src_expr.boolean.max()
    assert isinstance(max_, BooleanScalar)

    max_ = src_expr.int32.max()
    assert isinstance(max_, Int32Scalar)

    expr = src_expr.max()
    assert isinstance(expr, Summary)
    assert len(expr.fields) <= len(types._data_types)
    assert all(isinstance(node, Max) for node in expr.fields) is True


def test_count(src_expr):
    count = src_expr.string.count()
    assert isinstance(count, Int64Scalar)

    count = src_expr.string.unique().count()
    assert isinstance(count, Int64Scalar)

    expr = src_expr.count()
    assert isinstance(expr, Int64Scalar)


def test_sum(src_expr):
    sum = src_expr.string.sum()
    assert isinstance(sum, StringScalar)

    sum = src_expr.string.unique().sum()
    assert isinstance(sum, StringScalar)

    sum = src_expr.boolean.sum()
    assert isinstance(sum, Int64Scalar)

    sum = src_expr.int32.sum()
    assert isinstance(sum, Int32Scalar)

    sum = src_expr.decimal.sum()
    assert isinstance(sum, DecimalScalar)

    expr = src_expr.sum()
    assert isinstance(expr, Summary)
    assert len(expr.fields) <= len(types._data_types)
    assert all(isinstance(node, Sum) for node in expr.fields) is True


def test_var(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.string.var())

    var = src_expr.int8.var()
    assert isinstance(var, Float64Scalar)

    var = src_expr.decimal.var()
    assert isinstance(var, DecimalScalar)

    expr = src_expr.var()
    assert isinstance(expr, Summary)
    assert len(expr.fields) <= len(types._data_types)
    assert all(isinstance(node, Var) for node in expr.fields) is True


def test_std(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.boolean.var())

    std = src_expr.int64.std()
    assert isinstance(std, Float64Scalar)

    std = src_expr.decimal.std()
    assert isinstance(std, DecimalScalar)

    expr = src_expr.std()
    assert isinstance(expr, Summary)
    assert len(expr.fields) <= len(types._data_types)
    assert all(isinstance(node, Std) for node in expr.fields) is True


def test_mean(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.datetime.mean())

    mean = src_expr.float32.mean()
    assert isinstance(mean, Float64Scalar)

    mean = src_expr.decimal.std()
    assert isinstance(mean, DecimalScalar)

    expr = src_expr.mean()
    assert isinstance(expr, Summary)
    assert len(expr.fields) <= len(types._data_types)
    assert all(isinstance(node, Mean) for node in expr.fields) is True


def test_median(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.string.median())

    median = src_expr.float64.median()
    assert isinstance(median, Float64Scalar)

    median = src_expr.decimal.median()
    assert isinstance(median, DecimalScalar)

    expr = src_expr.median()
    assert isinstance(expr, Summary)
    assert len(expr.fields) <= len(types._data_types)
    assert all(isinstance(node, Median) for node in expr.fields) is True


def test_any(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.string.any())
    pytest.raises(AttributeError, lambda: src_expr.int64.any())

    any_ = src_expr.boolean.any()
    assert isinstance(any_, BooleanScalar)

    expr = src_expr.any()
    assert isinstance(expr, Summary)
    assert len(expr.fields) <= len(types._data_types)
    assert all(isinstance(node, Any) for node in expr.fields) is True


def test_all(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.string.all())
    pytest.raises(AttributeError, lambda: src_expr.int64.all())

    any_ = src_expr.boolean.all()
    assert isinstance(any_, BooleanScalar)

    expr = src_expr.all()
    assert isinstance(expr, Summary)
    assert len(expr.fields) <= len(types._data_types)
    assert all(isinstance(node, All) for node in expr.fields) is True


def test_cat(src_expr):
    pytest.raises(AttributeError, lambda: src_expr.int64.cat(sep=','))
    pytest.raises(AttributeError, lambda: src_expr.float.cat(sep=','))
    pytest.raises(AttributeError, lambda: src_expr.cat(sep=','))

    cat = src_expr.string.cat(sep=',')
    assert isinstance(cat, StringScalar)


def test_agg(src_expr):
    class Agg(object):
        def buffer(self):
            return [0]

        def __call__(self, buffer, val):
            buffer[0] += val

        def merge(self, buffer, pbuffer):
            buffer[0] += pbuffer[0]

        def getvalue(self, buffer):
            return buffer[0]

    expr = src_expr.int64.agg(Agg)

    assert isinstance(expr, Aggregation)
    assert expr.dtype == types.int64

    if not LESS_PY35:
        l = locals().copy()
        six.exec_(textwrap.dedent("""
        class Agg(object):
            def buffer(self):
                return [0]

            def __call__(self, buffer, val):
                buffer[0] += val

            def merge(self, buffer, pbuffer):
                buffer[0] += pbuffer[0]

            def getvalue(self, buffer) -> float:
                return buffer[0]

        expr = src_expr.int64.agg(Agg)
        """), globals(), l)
        expr = l['expr']
        assert isinstance(expr, Aggregation)
        assert isinstance(expr.dtype, types.Float)


def test_to_list(src_expr):
    expr = src_expr.int64.tolist()
    assert isinstance(expr, ListScalar)
    assert expr.dtype == types.validate_data_type('list<int64>')

    expr = src_expr.tolist()
    assert isinstance(expr, Summary)
    assert len(expr.fields) <= len(types._data_types)
    assert all(isinstance(node, ToList) for node in expr.fields) is True
