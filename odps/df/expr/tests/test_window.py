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

import pytest

from ..arithmetic import *
from ..window import *
from ..errors import ExpressionError
from ... import types


@pytest.fixture
def src_expr():
    schema = TableSchema.from_lists(types._data_types.keys(), types._data_types.values())
    return CollectionExpr(_source_data=None, _schema=schema)


def test_cum_sum(src_expr):
    grouped = src_expr.groupby('boolean')

    pytest.raises(AttributeError, lambda: grouped.datetime.cumsum())

    cumsum = grouped.sort('int8').int16.cumsum()

    assert isinstance(cumsum, CumSum)
    assert isinstance(cumsum, Int16SequenceExpr)
    assert [by.name for by in cumsum._partition_by] == ['boolean', ]
    assert [by.name for by in cumsum._order_by] == ['int8', ]


def test_cum_max(src_expr):
    cummax = src_expr.groupby(['int16', 'float64']).boolean.cummax()
    assert isinstance(cummax, CumMax)
    assert isinstance(cummax, BooleanSequenceExpr)
    assert [by.name for by in cummax._partition_by] == ['int16', 'float64']


def test_cum_min(src_expr):
    cummin = src_expr.groupby('datetime').int16.cummin()
    assert isinstance(cummin, CumMin)
    assert isinstance(cummin, Int16SequenceExpr)
    assert [by.name for by in cummin._partition_by] == ['datetime', ]


def test_cum_mean(src_expr):
    cummean = src_expr.groupby(src_expr.int16 * 2).sort(src_expr.float32 + 1).int64.cummean()
    assert isinstance(cummean, CumMean)
    assert isinstance(cummean, Float64SequenceExpr)
    assert [by.name for by in cummean._partition_by] == ['int16', ]
    assert len(cummean._partition_by) == 1
    assert isinstance(cummean._partition_by[0], Multiply)
    assert [by.name for by in cummean._order_by] == ['float32', ]
    assert len(cummean._order_by) == 1
    assert isinstance(cummean._order_by[0].input, Add)


def test_cum_median(src_expr):
    cummedian = src_expr.groupby(['float64', src_expr.string]).int32.cummedian(
            preceding=3, following=10)
    assert isinstance(cummedian, CumMedian)
    assert isinstance(cummedian, Float64SequenceExpr)
    assert cummedian._preceding == 3
    assert cummedian._following == 10


def test_cum_count(src_expr):
    cumcount = src_expr.groupby('decimal').string.cumcount().unique()
    assert isinstance(cumcount, CumCount)
    assert isinstance(cumcount, Int64SequenceExpr)
    assert cumcount._distinct

    s = src_expr.groupby('decimal').sort('string').string
    pytest.raises(ExpressionError, lambda: s.cumcount(unique=True))
    s = src_expr.groupby('decimal').string.cumcount(unique=True)
    assert [by.name for by in s._partition_by] == ['decimal', ]


def test_cum_std(src_expr):
    grouped = src_expr.groupby('datetime')

    pytest.raises(AttributeError, lambda: grouped.string.cumstd())

    cumstd = grouped.decimal.cumstd(preceding=(10, 3))
    assert isinstance(cumstd, CumStd)
    assert isinstance(cumstd, DecimalSequenceExpr)
    assert list(cumstd.preceding) == [10, 3]

    cumstd = grouped.int16.cumstd(following=(4, 8))
    assert isinstance(cumstd, Float64SequenceExpr)
    assert list(cumstd.following) == [4, 8]

    pytest.raises(AssertionError, lambda: grouped.int32.cumstd(preceding=(4, 5)))
    pytest.raises(AssertionError, lambda: grouped.int32.cumstd(following=(5, 4)))

    pytest.raises(ValueError,
                  lambda: grouped.int64.cumstd(preceding=(10, 3), following=20))
    pytest.raises(ValueError,
                  lambda: grouped.int64.cumstd(preceding=20, following=(3, 10)))
    pytest.raises(ValueError,
                  lambda: grouped.int64.cumstd(preceding=(10, 3), following=(3, 10)))


def test_rank(src_expr):
    rank = src_expr.groupby('boolean').sort(lambda x: x['float64']).rank()

    assert isinstance(rank, Rank)
    assert isinstance(rank, Int64SequenceExpr)
    assert [by.name for by in rank._partition_by] == ['boolean', ]
    assert [by.name for by in rank._order_by] == ['float64', ]


def test_dense_rank(src_expr):
    grouped = src_expr.groupby('datetime')
    denserank = grouped.boolean.dense_rank()

    assert denserank._input is grouped.dense_rank(sort='boolean')._input
    assert isinstance(denserank, DenseRank)
    assert isinstance(denserank, Int64SequenceExpr)


def test_percent_rank(src_expr):
    percentrank = src_expr.groupby('boolean').percent_rank(sort='int16')

    assert isinstance(percentrank, PercentRank)
    assert isinstance(percentrank, Float64SequenceExpr)


def test_row_number(src_expr):
    rownumber = src_expr.groupby('string').row_number(sort='int8').rename('rank')

    assert rownumber.name == 'rank'
    assert isinstance(rownumber, RowNumber)
    assert isinstance(rownumber, Int64SequenceExpr)


def test_lead(src_expr):
    grouped = src_expr.groupby(src_expr.string)

    pytest.raises(AttributeError, lambda: grouped.lead())

    lead = grouped.int8.lead(5, default=10)
    assert isinstance(lead, Lead)
    assert isinstance(lead, Int8SequenceExpr)
    assert lead._offset == 5
    assert lead._default == 10


def test_lag(src_expr):
    lag = src_expr.groupby('string').decimal.lag(-5, sort=['int8'])

    assert isinstance(lag, Lag)
    assert isinstance(lag, DecimalSequenceExpr)
    assert lag._offset == -5
    assert [by.name for by in lag._partition_by] == ['string', ]
    assert [by.name for by in lag._order_by] == ['int8', ]

    lag = src_expr.groupby('string').sort('int8').decimal.lag(1)

    assert isinstance(lag, Lag)
    assert isinstance(lag, DecimalSequenceExpr)
    assert lag._offset == 1
    assert [by.name for by in lag._partition_by] == ['string', ]
    assert [by.name for by in lag._order_by] == ['int8', ]