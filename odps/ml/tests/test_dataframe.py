# encoding: utf-8
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

from __future__ import print_function

import pytest

from ...df import DataFrame
from ...tests.core import tn, ci_skip_case, pandas_case
from ...config import options
from ...tests.core import pandas_case
from ..classifiers import *
from ..feature import *
from ..tests.base import MLTestUtil

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
IONOSPHERE_TABLE_TWO_PARTS = tn('pyodps_test_ml_ionosphere_two_parts')
IONOSPHERE_SORTED_TABLE = tn('pyodps_test_ml_iono_sorted')
IONOSPHERE_SORTED_TABLE_PART = tn('pyodps_test_ml_iono_sorted_part')
IONOSPHERE_PREDICTED_1 = tn('pyodps_test_ml_iono_predicted_1')
IONOSPHERE_PREDICTED_2 = tn('pyodps_test_ml_iono_predicted_2')
IONOSPHERE_SPLIT_1 = tn('pyodps_test_ml_iono_split_1')
IONOSPHERE_SPLIT_2 = tn('pyodps_test_ml_iono_split_2')


@pytest.fixture
def utils(odps, tunnel):
    return MLTestUtil(odps, tunnel)


@ci_skip_case
def test_df_store(odps, utils):
    utils.delete_table(IONOSPHERE_SORTED_TABLE_PART)
    utils.create_ionosphere_two_parts(IONOSPHERE_TABLE_TWO_PARTS)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE_TWO_PARTS)).filter_parts('part1=1,part2=2')
    odps.delete_table(IONOSPHERE_SORTED_TABLE_PART)
    sorted_df = df.groupby(df['class']).agg(df.a01.count().rename('count')).sort('class', ascending=False)
    sorted_df.persist(IONOSPHERE_SORTED_TABLE_PART)


@ci_skip_case
def test_df_method(odps, utils):
    utils.create_ionosphere(IONOSPHERE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    sorted_df = df.groupby(df['class']).agg(df.a01.count().rename('count')).sort('class', ascending=False)
    sorted_df.to_pandas()


@pandas_case
def test_df_consecutive(odps, utils):
    utils.create_ionosphere(IONOSPHERE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    df = df[df['a04'] != 0]
    df = df.roles(label='class')
    df.head(10)
    df['b01'] = df['a06']
    train, test = df.split(0.6)
    lr = LogisticRegression(epsilon=0.01)
    model = lr.train(train)
    predicted = model.predict(test)
    predicted['appended_col'] = predicted['prediction_score'] * 2
    predicted.to_pandas()


def test_sequential_execute(odps, utils):
    utils.create_ionosphere(IONOSPHERE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
    train, test = df.split(0.6)
    lr = LogisticRegression(epsilon=0.01)
    model = lr.train(train)
    predicted = model.predict(test)
    predicted.count().execute()
    model = lr.train(predicted)
    predicted2 = model.predict(test)
    predicted2.count().execute()


def test_df_multiple_persist(odps, utils):
    odps.delete_table(IONOSPHERE_PREDICTED_1, if_exists=True)
    odps.delete_table(IONOSPHERE_PREDICTED_2, if_exists=True)

    utils.create_ionosphere(IONOSPHERE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
    lr = LogisticRegression(epsilon=0.01)
    model = lr.train(df)
    predicted = model.predict(df)
    predicted.persist(IONOSPHERE_PREDICTED_1)
    predicted.persist(IONOSPHERE_PREDICTED_2)
    assert odps.exist_table(IONOSPHERE_PREDICTED_1)
    assert odps.exist_table(IONOSPHERE_PREDICTED_2)


def test_persist_split(odps, utils):
    odps.delete_table(IONOSPHERE_SPLIT_1, if_exists=True)
    odps.delete_table(IONOSPHERE_SPLIT_2, if_exists=True)

    utils.create_ionosphere(IONOSPHERE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    split1, split2 = df.split(0.6)
    split1.persist(IONOSPHERE_SPLIT_1)
    split2.persist(IONOSPHERE_SPLIT_2)
    assert odps.exist_table(IONOSPHERE_SPLIT_1)
    assert odps.exist_table(IONOSPHERE_SPLIT_2)


@pandas_case
def test_df_combined(odps, utils):
    utils.create_ionosphere(IONOSPHERE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    df = df[df['a04'] != 0]
    df = df['a01', df.a05.map(lambda v: v * 2).rename('a05'), 'a06', 'class']
    df = df.roles(label='class')
    df = df[df.a05 != 0].cache()
    df = df[df.a05, ((df.a06 + 1) / 2).rename('a06'), 'class']
    train, test = df.split(0.6)
    lr = LogisticRegression(epsilon=0.01)
    model = lr.train(train)
    predicted = model.predict(test)
    (- 1.0 * ((predicted['class'] * predicted.prediction_score.log().rename('t')).rename('t1') + (
    (1 - predicted['class']) * (1 - predicted.prediction_score).log().rename('t0')).rename('t2')).rename(
        't3').sum() / predicted.prediction_score.count()).rename('t4').execute()


@pandas_case
def test_pd_df(odps, utils):
    utils.create_ionosphere(IONOSPHERE_TABLE)
    DataFrame(odps.get_table(IONOSPHERE_TABLE)).to_pandas()


@ci_skip_case
def test_direct_method(odps, utils):
    utils.create_ionosphere(IONOSPHERE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
    train, test = df.split(0.6)
    lr = LogisticRegression(epsilon=0.01)
    model = lr.train(train)
    predicted = model.predict(test)
    predicted.to_pandas()


@ci_skip_case
def test_dynamic_output(odps, utils):
    utils.create_ionosphere(IONOSPHERE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    df = df.roles(label=df['class'])
    filtered, importance = select_features(df)
    print(filtered.describe().execute())


@ci_skip_case
def test_ml_end(odps, utils):
    old_interactive = options.interactive
    options.interactive = True
    utils.create_ionosphere(IONOSPHERE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE)).sample(n=20)
    repr(df)
    options.interactive = old_interactive
