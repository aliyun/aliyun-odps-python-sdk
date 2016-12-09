# encoding: utf-8
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

from __future__ import print_function

from odps.df import DataFrame
from odps.ml.classifiers import *
from odps.ml.feature import *
from odps.tests.core import tn, ci_skip_case, pandas_case
from odps.config import options
from odps.ml.tests.base import MLTestBase
from odps.ml.utils import drop_table
from odps.tests.core import pandas_case

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
IONOSPHERE_TABLE_TWO_PARTS = tn('pyodps_test_ml_ionosphere_two_parts')
IONOSPHERE_SORTED_TABLE = tn('pyodps_test_ml_iono_sorted')
IONOSPHERE_SORTED_TABLE_PART = tn('pyodps_test_ml_iono_sorted_part')
IONOSPHERE_PREDICTED_1 = tn('pyodps_test_ml_iono_predicted_1')
IONOSPHERE_PREDICTED_2 = tn('pyodps_test_ml_iono_predicted_2')


class Test(MLTestBase):
    @ci_skip_case
    def test_df_store(self):
        self.delete_table(IONOSPHERE_SORTED_TABLE_PART)
        self.create_ionosphere_two_parts(IONOSPHERE_TABLE_TWO_PARTS)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE_TWO_PARTS)).filter_partition('part1=1,part2=2')
        drop_table(self.odps, IONOSPHERE_SORTED_TABLE_PART, async=False)
        sorted_df = df.groupby(df['class']).agg(df.a01.count().rename('count')).sort('class', ascending=False)
        sorted_df.persist(IONOSPHERE_SORTED_TABLE_PART)

    @ci_skip_case
    def test_df_method(self):
        self.create_ionosphere(IONOSPHERE_TABLE)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))
        sorted_df = df.groupby(df['class']).agg(df.a01.count().rename('count')).sort('class', ascending=False)
        sorted_df.to_pandas()

    @pandas_case
    def test_df_consecutive(self):
        self.create_ionosphere(IONOSPHERE_TABLE)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))
        df = df[df['a04'] != 0]
        df = df.roles(label='class')
        df.head(10)
        train, test = df.split(0.6)
        lr = LogisticRegression(epsilon=0.01)
        model = lr.train(train)
        predicted = model.predict(test)
        predicted.to_pandas()

    def test_df_multiple_persist(self):
        self.odps.delete_table(IONOSPHERE_PREDICTED_1, if_exists=True)
        self.odps.delete_table(IONOSPHERE_PREDICTED_2, if_exists=True)

        self.create_ionosphere(IONOSPHERE_TABLE)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
        lr = LogisticRegression(epsilon=0.01)
        model = lr.train(df)
        predicted = model.predict(df)
        predicted.persist(IONOSPHERE_PREDICTED_1)
        predicted.persist(IONOSPHERE_PREDICTED_2)
        assert self.odps.exist_table(IONOSPHERE_PREDICTED_1)
        assert self.odps.exist_table(IONOSPHERE_PREDICTED_2)

    @pandas_case
    def test_df_combined(self):
        self.create_ionosphere(IONOSPHERE_TABLE)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))
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
    def test_pd_df(self):
        self.create_ionosphere(IONOSPHERE_TABLE)
        pd_df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).to_pandas()

    @ci_skip_case
    def test_direct_method(self):
        self.create_ionosphere(IONOSPHERE_TABLE)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
        train, test = df.split(0.6)
        lr = LogisticRegression(epsilon=0.01)
        model = lr.train(train)
        predicted = model.predict(test)
        predicted.to_pandas()

    @ci_skip_case
    def test_dynamic_output(self):
        self.create_ionosphere(IONOSPHERE_TABLE)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))
        df = df.roles(label=df['class'])
        filtered, importance = select_features(df)
        print(filtered.describe().execute())

    @ci_skip_case
    def test_ml_end(self):
        old_interactive = options.interactive
        options.interactive = True
        self.create_ionosphere(IONOSPHERE_TABLE)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).sample(n=20)
        repr(df)
        options.interactive = old_interactive
