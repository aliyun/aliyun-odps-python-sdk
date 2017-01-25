# encoding: utf-8
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from odps.df import DataFrame
from odps.ml import merge_data
from odps.ml.preprocess import *
from odps.ml.tests.base import MLTestBase, tn, ci_skip_case

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
IONOSPHERE_NORMALIZED_TABLE = tn('pyodps_test_ml_iono_normalized')
IONOSPHERE_STANDARDIZED_TABLE = tn('pyodps_test_ml_iono_standardized')
IONOSPHERE_RANDOM_SAMPLE_TABLE = tn('pyodps_test_ml_iono_rand_sample')
IONOSPHERE_WEIGHTED_SAMPLE_TABLE = tn('pyodps_test_ml_iono_weight_sample')
IONOSPHERE_APPEND_ID_TABLE = tn('pyodps_test_ml_iono_append_id')
IONOSPHERE_MERGED_TABLE = tn('pyodps_test_ml_iono_merged')
IONOSPHERE_PRINCOMP_TABLE = tn('pyodps_test_ml_iono_princomp')
IONOSPHERE_ABNORMAL_TABLE = tn('pyodps_test_ml_iono_abnormal')
USER_ITEM_TABLE = tn('pyodps_test_ml_user_item')
USER_ITEM_UNPIVOT_TABLE = tn('pyodps_test_ml_unpivot_user_item')


class TestPreprocess(MLTestBase):
    def setUp(self):
        super(TestPreprocess, self).setUp()
        self.create_ionosphere(IONOSPHERE_TABLE)

    @ci_skip_case
    def test_merge(self):
        self.delete_table(IONOSPHERE_MERGED_TABLE)
        ds = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))
        merged_df = merge_data(ds, ds, auto_rename=True)
        merged_df.persist(IONOSPHERE_MERGED_TABLE)
        assert self.odps.exist_table(IONOSPHERE_MERGED_TABLE)

    @ci_skip_case
    def test_sample(self):
        self.delete_table(IONOSPHERE_WEIGHTED_SAMPLE_TABLE)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).label_field('class')
        df.sample(0.5, replace=True).persist(IONOSPHERE_RANDOM_SAMPLE_TABLE)
        assert self.odps.exist_table(IONOSPHERE_RANDOM_SAMPLE_TABLE)
        df['a01', 'a02', ((df.a05 + 1) / 2).rename('a05')].sample(0.5, prob_field='a05', replace=True).persist(
            IONOSPHERE_WEIGHTED_SAMPLE_TABLE)
        assert self.odps.exist_table(IONOSPHERE_WEIGHTED_SAMPLE_TABLE)