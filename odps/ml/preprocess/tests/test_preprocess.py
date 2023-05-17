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

from ....df import DataFrame
from ... import merge_data
from ...preprocess import *
from ...tests.base import MLTestUtil, tn, ci_skip_case

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
IONOSPHERE_RANDOM_SAMPLE_TABLE = tn('pyodps_test_ml_iono_rand_sample')
IONOSPHERE_WEIGHTED_SAMPLE_TABLE = tn('pyodps_test_ml_iono_weight_sample')
IONOSPHERE_APPEND_ID_TABLE = tn('pyodps_test_ml_iono_append_id')
IONOSPHERE_MERGED_TABLE = tn('pyodps_test_ml_iono_merged')
IONOSPHERE_PRINCOMP_TABLE = tn('pyodps_test_ml_iono_princomp')
IONOSPHERE_ABNORMAL_TABLE = tn('pyodps_test_ml_iono_abnormal')
USER_ITEM_TABLE = tn('pyodps_test_ml_user_item')
USER_ITEM_UNPIVOT_TABLE = tn('pyodps_test_ml_unpivot_user_item')


@pytest.fixture
def utils(odps, tunnel):
    util = MLTestUtil(odps, tunnel)
    util.create_ionosphere(IONOSPHERE_TABLE)
    return util


@ci_skip_case
def test_merge(odps, utils):
    utils.delete_table(IONOSPHERE_MERGED_TABLE)
    ds = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    merged_df = merge_data(ds, ds, auto_rename=True)
    merged_df.persist(IONOSPHERE_MERGED_TABLE)
    assert odps.exist_table(IONOSPHERE_MERGED_TABLE)


@ci_skip_case
def test_sample(odps, utils):
    utils.delete_table(IONOSPHERE_WEIGHTED_SAMPLE_TABLE)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE)).label_field('class')
    df.sample(0.5, replace=True).persist(IONOSPHERE_RANDOM_SAMPLE_TABLE)
    assert odps.exist_table(IONOSPHERE_RANDOM_SAMPLE_TABLE)
    df['a01', 'a02', ((df.a05 + 1) / 2).rename('a05')].sample(0.5, prob_field='a05', replace=True).persist(
        IONOSPHERE_WEIGHTED_SAMPLE_TABLE)
    assert odps.exist_table(IONOSPHERE_WEIGHTED_SAMPLE_TABLE)