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
from ....config import options
from ...statistics import *
from ...tests.base import MLTestUtil, tn
from ...utils import TEMP_TABLE_PREFIX

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
IRIS_TABLE = tn('pyodps_test_ml_iris')
IONOSPHERE_PRINCOMP_TABLE = tn('pyodps_test_ml_iono_princomp')
IONOSPHERE_FEATURE_STATS = tn('pyodps_test_ml_iono_feature_stats')
IONOSPHERE_REPLACE_WOE = tn('pyodps_test_ml_iono_replace_woe')
IONOSPHERE_QUANTILE_TABLE = tn('pyodps_test_ml_iono_quantile')


@pytest.fixture
def utils(odps, tunnel):
    util = MLTestUtil(odps, tunnel)
    util.create_ionosphere(IONOSPHERE_TABLE)
    return util


def test_histograms(odps, utils):
    options.ml.dry_run = True

    ds = DataFrame(odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
    histograms(ds, _cases=utils.gen_check_params_case({
        'outputTableName': TEMP_TABLE_PREFIX + '_histograms',
        'selectedColNames': ','.join('a%02d' % i for i in range(1, 35)),
        'intervalNum': '10', 'inputTableName': IONOSPHERE_TABLE}))


def test_t_test(odps, utils):
    options.ml.dry_run = True

    ds = DataFrame(odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
    t_test(ds, x_col='a04', _cases=utils.gen_check_params_case(
        {'mu': '0', 'outputTableName': TEMP_TABLE_PREFIX + '_t_test', 'confidenceLevel': '0.95',
         'xTableName': odps.project + '.' + IONOSPHERE_TABLE, 'alternative': 'two.sided', 'xColName': 'a04'}))
    t_test(ds, x_col='a04', y_col='a05', _cases=utils.gen_check_params_case(
        {'yTableName': odps.project + '.' + IONOSPHERE_TABLE, 'yColName': 'a05', 'mu': '0',
         'outputTableName': TEMP_TABLE_PREFIX + '_t_test', 'confidenceLevel': '0.95',
         'xTableName': odps.project + '.' + IONOSPHERE_TABLE, 'alternative': 'two.sided',
         'xColName': 'a04'}))


def test_chisquare(odps, utils):
    options.ml.dry_run = True

    df = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    chi_square(df, x_col=df.a01, y_col='class', _cases=utils.gen_check_params_case(
        {'yColName': 'class', 'xColName': 'a01', 'outputDetailTableName': 'tmp_pyodps__chi_square',
         'outputTableName': 'tmp_pyodps__chi_square',
         'inputTableName': tn('pyodps_test_ml_ionosphere')}))


def test_cov(odps, utils):
    options.ml.dry_run = True

    df = DataFrame(odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
    covariance(df, _cases=utils.gen_check_params_case(
        {'outputTableName': 'tmp_pyodps__covariance',
         'selectedColNames': ','.join('a%02d' % i for i in range(1, 35)),
         'inputTableName': tn('pyodps_test_ml_ionosphere')}))


def test_mat_pearson(odps, utils):
    options.ml.dry_run = True

    df = DataFrame(odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
    matrix_pearson(df, _cases=utils.gen_check_params_case(
        {'outputTableName': 'tmp_pyodps__matrix_pearson',
         'selectedColNames': ','.join('a%02d' % i for i in range(1, 35)),
         'inputTableName': tn('pyodps_test_ml_ionosphere')}))


def test_quantile(odps, utils):
    options.ml.dry_run = True

    df = DataFrame(odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
    qt = quantile(df, _cases=utils.gen_check_params_case(
        {'inputTableName': tn('pyodps_test_ml_ionosphere'), 'outputTableName': tn('pyodps_test_ml_iono_quantile'),
         'colName': ','.join('a%02d' % i for i in range(1, 35)),
         'N': '100'}))
    qt.persist(IONOSPHERE_QUANTILE_TABLE)
