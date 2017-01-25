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
from odps.config import options
from odps.ml.utils import TEMP_TABLE_PREFIX
from odps.ml.regression import *
from odps.ml.feature import *
from odps.ml.statistics import *
from odps.ml.tests.base import MLTestBase, tn, otm, ci_skip_case
from odps.ml.metrics import *

import logging
logger = logging.getLogger(__name__)

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
XGBOOST_OUT_TABLE = tn('pyodps_test_xgboost_out')
GBDT_OUT_TABLE = tn('pyodps_test_gbdt_out')
LINEAR_REGRESSION_OUT_TABLE = tn('pyodps_test_linear_reg_out')
LINEAR_SVR_OUT_TABLE = tn('pyodps_test_linear_svr_out')
LASSO_OUT_TABLE = tn('pyodps_test_lasso_out')
RIDGE_OUT_TABLE = tn('pyodps_test_ridge_out')

MODEL_NAME = tn('pyodps_test_out_model')


class TestMLRegression(MLTestBase):
    def setUp(self):
        super(TestMLRegression, self).setUp()
        self.create_ionosphere(IONOSPHERE_TABLE)

        options.runner.dry_run = True

    def test_mock_xgboost(self):
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
        splited = df.split(0.6)

        xgboost = Xgboost()
        model = xgboost.train(splited[0])._add_case(self.gen_check_params_case({
            'labelColName': 'class', 'modelName': MODEL_NAME, 'colsample_bytree': '1', 'silent': '1',
            'eval_metric': 'error', 'eta': '0.3', 'inputTableName': TEMP_TABLE_PREFIX + '0_split_2_1', 'max_delta_step': '0',
            'base_score': '0.5', 'seed': '0', 'min_child_weight': '1', 'objective': 'reg:linear',
            'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
            'max_depth': '6', 'gamma': '0', 'booster': 'gbtree'}))
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])._add_case(self.gen_check_params_case({
            'modelName': MODEL_NAME, 'appendColNames': ','.join('a%02d' % i for i in range(1, 35)) + ',class',
            'outputTableName': XGBOOST_OUT_TABLE, 'inputTableName': TEMP_TABLE_PREFIX + '0_split_2_2'}))
        # persist is an operational node which will trigger execution of the flow
        predicted.persist(XGBOOST_OUT_TABLE)

    def test_mock_gbdt(self):
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
        splited = df.split(0.6)

        gbdt = GBDT(min_leaf_sample_count=10)
        model = gbdt.train(splited[0])._add_case(self.gen_check_params_case({
            'tau': '0.6', 'modelName': MODEL_NAME, 'inputTableName': TEMP_TABLE_PREFIX + '0_split_2_1', 'maxLeafCount': '32',
            'shrinkage': '0.05', 'featureSplitValueMaxSize': '500', 'featureRatio': '0.6', 'testRatio': '0.0',
            'newtonStep': '0', 'randSeed': '0', 'sampleRatio': '0.6', 'p': '1', 'treeCount': '500', 'metricType': '2',
            'labelColName': 'class', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
            'minLeafSampleCount': '10', 'lossType': '3', 'maxDepth': '11'}))
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])._add_case(self.gen_check_params_case({
            'modelName': MODEL_NAME, 'appendColNames': ','.join('a%02d' % i for i in range(1, 35)) + ',class',
            'outputTableName': GBDT_OUT_TABLE, 'inputTableName': TEMP_TABLE_PREFIX + '0_split_2_2'}))
        # persist is an operational node which will trigger execution of the flow
        predicted.persist(GBDT_OUT_TABLE)

    @ci_skip_case
    def test_linear(self):
        options.runner.dry_run = False
        self.delete_table(LINEAR_REGRESSION_OUT_TABLE)
        self.delete_offline_model(MODEL_NAME)

        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).roles(label='class')
        splited = df.split(0.6)

        algo = LinearRegression()
        model = algo.train(splited[0])
        model.persist(MODEL_NAME)

        logging.info('Importance: ', regression_importance(splited[1], model))

        predicted = model.predict(splited[1])
        # persist is an operational node which will trigger execution of the flow
        predicted.persist(LINEAR_REGRESSION_OUT_TABLE)

        logging.info('MSE: ', mean_squared_error(predicted, 'class'))
        logging.info('MAE: ', mean_absolute_error(predicted, 'class'))
        logging.info('HIST: ', residual_histogram(predicted, 'class'))
        logging.info('MSE: ', pearson(predicted, col1='class'))
