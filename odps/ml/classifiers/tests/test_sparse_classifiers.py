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

import logging

from odps.df import DataFrame
from odps.config import options
from odps.ml.utils import TEMP_TABLE_PREFIX
from odps.ml.classifiers import *
from odps.ml.metrics.classification import roc_curve
from odps.ml.tests.base import MLTestBase, tn, ci_skip_case

logger = logging.getLogger(__name__)

IRIS_KV_TABLE = tn('pyodps_test_ml_iris_sparse')

LR_TEST_TABLE = tn('pyodps_lr_output_table')
XGBOOST_TEST_TABLE = tn('pyodps_xgboost_output_table')

MODEL_NAME = tn('pyodps_test_out_model')


class TestSparseClassifiers(MLTestBase):
    def setUp(self):
        super(TestSparseClassifiers, self).setUp()
        self.create_iris_kv(IRIS_KV_TABLE)
        self.df = DataFrame(self.odps.get_table(IRIS_KV_TABLE)).label_field('category').key_value('content')

    def tearDown(self):
        super(TestSparseClassifiers, self).tearDown()

    @ci_skip_case
    def test_logistic_regression(self):
        options.runner.dry_run = False
        self.delete_table(LR_TEST_TABLE)
        self.delete_offline_model(MODEL_NAME)

        splited = self.df.split(0.6)

        lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
        model = lr.train(splited[0])
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])
        # persist is an operational node which will trigger execution of the flow
        predicted.persist(LR_TEST_TABLE)

        fpr, tpr, thresh = roc_curve(predicted, "category")
        assert len(fpr) == len(tpr) and len(thresh) == len(fpr)

    def test_mock_xgboost(self):
        options.runner.dry_run = True

        splited = self.df.split(0.6)

        lr = Xgboost()
        model = lr.train(splited[0])._add_case(self.gen_check_params_case(
                {'labelColName': 'category', 'modelName': MODEL_NAME, 'colsample_bytree': '1', 'silent': '1',
                 'eval_metric': 'error', 'eta': '0.3', 'itemDelimiter': ',', 'kvDelimiter': ':',
                 'inputTableName': TEMP_TABLE_PREFIX + '0_split_2_1', 'max_delta_step': '0', 'enableSparse': 'true',
                 'base_score': '0.5', 'seed': '0', 'min_child_weight': '1', 'objective': 'binary:logistic',
                 'featureColNames': 'content', 'max_depth': '6', 'gamma': '0', 'booster': 'gbtree'}))
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])._add_case(self.gen_check_params_case(
                {'itemDelimiter': ',', 'modelName': MODEL_NAME, 'appendColNames': 'content,category',
                 'inputTableName': TEMP_TABLE_PREFIX + '0_split_2_2', 'enableSparse': 'true',
                 'outputTableName': XGBOOST_TEST_TABLE, 'kvDelimiter': ':', 'featureColNames': 'content'}))
        # persist operational node which will trigger execution of the flow
        predicted.persist(XGBOOST_TEST_TABLE)
