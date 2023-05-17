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

import logging

import pytest

from ....df import DataFrame
from ....config import options
from ...utils import TEMP_TABLE_PREFIX
from ...classifiers import *
from ...metrics.classification import roc_curve
from ...tests.base import tn, ci_skip_case, MLTestUtil

logger = logging.getLogger(__name__)

IRIS_KV_TABLE = tn('pyodps_test_ml_iris_sparse')

LR_TEST_TABLE = tn('pyodps_lr_output_table')
XGBOOST_TEST_TABLE = tn('pyodps_xgboost_output_table')

MODEL_NAME = tn('pyodps_test_out_model')


@pytest.fixture
def utils(odps, tunnel):
    util = MLTestUtil(odps, tunnel)
    util.create_iris_kv(IRIS_KV_TABLE)
    util.df = DataFrame(odps.get_table(IRIS_KV_TABLE)).label_field('category').key_value('content')
    return util


@ci_skip_case
def test_logistic_regression(utils):
    options.ml.dry_run = False
    utils.delete_table(LR_TEST_TABLE)
    utils.delete_offline_model(MODEL_NAME)

    splited = utils.df.split(0.6)

    lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
    model = lr.train(splited[0])
    model.persist(MODEL_NAME)

    predicted = model.predict(splited[1])
    # persist is an operational node which will trigger execution of the flow
    predicted.persist(LR_TEST_TABLE)

    fpr, tpr, thresh = roc_curve(predicted, "category")
    assert len(fpr) == len(tpr) and len(thresh) == len(fpr)


def test_mock_xgboost(utils):
    options.ml.dry_run = True

    splited = utils.df.split(0.6)

    lr = Xgboost()
    model = lr.train(splited[0])._add_case(utils.gen_check_params_case(
            {'labelColName': 'category', 'modelName': MODEL_NAME, 'colsample_bytree': '1', 'silent': '1',
             'eval_metric': 'error', 'eta': '0.3', 'itemDelimiter': ',', 'kvDelimiter': ':',
             'inputTableName': TEMP_TABLE_PREFIX + '_split', 'max_delta_step': '0', 'enableSparse': 'true',
             'base_score': '0.5', 'seed': '0', 'min_child_weight': '1', 'objective': 'binary:logistic',
             'featureColNames': 'content', 'max_depth': '6', 'gamma': '0', 'booster': 'gbtree'}))
    model.persist(MODEL_NAME)

    predicted = model.predict(splited[1])._add_case(utils.gen_check_params_case(
            {'itemDelimiter': ',', 'modelName': MODEL_NAME, 'appendColNames': 'content,category',
             'inputTableName': TEMP_TABLE_PREFIX + '_split', 'enableSparse': 'true',
             'outputTableName': XGBOOST_TEST_TABLE, 'kvDelimiter': ':', 'featureColNames': 'content'}))
    # persist operational node which will trigger execution of the flow
    predicted.persist(XGBOOST_TEST_TABLE)
