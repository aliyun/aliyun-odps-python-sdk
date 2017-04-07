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

import logging

from odps.config import options
from odps.df import DataFrame
from odps.ml.classifiers import *
from odps.ml.feature import *
from odps.ml.utils import TEMP_TABLE_PREFIX
from odps.ml.metrics import roc_curve, roc_auc_score, confusion_matrix
from odps.ml.tests.base import MLTestBase, tn, ci_skip_case

logger = logging.getLogger(__name__)

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')

LR_TEST_TABLE = tn('pyodps_lr_output_table')
XGBOOST_TEST_TABLE = tn('pyodps_xgboost_output_table')
RANDOM_FORESTS_TEST_TABLE = tn('pyodps_random_forests_output_table')
GBDT_LR_TEST_TABLE = tn('pyodps_gbdt_lr_output_table')
LINEAR_SVM_TEST_TABLE = tn('pyodps_linear_svm_output_table')
NAIVE_BAYES_TEST_TABLE = tn('pyodps_naive_bayes_output_table')
KNN_TEST_TABLE = tn('pyodps_knn_output_table')

MODEL_NAME = tn('pyodps_test_out_model')


class Test(MLTestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.maxDiff = None
        self.create_ionosphere(IONOSPHERE_TABLE)
        self.df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).label_field('class')

    def test_mock_logistic_regression(self):
        options.ml.dry_run = True

        splited = self.df.split(0.6)
        labeled_data = splited[0]

        lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
        model = lr.train(labeled_data, core_num=1, core_mem=1024)._add_case(self.gen_check_params_case(
            {'labelColName': 'class', 'modelName': MODEL_NAME, 'inputTableName': TEMP_TABLE_PREFIX + '_split',
             'epsilon': '0.001', 'regularizedLevel': '1', 'regularizedType': 'l1', 'maxIter': '50',
             'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
             'coreNum': '1', 'memSizePerCore': '1024'}))
        model.persist(MODEL_NAME)

        lr = LogisticRegression(epsilon=0.001).set_max_iter(100)
        model = lr.train(labeled_data)._add_case(self.gen_check_params_case(
            {'labelColName': 'class', 'modelName': MODEL_NAME, 'inputTableName': TEMP_TABLE_PREFIX + '_split',
             'epsilon': '0.001', 'regularizedLevel': '1', 'regularizedType': 'l1', 'maxIter': '100',
             'featureColNames': ','.join('a%02d' % i for i in range(1, 35))}))
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])._add_case(self.gen_check_params_case(
            {'modelName': MODEL_NAME, 'appendColNames': ','.join('a%02d' % i for i in range(1, 35)) + ',class',
             'outputTableName': LR_TEST_TABLE, 'inputTableName': TEMP_TABLE_PREFIX + '_split'}))
        predicted.persist(LR_TEST_TABLE)

    def test_mock_xgboost(self):
        options.ml.dry_run = True

        splited = self.df.split(0.6)

        labeled_data = splited[0].label_field("class")
        xgboost = Xgboost(silent=1).set_eta(0.3)
        model = xgboost.train(labeled_data)._add_case(self.gen_check_params_case(
            {'labelColName': 'class', 'modelName': MODEL_NAME, 'colsample_bytree': '1', 'silent': '1',
             'eval_metric': 'error', 'eta': '0.3', 'inputTableName': TEMP_TABLE_PREFIX + '_split',
             'max_delta_step': '0', 'base_score': '0.5', 'seed': '0', 'min_child_weight': '1',
             'objective': 'binary:logistic', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
             'max_depth': '6', 'gamma': '0', 'booster': 'gbtree'}))
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])
        predicted.persist(XGBOOST_TEST_TABLE)

    def test_mock_random_forests(self):
        options.ml.dry_run = True

        splited = self.df.split(0.6)

        labeled_data = splited[0].label_field("class")
        rf = RandomForests(tree_num=10).set_max_tree_deep(10)
        model = rf.train(labeled_data)._add_case(self.gen_check_params_case(
            {'labelColName': 'class', 'maxRecordSize': '100000', 'inputTableName': TEMP_TABLE_PREFIX + '_split',
             'maxTreeDeep': '10', 'treeNum': '10', 'isFeatureContinuous': ','.join(['1', ] * 34),
             'minNumObj': '2', 'randomColNum': '-1', 'modelName': MODEL_NAME, 'minNumPer': '-1',
             'featureColNames': ','.join('a%02d' % i for i in range(1, 35))}))
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])
        predicted.persist(RANDOM_FORESTS_TEST_TABLE)

    @ci_skip_case
    def test_random_forests(self):
        self.odps.delete_table(RANDOM_FORESTS_TEST_TABLE, if_exists=True)

        splited = self.df.split(0.6)

        labeled_data = splited[0].label_field("class")
        rf = RandomForests(tree_num=10)
        model = rf.train(labeled_data)
        print(model.segments[0])

        predicted = model.predict(splited[1])
        # persist is an operational node which will trigger execution of the flow
        predicted.persist(RANDOM_FORESTS_TEST_TABLE)

        print(confusion_matrix(predicted))
        print(rf_importance(labeled_data, model)._repr_html_())

    def test_mock_gbdt_lr(self):
        options.ml.dry_run = True

        splited = self.df.split(0.6)

        labeled_data = splited[0].label_field("class")

        gbdt_lr = GBDTLR(tree_count=500, min_leaf_sample_count=10).set_shrinkage(0.05)
        model = gbdt_lr.train(labeled_data)._add_case(self.gen_check_params_case(
            {'labelColName': 'class', 'modelName': MODEL_NAME, 'inputTableName': TEMP_TABLE_PREFIX + '_split',
             'maxLeafCount': '32', 'shrinkage': '0.05', 'featureSplitValueMaxSize': '500', 'featureRatio': '0.6',
             'testRatio': '0.0', 'randSeed': '0', 'sampleRatio': '0.6', 'treeCount': '500', 'metricType': '2',
             'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
             'minLeafSampleCount': '10', 'maxDepth': '11'}))
        model.persist(MODEL_NAME)

        gbdt_lr = GBDTLR(tree_count=500).set_shrinkage(0.05)
        model = gbdt_lr.train(labeled_data)._add_case(self.gen_check_params_case(
            {'labelColName': 'class', 'modelName': MODEL_NAME, 'inputTableName': TEMP_TABLE_PREFIX + '_split',
             'maxLeafCount': '32', 'shrinkage': '0.05', 'featureSplitValueMaxSize': '500', 'featureRatio': '0.6',
             'testRatio': '0.0', 'randSeed': '0', 'sampleRatio': '0.6', 'treeCount': '500', 'metricType': '2',
             'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
             'minLeafSampleCount': '500', 'maxDepth': '11'}))
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])
        predicted.persist(GBDT_LR_TEST_TABLE)

    @ci_skip_case
    def test_gbdt_lr(self):
        options.ml.dry_run = False
        self.delete_offline_model(MODEL_NAME)

        splited = self.df.split(0.6)

        labeled_data = splited[0].label_field("class")
        gbdt_lr = GBDTLR(tree_count=10, min_leaf_sample_count=10).set_shrinkage(0.05)
        model = gbdt_lr.train(labeled_data)
        model.persist(MODEL_NAME)

        print(gbdt_importance(labeled_data, model)._repr_html_())

    def test_mock_linear_svm(self):
        options.ml.dry_run = True
        splited = self.df.split(0.6)

        labeled_data = splited[0].label_field("class")
        svm = LinearSVM(epsilon=0.001).set_cost(1)
        model = svm.train(labeled_data)._add_case(self.gen_check_params_case(
            {'labelColName': 'class', 'positiveCost': '1', 'modelName': MODEL_NAME,
             'inputTableName': TEMP_TABLE_PREFIX + '_split', 'epsilon': '0.001', 'negativeCost': '1',
             'featureColNames': ','.join('a%02d' % i for i in range(1, 35))}))
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])
        predicted.persist(LINEAR_SVM_TEST_TABLE)

    def test_mock_naive_bayes(self):
        options.ml.dry_run = True

        splited = self.df.split(0.6)

        labeled_data = splited[0].label_field("class")
        naive_bayes = NaiveBayes()
        model = naive_bayes.train(labeled_data)._add_case(self.gen_check_params_case(
            {'isFeatureContinuous': ','.join(['1', ] * 34), 'labelColName': 'class',
             'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
             'modelName': MODEL_NAME, 'inputTableName': TEMP_TABLE_PREFIX + '_split'}))
        model.persist(MODEL_NAME)

        predicted = model.predict(splited[1])
        predicted.persist(NAIVE_BAYES_TEST_TABLE)

    def test_mock_knn(self):
        options.ml.dry_run = True

        splited = self.df.split(0.6)
        labeled_data = splited[0].label_field("class")
        algo = KNN(k=2)
        predicted = algo.transform(labeled_data, splited[1])._add_case(self.gen_check_params_case(
            {'trainFeatureColNames': ','.join('a%02d' % i for i in range(1, 35)),
             'appendColNames': ','.join('a%02d' % i for i in range(1, 35)) + ',class',
             'k': '2', 'trainLabelColName': 'class', 'outputTableName': KNN_TEST_TABLE,
             'trainTableName': TEMP_TABLE_PREFIX + '_split', 'predictTableName': TEMP_TABLE_PREFIX + '_split',
             'predictFeatureColNames': ','.join('a%02d' % i for i in range(1, 35))}))
        predicted.persist(KNN_TEST_TABLE)

    @ci_skip_case
    def test_logistic_regression(self):
        options.ml.dry_run = False

        splited = self.df.split(0.6)

        lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
        model = lr.train(splited[0])
        predicted = model.predict(splited[1])
        # persist is an operational node which will trigger execution of the flow
        predicted.persist(LR_TEST_TABLE, drop_table=True)

        expr = roc_curve(predicted, execute_now=False)
        fpr, tpr, thresh = expr.execute()
        print(roc_auc_score(predicted))
        assert len(fpr) == len(tpr) and len(thresh) == len(fpr)
