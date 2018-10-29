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

from odps import options, DataFrame
from odps.ml.feature import *
from odps.ml.expr import PmmlModel
from odps.ml.tests.base import MLTestBase, tn

TEST_LR_MODEL_NAME = tn('pyodps_test_lr_model')
IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
SELECT_FEATURE_OUTPUT_TABLE = tn('pyodps_test_ml_select_feature_output')


class Test(MLTestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.create_test_pmml_model(TEST_LR_MODEL_NAME)
        self.create_ionosphere(IONOSPHERE_TABLE)
        self.df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).label_field('class')
        self.model = PmmlModel(_source_data=self.odps.get_offline_model(TEST_LR_MODEL_NAME))
        options.ml.dry_run = True

    def test_rf_importance(self):
        rf_importance(self.df, self.model, core_num=1, core_mem=1024, _cases=self.gen_check_params_case({
            'labelColName': 'class', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
            'modelName': TEST_LR_MODEL_NAME, 'outputTableName': 'tmp_pyodps__rf_importance',
            'inputTableName': IONOSPHERE_TABLE, 'coreNum': '1', 'memSizePerCore': '1024'
        }))

    def test_gbdt_importance(self):
        gbdt_importance(self.df, self.model, _cases=self.gen_check_params_case({
            'labelColName': 'class', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
            'modelName': TEST_LR_MODEL_NAME, 'outputTableName': 'tmp_pyodps__gbdt_importance',
            'inputTableName': IONOSPHERE_TABLE
        }))

    def test_regression_importance(self):
        regression_importance(self.df, self.model, _cases=self.gen_check_params_case({
            'labelColName': 'class', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
            'modelName': TEST_LR_MODEL_NAME, 'outputTableName': 'tmp_pyodps__regression_importance',
            'inputTableName': IONOSPHERE_TABLE
        }))

    def test_select_features(self):
        output, importance = select_features(self.df)
        output._add_case(self.gen_check_params_case({
            'inputTable': self.odps.project + '.' + IONOSPHERE_TABLE, 'labelCol': 'class', 'selectMethod': 'iv',
            'selectedCols': ','.join('a%02d' % i for i in range(1, 35)), 'topN': '10',
            'featImportanceTable': 'tmp_pyodps__select_features',
            'outputTable': SELECT_FEATURE_OUTPUT_TABLE
        }))
        output.persist(SELECT_FEATURE_OUTPUT_TABLE)
