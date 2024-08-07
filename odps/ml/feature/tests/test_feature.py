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
import pytest

from .... import options, DataFrame
from ....tests.core import get_test_unique_name
from ...feature import *
from ...expr import PmmlModel
from ...tests.base import MLTestUtil, tn

TEST_LR_MODEL_NAME = tn('pyodps_test_lr_model')
IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
SELECT_FEATURE_OUTPUT_TABLE = tn('pyodps_test_ml_select_feature_output')


@pytest.fixture
def utils(odps, tunnel):
    util = MLTestUtil(odps, tunnel)
    util.model_name = TEST_LR_MODEL_NAME + get_test_unique_name(5)
    util.table_name = IONOSPHERE_TABLE + get_test_unique_name(5)

    util.create_test_pmml_model(util.model_name)
    util.create_ionosphere(util.table_name)
    util.df = DataFrame(odps.get_table(util.table_name)).label_field('class')
    util.model = PmmlModel(_source_data=odps.get_offline_model(util.model_name))
    options.ml.dry_run = True
    return util


def test_rf_importance(utils):
    rf_importance(utils.df, utils.model, core_num=1, core_mem=1024, _cases=utils.gen_check_params_case({
        'labelColName': 'class', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
        'modelName': utils.model_name, 'outputTableName': 'tmp_pyodps__rf_importance',
        'inputTableName': utils.table_name, 'coreNum': '1', 'memSizePerCore': '1024'
    }))


def test_gbdt_importance(utils):
    gbdt_importance(utils.df, utils.model, _cases=utils.gen_check_params_case({
        'labelColName': 'class', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
        'modelName': utils.model_name, 'outputTableName': 'tmp_pyodps__gbdt_importance',
        'inputTableName': utils.table_name
    }))


def test_regression_importance(utils):
    regression_importance(utils.df, utils.model, _cases=utils.gen_check_params_case({
        'labelColName': 'class', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
        'modelName': utils.model_name, 'outputTableName': 'tmp_pyodps__regression_importance',
        'inputTableName': utils.table_name
    }))


def test_select_features(odps, utils):
    output, importance = select_features(utils.df)
    output._add_case(utils.gen_check_params_case({
        'inputTable': odps.project + '.' + utils.table_name, 'labelCol': 'class', 'selectMethod': 'iv',
        'selectedCols': ','.join('a%02d' % i for i in range(1, 35)), 'topN': '10',
        'featImportanceTable': 'tmp_pyodps__select_features',
        'outputTable': SELECT_FEATURE_OUTPUT_TABLE
    }))
    output.persist(SELECT_FEATURE_OUTPUT_TABLE)
