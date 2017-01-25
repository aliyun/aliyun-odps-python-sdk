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

import json

from odps import utils
from odps.ml import utils as ml_utils
from odps.ml.models import TablesModel, PmmlModel, list_tables_model
from odps.ml.tests.base import MLTestBase, tn

TEST_LR_MODEL_NAME = tn('pyodps_test_lr_model')
TEST_TABLE_MODEL_NAME = tn('pyodps_table_model')
IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')


class TestBaseModel(MLTestBase):
    def test_odps_model(self):
        self.create_test_pmml_model(TEST_LR_MODEL_NAME)
        model = PmmlModel(self.odps.get_offline_model(TEST_LR_MODEL_NAME))
        self.assertEqual(model._bind_node.code_name, 'pmml_input')
        self.assertEqual(model._bind_node.parameters['modelName'], TEST_LR_MODEL_NAME)

    def test_tables_model(self):
        model_comment = dict(className='odps.ml.models.TablesModel', key='value')

        model_table_name1 = ''.join([ml_utils.TABLE_MODEL_PREFIX, TEST_TABLE_MODEL_NAME,
                                     ml_utils.TABLE_MODEL_SEPARATOR, 'st1'])
        self.odps.execute_sql('drop table if exists {0}'.format(model_table_name1))
        self.odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
            model_table_name1, utils.escape_odps_string(json.dumps(model_comment))
        ))
        model_table_name2 = ''.join([ml_utils.TABLE_MODEL_PREFIX, TEST_TABLE_MODEL_NAME,
                                     ml_utils.TABLE_MODEL_SEPARATOR, 'st2'])
        self.odps.execute_sql('drop table if exists {0}'.format(model_table_name2))
        self.odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
            model_table_name2, utils.escape_odps_string(json.dumps(model_comment))
        ))
        self.assertIn(TEST_TABLE_MODEL_NAME, list_tables_model(odps=self.odps))

        tables_model = TablesModel(self.odps, TEST_TABLE_MODEL_NAME)
        self.assertDictEqual(model_comment, tables_model._params)
        assert hasattr(tables_model, 'st1')
        assert hasattr(tables_model, 'st2')
