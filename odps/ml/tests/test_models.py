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
from odps.ml.tests.base import MLTestBase, tn

TEST_LR_MODEL_NAME = tn('pyodps_test_lr_model')
TEST_TABLE_MODEL_NAME = tn('pyodps_table_model')
TEST_TEMP_TABLE_MODEL_NAME = tn(utils.TEMP_TABLE_PREFIX + 'table_model')
IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')


class Test(MLTestBase):
    def testNonTemp(self):
        model_comment = dict(key='value')

        model_table_name1 = ml_utils.build_model_table_name(TEST_TABLE_MODEL_NAME, 'st1')
        self.odps.execute_sql('drop table if exists {0}'.format(model_table_name1))
        self.odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
            model_table_name1, utils.escape_odps_string(json.dumps(model_comment))
        ))
        model_table_name2 = ml_utils.build_model_table_name(TEST_TABLE_MODEL_NAME, 'st2')
        self.odps.execute_sql('drop table if exists {0}'.format(model_table_name2))
        self.odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
            model_table_name2, utils.escape_odps_string(json.dumps(model_comment))
        ))
        self.assertIn(TEST_TABLE_MODEL_NAME, [tn.name for tn in self.odps.list_tables_model()])
        self.assertTrue(self.odps.exist_tables_model(TEST_TABLE_MODEL_NAME))

        tables_model = self.odps.get_tables_model(TEST_TABLE_MODEL_NAME)
        self.assertDictEqual(model_comment, tables_model.params)
        self.assertIn('st1', tables_model.tables)
        self.assertIn('st2', tables_model.tables)

        self.odps.delete_tables_model(TEST_TABLE_MODEL_NAME)
        self.assertFalse(self.odps.exist_tables_model(TEST_TABLE_MODEL_NAME))

    def testTemp(self):
        model_comment = dict(key='value')

        model_table_name1 = ml_utils.build_model_table_name(TEST_TEMP_TABLE_MODEL_NAME, 'st1')
        self.odps.execute_sql('drop table if exists {0}'.format(model_table_name1))
        self.odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
            model_table_name1, utils.escape_odps_string(json.dumps(model_comment))
        ))
        model_table_name2 = ml_utils.build_model_table_name(TEST_TEMP_TABLE_MODEL_NAME, 'st2')
        self.odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
            model_table_name2, utils.escape_odps_string(json.dumps(model_comment))
        ))
        self.assertIn(TEST_TEMP_TABLE_MODEL_NAME, [tn.name for tn in self.odps.list_tables_model()])
        self.assertIn(TEST_TEMP_TABLE_MODEL_NAME,
                      [tn.name for tn in self.odps.list_tables_model(prefix=ml_utils.TEMP_TABLE_PREFIX)])
        self.assertTrue(self.odps.exist_tables_model(TEST_TEMP_TABLE_MODEL_NAME))

        tables_model = self.odps.get_tables_model(TEST_TEMP_TABLE_MODEL_NAME)
        self.assertDictEqual(model_comment, tables_model.params)
        self.assertIn('st1', tables_model.tables)
        self.assertIn('st2', tables_model.tables)

        self.odps.delete_tables_model(TEST_TEMP_TABLE_MODEL_NAME)
        self.assertFalse(self.odps.exist_tables_model(TEST_TEMP_TABLE_MODEL_NAME))
