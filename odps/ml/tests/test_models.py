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

import json

import pytest

from ... import utils as odps_utils
from ...tests.core import get_test_unique_name
from .. import utils as ml_utils
from .base import MLTestUtil, tn

TEST_LR_MODEL_NAME = tn('pyodps_test_lr_model' + get_test_unique_name(5))
TEST_TABLE_MODEL_NAME = tn('pyodps_table_model' + get_test_unique_name(5))
TEST_TEMP_TABLE_MODEL_NAME = tn(odps_utils.TEMP_TABLE_PREFIX + 'table_model')
IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')

pytestmark = pytest.mark.skip("cases might contribute to timeouts")


@pytest.fixture
def utils(odps, tunnel):
    return MLTestUtil(odps, tunnel)


def test_non_temp(odps, utils):
    model_comment = dict(key='value')

    model_table_name1 = ml_utils.build_model_table_name(TEST_TABLE_MODEL_NAME, 'st1')
    odps.execute_sql('drop table if exists {0}'.format(model_table_name1))
    odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
        model_table_name1, odps_utils.escape_odps_string(json.dumps(model_comment))
    ))
    model_table_name2 = ml_utils.build_model_table_name(TEST_TABLE_MODEL_NAME, 'st2')
    odps.execute_sql('drop table if exists {0}'.format(model_table_name2))
    odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
        model_table_name2, odps_utils.escape_odps_string(json.dumps(model_comment))
    ))
    assert TEST_TABLE_MODEL_NAME in [tn.name for tn in odps.list_tables_model()]
    assert odps.exist_tables_model(TEST_TABLE_MODEL_NAME) is True

    tables_model = odps.get_tables_model(TEST_TABLE_MODEL_NAME)
    assert model_comment == tables_model.params
    assert 'st1' in tables_model.tables
    assert 'st2' in tables_model.tables

    odps.delete_tables_model(TEST_TABLE_MODEL_NAME)
    assert odps.exist_tables_model(TEST_TABLE_MODEL_NAME) is False


def test_temp(odps, utils):
    model_comment = dict(key='value')

    model_table_name1 = ml_utils.build_model_table_name(TEST_TEMP_TABLE_MODEL_NAME, 'st1')
    odps.execute_sql('drop table if exists {0}'.format(model_table_name1))
    odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
        model_table_name1, odps_utils.escape_odps_string(json.dumps(model_comment))
    ))
    model_table_name2 = ml_utils.build_model_table_name(TEST_TEMP_TABLE_MODEL_NAME, 'st2')
    odps.execute_sql('create table if not exists {0} (col1 string) comment \'{1}\' lifecycle 1'.format(
        model_table_name2, odps_utils.escape_odps_string(json.dumps(model_comment))
    ))
    assert TEST_TEMP_TABLE_MODEL_NAME in [tn.name for tn in odps.list_tables_model()]
    assert TEST_TEMP_TABLE_MODEL_NAME in [tn.name for tn in odps.list_tables_model(prefix=ml_utils.TEMP_TABLE_PREFIX)]
    assert odps.exist_tables_model(TEST_TEMP_TABLE_MODEL_NAME) is True

    tables_model = odps.get_tables_model(TEST_TEMP_TABLE_MODEL_NAME)
    assert model_comment == tables_model.params
    assert 'st1' in tables_model.tables
    assert 'st2' in tables_model.tables

    odps.delete_tables_model(TEST_TEMP_TABLE_MODEL_NAME)
    assert odps.exist_tables_model(TEST_TEMP_TABLE_MODEL_NAME) is False
