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

from ...tests.core import tn
from ...df import DataFrame
from ...df.backends.frame import ResultFrame
from .. import PmmlModel
from ..tests.base import MLTestUtil

IRIS_TABLE = tn('pyodps_test_ml_iris_execute')
IRIS_TEST_OFFLINE_MODEL = tn('pyodps_test_iris_model_exec')


@pytest.fixture
def utils(odps, tunnel):
    return MLTestUtil(odps, tunnel)


def test_execute_table(odps, utils):
    utils.create_iris(IRIS_TABLE)
    df = DataFrame(odps.get_table(IRIS_TABLE)).append_id()
    result = df.execute()
    assert isinstance(result, ResultFrame)


def test_execute_model(odps, utils):
    from .. import classifiers
    from ..expr.models.pmml import PmmlRegressionResult

    utils.create_iris(IRIS_TABLE)
    df = DataFrame(odps.get_table(IRIS_TABLE)).roles(label='category')
    model = classifiers.LogisticRegression().train(df)
    result = model.execute()
    assert isinstance(result, PmmlRegressionResult)


def test_execute_after_model_create(odps, utils):
    from .. import classifiers
    from ..expr.models.pmml import PmmlRegressionResult

    utils.create_iris(IRIS_TABLE)

    df = DataFrame(odps.get_table(IRIS_TABLE)).roles(label='category')
    model = classifiers.LogisticRegression().train(df)
    persisted = model.persist(IRIS_TEST_OFFLINE_MODEL, drop_model=True)
    result = persisted.execute()
    assert isinstance(result, PmmlRegressionResult)

    expr = PmmlModel(odps.get_offline_model(IRIS_TEST_OFFLINE_MODEL))
    result = expr.execute()
    assert isinstance(result, PmmlRegressionResult)
