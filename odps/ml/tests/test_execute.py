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

from odps.tests.core import tn
from odps.df import DataFrame
from odps.df.backends.frame import ResultFrame
from odps.ml import PmmlModel
from odps.ml.tests.base import MLTestBase

IRIS_TABLE = tn('pyodps_test_ml_iris_execute')
IRIS_TEST_OFFLINE_MODEL = tn('pyodps_test_iris_model_exec')


class Test(MLTestBase):
    def testExecuteTable(self):
        self.create_iris(IRIS_TABLE)
        df = DataFrame(self.odps.get_table(IRIS_TABLE)).append_id()
        result = df.execute()
        self.assertIsInstance(result, ResultFrame)

    def testExecuteModel(self):
        from odps.ml import classifiers
        from odps.ml.expr.models.pmml import PmmlRegressionResult

        self.create_iris(IRIS_TABLE)
        df = DataFrame(self.odps.get_table(IRIS_TABLE)).roles(label='category')
        model = classifiers.LogisticRegression().train(df)
        result = model.execute()
        self.assertIsInstance(result, PmmlRegressionResult)

    def testExecuteAfterModelCreate(self):
        from odps.ml import classifiers
        from odps.ml.expr.models.pmml import PmmlRegressionResult

        self.create_iris(IRIS_TABLE)

        df = DataFrame(self.odps.get_table(IRIS_TABLE)).roles(label='category')
        model = classifiers.LogisticRegression().train(df)
        persisted = model.persist(IRIS_TEST_OFFLINE_MODEL, drop_model=True)
        result = persisted.execute()
        self.assertIsInstance(result, PmmlRegressionResult)

        expr = PmmlModel(self.odps.get_offline_model(IRIS_TEST_OFFLINE_MODEL))
        result = expr.execute()
        self.assertIsInstance(result, PmmlRegressionResult)
