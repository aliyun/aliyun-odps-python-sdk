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

from odps.df import DataFrame
from odps.ml.classifiers import LogisticRegression
from odps.ml.cross_validation import cross_val_score
from odps.ml.tests.base import MLTestBase, tn, ci_skip_case

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')


class TestCrossValidation(MLTestBase):
    def setUp(self):
        super(TestCrossValidation, self).setUp()
        self.create_ionosphere(IONOSPHERE_TABLE)
        self.df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE)).roles(label='class')

    def tearDown(self):
        super(TestCrossValidation, self).tearDown()

    @ci_skip_case
    def test_logistic_regression(self):
        lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
        print(cross_val_score(lr, self.df))
