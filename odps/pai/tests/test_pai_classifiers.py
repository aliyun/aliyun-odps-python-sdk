# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function

import os
from odps.pai.context import PAIContext
from odps.pai.algorithms.classifiers import *
from odps.pai.metrics.classification import roc_curve
from odps.tests.core import TestBase

IONOSPHERE_TABLE = 'pyodps_test_pai_ionosphere'
IONOSPHERE_FILE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/data/ionosphere.txt'


class TestPaiClassifiers(TestBase):
    def setUp(self):
        super(TestPaiClassifiers, self).setUp()
        self.create_test_table()
        self.pai_context = PAIContext(self.odps)

    def tearDown(self):
        self.pai_context.cleanup()
        self.odps.run_sql("drop table if exists " + IONOSPHERE_TABLE)
        super(TestPaiClassifiers, self).tearDown()

    def create_test_table(self):
        fields = ','.join('a%02d double' % i for i in range(1, 35)) + ', class bigint'
        self.odps.execute_sql("drop table if exists " + IONOSPHERE_TABLE)
        self.odps.execute_sql("create table %s (%s)" % (IONOSPHERE_TABLE, fields))

        upload_ss = self.tunnel.create_upload_session(IONOSPHERE_TABLE)
        writer = upload_ss.open_record_writer(0)

        for line in open(IONOSPHERE_FILE_PATH, 'r'):
            rec = upload_ss.new_record()
            cols = [float(c) if rec._columns[i].type == 'double' else int(c) for i, c in enumerate(line.split(','))]
            [rec.set(i, val) for i, val in enumerate(cols)]
            writer.write(rec)
        writer.close()
        upload_ss.commit([0,])

    def test_mock_logistic_regression(self):
        self.pai_context.set_config("execution.mock", True)

        dataset = self.pai_context.odps_data(IONOSPHERE_TABLE)
        splited = dataset.split(0.6)

        labeled_data = splited[0].set_label_field("class")
        # labeled_data = dataset.set_label_field("category")
        lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
        model = lr.train(labeled_data)
        model.store_odps("testOutModel")

        predicted = model.predict(splited[1])
        # predicted = model.predict(dataset)
        predicted.store_odps("testOut")

    def test_mock_xgboost(self):
        self.pai_context.set_config("execution.mock", True)

        dataset = self.pai_context.odps_data(IONOSPHERE_TABLE)
        splited = dataset.split(0.6)

        labeled_data = splited[0].set_label_field("class")
        # labeled_data = dataset.set_label_field("category")
        xgboost = Xgboost(silent=1).set_eta(0.3)
        model = xgboost.train(labeled_data)
        model.store_odps("testOutModel")

        predicted = model.predict(splited[1])
        # predicted = model.predict(dataset)
        predicted.store_odps("testOut")

    def test_random_forests(self):
        self.pai_context.set_config("execution.mock", True)

        dataset = self.pai_context.odps_data(IONOSPHERE_TABLE)
        splited = dataset.split(0.6)

        labeled_data = splited[0].set_label_field("class")
        # labeled_data = dataset.set_label_field("category")
        rf = RandomForests(tree_num=10).set_max_tree_deep(10)
        model = rf.train(labeled_data)
        model.store_odps("testOutModel")

        predicted = model.predict(splited[1])
        # predicted = model.predict(dataset)
        predicted.store_odps("testOut")

    def test_gbdt_lr(self):
        self.pai_context.set_config("execution.mock", True)

        dataset = self.pai_context.odps_data(IONOSPHERE_TABLE)
        splited = dataset.split(0.6)

        labeled_data = splited[0].set_label_field("class")
        gbdt_lr = GBDTLR(tree_count=500).set_shrinkage(0.05)
        model = gbdt_lr.train(labeled_data)
        model.store_odps("testOutModel")

        predicted = model.predict(splited[1])
        predicted.store_odps("testOut")

    def test_linear_svm(self):
        self.pai_context.set_config("execution.mock", True)

        dataset = self.pai_context.odps_data(IONOSPHERE_TABLE)
        splited = dataset.split(0.6)

        labeled_data = splited[0].set_label_field("class")
        svm = LinearSVM(epsilon=0.001).set_cost(1)
        model = svm.train(labeled_data)
        model.store_odps("testOutModel")

        predicted = model.predict(splited[1])
        predicted.store_odps("testOut")

    def test_naive_bayes(self):
        self.pai_context.set_config("execution.mock", True)

        dataset = self.pai_context.odps_data(IONOSPHERE_TABLE)
        splited = dataset.split(0.6)

        labeled_data = splited[0].set_label_field("class")
        naive_bayes = NaiveBayes()
        model = naive_bayes.train(labeled_data)
        model.store_odps("testOutModel")

        predicted = model.predict(splited[1])
        predicted.store_odps("testOut")

    def test_logistic_regression(self):
        self.pai_context.set_config("execution.mock", False)

        dataset = self.pai_context.odps_data(IONOSPHERE_TABLE)
        splited = dataset.split(0.6)

        labeled_data = splited[0].set_label_field("class")
        lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
        model = lr.train(labeled_data)
        model.store_odps("testOutModel")

        predicted = model.predict(splited[1])
        # store_odps is an operational node which will trigger execution of the flow
        predicted.store_odps("testOut")

        fpr, tpr, thresh = roc_curve(predicted, 1, "class")
        assert len(fpr) == len(tpr) and len(thresh) == len(fpr)
