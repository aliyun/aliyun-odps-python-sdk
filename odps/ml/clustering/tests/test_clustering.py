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
from odps.config import options
from odps.ml.clustering import *
from odps.ml.metrics import *
from odps.ml.tests.base import MLTestBase, tn, ci_skip_case

import logging
logger = logging.getLogger(__name__)

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
IONOSPHERE_CLUSTER_LABEL_TABLE = tn('pyodps_test_ml_iono_cluster_label')
IONOSPHERE_CLUSTER_MODEL = tn('pyodps_test_ml_kmeans_model')


class TestMLClustering(MLTestBase):
    def setUp(self):
        super(TestMLClustering, self).setUp()
        self.create_ionosphere(IONOSPHERE_TABLE)

    @ci_skip_case
    def test_kmeans(self):
        self.delete_table(IONOSPHERE_CLUSTER_LABEL_TABLE)
        self.delete_offline_model(IONOSPHERE_CLUSTER_MODEL)
        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))
        labeled, model = KMeans(center_count=3).transform(df.exclude_fields('class'))
        model.persist(IONOSPHERE_CLUSTER_MODEL, delay=True)
        pmml = model.load_pmml()
        print(pmml)
        eresult = calinhara_score(labeled, model)
        print(eresult)

    def test_mock_kmeans(self):
        options.ml.dry_run = True
        self.maxDiff = None

        df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))
        labeled, model = KMeans(center_count=3).transform(df.exclude_fields('class'))
        labeled._add_case(self.gen_check_params_case(
            {'inputTableName': IONOSPHERE_TABLE, 'centerCount': '3', 'distanceType': 'euclidean',
             'idxTableName': 'test_project_name.' + IONOSPHERE_CLUSTER_LABEL_TABLE, 'initCentersMethod': 'sample',
             'modelName': 'tmp_pyodps_k_means', 'appendColsIndex': ','.join('%d' % i for i in range(0, 35)),
             'selectedColNames': ','.join('a%02d' % i for i in range(1, 35)), 'loop': '100', 'accuracy': '0.0'}))
        labeled.persist(IONOSPHERE_CLUSTER_LABEL_TABLE, project='test_project_name')
