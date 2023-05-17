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

from ....df import DataFrame
from ....config import options
from ...clustering import *
from ...metrics import *
from ...tests.base import MLTestUtil, tn, ci_skip_case

import logging
logger = logging.getLogger(__name__)

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')
IONOSPHERE_CLUSTER_LABEL_TABLE = tn('pyodps_test_ml_iono_cluster_label')
IONOSPHERE_CLUSTER_MODEL = tn('pyodps_test_ml_kmeans_model')


@pytest.fixture
def utils(odps, tunnel):
    util = MLTestUtil(odps, tunnel)
    util.create_ionosphere(IONOSPHERE_TABLE)
    return util


@ci_skip_case
def test_kmeans(odps, utils):
    utils.delete_table(IONOSPHERE_CLUSTER_LABEL_TABLE)
    utils.delete_offline_model(IONOSPHERE_CLUSTER_MODEL)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    labeled, model = KMeans(center_count=3).transform(df.exclude_fields('class'))
    model.persist(IONOSPHERE_CLUSTER_MODEL, delay=True)
    pmml = model.load_pmml()
    print(pmml)
    eresult = calinhara_score(labeled, model)
    print(eresult)


def test_mock_kmeans(odps, utils):
    options.ml.dry_run = True

    df = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    labeled, model = KMeans(center_count=3).transform(df.exclude_fields('class'))
    labeled._add_case(utils.gen_check_params_case(
        {'inputTableName': IONOSPHERE_TABLE, 'centerCount': '3', 'distanceType': 'euclidean',
         'idxTableName': 'test_project_name.' + IONOSPHERE_CLUSTER_LABEL_TABLE, 'initCentersMethod': 'sample',
         'modelName': 'tmp_pyodps_k_means', 'appendColsIndex': ','.join('%d' % i for i in range(0, 35)),
         'selectedColNames': ','.join('a%02d' % i for i in range(1, 35)), 'loop': '100', 'accuracy': '0.0'}))
    labeled.persist(IONOSPHERE_CLUSTER_LABEL_TABLE, project='test_project_name')
