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

import logging
import sys

import pytest

from ....df import DataFrame
from ....config import options
from ...utils import TEMP_TABLE_PREFIX
from .. import *
from ..loader import load_classifiers
from ...tests.base import MLTestUtil, tn

logger = logging.getLogger(__name__)

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')

MODEL_NAME = tn('pyodps_test_out_model')


def register_algorithm():
    algo_def = XflowAlgorithmDef('MyNaiveBayes', project='algo_public', xflow_name='NaiveBayes')

    algo_def.add_port(PortDef.build_data_input()).add_port(PortDef.build_model_output())

    algo_def.add_param(ParamDef.build_input_table()).add_param(ParamDef.build_input_partitions())
    algo_def.add_param(ParamDef.build_model_name())
    algo_def.add_param(ParamDef.build_feature_col_names())
    algo_def.add_param(ParamDef.build_label_col_name())

    load_classifiers(algo_def, sys.modules[__name__])


@pytest.fixture
def utils(odps, tunnel):
    util = MLTestUtil(odps, tunnel)
    util.create_ionosphere(IONOSPHERE_TABLE)
    register_algorithm()
    return util


def test_custom_algo(odps, utils):
    options.ml.dry_run = True

    df = DataFrame(odps.get_table(IONOSPHERE_TABLE))
    splited = df.split(0.6)

    labeled_data = splited[0].label_field("class")
    naive_bayes = MyNaiveBayes()
    model = naive_bayes.train(labeled_data)._add_case(utils.gen_check_params_case(
            {'labelColName': 'class', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35)),
             'modelName': MODEL_NAME, 'inputTableName': TEMP_TABLE_PREFIX + '_split'}))
    model.persist(MODEL_NAME)

    predicted = model.predict(splited[1])
    predicted.persist(MODEL_NAME)
