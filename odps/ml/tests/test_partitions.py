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

import pytest

from ...config import options
from ...df import DataFrame
from ..utils import TEMP_TABLE_PREFIX
from ..classifiers import *
from .base import MLTestUtil, tn

logger = logging.getLogger(__name__)

IONOSPHERE_TABLE_ONE_PART = tn(TEMP_TABLE_PREFIX + 'ionosphere_one_part')
IONOSPHERE_TABLE_TWO_PARTS = tn(TEMP_TABLE_PREFIX + 'ionosphere_two_parts')
TEST_OUTPUT_TABLE_NAME = tn(TEMP_TABLE_PREFIX + 'out_parted')

MODEL_NAME = tn('pyodps_test_out_model')


@pytest.fixture
def utils(odps, tunnel):
    return MLTestUtil(odps, tunnel)


def test_logistic_one_part_input(odps, utils):
    options.ml.dry_run = True

    utils.create_ionosphere_one_part(IONOSPHERE_TABLE_ONE_PART)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE_ONE_PART)) \
        .filter_parts('part=0/part=1').roles(label='class')

    lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
    model = lr.train(df)._add_case(utils.gen_check_params_case(
            {'labelColName': 'class', 'modelName': MODEL_NAME,
             'inputTableName': IONOSPHERE_TABLE_ONE_PART, 'epsilon': '0.001',
             'inputTablePartitions': 'part=0,part=1', 'regularizedLevel': '1', 'regularizedType': 'l1',
             'maxIter': '50', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35))}))
    model.persist(MODEL_NAME)


def test_logistic_partition_df(odps, utils):
    options.ml.dry_run = True

    utils.create_ionosphere_one_part(IONOSPHERE_TABLE_ONE_PART)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE_ONE_PART).get_partition("part=0")) \
        .roles(label='class')

    lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
    model = lr.train(df)._add_case(utils.gen_check_params_case(
        {'labelColName': 'class', 'modelName': MODEL_NAME,
         'inputTableName': IONOSPHERE_TABLE_ONE_PART, 'epsilon': '0.001',
         'inputTablePartitions': "part=0", 'regularizedLevel': '1', 'regularizedType': 'l1',
         'maxIter': '50', 'featureColNames': ','.join('a%02d' % i for i in range(1, 35))}))
    model.persist(MODEL_NAME)


def test_logistic_two_part_input(odps, utils):
    options.ml.dry_run = True

    utils.create_ionosphere_two_parts(IONOSPHERE_TABLE_TWO_PARTS)
    df = DataFrame(odps.get_table(IONOSPHERE_TABLE_TWO_PARTS)) \
        .filter_parts('part1=0,part2=0/part1=1,part2=0').roles(label='class')

    lr = LogisticRegression(epsilon=0.001).set_max_iter(50)
    model = lr.train(df)._add_case(utils.gen_check_params_case(
            {'labelColName': 'class', 'modelName': MODEL_NAME,
             'inputTableName': IONOSPHERE_TABLE_TWO_PARTS, 'epsilon': '0.001',
             'inputTablePartitions': 'part1=0/part2=0,part1=1/part2=0', 'regularizedLevel': '1',
             'regularizedType': 'l1', 'maxIter': '50',
             'featureColNames': ','.join('a%02d' % i for i in range(1, 35))}))
    model.persist(MODEL_NAME)

    predicted = model.predict(df)._add_case(utils.gen_check_params_case(
            {'modelName': MODEL_NAME, 'appendColNames': ','.join('a%02d' % i for i in range(1, 35)) + ',class',
             'inputTableName': IONOSPHERE_TABLE_TWO_PARTS,
             'inputTablePartitions': 'part1=0/part2=0,part1=1/part2=0',
             'outputTableName': TEST_OUTPUT_TABLE_NAME}))
    predicted.persist(TEST_OUTPUT_TABLE_NAME)
