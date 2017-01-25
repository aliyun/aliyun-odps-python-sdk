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

from odps.config import options
from odps.df import DataFrame
from odps.ml.recommend import *
from odps.ml.tests.base import MLTestBase, tn

USER_ITEM_TABLE = tn('pyodps_test_ml_user_item_table')
USER_ITEM_PAYLOAD_TABLE = tn('pyodps_test_ml_user_item_payload_table')
ASSOC_RESULT_TABLE = tn('pyodps_test_ml_assoc_result')
ETREC_RESULT_TABLE = tn('pyodps_test_ml_etrec_result')
ALSCF_RESULT_TABLE = tn('pyodps_test_ml_als_cf_result')
ALSCF_RECOMMEND_TABLE = tn('pyodps_test_ml_als_cf_rec')
SVDCF_RESULT_TABLE = tn('pyodps_test_ml_svd_cf_result')
SVDCF_RECOMMEND_TABLE = tn('pyodps_test_ml_svd_cf_rec')


class TestRecommend(MLTestBase):
    def setUp(self):
        super(TestRecommend, self).setUp()
        options.runner.dry_run = True

    def test_etrec(self):
        self.create_user_item_table(USER_ITEM_PAYLOAD_TABLE, mode='agg')
        ds = DataFrame(self.odps.get_table(USER_ITEM_PAYLOAD_TABLE)) \
            .roles(rec_user_id='user', rec_item='item', rec_payload='payload')
        result = Etrec().transform(ds)._add_case(self.gen_check_params_case(
            {'itemDelimiter': ',', 'maxUserBehavior': '500', 'weight': '1.0',
             'inputTableName': USER_ITEM_PAYLOAD_TABLE, 'minUserBehavior': '2', 'topN': '2000',
             'outputTableName': ETREC_RESULT_TABLE, 'kvDelimiter': ':', 'inputTableFormat': 'user-item',
             'operator': 'add', 'alpha': '0.5', 'similarityType': 'wbcosine', 'selectedColNames': 'user,item,payload'}))
        result.persist(ETREC_RESULT_TABLE)
