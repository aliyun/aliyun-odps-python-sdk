# encoding: utf-8
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

from functools import partial

from odps.config import options
from odps.df import DataFrame
from odps.runner import set_retry_mode
from odps.runner.tests.base import RunnerTestBase, tn

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')


class Test(RunnerTestBase):
    def tearDown(self):
        super(Test, self).tearDown()
        set_retry_mode(False)

    def test_retry(self):
        rec_norm = []
        rec_err = []

        class CustomizedError(Exception):
            pass

        def _action(node, raises=False):
            if raises:
                rec_err.append(node.message)
                raise CustomizedError('Error')
            else:
                rec_norm.append(node.message)

        options.runner.retry_times = 3
        options.runner.dry_run = False
        self.create_ionosphere(IONOSPHERE_TABLE)
        df1 = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))

        set_retry_mode(False)
        df2 = self.mock_action(df1, 1, 'DF2', _action)
        df3 = self.mock_action(df2, 1, 'DF3', partial(_action, raises=True))
        self.assertRaises(CustomizedError, lambda: df3.persist('out_table'))
        self.assertEqual(rec_norm, ['DF2', ])
        self.assertEqual(rec_err, ['DF3', ] * 3)

        rec_norm = []
        rec_err = []
        set_retry_mode(True)
        df2 = self.mock_action(df1, 1, 'DF2', _action)
        df3 = self.mock_action(df2, 1, 'DF3', _action)
        df3.persist('out_table')
        self.assertEqual(rec_norm, ['DF3', ])
        self.assertEqual(rec_err, [])
