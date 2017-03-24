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

import functools

from odps import inter, options, utils
from odps.df import DataFrame
from odps.runner import RunnerContext, adapter_from_df
from odps.runner.tests.base import RunnerTestBase, tn

TEST_CONTEXT_ROOM = 'test_context_room'
TEST_LR_MODEL_NAME = tn('pyodps_test_lr_model')
TEST_TABLE_MODEL_NAME = tn('pyodps_ml_table_model')
IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')


class Test(RunnerTestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.ml_context = RunnerContext.instance()
        self.create_ionosphere(IONOSPHERE_TABLE)
        self.df = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))

    def test_construct(self):
        get_odps_tuple = lambda o: (o.account.access_id, o.account.secret_access_key, o.project, o.endpoint)

        with_odps = RunnerContext(self.odps)
        self.assertTupleEqual(get_odps_tuple(self.odps), get_odps_tuple(with_odps._odps))

        inter.teardown(TEST_CONTEXT_ROOM)
        inter.setup(*get_odps_tuple(self.odps), room=TEST_CONTEXT_ROOM)
        inter.enter(TEST_CONTEXT_ROOM)

        without_odps = RunnerContext()
        self.assertTupleEqual(get_odps_tuple(self.odps), get_odps_tuple(without_odps._odps))

        inter.teardown(TEST_CONTEXT_ROOM)

    def test_batch_run(self):
        options.runner.dry_run = False

        call_seq = []
        actions = []

        def action(df):
            call_seq.append('B')
            adapter = adapter_from_df(df)
            self.ml_context._run(adapter._bind_node, self.odps)
            call_seq.append('A')

        for idx in range(3):
            write_str = 'F%d' % idx
            def gen_fun(wobj):
                return lambda _: call_seq.append(wobj)

            f = gen_fun((write_str, 'U'))
            df_upper = self.mock_action(self.df, action=f)
            f = gen_fun((write_str, 'D'))
            df_lower = self.mock_action(df_upper, action=f)

            actions.append(functools.partial(action, df_lower))

        self.ml_context._batch_run_actions(actions, self.odps)

        self.assertListEqual(call_seq[:3], list('BBB'))
        self.assertListEqual(call_seq[-3:], list('AAA'))

        pairs = call_seq[3:9]
        for idx in range(3):
            write_str = 'F%d' % idx
            self.assertListEqual([p[1] for p in pairs if p[0] == write_str], list('UD'))
        for dir in 'UD':
            self.assertListEqual(sorted(p[0] for p in pairs if p[1] == dir), ['F0', 'F1', 'F2'])
