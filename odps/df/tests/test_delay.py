# -*- coding: utf-8 -*-
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
import unittest

from odps.tests.core import TestBase, tn
from odps.models import Schema
from odps.df import DataFrame, Delay


class Test(TestBase):

    def setup(self):
        test_table_name = tn('pyodps_test_delay')
        schema = Schema.from_lists(['id', 'name', 'value'], ['bigint', 'string', 'bigint'])

        self.odps.delete_table(test_table_name, if_exists=True)
        self.table = self.odps.create_table(test_table_name, schema)
        self.data = [[1, 'name1', 1], [2, 'name1', 2], [3, 'name1', 3],
                     [4, 'name2', 1], [5, 'name2', 2], [6, 'name2', 3]]

        with self.table.open_writer() as w:
            w.write(self.data)

        self.df = DataFrame(self.table)

    def testSyncExecute(self):
        delay = Delay()
        filtered = self.df[self.df.id > 0].cache()
        sub_futures = [filtered[filtered.value == i].execute(delay=delay) for i in range(1, 3)]
        delay.execute(timeout=10 * 60)

        self.assertTrue(all(f.done() for f in sub_futures))
        for i in range(1, 3):
            self.assertEqual(self._get_result(sub_futures[i - 1].result()), [d for d in self.data if d[2] == i])

        # execute on executed delay
        sub_future = filtered[filtered.value == 3].execute(delay=delay)
        delay.execute(timeout=10 * 60)
        self.assertTrue(sub_future.done())
        self.assertEqual(self._get_result(sub_future.result()), [d for d in self.data if d[2] == 3])

    def testAsyncExecute(self):
        def make_filter(df, cnt):
            def waiter(val, c):
                import time
                time.sleep(5 * c)
                return val

            f_df = df[df.value == cnt]
            return f_df[f_df.exclude('value'), f_df.value.map(functools.partial(waiter, cnt))]

        delay = Delay()
        filtered = self.df[self.df.id > 0].cache()
        sub_futures = [make_filter(filtered, i).execute(delay=delay) for i in range(1, 4)]
        future = delay.execute(async_=True, n_parallel=3)
        self.assertRaises(RuntimeError, lambda: delay.execute())

        for i in range(1, 4):
            self.assertFalse(future.done())
            self.assertFalse(any(f.done() for f in sub_futures[i - 1:]))
            self.assertTrue(all(f.done() for f in sub_futures[:i - 1]))
            self.assertEqual(self._get_result(sub_futures[i - 1].result()), [d for d in self.data if d[2] == i])
        self.assertTrue(all(f.done() for f in sub_futures))
        future.result(timeout=10 * 60)
        self.assertTrue(future.done())

    def testPersistExecute(self):
        delay = Delay()
        filtered = self.df[self.df.id > 0].cache()

        persist_table_name = tn('pyodps_test_delay_persist')
        schema = Schema.from_lists(['id', 'name', 'value'], ['bigint', 'string', 'bigint'],
                                   ['pt', 'ds'], ['string', 'string'])
        self.odps.delete_table(persist_table_name, if_exists=True)
        self.odps.create_table(persist_table_name, schema)

        future1 = filtered[filtered.value > 2].persist(persist_table_name, partition='pt=a,ds=d1', delay=delay)
        future2 = filtered[filtered.value < 2].persist(persist_table_name, partition='pt=a,ds=d2', delay=delay)

        delay.execute()
        df1 = future1.result()
        df2 = future2.result()

        self.assertEqual([c.lhs.name for c in df1.predicate.children()], ['pt', 'ds'])
        result1 = self._get_result(df1.execute())
        self.assertEqual([r[:-2] for r in result1], [d for d in self.data if d[2] > 2])
        self.assertEqual([c.lhs.name for c in df2.predicate.children()], ['pt', 'ds'])
        result2 = self._get_result(df2.execute())
        self.assertEqual([r[:-2] for r in result2], [d for d in self.data if d[2] < 2])

if __name__ == '__main__':
    unittest.main()
