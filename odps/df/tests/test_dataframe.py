# -*- coding: utf-8 -*-
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

from odps.tests.core import TestBase
from odps.compat import unittest
from odps.models import Schema
from odps.df import DataFrame


class Test(TestBase):

    def setup(self):
        test_table_name = 'pyodps_test_dataframe'
        schema = Schema.from_lists(['id', 'name'], ['bigint', 'string'])

        self.odps.delete_table(test_table_name, if_exists=True)
        self.table = self.odps.create_table(test_table_name, schema)

        with self.table.open_writer() as w:
            w.write([[1, 'name1'], [2, 'name2'], [3, 'name3']])

    def teardown(self):
        self.table.drop()

    def testDataFrame(self):
        df = DataFrame(self.table)

        self.assertEqual(3, df.count().execute())
        self.assertEqual(1, df[df.name == 'name1'].count())

    def testHeadAndTail(self):
        df = DataFrame(self.table)

        self.assertEqual(1, len(df.head(1)))
        self.assertEqual(2, len(df.head(2)))
        self.assertEqual([3, 'name3'], list(df.tail(1)[0]))

        r = df[df.name == 'name2'].head(1)
        self.assertEqual(1, len(r))
        self.assertEqual([2, 'name2'], list(r[0]))

        self.assertRaises(NotImplementedError, lambda: df[df.name == 'name2'].tail(1))


if __name__ == '__main__':
    unittest.main()
