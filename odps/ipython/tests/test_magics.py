#!/usr/bin/env python
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

from collections import namedtuple

from odps.config import options
from odps.tests.core import TestBase, tn
from odps.compat import unittest
from odps.ipython.magics import ODPSSql
from odps.df.backends.frame import ResultFrame
try:
    import IPython
    has_ipython = True
except ImportError:
    has_ipython = False


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.old_use_instance_tunnel = options.tunnel.use_instance_tunnel

    def tearDown(self):
        super(Test, self).tearDown()
        options.tunnel.use_instance_tunnel = self.old_use_instance_tunnel

    @unittest.skipIf(not has_ipython, 'Skipped when no IPython is detected.')
    def testExecuteSql(self):
        FakeShell = namedtuple('FakeShell', 'user_ns')
        user_ns = {}
        magic_class = ODPSSql(FakeShell(user_ns=user_ns))
        magic_class._odps = self.odps

        test1_sql = """res <<
        select 123 as result
        union
        --comment
        select 321 result;
        """
        result = magic_class.execute('', test1_sql)
        self.assertIsNone(result)
        self.assertIsNotNone(user_ns['res'])
        self.assertIsInstance(user_ns['res'], ResultFrame)

        self.assertListEqual(self._get_result(user_ns['res']), [[123], [321]])

        result = magic_class.execute("res2 << select '123'")
        self.assertIsNone(result)
        self.assertListEqual(self._get_result(user_ns['res2']), [['123']])

        test_table_name = tn('pyodps_t_test_sql_magic')
        test_content = [['line1'], ['line2']]
        self.odps.delete_table(test_table_name, if_exists=True)
        self.odps.create_table(test_table_name, 'col string', lifecycle=1)
        self.odps.write_table(test_table_name, test_content)

        options.tunnel.use_instance_tunnel = False
        result = magic_class.execute('select * from %s' % test_table_name)
        self.assertListEqual(self._get_result(result), test_content)

        options.tunnel.use_instance_tunnel = True
        result = magic_class.execute('select * from %s' % test_table_name)
        self.assertListEqual(self._get_result(result), test_content)

        result = magic_class.execute('show tables')
        self.assertTrue(len(result) > 0)

        table_name = tn('pyodps_test_magics_create_table_result')
        magic_class.execute('create table %s (col string) lifecycle 1' % table_name)
        magic_class.execute('drop table %s' % table_name)
