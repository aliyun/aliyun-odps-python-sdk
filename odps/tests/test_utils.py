#!/usr/bin/env python
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
from odps.utils import replace_sql_parameters


class Test(TestBase):
    def testReplaceSqlParameters(self):
        ns = {'test1': 'new_test1', 'test3': 'new_test3'}

        sql = 'select :test1 from dual where :test2 > 0 and f=:test3.abc'
        replaced_sql = replace_sql_parameters(sql, ns)

        expected = 'select new_test1 from dual where :test2 > 0 and f=new_test3.abc'
        self.assertEqual(expected, replaced_sql)


if __name__ == '__main__':
    unittest.main()
