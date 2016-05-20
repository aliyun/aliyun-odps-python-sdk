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

from odps.compat import unittest
from odps.udf import get_execution_context
from odps.udf import ExecutionContext
from odps.tests.core import TestBase


class TestExecutionContext(TestBase):
    
    def test_get_counter(self):
        ctx = get_execution_context()
        counters = ctx.get_counters()
        c = counters.get_counter("test_group", "test")
        self.assertEqual("test", c.get_name())
        self.assertEqual(0, c.get_value())

        c.increment(1)
        self.assertEqual(1, c.get_value())
    
    def test_single_counters(self):
        ctx1 = get_execution_context()
        counters = ctx1.get_counters()
        counters.get_counter("test_group", "test")
        self.assertEqual(1, counters.size())
        
        ctx2 = ExecutionContext()
        counters2 = ctx2.get_counters()
        counters2.get_counter("test_group2", "test2")
        self.assertEqual(2, counters2.size())

if __name__ == '__main__':
    unittest.main()
