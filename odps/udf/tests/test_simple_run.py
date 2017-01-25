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

from odps.compat import unittest
from odps.udf.tests.udf_examples import *
from odps.udf.tools import runners
from odps.tests.core import TestBase


class TestSimpleRun(TestBase):
    
    def test_udf(self):
        self.assertEqual([2,3], runners.simple_run(Plus, [(1,1), (2,1)]))
        self.assertEqual([None], runners.simple_run(Plus, [(None,1) ]))

    def test_udaf(self):
        self.assertEqual([2], runners.simple_run(Avg, [(1,),(2,),(3,)]))

    def test_udtf(self):
        self.assertEqual(['a', 'b', 'ok'], runners.simple_run(Explode, [('a|b',),]))


class TestDistributedCache(TestBase):

    @unittest.skip("Not implemented yet")
    def test_get_cache_table(self):
        from odps import distcache
        self.assertEqual(distcache.get_cache_table('dual'), ('0',))


if __name__ == '__main__':
    unittest.main()
