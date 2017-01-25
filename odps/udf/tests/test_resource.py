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

import odps.tests.core as ut
from odps import distcache


class TestDistCache(ut.TestBase):
    
    def test_get_cache_archive(self):
        self.assertIsNone(distcache.get_cache_archive('x'))
        self.assertIsNone(distcache.get_cache_archive('x', 'y'))

    def test_get_cache_tabledesc(self):
        self.assertIsNone(distcache.get_cache_tabledesc('x'))

    def test_get_cache_tableinfo(self):
        self.assertIsNone(distcache.get_cache_tableinfo('x'))
