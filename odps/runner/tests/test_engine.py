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

from odps.runner.tests.base import RunnerTestBase
from odps.runner.engine import BaseNodeEngine


class Test(RunnerTestBase):
    def test_param_format(self):
        self.assertEqual(BaseNodeEngine._format_value(None), None)
        self.assertEqual(BaseNodeEngine._format_value(True), 'true')
        self.assertEqual(BaseNodeEngine._format_value(False), 'false')
        self.assertEqual(BaseNodeEngine._format_value(''), '')
        self.assertEqual(BaseNodeEngine._format_value([]), None)
        self.assertEqual(BaseNodeEngine._format_value([1, 2, 3, 4]), '1,2,3,4')
        self.assertEqual(BaseNodeEngine._format_value(set([1, 2, 3, 4])), '1,2,3,4')
        self.assertEqual(BaseNodeEngine._format_value(12), '12')
