# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

from odps.tests.core import TestBase
from odps.compat import unittest


class Test(TestBase):
    def testOfflineModels(self):
        self.assertIs(self.odps.get_project().offline_models, self.odps.get_project().offline_models)
        size = len(list(self.odps.list_offline_models()))
        self.assertGreaterEqual(size, 0)

    def testInstanceExists(self):
        non_exists_offline_model = 'a_non_exists_offline_model'
        self.assertFalse(self.odps.exist_offline_model(non_exists_offline_model))


if __name__ == '__main__':
    unittest.main()
