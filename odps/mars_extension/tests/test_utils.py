# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import unittest

from odps.mars_extension.utils import rewrite_partition_predicate


class Test(unittest.TestCase):
    def testRewritePredicate(self):
        self.assertEqual(rewrite_partition_predicate('pt>20210125 & hh=08', ['pt', 'hh']).replace(' ', ''),
                         'pt>"20210125"&hh=="08"')
        self.assertEqual(rewrite_partition_predicate('pt>20210125 && hh=08', ['pt', 'hh']).replace(' ', ''),
                         'pt>"20210125"&hh=="08"')
        self.assertEqual(rewrite_partition_predicate('pt>20210125, w=SUN', ['pt', 'w']).replace(' ', ''),
                         'pt>"20210125"&w=="SUN"')
        with self.assertRaises(SyntaxError):
            rewrite_partition_predicate('pt>20210125 &&& w=SUN', ['pt', 'w'])
