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

import itertools

from odps.tests.core import TestBase, to_str
from odps.compat import unittest
from odps.models import Resource
from odps import errors

FILE_CONTENT = '''from odps.udf import annotate

@annotate("bigint,bigint->bigint")
class MyPlus(object):

   def evaluate(self, arg0, arg1):
       if None in (arg0, arg1):
           return None
       return arg0 + arg1
'''


class Test(TestBase):
    def testFunctions(self):
        self.assertIs(self.odps.get_project().functions, self.odps.get_project().functions)

        functions_model = self.odps.get_project().functions

        functions = list(functions_model.iterate(maxitems=400))
        size = len(functions)
        self.assertGreaterEqual(size, 0)

        for i, function in zip(itertools.count(1), functions):
            if i > 50:
                break
            self.assertTrue(all(map(lambda r: type(r) != Resource, function.resources)))

    def testFunctionExists(self):
        non_exists_function = 'a_non_exists_function'
        self.assertFalse(self.odps.exist_function(non_exists_function))

    def testFunction(self):
        functions = self.odps.list_functions()
        try:
            function = next(functions)
        except StopIteration:
            return

        self.assertIs(function, self.odps.get_function(function.name))

        self.assertIsNotNone(function._getattr('name'))
        self.assertIsNotNone(function._getattr('owner'))
        self.assertIsNotNone(function._getattr('creation_time'))
        self.assertIsNotNone(function._getattr('class_type'))
        self.assertIsNotNone(function._getattr('_resources'))

    def testCreateDeleteFunction(self):
        test_resource_name = 'pyodps_t_tmp_test_function_resource.py'
        test_function_name = 'pyodps_t_tmp_test_function'

        try:
            self.odps.delete_resource(test_resource_name)
        except errors.NoSuchObject:
            pass
        try:
            self.odps.delete_function(test_function_name)
        except errors.NoSuchObject:
            pass

        test_resource = self.odps.create_resource(
            test_resource_name, 'py', file_obj=FILE_CONTENT)

        test_function = self.odps.create_function(
            test_function_name,
            class_type=test_resource_name.split('.', 1)[0]+'.MyPlus',
            resources=[test_resource,])

        self.assertIsNotNone(test_function.name)
        self.assertIsNotNone(test_function.owner)
        self.assertIsNotNone(test_function.creation_time)
        self.assertIsNotNone(test_function.class_type)
        self.assertEqual(len(test_function.resources), 1)

        with self.odps.open_resource(name=test_resource_name, mode='r') as fp:
            self.assertEqual(to_str(fp.read()), to_str(FILE_CONTENT))

        test_resource.drop()
        test_function.drop()


if __name__ == '__main__':
    unittest.main()
