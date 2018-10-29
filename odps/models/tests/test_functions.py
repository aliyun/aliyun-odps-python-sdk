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

import itertools

from odps.tests.core import TestBase, to_str, tn
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
            try:
                self.assertTrue(all(map(lambda r: type(r) != Resource, function.resources)))
            except errors.ODPSError:
                continue

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

    def testCreateDeleteUpdateFunction(self):
        secondary_project = self.config.get('test', 'secondary_project')
        secondary_user = self.config.get('test', 'secondary_user')

        test_resource_name = tn('pyodps_t_tmp_test_function_resource') + '.py'
        test_resource_name2 = tn('pyodps_t_tmp_test_function_resource2') + '.py'
        test_resource_name3 = tn('pyodps_t_tmp_test_function_resource3') + '.py'
        test_function_name = tn('pyodps_t_tmp_test_function')
        test_function_name3 = tn('pyodps_t_tmp_test_function3')

        try:
            self.odps.delete_resource(test_resource_name)
        except errors.NoSuchObject:
            pass
        try:
            self.odps.delete_resource(test_resource_name2)
        except errors.NoSuchObject:
            pass
        try:
            self.odps.delete_function(test_function_name)
        except errors.NoSuchObject:
            pass
        try:
            self.odps.delete_function(test_function_name3)
        except errors.NoSuchObject:
            pass

        if secondary_project:
            try:
                self.odps.delete_resource(test_resource_name3, project=secondary_project)
            except errors.NoSuchObject:
                pass

        test_resource = self.odps.create_resource(
            test_resource_name, 'py', file_obj=FILE_CONTENT)

        test_function = self.odps.create_function(
            test_function_name,
            class_type=test_resource_name.split('.', 1)[0]+'.MyPlus',
            resources=[test_resource])

        self.assertIsNotNone(test_function.name)
        self.assertIsNotNone(test_function.owner)
        self.assertIsNotNone(test_function.creation_time)
        self.assertIsNotNone(test_function.class_type)
        self.assertEqual(len(test_function.resources), 1)

        with self.odps.open_resource(name=test_resource_name, mode='r') as fp:
            self.assertEqual(to_str(fp.read()), to_str(FILE_CONTENT))

        self.assertNotEqual(test_function.owner, secondary_user)

        test_resource2 = self.odps.create_resource(
            test_resource_name2, 'file', file_obj='Hello World'
        )
        test_function.resources.append(test_resource2)
        test_function.owner = secondary_user
        test_function.update()

        test_function_id = id(test_function)
        del test_function.project.functions[test_function.name]
        test_function = self.odps.get_function(test_function_name)
        self.assertNotEqual(test_function_id, id(test_function))
        self.assertEqual(len(test_function.resources), 2)
        self.assertEqual(test_function.owner, secondary_user)

        test_resource3 = None
        test_function3 = None
        if secondary_project:
            test_resource3 = self.odps.create_resource(
                test_resource_name3, 'py', file_obj=FILE_CONTENT, project=secondary_project)

            test_function3 = self.odps.create_function(
                test_function_name3,
                class_type=test_resource_name3.split('.', 1)[0]+'.MyPlus',
                resources=[test_resource3])

            self.assertEqual(test_function3.name, test_function_name3)
            self.assertIsNotNone(test_function3.owner)
            self.assertIsNotNone(test_function3.creation_time)
            self.assertIsNotNone(test_function3.class_type)
            self.assertEqual(len(test_function3.resources), 1)
            self.assertEqual(test_function3.resources[0].name, test_resource_name3)
            self.assertEqual(test_function3.resources[0].project, secondary_project)

        test_resource.drop()
        test_resource2.drop()
        if test_resource3:
            test_resource3.drop()
        if test_function3:
            test_function3.drop()
        test_function.drop()


if __name__ == '__main__':
    unittest.main()
