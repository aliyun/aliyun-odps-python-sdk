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

import six

from odps.tests.core import TestBase, to_str
from odps.compat import unittest
from odps import compat
from odps.models import Resource, FileResource, TableResource, Schema
from odps import errors, types

FILE_CONTENT = '''This is a pyodps test file!
Enjoy it'''
OVERWRITE_FILE_CONTENT = '''This is an overwritten pyodps test file!
Enjoy it'''


class Test(TestBase):
    def testResources(self):
        self.assertIs(self.odps.get_project().resources, self.odps.get_project().resources)

        size = len(list(self.odps.list_resources()))
        self.assertGreaterEqual(size, 0)

        for resource in self.odps.list_resources():
            self.assertIsInstance(resource, Resource._get_cls(resource.type))

    def testResourceExists(self):
        non_exists_resource = 'a_non_exists_resource'
        self.assertFalse(self.odps.exist_resource(non_exists_resource))

    def testTableResource(self):
        test_table_name = 'pyodps_t_tmp_resource_table'
        schema = Schema.from_lists(['id', 'name'], ['string', 'string'])
        self.odps.delete_table(test_table_name, if_exists=True)
        self.odps.create_table(test_table_name, schema)

        resource_name = 'pyodps_t_tmp_table_resource'
        try:
            self.odps.delete_resource(resource_name)
        except errors.NoSuchObject:
            pass
        res = self.odps.create_resource(resource_name, 'table', table_name=test_table_name)
        self.assertIsInstance(res, TableResource)
        self.assertEqual(res.get_source_table().name, test_table_name)
        self.assertIsNone(res.get_source_table_partition())
        self.assertIs(res, self.odps.get_resource(resource_name))

        del res.parent[resource_name]  # delete from cache

        self.assertIsNot(res, self.odps.get_resource(resource_name))
        res = self.odps.get_resource(resource_name)
        self.assertIsInstance(res, TableResource)
        self.assertEqual(res.get_source_table().name, test_table_name)
        self.assertIsNone(res.get_source_table_partition())

        test_table_name = 'pyodps_t_tmp_resource_table'
        test_table_partition = 'pt=test'
        schema = Schema.from_lists(['id', 'name'], ['string', 'string'], ['pt', ], ['string', ])
        self.odps.delete_table(test_table_name, if_exists=True)
        table = self.odps.create_table(test_table_name, schema)
        table.create_partition(test_table_partition)

        resource_name = 'pyodps_t_tmp_table_resource'
        res = res.update(partition=test_table_partition)
        self.assertIsInstance(res, TableResource)
        self.assertEqual(res.get_source_table().name, test_table_name)
        self.assertEqual(str(res.get_source_table_partition()),
                         str(types.PartitionSpec(test_table_partition)))
        self.assertIs(res, self.odps.get_resource(resource_name))

        self.odps.delete_resource(resource_name)
        self.odps.delete_table(test_table_name)

    def testFileResource(self):
        resource_name = 'pyodps_t_tmp_file_resource'

        try:
            self.odps.delete_resource(resource_name)
        except errors.ODPSError:
            pass

        resource = self.odps.create_resource(resource_name, 'file', file_obj=FILE_CONTENT)
        self.assertIsInstance(resource, FileResource)

        with resource.open(mode='r') as fp:
            self.assertRaises(IOError, lambda: fp.write('sss'))
            self.assertRaises(IOError, lambda: fp.writelines(['sss\n']))

            self.assertIsInstance(fp.read(), six.text_type)

            fp.seek(0, compat.SEEK_END)
            size = fp.tell()
            fp.seek(0)
            self.assertEqual(fp._size, size)

            self.assertEqual(to_str(fp.read()), to_str(FILE_CONTENT))
            fp.seek(1)
            self.assertEqual(to_str(fp.read()), to_str(FILE_CONTENT[1:]))

            fp.seek(0)
            self.assertEqual(to_str(fp.readline()), to_str(FILE_CONTENT.split('\n', 1)[0]+'\n'))

            fp.seek(0)
            add_newline = lambda s: s if s.endswith('\n') else s+'\n'
            self.assertEqual([to_str(add_newline(l)) for l in fp],
                             [to_str(add_newline(l)) for l in FILE_CONTENT.split('\n')])

            self.assertFalse(fp._need_commit)
            self.assertTrue(fp._opened)

        self.assertFalse(fp._opened)
        self.assertIsNone(fp._fp)

        with resource.open(mode='w') as fp:
            self.assertRaises(IOError, fp.read)
            self.assertRaises(IOError, fp.readline)
            self.assertRaises(IOError, fp.readlines)

            fp.writelines([OVERWRITE_FILE_CONTENT]*2)

            self.assertTrue(fp._need_commit)

            size = fp._size

        with resource.open(mode='r+') as fp:
            self.assertEqual(to_str(fp.read()), to_str(OVERWRITE_FILE_CONTENT*2))

            self.assertEqual(size, fp._size)

            fp.seek(0)
            fp.write(FILE_CONTENT)
            fp.truncate()

            self.assertTrue(fp._need_commit)

        with resource.open(mode='a') as fp:
            self.assertRaises(IOError, fp.read)
            self.assertRaises(IOError, fp.readline)
            self.assertRaises(IOError, fp.readlines)

            fp.write(OVERWRITE_FILE_CONTENT)

            self.assertTrue(fp._need_commit)

        with resource.open(mode='a+') as fp:
            self.assertEqual(to_str(fp.read()), to_str(FILE_CONTENT+OVERWRITE_FILE_CONTENT))
            fp.seek(1)
            fp.truncate()
            self.assertTrue(fp._need_commit)

        fp = resource.open(mode='r')
        self.assertEqual(to_str(fp.read()), to_str('T'))
        fp.close()

        with resource.open(mode='w+') as fp:
            self.assertEqual(len(fp.read()), 0)
            fp.write(FILE_CONTENT)

        with resource.open(mode='r+') as fp:
            self.assertEqual(to_str(fp.read()), FILE_CONTENT)

        resource.update(file_obj='update')
        with resource.open(mode='rb') as fp:
            self.assertIsInstance(fp.read(), six.binary_type)
            fp.seek(0)
            self.assertEqual(to_str(fp.read()), to_str('update'))

        self.odps.delete_resource(resource_name)


if __name__ == '__main__':
    unittest.main()
