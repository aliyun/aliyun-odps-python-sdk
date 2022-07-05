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

import zipfile

from odps.tests.core import TestBase, to_str, tn
from odps.compat import unittest, six, ConfigParser
from odps import compat
from odps.models import Resource, FileResource, TableResource, VolumeArchiveResource, \
    VolumeFileResource, Schema
from odps import errors, types

FILE_CONTENT = to_str("""
Proudly swept the rain by the cliffs
As it glided through the trees
Still following ever the bud
The ahihi lehua of the vale
""")
OVERWRITE_FILE_CONTENT = to_str("""
Farewell to thee, farewell to thee
The charming one who dwells in the shaded bowers
One fond embrace,
'Ere I depart
Until we meet again
Sweet memories come back to me
Bringing fresh remembrances
Of the past
Dearest one, yes, you are mine own
From you, true love shall never depart
""")


class Test(TestBase):
    def testResources(self):
        self.assertIs(self.odps.get_project().resources, self.odps.get_project().resources)

        size = len(list(self.odps.list_resources()))
        self.assertGreaterEqual(size, 0)

        for resource in self.odps.list_resources():
            self.assertIsInstance(resource, Resource._get_cls(resource.type))

        self.assertRaises(TypeError, lambda: self.odps.create_resource(
            'test_error', 'py', resource=['uvw']))

    def testResourceExists(self):
        non_exists_resource = 'a_non_exists_resource'
        self.assertFalse(self.odps.exist_resource(non_exists_resource))

    def testTableResource(self):
        try:
            secondary_project = self.config.get('test', 'secondary_project')
        except ConfigParser.NoOptionError:
            secondary_project = None

        test_table_name = tn('pyodps_t_tmp_resource_table')
        schema = Schema.from_lists(['id', 'name'], ['string', 'string'])
        self.odps.delete_table(test_table_name, if_exists=True)
        self.odps.create_table(test_table_name, schema)
        if secondary_project:
            self.odps.delete_table(test_table_name, if_exists=True, project=secondary_project)
            self.odps.create_table(test_table_name, schema, project=secondary_project)

        resource_name = tn('pyodps_t_tmp_table_resource')
        try:
            self.odps.delete_resource(resource_name)
        except errors.NoSuchObject:
            pass
        res = self.odps.create_resource(resource_name, 'table', table_name=test_table_name)
        self.assertIsInstance(res, TableResource)
        self.assertEqual(res.get_source_table().name, test_table_name)
        self.assertEqual(res.table.name, test_table_name)
        self.assertIsNone(res.get_source_table_partition())
        self.assertIs(res, self.odps.get_resource(resource_name))

        with res.open_writer() as writer:
            writer.write([0, FILE_CONTENT])
        with res.open_reader() as reader:
            rec = list(reader)[0]
            self.assertEqual(rec[1], FILE_CONTENT)

        del res.parent[resource_name]  # delete from cache

        self.assertIsNot(res, self.odps.get_resource(resource_name))
        res = self.odps.get_resource(resource_name)
        self.assertIsInstance(res, TableResource)
        self.assertEqual(res.get_source_table().name, test_table_name)
        self.assertIsNone(res.get_source_table_partition())

        test_table_partition = 'pt=test,sec=1'
        schema = Schema.from_lists(['id', 'name'], ['string', 'string'], ['pt', 'sec'], ['string', 'bigint'])
        self.odps.delete_table(test_table_name, if_exists=True)
        table = self.odps.create_table(test_table_name, schema)
        table.create_partition(test_table_partition)

        res = res.update(partition=test_table_partition)
        self.assertIsInstance(res, TableResource)
        self.assertEqual(res.get_source_table().name, test_table_name)
        self.assertEqual(res.table.name, test_table_name)
        self.assertEqual(str(res.get_source_table_partition()),
                         str(types.PartitionSpec(test_table_partition)))
        self.assertEqual(str(res.partition.spec),
                         str(types.PartitionSpec(test_table_partition)))
        self.assertIs(res, self.odps.get_resource(resource_name))

        test_table_partition = 'pt=test,sec=2'
        table.create_partition(test_table_partition)
        res = res.update(partition=test_table_partition)
        self.assertIsInstance(res, TableResource)
        self.assertEqual(res.get_source_table().name, test_table_name)
        self.assertEqual(str(res.get_source_table_partition()),
                         str(types.PartitionSpec(test_table_partition)))
        self.assertIs(res, self.odps.get_resource(resource_name))

        test_table_partition = types.PartitionSpec('pt=test,sec=3')
        table.create_partition(test_table_partition)
        res = res.update(partition=test_table_partition)
        self.assertIsInstance(res, TableResource)
        self.assertEqual(res.get_source_table().name, test_table_name)
        self.assertEqual(str(res.get_source_table_partition()),
                         str(test_table_partition))
        self.assertIs(res, self.odps.get_resource(resource_name))

        with res.open_writer() as writer:
            writer.write([0, FILE_CONTENT])
        with res.open_reader() as reader:
            rec = list(reader)[0]
            self.assertEqual(rec[1], FILE_CONTENT)

        if secondary_project:
            resource_name2 = tn('pyodps_t_tmp_table_resource2')
            try:
                self.odps.delete_resource(resource_name2)
            except errors.NoSuchObject:
                pass
            res = self.odps.create_resource(resource_name2, 'table', project_name=secondary_project,
                                            table_name=test_table_name)
            self.assertIsInstance(res, TableResource)
            self.assertEqual(res.get_source_table().project.name, secondary_project)
            self.assertEqual(res.get_source_table().name, test_table_name)
            self.assertEqual(res.table.project.name, secondary_project)
            self.assertEqual(res.table.name, test_table_name)
            self.assertIsNone(res.get_source_table_partition())
            self.assertIs(res, self.odps.get_resource(resource_name2))

            del res.parent[resource_name2]  # delete from cache

            self.assertIsNot(res, self.odps.get_resource(resource_name2))
            res = self.odps.get_resource(resource_name2)
            self.assertIsInstance(res, TableResource)
            self.assertEqual(res.get_source_table().project.name, secondary_project)
            self.assertEqual(res.get_source_table().name, test_table_name)
            self.assertIsNone(res.get_source_table_partition())

            test_table_partition = 'pt=test,sec=1'
            res = res.update(project_name=self.odps.project, partition=test_table_partition)
            self.assertIsInstance(res, TableResource)
            self.assertEqual(res.get_source_table().project.name, self.odps.project)
            self.assertEqual(res.get_source_table().name, test_table_name)
            self.assertEqual(str(res.partition.spec),
                             str(types.PartitionSpec(test_table_partition)))

            res = res.update(table_name=secondary_project + '.' + test_table_name, partition=None)
            self.assertIsInstance(res, TableResource)
            self.assertEqual(res.get_source_table().project.name, secondary_project)
            self.assertEqual(res.get_source_table().name, test_table_name)
            self.assertIsNone(res.get_source_table_partition())

        self.odps.delete_resource(resource_name)
        self.odps.delete_table(test_table_name)
        if secondary_project:
            self.odps.delete_table(test_table_name, project=secondary_project)

    def testTempFileResource(self):
        resource_name = tn('pyodps_t_tmp_file_resource')

        try:
            self.odps.delete_resource(resource_name)
        except errors.ODPSError:
            pass

        resource = self.odps.create_resource(resource_name, 'file', file_obj=FILE_CONTENT, temp=True)
        self.assertIsInstance(resource, FileResource)
        self.assertTrue(resource.is_temp_resource)

        self.odps.delete_resource(resource_name)

    def testFileResource(self):
        resource_name = tn('pyodps_t_tmp_file_resource')

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
                             [to_str(add_newline(l)) for l in FILE_CONTENT.splitlines()])

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
        self.assertEqual(to_str(fp.read()), FILE_CONTENT[0])
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

    def testVolumeArchiveResource(self):
        volume_name = tn('pyodps_t_tmp_resource_archive_volume')
        resource_name = tn('pyodps_t_tmp_volume_archive_resource') + '.zip'
        partition_name = 'test_partition'
        file_name = 'test_file.zip'
        try:
            self.odps.delete_volume(volume_name)
        except errors.ODPSError:
            pass
        try:
            self.odps.delete_resource(resource_name)
        except errors.ODPSError:
            pass

        file_io = six.BytesIO()
        zfile = zipfile.ZipFile(file_io, 'a', zipfile.ZIP_DEFLATED, False)
        zfile.writestr('file1.txt', FILE_CONTENT)
        zfile.writestr('file2.txt', OVERWRITE_FILE_CONTENT)
        zfile.close()

        self.odps.create_parted_volume(volume_name)
        with self.odps.open_volume_writer(volume_name, partition_name) as writer:
            writer.write(file_name, file_io.getvalue())

        volume_file = self.odps.get_volume_partition(volume_name, partition_name).files[file_name]
        self.odps.create_resource(resource_name, 'volumearchive', volume_file=volume_file)
        res = self.odps.get_resource(resource_name)
        self.assertIsInstance(res, VolumeArchiveResource)
        self.assertEqual(res.type, Resource.Type.VOLUMEARCHIVE)
        self.assertEqual(res.volume_path, volume_file.path)
        self.odps.delete_resource(resource_name)

    def testVolumeFileResource(self):
        volume_name = tn('pyodps_t_tmp_resource_file_volume')
        resource_name = tn('pyodps_t_tmp_volume_file_resource')
        partition_name = 'test_partition'
        file_name = 'test_file.txt'
        try:
            self.odps.delete_volume(volume_name)
        except errors.ODPSError:
            pass
        try:
            self.odps.delete_resource(resource_name)
        except errors.ODPSError:
            pass

        self.odps.create_parted_volume(volume_name)
        with self.odps.open_volume_writer(volume_name, partition_name) as writer:
            writer.write(file_name, FILE_CONTENT)

        volume_file = self.odps.get_volume_partition(volume_name, partition_name).files[file_name]
        self.odps.create_resource(resource_name, 'volumefile', volume_file=volume_file)
        res = self.odps.get_resource(resource_name)
        self.assertIsInstance(res, VolumeFileResource)
        self.assertEqual(res.type, Resource.Type.VOLUMEFILE)
        self.assertEqual(res.volume_path, volume_file.path)
        self.odps.delete_resource(resource_name)


if __name__ == '__main__':
    unittest.main()
