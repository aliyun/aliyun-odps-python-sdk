#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from __future__ import print_function

from odps.tests.core import TestBase, tn
from odps.compat import six
from odps.utils import to_str
from odps.models import PartedVolume, FSVolume, VolumeFSDir, VolumeFSFile

FILE_CONTENT = to_str("""
Four score and seven years ago our fathers brought forth,
upon this continent,
a new nation,
conceived in liberty,
and dedicated to the proposition that "all men are created equal"
""")
FILE_CONTENT2 = to_str("""
Were it to benefit my country I would lay down my life;
What then is risk to me?
""")
TEST_PARTED_VOLUME_NAME = tn('pyodps_test_parted_volume')
TEST_FS_VOLUME_NAME = tn('pyodps_test_fs_volume')

TEST_PARTITION_NAME = 'pyodps_test_partition'
TEST_FILE_NAME = 'test_output_file'
TEST_FILE_NAME2 = 'test_output_file2'
TEST_NEW_FILE_NAME = 'test_new_output_file'

TEST_DIR_NAME = 'pyodps_test_dir'


class Test(TestBase):
    def tearDown(self):
        if self.odps.exist_volume(TEST_PARTED_VOLUME_NAME):
            self.odps.delete_volume(TEST_PARTED_VOLUME_NAME)
        if self.odps.exist_volume(TEST_FS_VOLUME_NAME):
            self.odps.delete_volume(TEST_FS_VOLUME_NAME)
        super(Test, self).tearDown()

    def testVolumes(self):
        if self.odps.exist_volume(TEST_PARTED_VOLUME_NAME):
            self.odps.delete_volume(TEST_PARTED_VOLUME_NAME)
        self.odps.create_parted_volume(TEST_PARTED_VOLUME_NAME)

        if self.odps.exist_volume(TEST_FS_VOLUME_NAME):
            self.odps.delete_volume(TEST_FS_VOLUME_NAME)
        self.odps.create_fs_volume(TEST_FS_VOLUME_NAME)

        volume = self.odps.get_volume(TEST_PARTED_VOLUME_NAME)
        self.assertIsInstance(volume, PartedVolume)
        self.assertIs(volume, self.odps.get_volume(TEST_PARTED_VOLUME_NAME))
        volume.reload()
        self.assertEqual(volume.name, TEST_PARTED_VOLUME_NAME)

        volume = self.odps.get_volume(TEST_FS_VOLUME_NAME)
        self.assertIsInstance(volume, FSVolume)
        self.assertIs(volume, self.odps.get_volume(TEST_FS_VOLUME_NAME))
        volume.reload()
        self.assertEqual(volume.name, TEST_FS_VOLUME_NAME)

        self.assertTrue(self.odps.exist_volume(TEST_PARTED_VOLUME_NAME))
        self.assertTrue(self.odps.exist_volume(TEST_FS_VOLUME_NAME))
        self.assertFalse(self.odps.exist_volume('non_existing_volume'))

        for vol in self.odps.list_volumes():
            self.assertIsNotNone(vol.name)

    def testVolumePartitionAndFile(self):
        if self.odps.exist_volume(TEST_PARTED_VOLUME_NAME):
            self.odps.delete_volume(TEST_PARTED_VOLUME_NAME)
        self.odps.create_parted_volume(TEST_PARTED_VOLUME_NAME)

        vol = self.odps.get_volume(TEST_PARTED_VOLUME_NAME)
        partition_path = '/'.join(('', TEST_PARTED_VOLUME_NAME, TEST_PARTITION_NAME))
        partition = vol.get_partition(TEST_PARTITION_NAME)
        self.assertIs(partition, self.odps.get_volume_partition(partition_path))
        with partition.open_writer() as writer:
            writer.write(TEST_FILE_NAME, FILE_CONTENT)
            writer.write(TEST_FILE_NAME2, FILE_CONTENT2)
        partition.reload()
        self.assertEqual(partition.name, TEST_PARTITION_NAME)
        self.assertEqual(partition.length, len(FILE_CONTENT) + len(FILE_CONTENT2))
        self.assertEqual(partition.file_number, 2)

        file_path = '/'.join(('', TEST_PARTED_VOLUME_NAME, TEST_PARTITION_NAME, TEST_FILE_NAME))
        file_obj = self.odps.get_volume_file(file_path)
        self.assertEqual(file_obj.name, TEST_FILE_NAME)
        self.assertEqual(self.odps.project + '/volumes/' + file_path.lstrip('/'), file_obj.path)

        with partition.files[TEST_FILE_NAME].open_reader() as reader:
            out_content = reader.read()
            if not six.PY2:
                out_content = out_content.decode('utf-8')
            self.assertEqual(out_content, FILE_CONTENT)

        self.assertTrue(vol.exist_partition(TEST_PARTITION_NAME))
        self.assertFalse(vol.exist_partition('non_existing_partition'))

        for part in self.odps.list_volume_partitions(TEST_PARTED_VOLUME_NAME):
            self.assertIsNotNone(part.name)

        for f in partition.list_files():
            self.assertIsNotNone(f.name)
        self.assertEqual(len(list(self.odps.list_volume_files(partition_path))), 2)
        self.assertTrue(any(f.name == TEST_FILE_NAME for f in self.odps.list_volume_files(partition_path)))

        self.odps.delete_volume_partition(partition_path)
        self.assertFalse(self.odps.exist_volume_partition(partition_path))

    def testVolumeFS(self):
        if self.odps.exist_volume(TEST_FS_VOLUME_NAME):
            self.odps.delete_volume(TEST_FS_VOLUME_NAME)
        self.odps.create_fs_volume(TEST_FS_VOLUME_NAME)

        vol = self.odps.get_volume(TEST_FS_VOLUME_NAME)

        self.odps.create_volume_directory(vol.path + '/' + TEST_DIR_NAME)
        dir_obj = vol[TEST_DIR_NAME]
        self.assertIsInstance(dir_obj, VolumeFSDir)
        self.assertIs(dir_obj, self.odps.get_volume_file(vol.path + '/' + TEST_DIR_NAME))
        self.assertEqual(dir_obj.path, '/' + TEST_FS_VOLUME_NAME + '/' + TEST_DIR_NAME)
        self.assertTrue(any(f.path in (dir_obj.path, dir_obj.path + '/')
                            for f in self.odps.list_volume_files(vol.path)))

        with self.odps.open_volume_writer(dir_obj.path + '/' + TEST_FILE_NAME) as writer:
            writer.write(FILE_CONTENT)
        self.assertNotIn('non_existing_file', dir_obj)
        self.assertIn(TEST_FILE_NAME, dir_obj)
        self.assertTrue(any(f.basename == TEST_FILE_NAME
                            for f in self.odps.list_volume_files(dir_obj.path)))
        with self.odps.open_volume_reader(dir_obj.path + '/' + TEST_FILE_NAME) as reader:
            content = reader.read()
            self.assertEqual(to_str(content), FILE_CONTENT)

        file_obj = dir_obj[TEST_FILE_NAME]
        self.assertIsInstance(file_obj, VolumeFSFile)
        self.assertIs(file_obj, dir_obj[TEST_FILE_NAME])
        with file_obj.open_reader() as reader:
            content = reader.read()
            self.assertEqual(to_str(content), FILE_CONTENT)
        file_obj.replication = 5
        self.assertEqual(file_obj.replication, 5)

        old_dir_name = file_obj.dirname
        self.odps.move_volume_file(file_obj.path, './/' + TEST_NEW_FILE_NAME, replication=10)
        self.assertEqual(old_dir_name, file_obj.dirname)
        self.assertEqual(file_obj.basename, TEST_NEW_FILE_NAME)
        self.assertEqual(file_obj.replication, 10)
        self.assertNotIn(TEST_FILE_NAME, dir_obj)
        self.odps.delete_volume_file(file_obj.path)
        self.assertNotIn(TEST_NEW_FILE_NAME, dir_obj)

        dir_obj.delete()
        self.assertNotIn(TEST_DIR_NAME, vol)
