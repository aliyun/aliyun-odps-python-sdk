#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from __future__ import print_function

from odps.tests.core import TestBase

FILE_CONTENT = '''This is a pyodps test file!
Enjoy it'''
OVERWRITE_FILE_CONTENT = '''This is an overwritten pyodps test file!
Enjoy it'''
TEST_VOLUME_NAME = 'pyodps_test_volume'
TEST_PARTITION_NAME = 'pyodps_test_partition'
TEST_FILE_NAME = 'test_output_file'
TEST_FILE_CONTENT = 'Life is short.\r\nI use PyODPS.'


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        if not self.odps.exist_volume(TEST_VOLUME_NAME):
            self.odps.create_volume(TEST_VOLUME_NAME)

    def tearDown(self):
        self.odps.delete_volume(TEST_VOLUME_NAME)
        super(Test, self).setUp()

    def testVolumes(self):
        volume = self.odps.get_volume(TEST_VOLUME_NAME)
        volume.reload()
        assert volume.name == TEST_VOLUME_NAME

        assert self.odps.exist_volume(TEST_VOLUME_NAME)
        assert not self.odps.exist_volume('non_existing_volume')

        for vol in self.odps.list_volumes():
            assert vol.name

    def testVolumePartitionAndFile(self):
        vol = self.odps.get_volume(TEST_VOLUME_NAME)
        partition = vol.get_partition(TEST_PARTITION_NAME)
        with partition.open_writer() as writer:
            writer.write(TEST_FILE_NAME, TEST_FILE_CONTENT)
        partition.reload()
        assert partition.name == TEST_PARTITION_NAME

        with partition.open_reader(TEST_FILE_NAME) as reader:
            assert reader.read() == TEST_FILE_CONTENT

        assert vol.exist_partition(TEST_PARTITION_NAME)
        assert not vol.exist_partition('non_existing_partition')

        for part in vol.list_partitions():
            assert part.name

        for f in partition.list_files():
            assert f.name
        assert any(1 for f in partition.list_files() if f.name == TEST_FILE_NAME)

        vol.delete_partition(TEST_PARTITION_NAME)
        assert not vol.exist_partition(TEST_PARTITION_NAME)
