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

from odps.compat import lrange
from odps.tests.core import TestBase
from odps.tunnel import CompressOption

TEST_VOLUME_NAME = 'pyodps_test_volume'
TEST_PARTITION_NAME = 'pyodps_test_partition'
TEST_FILE_NAME = 'test_output_file'

TEST_BLOCK_SIZE = 1048500
TEST_MODULUS = 251


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        if not self.odps.exist_volume(TEST_VOLUME_NAME):
            self.odps.create_volume(TEST_VOLUME_NAME)

    def tearDown(self):
        self.odps.delete_volume(TEST_VOLUME_NAME)
        super(Test, self).setUp()

    def testTextUploadDownload(self):
        text_content = 'Life is short, \r\n Java is tedious.    \n\n\r\nI use PyODPS. \n\n'
        expect_lines = ['Life is short, \n', ' Java is tedious.    \n', '\n', '\n', 'I use PyODPS. \n', '\n']

        partition = self.odps.get_volume_partition(TEST_VOLUME_NAME, TEST_PARTITION_NAME)
        with partition.open_writer() as writer:
            writer.write(TEST_FILE_NAME, text_content)

        with partition.open_reader(TEST_FILE_NAME) as reader:
            actual_lines = [line for line in reader]
            assert expect_lines == actual_lines

    def testRawUploadDownloadGreenlet(self):
        from odps.tunnel import io
        io.RequestsIO = io.GreenletRequestsIO

        block = bytes(bytearray([iid % TEST_MODULUS for iid in lrange(TEST_BLOCK_SIZE)]))

        partition = self.odps.get_volume_partition(TEST_VOLUME_NAME, TEST_PARTITION_NAME)
        with partition.open_writer() as writer:
            writer.write(TEST_FILE_NAME, block)

        with partition.open_reader(TEST_FILE_NAME) as reader:
            assert reader.read() == block

    def testRawUploadDownloadThread(self):
        from odps.tunnel import io
        io.RequestsIO = io.ThreadRequestsIO

        block = bytes(bytearray([iid % TEST_MODULUS for iid in lrange(TEST_BLOCK_SIZE)]))

        partition = self.odps.get_volume_partition(TEST_VOLUME_NAME, TEST_PARTITION_NAME)
        with partition.open_writer() as writer:
            writer.write(TEST_FILE_NAME, block)

        with partition.open_reader(TEST_FILE_NAME) as reader:
            assert reader.read() == block

    def testZLibUploadDownload(self):
        block = bytes(bytearray([iid % TEST_MODULUS for iid in lrange(TEST_BLOCK_SIZE)]))

        comp_option = CompressOption(level=9)

        partition = self.odps.get_volume_partition(TEST_VOLUME_NAME, TEST_PARTITION_NAME)
        with partition.open_writer(compress_option=comp_option) as writer:
            writer.write(TEST_FILE_NAME, block, compress=True)

        with partition.open_reader(TEST_FILE_NAME, compress_option=comp_option) as reader:
            assert reader.read() == block
