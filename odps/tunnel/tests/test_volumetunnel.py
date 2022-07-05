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

from odps.compat import irange
from odps.tests.core import TestBase, tn
from odps.tunnel import CompressOption
from odps.tunnel.volumetunnel import VolumeTunnel

TEST_PARTED_VOLUME_NAME = tn('pyodps_test_p_volume')
TEST_FS_VOLUME_NAME = tn('pyodps_test_fs_volume')
TEST_PARTITION_NAME = 'pyodps_test_partition'
TEST_FILE_NAME = 'test_output_file'

TEST_BLOCK_SIZE = 1048500
TEST_MODULUS = 251


class Test(TestBase):
    def setUp(self):
        def wrap_fun(func):
            def wrapped(self, *args, **kwargs):
                ret = func(self, *args, **kwargs)
                repr(ret)
                return ret

            wrapped.__name__ = func.__name__
            wrapped.__doc__ = func.__doc__
            return wrapped

        self._old_create_download_session = VolumeTunnel.create_download_session
        self._old_create_upload_session = VolumeTunnel.create_upload_session

        VolumeTunnel.create_download_session = wrap_fun(
            self._old_create_download_session
        )
        VolumeTunnel.create_upload_session = wrap_fun(self._old_create_upload_session)
        super(Test, self).setUp()

    def tearDown(self):
        if self.odps.exist_volume(TEST_PARTED_VOLUME_NAME):
            self.odps.delete_volume(TEST_PARTED_VOLUME_NAME)
        if self.odps.exist_volume(TEST_FS_VOLUME_NAME):
            self.odps.delete_volume(TEST_FS_VOLUME_NAME)
        VolumeTunnel.create_download_session = self._old_create_download_session
        VolumeTunnel.create_upload_session = self._old_create_upload_session
        super(Test, self).tearDown()

    @staticmethod
    def _gen_byte_block():
        return bytes(bytearray([iid % TEST_MODULUS for iid in irange(TEST_BLOCK_SIZE)]))

    def _get_test_partition(self):
        if self.odps.exist_volume(TEST_PARTED_VOLUME_NAME):
            self.odps.delete_volume(TEST_PARTED_VOLUME_NAME)
        self.odps.create_parted_volume(TEST_PARTED_VOLUME_NAME)
        return self.odps.get_volume_partition(TEST_PARTED_VOLUME_NAME, TEST_PARTITION_NAME)

    def _get_test_fs(self):
        if self.odps.exist_volume(TEST_FS_VOLUME_NAME):
            self.odps.delete_volume(TEST_FS_VOLUME_NAME)
        self.odps.create_fs_volume(TEST_FS_VOLUME_NAME)
        return self.odps.get_volume(TEST_FS_VOLUME_NAME)

    def testTextUploadDownload(self):
        text_content = 'Life is short, \r\n Java is tedious.    \n\n\r\nI use PyODPS. \n\n'
        expect_lines = ['Life is short, \n', ' Java is tedious.    \n', '\n', '\n', 'I use PyODPS. \n', '\n']

        partition = self._get_test_partition()
        with partition.open_writer() as writer:
            writer.write(TEST_FILE_NAME, text_content)

        with partition.open_reader(TEST_FILE_NAME) as reader:
            actual_lines = [line for line in reader]
            assert expect_lines == actual_lines

    def testRawUploadDownloadGreenlet(self):
        block = self._gen_byte_block()

        partition = self._get_test_partition()
        with partition.open_writer() as writer:
            writer.write(TEST_FILE_NAME, block)

        with partition.open_reader(TEST_FILE_NAME) as reader:
            assert reader.read() == block

    def testRawUploadDownloadThread(self):
        from odps.tunnel import io
        io._FORCE_THREAD = True

        block = self._gen_byte_block()

        partition = self._get_test_partition()
        with partition.open_writer() as writer:
            writer.write(TEST_FILE_NAME, block)

        with partition.open_reader(TEST_FILE_NAME) as reader:
            assert reader.read() == block

    def testZLibUploadDownload(self):
        block = self._gen_byte_block()

        comp_option = CompressOption(level=9)

        partition = self._get_test_partition()
        with partition.open_writer(compress_option=comp_option) as writer:
            writer.write(TEST_FILE_NAME, block, compress=True)

        with partition.open_reader(TEST_FILE_NAME, compress_option=comp_option) as reader:
            assert reader.read() == block

    def testFSRawUploadDownload(self):
        block = self._gen_byte_block()

        vol = self._get_test_fs()
        with vol.open_writer(TEST_FILE_NAME) as writer:
            writer.write(block)

        with vol.open_reader(TEST_FILE_NAME) as reader:
            assert reader.read() == block

    def testFSZLibUploadDownload(self):
        block = self._gen_byte_block()

        comp_option = CompressOption(level=9)

        vol = self._get_test_fs()
        with vol.open_writer(TEST_FILE_NAME, compress_option=comp_option) as writer:
            writer.write(block)

        with vol.open_reader(TEST_FILE_NAME, compress_option=comp_option) as reader:
            parts = []
            while True:
                b = reader.read(10003)
                if not b:
                    break
                parts.append(b)
            self.assertEqual(bytes().join(parts), block)
