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

from collections import namedtuple

import pytest

from ...compat import irange, six
from ...errors import NoSuchObject
from ...tests.core import get_test_unique_name, tn
from .. import CompressOption
from ..volumetunnel import VolumeTunnel

TEST_PARTITION_NAME = 'pyodps_test_partition'
TEST_FILE_NAME = 'test_output_file'

TEST_BLOCK_SIZE = 1048500
TEST_MODULUS = 251


@pytest.fixture
def setup(odps):
    test_parted_vol_name = tn("pyodps_test_p_volume_" + get_test_unique_name(5))
    test_fs_vol_name = tn("pyodps_test_fs_volume" + get_test_unique_name(5))

    def gen_byte_block():
        return bytes(bytearray([iid % TEST_MODULUS for iid in irange(TEST_BLOCK_SIZE)]))

    def get_test_partition():
        if odps.exist_volume(test_parted_vol_name):
            odps.delete_volume(test_parted_vol_name)
        odps.create_parted_volume(test_parted_vol_name)
        return odps.get_volume_partition(test_parted_vol_name, TEST_PARTITION_NAME)

    def get_test_fs():
        if odps.exist_volume(test_fs_vol_name):
            odps.delete_volume(test_fs_vol_name)
        odps.create_fs_volume(test_fs_vol_name)
        return odps.get_volume(test_fs_vol_name)

    def wrap_fun(func):
        @six.wraps(func)
        def wrapped(self, *args, **kwargs):
            ret = func(self, *args, **kwargs)
            repr(ret)
            return ret

        return wrapped

    _old_create_download_session = VolumeTunnel.create_download_session
    _old_create_upload_session = VolumeTunnel.create_upload_session

    VolumeTunnel.create_download_session = wrap_fun(
        _old_create_download_session
    )
    VolumeTunnel.create_upload_session = wrap_fun(_old_create_upload_session)

    test_funcs = namedtuple(
        "TestFuncs", "gen_byte_block, get_test_partition, get_test_fs"
    )
    try:
        yield test_funcs(gen_byte_block, get_test_partition, get_test_fs)
    finally:
        try:
            odps.delete_volume(test_parted_vol_name)
        except NoSuchObject:
            pass
        try:
            odps.delete_volume(test_fs_vol_name)
        except NoSuchObject:
            pass
        VolumeTunnel.create_download_session = _old_create_download_session
        VolumeTunnel.create_upload_session = _old_create_upload_session


def test_text_upload_download(odps, setup):
    text_content = 'Life is short, \r\n Java is tedious.    \n\n\r\nI use PyODPS. \n\n'
    expect_lines = [
        'Life is short, \n', ' Java is tedious.    \n', '\n', '\n', 'I use PyODPS. \n', '\n'
    ]

    partition = setup.get_test_partition()
    with partition.open_writer() as writer:
        writer.write(TEST_FILE_NAME, text_content)

    with partition.open_reader(TEST_FILE_NAME) as reader:
        actual_lines = [line for line in reader]
        assert expect_lines == actual_lines


def test_raw_upload_download_greenlet(odps, setup):
    block = setup.gen_byte_block()

    partition = setup.get_test_partition()
    with partition.open_writer() as writer:
        writer.write(TEST_FILE_NAME, block)

    with partition.open_reader(TEST_FILE_NAME) as reader:
        assert reader.read() == block


def test_raw_upload_download_thread(odps, setup):
    from .. import io
    io._FORCE_THREAD = True

    block = setup.gen_byte_block()

    partition = setup.get_test_partition()
    with partition.open_writer() as writer:
        writer.write(TEST_FILE_NAME, block)

    with partition.open_reader(TEST_FILE_NAME) as reader:
        assert reader.read() == block
    with partition.open_reader(TEST_FILE_NAME, reopen=True) as reader:
        assert reader.read() == block


def test_z_lib_upload_download(odps, setup):
    block = setup.gen_byte_block()

    comp_option = CompressOption(level=9)

    partition = setup.get_test_partition()
    with partition.open_writer(compress_option=comp_option) as writer:
        writer.write(TEST_FILE_NAME, block, compress=True)

    with partition.open_reader(
        TEST_FILE_NAME, compress_option=comp_option
    ) as reader:
        assert reader.read() == block


def test_fs_raw_upload_download(odps, setup):
    block = setup.gen_byte_block()

    vol = setup.get_test_fs()
    with vol.open_writer(TEST_FILE_NAME) as writer:
        writer.write(block)

    with vol.open_reader(TEST_FILE_NAME) as reader:
        assert reader.read() == block


def test_fsz_lib_upload_download(odps, setup):
    block = setup.gen_byte_block()

    comp_option = CompressOption(level=9)

    vol = setup.get_test_fs()
    with vol.open_writer(TEST_FILE_NAME, compress_option=comp_option) as writer:
        writer.write(block)

    with vol.open_reader(TEST_FILE_NAME, compress_option=comp_option) as reader:
        parts = []
        while True:
            b = reader.read(10003)
            if not b:
                break
            parts.append(b)
        assert bytes().join(parts) == block
