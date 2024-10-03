#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import warnings

import pytest

try:
    import oss2
except ImportError:
    oss2 = None

from ...compat import six
from ...errors import NoSuchObject
from ...tests.core import tn
from ...utils import to_str
from .. import (
    ExternalVolume,
    ExternalVolumeDir,
    ExternalVolumeFile,
    FSVolume,
    FSVolumeDir,
    FSVolumeFile,
    PartedVolume,
)

FILE_CONTENT = to_str(
    """
Four score and seven years ago our fathers brought forth,
upon this continent,
a new nation,
conceived in liberty,
and dedicated to the proposition that "all men are created equal"
"""
)
FILE_CONTENT2 = to_str(
    """
Were it to benefit my country I would lay down my life;
What then is risk to me?
"""
)
TEST_PARTED_VOLUME_NAME = tn("pyodps_test_parted_volume")
TEST_FS_VOLUME_NAME = tn("pyodps_test_fs_volume")
TEST_EXT_VOLUME_NAME = tn("pyodps_test_external_volume")

TEST_PARTITION_NAME = "pyodps_test_partition"
TEST_FILE_NAME = "test_output_file"
TEST_FILE_NAME2 = "test_output_file2"
TEST_NEW_FILE_NAME = "test_new_output_file"

TEST_DIR_NAME = "pyodps_test_dir"


@pytest.fixture
def auto_teardown_volumes(odps):
    try:
        yield
    finally:
        if odps.exist_volume(TEST_PARTED_VOLUME_NAME):
            odps.delete_volume(TEST_PARTED_VOLUME_NAME)
        if odps.exist_volume(TEST_FS_VOLUME_NAME):
            odps.delete_volume(TEST_FS_VOLUME_NAME)


@pytest.fixture
def check_experimental(request):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", category=FutureWarning)
        yield
        if request.param:
            assert len(record) > 0, "No experimental warnings popped"
        else:
            assert len(record) == 0, "Unexpected experimental warnings popped"


def test_volumes(odps):
    if odps.exist_volume(TEST_PARTED_VOLUME_NAME):
        odps.delete_volume(TEST_PARTED_VOLUME_NAME)
    odps.create_parted_volume(TEST_PARTED_VOLUME_NAME)

    if odps.exist_volume(TEST_FS_VOLUME_NAME):
        odps.delete_volume(TEST_FS_VOLUME_NAME)
    odps.create_fs_volume(TEST_FS_VOLUME_NAME)

    volume = odps.get_volume(TEST_PARTED_VOLUME_NAME)
    assert isinstance(volume, PartedVolume)
    assert volume is odps.get_volume(TEST_PARTED_VOLUME_NAME)
    volume.reload()
    assert volume.name == TEST_PARTED_VOLUME_NAME

    volume = odps.get_volume(TEST_FS_VOLUME_NAME)
    assert isinstance(volume, FSVolume)
    assert volume is odps.get_volume(TEST_FS_VOLUME_NAME)
    volume.reload()
    assert volume.name == TEST_FS_VOLUME_NAME

    assert odps.exist_volume(TEST_PARTED_VOLUME_NAME) is True
    assert odps.exist_volume(TEST_FS_VOLUME_NAME) is True
    assert odps.exist_volume("non_existing_volume") is False

    for vol in odps.list_volumes():
        assert vol.name is not None


def test_volume_partition_and_file(odps):
    if odps.exist_volume(TEST_PARTED_VOLUME_NAME):
        odps.delete_volume(TEST_PARTED_VOLUME_NAME)
    odps.create_parted_volume(TEST_PARTED_VOLUME_NAME)

    vol = odps.get_volume(TEST_PARTED_VOLUME_NAME)
    partition_path = "/".join(("", TEST_PARTED_VOLUME_NAME, TEST_PARTITION_NAME))
    partition = vol.get_partition(TEST_PARTITION_NAME)
    assert partition is odps.get_volume_partition(partition_path)
    with partition.open_writer() as writer:
        writer.write(TEST_FILE_NAME, FILE_CONTENT)
        writer.write(TEST_FILE_NAME2, FILE_CONTENT2)
    partition.reload()
    assert partition.name == TEST_PARTITION_NAME
    assert partition.length == len(FILE_CONTENT) + len(FILE_CONTENT2)
    assert partition.file_number == 2

    file_path = "/".join(
        ("", TEST_PARTED_VOLUME_NAME, TEST_PARTITION_NAME, TEST_FILE_NAME)
    )
    file_obj = odps.get_volume_file(file_path)
    assert file_obj.name == TEST_FILE_NAME
    assert odps.project + "/volumes/" + file_path.lstrip("/") == file_obj.path

    with partition.files[TEST_FILE_NAME].open_reader() as reader:
        out_content = reader.read()
        if not six.PY2:
            out_content = out_content.decode("utf-8")
        assert out_content == FILE_CONTENT

    assert vol.exist_partition(TEST_PARTITION_NAME) is True
    assert vol.exist_partition("non_existing_partition") is False

    for part in odps.list_volume_partitions(TEST_PARTED_VOLUME_NAME):
        assert part.name is not None

    for f in partition.list_files():
        assert f.name is not None
    assert len(list(odps.list_volume_files(partition_path))) == 2
    assert (
        any(f.name == TEST_FILE_NAME for f in odps.list_volume_files(partition_path))
        is True
    )

    odps.delete_volume_partition(partition_path)
    assert odps.exist_volume_partition(partition_path) is False


@pytest.mark.parametrize("check_experimental", [True], indirect=True)
def test_volume_fs(odps, check_experimental):
    if odps.exist_volume(TEST_FS_VOLUME_NAME):
        odps.delete_volume(TEST_FS_VOLUME_NAME)
    odps.create_fs_volume(TEST_FS_VOLUME_NAME)

    vol = odps.get_volume(TEST_FS_VOLUME_NAME)

    odps.create_volume_directory(vol.path + "/" + TEST_DIR_NAME)
    dir_obj = vol[TEST_DIR_NAME]
    assert isinstance(dir_obj, FSVolumeDir)
    assert dir_obj is odps.get_volume_file(vol.path + "/" + TEST_DIR_NAME)
    assert dir_obj.path == "/" + TEST_FS_VOLUME_NAME + "/" + TEST_DIR_NAME
    assert (
        any(
            f.path in (dir_obj.path, dir_obj.path + "/")
            for f in odps.list_volume_files(vol.path)
        )
        is True
    )

    with odps.open_volume_writer(dir_obj.path + "/" + TEST_FILE_NAME) as writer:
        writer.write(FILE_CONTENT)
    assert "non_existing_file" not in dir_obj
    assert TEST_FILE_NAME in dir_obj
    assert (
        any(f.basename == TEST_FILE_NAME for f in odps.list_volume_files(dir_obj.path))
        is True
    )

    with odps.open_volume_reader(dir_obj.path + "/" + TEST_FILE_NAME) as reader:
        content = reader.read()
        assert to_str(content) == FILE_CONTENT

    file_obj = dir_obj[TEST_FILE_NAME]
    assert isinstance(file_obj, FSVolumeFile)
    assert file_obj is dir_obj[TEST_FILE_NAME]
    with file_obj.open_reader() as reader:
        content = reader.read()
        assert to_str(content) == FILE_CONTENT
    file_obj.replication = 5
    assert file_obj.replication == 5

    old_dir_name = file_obj.dirname
    odps.move_volume_file(file_obj.path, ".//" + TEST_NEW_FILE_NAME, replication=10)
    assert old_dir_name == file_obj.dirname
    assert file_obj.basename == TEST_NEW_FILE_NAME
    assert file_obj.replication == 10
    assert TEST_FILE_NAME not in dir_obj
    odps.delete_volume_file(file_obj.path)
    assert TEST_NEW_FILE_NAME not in dir_obj

    dir_obj.delete()
    assert TEST_DIR_NAME not in vol


@pytest.mark.parametrize("check_experimental", [False], indirect=True)
def test_external_volume(config, odps_daily, check_experimental):
    if not hasattr(config, "oss_bucket") or oss2 is None:
        pytest.skip("Need oss2 and config to run this test")

    test_dir_name = tn("test_oss_directory")

    try:
        odps_daily.delete_volume(TEST_EXT_VOLUME_NAME)
    except NoSuchObject:
        pass

    config.oss_bucket.put_object(test_dir_name + "/", b"")

    (
        oss_access_id,
        oss_secret_access_key,
        oss_bucket_name,
        oss_endpoint,
    ) = config.oss_config
    test_location = "oss://%s:%s@%s/%s/%s" % (
        oss_access_id,
        oss_secret_access_key,
        oss_endpoint,
        oss_bucket_name,
        test_dir_name,
    )

    vol = odps_daily.create_external_volume(
        TEST_EXT_VOLUME_NAME, location=test_location, auto_create_dir=True
    )
    try:
        assert isinstance(vol, ExternalVolume)

        test_read_file_name = "test_oss_file_read"
        config.oss_bucket.put_object(
            test_dir_name + "/" + test_read_file_name, FILE_CONTENT
        )
        assert isinstance(vol[test_read_file_name], ExternalVolumeFile)
        with vol[test_read_file_name].open_reader() as reader:
            assert reader.read() == FILE_CONTENT.encode()
        vol[test_read_file_name].delete()
        assert test_read_file_name not in vol

        test_write_file_name = "test_oss_file_write"
        with vol.open_writer(test_write_file_name) as writer:
            writer.write(FILE_CONTENT2.encode())
        assert any(test_write_file_name in f.path for f in vol)
        assert (
            config.oss_bucket.get_object(
                test_dir_name + "/" + test_write_file_name
            ).read()
            == FILE_CONTENT2.encode()
        )
        vol.delete(test_write_file_name)
        assert not any(test_write_file_name in f.path for f in vol)

        test_subdir_name = "test_oss_dir"
        dir_obj = vol.create_dir(test_subdir_name)
        assert isinstance(dir_obj, ExternalVolumeDir)
        with dir_obj.open_writer(test_write_file_name) as writer:
            writer.write(FILE_CONTENT2.encode())
        assert (
            config.oss_bucket.get_object(
                "/".join([test_dir_name, test_subdir_name, test_write_file_name])
            ).read()
            == FILE_CONTENT2.encode()
        )
        with dir_obj.open_reader(test_write_file_name) as reader:
            assert reader.read() == FILE_CONTENT2.encode()
        dir_obj.delete(recursive=True)
        assert not any(test_subdir_name in f.path for f in vol)
    finally:
        vol.drop(auto_remove_dir=True, recursive=True)
