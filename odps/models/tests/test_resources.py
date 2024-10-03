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

import os
import tempfile
import zipfile

import pytest

from ... import compat, errors, options, types
from ...compat import ConfigParser, UnsupportedOperation, futures, six
from ...tests.core import tn
from ...utils import to_text
from .. import (
    FileResource,
    Resource,
    TableResource,
    TableSchema,
    VolumeArchiveResource,
    VolumeFileResource,
)

FILE_CONTENT = to_text(
    """
Proudly swept the rain by the cliffs
As it glided through the trees
Still following ever the bud
The ahihi lehua of the vale
"""
)
OVERWRITE_FILE_CONTENT = to_text(
    """
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
"""
)


@pytest.fixture(autouse=True)
def reset_options():
    try:
        yield
    finally:
        options.resource_chunk_size = 64 << 20
        options.upload_resource_in_chunks = True


def test_resources(odps):
    assert odps.get_project().resources is odps.get_project().resources

    next(odps.list_resources())

    for idx, resource in enumerate(odps.list_resources()):
        if idx >= 20:
            break
        assert isinstance(resource, Resource._get_cls(resource.type))

    pytest.raises(
        TypeError, lambda: odps.create_resource("test_error", "py", resource=["uvw"])
    )


def test_resource_exists(odps):
    non_exists_resource = "a_non_exists_resource"
    assert odps.exist_resource(non_exists_resource) is False


def test_table_resource(config, odps):
    try:
        secondary_project = config.get("test", "secondary_project")
    except ConfigParser.NoOptionError:
        secondary_project = None

    test_table_name = tn("pyodps_t_tmp_resource_table")
    schema = TableSchema.from_lists(["id", "name"], ["string", "string"])
    odps.delete_table(test_table_name, if_exists=True)
    odps.create_table(test_table_name, schema)
    if secondary_project:
        odps.delete_table(test_table_name, if_exists=True, project=secondary_project)
        odps.create_table(test_table_name, schema, project=secondary_project)

    resource_name = tn("pyodps_t_tmp_table_resource")
    try:
        odps.delete_resource(resource_name)
    except errors.NoSuchObject:
        pass
    res = odps.create_resource(resource_name, "table", table_name=test_table_name)
    assert isinstance(res, TableResource)
    assert res.get_source_table().name == test_table_name
    assert res.table.name == test_table_name
    assert res.get_source_table_partition() is None
    assert res is odps.get_resource(resource_name)

    with res.open_writer() as writer:
        writer.write([0, FILE_CONTENT])
    with res.open_reader() as reader:
        rec = list(reader)[0]
        assert rec[1] == FILE_CONTENT

    del res.parent[resource_name]  # delete from cache

    assert res is not odps.get_resource(resource_name)
    res = odps.get_resource(resource_name)
    assert isinstance(res, TableResource)
    assert res.get_source_table().name == test_table_name
    assert res.get_source_table_partition() is None

    test_table_partition = "pt=test,sec=1"
    schema = TableSchema.from_lists(
        ["id", "name"], ["string", "string"], ["pt", "sec"], ["string", "bigint"]
    )
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, schema)
    table.create_partition(test_table_partition)

    res = res.update(partition=test_table_partition)
    assert isinstance(res, TableResource)
    assert res.get_source_table().name == test_table_name
    assert res.table.name == test_table_name
    assert str(res.get_source_table_partition()) == str(
        types.PartitionSpec(test_table_partition)
    )
    assert str(res.partition.spec) == str(types.PartitionSpec(test_table_partition))
    assert res is odps.get_resource(resource_name)

    test_table_partition = "pt=test,sec=2"
    table.create_partition(test_table_partition)
    res = res.update(partition=test_table_partition)
    assert isinstance(res, TableResource)
    assert res.get_source_table().name == test_table_name
    assert str(res.get_source_table_partition()) == str(
        types.PartitionSpec(test_table_partition)
    )
    assert res is odps.get_resource(resource_name)

    test_table_partition = types.PartitionSpec("pt=test,sec=3")
    table.create_partition(test_table_partition)
    res = res.update(partition=test_table_partition)
    assert isinstance(res, TableResource)
    assert res.get_source_table().name == test_table_name
    assert str(res.get_source_table_partition()) == str(test_table_partition)
    assert res is odps.get_resource(resource_name)

    with res.open_writer() as writer:
        writer.write([0, FILE_CONTENT])
    with res.open_reader() as reader:
        rec = list(reader)[0]
        assert rec[1] == FILE_CONTENT

    if secondary_project:
        resource_name2 = tn("pyodps_t_tmp_table_resource2")
        try:
            odps.delete_resource(resource_name2)
        except errors.NoSuchObject:
            pass
        res = odps.create_resource(
            resource_name2,
            "table",
            project_name=secondary_project,
            table_name=test_table_name,
        )
        assert isinstance(res, TableResource)
        assert res.get_source_table().project.name == secondary_project
        assert res.get_source_table().name == test_table_name
        assert res.table.project.name == secondary_project
        assert res.table.name == test_table_name
        assert res.get_source_table_partition() is None
        assert res is odps.get_resource(resource_name2)

        del res.parent[resource_name2]  # delete from cache

        assert res is not odps.get_resource(resource_name2)
        res = odps.get_resource(resource_name2)
        assert isinstance(res, TableResource)
        assert res.get_source_table().project.name == secondary_project
        assert res.get_source_table().name == test_table_name
        assert res.get_source_table_partition() is None

        test_table_partition = "pt=test,sec=1"
        res = res.update(project_name=odps.project, partition=test_table_partition)
        assert isinstance(res, TableResource)
        assert res.get_source_table().project.name == odps.project
        assert res.get_source_table().name == test_table_name
        assert str(res.partition.spec) == str(types.PartitionSpec(test_table_partition))

        res = res.update(
            table_name=secondary_project + "." + test_table_name, partition=None
        )
        assert isinstance(res, TableResource)
        assert res.get_source_table().project.name == secondary_project
        assert res.get_source_table().name == test_table_name
        assert res.get_source_table_partition() is None

    odps.delete_resource(resource_name)
    odps.delete_table(test_table_name)
    if secondary_project:
        odps.delete_table(test_table_name, project=secondary_project)


def test_temp_file_resource(odps):
    resource_name = tn("pyodps_t_tmp_file_resource")

    try:
        odps.delete_resource(resource_name)
    except errors.ODPSError:
        pass

    resource = odps.create_resource(
        resource_name, "file", fileobj=FILE_CONTENT, temp=True
    )
    assert isinstance(resource, FileResource)
    assert resource.is_temp_resource
    resource.reload()
    assert resource.is_temp_resource

    odps.delete_resource(resource_name)


def test_stream_file_resource(odps):
    options.resource_chunk_size = 1024
    content = OVERWRITE_FILE_CONTENT * 32
    resource_name = tn("pyodps_t_tmp_file_resource")

    del_pool = futures.ThreadPoolExecutor(10)
    res_to_del = [resource_name]
    for idx in range(10):
        res_to_del.append("%s.part.tmp.%06d" % (resource_name, idx))
    for res_name in res_to_del:
        del_pool.submit(odps.delete_resource, res_name)
    del_pool.shutdown(wait=True)

    try:
        temp_dir = tempfile.mkdtemp(prefix="pyodps-test-")
        temp_file = os.path.join(temp_dir, "res_source")
        with open(temp_file, "w") as out_file:
            out_file.write(content)
        with open(temp_file, "r") as in_file:
            odps.create_resource(resource_name, "file", fileobj=in_file)
        with odps.open_resource(resource_name, mode="r", stream=True) as res:
            assert res.read() == content
    finally:
        odps.delete_resource(resource_name)

    with odps.open_resource(resource_name, mode="w", stream=True) as res:
        pytest.raises(UnsupportedOperation, lambda: res.seek(0, os.SEEK_END))
        for offset in range(0, len(content), 1023):
            res.write(content[offset : offset + 1023])
            assert res.tell() == min(offset + 1023, len(content))
        pytest.raises(UnsupportedOperation, lambda: res.truncate(1024))

    with odps.open_resource(resource_name, mode="r", stream=True) as res:
        sio = compat.StringIO()
        for offset in range(0, len(content), 1025):
            sio.write(res.read(1025))
    assert sio.getvalue() == content

    with odps.open_resource(resource_name, mode="r", stream=True) as res:
        sio = compat.StringIO()
        for line in res:
            sio.write(line)
    assert sio.getvalue() == content

    odps.delete_resource(resource_name)

    with odps.open_resource(resource_name, mode="w", stream=True, temp=True) as res:
        lines = content.splitlines(True)
        for offset in range(0, len(lines), 50):
            res.writelines(lines[offset : offset + 50])

    with odps.open_resource(resource_name, mode="r", stream=True) as res:
        lines = res.readlines()
    res.reload()
    assert res.is_temp_resource
    assert "".join(lines) == content


def test_file_resource(odps):
    resource_name = tn("pyodps_t_tmp_file_resource")

    try:
        odps.delete_resource(resource_name)
    except errors.ODPSError:
        pass

    resource = odps.create_resource(resource_name, "file", fileobj=FILE_CONTENT)
    assert isinstance(resource, FileResource)
    resource.drop()

    # create resource with open_resource and write
    with odps.open_resource(
        resource_name, mode="w", type="file", comment="comment_data", temp=True
    ) as resource:
        resource.write(FILE_CONTENT)
    resource.reload()
    assert resource.is_temp_resource
    resource.drop()

    # create resource with full resource path
    with odps.open_resource(
        odps.project + "/resources/" + resource_name,
        mode="w",
        type="file",
        comment="comment_data",
        temp=True,
    ) as resource:
        resource.write(FILE_CONTENT)
    resource.reload()
    assert resource.is_temp_resource

    resource.reload()
    assert resource.comment == "comment_data"

    with resource.open(mode="r") as fp:
        pytest.raises(IOError, lambda: fp.write("sss"))
        pytest.raises(IOError, lambda: fp.writelines(["sss\n"]))

        assert isinstance(fp.read(), six.text_type)

        fp.seek(0, compat.SEEK_END)
        size = fp.tell()
        fp.seek(0)
        assert fp._size == size

        assert to_text(fp.read()) == to_text(FILE_CONTENT)
        fp.seek(1)
        assert to_text(fp.read()) == to_text(FILE_CONTENT[1:])

        fp.seek(0)
        assert to_text(fp.readline()) == to_text(FILE_CONTENT.split("\n", 1)[0] + "\n")

        fp.seek(0)
        add_newline = lambda s: s if s.endswith("\n") else s + "\n"
        assert [to_text(add_newline(line)) for line in fp] == [
            to_text(add_newline(line)) for line in FILE_CONTENT.splitlines()
        ]

        assert fp._fp._need_commit is False
        assert fp.opened is True

    assert fp.opened is False
    assert fp._fp is None

    with resource.open(mode="w") as fp:
        pytest.raises(IOError, fp.read)
        pytest.raises(IOError, fp.readline)
        pytest.raises(IOError, fp.readlines)

        fp.writelines([OVERWRITE_FILE_CONTENT] * 2)

        assert fp._fp._need_commit is True

        size = fp._size

    with resource.open(mode="r+") as fp:
        assert to_text(fp.read()) == to_text(OVERWRITE_FILE_CONTENT * 2)

        assert size == fp._size

        fp.seek(0)
        fp.write(FILE_CONTENT)
        fp.truncate()

        assert fp._fp._need_commit is True

    with resource.open(mode="a") as fp:
        pytest.raises(IOError, fp.read)
        pytest.raises(IOError, fp.readline)
        pytest.raises(IOError, fp.readlines)

        fp.write(OVERWRITE_FILE_CONTENT)

        assert fp._fp._need_commit is True

    with resource.open(mode="a+") as fp:
        assert to_text(fp.read()) == to_text(FILE_CONTENT + OVERWRITE_FILE_CONTENT)
        fp.seek(1)
        fp.truncate()
        assert fp._fp._need_commit is True
        # redundant closing should work as well
        fp.close()

    fp = resource.open(mode="r")
    assert to_text(fp.read()) == FILE_CONTENT[0]
    fp.close()

    with resource.open(mode="w+") as fp:
        assert len(fp.read()) == 0
        fp.write(FILE_CONTENT)

    with resource.open(mode="r+") as fp:
        assert to_text(fp.read()) == FILE_CONTENT

    resource.update(file_obj="update")
    with resource.open(mode="rb") as fp:
        assert isinstance(fp.read(), six.binary_type)
        fp.seek(0)
        assert to_text(fp.read()) == to_text("update")

    odps.delete_resource(resource_name)


def test_volume_archive_resource(odps):
    volume_name = tn("pyodps_t_tmp_resource_archive_volume")
    resource_name = tn("pyodps_t_tmp_volume_archive_resource") + ".zip"
    partition_name = "test_partition"
    file_name = "test_file.zip"
    try:
        odps.delete_volume(volume_name)
    except errors.ODPSError:
        pass
    try:
        odps.delete_resource(resource_name)
    except errors.ODPSError:
        pass

    file_io = six.BytesIO()
    zfile = zipfile.ZipFile(file_io, "a", zipfile.ZIP_DEFLATED, False)
    zfile.writestr("file1.txt", FILE_CONTENT)
    zfile.writestr("file2.txt", OVERWRITE_FILE_CONTENT)
    zfile.close()

    odps.create_parted_volume(volume_name)
    with odps.open_volume_writer(volume_name, partition_name) as writer:
        writer.write(file_name, file_io.getvalue())

    volume_file = odps.get_volume_partition(volume_name, partition_name).files[
        file_name
    ]
    odps.create_resource(resource_name, "volumearchive", volume_file=volume_file)
    res = odps.get_resource(resource_name)
    assert isinstance(res, VolumeArchiveResource)
    assert res.type == Resource.Type.VOLUMEARCHIVE
    assert res.volume_path == volume_file.path
    odps.delete_resource(resource_name)


def test_volume_file_resource(odps):
    volume_name = tn("pyodps_t_tmp_resource_file_volume")
    resource_name = tn("pyodps_t_tmp_volume_file_resource")
    partition_name = "test_partition"
    file_name = "test_file.txt"
    try:
        odps.delete_volume(volume_name)
    except errors.ODPSError:
        pass
    try:
        odps.delete_resource(resource_name)
    except errors.ODPSError:
        pass

    odps.create_parted_volume(volume_name)
    with odps.open_volume_writer(volume_name, partition_name) as writer:
        writer.write(file_name, FILE_CONTENT)

    volume_file = odps.get_volume_partition(volume_name, partition_name).files[
        file_name
    ]
    odps.create_resource(resource_name, "volumefile", volume_file=volume_file)
    res = odps.get_resource(resource_name)
    assert isinstance(res, VolumeFileResource)
    assert res.type == Resource.Type.VOLUMEFILE
    assert res.volume_path == volume_file.path
    odps.delete_resource(resource_name)
