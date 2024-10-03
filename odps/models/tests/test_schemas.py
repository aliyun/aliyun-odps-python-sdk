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
import time

import mock
import pytest

from ... import ODPS, options
from ...compat import BytesIO
from ...errors import NoSuchObject
from ...tests.core import force_drop_schema, tn

try:
    import pyarrow as pa
except ImportError:
    pa = None

TEST_SCHEMA_NAME = tn("pyodps_test_schema")
TEST_SCHEMA_NAME2 = tn("pyodps_test_schema2")
TEST_CLS_SCHEMA_NAME = tn("pyodps_test_cls_schema")
TEST_CLS_SCHEMA_NAME2 = tn("pyodps_test_cls_schema2")
TEST_RESOURCE_NAME = tn("pyodps_test_schema_res")


@pytest.fixture(scope="module", autouse=True)
def setup_module_schema(odps_with_schema):
    from ..schemas import _project_has_schema_api

    schemas = []
    _project_has_schema_api.pop(
        (odps_with_schema.endpoint, odps_with_schema.project), None
    )
    options.enable_schema = False

    for cls_schema_names in (TEST_CLS_SCHEMA_NAME, TEST_CLS_SCHEMA_NAME2):
        if odps_with_schema.exist_schema(cls_schema_names):
            schemas.append(odps_with_schema.get_schema(cls_schema_names))
        else:
            schemas.append(odps_with_schema.create_schema(cls_schema_names))
    try:
        yield
    finally:
        if "CI_MODE" in os.environ:
            for schema in schemas:
                force_drop_schema(schema)


@pytest.fixture(autouse=True)
def reset_schema_config(odps_with_schema):
    from ..schemas import _project_has_schema_api

    try:
        yield
    finally:
        _project_has_schema_api.pop(
            (odps_with_schema.endpoint, odps_with_schema.project), None
        )
        options.enable_schema = False


def _assert_schema_deleted(odps, schema_name):
    cnt = 0
    while odps.exist_schema(schema_name):
        time.sleep(10)
        cnt += 1
        if cnt > 120:
            break
    assert not odps.exist_schema(schema_name)


@pytest.mark.parametrize("legacy", (False, True))
def test_schemas(odps_with_schema, legacy):
    from ..schemas import _project_has_schema_api

    _project_has_schema_api[
        (odps_with_schema.endpoint, odps_with_schema.project)
    ] = not legacy

    schema_name = TEST_SCHEMA_NAME
    schema_name2 = TEST_SCHEMA_NAME2
    if odps_with_schema.exist_schema(schema_name):
        force_drop_schema(odps_with_schema.get_schema(schema_name))
        schema_name = "%s_%s" % (schema_name, int(time.time()))
    if odps_with_schema.exist_schema(schema_name2):
        force_drop_schema(odps_with_schema.get_schema(schema_name2))
        schema_name2 = "%s_%s" % (schema_name2, int(time.time()))

    for idx in range(3):
        try:
            assert not odps_with_schema.exist_schema(schema_name)
        except AssertionError:
            odps_with_schema.delete_schema(schema_name)

    schema = odps_with_schema.create_schema(schema_name)
    assert odps_with_schema.exist_schema(schema_name)
    assert schema.owner is not None

    assert any(s.name == schema_name for s in odps_with_schema.list_schemas())

    for idx in range(5):
        try:
            odps_with_schema.delete_schema(schema)
            _assert_schema_deleted(odps_with_schema, schema_name)
        except AssertionError:
            if idx >= 5:
                raise
        except NoSuchObject:
            break
        else:
            break

    schema = odps_with_schema.create_schema(schema_name2)
    assert odps_with_schema.exist_schema(schema_name2)

    for idx in range(5):
        try:
            schema.drop()
            _assert_schema_deleted(odps_with_schema, schema_name2)
        except AssertionError:
            if idx >= 5:
                raise
        except NoSuchObject:
            break
        else:
            break


def test_default_schema(odps_with_schema):
    new_odps = ODPS(
        project=odps_with_schema.project,
        endpoint=odps_with_schema.endpoint,
        account=odps_with_schema.account,
        schema=TEST_CLS_SCHEMA_NAME,
        overwrite_global=False,
    )

    project = new_odps.get_project()
    assert project._default_schema == TEST_CLS_SCHEMA_NAME

    schema = new_odps.get_schema()
    assert schema.project.name == odps_with_schema.project
    assert schema.name == TEST_CLS_SCHEMA_NAME

    res = new_odps.create_resource(
        TEST_RESOURCE_NAME, "file", fileobj=BytesIO(b"content")
    )
    assert new_odps.exist_resource(TEST_RESOURCE_NAME)
    assert res.schema.name == TEST_CLS_SCHEMA_NAME

    new_odps.schema = TEST_CLS_SCHEMA_NAME2
    assert project._default_schema == TEST_CLS_SCHEMA_NAME2

    assert not new_odps.exist_resource(TEST_RESOURCE_NAME)

    res.drop()


@pytest.mark.parametrize("schema_name", [TEST_CLS_SCHEMA_NAME, None])
def test_table_with_schema(odps_with_schema, schema_name):
    test_table_name = tn("pyodps_test_table_with_schema")
    test_partition = "pt='20220901'"

    odps = odps_with_schema

    if schema_name is None:
        options.enable_schema = True

    default_schema_name = "default" if odps.is_schema_namespace_enabled() else None

    odps.delete_table(test_table_name, schema=schema_name, if_exists=True)

    table = odps.create_table(
        test_table_name,
        ("col1 string", "pt string"),
        schema=schema_name,
        lifecycle=1,
    )
    assert table.get_schema().name == schema_name or default_schema_name

    tables = list(odps.list_tables(prefix=test_table_name, schema=schema_name))
    assert len(tables) >= 1
    assert tables[0].name == test_table_name
    assert tables[0].get_schema().name == schema_name or default_schema_name

    with table.open_writer(partition=test_partition, create_partition=True) as writer:
        writer.write([["test_record"]])
    assert table.exist_partition(test_partition)

    parts = list(table.partitions)
    assert len(parts) == 1
    assert parts[0].name == test_partition

    with table.open_reader(partition=test_partition) as reader:
        record = list(reader)[0]
        assert record[0] == "test_record"

    table.get_partition(test_partition).truncate()

    if pa is not None:
        with table.open_writer(partition=test_partition, arrow=True) as writer:
            arrow_array = pa.array(["abc", "def"])
            writer.write(pa.record_batch([arrow_array], names=["col1"]))

        with table.open_reader(
            reopen=True, partition=test_partition, arrow=True
        ) as reader:
            arrow_table = reader.read_all()
            assert arrow_table.num_rows == 2

    odps.execute_merge_files(table)
    odps.execute_merge_files(table.full_table_name)

    table.delete_partition(test_partition)
    assert not table.exist_partition(test_partition)

    table.drop()
    assert not odps.exist_table(test_table_name, schema=schema_name)


def test_get_table_with_schema_opt(odps_with_schema):
    odps = odps_with_schema
    test_table_name = tn("pyodps_test_table_with_schema2")

    try:
        options.enable_schema = True
        odps.delete_table(test_table_name, schema=TEST_CLS_SCHEMA_NAME, if_exists=True)
        odps.create_table(
            test_table_name, "col1 string", schema=TEST_CLS_SCHEMA_NAME, lifecycle=1
        )
        tb = odps.get_table(TEST_CLS_SCHEMA_NAME + "." + test_table_name)
        tb.reload()
        assert tb.name == test_table_name
        assert tb.get_schema().name == TEST_CLS_SCHEMA_NAME

        tb.drop()
    finally:
        options.enable_schema = False


def test_table_tenant_config(odps_with_schema):
    odps = odps_with_schema
    test_table_name = tn("pyodps_test_table_with_schema3")

    def _new_get_parameter(self, key, default=None):
        assert key == "odps.namespace.schema"
        return "true"

    with mock.patch("odps.models.tenant.Tenant.get_parameter", new=_new_get_parameter):
        assert odps.is_schema_namespace_enabled()

        odps.delete_table(test_table_name, if_exists=True)
        tb = odps.create_table(test_table_name, "col1 string", lifecycle=1)
        assert tb.get_schema().name == "default"

        tb = odps.get_table("default." + test_table_name)
        tb.reload()
        assert tb.name == test_table_name
        assert tb.get_schema().name == "default"

        tb.drop()


def test_file_resource_with_schema(odps_with_schema):
    test_file_res_name = tn("pyodps_test_file_resource")

    try:
        odps_with_schema.delete_resource(
            test_file_res_name, schema=TEST_CLS_SCHEMA_NAME
        )
    except NoSuchObject:
        pass

    res = odps_with_schema.create_resource(
        test_file_res_name,
        "file",
        fileobj=BytesIO(b"content"),
        schema=TEST_CLS_SCHEMA_NAME,
    )
    assert res.schema.name == TEST_CLS_SCHEMA_NAME

    assert odps_with_schema.exist_resource(
        test_file_res_name, schema=TEST_CLS_SCHEMA_NAME
    )

    resources = list(odps_with_schema.list_resources(schema=TEST_CLS_SCHEMA_NAME))
    assert 1 == len(resources)

    with odps_with_schema.open_resource(
        test_file_res_name, schema=TEST_CLS_SCHEMA_NAME, mode="rb"
    ) as res_reader:
        assert b"content" == res_reader.read()

    odps_with_schema.delete_resource(test_file_res_name, schema=TEST_CLS_SCHEMA_NAME)
    assert not odps_with_schema.exist_resource(
        test_file_res_name, schema=TEST_CLS_SCHEMA_NAME
    )


def test_table_resource_with_schema(odps_with_schema):
    test_table_res_name = tn("pyodps_test_table_resource")
    test_res_table_name = tn("pyodps_test_resource_table")

    try:
        odps_with_schema.delete_resource(
            test_table_res_name, schema=TEST_CLS_SCHEMA_NAME
        )
    except NoSuchObject:
        pass

    test_res_table = odps_with_schema.create_table(
        test_res_table_name,
        "col1 string",
        schema=TEST_CLS_SCHEMA_NAME2,
        lifecycle=1,
        if_not_exists=True,
    )

    res = odps_with_schema.create_resource(
        test_table_res_name,
        "table",
        schema=TEST_CLS_SCHEMA_NAME,
        table_name=test_res_table.full_table_name,
    )
    assert res.schema.name == TEST_CLS_SCHEMA_NAME
    assert res.table.full_table_name == test_res_table.full_table_name

    odps_with_schema.delete_resource(test_table_res_name, schema=TEST_CLS_SCHEMA_NAME)
    assert not odps_with_schema.exist_resource(
        test_table_res_name, schema=TEST_CLS_SCHEMA_NAME
    )

    odps_with_schema.delete_table(test_res_table_name, schema=TEST_CLS_SCHEMA_NAME2)


def test_function_with_resource(odps_with_schema):
    from .test_functions import FUNCTION_CONTENT

    test_func_res_name = tn("pyodps_test_function_res")
    test_func_res_file = test_func_res_name + ".py"
    test_func_name = tn("pyodps_test_function")

    odps = odps_with_schema

    try:
        odps.delete_resource(test_func_res_file, schema=TEST_CLS_SCHEMA_NAME2)
    except NoSuchObject:
        pass
    try:
        odps.delete_function(test_func_name, schema=TEST_CLS_SCHEMA_NAME)
    except NoSuchObject:
        pass

    content = BytesIO(FUNCTION_CONTENT.encode())
    res = odps.create_resource(
        test_func_res_file, "py", fileobj=content, schema=TEST_CLS_SCHEMA_NAME2
    )
    func = odps.create_function(
        test_func_name,
        class_type=test_func_res_name + ".MyPlus",
        resources=[res],
        schema=TEST_CLS_SCHEMA_NAME,
    )
    assert func.schema.name == TEST_CLS_SCHEMA_NAME

    assert odps_with_schema.exist_function(test_func_name, schema=TEST_CLS_SCHEMA_NAME)

    funcs = list(odps_with_schema.list_functions(schema=TEST_CLS_SCHEMA_NAME))
    assert 1 == len(funcs)
    assert funcs[0].name == test_func_name
    assert funcs[0].schema.name == TEST_CLS_SCHEMA_NAME
    assert funcs[0].resources[0].name == test_func_res_file
    assert funcs[0].resources[0].schema.name == TEST_CLS_SCHEMA_NAME2

    odps.delete_function(test_func_name, schema=TEST_CLS_SCHEMA_NAME)
    assert not odps_with_schema.exist_function(
        test_func_name, schema=TEST_CLS_SCHEMA_NAME
    )
    odps.delete_resource(test_func_res_file, schema=TEST_CLS_SCHEMA_NAME2)
