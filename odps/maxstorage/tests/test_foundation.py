# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

"""Unit tests for odps.maxstorage foundation: errors, identifiers, enums,
types, schema, options, requests, responses."""

from io import BytesIO

import pytest

from ...types import Array, Bigint, Blob
from ..errors import StorageClientError, StorageServiceError
from ..models import InstanceIdentifier, TableIdentifier
from ..models.schema import ReadSchema, WriteSchema, _contains_blob, _has_nested_blob
from ..models.types import parse_write_schema_type
from ..options import BlobWriteItem

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def test_error_messages():
    err = StorageClientError("bad request")
    assert "bad request" in str(err)
    # StorageServiceError takes (http_status, error_code, message, request_id)
    err2 = StorageServiceError(500, "InternalError", "server error")
    assert "server error" in str(err2)
    assert err2.http_status == 500


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------


def test_table_identifier():
    tid = TableIdentifier("proj", "tbl", "sch")
    assert tid.project == "proj"
    assert tid.table == "tbl"
    assert tid.schema == "sch"
    target = tid.to_target()
    assert "projects.proj" in target
    assert "tables.tbl" in target
    assert "schemas.sch" in target


def test_table_identifier_no_schema():
    tid = TableIdentifier("proj", "tbl")
    target = tid.to_target()
    assert target == "projects.proj.tables.tbl"


def test_instance_identifier():
    iid = InstanceIdentifier("proj", "inst123")
    assert iid.project == "proj"
    assert iid.instance == "inst123"
    target = iid.to_target()
    assert target == "projects.proj.instances.inst123"


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------


def test_parse_write_schema_type_blob():
    nested = {}
    type_info = {"Type": 22, "MemberName": "data", "ColumnId": 2, "Nullable": True}
    dt = parse_write_schema_type(type_info, "data", nested)

    assert isinstance(dt, Blob)


def test_parse_write_schema_type_array():
    nested = {}
    type_info = {
        "Type": 17,
        "MemberName": "arr",
        "ColumnId": 5,
        "Nullable": True,
        "SubTypes": [{"Type": 0, "ColumnId": 6, "Nullable": True}],
    }
    dt = parse_write_schema_type(type_info, "arr", nested)

    assert isinstance(dt, Array)
    assert isinstance(dt.value_type, Bigint)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def _blob_col(name, cid, type_code=22, subtypes=None):
    """Build a single write-schema column dict for BLOB (type 22) or a
    container type with the given subtypes."""
    ct = {"MemberName": name, "ColumnId": cid, "Type": type_code, "Nullable": True}
    if subtypes:
        ct["SubTypes"] = subtypes
    return {"columnType": ct}


def _ws(*cols):
    return WriteSchema.from_dict({"DataColumns": list(cols), "SystemColumns": []})


def test_write_schema_from_dict():
    raw = {
        "DataColumns": [
            {
                "columnType": {
                    "MemberName": "id",
                    "ColumnId": 0,
                    "Type": 0,
                    "Nullable": True,
                }
            },
            {
                "columnType": {
                    "MemberName": "name",
                    "ColumnId": 1,
                    "Type": 4,
                    "Nullable": True,
                }
            },
            {
                "columnType": {
                    "MemberName": "data",
                    "ColumnId": 2,
                    "Type": 22,
                    "Nullable": True,
                }
            },
        ],
        "SystemColumns": [
            {
                "columnType": {
                    "MemberName": "__operation",
                    "ColumnId": 3,
                    "Type": 6,
                    "Nullable": True,
                }
            },
        ],
    }
    ws = WriteSchema.from_dict(raw)
    assert len(ws.columns) == 3
    assert ws.columns[0].name == "id"
    assert ws.columns[2].name == "data"
    assert ws.columns[2].column_id == 2
    assert len(ws.system_columns) == 1
    assert ws.system_columns[0].name == "__operation"


def test_write_schema_blob_detection():
    ws = _ws(_blob_col("data", 2))
    blob_ids = ws.find_all_blob_column_ids()
    assert blob_ids == {"data": 2}
    assert _contains_blob(ws.columns[0].type)
    assert not ws.has_nested_blob_columns()


def test_write_schema_nested_blob():
    ws = _ws(_blob_col("arr", 0, 17, [{"Type": 22, "ColumnId": 1, "Nullable": True}]))
    blob_ids = ws.find_all_blob_column_ids()
    assert "arr.element" in blob_ids
    assert ws.has_nested_blob_columns()
    assert _has_nested_blob(ws.columns[0].type)


def test_resolve_blob_column_name_auto_single():
    ws = _ws(_blob_col("img", 3))
    assert ws.resolve_blob_column_name() == ("img", 3)


@pytest.mark.parametrize(
    "cols, match",
    [
        # two top-level BLOB columns
        ([_blob_col("a", 1), _blob_col("b", 2)], "Cannot auto-select"),
        # one top-level BLOB plus a nested BLOB inside ARRAY<BLOB>
        (
            [
                _blob_col("a", 1),
                _blob_col(
                    "arr", 0, 17, [{"Type": 22, "ColumnId": 2, "Nullable": True}]
                ),
            ],
            "nested BLOB",
        ),
    ],
    ids=["multi-blob", "nested-blob"],
)
def test_resolve_blob_column_name_auto_blocked(cols, match):
    ws = _ws(*cols)
    with pytest.raises(StorageClientError, match=match):
        ws.resolve_blob_column_name()


def test_resolve_blob_column_name_explicit():
    ws = _ws(_blob_col("a", 1), _blob_col("b", 2))
    assert ws.resolve_blob_column_name("a") == ("a", 1)
    assert ws.resolve_blob_column_name("b") == ("b", 2)


def test_resolve_blob_column_name_explicit_unknown():
    ws = _ws(_blob_col("a", 1))
    with pytest.raises(StorageClientError, match="Unknown BLOB column name"):
        ws.resolve_blob_column_name("nope")


def test_read_schema_from_dict():
    raw = {
        "DataColumns": [
            {"Name": "id", "Type": "bigint", "Comment": ""},
            {"Name": "name", "Type": "string", "Comment": ""},
        ],
    }
    rs = ReadSchema.from_dict(raw)
    assert len(rs.columns) == 2
    assert rs.columns[0].name == "id"
    assert rs.columns[1].name == "name"


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def test_blob_write_item_serialization():
    item = BlobWriteItem(data=b"hello", column_id=0)
    buf = BytesIO()
    item.write_frame_to(buf)
    serialized = buf.getvalue()
    assert isinstance(serialized, (bytes, bytearray))
    assert len(serialized) > 0
