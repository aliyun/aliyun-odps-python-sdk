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

from ...types import Array, Bigint, Blob, Double, Float, Vector
from ..errors import StorageClientError, StorageServiceError
from ..models import InstanceIdentifier, TableIdentifier
from ..models.requests import (
    BatchCompatibleCommitRequest,
    CreateTableReadSessionRequest,
)
from ..models.responses import (
    BatchCompatibleSessionResponse,
    CreateTableReadSessionResponse,
)
from ..models.schema import ReadSchema, WriteSchema, _contains_blob, _has_nested_blob
from ..models.types import parse_write_schema_type
from ..options import BlobWriteItem, IncrementalReadOptions, SplitOptions

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
    assert nested == {"arr": 5, "arr.element": 6}


@pytest.mark.parametrize(
    "sub_type_code,element_type,dimension",
    [
        (14, Float, 128),
        (1, Double, 256),
    ],
)
def test_parse_write_schema_type_vector(sub_type_code, element_type, dimension):
    """Wire code 23 maps to VECTOR with the given element type and dimension."""
    type_info = {
        "Type": 23,
        "MemberName": "vec",
        "ColumnId": 7,
        "Nullable": True,
        "SubTypes": [{"Type": sub_type_code, "ColumnId": 8, "Nullable": True}],
        "VectorDimension": dimension,
    }
    nested = {}
    dt = parse_write_schema_type(type_info, "vec", nested)

    assert isinstance(dt, Vector)
    assert isinstance(dt.element_type, element_type)
    assert dt.dimension == dimension
    assert nested == {"vec": 7, "vec.element": 8}


@pytest.mark.parametrize(
    "type_info,match",
    [
        # Missing VectorDimension -> defaults to 0 -> rejected.
        (
            {
                "Type": 23,
                "SubTypes": [{"Type": 14, "ColumnId": 8, "Nullable": True}],
            },
            "positive VectorDimension",
        ),
        # Wrong number of sub-types.
        (
            {
                "Type": 23,
                "SubTypes": [
                    {"Type": 14, "ColumnId": 8, "Nullable": True},
                    {"Type": 1, "ColumnId": 9, "Nullable": True},
                ],
                "VectorDimension": 128,
            },
            "exactly one sub-type",
        ),
    ],
)
def test_parse_write_schema_type_vector_invalid(type_info, match):
    with pytest.raises(ValueError, match=match):
        parse_write_schema_type(type_info, "vec", {})


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


# ---------------------------------------------------------------------------
# v2 client capability wire fields
# ---------------------------------------------------------------------------


_V2_OPT_IN_FIELDS = [
    pytest.param(
        SplitOptions(),
        SplitOptions(bucket_sort=True),
        "BucketSort",
        True,
        id="split-options-bucket-sort",
    ),
    pytest.param(
        IncrementalReadOptions(),
        IncrementalReadOptions(start_txn_id="txn-1"),
        "StartTxnId",
        "txn-1",
        id="incremental-read-start-txn-id",
    ),
    pytest.param(
        CreateTableReadSessionRequest(),
        CreateTableReadSessionRequest(enable_estimate_stats=True),
        "EnableEstimateStats",
        True,
        id="read-session-estimate-stats",
    ),
    pytest.param(
        CreateTableReadSessionRequest(),
        CreateTableReadSessionRequest(enable_extended_arrow_ipc=True),
        "EnableExtendedArrowIPC",
        True,
        id="read-session-extended-arrow-ipc",
    ),
    pytest.param(
        CreateTableReadSessionRequest(),
        CreateTableReadSessionRequest(unique_map_key=True),
        "UniqueMapKey",
        True,
        id="read-session-unique-map-key",
    ),
]


@pytest.mark.parametrize("default,opted_in,wire_key,expected", _V2_OPT_IN_FIELDS)
def test_v2_wire_field_is_opt_in(default, opted_in, wire_key, expected):
    """New v2 fields stay off the wire until explicitly requested."""
    assert wire_key not in default.to_dict()
    assert opted_in.to_dict()[wire_key] == expected


def test_incremental_read_options_wire_mapping():
    assert IncrementalReadOptions(
        version="Auto", from_=100, to=200, start_txn_id="txn-1"
    ).to_dict() == {"Mode": "Auto", "From": 100, "To": 200, "StartTxnId": "txn-1"}


def test_split_options_defaults_to_size_split_mode():
    assert SplitOptions().to_dict()["SplitMode"] == "Size"


def test_create_read_session_response_v2_fields():
    resp = CreateTableReadSessionResponse.from_dict(
        {
            "SessionId": "s1",
            "ExtendedArrowIPCEnabled": True,
            "LatestTxnId": "txn-9",
            "RecordCount": 123,
        }
    )
    assert resp.extended_arrow_ipc_enabled is True
    assert resp.latest_txn_id == "txn-9"
    assert resp.record_count == 123

    # Absent RecordCount is unknown (-1), not zero.
    empty = CreateTableReadSessionResponse.from_dict({"SessionId": "s2"})
    assert empty.record_count == -1
    assert empty.extended_arrow_ipc_enabled is False
    assert empty.latest_txn_id is None

    # Default instance also reports -1.
    assert CreateTableReadSessionResponse().record_count == -1


def test_batch_compatible_session_response_v2_fields():
    resp = BatchCompatibleSessionResponse.from_dict(
        {
            "SessionId": "w1",
            "SupportedDataFormat": [
                {"Type": "Arrow", "Version": "V5"},
                {"Type": "DataBlock", "Version": "V1"},
            ],
            "RequiredOrdering": [
                {
                    "Name": "pk",
                    "SortDirection": "ASCENDING",
                    "NullOrdering": "NULLS_FIRST",
                }
            ],
            "RequiredDistribution": {
                "Type": "Hash",
                "ClusterKeys": ["pk"],
                "BucketsNumber": 16,
            },
        }
    )
    assert resp.session_id == "w1"
    assert [fmt.type for fmt in resp.supported_data_format] == ["Arrow", "DataBlock"]
    assert len(resp.required_ordering) == 1
    assert resp.required_ordering[0].name == "pk"
    assert resp.required_ordering[0].sort_direction == "ASCENDING"
    assert resp.required_ordering[0].null_ordering == "NULLS_FIRST"
    assert resp.required_distribution.type == "Hash"
    assert resp.required_distribution.cluster_keys == ["pk"]
    assert resp.required_distribution.buckets_number == 16

    # Absent optional blocks default to empty / None.
    empty = BatchCompatibleSessionResponse.from_dict({"SessionId": "w2"})
    assert empty.supported_data_format == []
    assert empty.required_ordering == []
    assert empty.required_distribution is None


# ---------------------------------------------------------------------------
# batch-compatible commit request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeout", [None, 1, 30, 300, 600])
def test_batch_compatible_commit_request_wait_timeout_wire_field(timeout):
    body = BatchCompatibleCommitRequest(
        commit_messages=["m0"], wait_flying_writers_timeout_seconds=timeout
    ).to_dict()

    assert body["CommitMessages"] == ["m0"]
    if timeout is None:
        # Inert by default: existing wire bodies are unchanged.
        assert "WaitFlyingWritersTimeoutSeconds" not in body
    else:
        assert body["WaitFlyingWritersTimeoutSeconds"] == timeout


@pytest.mark.parametrize("timeout", [0, -1, 601, "30", 1.5, True])
def test_batch_compatible_commit_request_wait_timeout_validation(timeout):
    with pytest.raises(ValueError, match=r"1, 600"):
        BatchCompatibleCommitRequest(
            commit_messages=[], wait_flying_writers_timeout_seconds=timeout
        )
