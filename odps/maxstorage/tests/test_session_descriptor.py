# -*- coding: utf-8 -*-
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

"""Tests for cross-process session descriptors (export / restore / JSON).

Covers the descriptor model itself (canonical JSON, whitelisting,
validation), read-session export/restore, and write-session export/restore
(including the descriptor-only restrictions after restore).
"""

import json

import mock
import pytest

from .. import SessionDescriptor, SessionKind
from ..base import MaxStorageClient
from ..errors import StorageClientError
from ..models.enums import SplitMode, WriteMode
from ..models.identifier import TableIdentifier
from ..models.requests import CreateTableReadSessionRequest
from ..models.responses import (
    BatchCompatibleSessionResponse,
    CreateTableReadSessionResponse,
)
from ..options import IncrementalReadOptions, SplitOptions
from ..read.table_session import TableReadSession
from ..write.session import TableWriteSession

TABLE = {"project": "p", "schema": "s", "table": "t"}


def _read_metadata(**over):
    meta = {
        "SessionId": "r-1",
        "SessionStatus": "NORMAL",
        "SplitMode": "RowOffset",
        "SplitsCount": 3,
        "RecordCount": 8,
        "DataSchema": {
            "DataColumns": [
                {"Name": "a", "Type": "bigint", "Nullable": False, "ColumnId": 0},
                {
                    "Name": "b",
                    "Type": "string",
                    "Nullable": True,
                    "Comment": "",
                    "ColumnId": 1,
                },
                {
                    "Name": "c",
                    "Type": "array<struct<S:bigint>>",
                    "Nullable": True,
                    "ColumnId": 2,
                },
            ],
            "PartitionColumns": [],
            "SystemColumns": [],
        },
        "SupportedDataFormat": [],
        "EnableLargeString": False,
    }
    meta.update(over)
    return meta


def _write_metadata(**over):
    meta = {
        "SessionId": "w-1",
        "SessionStatus": "NORMAL",
        "DataSchema": {
            "DataColumns": [
                {"Name": "a", "Type": "bigint", "Nullable": False},
                {"Name": "b", "Type": "string", "Nullable": True},
            ],
            "PartitionColumns": [],
        },
        "MaxBlockNumber": 16,
        "EnhanceWriteCheck": True,
        "CommitStarted": False,
    }
    meta.update(over)
    return meta


class NoRpcStub:
    """A stub that fails on any RPC — restored sessions must not call one."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise AssertionError("unexpected RPC: %s" % name)

        return fail


def _reparse(root):
    """Round-trip a raw descriptor dict through the JSON parser."""
    return SessionDescriptor.from_json(json.dumps(root))


def _mutate_metadata(descriptor, **over):
    """*descriptor* with canonical wire-metadata overrides applied."""
    root = json.loads(descriptor.to_json())
    root["metadata"].update(over)
    return _reparse(root)


def _make_read_descriptor(**create_kwargs):
    kwargs = {"route_token": "route-1", "rows_per_split": 3}
    kwargs.update(create_kwargs)
    return SessionDescriptor.create(
        SessionKind.BATCH_READ, TABLE, _read_metadata(), **kwargs
    )


def _make_write_descriptor(**create_kwargs):
    kwargs = {
        "route_token": "route-w",
        "arrow": {"timestampUnit": "NANO", "datetimeUnit": "MILLI"},
    }
    kwargs.update(create_kwargs)
    return SessionDescriptor.create(
        SessionKind.BATCH_WRITE, TABLE, _write_metadata(), **kwargs
    )


def _restore_read(descriptor):
    """Restore a read session from a descriptor through the public path."""
    return TableReadSession.from_descriptor(
        NoRpcStub(), SessionDescriptor.from_json(descriptor.to_json())
    )


def _restore_write(descriptor):
    """Restore a write session from a descriptor through the public path."""
    return TableWriteSession.from_descriptor(
        NoRpcStub(), SessionDescriptor.from_json(descriptor.to_json())
    )


# ---------------------------------------------------------------------------
# descriptor model
# ---------------------------------------------------------------------------


def test_descriptor_json_round_trip_and_accessors():
    d = _make_write_descriptor()
    restored = SessionDescriptor.from_json(d.to_json())

    assert restored == d
    assert restored.kind == SessionKind.BATCH_WRITE
    assert restored.table == TABLE
    tid = restored.table_identifier
    assert (tid.project, tid.table, tid.schema) == ("p", "t", "s")
    assert restored.session_id == "w-1"
    assert restored.session_status == "NORMAL"
    assert restored.route_token == "route-w"
    assert restored.arrow == {"timestampUnit": "NANO", "datetimeUnit": "MILLI"}

    # metadata is exposed in wire (PascalCase) form
    assert restored.metadata["SessionId"] == "w-1"
    assert restored.metadata["MaxBlockNumber"] == 16
    assert restored.metadata["CommitStarted"] is False

    # a read descriptor carries rowsPerSplit
    rd = _make_read_descriptor()
    assert rd.rows_per_split == 3
    assert json.loads(rd.to_json())["readOptions"]["rowsPerSplit"] == 3


def test_descriptor_default_fields_are_optional():
    d = SessionDescriptor.create(
        SessionKind.BATCH_WRITE,
        {"project": "p", "table": "t"},  # no schema
        _write_metadata(),
        arrow={"timestampUnit": "NANO", "datetimeUnit": "MILLI"},
    )
    restored = SessionDescriptor.from_json(d.to_json())
    assert restored == d
    assert restored.table == {"project": "p", "table": "t"}


def test_descriptor_whitelist_strips_unknown_fields():
    root = json.loads(_make_read_descriptor().to_json())
    root["extraRoot"] = "nope"
    root["metadata"]["extraMeta"] = "nope"
    root["metadata"]["dataSchema"]["extraSchema"] = "nope"
    root["metadata"]["dataSchema"]["dataColumns"][0]["extraCol"] = "nope"

    restored = _reparse(root)
    assert "extraRoot" not in restored._root
    assert "extraMeta" not in restored._root["metadata"]
    schema = restored._root["metadata"]["dataSchema"]
    assert "extraSchema" not in schema
    assert "extraCol" not in schema["dataColumns"][0]
    # whitelisted fields survive
    assert schema["dataColumns"][0]["columnId"] == 0
    assert schema["dataColumns"][2]["type"] == "array<struct<S:bigint>>"
    assert restored.route_token == "route-1"


@pytest.mark.parametrize(
    "mutate,error_match",
    [
        # envelope and table identity
        (lambda r: r.update(version=2), "version"),
        (lambda r: r.update(provider="v1"), "v2 session descriptor"),
        (lambda r: r.update(kind="batch-scan"), "kind"),
        (lambda r: r.__setitem__("table", "scalar"), "table"),
        (lambda r: r["table"].pop("project"), "project"),
        (lambda r: r["table"].pop("table"), "table"),
        # session metadata
        (lambda r: r.__setitem__("metadata", "scalar"), "metadata"),
        (lambda r: r["metadata"].pop("sessionId"), "sessionId"),
        (lambda r: r["metadata"].pop("sessionStatus"), "sessionStatus"),
        (lambda r: r["metadata"].__setitem__("dataSchema", None), "dataSchema"),
        # read split plan
        (lambda r: r["metadata"].update(splitMode="Teleport"), "split mode"),
        (lambda r: r["metadata"].update(splitMode=None), "splitMode"),
        (
            lambda r: r["metadata"].update(splitMode="Size", splitsCount=-1),
            "splitsCount",
        ),
        (
            lambda r: r["metadata"].update(splitMode="Size", splitsCount=True),
            "splitsCount",
        ),
        (lambda r: r["metadata"].update(recordCount=-2), "recordCount"),
        (
            lambda r: r["metadata"].update(splitMode="Bucket", splitBucketId=[0, 1]),
            "splitsCount",
        ),
        (
            lambda r: r["metadata"].update(
                splitMode="Bucket", splitsCount=2, splitBucketId=[-1, 1]
            ),
            "Invalid bucket id",
        ),
    ],
)
def test_descriptor_validation_rejects_bad_payloads(mutate, error_match):
    root = json.loads(_make_read_descriptor().to_json())
    mutate(root)
    with pytest.raises(ValueError, match=error_match):
        _reparse(root)


@pytest.mark.parametrize(
    "mutate,error_match",
    [
        (lambda r: r.pop("arrow"), "arrow"),
        (lambda r: r.__setitem__("arrow", {}), "timestampUnit"),
        (lambda r: r["arrow"].__setitem__("timestampUnit", "HOUR"), "unit"),
        (lambda r: r["arrow"].__setitem__("datetimeUnit", "MINUTE"), "unit"),
    ],
)
def test_descriptor_validation_arrow(mutate, error_match):
    root = json.loads(_make_write_descriptor().to_json())
    mutate(root)
    with pytest.raises(ValueError, match=error_match):
        _reparse(root)


def test_descriptor_zero_rows_per_split_is_absent():
    # a zero value is treated as "absent" on the create path
    d = SessionDescriptor.create(
        SessionKind.BATCH_READ, TABLE, _read_metadata(), rows_per_split=0
    )
    assert "readOptions" not in d.to_json()


@pytest.mark.parametrize("rows", [0, -3])
def test_descriptor_rejects_non_positive_rows_per_split(rows):
    root = json.loads(_make_read_descriptor().to_json())
    root["readOptions"]["rowsPerSplit"] = rows
    with pytest.raises(ValueError, match="rowsPerSplit"):
        _reparse(root)


# ---------------------------------------------------------------------------
# read session export / restore
# ---------------------------------------------------------------------------


def _plain_read_session(meta, request=None):
    """Build a read session straight from wire metadata, without any RPC."""
    return TableReadSession(
        NoRpcStub(),
        TableIdentifier("p", "t", "s"),
        request if request is not None else CreateTableReadSessionRequest(),
        _response=CreateTableReadSessionResponse.from_dict(meta),
    )


def _driver_read_session(split_options=None, incremental=None):
    """Build a driver-side session backed by a stub that never gets called."""
    return _plain_read_session(
        _read_metadata(),
        CreateTableReadSessionRequest(
            split_options=split_options,
            incremental_read=incremental is not None,
            incremental_read_options=incremental,
        ),
    )


def test_read_export_restore_round_trip_preserves_rows_per_split():
    session = _driver_read_session(
        split_options=SplitOptions(split_mode=SplitMode.ROW_OFFSET, split_number=3)
    )
    d = session.to_descriptor()
    assert d.rows_per_split == 3

    restored = _restore_read(d)
    # splits are materialized from recordCount / rowsPerSplit, no RPC
    assert [(sp.offset, sp.length) for sp in restored.splits] == [
        (0, 3),
        (3, 3),
        (6, 2),
    ]
    assert restored.id == "r-1"
    assert [c.name for c in restored.table_schema.columns] == ["a", "b", "c"]
    # re-export is byte-identical
    assert restored.to_descriptor() == d


def test_read_export_keeps_partition_columns_out_of_data_columns():
    # ``OdpsSchema.columns`` is data + partitions; serializing it as
    # DataColumns duplicated every partition column on the restored schema.
    meta = _read_metadata()
    meta["DataSchema"]["PartitionColumns"] = [
        {"Name": "pt", "Type": "string", "Nullable": True}
    ]
    session = _plain_read_session(meta)
    d = session.to_descriptor()
    wire = json.loads(d.to_json())["metadata"]["dataSchema"]
    assert [c["name"] for c in wire["dataColumns"]] == ["a", "b", "c"]
    assert [c["name"] for c in wire["partitionColumns"]] == ["pt"]

    restored = _restore_read(d)
    assert [c.name for c in restored.table_schema.simple_columns] == ["a", "b", "c"]
    assert [c.name for c in restored.table_schema.partitions] == ["pt"]
    assert restored.to_descriptor() == d


def test_read_restore_preserves_wire_type_strings():
    d = _make_read_descriptor()
    assert _restore_read(d).to_descriptor() == d
    wire_cols = json.loads(d.to_json())["metadata"]["dataSchema"]["dataColumns"]
    assert wire_cols[2]["type"] == "array<struct<S:bigint>>"


def test_read_export_infers_rows_per_split_from_materialized_splits():
    # ROW_OFFSET without explicit split options: the request carries no chunk
    # size, so the export falls back to the materialized splits.
    session = _plain_read_session(_read_metadata(RecordCount=10))
    d = session.to_descriptor()
    assert [sp.length for sp in session.splits] == [10]
    assert d.rows_per_split == 10
    assert _restore_read(d).to_descriptor() == d


def test_read_export_split_options_without_chunk_size_uses_single_split():
    # A request that only asked the server to plan splits exposes no chunk
    # size, so the plan degenerates to one split over the whole range
    # instead of one split per row.
    session = _driver_read_session(
        split_options=SplitOptions(split_mode=SplitMode.ROW_OFFSET, split_number=0)
    )
    assert [(sp.offset, sp.length) for sp in session.splits] == [(0, 8)]
    assert session.to_descriptor().rows_per_split == 8


def test_read_restore_row_offset_requires_rows_per_split():
    meta = _read_metadata(RecordCount=8, SplitsCount=3)
    d = SessionDescriptor.create(
        SessionKind.BATCH_READ, TABLE, meta
    )  # no rows_per_split
    with pytest.raises(ValueError, match="rowsPerSplit"):
        _restore_read(d)


def test_read_restore_empty_table_allows_missing_rows_per_split():
    meta = _read_metadata(RecordCount=0, SplitsCount=0)
    d = SessionDescriptor.create(SessionKind.BATCH_READ, TABLE, meta)
    restored = _restore_read(d)
    assert restored.splits == []
    assert restored.to_descriptor() == d


def test_read_export_incremental_options_come_from_request():
    session = _driver_read_session(
        incremental=IncrementalReadOptions(version="VersionRange", from_=5, to=9)
    )
    d = session.to_descriptor()
    assert d.kind == SessionKind.INCREMENTAL_READ
    assert d.metadata["IncrementalReadOptions"] == {
        "Mode": "VersionRange",
        "StartVersion": 5,
        "EndVersion": 9,
    }
    assert _restore_read(d).to_descriptor() == d


def test_read_export_supported_data_format_round_trip():
    meta = _read_metadata(
        SupportedDataFormat=[
            {"Type": "Arrow", "Version": "V5"},
        ]
    )
    d = SessionDescriptor.create(SessionKind.BATCH_READ, TABLE, meta, rows_per_split=3)
    restored = _restore_read(d)
    assert restored.supported_data_formats[0].type == "Arrow"
    assert restored.to_descriptor() == d


# ---------------------------------------------------------------------------
# write session export / restore
# ---------------------------------------------------------------------------


class FakeBatchWriteStub(NoRpcStub):
    """Stand-in ``StorageStub`` returning a canned batch-compatible session."""

    def __init__(self, fail_commit=False):
        self.fail_commit = fail_commit
        self.abort_calls = []

    def create_batch_compatible_session(self, table_id, request):
        return BatchCompatibleSessionResponse.from_dict(_write_metadata())

    def commit_batch_compatible_session(self, *args, **kwargs):
        if self.fail_commit:
            raise StorageClientError("commit failed")
        return BatchCompatibleSessionResponse.from_dict(
            _write_metadata(SessionStatus="COMMITTED", CommitStarted=True)
        )

    def abort_batch_compatible_session(self, *args, **kwargs):
        self.abort_calls.append(args)
        return BatchCompatibleSessionResponse.from_dict(
            _write_metadata(SessionStatus="ABORTED")
        )


class AbortOutcomeStub(FakeBatchWriteStub):
    """Coordinator stub whose remote abort either succeeds or raises."""

    def __init__(self, fail_abort=False):
        super().__init__()
        self.fail_abort = fail_abort
        self.abort_calls = 0

    def abort_table_write_session(self, *args, **kwargs):
        self.abort_calls += 1
        if self.fail_abort:
            raise StorageClientError("abort failed")
        return BatchCompatibleSessionResponse.from_dict(
            _write_metadata(SessionStatus="ABORTED")
        )


def _coordinator_write_session(stub=None):
    return TableWriteSession(
        stub or FakeBatchWriteStub(),
        TableIdentifier("p", "t", "s"),
        write_mode=WriteMode.BATCH_COMPATIBLE,
    )


def test_write_export_restore_round_trip():
    session = _coordinator_write_session()
    d = session.to_descriptor()
    assert d.kind == SessionKind.BATCH_WRITE
    assert d.arrow["timestampUnit"] in ("NANO", "MICRO")

    restored = _restore_write(d)
    assert restored.descriptor_only
    assert restored.id == "w-1"
    assert restored.max_block_number == 16
    wire_cols = d.metadata["DataSchema"]["DataColumns"]
    assert [c["Name"] for c in wire_cols] == ["a", "b"]
    assert restored.to_descriptor() == d


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: s.commit(block_results=[]), id="commit"),
        pytest.param(lambda s: s.abort(), id="abort"),
    ],
)
def test_write_restore_rejects_commit_and_abort(call):
    restored = _restore_write(_make_write_descriptor())
    with pytest.raises(StorageClientError, match="cannot commit or abort"):
        call(restored)
    # close() must not abort the global session
    restored.close()


@pytest.mark.parametrize(
    "metadata_over,error_match",
    [
        ({"commitStarted": True}, "after commit started"),
        ({"sessionStatus": "COMMITTING"}, "outside the NORMAL"),
    ],
)
def test_write_restore_blocks_new_writers(metadata_over, error_match):
    d = _mutate_metadata(_make_write_descriptor(), **metadata_over)
    restored = _restore_write(d)
    with pytest.raises(StorageClientError, match=error_match):
        restored.open_block_writer(0, 0)


@pytest.mark.parametrize(
    "fail_commit,error_match,status",
    [
        (False, None, "COMMITTED"),
        (True, "commit failed", "NORMAL"),
    ],
)
def test_write_export_after_commit(fail_commit, error_match, status):
    session = _coordinator_write_session(FakeBatchWriteStub(fail_commit=fail_commit))
    if error_match is None:
        session.commit(block_results=[])
    else:
        with pytest.raises(StorageClientError, match=error_match):
            session.commit(block_results=[])

    d = session.to_descriptor()
    # after a failed commit the status stays NORMAL but CommitStarted is
    # true, so restored workers must not open new writers
    assert d.metadata["SessionStatus"] == status
    assert d.metadata["CommitStarted"] is True

    with pytest.raises(StorageClientError, match="after commit started"):
        _restore_write(d).open_block_writer(0, 0)


@pytest.mark.parametrize("fail_abort", [False, True])
def test_write_close_finalizes_handle(fail_abort):
    stub = AbortOutcomeStub(fail_abort=fail_abort)
    session = _coordinator_write_session(stub)
    session.close()

    assert stub.abort_calls == 1
    # the handle stays closed even when the remote abort failed
    with pytest.raises(StorageClientError, match="already closed"):
        session.open_block_writer(0, 0)
    with pytest.raises(StorageClientError, match="already closed"):
        session.commit(block_results=[])


@pytest.mark.parametrize("fail_abort,status", [(False, "ABORTED"), (True, "UNKNOWN")])
def test_write_export_after_close_never_reports_normal(fail_abort, status):
    session = _coordinator_write_session(AbortOutcomeStub(fail_abort=fail_abort))
    session.close()

    d = session.to_descriptor()
    assert d.metadata["SessionStatus"] == status
    with pytest.raises(StorageClientError, match="outside the NORMAL"):
        _restore_write(d).open_block_writer(0, 0)


def test_write_export_rejects_non_batch_compatible_mode():
    stub = mock.MagicMock()
    stub.create_table_write_session.return_value = mock.MagicMock(
        session_id="sess-1", route_token="rt"
    )
    session = TableWriteSession(stub, TableIdentifier("p", "t", "s"))
    with pytest.raises(NotImplementedError, match="BATCH_COMPATIBLE"):
        session.to_descriptor()


# ---------------------------------------------------------------------------
# cross-kind restore guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "session_cls,descriptor_factory,error_match",
    [
        (TableReadSession, _make_write_descriptor, "read session descriptor"),
        (TableWriteSession, _make_read_descriptor, "batch-write session descriptor"),
    ],
)
def test_restore_rejects_descriptor_of_other_kind(
    session_cls, descriptor_factory, error_match
):
    with pytest.raises(ValueError, match=error_match):
        session_cls.from_descriptor(NoRpcStub(), descriptor_factory())


# ---------------------------------------------------------------------------
# client integration
# ---------------------------------------------------------------------------


def _client_with_stub(stub):
    client = MaxStorageClient(
        account=mock.MagicMock(), endpoint="http://tunnel", project="p"
    )
    client._stub = stub
    return client


@pytest.mark.parametrize(
    "factory,descriptor_factory,session_id",
    [
        (MaxStorageClient.create_table_read_session, _make_read_descriptor, "r-1"),
        (MaxStorageClient.create_table_write_session, _make_write_descriptor, "w-1"),
    ],
)
def test_client_restores_session_from_descriptor(
    factory, descriptor_factory, session_id
):
    descriptor = descriptor_factory()
    # NoRpcStub raises on any RPC, so restoration must be purely local
    client = _client_with_stub(NoRpcStub())
    session = factory(client, descriptor=descriptor)
    assert session.id == session_id
    assert session.to_descriptor() == descriptor


@pytest.mark.parametrize("table", ["dt", TableIdentifier("p", "t", "s")])
def test_client_rejects_table_and_descriptor_together(table):
    client = _client_with_stub(NoRpcStub())
    with pytest.raises(ValueError, match="not both"):
        client.create_table_read_session(table, descriptor=_make_read_descriptor())
    with pytest.raises(ValueError, match="not both"):
        client.create_table_write_session(table, descriptor=_make_write_descriptor())


@pytest.mark.parametrize(
    "factory,descriptor_factory",
    [
        (MaxStorageClient.create_table_read_session, _make_read_descriptor),
        (MaxStorageClient.create_table_write_session, _make_write_descriptor),
    ],
)
def test_client_rejects_descriptor_with_session_id(factory, descriptor_factory):
    client = _client_with_stub(NoRpcStub())
    with pytest.raises(ValueError, match="session_id"):
        factory(client, descriptor=descriptor_factory(), session_id="s-1")


def test_client_requires_table_or_descriptor():
    client = _client_with_stub(NoRpcStub())
    with pytest.raises(ValueError, match="table is required"):
        client.create_table_read_session()
    with pytest.raises(ValueError, match="table is required"):
        client.create_table_write_session()
