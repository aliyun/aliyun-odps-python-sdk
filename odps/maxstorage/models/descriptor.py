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

"""Versioned, credential-free session descriptors.

A :class:`SessionDescriptor` is an immutable, versioned description of a
storage session.  A driver exports it from a live session, transfers the
JSON through any application channel, and a worker restores a local session
binding from it — without creating or reloading a remote session.  The
restoring process supplies its own client and credentials; descriptors never
carry clients, credentials, or runtime resources.

Write descriptors carry explicit Arrow encoding units.  Read descriptors
additionally preserve the row-offset chunk size when the session was planned
with ``ROW_OFFSET`` splits.  Treat descriptors as internal task data: they
contain session identifiers and opaque routing data, so do not log or
publicly share them.
"""

import json
from typing import Any, Dict, Optional

from .enums import SessionKind
from .identifier import TableIdentifier
from .requests import _DEFAULT_ARROW_OPTIONS

__all__ = [
    "SessionDescriptor",
    "DESCRIPTOR_VERSION",
    "DESCRIPTOR_PROVIDER",
]

DESCRIPTOR_VERSION = 1
DESCRIPTOR_PROVIDER = "v2"

#: Canonical Arrow unit names carried by write descriptors.
ARROW_UNITS = ("SECOND", "MILLI", "MICRO", "NANO")

_MAX_SPLIT_COUNT = 2**31 - 1

# These names define the descriptor schema, independently of response
# evolution.  Wire responses use PascalCase; the canonical descriptor form
# stores the same fields in camelCase.
_ROOT_WHITELIST = (
    "SessionId",
    "SessionStatus",
    "DataSchema",
    "SupportedDataFormat",
    "SplitMode",
    "SplitsCount",
    "RecordCount",
    "SplitBucketId",
    "SessionStats",
    "ExtendedArrowIPCEnabled",
    "EnableLargeString",
    "LatestVersion",
    "LatestTxnId",
    "ExpirationTime",
    "IncrementalReadOptions",
    "MaxBlockNumber",
    "EnhanceWriteCheck",
    "RequiredOrdering",
    "RequiredDistribution",
    "CommitStarted",
)

_CHILD_WHITELISTS = {
    "DataSchema": ("DataColumns", "PartitionColumns", "SystemColumns", "BlobColumns"),
    "SupportedDataFormat": ("Type", "Version"),
    "SessionStats": ("EstimatedSize", "EstimatedRowCount"),
    "IncrementalReadOptions": (
        "StartTimeStamp",
        "EndTimeStamp",
        "StartVersion",
        "EndVersion",
        "Mode",
        "StartTxnId",
    ),
    "RequiredOrdering": ("Name", "SortDirection", "NullOrdering"),
    "RequiredDistribution": ("Type", "ClusterKeys", "BucketsNumber"),
}
_COLUMN_WHITELIST = ("Name", "Type", "Comment", "Nullable", "ColumnId")
for _list_name in ("DataColumns", "PartitionColumns", "SystemColumns", "BlobColumns"):
    _CHILD_WHITELISTS[_list_name] = _COLUMN_WHITELIST

# The descriptor envelope, in canonical (camelCase) serialization order.
_DESCRIPTOR_KEYS = (
    "version",
    "provider",
    "kind",
    "table",
    "metadata",
    "routeToken",
    "arrow",
    "readOptions",
)
_TABLE_KEYS = ("project", "table", "schema")
_ARROW_KEYS = ("timestampUnit", "datetimeUnit")


def _to_camel(name):
    return name[:1].lower() + name[1:]


def _to_pascal(name):
    return name[:1].upper() + name[1:]


def _filter(source, names, key_in, key_out):
    """Whitelist-filter ``source``, keyed by ``key_in`` / ``key_out``.

    Each whitelisted wire name maps to ``key_in(name)`` in ``source`` and to
    ``key_out(name)`` in the result — :func:`_to_pascal` reads or writes wire
    (PascalCase) names, :func:`_to_camel` the canonical (camelCase) ones.
    ``None`` values are dropped, and nested objects and arrays are filtered
    recursively through the child whitelists, so unknown server fields can
    never leak into (or out of) a descriptor.
    """
    result = {}
    for wire in names:
        element = source.get(key_in(wire))
        if element is None:
            continue
        children = _CHILD_WHITELISTS.get(wire)
        if children is not None:
            if isinstance(element, list):
                element = [
                    _filter(item, children, key_in, key_out)
                    if isinstance(item, dict)
                    else item
                    for item in element
                ]
            elif isinstance(element, dict):
                element = _filter(element, children, key_in, key_out)
        result[key_out(wire)] = element
    return result


def _normalize(root):
    """Project a parsed descriptor onto the canonical field set.

    Mirrors typed deserialization: unknown keys in the incoming JSON are
    dropped instead of being carried through and re-exported.  Run after
    ``_validate`` so the field types are known to be well-formed.
    """
    filtered = {k: root[k] for k in _DESCRIPTOR_KEYS if root.get(k) is not None}
    if "table" in filtered:
        table = filtered["table"]
        filtered["table"] = {k: table[k] for k in _TABLE_KEYS if k in table}
    if "metadata" in filtered:
        filtered["metadata"] = _filter(
            filtered["metadata"], _ROOT_WHITELIST, _to_camel, _to_camel
        )
    if "arrow" in filtered:
        arrow = filtered["arrow"]
        filtered["arrow"] = {k: arrow[k] for k in _ARROW_KEYS if k in arrow}
    if "readOptions" in filtered:
        options = filtered["readOptions"]
        if "rowsPerSplit" in options:
            filtered["readOptions"] = {"rowsPerSplit": options["rowsPerSplit"]}
        else:
            del filtered["readOptions"]
    return filtered


def _required(obj, key):
    if obj is None:
        raise ValueError("Missing session descriptor field: %s" % key)
    value = obj.get(key)
    if value is None or (isinstance(value, str) and not value):
        raise ValueError("Missing session descriptor field: %s" % key)
    return value


def _non_negative_int(obj, key):
    value = _required(obj, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Invalid non-negative integer: %s" % key)
    return value


def _validate_arrow(arrow):
    if not isinstance(arrow, dict):
        raise ValueError("Missing session descriptor field: arrow")
    for key in ("timestampUnit", "datetimeUnit"):
        if _required(arrow, key) not in ARROW_UNITS:
            raise ValueError("Invalid session descriptor arrow unit: %s" % key)


def _validate(root):
    """Validate a canonical descriptor dict; raises ``ValueError`` on flaws."""
    if not isinstance(root, dict):
        raise ValueError("Session descriptor must be a JSON object")

    version = root.get("version")
    if isinstance(version, bool) or version != DESCRIPTOR_VERSION:
        raise ValueError("Unsupported session descriptor version")

    if _required(root, "provider") != DESCRIPTOR_PROVIDER:
        raise ValueError("Expected v2 session descriptor")

    kind = SessionKind.from_string(_required(root, "kind"))

    table = root.get("table")
    if not isinstance(table, dict):
        raise ValueError("Missing session descriptor field: table")
    _required(table, "project")
    _required(table, "table")

    metadata = root.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Missing session descriptor field: metadata")
    _required(metadata, "sessionId")
    _required(metadata, "sessionStatus")
    if not isinstance(metadata.get("dataSchema"), dict):
        raise ValueError("Missing session descriptor dataSchema")

    if kind == SessionKind.BATCH_WRITE:
        _validate_arrow(root.get("arrow"))
    else:
        mode = _required(metadata, "splitMode")
        if mode == "Bucket":
            count = _non_negative_int(metadata, "splitsCount")
            buckets = metadata.get("splitBucketId")
            if not isinstance(buckets, list) or len(buckets) != count:
                raise ValueError("Bucket count must match splitsCount")
            for bucket in buckets:
                if (
                    isinstance(bucket, bool)
                    or not isinstance(bucket, int)
                    or bucket < 0
                ):
                    raise ValueError("Invalid bucket id")
            if count > _MAX_SPLIT_COUNT:
                raise ValueError("splitsCount exceeds supported range")
        elif mode == "Size":
            if _non_negative_int(metadata, "splitsCount") > _MAX_SPLIT_COUNT:
                raise ValueError("splitsCount exceeds supported range")
        elif mode == "RowOffset":
            _non_negative_int(metadata, "recordCount")
        else:
            raise ValueError("Unsupported read split mode")

    read_options = root.get("readOptions")
    if read_options is not None:
        if not isinstance(read_options, dict):
            raise ValueError("Invalid session descriptor readOptions")
        rows = read_options.get("rowsPerSplit")
        if rows is not None and (
            isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
        ):
            raise ValueError("rowsPerSplit must be positive")


class SessionDescriptor:
    """Immutable, versioned description of a storage session.

    Export with a session's ``to_descriptor()``, serialize with
    :meth:`to_json`, and parse on the worker with :meth:`from_json`.  Bind to
    a worker session through ``TableReadSession.from_descriptor`` /
    ``TableWriteSession.from_descriptor`` or the matching
    :class:`~odps.maxstorage.MaxStorageClient` factory methods.  Restoration
    never creates or reloads a remote session; a restored write session has
    only block-writing authority.
    """

    __slots__ = ("_root",)

    def __init__(self, root):
        self._root = root

    @classmethod
    def create(
        cls,
        kind,
        table,
        metadata,
        route_token=None,
        arrow=None,
        rows_per_split=None,
    ):
        """Build a descriptor from wire-form (PascalCase) session metadata.

        Parameters
        ----------
        kind : SessionKind or str
            The session purpose; a string value must equal the enum value
            (``"batch-read"`` / ``"incremental-read"`` / ``"batch-write"``).
        table : TableIdentifier or dict
            Table identity (``project`` / ``table`` / optional ``schema``).
        metadata : dict
            Wire-form session metadata; unknown fields are dropped.
        route_token : str, optional
            Opaque routing token from the session response.
        arrow : dict, optional
            Write encoding units, e.g.
            ``{"timestampUnit": "NANO", "datetimeUnit": "MILLI"}``.
        rows_per_split : int, optional
            Row-offset chunk size for ``ROW_OFFSET`` read sessions.
        """
        root = {
            "version": DESCRIPTOR_VERSION,
            "provider": DESCRIPTOR_PROVIDER,
            "kind": SessionKind.from_string(kind).value,
            "table": _table_to_canonical(table),
            "metadata": _filter(metadata, _ROOT_WHITELIST, _to_pascal, _to_camel),
        }
        if route_token is not None:
            root["routeToken"] = route_token
        if arrow is not None:
            root["arrow"] = dict(arrow)
        if rows_per_split:
            root["readOptions"] = {"rowsPerSplit": rows_per_split}
        _validate(root)
        return cls(_normalize(root))

    @classmethod
    def from_json(cls, text):
        """Parse a descriptor from JSON without network or config access."""
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")
        try:
            root = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid session descriptor JSON") from exc
        _validate(root)
        return cls(_normalize(root))

    def to_json(self):
        """Serialize to the canonical JSON form."""
        return json.dumps(self._root)

    # -- accessors ---------------------------------------------------------

    @property
    def kind(self) -> SessionKind:
        """The session purpose, as a :class:`~odps.maxstorage.SessionKind`."""
        return SessionKind(self._root["kind"])

    @property
    def table(self) -> Dict[str, Any]:
        """Table identity: ``project`` / ``table`` / optional ``schema``."""
        return dict(self._root["table"])

    @property
    def table_identifier(self):
        """The table identity as a :class:`TableIdentifier`."""
        table = self._root["table"]
        return TableIdentifier(
            table.get("project"), table.get("table"), table.get("schema")
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Session metadata in wire (PascalCase) form."""
        return _filter(self._root["metadata"], _ROOT_WHITELIST, _to_camel, _to_pascal)

    @property
    def route_token(self) -> Optional[str]:
        """Opaque routing token, or ``None``."""
        return self._root.get("routeToken")

    @property
    def arrow(self) -> Optional[Dict[str, str]]:
        """Write Arrow encoding units, or ``None`` for read descriptors."""
        arrow = self._root.get("arrow")
        return dict(arrow) if arrow is not None else None

    @property
    def rows_per_split(self) -> int:
        """Row-offset chunk size (``0`` when the descriptor carries none)."""
        read_options = self._root.get("readOptions")
        if not read_options:
            return 0
        return read_options.get("rowsPerSplit") or 0

    @property
    def session_id(self) -> str:
        """The session identifier from the metadata."""
        return self._root["metadata"]["sessionId"]

    @property
    def session_status(self) -> str:
        """The last observed session status from the metadata."""
        return self._root["metadata"]["sessionStatus"]

    def __eq__(self, other):
        if not isinstance(other, SessionDescriptor):
            return NotImplemented
        return self._root == other._root

    def __repr__(self):
        return "SessionDescriptor(kind=%r, table=%r, session_id=%r)" % (
            self.kind,
            self._root.get("table"),
            self._root.get("metadata", {}).get("sessionId"),
        )


def _table_to_canonical(table):
    if isinstance(table, dict):
        project = table.get("project")
        name = table.get("table")
        schema = table.get("schema")
    else:
        project = table.project
        name = table.table
        schema = table.schema
    result = {"project": project, "table": name}
    if schema is not None:
        result["schema"] = schema
    return result


# ---------------------------------------------------------------------------
# wire serialization helpers
#
# Sessions export their state as wire-form (PascalCase) metadata; keeping the
# serializers next to the whitelists that filter them keeps the two in sync.
# ---------------------------------------------------------------------------


def column_to_wire(name, type_, comment=None, nullable=True, column_id=None):
    """Serialize one column to its wire (PascalCase) form."""
    wire = {"Name": name, "Type": type_}
    if comment is not None:
        wire["Comment"] = comment
    wire["Nullable"] = bool(nullable)
    if column_id is not None:
        wire["ColumnId"] = column_id
    return wire


def _read_column_to_wire(column):
    return column_to_wire(
        column.name,
        # Prefer the exact wire string captured at parse time; fall back to
        # the type's display form for programmatically-built columns.
        getattr(column, "wire_type", None) or str(column.type),
        column.comment,
        column.nullable,
        column.column_id,
    )


def read_schema_to_wire(schema):
    """Serialize a read-session :class:`ReadSchema` to its wire form."""
    if schema is None:
        return None
    return {
        # ``OdpsSchema.columns`` is data + partition columns; only the data
        # columns belong under DataColumns, or the partitions would be
        # duplicated into the restored schema.
        "DataColumns": [_read_column_to_wire(col) for col in schema.simple_columns],
        "PartitionColumns": [_read_column_to_wire(col) for col in schema.partitions],
        "SystemColumns": [_read_column_to_wire(col) for col in schema.system_columns],
    }


def write_schema_to_wire(schema):
    """Serialize a batch-compatible data schema to its wire form."""
    if schema is None:
        return None
    return {
        "DataColumns": [
            column_to_wire(col.name, col.type, col.comment, col.nullable)
            for col in schema.data_columns
        ],
        "PartitionColumns": [
            column_to_wire(col.name, col.type, col.comment, col.nullable)
            for col in schema.partition_columns
        ],
    }


def incremental_options_to_wire(options):
    """Serialize incremental-read options to the descriptor wire form.

    Accepts either a raw response dict (already wire-form) or an
    ``IncrementalReadOptions`` request object whose version boundaries may
    be versions (int) or timestamps (str).
    """
    if options is None:
        return None
    if isinstance(options, dict):
        return dict(options)
    wire = {}
    if getattr(options, "version", None) is not None:
        wire["Mode"] = options.version
    if options.from_ is not None:
        key = "StartVersion" if isinstance(options.from_, int) else "StartTimeStamp"
        wire[key] = options.from_
    if options.to is not None:
        key = "EndVersion" if isinstance(options.to, int) else "EndTimeStamp"
        wire[key] = options.to
    return wire


def write_arrow_units():
    """The Arrow encoding units negotiated at session creation.

    Batch-compatible sessions are always created with the default units,
    which mirror the request defaults (``nano`` timestamps / ``milli``
    datetimes); descriptors record them in canonical unit-name form.
    """
    return {
        "timestampUnit": str(_DEFAULT_ARROW_OPTIONS["TimestampUnit"]).upper(),
        "datetimeUnit": str(_DEFAULT_ARROW_OPTIONS["DatetimeUnit"]).upper(),
    }
