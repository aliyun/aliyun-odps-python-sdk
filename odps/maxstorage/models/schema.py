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

"""Schema classes for :mod:`odps.maxstorage`.

The maxstorage schemas reuse :class:`odps.types.OdpsSchema` and PyODPS data
types rather than defining standalone ``ColumnTypeInfo``.  ``Column`` subclasses
:class:`odps.types.Column` to add ``column_id`` (server-assigned, needed by blob
operations).  ``ReadSchema`` and ``WriteSchema`` have **different JSON
conventions** (see DESIGN.md §6.1) and must not share a parser.
"""

from ...types import Array, Blob
from ...types import Column as _BaseColumn
from ...types import Map, OdpsSchema, Struct
from ..errors import StorageClientError
from .types import (
    _get_bool,
    _get_long,
    _get_str,
    parse_read_schema_type,
    parse_write_schema_type,
)


class Column(_BaseColumn):
    """maxstorage column — subclasses :class:`odps.types.Column` and adds
    ``column_id`` (server-assigned identifier needed by blob operations).

    Because it inherits name/type/comment/nullable, it is transparent to all
    OdpsSchema facilities (casing, ``__getitem__``, ``__contains__``,
    ``get_column``, ``get_type``).
    """

    def __init__(
        self,
        name=None,
        typo=None,
        comment=None,
        nullable=True,
        column_id=None,
        label=None,
        **kw
    ):
        super().__init__(
            name=name, typo=typo, comment=comment, label=None, nullable=nullable, **kw
        )
        self.column_id = column_id
        self.label = label

    @property
    def is_distribution_key(self):
        return getattr(self, "_is_distribution_key", False)

    @is_distribution_key.setter
    def is_distribution_key(self, value):
        self._is_distribution_key = value


class StorageSchema(OdpsSchema):
    """Base schema for ReadSchema/WriteSchema.

    Subclasses :class:`odps.types.OdpsSchema` to inherit casing,
    ``__getitem__``, ``__contains__``, ``get_column``, ``get_type``,
    ``from_lists``, and the ``columns``/``simple_columns``/``partitions``
    properties.  Adds ``system_columns``.
    """

    def __init__(self, columns=None, partitions=None, system_columns=None):
        super().__init__(columns=columns, partitions=partitions)
        self._system_columns = system_columns or []

    @property
    def system_columns(self):
        return self._system_columns

    @property
    def blob_columns(self):
        """A StorageSchema view of BLOB-type columns."""
        cols = [c for c in self._columns if isinstance(c.type, Blob)]
        return StorageSchema(columns=cols)


class ReadSchema(StorageSchema):
    """Deserialized from a read-session response.

    JSON is **flat PascalCase**: ``[{Name, Type (string), Comment, Nullable,
    ColumnId}, ...]``.
    """

    @classmethod
    def from_dict(cls, raw):
        """Parse a raw ``TableSchema``/``DataSchema`` dict (ReadSchema format)."""
        if not raw:
            return cls()
        data_cols = _parse_column_list(raw.get("DataColumns"))
        part_cols = _parse_column_list(raw.get("PartitionColumns"))
        sys_cols = _parse_column_list(raw.get("SystemColumns"))
        return cls(columns=data_cols, partitions=part_cols, system_columns=sys_cols)


class WriteSchema(StorageSchema):
    """Deserialized from a write-session response.

    JSON is **nested camelCase**: ``[{comment, label, columnType: {MemberName,
    ColumnId, Nullable, Type (int), SubTypes, ...}}, ...]``.

    ``find_all_blob_column_ids()`` walks the column tree and returns
    ``{dot_path: column_id}`` for every BLOB column (top-level and nested).
    """

    def __init__(
        self, columns=None, partitions=None, system_columns=None, nested_column_ids=None
    ):
        super().__init__(
            columns=columns, partitions=partitions, system_columns=system_columns
        )
        self._nested_column_ids = nested_column_ids or {}

    @classmethod
    def from_dict(cls, raw):
        """Parse a raw ``TableSchema`` dict (WriteSchema format)."""
        if not raw:
            return cls()
        nested_column_ids = {}
        data_cols = _parse_write_column_list(raw.get("DataColumns"), nested_column_ids)
        sys_cols = _parse_write_column_list(raw.get("SystemColumns"), nested_column_ids)
        return cls(
            columns=data_cols,
            system_columns=sys_cols,
            nested_column_ids=nested_column_ids,
        )

    @property
    def nested_column_ids(self):
        return self._nested_column_ids

    def get_nested_column_id(self, path):
        return self._nested_column_ids.get(path)

    def find_all_blob_column_ids(self):
        """``{dot_path: column_id}`` for every BLOB column (top-level + nested)."""
        blob_ids = {}
        for col in self._columns:
            _walk_for_blobs(
                col.type, col.name, col.column_id, blob_ids, self._nested_column_ids
            )
        for col in self._system_columns:
            _walk_for_blobs(
                col.type, col.name, col.column_id, blob_ids, self._nested_column_ids
            )
        return blob_ids

    def resolve_blob_column_name(self, column_name=None):
        """Resolve *column_name* to a ``(name, column_id)`` BLOB-column pair.

        When *column_name* is ``None`` and the schema has exactly one
        **top-level pure-BLOB** column (not nested inside
        ARRAY/STRUCT/MAP, and no other BLOB column of any kind), that
        column is selected automatically.  Any nested BLOB column, or a
        second top-level BLOB column, disqualifies auto-selection so the
        caller must name the column explicitly.  An explicit
        *column_name* is validated against the known BLOB columns and
        returned unchanged.
        """
        blob_ids = self.find_all_blob_column_ids()
        if column_name is None:
            pure = list(self.blob_columns.columns)
            nested = self.has_nested_blob_columns()
            if len(pure) != 1 or nested:
                raise StorageClientError(
                    "Cannot auto-select a BLOB column: found "
                    f"{len(pure)} top-level BLOB column(s)"
                    + (" plus nested BLOB column(s)" if nested else "")
                    + ". Specify column_name explicitly. "
                    f"Available: {', '.join(sorted(blob_ids)) or '(none)'}"
                )
            column_name = pure[0].name
        if column_name not in blob_ids:
            raise StorageClientError(
                f"Unknown BLOB column name {column_name!r}; available: "
                f"{', '.join(sorted(blob_ids)) or '(none)'}"
            )
        return column_name, blob_ids[column_name]

    def has_nested_blob_columns(self):
        """True if any column contains a BLOB nested inside ARRAY/STRUCT/MAP."""
        for col in list(self._columns) + list(self._system_columns):
            if _has_nested_blob(col.type):
                return True
        return False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_column_list(raw):
    """Parse a ReadSchema column list (flat PascalCase, string types)."""
    if not raw:
        return []
    if isinstance(raw, dict):
        return []
    columns = []
    for col_json in raw:
        name = col_json.get("Name")
        type_str = col_json.get("Type")
        comment = col_json.get("Comment")
        nullable = col_json.get("Nullable", True)
        column_id = col_json.get("ColumnId")
        col = Column(
            name=name,
            typo=parse_read_schema_type(type_str),
            comment=comment,
            nullable=bool(nullable),
            column_id=column_id,
        )
        columns.append(col)
    return columns


def _parse_write_column_list(raw, nested_column_ids):
    """Parse a WriteSchema column list (nested camelCase, int type codes)."""
    if not raw:
        return []
    if isinstance(raw, dict):
        return []
    columns = []
    for col_json in raw:
        type_info = col_json.get("columnType")
        if type_info is None:
            continue
        name = _get_str(type_info, "MemberName", "")
        column_id = _get_long(type_info, "ColumnId", -1)
        nullable = _get_bool(type_info, "Nullable", True)
        comment = _get_str(col_json, "comment", "")
        label = _get_str(col_json, "label", "")

        data_type = parse_write_schema_type(type_info, name, nested_column_ids)

        col = Column(
            name=name,
            typo=data_type,
            comment=comment,
            nullable=nullable,
            column_id=column_id if column_id != -1 else None,
            label=label,
        )
        # Distribution key flag — used by TableArrowBlobUploadWriter for PK detection
        if _get_bool(type_info, "IsDistributionKey", False):
            col.is_distribution_key = True
        columns.append(col)
    return columns


def _walk_for_blobs(data_type, path, column_id, blob_ids, nested_column_ids):
    """Walk a type tree, recording BLOB column IDs at every level."""
    if isinstance(data_type, Blob):
        if column_id is not None:
            blob_ids[path] = column_id
        return
    if isinstance(data_type, Array):
        elem_path = f"{path}.element"
        elem_id = nested_column_ids.get(elem_path)
        _walk_for_blobs(
            data_type.value_type, elem_path, elem_id, blob_ids, nested_column_ids
        )
    elif isinstance(data_type, Map):
        val_path = f"{path}.value"
        val_id = nested_column_ids.get(val_path)
        _walk_for_blobs(
            data_type.value_type, val_path, val_id, blob_ids, nested_column_ids
        )
    elif isinstance(data_type, Struct):
        for fname, ftype in zip(data_type.field_names, data_type.field_types):
            child_path = f"{path}.{fname}"
            child_id = nested_column_ids.get(child_path)
            _walk_for_blobs(ftype, child_path, child_id, blob_ids, nested_column_ids)


def _has_nested_blob(data_type):
    """True if a BLOB column appears nested inside ARRAY/STRUCT/MAP."""
    if isinstance(data_type, Array):
        return _contains_blob(data_type.value_type)
    if isinstance(data_type, Map):
        return _contains_blob(data_type.value_type)
    if isinstance(data_type, Struct):
        return any(_contains_blob(ft) for ft in data_type.field_types)
    return False


def _contains_blob(data_type):
    """True if data_type is BLOB or contains a nested BLOB."""
    if isinstance(data_type, Blob):
        return True
    if isinstance(data_type, (Array, Map, Struct)):
        return _has_nested_blob(data_type)
    return False
