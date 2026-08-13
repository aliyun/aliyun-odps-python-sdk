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

"""Storage-API type parsing.

Two distinct parsers live here:

* ``parse_read_schema_type`` -- ReadSchema path: type is a *string*
  (``"bigint"``, ``"array<int>"``) parsed by ``odps.types.validate_data_type``.
* ``parse_write_schema_type`` -- WriteSchema path: type is a nested JSON object
  with an *int* code (wire codes that do **not** match
  PyODPS ``DataType._type_id``).  This module owns the int-code map and the
  recursive ``columnType`` walker that also accumulates dot-path column IDs.
"""

from ...types import (
    Array,
    Bigint,
    Binary,
    Blob,
    Boolean,
    Char,
    Date,
    Datetime,
    Decimal,
    Double,
    Float,
    IntervalDayTime,
    IntervalYearMonth,
    Json,
    Map,
    Smallint,
    String,
    Struct,
    Timestamp,
    TimestampNTZ,
    Tinyint,
    Varchar,
    Variant,
    validate_data_type,
)

# Storage-API wire int-code -> DataType singleton or factory.
# These are wire int-code values, NOT PyODPS _type_id values.
_TYPE_CODE_MAP = {
    0: Bigint(),
    1: Double(),
    2: Boolean(),
    3: Datetime(),
    4: String(),  # deprecated alias
    5: "decimal",  # parameterized — recursive builder
    6: Tinyint(),
    7: Smallint(),
    8: "int",  # PyODPS has no standalone Int; use int32-equivalent
    9: "char",  # parameterized
    10: "varchar",  # parameterized
    11: Binary(),
    12: Date(),
    13: Timestamp(),
    14: Float(),
    15: IntervalYearMonth(),
    16: IntervalDayTime(),
    17: "array",  # recursive builder
    18: "map",  # recursive builder
    19: "struct",  # recursive builder
    20: Json(),
    21: TimestampNTZ(),  # alt code
    22: Blob(),
    23: Variant(),
}


def parse_read_schema_type(type_string):
    """Parse a ReadSchema ``Type`` string (e.g. ``"bigint"``, ``"array<int>"``)."""
    return validate_data_type(type_string)


def parse_write_schema_type(type_info_json, path, nested_column_ids):
    """Recursively parse a WriteSchema ``columnType`` JSON object.

    1. Record ``ColumnId`` (if present and != -1) into ``nested_column_ids[path]``.
    2. Look up ``Type`` (int code) in ``_TYPE_CODE_MAP``.
    3. For parameterized types (Decimal/Char/Varchar): pass Precision/Scale/
       SpecifiedLength to the constructor.
    4. For complex types (Array/Map/Struct): recurse into SubTypes, building
       dot-paths with MemberName, then construct the PyODPS type.
    5. For primitives: return the singleton from the map.

    Returns the :class:`odps.types.DataType` instance.  ColumnId/MemberName
    are consumed for ``nested_column_ids`` but not stored on the DataType.
    """
    column_id = _get_long(type_info_json, "ColumnId", -1)
    if column_id != -1:
        nested_column_ids[path] = column_id

    type_code = _get_int(type_info_json, "Type", -1)
    entry = _TYPE_CODE_MAP.get(type_code)

    if entry is None:
        # Unknown code — treat as String to avoid hard failure.
        return String()

    if entry == "decimal":
        # Default to None when the server omits Precision/Scale so we build
        # an unlimited-precision Decimal() rather than Decimal(0, 0), which
        # raises ValueError("Decimal precision < 1").
        precision = _get_int(type_info_json, "Precision", None)
        scale = _get_int(type_info_json, "Scale", None)
        return Decimal(precision, scale)

    if entry == "char":
        length = _get_int(type_info_json, "SpecifiedLength", 0)
        return Char(length)

    if entry == "varchar":
        length = _get_int(type_info_json, "SpecifiedLength", 0)
        return Varchar(length)

    if entry == "array":
        sub_types = type_info_json.get("SubTypes") or []
        if len(sub_types) != 1:
            raise ValueError("ARRAY type must have exactly one sub-type.")
        elem = sub_types[0]
        elem_name = _get_str(elem, "MemberName", "element") or "element"
        elem_type = parse_write_schema_type(
            elem, f"{path}.{elem_name}", nested_column_ids
        )
        return Array(elem_type)

    if entry == "map":
        sub_types = type_info_json.get("SubTypes") or []
        if len(sub_types) != 2:
            raise ValueError("MAP type must have exactly two sub-types.")
        key_sub, val_sub = sub_types[0], sub_types[1]
        key_name = _get_str(key_sub, "MemberName", "key") or "key"
        val_name = _get_str(val_sub, "MemberName", "value") or "value"
        key_type = parse_write_schema_type(
            key_sub, f"{path}.{key_name}", nested_column_ids
        )
        val_type = parse_write_schema_type(
            val_sub, f"{path}.{val_name}", nested_column_ids
        )
        return Map(key_type, val_type)

    if entry == "struct":
        sub_types = type_info_json.get("SubTypes") or []
        field_names = []
        field_types = []
        for sub in sub_types:
            fname = _get_str(sub, "MemberName", "")
            if not fname:
                raise ValueError("Struct member must have a 'MemberName'.")
            ftype = parse_write_schema_type(sub, f"{path}.{fname}", nested_column_ids)
            field_names.append(fname)
            field_types.append(ftype)
        return Struct(list(zip(field_names, field_types)))

    if entry == "int":
        # Wire code 8 is INT (int32). PyODPS exposes this via validate_data_type.
        return validate_data_type("int")

    # Primitive singleton from the map.
    return entry


def _get_str(obj, key, default=""):
    val = obj.get(key)
    return default if val is None else val


def _get_int(obj, key, default=0):
    val = obj.get(key)
    if val is None:
        return default
    return int(val)


def _get_long(obj, key, default=-1):
    val = obj.get(key)
    if val is None:
        return default
    return int(val)


def _get_bool(obj, key, default=True):
    val = obj.get(key)
    if val is None:
        return default
    return bool(val)
