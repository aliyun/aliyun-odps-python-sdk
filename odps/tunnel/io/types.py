#!/usr/bin/env python
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

from ... import types as odps_types

try:
    import pyarrow as pa
except (AttributeError, ImportError):
    pa = None


if pa is not None:
    TIMESTAMP_STRUCT_TYPE = pa.struct([("sec", pa.int64()), ("nano", pa.int32())])

    _ARROW_TO_ODPS_TYPE = {
        pa.string(): odps_types.string,
        pa.binary(): odps_types.binary,
        pa.int8(): odps_types.tinyint,
        pa.int16(): odps_types.smallint,
        pa.int32(): odps_types.int_,
        pa.int64(): odps_types.bigint,
        pa.bool_(): odps_types.boolean,
        pa.float32(): odps_types.float_,
        pa.float64(): odps_types.double,
        pa.date32(): odps_types.date,
        pa.timestamp("ms"): odps_types.datetime,
        pa.timestamp("ns"): odps_types.timestamp,
        TIMESTAMP_STRUCT_TYPE: odps_types.timestamp,
    }
    _ODPS_TO_ARROW_TYPE = {
        odps_types.string: pa.string(),
        odps_types.binary: pa.binary(),
        odps_types.tinyint: pa.int8(),
        odps_types.smallint: pa.int16(),
        odps_types.int_: pa.int32(),
        odps_types.bigint: pa.int64(),
        odps_types.boolean: pa.bool_(),
        odps_types.float_: pa.float32(),
        odps_types.double: pa.float64(),
        odps_types.date: pa.date32(),
        odps_types.datetime: pa.timestamp("ms"),
        odps_types.timestamp: pa.timestamp("ns"),
        odps_types.timestamp_ntz: pa.timestamp("ns"),
    }
else:
    _ARROW_TO_ODPS_TYPE = {}
    _ODPS_TO_ARROW_TYPE = {}
    TIMESTAMP_STRUCT_TYPE = None

if hasattr(pa, "large_string"):
    _ARROW_TO_ODPS_TYPE[pa.large_string()] = odps_types.string
if hasattr(pa, "large_binary"):
    _ARROW_TO_ODPS_TYPE[pa.large_binary()] = odps_types.binary


def odps_type_to_arrow_type(odps_type):
    if odps_type in _ODPS_TO_ARROW_TYPE:
        col_type = _ODPS_TO_ARROW_TYPE[odps_type]
    else:
        if isinstance(odps_type, odps_types.Array):
            col_type = pa.list_(odps_type_to_arrow_type(odps_type.value_type))
        elif isinstance(odps_type, odps_types.Map):
            col_type = pa.map_(
                odps_type_to_arrow_type(odps_type.key_type),
                odps_type_to_arrow_type(odps_type.value_type),
            )
        elif isinstance(odps_type, odps_types.Decimal):
            precision = odps_type.precision or odps_types.Decimal._default_precision
            scale = odps_type.scale or odps_types.Decimal._default_scale
            if odps_type.precision is None and not hasattr(pa, "decimal256"):
                # need to be less than minimal allowed digits of pa.decimal128
                precision = min(precision, 38)
            decimal_cls = getattr(pa, "decimal256") if precision > 38 else pa.decimal128
            col_type = decimal_cls(precision, scale)
        elif isinstance(odps_type, odps_types.Struct):
            fields = [
                (k, odps_type_to_arrow_type(v))
                for k, v in odps_type.field_types.items()
            ]
            col_type = pa.struct(fields)
        elif isinstance(odps_type, odps_types.IntervalDayTime):
            col_type = pa.struct([("sec", pa.int64()), ("nano", pa.int32())])
        elif isinstance(odps_type, (odps_types.Char, odps_types.Varchar)):
            col_type = pa.string()
        else:
            raise TypeError(f"Unsupported type: {odps_type}")
    return col_type


def odps_schema_to_arrow_schema(odps_schema):
    arrow_schema = []
    for col in odps_schema.simple_columns:
        col_name = col.name
        col_type = odps_type_to_arrow_type(col.type)
        arrow_schema.append(pa.field(col_name, col_type))

    return pa.schema(arrow_schema)


def _is_timestamp_struct_type(arrow_type):
    if not isinstance(arrow_type, pa.StructType):
        return False
    if arrow_type.num_fields != 2:
        return False
    f0, f1 = arrow_type[0], arrow_type[1]
    return (
        f0.name == "sec"
        and f0.type == pa.int64()
        and f1.name == "nano"
        and f1.type == pa.int32()
    )


def arrow_type_to_odps_type(arrow_type):
    arrow_decimal_types = (pa.Decimal128Type,)
    if hasattr(pa, "Decimal256Type"):
        arrow_decimal_types += (pa.Decimal256Type,)

    if arrow_type in _ARROW_TO_ODPS_TYPE:
        col_type = _ARROW_TO_ODPS_TYPE[arrow_type]
    elif _is_timestamp_struct_type(arrow_type):
        col_type = odps_types.timestamp
    else:
        if isinstance(arrow_type, pa.ListType):
            col_type = odps_types.Array(arrow_type_to_odps_type(arrow_type.value_type))
        elif isinstance(arrow_type, pa.MapType):
            col_type = odps_types.Map(
                arrow_type_to_odps_type(arrow_type.key_type),
                arrow_type_to_odps_type(arrow_type.item_type),
            )
        elif isinstance(arrow_type, arrow_decimal_types):
            precision = arrow_type.precision or odps_types.Decimal._default_precision
            scale = arrow_type.scale or odps_types.Decimal._default_scale
            col_type = odps_types.Decimal(precision, scale)
        elif isinstance(arrow_type, pa.StructType):
            fields = [
                (arrow_type[idx].name, arrow_type_to_odps_type(arrow_type[idx].type))
                for idx in range(arrow_type.num_fields)
            ]
            col_type = odps_types.Struct(fields)
        else:
            raise TypeError(f"Unsupported type: {arrow_type}")
    return col_type


def arrow_schema_to_odps_schema(arrow_schema):
    odps_cols = []
    for col_name, pa_type in zip(arrow_schema.names, arrow_schema.types):
        col_type = arrow_type_to_odps_type(pa_type)
        odps_cols.append(odps_types.Column(col_name, col_type))

    return odps_types.OdpsSchema(odps_cols)
