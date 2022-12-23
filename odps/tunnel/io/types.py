#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2022 Alibaba Group Holding Ltd.
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
    _ODPS_ARROW_TYPE_MAPPING = {
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
        odps_types.datetime: pa.timestamp('ms'),
        odps_types.timestamp: pa.timestamp('ns')
    }
else:
    _ODPS_ARROW_TYPE_MAPPING = {}


def odps_type_to_arrow_type(odps_type):
    from ... import types

    if odps_type in _ODPS_ARROW_TYPE_MAPPING:
        col_type = _ODPS_ARROW_TYPE_MAPPING[odps_type]
    else:
        if isinstance(odps_type, types.Array):
            col_type = pa.list_(odps_type_to_arrow_type(odps_type.value_type))
        elif isinstance(odps_type, types.Map):
            col_type = pa.map_(
                odps_type_to_arrow_type(odps_type.key_type),
                odps_type_to_arrow_type(odps_type.value_type),
            )
        elif isinstance(odps_type, types.Decimal):
            col_type = pa.decimal128(odps_type.precision, odps_type.scale)
        else:
            raise TypeError('Unsupported type: {}'.format(odps_type))
    return col_type


def odps_schema_to_arrow_schema(odps_schema):

    arrow_schema = []
    for schema in odps_schema.simple_columns:
        col_name = schema.name
        col_type = odps_type_to_arrow_type(schema.type)
        arrow_schema.append(pa.field(col_name, col_type))

    return pa.schema(arrow_schema)
