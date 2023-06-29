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

import threading
from collections import OrderedDict

from ....models import TableSchema
from .... import types as odps_types
from ... import types as df_types
from ....compat import six
from ....config import options


_odps_to_df_types = {
    odps_types.tinyint: df_types.int8,
    odps_types.smallint: df_types.int16,
    odps_types.int_: df_types.int32,
    odps_types.bigint: df_types.int64,
    odps_types.float_: df_types.float32,
    odps_types.double: df_types.float64,
    odps_types.string: df_types.string,
    odps_types.datetime: df_types.datetime,
    odps_types.boolean: df_types.boolean,
    odps_types.binary: df_types.binary,
    odps_types.timestamp: df_types.timestamp,
    odps_types.timestamp_ntz: df_types.timestamp,
    odps_types.date: df_types.date,
    odps_types.json: df_types.json,
}

_df_to_odps_types = {
    df_types.int8: odps_types.bigint,
    df_types.int16: odps_types.bigint,
    df_types.int32: odps_types.bigint,
    df_types.int64: odps_types.bigint,
    df_types.float32: odps_types.double,
    df_types.float64: odps_types.double,
    df_types.boolean: odps_types.boolean,
    df_types.string: odps_types.string,
    df_types.datetime: odps_types.datetime,
    df_types.binary: odps_types.string,
    df_types.timestamp: odps_types.timestamp,
    df_types.date: odps_types.date,
    df_types.json: odps_types.json,
}

_df_to_odps_types2 = {
    df_types.int8: odps_types.tinyint,
    df_types.int16: odps_types.smallint,
    df_types.int32: odps_types.int_,
    df_types.int64: odps_types.bigint,
    df_types.float32: odps_types.float_,
    df_types.float64: odps_types.double,
    df_types.boolean: odps_types.boolean,
    df_types.string: odps_types.string,
    df_types.datetime: odps_types.datetime,
    df_types.binary: odps_types.binary,
    df_types.timestamp: odps_types.timestamp,
    df_types.date: odps_types.date,
    df_types.json: odps_types.json,
}


def is_odps_type(typ):
    return typ in _odps_to_df_types


def odps_type_to_df_type(odps_type):
    if isinstance(odps_type, six.string_types):
        odps_type = odps_types.validate_data_type(odps_type)

    if odps_type in _odps_to_df_types:
        return _odps_to_df_types[odps_type]
    elif isinstance(odps_type, odps_types.Decimal):
        return df_types.decimal
    elif isinstance(odps_type, (odps_types.Varchar, odps_types.Char)):
        return df_types.string
    elif isinstance(odps_type, odps_types.Array):
        return df_types.List(odps_type_to_df_type(odps_type.value_type))
    elif isinstance(odps_type, odps_types.Map):
        return df_types.Dict(odps_type_to_df_type(odps_type.key_type),
                             odps_type_to_df_type(odps_type.value_type))
    elif isinstance(odps_type, odps_types.Struct):
        type_dict = OrderedDict(
            (k, odps_type_to_df_type(v)) for k, v in odps_type.field_types.items()
        )
        return df_types.Struct(type_dict)
    else:
        raise KeyError(repr(odps_type))


def odps_schema_to_df_schema(odps_schema):
    names = [col.name for col in odps_schema.columns]
    types = [odps_type_to_df_type(col.type) for col in odps_schema.columns]

    return TableSchema.from_lists(names, types)


def df_type_to_odps_type(df_type, use_odps2_types=None, project=None):
    if use_odps2_types is None:
        use_odps2_types = get_local_use_odps2_types(project)

    if use_odps2_types:
        df_to_odps_types = _df_to_odps_types2
    else:
        df_to_odps_types = _df_to_odps_types
    if isinstance(df_type, six.string_types):
        df_type = df_types.validate_data_type(df_type)

    if df_type in df_to_odps_types:
        return df_to_odps_types[df_type]
    elif df_type == df_types.decimal:
        return odps_types.Decimal()
    elif isinstance(df_type, df_types.List):
        return odps_types.Array(
            df_type_to_odps_type(df_type.value_type, use_odps2_types=use_odps2_types)
        )
    elif isinstance(df_type, df_types.Dict):
        return odps_types.Map(
            df_type_to_odps_type(df_type.key_type, use_odps2_types=use_odps2_types),
            df_type_to_odps_type(df_type.value_type, use_odps2_types=use_odps2_types),
        )
    elif isinstance(df_type, df_types.Struct):
        type_dict = OrderedDict(
            (k, df_type_to_odps_type(v, use_odps2_types=use_odps2_types))
            for k, v in df_type.field_types.items()
        )
        return odps_types.Struct(type_dict)
    else:
        raise KeyError(repr(df_type))


def df_schema_to_odps_schema(df_schema, ignorecase=False, project=None):
    names = [col.name.lower() if ignorecase else col.name
             for col in df_schema._columns]
    types = [
        df_type_to_odps_type(col.type, project=project)
        for col in df_schema._columns
    ]

    partition_names, partition_types = None, None
    if df_schema.partitions:
        partition_names = [
            col.name.lower() if ignorecase else col.name
            for col in df_schema.partitions
        ]
        partition_types = [
            df_type_to_odps_type(col.type, project=project)
            for col in df_schema.partitions
        ]

    return TableSchema.from_lists(
        names, types, partition_names=partition_names, partition_types=partition_types
    )



_use_odps2_types_local = threading.local()


def set_local_use_odps2_types(val):
    _use_odps2_types_local.value = val


def get_local_use_odps2_types(project=None):
    if options.sql.use_odps2_extension is None and project is not None:
        project_prop = project.properties.get("odps.sql.type.system.odps2")
        ret = ("true" == (project_prop or "false").lower())
        set_local_use_odps2_types(ret)
    elif getattr(_use_odps2_types_local, "value", None) is not None:
        ret = _use_odps2_types_local.value
    else:
        ret = bool(options.sql.use_odps2_extension)
    return ret
