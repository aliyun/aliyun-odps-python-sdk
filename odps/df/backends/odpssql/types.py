#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from ....models import Schema
from .... import types as odps_types
from ... import types as df_types
from ....compat import six


_odps_to_df_types = {
    odps_types.bigint: df_types.int64,
    odps_types.double: df_types.float64,
    odps_types.string: df_types.string,
    odps_types.datetime: df_types.datetime,
    odps_types.boolean: df_types.boolean,
    odps_types.decimal: df_types.decimal
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
    df_types.decimal: odps_types.decimal,
    df_types.datetime: odps_types.datetime
}


def odps_type_to_df_type(odps_type):
    if isinstance(odps_type, six.string_types):
        odps_type = odps_types.validate_data_type(odps_type)

    return _odps_to_df_types[odps_type]


def odps_schema_to_df_schema(odps_schema):
    names = [col.name for col in odps_schema.columns]
    types = [odps_type_to_df_type(col.type) for col in odps_schema.columns]

    return Schema.from_lists(names, types)


def df_type_to_odps_type(df_type):
    if isinstance(df_type, six.string_types):
        df_type = df_types.validate_data_type(df_type)

    return _df_to_odps_types[df_type]


def df_schema_to_odps_schema(df_schema, ignorecase=False):
    names = [col.name.lower() if ignorecase else col.name
             for col in df_schema._columns]
    types = [df_type_to_odps_type(col.type) for col in df_schema._columns]

    partition_names, partition_types = None, None
    if df_schema.partitions:
        partition_names = [col.name.lower() if ignorecase else col.name
                           for col in df_schema.partitions]
        partition_types = [df_type_to_odps_type(col.type)
                           for col in df_schema.partitions]

    return Schema.from_lists(names, types,
                             partition_names=partition_names,
                             partition_types=partition_types)