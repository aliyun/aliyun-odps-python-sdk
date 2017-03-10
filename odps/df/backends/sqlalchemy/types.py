#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


try:
    import sqlalchemy
    has_sqlalchemy = True
except ImportError:
    has_sqlalchemy = False

from ... import types
from ....compat import OrderedDict, six
from ....models import Schema


_sqlalchemy_to_df_types = OrderedDict()
_df_to_sqlalchemy_types = OrderedDict()

if has_sqlalchemy:
    _sqlalchemy_to_df_types[sqlalchemy.CHAR] = types.int8
    _sqlalchemy_to_df_types[sqlalchemy.SmallInteger] = types.int16
    _sqlalchemy_to_df_types[sqlalchemy.BigInteger] = types.int64
    _sqlalchemy_to_df_types[sqlalchemy.Integer] = types.int32
    _sqlalchemy_to_df_types[sqlalchemy.Float] = types.float64
    _sqlalchemy_to_df_types[sqlalchemy.String] = types.string
    _sqlalchemy_to_df_types[sqlalchemy.Boolean] = types.boolean
    _sqlalchemy_to_df_types[sqlalchemy.DECIMAL] = types.decimal
    _sqlalchemy_to_df_types[sqlalchemy.DateTime] = types.datetime

    _df_to_sqlalchemy_types = dict((v, k) for k, v in six.iteritems(_sqlalchemy_to_df_types))
    _df_to_sqlalchemy_types[types.string] = sqlalchemy.Text  # we store text


def sqlalchemy_to_df_type(sqlalchemy_type):
    for sqlalchemy_type_cls, df_type in six.iteritems(_sqlalchemy_to_df_types):
        if isinstance(sqlalchemy_type, sqlalchemy_type_cls):
            return df_type

    raise ValueError(
        'Cannot convert SQLAlchemy type %s to dataframe type' % sqlalchemy_type)


def sqlalchemy_to_df_schema(sqlalchemy_columns):
    names, types = [], []
    for c in sqlalchemy_columns:
        names.append(c.name)
        types.append(sqlalchemy_to_df_type(c.type))

    return Schema.from_lists(names, types)


def df_type_to_sqlalchemy_type(df_type, engine=None):
    if isinstance(df_type, six.string_types):
        df_type = types.validate_data_type(df_type)

    if isinstance(df_type, types.Datetime):
        return sqlalchemy.DateTime(timezone=True)
    elif df_type == types.decimal:
        return sqlalchemy.DECIMAL(36, 18)

    if engine and engine.name == 'mysql' and df_type == types.float64:
        from sqlalchemy.dialects.mysql.types import DOUBLE, DECIMAL
        return DOUBLE()
    return _df_to_sqlalchemy_types[df_type]


def df_schema_to_sqlalchemy_columns(df_schema, ignorecase=False, engine=None):
    names = [col.name.lower() if ignorecase else col.name
             for col in df_schema._columns]
    types = [df_type_to_sqlalchemy_type(col.type, engine=engine) for col in df_schema._columns]

    return [sqlalchemy.Column(n, t) for n, t in zip(names, types)]