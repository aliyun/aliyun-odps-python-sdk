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

from datetime import datetime
from decimal import Decimal

try:
    import numpy as np
    has_np = True
except ImportError:
    has_np = False

from ... import types
from ....models import Schema
from ....compat import six

_np_to_df_types = dict()
_df_to_np_types = dict()

if has_np:
    _np_int_types = list(map(np.dtype, [np.int_, np.int8, np.int16, np.int32, np.int64]))
    _np_float_types = list(map(np.dtype, [np.float, np.float32, np.float64]))

    for np_type in _np_int_types + _np_float_types:
        _np_to_df_types[np_type] = getattr(types, str(np_type))

    _np_to_df_types[np.dtype(np.bool)] = types.boolean
    _np_to_df_types[np.dtype(np.str)] = types.string

    _df_to_np_types = dict((v, k) for k, v in six.iteritems(_np_to_df_types))


def np_type_to_df_type(dtype, arr=None):
    if dtype in _np_to_df_types:
        return _np_to_df_types[dtype]

    if arr is None or len(arr) == 0:
        raise TypeError('Unknown dtype: %s' % dtype)

    for it in arr:
        if it is None:
            continue
        if isinstance(it, six.string_types):
            return types.string
        elif isinstance(it, datetime):
            return types.datetime
        elif isinstance(it, Decimal):
            return types.decimal
        else:
            raise TypeError('Unknown dtype: %s' % dtype)

    raise TypeError('Unknown dtype: %s' % dtype)


def pd_to_df_schema(pd_df):
    if pd_df.index.name is not None:
        pd_df.reset_index(inplace=True)

    dtypes = pd_df.dtypes
    names = list(pd_df.columns.values)

    df_types = []
    for i in range(len(dtypes)):
        arr = pd_df.ix[:, i]
        df_types.append(np_type_to_df_type(dtypes[i], arr))

    return Schema.from_lists(names, df_types)


def df_type_to_np_type(df_type):
    if df_type in _df_to_np_types:
        return _df_to_np_types[df_type]

    if df_type == types.datetime:
        return np.dtype(datetime)
    elif df_type == types.decimal:
        return np.dtype(Decimal)
    else:
        raise TypeError('Unknown DataFrame dtype: %s' % df_type)


def df_schema_to_pd_dtypes(df_schema):
    names = [col.name for col in df_schema.columns]
    dtypes = [df_type_to_np_type(col.type) for col in df_schema.types]

    import pandas as pd
    return pd.Series(dtypes, index=names)

