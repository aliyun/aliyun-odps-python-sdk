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

from datetime import datetime, date
from decimal import Decimal

try:
    import numpy as np
    has_np = True
except ImportError:
    has_np = False

from ... import types
from ....models import Schema
from ....compat import six, OrderedDict

_np_to_df_types = dict()
_df_to_np_types = dict()

if has_np:
    _np_int_types = list(map(np.dtype, [np.int_, np.int8, np.int16, np.int32, np.int64]))
    _np_float_types = list(map(np.dtype, [np.float_, np.float32, np.float64]))

    for np_type in _np_int_types + _np_float_types:
        _np_to_df_types[np_type] = getattr(types, str(np_type))

    _np_to_df_types[np.dtype(np.bool_)] = types.boolean
    _np_to_df_types[np.dtype(np.str_)] = types.string

    _df_to_np_types = dict((v, k) for k, v in six.iteritems(_np_to_df_types))


def np_type_to_df_type(dtype, arr=None, unknown_as_string=False, name=None):
    if dtype in _np_to_df_types:
        return _np_to_df_types[dtype]

    name = ', field: ' + name if name else ''
    if arr is None or len(arr) == 0:
        if dtype == np.dtype(object) and unknown_as_string:
            return types.string
        raise TypeError('Unknown dtype: %s%s' % (dtype, name))

    for it in arr:
        if it is None:
            continue
        if isinstance(it, six.string_types):
            return types.string
        elif isinstance(it, six.integer_types):
            return types.int64
        elif isinstance(it, float):
            return types.float64
        elif isinstance(it, datetime):
            return types.datetime
        elif isinstance(it, Decimal):
            return types.decimal
        elif unknown_as_string:  # not inferred
            return types.string
        else:
            raise TypeError('Unknown dtype: %s%s' % (dtype, name))

    if unknown_as_string:
        return types.string
    raise TypeError('Unknown dtype: %s' % dtype)


def pd_to_df_schema(pd_df, unknown_as_string=False, as_type=None):
    if pd_df.index.name is not None:
        pd_df.reset_index(inplace=True)

    dtypes = pd_df.dtypes
    names = list(pd_df.columns.values)

    df_types = []
    for i in range(len(dtypes)):
        arr = pd_df.iloc[:, i]
        if as_type and names[i] in as_type:
            df_types.append(as_type[names[i]])
            continue
        df_types.append(np_type_to_df_type(dtypes.iloc[i], arr,
                                           unknown_as_string=unknown_as_string,
                                           name=names[i]))

    return Schema.from_lists(names, df_types)


def df_type_to_np_type(df_type):
    if df_type in _df_to_np_types:
        return _df_to_np_types[df_type]

    if df_type == types.datetime:
        return np.dtype(datetime)
    elif df_type == types.date:
        return np.dtype(date)
    elif df_type == types.timestamp:
        return np.datetime64(0, 'ns').dtype
    elif df_type == types.decimal:
        return np.dtype(Decimal)
    elif df_type == types.binary:
        return np.dtype('bytes')
    else:
        raise TypeError('Unknown DataFrame dtype: %s' % df_type)


def df_schema_to_pd_dtypes(df_schema):
    names = [col.name for col in df_schema.columns]
    dtypes = [df_type_to_np_type(col.type) for col in df_schema.types]

    import pandas as pd
    return pd.Series(dtypes, index=names)


def cast_composite_sequence(pd_seq, df_type):
    def _cast(v, df_type):
        if v is None:
            return None
        elif isinstance(df_type, types.Primitive):
            return df_type_to_np_type(df_type).type(v)
        elif isinstance(df_type, types.List) and isinstance(v, list):
            return [_cast(it, df_type.value_type) for it in v]
        elif isinstance(df_type, types.Dict) and isinstance(v, dict):
            return OrderedDict((_cast(it_key, df_type.key_type), _cast(it_value, df_type.value_type))
                               for it_key, it_value in six.iteritems(v))
        else:
            raise TypeError('Cannot convert value %r into dtype %s' % (v, df_type))

    return pd_seq.map(lambda v: _cast(v, df_type))
