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

import inspect

from .core import DataFrame, CollectionExpr
from .expr.expressions import SequenceExpr, Scalar, BuiltinFunction, RandomScalar
from .expr.element import switch, FuncFactory
from .expr.datetimes import year, month, day, hour, minute, second, millisecond
from .expr.reduction import aggregate, agg
from .utils import output_types, output_names, output
from .delay import Delay
from ..compat import six


def NullScalar(tp):
    return Scalar(_value_type=tp)

func = FuncFactory()

try:
    import pandas as pd
    from pandas.io.api import *

    def wrap(func):
        def inner(*args, **kwargs):
            as_type = kwargs.pop('as_type', None)
            unknown_as_string = kwargs.pop('unknown_as_string', False)

            res = func(*args, **kwargs)
            if isinstance(res, pd.DataFrame):
                return DataFrame(res, as_type=as_type,
                                 unknown_as_string=unknown_as_string)
            return res
        return inner

    for k, v in six.iteritems(dict(locals())):
        if k.startswith('read_') and inspect.isfunction(v):
            locals()[k] = wrap(v)
except ImportError:
    pass
