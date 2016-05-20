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

import inspect

from .core import DataFrame
from .expr.expressions import Scalar, BuiltinFunction
from .expr.element import switch
from .expr.datetimes import year, month, day, hour, minute, second, millisecond
from .utils import output_types, output_names, output
from ..compat import six

try:
    import pandas as pd
    from pandas.io.api import *

    def wrap(func):
        def inner(*args, **kwargs):
            res = func(*args, **kwargs)
            if isinstance(res, pd.DataFrame):
                return DataFrame(res)
            return res
        return inner

    for k, v in six.iteritems(dict(locals())):
        if k.startswith('read_') and inspect.isfunction(v):
            locals()[k] = wrap(v)
except ImportError:
    pass