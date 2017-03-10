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

import datetime
import time

from ..analyzer import BaseAnalyzer
from ...expr.expressions import *
from ...expr.element import IntToDatetime
from ...expr.arithmetic import Power
from ...expr.datetimes import UnixTimestamp
from ...expr.reduction import GroupedSequenceReduction, \
    Moment, GroupedMoment, Kurtosis, GroupedKurtosis

try:
    import numpy as np
except ImportError:
    pass


class Analyzer(BaseAnalyzer):
    def visit_builtin_function(self, expr):
        try:
            collection = six.next(iter(n for n in self._dag.iter_descendants(expr)
                                       if isinstance(n, CollectionExpr))).input
        except StopIteration:
            raise NotImplementedError

        if isinstance(expr, RandomScalar):
            seed = expr._func_args[0] if len(expr._func_args) >= 1 else None
            if seed is not None:
                np.random.seed(seed)

            col = getattr(collection, collection.schema.names[0])
            self._sub(expr, col.map(lambda v: np.random.rand()).astype('float'))
        else:
            raise NotImplementedError

    def visit_element_op(self, expr):
        if isinstance(expr, IntToDatetime):
            sub = expr.input.map(lambda n: datetime.datetime.fromtimestamp(n))
            self._sub(expr, sub)
        else:
            raise NotImplementedError

    def visit_binary_op(self, expr):
        if isinstance(expr, Power) and isinstance(expr._lhs.dtype, types.Integer) and \
                isinstance(expr._rhs.dtype, types.Integer):
            self._sub(expr._lhs, expr._lhs.astype('float'), parents=[expr])
        else:
            raise NotImplementedError

    def visit_datetime_op(self, expr):
        if isinstance(expr, UnixTimestamp):
            sub = expr.input.map(lambda d: time.mktime(d.timetuple()))
            self._sub(expr, sub)
        else:
            raise NotImplementedError

    def visit_reduction(self, expr):
        if isinstance(expr, (Moment, GroupedMoment)):
            order = expr._order
            center = expr._center

            sub = self._get_moment_sub_expr(expr, expr.input, order, center)
            sub = sub.rename(expr.name)
            self._sub(expr, sub)
            return
        elif isinstance(expr, (Kurtosis, GroupedKurtosis)):
            std = expr.input.std(ddof=0)
            if isinstance(expr, GroupedSequenceReduction):
                std = std.to_grouped_reduction(expr._grouped)
            cnt = expr.input.count()
            if isinstance(expr, GroupedSequenceReduction):
                cnt = cnt.to_grouped_reduction(expr._grouped)
            m4 = self._get_moment_sub_expr(expr, expr.input, 4, True)
            sub = 1.0 / (cnt - 2) / (cnt - 3) * ((cnt * cnt - 1) * m4 / (std ** 4) - 3 * (cnt - 1) ** 2)
            sub = sub.rename(expr.name)
            self._sub(expr, sub)
            return

        raise NotImplementedError
