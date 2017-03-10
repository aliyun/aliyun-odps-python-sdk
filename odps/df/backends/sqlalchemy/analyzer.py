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

from ..analyzer import BaseAnalyzer
from ...expr.element import Cut, Between
from ...expr.arithmetic import *
from ...expr.math import Log1p, Expm1
from ...expr.strings import *
from ...expr.window import CumSum
from ...expr.reduction import *
from ...expr.collections import PivotCollectionExpr
from ... import types


class Analyzer(BaseAnalyzer):
    def visit_element_op(self, expr):
        if isinstance(expr, Between) and not expr.inclusive:
            sub = ((expr.left < expr.input) & (expr.input.copy() < expr.right))
            self._sub(expr, sub.rename(expr.name))
        elif isinstance(expr, Cut):
            sub = self._get_cut_sub_expr(expr)
            self._sub(expr, sub)

    def visit_value_counts(self, expr):
        self._sub(expr, self._get_value_counts_sub_expr(expr))

    def visit_binary_op(self, expr):
        if isinstance(expr, Divide) and isinstance(expr._lhs.dtype, types.Integer) and \
                isinstance(expr._rhs.dtype, types.Integer):
            self._sub(expr._lhs, expr._lhs.astype('float'), parents=[expr])
        elif isinstance(expr, FloorDivide):
            lhs = (expr._lhs < 0).ifelse(expr._lhs - 1, expr._lhs)
            self._sub(expr._lhs, lhs, parents=[expr])

    def visit_unary_op(self, expr):
        if isinstance(expr, Invert) and isinstance(expr.input.dtype, types.Integer):
            sub = -expr._input - 1
            self._sub(expr, sub)
        elif isinstance(expr, Negate) and expr.input.dtype == types.boolean:
            input = expr._input.ifelse(1, 0)
            self._sub(expr.input, input)

    def visit_math(self, expr):
        if isinstance(expr, Log1p):
            sub = (1 + expr.input).log()
            self._sub(expr, sub)
        elif isinstance(expr, Expm1):
            sub = expr.input.exp() - 1
            self._sub(expr, sub)

    def visit_string_op(self, expr):
        if isinstance(expr, Ljust):
            rest = expr.width - expr.input.len()
            sub = expr.input + \
                     (rest >= 0).ifelse(expr._fillchar.repeat(rest), '')
            self._sub(expr, sub.rename(expr.name))
        elif isinstance(expr, Rjust):
            rest = expr.width - expr.input.len()
            sub = (rest >= 0).ifelse(expr._fillchar.repeat(rest), '') + expr.input
            self._sub(expr, sub.rename(expr.name))
        elif isinstance(expr, Zfill):
            fillchar = Scalar('0')
            rest = expr.width - expr.input.len()
            sub = (rest >= 0).ifelse(fillchar.repeat(rest), '') + expr.input
            self._sub(expr, sub.rename(expr.name))
        elif isinstance(expr, CatStr):
            input = expr.input
            others = expr._others if isinstance(expr._others, Iterable) else (expr._others, )
            for other in others:
                if expr.na_rep is not None:
                    for e in (input, ) + tuple(others):
                        self._sub(e, e.fillna(expr.na_rep), parents=(expr, ))
                    return
                else:
                    if expr._sep is not None:
                        input = other.isnull().ifelse(input, input + expr._sep + other)
                    else:
                        input = other.isnull().ifelse(input, input + other)
            self._sub(expr, input.rename(expr.name))

    def visit_cum_window(self, expr):
        if isinstance(expr, CumSum) and expr.input.dtype == types.boolean:
            sub = expr.input.ifelse(1, 0)
            self._sub(expr.input, sub, parents=[expr])

    def _visit_pivot(self, expr):
        sub = self._get_pivot_sub_expr(expr)
        self._sub(expr, sub)

    def _visit_pivot_table(self, expr):
        sub = self._get_pivot_table_sub_expr(expr)
        self._sub(expr, sub)

    def visit_pivot(self, expr):
        if isinstance(expr, PivotCollectionExpr):
            self._visit_pivot(expr)
        else:
            self._visit_pivot_table(expr)

    def visit_reduction(self, expr):
        if isinstance(expr, (Max, GroupedMax, Min, GroupedMin, Sum, GroupedSum)) and \
                expr.input.dtype == types.boolean:
            self._sub(expr.input, expr.input.ifelse(1, 0), parents=[expr])

        if isinstance(expr, (Sum, GroupedSum)) and expr.input.dtype == types.string:
            sub = expr.input.cat(sep='')
            if isinstance(expr, GroupedSum):
                sub = sub.to_grouped_reduction(expr._grouped)
            sub = sub.rename(expr.name)
            self._sub(expr, sub)
        elif isinstance(expr, (Any, GroupedAny)):
            input = expr.input.ifelse(1, 0).max() == 1
            if isinstance(expr, GroupedAny):
                input = input.to_grouped_reduction(expr._grouped)
            sub = input.rename(expr.name)
            self._sub(expr, sub)
        elif isinstance(expr, (All, GroupedAll)):
            input = expr.input.ifelse(1, 0).min() == 1
            if isinstance(expr, GroupedAll):
                input = input.to_grouped_reduction(expr._grouped)
            sub = input.rename(expr.name)
            self._sub(expr, sub)
        elif isinstance(expr, (Moment, GroupedMoment)):
            order = expr._order
            center = expr._center

            sub = self._get_moment_sub_expr(expr, expr.input, order, center)
            sub = sub.rename(expr.name)
            self._sub(expr, sub)
        elif isinstance(expr, (Skewness, GroupedSkewness)):
            std = expr.input.std(ddof=1)
            if isinstance(expr, GroupedSequenceReduction):
                std = std.to_grouped_reduction(expr._grouped)
            cnt = expr.input.count()
            if isinstance(expr, GroupedSequenceReduction):
                cnt = cnt.to_grouped_reduction(expr._grouped)
            sub = self._get_moment_sub_expr(expr, expr.input, 3, True) / (std ** 3)
            sub *= (cnt ** 2) / (cnt - 1) / (cnt - 2)
            sub = sub.rename(expr.name)
            self._sub(expr, sub)
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
