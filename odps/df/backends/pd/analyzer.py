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

import itertools

from ..core import Backend
from ...expr.expressions import *
from ...expr.reduction import GroupedSequenceReduction, \
    Moment, GroupedMoment, Kurtosis, GroupedKurtosis
from ...utils import traverse_until_source
from .... import compat

try:
    import numpy as np
except ImportError:
    pass


class Analyzer(Backend):
    def __init__(self, expr_dag, traversed=None, on_sub=None):
        self._dag = expr_dag
        self._indexer = itertools.count(0)
        self._traversed = traversed or set()
        self._on_sub = on_sub

    def analyze(self):
        for node in self._iter():
            self._visit_node(node)
            self._traversed.add(id(node))

        return self._dag.root

    def _iter(self):
        for node in traverse_until_source(self._dag, top_down=True,
                                          traversed=self._traversed):
            yield node

        while True:
            all_traversed = True
            for node in traverse_until_source(self._dag, top_down=True):
                if id(node) not in self._traversed:
                    all_traversed = False
                    yield node
            if all_traversed:
                break

    def _visit_node(self, node):
        try:
            node.accept(self)
        except NotImplementedError:
            return

    def _sub(self, expr, sub, parents=None):
        self._dag.substitute(expr, sub, parents=parents)
        if self._on_sub:
            self._on_sub(expr, sub)

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

    @staticmethod
    def _get_moment_sub_expr(expr, _input, order, center):
        def _group_mean(e):
            m = e.mean()
            if isinstance(expr, GroupedSequenceReduction):
                m = m.to_grouped_reduction(expr._grouped)
            return m

        def _order(e, o):
            if o == 1:
                return e
            else:
                return e ** o

        if not center:
            if order == 0:
                sub = Scalar(1)
            else:
                sub = _group_mean(_input ** order)
        else:
            if order == 0:
                sub = Scalar(1)
            elif order == 1:
                sub = Scalar(0)
            else:
                sub = _group_mean(_input ** order)
                divided = 1
                divisor = 1
                for o in compat.irange(1, order):
                    divided *= order - o + 1
                    divisor *= o
                    part_item = divided / divisor * _group_mean(_order(_input, order - o)) \
                                * (_order(_group_mean(_input), o))
                    if o & 1:
                        sub -= part_item
                    else:
                        sub += part_item
                part_item = _group_mean(_input) ** order
                if order & 1:
                    sub -= part_item
                else:
                    sub += part_item
        return sub

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

        raise NotImplementedError
