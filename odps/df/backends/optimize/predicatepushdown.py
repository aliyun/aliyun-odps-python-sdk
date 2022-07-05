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

import operator
import itertools

from ..core import Backend
from ...expr.expressions import ProjectCollectionExpr, FilterCollectionExpr, \
    LateralViewCollectionExpr, Column, SequenceExpr
from ...expr.element import IsIn, Func, MappedExpr
from ...expr.merge import InnerJoin
from ...expr.arithmetic import And
from ...expr.reduction import SequenceReduction
from ...expr.window import Window
from ...expr.merge import UnionCollectionExpr
from ...utils import traverse_until_source
from .utils import change_input, copy_sequence
from ....compat import reduce


class PredicatePushdown(Backend):
    def __init__(self, dag):
        self._dag = dag
        self._traversed = set()

    def pushdown(self):
        for node in traverse_until_source(self._dag, top_down=True, traversed=self._traversed):
            try:
                node.accept(self)
            except NotImplementedError:
                continue

        return self._dag.root

    def visit_filter_collection(self, expr):
        from .core import Optimizer

        predicates = self._split_filter_into_predicates(expr)

        if any(isinstance(seq, (Window, SequenceReduction, Func))
               for seq in itertools.chain(*expr.all_path(expr.input, strict=True))):
            # if any reduction like sum or window op exists, skip
            return
        if any(isinstance(p, IsIn) and
                (p._values is None or
                 (len(p._values) == 1 and isinstance(p._values[0], SequenceExpr)))
               for p in predicates):
            return

        filters = [expr]
        input = expr.input
        while isinstance(input, FilterCollectionExpr):
            filters.append(input)
            input = input.input

        def sub():
            if len(filters) > 1:
                new_expr = Optimizer.get_compact_filters(self._dag, *filters)
                old_predicate = expr._predicate
                expr._predicate = new_expr.predicate
                self._dag.substitute(old_predicate, expr._predicate, parents=[expr])
                [self._traversed.add(id(i)) for i in filters[1:]]
                return True
        if isinstance(input, ProjectCollectionExpr):
            if any(isinstance(seq, (Window, SequenceReduction, MappedExpr))
                   for seq in input.traverse(top_down=True, unique=True,
                                             stop_cond=lambda x: x is input.input)):
                # if any reduction like sum or window op exists, skip
                return
            if sub():
                predicates = self._split_filter_into_predicates(expr)
            predicate = expr.predicate
            remain = None
            if isinstance(input, LateralViewCollectionExpr):
                udtf_columns = set()
                for lv in input.lateral_views:
                    udtf_columns.update(lv.schema.names)
                pushes = []
                remains = []
                for p in predicates:
                    cols = [col.source_name for path in p.all_path(input, strict=True)
                            for col in path if isinstance(col, Column)]
                    if not (set(cols) & udtf_columns):
                        pushes.append(p)
                    else:
                        remains.append(p)
                if not pushes:
                    return
                predicate = reduce(operator.and_, pushes)
                if remains:
                    remain = reduce(operator.and_, remains)
            self._push_down_through_projection(predicate, expr, input)
            if remain is None:
                self._dag.substitute(expr, expr.input)
            else:
                expr.substitute(expr._predicate, remain, dag=self._dag)
        elif isinstance(input, InnerJoin):
            if sub():
                predicates = self._split_filter_into_predicates(expr)
            remains = []
            predicates = predicates[::-1]
            for i, predicate in enumerate(predicates):
                collection = self._predicate_on_same_collection(
                    predicate, expr.input, input)
                if collection is not False:
                    self._push_down_through_join(predicate, expr, input, collection)
                else:
                    remains.append(i)
            if len(remains) == 0:
                self._dag.substitute(expr, expr.input)
            else:
                expr.substitute(expr._predicate,
                                reduce(operator.and_, [predicates[i] for i in remains]),
                                dag=self._dag)
        elif isinstance(input, UnionCollectionExpr):
            sub()
            self._push_down_through_union(expr.predicate, expr, input)
            self._dag.substitute(expr, expr.input)

    def visit_join(self, expr):
        if not isinstance(expr, InnerJoin):
            return

        predicates = expr._predicate
        if predicates is None:
            return

        splits = self._split_predicates(predicates)
        rests = []

        left_filters, right_filters = [], []
        for i, p in enumerate(splits):
            collection = None
            skip = False
            for node in (expr.lhs, expr.rhs):
                if p.is_ancestor(node):
                    if collection is not None:
                        skip = True
                        break
                    collection = node
            if skip:
                rests.append(i)
                continue

            if collection is expr.lhs:
                left_filters.append(p)
            else:
                right_filters.append(p)

        if left_filters:
            left_filters = [copy_sequence(f, expr.lhs) for f in left_filters]
            new_lhs = expr.lhs.filter(*left_filters)
            self._dag.substitute(expr.lhs, new_lhs)
        if right_filters:
            right_filters = [copy_sequence(f, expr.rhs) for f in right_filters]
            new_rhs = expr.rhs.filter(*right_filters)
            self._dag.substitute(expr.rhs, new_rhs)

        if len(rests) != len(splits):
            new_predicates = [splits[i] for i in rests]
            join = expr.copy()
            join._predicate = new_predicates
            self._dag.substitute(expr, join)

    @classmethod
    def _split_predicates(cls, predicats):
        splits = [p for p in predicats if not isinstance(p, And)]
        nodes = list(p for p in predicats if isinstance(p, And))
        while len(nodes) > 0:
            node = nodes.pop(0)
            for s in (node.lhs, node.rhs):
                if not isinstance(s, And):
                    splits.append(s)
                else:
                    nodes.append(s)

        return splits

    @classmethod
    def _split_filter_into_predicates(cls, expr):
        predicate = expr.predicate
        if not isinstance(predicate, And):
            return [predicate]

        return cls._split_predicates([predicate])

    @classmethod
    def _predicate_on_same_collection(cls, predicate, input, joined):
        origins = set()

        cols = [col.source_name for path in predicate.all_path(input, strict=True)
                for col in path if isinstance(col, Column)]
        for col in cols:
            origins.add(joined._column_origins[col][0])
            if len(origins) > 1:
                return False

        # todo maybe broadcast in every collection in future
        if len(origins) == 0:
            return False

        return (joined.lhs, joined.rhs)[origins.pop()]

    def _push_down_through_projection(self, predicate, expr, projection):
        def get_project_field(n, col):
            field = projection._fields[projection.schema._name_indexes[col]]
            return copy_sequence(field, projection.input, self._dag)
        if isinstance(predicate, Column) and predicate.input is expr.input:
            # filter on column
            predicate = get_project_field(projection.input, predicate.source_name)
            if predicate.is_renamed():
                predicate = predicate.rename(predicate.name)
        else:
            change_input(predicate, expr.input, projection.input, get_project_field, self._dag)

        filter_expr = FilterCollectionExpr(
            projection.input, predicate, _schema=projection.input.schema)
        get_filter_field = lambda filter_node, col: filter_node[col]
        change_input(projection, projection.input, filter_expr, get_filter_field, self._dag)

    def _push_down_through_join(self, predicate, expr, joined, collection):
        def get_join_field(node, col):
            return node[joined._column_origins[col][1]]

        change_input(predicate, expr.input, collection, get_join_field, self._dag)

        filter_expr = FilterCollectionExpr(
            collection, predicate, _schema=collection.schema)
        get_field = lambda n, col: n[col]
        change_input(joined, collection, filter_expr, get_field, self._dag)

    def _push_down_through_union(self, predicate, expr, union):
        get_field = lambda n, col: n[col]

        predicates = [copy_sequence(predicate, expr.input, self._dag) for _ in range(2)]
        for predicate, collection in zip(predicates, (union.lhs, union.rhs)):
            change_input(predicate, expr.input, collection, get_field, self._dag)

            filter_expr = FilterCollectionExpr(collection, predicate, _schema=collection.schema)
            change_input(union, collection, filter_expr, get_field, self._dag)
