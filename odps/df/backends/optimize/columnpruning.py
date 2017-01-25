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

from ..core import Backend
from ...expr.core import ExprProxy
from ...expr.expressions import Column, CollectionExpr, FilterCollectionExpr
from ...expr.collections import SortedCollectionExpr, SliceCollectionExpr
from ...expr.merge import JoinCollectionExpr, UnionCollectionExpr
from ....models import Schema
from ...utils import is_project_expr, traverse_until_source


class ColumnPruning(Backend):
    def __init__(self, dag):
        self._dag = dag

        # As some collections like join and union expr do not have fields,
        # we store them here to illustrate the fields which should keep.
        # Specially, for the JoinCollectionExpr,
        # we store a 2-tuple.
        self._remain_columns = dict()

    def prune(self):
        for node in traverse_until_source(self._dag, top_down=True):
            try:
                node.accept(self)
            except NotImplementedError:
                continue

        return self._dag.root

    def _need_prune(self, expr):
        parents = self._dag.successors(expr)
        if len(parents) == 0:
            return False, set()

        columns = set()
        for p in parents:
            if isinstance(p, Column) and p.input is expr:
                columns.add(p.source_name)
            elif isinstance(p, JoinCollectionExpr):
                proxy = ExprProxy(p)
                if proxy not in self._remain_columns:
                    continue
                lcols, rcols = self._remain_columns[proxy]
                if p._get_child(p._lhs) is expr:
                    columns = columns.union(lcols)
                elif p._get_child(p._rhs) is expr:
                    columns = columns.union(rcols)
                else:
                    continue
            elif isinstance(p, UnionCollectionExpr):
                proxy = ExprProxy(p)
                if proxy not in self._remain_columns:
                    continue
                cols = self._remain_columns[proxy]
                if p._lhs is expr or p._rhs is expr:
                    columns = columns.union(cols)
            else:
                proxy = ExprProxy(p)
                if proxy not in self._remain_columns:
                    continue
                if hasattr(p, 'input') and p.input is expr:
                    columns = columns.union(self._remain_columns[proxy])

        if len(columns) == 0:
            return False, set()

        if len(set(expr.schema.names) - columns) > 0:
            return True, sorted(columns)

        return False, sorted(columns)

    def _need_project_on_source(self, expr):
        def no_project(node):
            if is_project_expr(node):
                # has been projected out, just skip
                return True
            parents = self._dag.successors(node)
            if len(parents) == 1 and isinstance(parents[0], Column):
                return True
            if isinstance(node, (FilterCollectionExpr, SortedCollectionExpr,
                                 SliceCollectionExpr)) or node is expr:
                collection_parents = [n for n in parents
                                      if isinstance(n, CollectionExpr)]
                if len(collection_parents) == 1 and no_project(collection_parents[0]):
                    return True
            return False

        return not no_project(expr)

    def visit_source_collection(self, expr):
        if not self._need_project_on_source(expr):
            return

        prune, columns = self._need_prune(expr)
        if not prune:
            return

        collection = expr[list(columns)]
        self._dag.substitute(expr, collection)

    def visit_project_collection(self, expr):
        prune, columns = self._need_prune(expr)
        if not prune:
            return

        new_projected = expr.input.select(
            [c for c in expr._fields if c.name in columns])
        self._dag.substitute(expr, new_projected)

    def visit_apply_collection(self, expr):
        # the apply cannot be pruned because the input
        # and output of apply is defined by the user
        return

    def _visit_all_columns_project_collection(self, expr):
        prune, columns = self._need_prune(expr)
        self._remain_columns[ExprProxy(expr)] = columns or expr.input.schema.names
        if not prune:
            return

        expr._schema = Schema(columns=[expr._schema[c] for c in columns])

    def visit_filter_collection(self, expr):
        self._visit_all_columns_project_collection(expr)

    def visit_filter_partition_collection(self, expr):
        prune, columns = self._need_prune(expr)
        if not prune:
            return

        expr._fields = [f for f in expr._fields if f.name in columns]
        expr._schema = Schema.from_lists([f.name for f in expr._fields],
                                         [f.dtype for f in expr._fields])

    def visit_slice_collection(self, expr):
        self._visit_all_columns_project_collection(expr)

    def _visit_grouped(self, expr):
        prune, columns = self._need_prune(expr)
        if not prune:
            return

        if expr._fields is None:
            expr._fields = expr._by + expr._aggregations
        expr._fields = [f for f in expr._fields if f.name in columns]
        expr._schema = Schema(columns=[expr._schema[c] for c in columns])

    def visit_groupby(self, expr):
        self._visit_grouped(expr)

    def visit_mutate(self, expr):
        self._visit_grouped(expr)

    def visit_reshuffle(self, expr):
        self._visit_all_columns_project_collection(expr)

    def visit_value_counts(self, expr):
        # skip, by and count cannot be pruned
        return

    def visit_sort(self, expr):
        self._visit_all_columns_project_collection(expr)

    def visit_distinct(self, expr):
        # skip, the distinct fields cannot be pruned
        return

    def visit_sample(self, expr):
        self._remain_columns[ExprProxy(expr)] = expr.input.schema.names

    def visit_join(self, expr):
        prune, columns = self._need_prune(expr)

        columns = columns or expr.schema.names
        lhs_cols, rhs_cols = [], []
        for col in columns:
            idx, origin_col = expr._column_origins[col]
            (lhs_cols, rhs_cols)[idx].append(origin_col)

        self._remain_columns[ExprProxy(expr)] = (lhs_cols, rhs_cols)

        if not prune:
            return
        expr._schema = Schema(columns=[expr._schema[col] for col in columns])

    def visit_union(self, expr):
        prune, columns = self._need_prune(expr)
        self._remain_columns[ExprProxy(expr)] = columns or expr.schema.names
        if not prune:
            return

        expr._schema = Schema(columns=[expr._schema[c] for c in columns])