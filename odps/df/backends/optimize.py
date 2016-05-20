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

from .core import Backend
from ...models import Schema
from ..expr.expressions import *
from ..expr.groupby import GroupByCollectionExpr, GroupbyAppliedCollectionExpr, MutateCollectionExpr
from ..expr.reduction import SequenceReduction, GroupedSequenceReduction
from ..expr.collections import DistinctCollectionExpr, RowAppliedCollectionExpr
from ..expr.utils import get_attrs
from ..expr.merge import JoinCollectionExpr
from ..expr.window import Window


class Optimizer(Backend):
    def __init__(self, expr, traversed=None, use_cache=None):
        self._expr = expr
        self._traversed = traversed or set()
        self._use_cache = use_cache if use_cache is not None \
            else options.df.use_cache

    def optimize(self):
        for node in self._expr.traverse(top_down=True, unique=True,
                                        traversed=self._traversed):
            try:
                node.accept(self)
            except NotImplementedError:
                continue

        return self._expr

    def _sub(self, expr, to_sub, parents):
        if not parents and expr is self._expr:
            self._expr = to_sub
        else:
            [p.substitute(expr, to_sub) for p in parents]

    def visit_filter_collection(self, expr):
        if not options.df.optimize:
            return

        if isinstance(expr.input, GroupByCollectionExpr) and \
                not expr.input.optimize_banned:
            # move filter on GroupBy to GroupBy's having
            predicate = self._broadcast_field(expr.predicate, expr.input)
            if predicate is None:
                predicate = expr.predicate

            having = expr.input.having
            if having is not None:
                predicates = having & predicate
            else:
                predicates = predicate

            parents = expr.parents

            attrs = get_attrs(expr.input)
            attrs_values = dict((attr, getattr(expr.input, attr)) for attr in attrs)
            attrs_values['_having'] = predicates
            del attrs_values['_cached_args']
            groupby_collection = type(expr.input)(**attrs_values)

            self._sub(expr, groupby_collection, parents)

        raise NotImplementedError

    def visit_project_collection(self, expr):
        # Summary does not attend here
        if not options.df.optimize:
            return

        parents = expr.parents

        compacted = self._visit_need_compact_collection(expr)
        if compacted:
            expr = compacted

        if isinstance(expr, ProjectCollectionExpr) and \
                isinstance(expr.input, GroupByCollectionExpr) and \
                not expr.input.optimize_banned:
            # compact projection into Groupby
            selects = []

            for field in expr._fields:
                selects.append(self._broadcast_field(field, expr.input) or field)

            attrs = get_attrs(expr.input)
            attrs_values = dict((attr, getattr(expr.input, attr)) for attr in attrs)
            attrs_values['_fields'] = selects
            attrs_values['_schema'] = Schema.from_lists([f.name for f in selects],
                                                        [f.dtype for f in selects])
            del attrs_values['_cached_args']
            groupby_collection = type(expr.input)(**attrs_values)

            self._sub(expr, groupby_collection, parents)
            return

    def visit_groupby(self, expr):
        if not options.df.optimize:
            return

        # we do not do compact on the projections from Join
        input = expr.input
        while isinstance(input, ProjectCollectionExpr):
            input = input._input
        if isinstance(input, JoinCollectionExpr):
            return

        if len(expr._aggregations) == 1 and \
                isinstance(expr._aggregations[0], GroupedSequenceReduction) and \
                isinstance(expr._aggregations[0]._input, CollectionExpr):
            # we just skip the case: df.groupby(***).count()
            return

        self._visit_need_compact_collection(expr)

    def visit_distinct(self, expr):
        if not options.df.optimize:
            return
        self._visit_need_compact_collection(expr)

    def visit_apply_collection(self, expr):
        if not options.df.optimize:
            return
        self._visit_need_compact_collection(expr)

    def _visit_need_compact_collection(self, expr):
        parents = expr.parents

        compacted = self._compact(expr)
        if compacted is None:
            return

        self._sub(expr, compacted, parents)
        return compacted

    def _compact(self, expr):
        to_compact = [expr, ]

        for node in expr.traverse(top_down=True, unique=True):
            if node is expr:
                continue
            if not isinstance(node, CollectionExpr):
                continue

            # We do not handle collection with Scalar column or window function here
            # TODO think way to compact in this situation
            if isinstance(node, ProjectCollectionExpr) and \
                    not node.optimize_banned and \
                    not any(isinstance(n, Window) for n in node._fields):
                valid = True
                for it in itertools.chain(*(node.all_path(to_compact[-1]))):
                    if isinstance(it, SequenceReduction):
                        valid = False
                        break
                if not valid:
                    break
                to_compact.append(node)
            else:
                break

        if len(to_compact) <= 1:
            return

        changed = False
        for field in self._get_fields(expr):
            if not isinstance(field, SequenceExpr):
                continue
            broadcast_field = self._broadcast_field(field, *to_compact[1:][::-1])
            if broadcast_field is not None:
                changed = True
                expr.substitute(field, broadcast_field)

        if changed:
            expr.substitute(expr.input, to_compact[-1].input)
            return expr

    def _broadcast_field(self, expr, *collects):
        changed = False
        retval = expr

        collection = collects[-1]
        for path in expr.all_path(collection, strict=True):
            cols = [it for it in path if isinstance(it, Column)]
            assert len(cols) <= 1
            assert len([it for it in path if isinstance(it, CollectionExpr)]) == 1
            if len(cols) == 1:
                col = cols[0]

                col_name = col.source_name or col.name
                field = self._get_field(collection, col_name)
                if col.is_renamed():
                    field = field.rename(col.name)
                else:
                    field = field.copy()

                parents = col.parents
                self._sub(col, field, parents)
                changed = True
                if col is retval:
                    retval = field

                if isinstance(field, Scalar) and field._value is not None:
                    continue
                if len(collects) > 1:
                    self._broadcast_field(field, *collects[:-1]) or field
            else:
                path[-2].substitute(collection, collection.input)

        if changed:
            return retval

    def _get_fields(self, collection):
        if isinstance(collection, (ProjectCollectionExpr, Summary)):
            return collection.fields
        elif isinstance(collection, DistinctCollectionExpr):
            return collection.unique_fields
        elif isinstance(collection, (GroupByCollectionExpr, MutateCollectionExpr)):
            return collection.fields
        elif isinstance(collection, (RowAppliedCollectionExpr, GroupbyAppliedCollectionExpr)):
            return collection.fields

    def _get_field(self, collection, name):
        # FIXME: consider name with upper letters
        if isinstance(collection, GroupByCollectionExpr):
            return collection._name_to_exprs()[name]

        name_idxes = collection.schema._name_indexes
        if name.lower() in name_idxes:
            idx = name_idxes[name.lower()]
        else:
            idx = name_idxes[name]
        return self._get_fields(collection)[idx]