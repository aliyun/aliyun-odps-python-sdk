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
from ..expr.expressions import *
from ..expr.groupby import GroupByCollectionExpr, SequenceGroupBy
from ..expr.reduction import SequenceReduction
from ..expr.collections import DistinctCollectionExpr
from ..expr.utils import get_attrs


class Optimizer(Backend):
    def __init__(self, expr, memo=None):
        self._expr = expr
        self._memo = dict() if memo is None else memo

    def optimize(self):
        for node in self._expr.traverse(parent_cache=self._memo, top_down=True,
                                        unique=True):
            try:
                node.accept(self)
            except NotImplementedError:
                continue

        return self._memo.get(id(self._expr)) or self._expr

    def _get_parent(self, expr):
        return self._memo.get(id(expr))

    def _sub(self, expr, to_sub):
        parents = self._get_parent(expr)
        if parents is None:
            self._expr = to_sub
        else:
            [p.substitute(expr, to_sub, parent_cache=self._memo)
             for p in set(parents)]

    def visit_filter_collection(self, expr):
        if isinstance(expr.input, GroupByCollectionExpr):
            predicate = self._broadcast_field(expr.predicate, expr.input)

            having = expr.input.having
            if having is not None:
                predicates = having & predicate
            else:
                predicates = predicate

            attrs = get_attrs(expr.input)
            attrs_values = dict((attr, getattr(expr.input, attr)) for attr in attrs)
            groupby_collection = type(expr.input)(**attrs_values)
            groupby_collection._having = predicates

            self._sub(expr, groupby_collection)

        raise NotImplementedError

    def visit_project_collection(self, expr):
        # Summary does not attend here
        self._visit_need_compact_collection(expr)

    def visit_groupby(self, expr):
        self._visit_need_compact_collection(expr)

    def visit_distinct(self, expr):
        self._visit_need_compact_collection(expr)

    def _visit_need_compact_collection(self, expr):
        compacted = self._compact(expr)
        if compacted is None:
            return

        self._sub(expr, compacted)

    def _add_parent_memo(self, expr, parent):
        if id(expr) not in self._memo:
            self._memo[id(expr)] = set([parent, ])
        else:
            self._memo[id(expr)].add(parent)

    def _compact(self, expr):
        to_compact = [expr, ]

        for node in expr.traverse(top_down=True, unique=True):
            if node is expr:
                continue
            if not isinstance(node, CollectionExpr):
                continue

            if isinstance(node, ProjectCollectionExpr) and \
                    not node.optimize_banned:
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
            self._add_parent_memo(field, expr)

            if not isinstance(field, SequenceExpr):
                continue
            broadcast_field = self._broadcast_field(field, to_compact[-1])
            if broadcast_field is not None:
                changed = True
                expr.substitute(field, broadcast_field, parent_cache=self._memo)

        if changed:
            expr.substitute(expr.input, to_compact[-1].input,
                            parent_cache=self._memo)
            return expr

    def _broadcast_field(self, expr, collection):
        changed = False
        for path in expr.all_path(collection, strict=True):
            col = None
            for i, node in enumerate(path):
                if isinstance(node, Column):
                    col = node
                    if i > 0:
                        self._add_parent_memo(col, path[i - 1])
                    continue
                elif isinstance(node, SequenceExpr):
                    continue
                elif isinstance(node, Scalar):
                    break

                # find a collection
                if col is not None:
                    field = self._get_field(node, col.source_name or col.name)

                    parents = set(self._memo[id(col)])
                    for parent in parents:
                        parent.substitute(col, field, parent_cache=self._memo)
                        self._add_parent_memo(field, parent)

                    col = field
                else:
                    if i > 0:
                        parent = path[i - 1]
                        parent.substitute(node, collection.input)
                        self._add_parent_memo(collection.input, parent)

                changed = True

        if changed:
            return expr

    def _get_fields(self, collection):
        if isinstance(collection, (ProjectCollectionExpr, Summary)):
            return collection.fields
        elif isinstance(collection, DistinctCollectionExpr):
            return collection.unique_fields
        elif isinstance(collection, GroupByCollectionExpr):
            return collection.fields

    def _get_field(self, collection, name):
        idx = collection.schema._name_indexes[name.lower()]
        return self._get_fields(collection)[idx]