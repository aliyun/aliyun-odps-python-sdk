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
from ...expr.merge import JoinCollectionExpr


class Analyzer(Backend):
    def __init__(self, expr):
        self._expr = expr
        self._indexer = itertools.count(0)

        self._iters = []

    def analyze(self):
        traversed = set()

        for node in self._iter():
            self._visit_node(node, traversed)

        return self._expr

    def _iter(self):
        for node in self._expr.traverse(top_down=True):
            yield node

        for node in itertools.chain(
                *(it.traverse(top_down=True)
                  for it in self._iters)):
            yield node

    def _visit_node(self, node, traversed):
        if id(node) not in traversed:
            traversed.add(id(node))
            try:
                node.accept(self)
            except NotImplementedError:
                return

    def visit_project_collection(self, expr):
        if not isinstance(expr, ProjectCollectionExpr) or \
                not isinstance(expr.input, JoinCollectionExpr):
            return

        joined = expr.input
        lhs = joined._get_child(expr.input._lhs)
        rhs = joined._get_child(expr.input._rhs)

        for field in expr._fields:
            if field.is_ancestor(joined):
                continue

            idx = 0 if field.is_ancestor(lhs) else 1
            child = (lhs, rhs)[idx]

            for path in field.all_path(child, strict=True):
                # TODO modification may not be applied to the paths
                if not isinstance(path[-2], Column):
                    continue

                col = path[-2]

                src_name = col.source_name
                if src_name in joined._renamed_columns:
                    src_name = joined._renamed_columns[src_name][idx]

                new_col = joined[src_name]
                if col.is_renamed():
                    new_col = new_col.rename(col._name)

                parent = expr if len(path) < 3 else path[-3]
                parent.substitute(col, new_col)
                self._iters.append(new_col)
