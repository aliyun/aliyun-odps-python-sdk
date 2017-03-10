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
import operator

from ..rewriter import BaseRewriter
from ...expr.merge import *
from ...utils import traverse_until_source


class Rewriter(BaseRewriter):
    def visit_project_collection(self, expr):
        self._rewrite_reduction_in_projection(expr)

    def visit_filter_collection(self, expr):
        self._rewrite_reduction_in_filter(expr)

    def visit_column(self, expr):
        if isinstance(expr.input, JoinCollectionExpr):
            input = expr.input
            name = expr.source_name
            while isinstance(input, JoinCollectionExpr):
                idx, name = input._column_origins[name]
                input = (input.lhs, input.rhs)[idx]

            self._sub(expr, input[name].rename(expr.name))

    def visit_join(self, expr):
        if expr._predicate and isinstance(expr._predicate, list):
            expr._predicate = reduce(operator.and_, expr._predicate)

        for node in (expr.rhs,):
            parents = self._parents(node)
            if isinstance(node, JoinCollectionExpr):
                projection = JoinProjectCollectionExpr(
                    _input=node, _schema=node.schema,
                    _fields=node._fetch_fields())
                self._sub(node, projection, parents)
            elif isinstance(node, JoinProjectCollectionExpr):
                self._sub(node.input, node, parents)

        need_project = [False, ]

        def walk(node):
            if isinstance(node, JoinCollectionExpr) and \
                    node.column_conflict:
                need_project[0] = True
                return

            if isinstance(node, JoinCollectionExpr):
                walk(node.lhs)

        walk(expr)

        if need_project[0]:
            parents = self._parents(expr)
            if not parents or \
                    not any(isinstance(parent, (ProjectCollectionExpr, JoinCollectionExpr))
                            for parent in parents):
                to_sub = expr[expr]
                self._sub(expr, to_sub, parents)