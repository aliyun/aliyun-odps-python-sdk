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

from ..expr.core import ExprProxy, ExprDAG


class ExecuteContext(object):
    def __init__(self):
        self._expr_to_dag = dict()
        self._expr_to_copied = dict()

    def register_to_copy_expr(self, src_expr, expr=None, rebuilt=False, on_copy=None):
        if not rebuilt and ExprProxy(src_expr) in self._expr_to_copied:
            return self._expr_to_copied[ExprProxy(src_expr)]

        if expr is None:
            copied = src_expr.copy_tree(on_copy=on_copy)
        else:
            copied = expr.copy_tree(on_copy=on_copy)

        proxy = ExprProxy(src_expr, self._expr_to_copied)
        self._expr_to_copied[proxy] = copied

        return copied

    def is_dag_built(self, expr):
        return ExprProxy(expr) in self._expr_to_dag

    def build_dag(self, src_expr, expr, rebuilt=False, dag=None):
        if dag is not None:
            return ExprDAG(expr, dag)
        if not rebuilt and self.is_dag_built(src_expr):
            return self.get_dag(src_expr)

        dag = expr.to_dag(copy=False)
        self._expr_to_dag[ExprProxy(src_expr, self._expr_to_dag)] = dag

        return dag

    def get_dag(self, expr):
        return self._expr_to_dag[ExprProxy(expr)]

context = ExecuteContext()
