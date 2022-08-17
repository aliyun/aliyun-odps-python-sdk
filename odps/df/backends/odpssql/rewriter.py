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

from ..rewriter import BaseRewriter
from ...expr.reduction import Cat, GroupedCat
from ...expr.window import *
from ...expr.merge import *
from ...expr.expressions import LateralViewCollectionExpr
from ...expr.element import Func
from ...expr.utils import get_attrs
from ....errors import NoSuchObject
from ...utils import is_source_collection
from ....models import Table


class Rewriter(BaseRewriter):
    def visit_project_collection(self, expr):
        self._rewrite_reduction_in_projection(expr)

    def visit_filter_collection(self, expr):
        self._rewrite_reduction_in_filter(expr)

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

    def visit_lateral_view(self, expr):
        parents = self._parents(expr)
        if not parents \
                or not any(isinstance(parent, ProjectCollectionExpr) \
                           and not isinstance(parent, LateralViewCollectionExpr) for parent in parents):
            to_sub = ProjectCollectionExpr(
                _input=expr, _schema=expr.schema, _fields=expr._fields
            )
            self._sub(expr, to_sub, parents)

    def _handle_function(self, expr, raw_inputs):
        # Since Python UDF cannot support decimal field,
        # We will try to replace the decimal input with string.
        # If the output is decimal, we will also try to replace it with string,
        # and then cast back to decimal
        def no_output_decimal():
            if isinstance(expr, (SequenceExpr, Scalar)):
                return expr.dtype != types.decimal
            else:
                return all(t != types.decimal for t in expr.schema.types)

        if isinstance(expr, Func):
            return

        if all(input.dtype != types.decimal for input in raw_inputs) and \
                no_output_decimal():
            return

        inputs = list(raw_inputs)
        for input in raw_inputs:
            if input.dtype == types.decimal:
                self._sub(input, input.astype('string'), parents=[expr, ])
        if hasattr(expr, '_raw_inputs'):
            expr._raw_inputs = inputs
        else:
            assert len(inputs) == 1
            expr._raw_input = inputs[0]

        attrs = get_attrs(expr)
        attr_values = dict((attr, getattr(expr, attr, None)) for attr in attrs)
        if isinstance(expr, (SequenceExpr, Scalar)):
            if expr.dtype == types.decimal:
                if isinstance(expr, SequenceExpr):
                    attr_values['_data_type'] = types.string
                    attr_values['_source_data_type'] = types.string
                else:
                    attr_values['_value_type'] = types.string
                    attr_values['_source_value_type'] = types.string
            sub = type(expr)._new(**attr_values)

            if expr.dtype == types.decimal:
                sub = sub.astype('decimal')
        else:
            names = expr.schema.names
            tps = expr.schema.types
            cast_names = set()
            if any(tp == types.decimal for tp in tps):
                new_tps = []
                for name, tp in zip(names, tps):
                    if tp != types.decimal:
                        new_tps.append(tp)
                        continue
                    new_tps.append(types.string)
                    cast_names.add(name)
                if len(cast_names) > 0:
                    attr_values['_schema'] = Schema.from_lists(names, new_tps)
            sub = type(expr)(**attr_values)

            if len(cast_names) > 0:
                fields = []
                for name in names:
                    if name in cast_names:
                        fields.append(sub[name].astype('decimal'))
                    else:
                        fields.append(name)
                sub = sub[fields]

        self._sub(expr, sub)

    def visit_function(self, expr):
        self._handle_function(expr, expr._inputs)

    def visit_reshuffle(self, expr):
        if isinstance(expr.input, JoinCollectionExpr):
            sub = JoinProjectCollectionExpr(
                _input=expr.input, _schema=expr.input.schema,
                _fields=expr.input._fetch_fields())
            self._sub(expr.input, sub)

    def visit_apply_collection(self, expr):
        if (
            isinstance(expr._input, JoinCollectionExpr)
            and (expr._input._mapjoin or expr._input._skewjoin)
        ):
            node = expr._input
            projection = JoinProjectCollectionExpr(
                _input=node, _schema=node.schema,
                _fields=node._fetch_fields())
            self._sub(node, projection)
        self._handle_function(expr, expr._fields)

    def visit_user_defined_aggregator(self, expr):
        self._handle_function(expr, [expr.input, ])

    def visit_column(self, expr):
        if is_source_collection(expr.input) and isinstance(expr._input._source_data, Table):
            try:
                if expr.input._source_data.schema.is_partition(expr.source_name) and \
                                expr.dtype != types.string:
                    expr._source_data_type = types.string
            except NoSuchObject:
                return

    def visit_reduction(self, expr):
        if isinstance(expr, (Cat, GroupedCat)):
            if expr._na_rep is not None:
                input = expr.input.fillna(expr._na_rep)
                self._sub(expr.input, input, parents=(expr, ))