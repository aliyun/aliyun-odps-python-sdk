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
from ...expr.expressions import Summary
from ...expr.reduction import SequenceReduction
from ...expr.window import *
from ...expr.merge import *
from ...expr.utils import get_attrs
from ....errors import NoSuchObject
from ...utils import is_source_collection


class Rewriter(Backend):
    def __init__(self, dag, traversed=None):
        self._expr = dag.root
        self._dag = dag
        self._indexer = itertools.count(0)
        self._traversed = traversed or set()

    def rewrite(self):
        for node in self._iter():
            self._traversed.add(id(node))
            self._visit_node(node)

        return self._dag.root

    def _iter(self):
        for node in self._expr.traverse(top_down=True, unique=True,
                                        traversed=self._traversed):
            yield node

        while True:
            all_traversed = True
            for node in self._expr.traverse(top_down=True, unique=True):
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

    def _sub(self, expr, to_sub, parents=None):
        self._dag.substitute(expr, to_sub, parents=parents)

    def _parents(self, expr):
        return self._dag.successors(expr)

    def visit_project_collection(self, expr):
        # FIXME how to handle nested reduction?
        if isinstance(expr, Summary):
            return

        collection = expr.input

        sink_selects = []
        columns = set()
        to_replace = []

        windows_rewrite = False
        for field in expr.fields:
            has_window = False
            traversed = set()
            for path in field.all_path(collection, strict=True):
                for node in path:
                    if id(node) in traversed:
                        continue
                    else:
                        traversed.add(id(node))
                    if isinstance(node, SequenceReduction):
                        windows_rewrite = True
                        has_window = True

                        win = self._reduction_to_window(node)
                        window_name = '%s_%s' % (win.name, next(self._indexer))
                        sink_selects.append(win.rename(window_name))
                        to_replace.append((node, window_name))
                        break
                    elif isinstance(node, Column):
                        if node.input is not collection:
                            continue

                        if node.name in columns:
                            to_replace.append((node, node.name))
                            continue
                        columns.add(node.name)
                        select_field = collection[node.source_name]
                        if node.is_renamed():
                            select_field = select_field.rename(node.name)
                        sink_selects.append(select_field)
                        to_replace.append((node, node.name))

            if has_window:
                field._name = field.name

        if not windows_rewrite:
            return

        get = lambda x: x.name if not isinstance(x, six.string_types) else x
        projected = collection[sorted(sink_selects, key=get)]
        projected.optimize_banned = True  # TO prevent from optimizing
        expr.substitute(collection, projected, dag=self._dag)

        for col, col_name in to_replace:
            self._sub(col, projected[col_name])

    def visit_filter_collection(self, expr):
        # FIXME how to handle nested reduction?
        collection = expr.input

        sink_selects = []
        columns = set()
        to_replace = []

        windows_rewrite = False
        traversed = set()
        for path in expr.predicate.all_path(collection, strict=True):
            for node in path:
                if id(node) in traversed:
                    continue
                else:
                    traversed.add(id(node))
                if isinstance(node, SequenceReduction):
                    windows_rewrite = True

                    win = self._reduction_to_window(node)
                    window_name = '%s_%s' % (win.name, next(self._indexer))
                    sink_selects.append(win.rename(window_name))
                    to_replace.append((node, window_name))
                    break
                elif isinstance(node, Column):
                    if node.input is not collection:
                        continue
                    if node.name in columns:
                        to_replace.append((node, node.name))
                        continue
                    columns.add(node.name)
                    select_field = collection[node.source_name]
                    if node.is_renamed():
                        select_field = select_field.rename(node.name)
                    sink_selects.append(select_field)
                    to_replace.append((node, node.name))

        for column_name in expr.schema.names:
            if column_name in columns:
                continue
            columns.add(column_name)
            sink_selects.append(column_name)

        if not windows_rewrite:
            return

        get = lambda x: x.name if not isinstance(x, six.string_types) else x
        projected = collection[sorted(sink_selects, key=get)]
        projected.optimize_banned = True  # TO prevent from optimizing
        expr.substitute(collection, projected, dag=self._dag)

        for col, col_name in to_replace:
            self._sub(col, projected[col_name])

        to_sub = expr[expr.schema.names]
        self._sub(expr, to_sub)

    def _reduction_to_window(self, expr):
        clazz = 'Cum' + expr.node_name
        return globals()[clazz](_input=expr.input, _data_type=expr.dtype)

    def visit_join(self, expr):
        if expr._predicate:
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

    def _handle_groupby_applied_collection(self, expr):
        if isinstance(expr.input, JoinCollectionExpr):
            sub = JoinProjectCollectionExpr(
                _input=expr.input, _schema=expr.input.schema,
                _fields=expr.input._fetch_fields())
            self._sub(expr.input, sub)

    def visit_apply_collection(self, expr):
        if isinstance(expr, GroupbyAppliedCollectionExpr):
            self._handle_groupby_applied_collection(expr)
        self._handle_function(expr, expr._fields)

    def visit_user_defined_aggregator(self, expr):
        self._handle_function(expr, [expr.input, ])

    def visit_column(self, expr):
        if is_source_collection(expr.input):
            try:
                if expr.input._source_data.schema.is_partition(expr.source_name) and \
                                expr.dtype != types.string:
                    expr._source_data_type = types.string
            except NoSuchObject:
                return
