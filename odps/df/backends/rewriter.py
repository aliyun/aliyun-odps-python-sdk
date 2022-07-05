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


import itertools

from .core import Backend
from ..expr.expressions import Summary
from ..expr.reduction import SequenceReduction
from ..expr.window import *  # don't remove
from ..expr.merge import *
from ..utils import traverse_until_source


class BaseRewriter(Backend):
    def __init__(self, expr_dag, traversed=None):
        self._dag = expr_dag
        self._indexer = itertools.count(0)
        self._traversed = traversed or set()

    def rewrite(self):
        for node in self._iter():
            self._traversed.add(id(node))
            self._visit_node(node)

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

    def _sub(self, expr, to_sub, parents=None):
        self._dag.substitute(expr, to_sub, parents=parents)

    def _parents(self, expr):
        return self._dag.successors(expr)

    def _rewrite_reduction_in_projection(self, expr):
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
                    if isinstance(node, SequenceReduction) and not node._need_cache:
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
            self._sub(col, projected[col_name].rename(col.name))

    def _rewrite_reduction_in_filter(self, expr):
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
