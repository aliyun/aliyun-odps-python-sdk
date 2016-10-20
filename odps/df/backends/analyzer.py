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

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

from ...models import Table
from ..expr.expressions import CollectionExpr
from ..expr.core import ExprDictionary
from .core import Backend, EngineTypes as Engines
from .pd.analyzer import Analyzer as PandasAnalyzer
from .odpssql.analyzer import Analyzer as ODPSAnalyzer
from .errors import NoBackendFound


def available_engines(sources):
    engines = set()

    for src in sources:
        if isinstance(src, Table):
            engines.add(Engines.ODPS)
        elif has_pandas and isinstance(src, pd.DataFrame):
            engines.add(Engines.PANDAS)
        else:
            raise NoBackendFound('No backend found for data source: %s' % src)

    return engines


class EngineSelecter(object):
    def __init__(self):
        self._node_engines = ExprDictionary()

    def choose_backend(self, node):
        if node in self._node_engines:
            return self._node_engines[node]

        available_engine_size = len(Engines)

        engines = set()
        children = node.children()
        if len(children) == 0:
            if not isinstance(node, CollectionExpr):
                self._node_engines[node] = None
                return
            engines.update(available_engines(node._data_source()))
        else:
            for n in children:
                e = self.choose_backend(n)
                if e and e not in engines:
                    engines.add(e)
                if len(engines) == available_engine_size:
                    break

        if len(engines) == 1:
            self._node_engines[node] = engines.pop()
        else:
            # Now if we do an operation like join
            # which comes from different data sources
            # we just use ODPS to caculate
            # in the future, we will choose according to the scale of data
            self._node_engines[node] = Engines.ODPS

        return self._node_engines[node]

    def has_diff_data_sources(self, node, no_cache=False):
        if no_cache:
            return len(available_engines(node.data_source())) > 1

        children = node.children()
        if len(children) == 0:
            return False
        return len(set(filter(lambda x: x is not None,
                              (self.choose_backend(c) for c in children)))) > 1

    def _has_data_source(self, node, src):
        children = node.children()
        if len(children) == 0:
            return src == self.choose_backend(node)
        return src in set(self.choose_backend(c) for c in node.children())

    def has_pandas_data_source(self, node):
        return self._has_data_source(node, Engines.PANDAS)

    def has_odps_data_source(self, node):
        return self._has_data_source(node, Engines.ODPS)


class Analyzer(Backend):
    """
    Analyze each node, and do two things:

    1) decide the backend for the specified expr
    2) use the backend's analyze to analyze the expr
    """

    def __init__(self, selecter, dag, traversed=None, on_sub=None):
        self._selecter = selecter
        self._expr = dag.root
        self._dag = dag
        self._traversed = traversed or set()

        self._pandas_analyzer = PandasAnalyzer(
            dag, traversed=traversed, on_sub=on_sub)
        self._odps_analyzer = ODPSAnalyzer(
            dag, traversed=traversed, on_sub=on_sub)

    def analyze(self):
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
            node.accept(self._get_analyzer(node))
        except NotImplementedError:
            return

    def _get_analyzer(self, node):
        backend = self._selecter.choose_backend(node)
        return self._pandas_analyzer if backend == Engines.PANDAS \
            else self._odps_analyzer
