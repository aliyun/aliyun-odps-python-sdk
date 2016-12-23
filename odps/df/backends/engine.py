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

import time
import uuid
import itertools

try:
    import pandas as pd
except ImportError:
    pass

from .core import Engine
from .odpssql.engine import ODPSSQLEngine
from .pd.engine import PandasEngine
from .selecter import available_engines, Engines, EngineSelecter
from .formatter import ExprExecutionGraphFormatter
from .. import Scalar
from ..expr.core import ExprDAG
from ..expr.expressions import CollectionExpr, SequenceExpr
from ..expr.merge import JoinCollectionExpr, UnionCollectionExpr
from ..expr.element import IsIn
from ...models import Table
from ... import ODPS
from ... import options
from ...utils import gen_repr_object


def get_default_engine(expr):
    if expr._engine:
        return expr._engine

    srcs = list(expr.data_source())
    engines = list(available_engines(srcs))

    if len(engines) == 1:
        engine = engines[0]
        src = srcs[0]
        if engine == Engines.ODPS:
            odps = src.odps
        elif engine == Engines.PANDAS:
            if options.account is not None and \
                    options.end_point is not None and options.default_project is not None:
                odps = ODPS._from_account(options.account, options.default_project,
                                          endpoint=options.end_point,
                                          tunnel_endpoint=options.tunnel_endpoint)
            else:
                odps = None
        else:
            raise NotImplementedError
    else:
        table_src = next(it for it in srcs if isinstance(it, Table))
        odps = table_src.odps

    return MixedEngine(odps, engines)


class MixedEngine(Engine):
    def __init__(self, odps, engines=None):
        self._odps = odps
        self._engines = engines
        self._generated_table_names = []

        self._selecter = EngineSelecter()

        self._pandas_engine = PandasEngine(self._odps)
        self._odpssql_engine = ODPSSQLEngine(self._odps)

    def stop(self):
        self._pandas_engine.stop()
        self._odpssql_engine.stop()

    def _gen_table_name(self):
        table_name = self._odpssql_engine._gen_table_name()
        self._generated_table_names.append(table_name)
        return table_name

    def _get_backend(self, expr):
        return self._odpssql_engine if self._selecter.select(expr) == Engines.ODPS \
            else self._pandas_engine

    def _delegate(self, method, expr_dag, dag, expr, **kwargs):
        return getattr(self._get_backend(expr_dag.root), method)(expr_dag, dag, expr, **kwargs)

    def _cache(self, expr_dag, dag, expr, **kwargs):
        return self._delegate('_cache', expr_dag, dag, expr, **kwargs)

    def _handle_dep(self, expr_dag, dag, expr, **kwargs):
        return self._delegate('_handle_dep', expr_dag, dag, expr, **kwargs)

    def _execute(self, expr_dag, dag, expr, **kwargs):
        return self._delegate('_execute', expr_dag, dag, expr, **kwargs)

    def _persist(self, name, expr_dag, dag, expr, **kwargs):
        return self._get_backend(expr_dag.root)._persist(name, expr_dag, dag, expr, **kwargs)

    def _handle_join_or_union(self, expr_dag, dag, _, **kwargs):
        root = expr_dag.root
        if not self._selecter.has_diff_data_sources(root, no_cache=True):
            return

        to_execute = root._lhs if self._selecter.has_pandas_data_source(root._lhs) \
            else root._rhs
        table_name = self._gen_table_name()
        sub = CollectionExpr(_source_data=self._odps.get_table(table_name),
                             _schema=to_execute.schema)
        sub.add_deps(to_execute)
        expr_dag.substitute(to_execute, sub)
        return self._pandas_engine._persist(
            table_name, ExprDAG(to_execute, dag=expr_dag), dag, to_execute, **kwargs)

    def _handle_isin(self, expr_dag, dag, expr, **kwargs):
        if not self._selecter.has_diff_data_sources(expr_dag.root, no_cache=True):
            return

        seq = expr._values[0]
        expr._values = None
        execute_node = self._pandas_engine._execute(
            ExprDAG(seq, dag=expr_dag), dag, seq, **kwargs)

        def callback(res):
            vals = res[:, 0]
            expr._values = tuple(Scalar(val) for val in vals)
        execute_node.callback = callback

        return execute_node

    def _handle_function(self, expr_dag, dag, _, **kwargs):
        root = expr_dag.root

        # if expr input comes from an ODPS table
        is_root_input_from_odps = \
            self._selecter.has_odps_data_source(root.children()[0])

        for i, collection in enumerate(root._collection_resources):
            # if collection resource comes from an ODPS table
            is_source_from_odps = self._selecter.has_odps_data_source(collection)

            if is_root_input_from_odps and not is_source_from_odps:
                table_name = self._gen_table_name()
                sub = CollectionExpr(_source_data=self._odps.get_table(table_name),
                                     _schema=collection.schema)
                sub.add_deps(collection)
                expr_dag.substitute(collection, sub)
                self._pandas_engine._persist(
                    table_name, ExprDAG(collection, dag=expr_dag), dag, collection, **kwargs)
            elif not is_root_input_from_odps and is_source_from_odps:
                sub = CollectionExpr(_source_data=pd.DataFrame(),
                                     _schema=collection.schema)
                sub.add_deps(collection)
                expr_dag.substitute(collection, sub)
                execute_node = self._odpssql_engine._execute(
                    ExprDAG(collection, dag=expr_dag), dag, collection, **kwargs)

                def callback(res):
                    sub._source_data = res.values
                execute_node.callback = callback
            else:
                continue

    def _dispatch(self, expr_dag, expr, ctx):
        funcs = []

        handle = None
        if isinstance(expr, (JoinCollectionExpr, UnionCollectionExpr)) and \
                self._selecter.has_diff_data_sources(expr):
            handle = self._handle_join_or_union
        elif isinstance(expr, IsIn) and \
                self._selecter.has_diff_data_sources(expr):
            handle = self._handle_isin
        elif hasattr(expr, '_func') and expr._collection_resources and \
                self._selecter.has_diff_data_sources(expr):
            handle = self._handle_function
        if handle is not None:
            funcs.append(handle)

        h = super(MixedEngine, self)._dispatch(expr_dag, expr, ctx)
        if h is not None:
            if handle is None:
                return h
            funcs.append(h)

        if funcs:
            def f(*args, **kwargs):
                for func in funcs:
                    func(*args, **kwargs)

            return f

    def visualize(self, expr):
        dag = self.compile(expr)
        try:
            formatter = ExprExecutionGraphFormatter(dag)
            return gen_repr_object(svg=formatter._repr_svg_())
        finally:
            self._generated_table_names = []

