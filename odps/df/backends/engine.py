#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from __future__ import absolute_import

try:
    import pandas as pd
except ImportError:
    pass

from .core import Engine
from .odpssql.engine import ODPSSQLEngine
from .pd.engine import PandasEngine
from .seahawks.engine import SeahawksEngine
from .seahawks.models import SeahawksTable
from .sqlalchemy.engine import SQLAlchemyEngine
from .selecter import available_engines, Engines, EngineSelecter
from .formatter import ExprExecutionGraphFormatter
from .context import context
from .utils import process_persist_kwargs
from .. import Scalar
from ..expr.core import ExprDAG
from ..expr.expressions import CollectionExpr
from ..expr.merge import JoinCollectionExpr, UnionCollectionExpr
from ..expr.element import IsIn
from ...models import Table
from ... import ODPS
from ... import options
from ...utils import gen_repr_object


def get_default_engine(*exprs):
    odps = None
    engines = set()

    for expr in exprs:
        if expr._engine:
            return expr._engine

        srcs = list(expr.data_source())
        expr_engines = list(available_engines(srcs))
        engines.update(set(expr_engines))

        if len(expr_engines) == 1:
            engine = expr_engines[0]
            src = srcs[0]
            if engine in (Engines.ODPS, Engines.ALGO):
                expr_odps = src.odps
            elif engine in (Engines.PANDAS, Engines.SQLALCHEMY):
                expr_odps = None
            else:
                raise NotImplementedError
        else:
            table_src = next(it for it in srcs if hasattr(it, 'odps'))
            expr_odps = table_src.odps
        if expr_odps is not None:
            odps = expr_odps

    if odps is None and options.account is not None and \
                    options.end_point is not None and options.default_project is not None:
        odps = ODPS._from_account(options.account, options.default_project,
                                  endpoint=options.end_point,
                                  tunnel_endpoint=options.tunnel.endpoint)

    return MixedEngine(odps, list(engines))


class MixedEngine(Engine):
    def __init__(self, odps, engines=None):
        self._odps = odps
        self._engines = engines
        self._generated_table_names = []

        self._selecter = EngineSelecter()

        self._pandas_engine = PandasEngine(self._odps)
        self._odpssql_engine = ODPSSQLEngine(self._odps)
        self._seahawks_engine = SeahawksEngine(self._odps)
        self._sqlalchemy_engine = SQLAlchemyEngine(self._odps)

        from ...ml.engine import OdpsAlgoEngine
        self._xflow_engine = OdpsAlgoEngine(self._odps)

    def stop(self):
        self._pandas_engine.stop()
        self._odpssql_engine.stop()
        self._seahawks_engine.stop()
        self._xflow_engine.stop()

    def _gen_table_name(self):
        table_name = self._odpssql_engine._gen_table_name()
        self._generated_table_names.append(table_name)
        return table_name

    def _get_backend(self, expr_dag):
        engine = self._selecter.select(expr_dag)
        if engine == Engines.ODPS:
            return self._odpssql_engine
        elif engine == Engines.PANDAS:
            return self._pandas_engine
        elif engine == Engines.SQLALCHEMY:
            return self._sqlalchemy_engine
        elif engine == Engines.ALGO:
            return self._xflow_engine
        else:
            assert engine == Engines.SEAHAWKS
            return self._seahawks_engine

    def _delegate(self, method, expr_dag, dag, expr, **kwargs):
        return getattr(self._get_backend(expr_dag), method)(expr_dag, dag, expr, **kwargs)

    def _cache(self, expr_dag, dag, expr, **kwargs):
        return self._delegate('_cache', expr_dag, dag, expr, **kwargs)

    def _handle_dep(self, expr_dag, dag, expr, **kwargs):
        return self._delegate('_handle_dep', expr_dag, dag, expr, **kwargs)

    def _execute(self, expr_dag, dag, expr, **kwargs):
        return self._delegate('_execute', expr_dag, dag, expr, **kwargs)

    def _persist(self, name, expr_dag, dag, expr, **kwargs):
        return self._get_backend(expr_dag)._persist(name, expr_dag, dag, expr, **kwargs)

    def _handle_join_or_union(self, expr_dag, dag, _, **kwargs):
        root = expr_dag.root
        if not self._selecter.has_diff_data_sources(root, no_cache=True):
            return

        to_execute = root._lhs if not self._selecter.has_odps_data_source(root._lhs) \
            else root._rhs
        table_name = self._gen_table_name()
        sub = CollectionExpr(_source_data=self._odps.get_table(table_name),
                             _schema=to_execute.schema)
        sub.add_deps(to_execute)
        expr_dag.substitute(to_execute, sub)

        # prevent the kwargs come from `persist`
        process_persist_kwargs(kwargs)

        execute_dag = ExprDAG(to_execute, dag=expr_dag)
        return self._get_backend(execute_dag)._persist(
            table_name, execute_dag, dag, to_execute, **kwargs)

    def _handle_isin(self, expr_dag, dag, expr, **kwargs):
        if not self._selecter.has_diff_data_sources(expr_dag.root, no_cache=True):
            return

        seq = expr._values[0]
        expr._values = None
        execute_dag = ExprDAG(seq, dag=expr_dag)
        execute_node = self._get_backend(execute_dag)._execute(
            execute_dag, dag, seq, **kwargs)

        def callback(res):
            vals = res[:, 0].tolist()
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

                # prevent the kwargs come from `persist`
                process_persist_kwargs(kwargs)

                execute_dag = ExprDAG(collection, dag=expr_dag)
                self._get_backend(execute_dag)._persist(
                    table_name, execute_dag, dag, collection, **kwargs)
            elif not is_root_input_from_odps and is_source_from_odps:
                if not self._selecter.has_pandas_data_source(root.children()[0]):
                    raise NotImplementedError
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
        from ...ml.expr import AlgoExprMixin
        funcs = []

        if isinstance(expr, AlgoExprMixin):
            return self._xflow_engine._dispatch(expr_dag, expr, ctx)

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

    def _get_cached_sub_expr(self, cached_expr, ctx=None):
        ctx = ctx or context
        data = ctx.get_cached(cached_expr)
        if self._selecter.force_odps and isinstance(data, SeahawksTable):
            # skip seahawks heap table
            return
        return super(MixedEngine, self)._get_cached_sub_expr(cached_expr, ctx=ctx)

    def visualize(self, expr):
        dag = self.compile(expr)
        try:
            formatter = ExprExecutionGraphFormatter(dag)
            return gen_repr_object(svg=formatter._repr_svg_())
        finally:
            self._generated_table_names = []

    def compile(self, *expr_args_kwargs):
        src_expr_args_kwargs = tuple(expr_args_kwargs)
        exprs_dags, expr_args_kwargs = self._process(*expr_args_kwargs)

        if not self._selecter.force_odps and any(it[0] == '_persist' for it in expr_args_kwargs):
            self._selecter.force_odps = True
            return self.compile(*src_expr_args_kwargs)

        engine_types = [self._selecter.select(expr_dag) for expr_dag in exprs_dags]
        if any(engine_type == Engines.ODPS for engine_type in engine_types):
            self._selecter.force_odps = True
            if any(engine_type == Engines.SEAHAWKS for engine_type in engine_types):
                return self.compile(*src_expr_args_kwargs)

        try:
            dag = super(MixedEngine, self)._compile_dag(expr_args_kwargs, exprs_dags)
        except NotImplementedError:
            self._selecter.force_odps = True
            return self.compile(*src_expr_args_kwargs)

        if any(engine_type == Engines.SEAHAWKS for engine_type in engine_types):
            def fallback():
                self._selecter.force_odps = True
                return self.compile(*src_expr_args_kwargs)

            def need_fallback(e):
                try:
                    import sqlalchemy
                    exceptions = (NotImplementedError, sqlalchemy.exc.DatabaseError)
                except ImportError:
                    exceptions = (NotImplementedError,)

                if not isinstance(e, exceptions):
                    return False
                if not isinstance(e, NotImplementedError) and \
                        'AXF Exception' not in str(e):
                    # the seahawks error
                    return False
                return True

            if not all(not hasattr(n, 'verify') or n.verify()
                       for n in dag.indep_nodes()):
                # we only verify the independent nodes
                dag = fallback()

            dag.fallback = fallback
            dag.need_fallback = need_fallback
        return dag
