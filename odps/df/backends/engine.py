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
from .odpssql.engine import ODPSEngine
from .pd.engine import PandasEngine
from .optimize import Optimizer
from .analyzer import available_engines, Engines, Analyzer, EngineSelecter
from .context import context
from .formatter import ExprExecutionGraphFormatter
from .. import Scalar
from ..expr.core import ExprProxy
from ..expr.expressions import CollectionExpr, SequenceExpr
from ..expr.merge import JoinCollectionExpr, UnionCollectionExpr
from ..expr.element import IsIn
from ..utils import is_source_collection
from ...models import Table, Schema
from ... import ODPS
from ... import options
from ...dag import DAG
from ...utils import init_progress_ui, TEMP_TABLE_PREFIX, gen_repr_object
from ...tempobj import register_temp_table
from ...ui.progress import create_instance_group
from ...compat import OrderedDict, six


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

    return MixedEngine(odps)


class MixedEngine(Engine):
    def __init__(self, odps):
        self._odps = odps
        self._generated_table_names = []

        self._selecter = EngineSelecter()

        self._pandas_engine = PandasEngine(self._odps, global_optimize=False)
        self._odpssql_engine = ODPSEngine(self._odps, global_optimize=False)

    def _gen_table_name(self):
        table_name = '%s%s_%s' % (TEMP_TABLE_PREFIX, int(time.time()),
                                  str(uuid.uuid4()).replace('-', '_'))
        register_temp_table(self._odps, table_name)
        self._generated_table_names.append(table_name)
        return table_name

    def _sub(self, to_sub, sub, dag):
        dag.substitute(to_sub, sub)

    def _add_node(self, dag, expr, func, copied):
        nodes = dag.nodes()

        curr = (expr, func, copied)
        dag.add_node(curr)

        for node in nodes:
            for n, e in ((node[0], expr), (node[2], copied)):
                if n is None or e is None:
                    continue
                if e.is_ancestor(n):
                    dag.add_edge(node, curr)
                elif n.is_ancestor(e):
                    dag.add_edge(curr, node)

        return curr

    def _compile_join_or_union_node(self, node, src_node, node_dag, dag, callbacks):
        if not self._selecter.has_diff_data_sources(node, no_cache=True):
            return

        # Right now, we just persit DataFrame from pandas to ODPS
        to_sub, src_to_sub = next(
            (it, src_it) for it, src_it in [(node._lhs, src_node._lhs), (node._rhs, src_node._rhs)]
            if self._selecter.has_pandas_data_source(it))
        table_name = self._gen_table_name()

        sub = CollectionExpr(_source_data=self._odps.get_table(table_name),
                             _schema=to_sub.schema)

        cb = lambda: self._sub(sub, to_sub, node_dag)

        def func(**kwargs):
            if 'lifecycle' not in kwargs:
                kwargs['lifecycle'] = options.temp_lifecycle
            self._pandas_engine.persist(to_sub, table_name, src_expr=src_to_sub,
                                        dag=node_dag, **kwargs)
            callbacks.remove(cb)

        self._sub(to_sub, sub, node_dag)
        self._add_node(dag, src_to_sub, func, to_sub)
        callbacks.append(cb)

    def _compile_isin_node(self, node, src_node, node_dag, dag, callbacks):
        if not self._selecter.has_diff_data_sources(node, no_cache=True):
            return

        seq = node._values[0]
        src_seq = src_node._values[0]

        args = node.args
        def cb():
            for arg_name, arg in zip(node._args, args):
                setattr(node, arg_name, arg)

        def func(**kwargs):
            vals = list(self._pandas_engine.execute(seq, src_expr=src_seq,
                                                    dag=node_dag, **kwargs)[:, 0])
            node._values = tuple(Scalar(val) for val in vals)
            callbacks.remove(cb)

        for arg_name in node._args[1:]:
            setattr(node, arg_name, None)
        self._add_node(dag, src_seq, func, seq)
        callbacks.append(cb)

    def _compile_cache_node(self, node, src_node, node_dag, dag, callbacks):
        engines = list(available_engines(node.data_source()))
        if len(engines) == 1:
            executor = self._odpssql_engine if engines[0] == Engines.ODPS \
                else self._pandas_engine
        else:
            # we now just upload pandas data to ODPS
            executor = self._odpssql_engine

        try:
            import pandas as pd
        except ImportError:
            pass

        if executor is self._odpssql_engine:
            if isinstance(node, (CollectionExpr, SequenceExpr)):
                tmp_table_name = self._gen_table_name()
                if isinstance(node, CollectionExpr):
                    schema = node._schema
                else:
                    schema = Schema.from_lists([node.name], [node.dtype])
                sub = CollectionExpr(_source_data=self._odps.get_table(tmp_table_name),
                                     _schema=schema)
                if isinstance(node, SequenceExpr):
                    sub = sub[node.name]
            else:
                sub = Scalar(_value=None, _value_type=node.dtype)
        else:
            if isinstance(node, (CollectionExpr, SequenceExpr)):
                if isinstance(node, CollectionExpr):
                    schema = node._schema
                else:
                    schema = Schema.from_lists([node.name], [node.dtype])
                sub = CollectionExpr(_source_data=pd.DataFrame(),
                                     _schema=schema)
                if isinstance(node, SequenceExpr):
                    sub = sub[node.name]
            else:
                sub = Scalar(_value=None, _value_type=node.dtype)

        def func(**kwargs):
            kwargs['src_expr'] = src_node
            kwargs['dag'] = node_dag
            if executor is self._odpssql_engine:
                if isinstance(node, (CollectionExpr, SequenceExpr)):
                    tmp_table_name = sub._source_data.name
                    if 'lifecycle' not in kwargs:
                        kwargs['lifecycle'] = options.temp_lifecycle
                    executor.persist(node, tmp_table_name, **kwargs)
                else:
                    val = executor.execute(node, **kwargs)
                    sub._value = val
            else:
                if isinstance(node, (CollectionExpr, SequenceExpr)):
                    data = executor.execute(node, **kwargs).values
                    sub._source_data = data
                else:
                    sub._value = executor.execute(node, **kwargs)

        self._sub(node, sub, node_dag)
        self._add_node(dag, src_node, func, node)
        callbacks.append(lambda: self._sub(sub, node, node_dag))

        return sub

    def _handle_dep(self, node, dag):
        for d in node._deps:
            if isinstance(d, tuple):
                dep, dep_callback = d
            else:
                dep, dep_callback = d, None

            def func(**kwargs):
                engine = self._odpssql_engine \
                    if self._selecter.has_odps_data_source(dep) else self._pandas_engine
                result = engine.execute(dep, **kwargs)
                if dep_callback:
                    dep_callback(result, node)

            self._add_node(dag, None, func, dep)

    def _is_source_data(self, node):
        if is_source_collection(node):
            return True
        if isinstance(node, Scalar) and node._value is not None:
            return True
        return False

    def _has_cached(self, node, use_cache):
        if not use_cache:
            return False
        if self._is_source_data(node):
            return False
        if node._cache_data is not None:
            return True
        return False

    def _need_cache(self, node, use_cache):
        if not isinstance(node, (CollectionExpr, SequenceExpr, Scalar)):
            return False
        if hasattr(node, '_source_data') and node._source_data is not None:
            return False
        if node._need_cache and not use_cache:
            # we ignore the cached data, and recompute.
            return True
        return node._need_cache and node._cache_data is None

    def _compile_function(self, node, src_node, node_dag, dag, callbacks):
        # if node input comes from an ODPS table
        is_node_input_from_table = \
            isinstance(next(node.children()[0].data_source()), Table)
        for i, collection in enumerate(node._collection_resources):
            # if the collection resource comes from an ODPS table
            is_source_from_table = \
                isinstance(next(collection.data_source()), Table)
            src_collection = src_node._collection_resources[i]

            if is_node_input_from_table and not is_source_from_table:
                table_name = self._gen_table_name()

                sub = CollectionExpr(_source_data=self._odps.get_table(table_name),
                                     _schema=collection.schema)

                cb = lambda: self._sub(sub, collection, node_dag)

                def func(**kwargs):
                    self._pandas_engine.persist(collection, table_name,
                                                src_expr=src_collection,
                                                dag=node_dag, **kwargs)
                    callbacks.remove(cb)

            elif not is_node_input_from_table and is_source_from_table:
                sub = CollectionExpr(_source_data=pd.DataFrame(),
                                     _schema=collection.schema)

                cb = lambda: self._sub(sub, collection, node_dag)

                def func(**kwargs):
                    res = self._odpssql_engine.execute(
                        collection, src_expr=src_collection, dag=node_dag, **kwargs)
                    data = res.values
                    sub._source_data = data
            else:
                continue

            self._sub(collection, sub, node_dag)
            self._add_node(dag, src_collection, func, collection)
            callbacks.append(cb)

    def _handle_cache(self, expr, node_dag, use_cache, callbacks):
        if not use_cache:
            return

        if expr._cache_data is not None:
            if hasattr(expr, '_source_data') and expr._source_data is not None:
                return

            sub = None
            if isinstance(expr, CollectionExpr):
                sub = CollectionExpr(_source_data=expr._cache_data,
                                     _schema=expr._schema)
            elif isinstance(expr, Scalar):
                sub = Scalar(_value=expr._cache_data, _value_type=expr.dtype)

            if sub:
                self._sub(expr, sub, node_dag)
                cb = lambda: self._sub(sub, expr, node_dag)
                callbacks.append(cb)
                return sub

    def _analyze(self, expr_dag, execute_dag):
        def sub_has_deps(_, new_expr):
            if new_expr._deps is not None:
                self._handle_dep(new_expr, execute_dag)

        Analyzer(self._selecter, expr_dag, on_sub=sub_has_deps).analyze()
        return Optimizer(expr_dag).optimize()

    def _compile(self, expr, use_cache=None):
        use_cache = use_cache if use_cache is not None else options.df.use_cache

        src_expr = expr
        dag = DAG()
        callbacks = []

        need_cache = OrderedDict()
        to_compile = OrderedDict()
        cached = OrderedDict()

        def on_copy(src_node, node):
            if isinstance(node, (JoinCollectionExpr, UnionCollectionExpr)) \
                    and self._selecter.has_diff_data_sources(node):
                to_compile[ExprProxy(src_node)] = (node, self._compile_join_or_union_node)
            elif isinstance(node, IsIn) and self._selecter.has_diff_data_sources(node):
                to_compile[ExprProxy(src_node)] = (node, self._compile_isin_node)
            elif hasattr(node, '_func') and node._collection_resources and \
                    self._selecter.has_diff_data_sources(node):
                to_compile[ExprProxy(src_node)] = (node, self._compile_function)

            if self._need_cache(node, use_cache):
                need_cache[ExprProxy(src_node)] = (node, self._compile_cache_node)
            elif hasattr(node, '_cache_data') and node._cache_data is not None:
                cached[ExprProxy(src_node)] = (node, self._handle_cache)

        expr = context.register_to_copy_expr(src_expr, rebuilt=True, on_copy=on_copy)
        node_dag = context.build_dag(src_expr, expr, rebuilt=True)

        skip = False
        # substitute the calculated node first
        for node, func in list(six.itervalues(cached))[::-1]:
            if use_cache and node is node_dag.root:
                skip=True
            func(node, node_dag, use_cache, callbacks)
            if skip:
                break

        if not skip:
            # handle the cache node which has not been calculated
            for p, val in six.iteritems(need_cache):
                src_node = p()
                node, func = val
                func(node, src_node, node_dag, dag, callbacks)
                # we do optimize here
                op_dag = context.build_dag(None, node, dag=node_dag)
                self._analyze(op_dag, dag)

            # handle the node which comes from different data sources
            for p, val in six.iteritems(to_compile):
                src_node = p()
                node, func = val
                func(node, src_node, node_dag, dag, callbacks)

        expr = self._analyze(node_dag, dag)
        engines = list(available_engines(expr.data_source()))
        if len(engines) == 0 and isinstance(expr, Scalar):
            def func(*args, **kwargs):
                return expr._value
        else:
            assert len(engines) == 1
            executor = self._odpssql_engine if engines[0] == Engines.ODPS \
                else self._pandas_engine

            def func(method, *args, **kwargs):
                kwargs['src_expr'] = src_expr
                kwargs['dag'] = node_dag
                args = (node_dag.root, ) + args[1:]  # expr may change during running
                return getattr(executor, method)(*args, **kwargs)

        self._add_node(dag, src_expr, func, expr)

        return dag, expr, callbacks

    def _compile_dag(self, expr):
        dag, expr, callbacks = self._compile(expr)
        [cb() for cb in callbacks]

        self._generated_table_names = []
        return dag

    def compile(self, expr, use_cache=None):
        dag, expr, callbacks = self._compile(expr)
        try:
            formatter = ExprExecutionGraphFormatter(expr, dag)
            return gen_repr_object(html=formatter._to_html(),
                                   text=formatter._to_str())
        finally:
            self._generated_table_names = []
            [cb() for cb in callbacks]

    def _run(self, expr, method, *args, **kwargs):
        ui = kwargs.get('ui', init_progress_ui())
        use_cache = kwargs.pop('use_cache', None)
        src_expr = expr

        if isinstance(expr, Scalar) and expr.value is not None:
            ui.update(1)
            ui.close()
            return expr.value

        node_name = getattr(expr, 'node_name', expr.__class__.__name__)
        group = create_instance_group('DataFrame Operation[%s]' % node_name)
        ui.add_keys(group)

        dag, expr, callbacks = self._compile(expr, use_cache=use_cache)

        try:
            nodes = dag.topological_sort()
            stages_size = len(nodes)
            stage_progress = 1 / float(stages_size)

            succeeded = False
            try:
                kwargs['group'] = group

                curr = 0.0
                for _, func, _ in nodes[:-1]:
                    kw = dict(kwargs)
                    kw['finish'] = False

                    # fix, prevent the `partition` and `partitions` from passing to the caching table
                    kw.pop('partition', None)
                    kw.pop('partitions', None)

                    func(start_progress=curr, max_progress=stage_progress, ui=ui, **kw)
                    curr += stage_progress

                func = nodes[-1][1]
                kwargs['start_progress'] = curr
                kwargs['max_progress'] = stage_progress
                kwargs['src_expr'] = src_expr
                kwargs['ui'] = ui
                res = func(method, expr, *args, **kwargs)

                succeeded = True
            finally:
                ui.update(1)
                ui.close()

                notify = kwargs.get('notify', True)
                if notify:
                    if succeeded:
                        ui.notify('DataFrame execution succeeded')
                    else:
                        ui.notify('DataFrame execution failed')

            if isinstance(src_expr, Scalar):
                return res
            return res
        finally:
            [cb() for cb in callbacks]

    def execute(self, expr, **kwargs):
        return self._run(expr, 'execute', **kwargs)

    def persist(self, expr, name, **kwargs):
        return self._run(expr, 'persist', name, **kwargs)

    def visualize(self, expr):
        dag, expr, callbacks = self._compile(expr)
        try:
            formatter = ExprExecutionGraphFormatter(expr, dag)
            return gen_repr_object(svg=formatter._repr_svg_())
        finally:
            self._generated_table_names = []
            [cb() for cb in callbacks]

    def cleanup_tmp(self):
        for t in self._generated_table_names:
            self._odps.delete_table(t, if_exists=True, async=True)

