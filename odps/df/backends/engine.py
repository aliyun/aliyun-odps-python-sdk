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

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

from .core import Engine
from .odpssql.engine import ODPSEngine
from .pd.engine import PandasEngine
from .errors import NoBackendFound
from .optimize import Optimizer
from .formatter import ExprExecutionGraphFormatter
from .. import Scalar
from ..expr.expressions import CollectionExpr, SequenceExpr
from ..expr.merge import JoinCollectionExpr, UnionCollectionExpr
from ..expr.element import IsIn
from ...models import Table, Schema
from ... import ODPS
from ... import options
from ...dag import DAG
from ...utils import init_progress_ui, TEMP_TABLE_PREFIX, gen_repr_object
from ...tempobj import register_temp_table
from ...ui.progress import create_instance_group
from ...compat import Enum


class Engines(Enum):
    ODPS = 'ODPS'
    PANDAS = 'PANDAS'


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


def _build_odps_from_table(table):
    client = table._client

    account = client.account
    endpoint = client.endpoint
    project = client.project

    return ODPS(account.access_id, account.secret_access_key,
                project, endpoint=endpoint)


def get_default_engine(expr):
    if expr._engine:
        return expr._engine

    srcs = list(expr.data_source())
    engines = list(available_engines(srcs))

    if len(engines) == 1:
        engine = engines[0]
        src = srcs[0]
        if engine == Engines.ODPS:
            odps = _build_odps_from_table(src)
        elif engine == Engines.PANDAS:
            if options.access_id is not None and options.access_key is not None and \
                    options.end_point is not None and options.default_project is not None:
                odps = ODPS(options.access_id, options.access_key, options.default_project,
                         endpoint=options.end_point, tunnel_endpoint=options.tunnel_endpoint)
            else:
                odps = None
        else:
            raise NotImplementedError
    else:
        table_src = next(it for it in srcs if isinstance(it, Table))
        odps = _build_odps_from_table(table_src)

    return MixedEngine(odps)


class MixedEngine(Engine):
    def __init__(self, odps):
        self._odps = odps
        self._generated_table_names = []

        self._pandas_engine = PandasEngine(self._odps, global_optimize=False)
        self._odpssql_engine = ODPSEngine(self._odps, global_optimize=False)

    def _need_handle_sources(self, expr):
        return len(available_engines(expr.data_source())) > 1

    def _calc_pandas(self, expr):
        try:
            import pandas as pd

            return isinstance(next(expr.data_source()), pd.DataFrame)
        except ImportError:
            return False

    def _gen_table_name(self):
        table_name = '%s%s_%s' % (TEMP_TABLE_PREFIX, int(time.time()),
                                  str(uuid.uuid4()).replace('-', '_'))
        register_temp_table(self._odps, table_name)
        self._generated_table_names.append(table_name)
        return table_name

    def _sub(self, to_sub, sub):
        parents = to_sub.parents
        for parent in parents:
            parent.substitute(to_sub, sub)

    def _add_node(self, dag, expr, func, src_expr=None):
        src_expr = src_expr or expr
        nodes = dag.nodes()

        curr = (expr, func, src_expr)
        dag.add_node(curr)

        for node in nodes:
            e = node[0]
            if src_expr.is_ancestor(e):
                dag.add_edge(node, curr)

    def _compile_join_or_union_node(self, node, dag, callbacks):
        if not self._need_handle_sources(node):
            return

        # Right now, we just persit DataFrame from pandas to ODPS
        to_sub = next(it for it in (node._lhs, node._rhs) if self._calc_pandas(it))
        table_name = self._gen_table_name()

        sub = CollectionExpr(_source_data=self._odps.get_table(table_name),
                             _schema=to_sub.schema)

        cb = lambda: self._sub(sub, to_sub)

        def func(**kwargs):
            if 'lifecycle' not in kwargs:
                kwargs['lifecycle'] = options.temp_lifecycle
            self._pandas_engine.persist(to_sub, table_name, **kwargs)
            callbacks.remove(cb)

        self._sub(to_sub, sub)
        self._add_node(dag, node, func, src_expr=to_sub)
        callbacks.append(cb)

    def _compile_isin_node(self, node, dag, callbacks):
        if not self._need_handle_sources(node):
            return

        seq = node._values[0]

        def cb():
            node._cached_args = None

        def func(**kwargs):
            vals = list(self._pandas_engine.execute(seq, **kwargs)[:, 0])
            node._cached_args = tuple(
                node._cached_args[:1] + (tuple(Scalar(it) for it in vals), ))
            callbacks.remove(cb)

        node._cached_args = tuple(node._cached_args[:1] + ((None, ), ))
        self._add_node(dag, node, func)
        callbacks.append(cb)

    def _compile_cache_node(self, node, dag, callbacks):
        engines = list(available_engines(node.data_source()))
        assert len(engines) == 1
        executor = self._odpssql_engine if engines[0] == Engines.ODPS \
            else self._pandas_engine

        try:
            import pandas as pd
        except ImportError:
            pass

        if engines[0] == Engines.ODPS:
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
            if engines[0] == Engines.ODPS:
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

        self._sub(node, sub)
        self._add_node(dag, sub, func, src_expr=node)
        callbacks.append(lambda: self._sub(sub, node))

        return sub

    def _is_source_data(self, node):
        if isinstance(node, CollectionExpr) and node._source_data is not None:
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

    def _compile_function(self, node, dag, callbacks):
        if not self._need_handle_sources(node):
            return

        # if node input comes from an ODPS table
        is_node_input_from_table = \
            isinstance(next(node.children()[0].data_source()), Table)
        for collection in node._collection_resources:
            # if the collection resource comes from an ODPS table
            is_source_from_table = \
                isinstance(next(collection.data_source()), Table)

            if is_node_input_from_table and not is_source_from_table:
                table_name = self._gen_table_name()

                sub = CollectionExpr(_source_data=self._odps.get_table(table_name),
                                     _schema=collection.schema)

                cb = lambda: self._sub(sub, collection)

                def func(**kwargs):
                    self._pandas_engine.persist(collection, table_name, **kwargs)
                    callbacks.remove(cb)

            elif not is_node_input_from_table and is_source_from_table:
                sub = CollectionExpr(_source_data=pd.DataFrame(),
                                     _schema=collection.schema)

                cb = lambda: self._sub(sub, collection)

                def func(**kwargs):
                    data = collection.to_pandas(**kwargs)
                    sub._source_data = data
            else:
                continue

            self._sub(collection, sub)
            self._add_node(dag, node, func, src_expr=collection)
            callbacks.append(cb)

    def _handle_cache(self, expr, use_cache, callbacks):
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
                self._sub(expr, sub)
                cb = lambda: self._sub(sub, expr)
                callbacks.append(cb)
                return sub

    def _compile(self, expr, use_cache=None):
        use_cache = use_cache if use_cache is not None else options.df.use_cache

        dag = DAG()

        callbacks = []

        for node in expr.traverse(unique=True):
            if isinstance(node, (JoinCollectionExpr, UnionCollectionExpr)):
                self._compile_join_or_union_node(node, dag, callbacks)
            elif isinstance(node, IsIn):
                self._compile_isin_node(node, dag, callbacks)
            elif hasattr(node, '_func') and node._collection_resources:
                self._compile_function(node, dag, callbacks)
            if self._need_cache(node, use_cache):
                sub = self._compile_cache_node(node, dag, callbacks)
                if node is expr:
                    expr = sub
                    break
            elif hasattr(node, '_cache_data') and node._cache_data is not None:
                sub = self._handle_cache(node, use_cache, callbacks)
                if sub is not None and node is expr:
                    expr = sub
                    break

        engines = list(available_engines(expr.data_source()))
        assert len(engines) == 1
        executor = self._odpssql_engine if engines[0] == Engines.ODPS \
            else self._pandas_engine

        def func(method, *args, **kwargs):
            return getattr(executor, method)(*args, **kwargs)

        self._add_node(dag, expr, func)

        return dag, expr, callbacks

    def _pre_process(self, expr, use_cache=None):
        return Optimizer(expr, use_cache=use_cache).optimize()

    def _compile_dag(self, expr, use_cache=None):
        expr = self._pre_process(expr, use_cache=use_cache)

        dag, expr, callbacks = self._compile(expr)
        [cb() for cb in callbacks]

        self._generated_table_names = []
        return dag

    def compile(self, expr, use_cache=None):
        expr = self._pre_process(expr, use_cache=use_cache)

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
        group = create_instance_group('DataFrame Operation[%s] execution details' % node_name)
        ui.add_keys(group)

        expr = self._pre_process(expr, use_cache=use_cache)
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
                    func(start_progress=curr, max_progress=stage_progress, ui=ui, **kwargs)
                    curr += stage_progress

                func = nodes[-1][1]
                kwargs['start_progress'] = curr
                kwargs['max_progress'] = stage_progress
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
        expr = self._pre_process(expr)

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

