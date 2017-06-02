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

from collections import Iterable
import itertools
import time
import types
import sys
import threading
from operator import itemgetter

from ...compat import six, Enum
from ...models import Resource
from ...config import options
from ...dag import DAG
from ...utils import init_progress_ui
from ...ui.progress import create_instance_group
from ...compat import futures
from .. import utils
from ..expr.expressions import Expr, CollectionExpr, Scalar
from ..expr.core import ExprDictionary, ExprDAG
from .context import context, ExecuteContext
from .errors import DagDependencyError, CompileError
from .formatter import ExprExecutionGraphFormatter


class EngineTypes(Enum):
    ODPS = 'ODPS'
    PANDAS = 'PANDAS'
    SEAHAWKS = 'SEAHAWKS'
    SQLALCHEMY = 'SQLALCHEMY'
    ALGO = 'ALGO'


class Backend(object):

    def visit_source_collection(self, expr):
        raise NotImplementedError

    def visit_project_collection(self, expr):
        raise NotImplementedError

    def visit_apply_collection(self, expr):
        raise NotImplementedError

    def visit_filter_collection(self, expr):
        raise NotImplementedError

    def visit_filter_partition_collection(self, expr):
        raise NotImplementedError

    def visit_algo(self, expr):
        raise NotImplementedError

    def visit_slice_collection(self, expr):
        raise NotImplementedError

    def visit_element_op(self, expr):
        raise NotImplementedError

    def visit_binary_op(self, expr):
        raise NotImplementedError

    def visit_unary_op(self, expr):
        raise NotImplementedError

    def visit_math(self, expr):
        raise NotImplementedError

    def visit_string_op(self, expr):
        raise NotImplementedError

    def visit_datetime_op(self, expr):
        raise NotImplementedError

    def visit_groupby(self, expr):
        raise NotImplementedError

    def visit_mutate(self, expr):
        raise NotImplementedError

    def visit_reshuffle(self, expr):
        raise NotImplementedError

    def visit_value_counts(self, expr):
        raise NotImplementedError

    def visit_sort(self, expr):
        raise NotImplementedError

    def visit_sort_column(self, expr):
        raise NotImplementedError

    def visit_distinct(self, expr):
        raise NotImplementedError

    def visit_sample(self, expr):
        raise NotImplementedError

    def visit_pivot(self, expr):
        raise NotImplementedError

    def visit_reduction(self, expr):
        raise NotImplementedError

    def visit_user_defined_aggregator(self, expr):
        raise NotImplementedError

    def visit_column(self, expr):
        raise NotImplementedError

    def visit_function(self, expr):
        raise NotImplementedError

    def visit_builtin_function(self, expr):
        raise NotImplementedError

    def visit_sequence(self, expr):
        raise NotImplementedError

    def visit_cum_window(self, expr):
        raise NotImplementedError

    def visit_rank_window(self, expr):
        raise NotImplementedError

    def visit_shift_window(self, expr):
        raise NotImplementedError

    def visit_scalar(self, expr):
        raise NotImplementedError

    def visit_join(self, expr):
        raise NotImplementedError

    def visit_cast(self, expr):
        raise NotImplementedError

    def visit_union(self, expr):
        raise NotImplementedError

    def visit_concat(self, expr):
        raise NotImplementedError

    def visit_append_id(self, expr):
        raise NotImplementedError

    def visit_split(self, expr):
        raise NotImplementedError

    def visit_extract_kv(self, expr):
        raise NotImplementedError


class ExecuteNode(object):
    def __init__(self, expr_dag, result_index=None, callback=None):
        self.expr_dag = expr_dag
        self.result_index = result_index
        self.callback = callback

    @property
    def expr(self):
        return self.expr_dag.root

    def run(self, **execute_kw):
        raise NotImplementedError

    def __call__(self, ui=None, progress_proportion=None):
        res = self.run(ui=ui, progress_proportion=progress_proportion)
        if self.callback:
            self.callback(res)
        return res

    def __repr__(self):
        raise NotImplementedError

    def _repr_html_(self):
        raise NotImplementedError


class ExecuteDAG(DAG):
    def _run(self, ui, progress_proportion=1.0):
        curr_progress = ui.current_progress() or 0
        try:
            calls = self.topological_sort()
            results = [None] * len(calls)

            result_idx = dict()
            for i, call in enumerate(calls):
                res = call(ui=ui, progress_proportion=progress_proportion / len(calls))
                results[i] = res
                if call.result_index is not None:
                    result_idx[call.result_index] = i

            return [results[result_idx[idx]] for idx in sorted(result_idx)]
        except Exception as e:
            if self._can_fallback() and self._need_fallback(e):
                ui.update(curr_progress)
                return self.fallback()._run(ui, progress_proportion)

            raise

    def _run_in_parallel(self, ui, n_parallel, async=False, timeout=None, progress_proportion=1.0):
        submits_lock = threading.RLock()
        submits = dict()
        user_wait = dict()
        result_wait = dict()
        results = dict()

        curr_progress = ui.current_progress() or 0

        def actual_run(dag=None, is_fallback=False):
            dag = dag or self
            calls = dag.topological_sort()
            result_calls = sorted([c for c in calls if c.result_index is not None],
                                  key=lambda x: x.result_index)
            fallback = threading.Event()

            if is_fallback:
                ui.update(curr_progress)

            def close_ui(*_):
                with submits_lock:
                    if all(call in submits and call in results for call in result_calls):
                        ui.close()

            executor = futures.ThreadPoolExecutor(max_workers=n_parallel)

            for call in calls:
                if call.result_index is not None and is_fallback:
                    # if is fallback, we do not create new future
                    # cause the future objects have been passed to user
                    future = result_wait[call.result_index]
                else:
                    future = futures.Future()
                user_wait[call] = future
                if call.result_index is not None:
                    future.add_done_callback(close_ui)
                    if not is_fallback:
                        result_wait[call.result_index] = future

            for call in calls:
                def run(func):
                    try:
                        if fallback.is_set():
                            raise DagDependencyError('Node execution failed due to callback')

                        if call.result_index is None or not is_fallback:
                            user_wait[func].set_running_or_notify_cancel()

                        prevs = dag.predecessors(func)
                        if prevs:
                            fs = [user_wait[p] for p in prevs]
                            for f in fs:
                                if f.exception():
                                    raise DagDependencyError('Node execution failed due to failure of '
                                                             'previous node, exception: %s' % f.exception())

                        res = func(ui=ui, progress_proportion=progress_proportion / len(calls))
                        results[func] = res
                        user_wait[func].set_result(res)
                        return res
                    except:
                        e, tb = sys.exc_info()[1:]
                        if not is_fallback and self._can_fallback() and self._need_fallback(e):
                            if not fallback.is_set():
                                fallback.set()
                                new_dag = dag.fallback()
                                actual_run(new_dag, True)
                        if not fallback.is_set():
                            results[func] = (e, tb)
                            if six.PY2:
                                user_wait[func].set_exception_info(e, tb)
                            else:
                                user_wait[func].set_exception(e)
                        raise
                    finally:
                        with submits_lock:
                            for f in dag.successors(func):
                                if f in submits:
                                    continue
                                prevs = dag.predecessors(f)
                                if all(p in submits and user_wait[p].done() for p in prevs):
                                    submits[f] = executor.submit(run, f)

                if not dag.predecessors(call):
                    with submits_lock:
                        submits[call] = executor.submit(run, call)

            if not async:
                dones, _ = futures.wait(user_wait.values())
                for done in dones:
                    done.result()
                return [results[c] for c in
                        sorted([c for c in calls if c.result_index is not None],
                               key=lambda x: x.result_index)]

            if timeout:
                futures.wait(user_wait.values(), timeout=timeout)

        actual_run()
        if not async:
            return [it[1].result() for it in sorted(result_wait.items(), key=itemgetter(0))]
        else:
            return [it[1] for it in sorted(result_wait.items(), key=itemgetter(0))]

    def execute(self, ui=None, async=False, n_parallel=1, timeout=None,
                close_and_notify=True, progress_proportion=1.0):
        ui = ui or init_progress_ui(lock=async)
        succeeded = False
        if not async:
            try:
                if n_parallel <= 1:
                    results = self._run(ui, progress_proportion)
                else:
                    results = self._run_in_parallel(ui, n_parallel, progress_proportion=progress_proportion)
                succeeded = True
                return results
            finally:
                if close_and_notify or succeeded:
                    ui.close()

                    if succeeded:
                        ui.notify('DataFrame execution succeeded')
                    else:
                        ui.notify('DataFrame execution failed')
        else:
            try:
                fs = self._run_in_parallel(ui, n_parallel, async=async, timeout=timeout,
                                           progress_proportion=progress_proportion)
                succeeded = True
                return fs
            finally:
                if succeeded:
                    ui.notify('DataFrame execution submitted')
                else:
                    ui.notify('DataFrame execution failed to summit')

    def _can_fallback(self):
        return hasattr(self, 'fallback') and self.fallback is not None

    def _need_fallback(self, e):
        return hasattr(self, 'need_fallback') and self.need_fallback(e)

    def __repr__(self):
        return ExprExecutionGraphFormatter(self)._to_str()

    def _repr_html_(self):
        return ExprExecutionGraphFormatter(self)._to_html()


class Engine(object):
    def stop(self):
        pass

    @classmethod
    def _convert_table(cls, expr):
        if isinstance(expr, Expr):
            return utils.to_collection(expr)

        expr_dag = expr
        root = utils.to_collection(expr_dag.root)
        if root is not expr_dag.root:
            new_expr_dag = ExprDAG(root, dag=expr_dag)
            new_expr_dag.ensure_all_nodes_in_dag()
            return new_expr_dag

        return expr_dag

    def _cache(self, expr_dag, dag, expr, **kwargs):
        # should return the data
        raise NotImplementedError

    def _dispatch(self, expr_dag, expr, ctx):
        if expr._need_cache:
            if not ctx.is_cached(expr):
                def h():
                    def inner(*args, **kwargs):
                        ret = self._cache(*args, **kwargs)
                        if ret:
                            data, node = ret
                            ctx.cache(expr, data)
                            return node
                    return inner
                return h()
            else:
                cached = ctx.get_cached(expr)
                if isinstance(expr, CollectionExpr):
                    cached = cached.copy()
                expr_dag.substitute(expr, cached)
        elif expr._deps:
            return self._handle_dep

    def _new_analyzer(self, expr_dag, on_sub=None):
        raise NotImplementedError

    def _new_rewriter(self, expr_dag):
        return

    def _analyze(self, expr_dag, dag, **kwargs):
        from .optimize import Optimizer

        def sub_has_dep(_, sub):
            if sub._deps is not None:
                kw = dict(kwargs)
                kw['finish'] = False
                kw.pop('head', None)
                kw.pop('tail', None)
                self._handle_dep(ExprDAG(sub, dag=expr_dag), dag, sub, **kw)

        # analyze first
        self._new_analyzer(expr_dag, on_sub=sub_has_dep).analyze()
        # optimize
        return Optimizer(expr_dag).optimize()

    def _rewrite(self, expr_dag):
        # rewrite if exist
        rewriter = self._new_rewriter(expr_dag)
        if rewriter:
            return rewriter.rewrite()
        return expr_dag.root

    def _new_execute_node(self, expr_dag):
        return ExecuteNode(expr_dag)

    def _handle_dep(self, expr_dag, dag, expr, **kwargs):
        root = expr_dag.root

        execute_nodes = []
        for dep in root._deps:
            if isinstance(dep, tuple):
                if len(dep) == 3:
                    node, action, callback = dep
                else:
                    node, callback = dep
                    action = '_execute'
            else:
                node, action, callback = dep, '_execute', None

            if callback:
                def dep_callback(res):
                    callback(res, expr)
            else:
                dep_callback = None

            execute_node = getattr(self, action)(ExprDAG(node, dag=expr_dag), dag, node,
                                                 analyze=False, **kwargs)
            execute_node.callback = dep_callback
            execute_nodes.append(execute_node)

        return execute_nodes

    def _handle_expr_args_kwargs(self, expr_args_kwargs):
        if len(expr_args_kwargs) == 1 and not isinstance(expr_args_kwargs[0], Expr) and \
                all(isinstance(it, Expr) for it in expr_args_kwargs[0]):
            expr_args_kwargs = expr_args_kwargs[0]
        if all(isinstance(it, Expr) for it in expr_args_kwargs):
            expr_args_kwargs = [('_execute', it, (), {}) for it in expr_args_kwargs]

        return expr_args_kwargs

    def _process(self, *expr_args_kwargs):
        expr_args_kwargs = self._handle_expr_args_kwargs(expr_args_kwargs)

        def h(e):
            if isinstance(e, Scalar) and e.name is None:
                return e.rename('__rand_%s' % int(time.time()))
            if isinstance(e, CollectionExpr) and hasattr(e, '_proxy') and \
                    e._proxy is not None:
                return e._proxy
            return e

        src_exprs = [h(it[1]) for it in expr_args_kwargs]
        exprs_dags = self._build_expr_dag([self._convert_table(e) for e in src_exprs])

        return exprs_dags, expr_args_kwargs

    def _compile_dag(self, expr_args_kwargs, exprs_dags):
        ctx = ExecuteContext()  # expr -> new_expr
        dag = ExecuteDAG()

        for idx, it, expr_dag in zip(itertools.count(0), expr_args_kwargs, exprs_dags):
            action, src_expr, args, kwargs = it

            for node in expr_dag.traverse():
                if hasattr(self, '_selecter') and not self._selecter.force_odps and hasattr(node, '_algo'):
                    raise NotImplementedError
                h = self._dispatch(expr_dag, node, ctx)
                if h:
                    kw = dict(kwargs)
                    kw['finish'] = False
                    if node is expr_dag.root:
                        node_dag = expr_dag
                    else:
                        node_dag = ExprDAG(node, dag=expr_dag)
                    h(node_dag, dag, node, **kw)

            args = args + (expr_dag, dag, src_expr)
            n = getattr(self, action)(*args, **kwargs)
            n.result_index = idx

        return dag

    def compile(self, *expr_args_kwargs):
        exprs_dags, expr_args_kwargs = self._process(*expr_args_kwargs)
        return self._compile_dag(expr_args_kwargs, exprs_dags)

    def _action(self, *exprs_args_kwargs, **kwargs):
        ui = kwargs.pop('ui', None)
        progress_proportion = kwargs.pop('progress_proportion', 1.0)
        async = kwargs.pop('async', False)
        n_parallel = kwargs.pop('n_parallel', 1)
        timeout = kwargs.pop('timeout', None)
        batch = kwargs.pop('batch', False)
        action = kwargs.pop('action', None)

        def transform(*exprs_args_kw):
            for expr_args_kwargs in exprs_args_kw:
                if len(expr_args_kwargs) == 3:
                    expr, args, kw = expr_args_kwargs
                    act = action
                else:
                    act, expr, args, kw = expr_args_kwargs

                kw = kw.copy()
                kw.update(kwargs)
                yield act, expr, args, kw

        dag = self.compile(*transform(*exprs_args_kwargs))
        try:
            res = self._execute_dag(dag, ui=ui, async=async, n_parallel=n_parallel,
                                    timeout=timeout, progress_proportion=progress_proportion)
        except KeyboardInterrupt:
            self.stop()
            sys.exit(1)

        if not batch:
            return res[0]
        return res

    def _do_execute(self, expr_dag, expr, **kwargs):
        raise NotImplementedError

    def _execute(self, expr_dag, dag, expr, **kwargs):
        # analyze first
        analyze = kwargs.pop('analyze', True)
        if analyze:
            kw = dict(kwargs)
            kw.pop('execute_kw', None)
            self._analyze(expr_dag, dag, **kw)

        engine = self

        execute_node = self._new_execute_node(expr_dag)
        group_key = kwargs.get('group') or self._create_progress_group(expr)

        def run(s, **execute_kw):
            kw = dict(kwargs)
            kw.update(kw.pop('execute_kw', dict()))
            kw.update(execute_kw)
            kw['group'] = group_key

            if 'ui' in kw:
                kw['ui'].add_keys(group_key)
            result = engine._do_execute(expr_dag, expr, **kw)
            if 'ui' in kw:
                kw['ui'].remove_keys(group_key)

            return result

        execute_node.run = types.MethodType(run, execute_node)
        self._add_node(execute_node, dag)

        return execute_node

    @classmethod
    def _reorder(cls, expr, table, cast=False, with_partitions=False):
        from .odpssql.engine import types as odps_engine_types
        from .. import NullScalar

        df_schema = odps_engine_types.odps_schema_to_df_schema(table.schema)
        expr_schema = expr.schema.to_ignorecase_schema()
        expr_table_schema = odps_engine_types.df_schema_to_odps_schema(expr_schema)
        case_dict = dict((n.lower(), n) for n in expr.schema.names)

        for col in expr_table_schema.columns:
            if col.name.lower() not in expr_table_schema:
                raise CompileError('Column %s does not exist in table' % col.name)
            t_col = table.schema[col.name.lower()]
            if not cast and not t_col.type.can_implicit_cast(col.type):
                raise CompileError('Cannot implicitly cast column %s from %s to %s.' % (
                    col.name, col.type, t_col.type))

        if table.schema.names == expr_schema.names and \
                        df_schema.types[:len(table.schema.names)] == expr_schema.types:
            return expr

        def field(name):
            expr_name = case_dict[name]
            if expr[expr_name].dtype == df_schema[name].type:
                return expr[expr_name]
            elif df_schema[name].type.can_implicit_cast(expr[expr_name].dtype) or cast:
                return expr[expr_name].astype(df_schema[name].type)
            else:
                raise CompileError('Column %s\'s type does not match, expect %s, got %s' % (
                    expr_name, expr[expr_name].dtype, df_schema[name].type))
        names = [c.name for c in table.schema.columns] if with_partitions else table.schema.names
        return expr[[field(name) if name in expr_schema else NullScalar(df_schema[name].type).rename(name)
                     for name in names]]

    def _do_persist(self, expr_dag, expr, name, **kwargs):
        raise NotImplementedError

    def _persist(self, name, expr_dag, dag, expr, **kwargs):
        # analyze first
        analyze = kwargs.pop('analyze', True)
        if analyze:
            self._analyze(expr_dag, dag, **kwargs)

        engine = self

        execute_node = self._new_execute_node(expr_dag)
        group_key = self._create_progress_group(expr)

        def run(s, **execute_kw):
            kw = dict(kwargs)
            kw.update(execute_kw)
            kw['group'] = group_key

            if 'ui' in kw:
                kw['ui'].add_keys(group_key)
            result = engine._do_persist(expr_dag, expr, name, **kw)
            if 'ui' in kw:
                kw['ui'].remove_keys(group_key)

            return result

        execute_node.run = types.MethodType(run, execute_node)
        self._add_node(execute_node, dag)

        return execute_node

    @classmethod
    def _handle_params(cls, *expr_args_kwargs, **kwargs):
        if isinstance(expr_args_kwargs[0], Expr):
            return [(expr_args_kwargs[0], expr_args_kwargs[1:], {})], kwargs
        elif isinstance(expr_args_kwargs[0], Iterable) and \
                all(isinstance(e, Expr) for e in expr_args_kwargs[0]):
            args = expr_args_kwargs[1:]
            kwargs['batch'] = True
            return [(e, args, {}) for e in expr_args_kwargs[0]], kwargs
        else:
            kwargs['batch'] = True
            return expr_args_kwargs, kwargs

    @staticmethod
    def _create_ui(**kwargs):
        existing_ui = kwargs.get('ui')
        if existing_ui:
            return existing_ui

        ui = init_progress_ui(lock=kwargs.get('async', False), use_console=not kwargs.get('async', False))
        ui.status('Preparing')
        return ui

    @staticmethod
    def _create_progress_group(expr):
        node_name = getattr(expr, 'node_name', expr.__class__.__name__)
        return create_instance_group(node_name)

    def execute(self, *exprs_args_kwargs, **kwargs):
        exprs_args_kwargs, kwargs = self._handle_params(*exprs_args_kwargs, **kwargs)
        kwargs['ui'] = self._create_ui(**kwargs)
        kwargs['action'] = '_execute'
        return self._action(*exprs_args_kwargs, **kwargs)

    def persist(self, *exprs_args_kwargs, **kwargs):
        exprs_args_kwargs, kwargs = self._handle_params(*exprs_args_kwargs, **kwargs)
        kwargs['ui'] = self._create_ui(**kwargs)
        kwargs['action'] = '_persist'
        return self._action(*exprs_args_kwargs, **kwargs)

    def batch(self, *action_exprs_args_kwargs, **kwargs):
        args = []
        for action_expr_args_kwargs in action_exprs_args_kwargs:
            action, others = action_expr_args_kwargs[0], action_expr_args_kwargs[1:]
            action = '_%s' % action if not action.startswith('_') else action
            args.append((action, ) + tuple(others))
        kwargs = kwargs.copy()
        kwargs['batch'] = True
        return self._action(*args, **kwargs)

    def _get_cached_sub_expr(self, cached_expr, ctx=None):
        ctx = ctx or context
        data = ctx.get_cached(cached_expr)
        return cached_expr.get_cached(data)

    def _build_expr_dag(self, exprs, on_copy=None):
        cached_exprs = ExprDictionary()

        def find_cached(_, n):
            if context.is_cached(n) and hasattr(n, 'get_cached'):
                cached_exprs[n] = True

        if on_copy is not None:
            if not isinstance(on_copy, Iterable):
                on_copy = (on_copy, )
            else:
                on_copy = tuple(on_copy)
            on_copy = on_copy + (find_cached, )
        else:
            on_copy = find_cached

        res = tuple(expr.to_dag(copy=True, on_copy=on_copy, validate=False)
                    for expr in exprs)
        for cached in cached_exprs:
            sub = self._get_cached_sub_expr(cached)
            if sub is not None:
                for dag in res:
                    if dag.contains_node(cached):
                        dag.substitute(cached, sub)

        return res

    def _add_node(self, dag_node, dag):
        nodes = dag.nodes()
        dag.add_node(dag_node)
        for node in nodes:
            node_expr = node.expr
            if dag_node.expr.is_ancestor(node_expr):
                dag.add_edge(node, dag_node)
            elif node_expr.is_ancestor(dag_node.expr):
                dag.add_edge(dag_node, node)

    @classmethod
    def _execute_dag(cls, dag, ui=None, async=False, n_parallel=1, timeout=None, close_and_notify=True,
                     progress_proportion=1.0):
        return dag.execute(ui=ui, async=async, n_parallel=n_parallel, timeout=timeout,
                           close_and_notify=close_and_notify, progress_proportion=progress_proportion)

    def _get_libraries(self, libraries):
        def conv(libs):
            if isinstance(libs, (six.binary_type, six.text_type, Resource)):
                return conv([libs, ])
            if libs is None:
                return None
            new_libs = []
            for lib in libs:
                if isinstance(lib, Resource):
                    new_libs.append(lib)
                else:
                    new_libs.append(self._odps.get_resource(lib))
            return new_libs

        libraries = conv(libraries) or []
        if options.df.libraries is not None:
            libraries.extend(conv(options.df.libraries))
        if len(libraries) == 0:
            return
        return list(set(libraries))
