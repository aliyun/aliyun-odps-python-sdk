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

from ...compat import six, Enum
from ...dag import DAG
from ...utils import init_progress_ui
from ...ui.progress import create_instance_group
from ...compat import futures
from .. import utils
from ..expr.expressions import Expr, CollectionExpr, Scalar
from ..expr.core import ExprDictionary, ExprDAG
from .context import context, ExecuteContext
from .formatter import ExprExecutionGraphFormatter


class EngineTypes(Enum):
    ODPS = 'ODPS'
    PANDAS = 'PANDAS'
    SEAHAWKS = 'SEAHAWKS'
    SQLALCHEMY = 'SQLALCHEMY'


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
    def _run(self, ui):
        calls = self.topological_sort()
        results = [None] * len(calls)

        result_idx = dict()
        for i, call in enumerate(calls):
            res = call(ui=ui, progress_proportion=1.0 / len(calls))
            results[i] = res
            if call.result_index is not None:
                result_idx[call.result_index] = i

        return [results[result_idx[idx]] for idx in sorted(result_idx)]

    def _run_in_parallel(self, ui, n_parallel, async=False, timeout=None):
        submits_lock = threading.RLock()
        submits = dict()
        user_wait = dict()
        results = dict()

        calls = self.topological_sort()

        def close_ui(*_):
            with submits_lock:
                if all(call in submits and results[call] is not None for call in calls):
                    ui.close()

        executor = futures.ThreadPoolExecutor(max_workers=n_parallel)

        for call in calls:
            future = futures.Future()
            if call.result_index is not None:
                future.add_done_callback(close_ui)
            user_wait[call] = future

        for call in calls:
            def run(func):
                try:
                    prevs = self.predecessors(func)
                    if prevs:
                        fs = [user_wait[p] for p in prevs]
                        for f in fs:
                            if f.exception():
                                raise RuntimeError('Node execution failed due to failure '
                                                   'of previous node, exception: %s' % f.exception())

                    user_wait[func].set_running_or_notify_cancel()
                    res = func(ui=ui, progress_proportion=1.0 / len(calls))
                    results[func] = res
                    user_wait[func].set_result(res)
                    return res
                except:
                    e, tb = sys.exc_info()[1:]
                    if six.PY2:
                        user_wait[func].set_exception_info(e, tb)
                    else:
                        user_wait[func].set_exception(e)
                    raise
                finally:
                    with submits_lock:
                        for f in self.successors(func):
                            if f in submits:
                                continue
                            prevs = self.predecessors(f)
                            if all(p in submits and user_wait[p].done() for p in prevs):
                                submits[f] = executor.submit(run, f)

            if not self.predecessors(call):
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
        return [user_wait[c] for c in
                sorted([c for c in calls if c.result_index is not None],
                       key=lambda x: x.result_index)]

    def execute(self, ui=None, async=False, n_parallel=1, timeout=None,
                close_and_notify=True):
        ui = ui or init_progress_ui(lock=async)
        succeeded = False
        if not async:
            try:
                if n_parallel <= 1:
                    results = self._run(ui)
                else:
                    results = self._run_in_parallel(ui, n_parallel)
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
                fs = self._run_in_parallel(ui, n_parallel, async=async, timeout=timeout)
                succeeded = True
                return fs
            finally:
                if succeeded:
                    ui.notify('DataFrame execution submitted')
                else:
                    ui.notify('DataFrame execution failed to summit')

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
        # optimzie
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

        def rename(e):
            if isinstance(e, Scalar) and e.name is None:
                return e.rename('__rand_%s' % int(time.time()))
            return e

        src_exprs = [rename(it[1]) for it in expr_args_kwargs]
        exprs_dags = self._build_expr_dag([self._convert_table(e) for e in src_exprs])

        return exprs_dags, expr_args_kwargs

    def _compile_dag(self, expr_args_kwargs, exprs_dags):
        ctx = ExecuteContext()  # expr -> new_expr
        dag = ExecuteDAG()

        for idx, it, expr_dag in zip(itertools.count(0), expr_args_kwargs, exprs_dags):
            action, src_expr, args, kwargs = it

            for node in expr_dag.traverse():
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

                kw = dict(kw)
                kw.update(kwargs)
                yield act, expr, args, kw

        dag = self.compile(*transform(*exprs_args_kwargs))
        try:
            res = self._execute_dag(dag, ui=ui, async=async, n_parallel=n_parallel,
                                    timeout=timeout)
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

        def run(s, **execute_kw):
            kw = dict(kwargs)
            kw.update(kw.pop('execute_kw', dict()))
            kw.update(execute_kw)
            return engine._do_execute(expr_dag, expr, **kw)

        execute_node.run = types.MethodType(run, execute_node)
        self._add_node(execute_node, dag)

        return execute_node

    def _do_persist(self, expr_dag, expr, name, **kwargs):
        raise NotImplementedError

    def _persist(self, name, expr_dag, dag, expr, **kwargs):
        # analyze first
        analyze = kwargs.pop('analyze', True)
        if analyze:
            self._analyze(expr_dag, dag, **kwargs)

        engine = self

        execute_node = self._new_execute_node(expr_dag)

        def run(s, **execute_kw):
            kw = dict(kwargs)
            kw.update(execute_kw)
            return engine._do_persist(expr_dag, expr, name, **kw)

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

    def _create_ui_and_group(self, expr_args_kwargs, **kwargs):
        ui = kwargs.get('ui', init_progress_ui(lock=kwargs.get('async', False)))

        batch = kwargs.get('batch', False)
        if not batch:
            expr = expr_args_kwargs[0][0]
            node_name = getattr(expr, 'node_name', expr.__class__.__name__)
            group = create_instance_group('DataFrame Operation[%s]' % node_name)
        else:
            group = create_instance_group('DataFrame Operations')

        ui.add_keys(group)
        return ui, group

    def execute(self, *exprs_args_kwargs, **kwargs):
        exprs_args_kwargs, kwargs = self._handle_params(*exprs_args_kwargs, **kwargs)
        kwargs['ui'], kwargs['group'] = self._create_ui_and_group(exprs_args_kwargs, **kwargs)
        kwargs['action'] = '_execute'
        return self._action(*exprs_args_kwargs, **kwargs)

    def persist(self, *exprs_args_kwargs, **kwargs):
        exprs_args_kwargs, kwargs = self._handle_params(*exprs_args_kwargs, **kwargs)
        kwargs['ui'], kwargs['group'] = self._create_ui_and_group(exprs_args_kwargs, **kwargs)
        kwargs['action'] = '_persist'
        return self._action(*exprs_args_kwargs, **kwargs)

    def batch(self, *action_exprs_args_kwargs, **kwargs):
        args = []
        for action_expr_args_kwargs in action_exprs_args_kwargs:
            action, others = action_expr_args_kwargs[0], action_expr_args_kwargs[1:]
            action = '_%s' % action if not action.startswith('_') else action
            args.append((action, ) + tuple(others))
        return self._action(*args, **kwargs)

    def _get_cached_sub_expr(self, cached_expr, ctx=None):
        ctx = ctx or context
        data = ctx.get_cached(cached_expr)
        if isinstance(cached_expr, CollectionExpr):
            return CollectionExpr(_source_data=data,
                                  _schema=cached_expr._schema)
        else:
            assert isinstance(cached_expr, Scalar)
            return Scalar(_value=data, _value_type=cached_expr.dtype)

    def _build_expr_dag(self, exprs, on_copy=None):
        cached_exprs = ExprDictionary()

        def find_cached(_, n):
            if context.is_cached(n) and isinstance(n, (CollectionExpr, Scalar)):
                cached_exprs[n] = True

        if on_copy is not None:
            if not isinstance(on_copy, Iterable):
                on_copy = (on_copy, )
            else:
                on_copy = tuple(on_copy)
            on_copy = on_copy + (find_cached, )
        else:
            on_copy = find_cached

        res = tuple(expr.to_dag(copy=True, on_copy=on_copy) for expr in exprs)
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
    def _execute_dag(cls, dag, ui=None, async=False, n_parallel=1, timeout=None, close_and_notify=True):
        return dag.execute(ui=ui, async=async, n_parallel=n_parallel, timeout=timeout,
                           close_and_notify=close_and_notify)
