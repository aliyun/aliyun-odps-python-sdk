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

from collections import namedtuple

from ..core import Backend
from ...expr.expressions import *
from ...expr.reduction import GroupedSequenceReduction, GroupedCount, Count
from ...expr.merge import JoinCollectionExpr
from ...expr.groupby import GroupBy
from ...expr.datetimes import DTScalar
from ...expr import element
from ...expr import arithmetic
from ....dag import DAG
from ..errors import CompileError
from . import types
from ... import types as df_types

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pass


BINARY_OP_TO_PANDAS = {
    'Add': operator.add,
    'Substract': operator.sub,
    'Multiply': operator.mul,
    'Divide': operator.div if six.PY2 else operator.truediv,
    'FloorDivide': operator.floordiv,
    'Power': operator.pow,
    'Greater': operator.gt,
    'GreaterEqual': operator.ge,
    'Less': operator.lt,
    'LessEqual': operator.le,
    'Equal': operator.eq,
    'NotEqual': operator.ne,
    'And': operator.and_,
    'Or': operator.or_
}


UNARY_OP_TO_PANDAS = {
    'Negate': operator.neg,
    'Invert': operator.inv,
    'Abs': operator.abs
}


class PandasCompiler(Backend):
    """
    PandasCompiler will compile an Expr into a DAG
    in which each node is a pair of <expr, function>.
    """

    def __init__(self):
        self._dag = DAG()
        self._expr_to_dag_node = dict()
        self._callbacks = list()

    def compile(self, expr):
        try:
            return self._compile(expr)
        finally:
            self._cleanup()

    def _cleanup(self):
        for callback in self._callbacks:
            callback()
        self._callbacks = list()

    def _compile(self, expr, traversed=None):
        if traversed is None:
            traversed = set()

        root = self._retrieve_until_find_root(expr)

        if root is not None:
            self._compile_join_node(root, traversed)
            traversed.add(id(root))

        for node in expr.traverse():
            if id(node) not in traversed:
                node.accept(self)
                traversed.add(id(node))

        return self._dag

    def _compile_join_node(self, expr, traversed):
        nodes = []

        self._compile(expr._lhs)
        nodes.append(expr._lhs)
        self._compile(expr._rhs)
        nodes.append(expr._rhs)
        for node in self._find_all_equalizations(expr._predicate, expr._lhs, expr._rhs):
            nodes.append(node._lhs)
            self._compile(node._lhs, traversed)
            nodes.append(node._rhs)
            self._compile(node._rhs, traversed)

        expr.accept(self)
        for node in nodes:
            self._dag.add_edge(self._expr_to_dag_node[node], self._expr_to_dag_node[expr])

        cached_args = expr.args

        def cb():
            expr._cached_args = cached_args
        self._callbacks.append(cb)

        expr._cached_args = [None] * len(expr.args)

    @classmethod
    def _retrieve_until_find_root(cls, expr):
        for node in expr.traverse(top_down=True, unique=True):
            if isinstance(node, JoinCollectionExpr):
                return node

    def _add_node(self, expr, handle):
        children = expr.children()

        node = (expr, handle)
        self._dag.add_node(node)
        self._expr_to_dag_node[expr] = node

        predecessors = [self._expr_to_dag_node[child]
                        for child in children]
        [self._dag.add_edge(p, node) for p in predecessors]

    def visit_source_collection(self, expr):
        df = next(expr.data_source())

        if not isinstance(df, pd.DataFrame):
            raise ValueError('Expr data must be a pandas DataFrame.')

        handle = lambda _: df.copy()  # make a copy to avoid modify
        self._add_node(expr, handle)

    @classmethod
    def _get_children_vals(cls, kw, expr=None, children=None):
        children = children or expr.children()
        return [kw.get(child) for child in children]

    def visit_project_collection(self, expr):
        def handle(kw):
            children = expr.children()

            fields = self._get_children_vals(kw, children=children)[1:]
            names = expr.schema.names

            if isinstance(expr, Summary):
                size = 1
            else:
                size = max(len(f) for f, e in zip(fields, expr._fields)
                           if isinstance(e, SequenceExpr))
            for i in range(len(fields)):
                child = children[1:][i]
                if isinstance(child, Scalar):
                    fields[i] = pd.Series([fields[i]] * size)

            return pd.concat(fields, axis=1, keys=names)

        self._add_node(expr, handle)

    def visit_filter_collection(self, expr):
        def handle(kw):
            df, predicate = tuple(self._get_children_vals(kw, expr))
            return df[predicate]

        self._add_node(expr, handle)

    def visit_slice_collection(self, expr):
        def handle(kw):
            children_vals = self._get_children_vals(kw, expr)

            df = children_vals[0]
            start, end, step = expr.start, expr.stop, expr.step

            return df[start: end: step]

        self._add_node(expr, handle)

    def visit_element_op(self, expr):
        def handle(kw):
            children_vals = self._get_children_vals(kw, expr)
            input, args = children_vals[0], children_vals[1:]

            if isinstance(expr.input, Scalar):
                input = pd.Series([input])

            def run():
                if isinstance(expr, element.IsNull):
                    return input.isnull()
                elif isinstance(expr, element.NotNull):
                    return input.notnull()
                elif isinstance(expr, element.FillNa):
                    return input.fillna(args[0])
                elif isinstance(expr, element.IsIn):
                    if isinstance(expr._values[0], SequenceExpr):
                        return input.isin(list(args[0]))
                    else:
                        return input.isin(args)
                elif isinstance(expr, element.NotIn):
                    if isinstance(expr._values[0], SequenceExpr):
                        return ~input.isin(list(args[0]))
                    else:
                        return ~input.isin(args)
                elif isinstance(expr, element.IfElse):
                    return np.where(input, args[0], args[1])
                elif isinstance(expr, element.Switch):
                    case = None if expr.case is None else kw.get(expr.case)
                    default = None if expr.default is None else kw.get(expr.default)

                    conditions = [kw.get(it) for it in expr.conditions]
                    thens = [kw.get(it) for it in expr.thens]
                    if case is not None:
                        conditions = [case == condition for condition in conditions]

                    size = max(len(val) for e, val in zip(expr.conditions + expr.thens, conditions + thens)
                               if isinstance(e, SequenceExpr))

                    curr = pd.Series([None] * size)
                    for condition, then in zip(conditions, thens):
                        curr = curr.where(-condition, then)
                    return curr.fillna(default)
                elif isinstance(expr, element.Between):
                    return input.between(*args)
                elif isinstance(expr, element.Cut):
                    bins = [bin.value for bin in expr.bins]
                    if expr.include_under:
                        bins.insert(0, -float('inf'))
                    if expr.include_over:
                        bins.append(float('inf'))
                    labels = [l.value for l in expr.labels]
                    return pd.cut(input, bins, right=expr.right, labels=labels,
                                  include_lowest=expr.include_lowest)

            if isinstance(expr.input, Scalar):
                return run()[0]
            else:
                return run()

        self._add_node(expr, handle)

    def visit_binary_op(self, expr):
        def handle(kw):
            children_vals = self._get_children_vals(kw, expr)

            if expr.lhs.dtype == df_types.datetime and expr.rhs.dtype == df_types.datetime:
                return (pd.to_datetime(children_vals[0]) - pd.to_datetime(children_vals[1])) / \
                       np.timedelta64(1, 'ms')

            op = BINARY_OP_TO_PANDAS[expr.node_name]
            return op(*children_vals)

        self._add_node(expr, handle)

    def visit_unary_op(self, expr):
        def handle(kw):
            children_vals = self._get_children_vals(kw, expr)
            op = UNARY_OP_TO_PANDAS[expr.node_name]
            return op(*children_vals)

        self._add_node(expr, handle)

    def visit_math(self, expr):
        def handle(kw):
            children_vals = self._get_children_vals(kw, expr)
            if isinstance(expr, math.Log) and expr._base is not None:
                base = expr._base.value
                return np.log(children_vals[0]) / np.log(base)
            else:
                op = getattr(np, expr.node_name.lower())
                return op(*children_vals)

        self._add_node(expr, handle)

    def visit_string_op(self, expr):
        def handle(kw):
            children_vals = self._get_children_vals(kw, expr)

            input = children_vals[0]
            if isinstance(expr.input, Scalar):
                input = pd.Series([input])

            assert len(expr._args) == len(expr.args)
            kv = dict((name.lstrip('_'), kw.get(arg))
                      for name, arg in zip(expr._args[1:], expr.args[1:]))
            op = expr.node_name.lower()

            if op == 'get':
                res = getattr(getattr(input, 'str'), op)(children_vals[1])
            else:
                if op == 'slice':
                    kv['stop'] = kv.pop('end', None)
                res = getattr(getattr(input, 'str'), op)(**kv)
            if isinstance(expr.input, Scalar):
                return res[0]
            else:
                return res

        self._add_node(expr, handle)

    def visit_datetime_op(self, expr):
        def handle(kw):
            children_vals = self._get_children_vals(kw, expr)

            input = children_vals[0]
            if isinstance(expr.input, Scalar):
                input = pd.Series([input])

            assert len(children_vals) == len(expr.args)
            kv = dict(zip([arg.lstrip('_') for arg in expr._args[1:]],
                          children_vals[1:]))
            op = expr.node_name.lower()

            res = getattr(getattr(input, 'dt'), op)
            if not isinstance(res, pd.Series):
                res = res(**kv)

            if isinstance(expr.input, Scalar):
                return res[0]
            else:
                return res

        self._add_node(expr, handle)

    def visit_groupby(self, expr):
        def handle(kw):
            fields_exprs = expr._fields or expr._by + expr._aggregations
            names = [fields_expr.name for fields_expr in fields_exprs]

            agg_ids = set(id(agg) for agg in expr._aggregations)
            agg_field_exprs = [field for field in fields_exprs
                               if id(field) in agg_ids]
            agg_fields = [kw.get(field) for field in agg_field_exprs]

            length = max(len(it) for it in agg_fields)
            for i in range(len(agg_fields)):
                if len(agg_fields[i]) == 1:
                    agg_fields[i] = agg_fields[i] * length

            df = pd.concat(agg_fields, axis=1)
            if expr._having is not None:
                df = df[kw.get(expr._having)]

            if len(expr._by) == 1:
                index_name = str(uuid.uuid4())
            else:
                index_name = [str(uuid.uuid4()) for _ in range(len(expr._by))]
            df.index.rename(index_name, inplace=True)
            df.reset_index(inplace=True)

            by_names = [by.name for by in expr._by]
            agg_names = [f.name for f in agg_field_exprs]

            return pd.DataFrame(df.values, columns=by_names+agg_names)[names]

        self._add_node(expr, handle)

    def visit_mutate(self, expr):
        raise NotImplementedError

    def visit_value_counts(self, expr):
        def handle(kw):
            by = kw.get(expr._by)

            df = by.value_counts().to_frame()
            df.reset_index(inplace=True)
            return pd.DataFrame(df.values, columns=expr.schema.names)

        self._add_node(expr, handle)

    def visit_sort(self, expr):
        def handle(kw):
            input = kw.get(expr.input)
            names = expr.schema.names

            sorted_fields = []
            for field in expr._sorted_fields:
                name = str(uuid.uuid4())
                sorted_fields.append(name)
                input[name] = kw.get(field)

            return input.sort_values(sorted_fields, ascending=expr._ascending)[names]

        self._add_node(expr, handle)

    def visit_sort_column(self, expr):
        def handle(kw):
            input = kw.get(expr.input)
            if isinstance(expr.input, CollectionExpr):
                return input[expr._source_name]
            else:
                return input

        self._add_node(expr, handle)

    def visit_distinct(self, expr):
        def handle(kw):
            children_vals = self._get_children_vals(kw, expr)
            fields = children_vals[1:]

            return pd.concat(fields, axis=1, keys=expr.schema.names).drop_duplicates()

        self._add_node(expr, handle)

    def _compile_grouped_reduction(self, kw, expr):
        if isinstance(expr.raw_input, GroupBy) and \
                isinstance(expr, GroupedCount):
            df = kw.get(expr.input)
            grouped = expr.raw_input

            bys = [[kw.get(by), ] if isinstance(by, Scalar) else kw.get(by)
                   for by in grouped._by]
            if any(isinstance(e, SequenceExpr) for e in grouped._by):
                size = max(len(by) for by, e in zip(bys, grouped._by)
                           if isinstance(e, SequenceExpr))
            else:
                size = len(df)
            bys = [(by * size if len(by) == 1 else by) for by in bys]
            return df.groupby(bys).size()

        df = kw.get(expr.input.input)
        field = expr.raw_input

        grouped = field.input
        bys = [[kw.get(by), ] if isinstance(by, Scalar) else kw.get(by)
               for by in grouped._by]
        if any(isinstance(e, SequenceExpr) for e in grouped._by):
            size = max(len(by) for by, e in zip(bys, grouped._by)
                       if isinstance(e, SequenceExpr))
        else:
            size = len(grouped)
        bys = [(by * size if len(by) == 1 else by) for by in bys]

        kv = dict()
        if hasattr(expr, '_ddof'):
            kv['ddof'] = expr._ddof
        op = expr.node_name.lower()
        op = 'size' if op == 'count' else op

        return getattr(getattr(df.groupby(bys), field.name), op)(**kv)

    def visit_reduction(self, expr):
        def handle(kw):
            if isinstance(expr, GroupedSequenceReduction):
                return self._compile_grouped_reduction(kw, expr)

            children_vals = self._get_children_vals(kw, expr)

            kv = dict()
            if hasattr(expr, '_ddof'):
                kv['ddof'] = expr._ddof
            op = expr.node_name.lower()
            op = 'size' if op == 'count' else op

            input = children_vals[0]
            if isinstance(expr, Count) and isinstance(expr.input, CollectionExpr):
                return len(input)
            return getattr(input, op)(**kv)

        self._add_node(expr, handle)

    def visit_column(self, expr):
        def handle(kw):
            chidren_vals = self._get_children_vals(kw, expr)
            return chidren_vals[0][expr._source_name]

        self._add_node(expr, handle)

    def visit_function(self, expr):
        def handle(kw):
            if not expr._multiple:
                input = self._get_children_vals(kw, expr)[0]

                if isinstance(expr.inputs[0], Scalar):
                    input = pd.Series([input])

                func = expr._func
                args = expr._func_args
                kwargs = expr._func_kwargs
                if args is not None and len(args) > 0:
                    raise NotImplementedError
                if kwargs is not None and len(kwargs) > 0:
                    raise NotImplementedError

                res = input.map(func)
                if isinstance(expr.inputs[0], Scalar):
                    return res[0]
                return res
            else:
                collection = next(it for it in expr.traverse(top_down=True, unique=True)
                                  if isinstance(it, CollectionExpr))
                input = kw.get(collection)

                def func(s):
                    names = [f.name for f in expr.inputs]
                    t = namedtuple('NamedArgs', names)
                    row = t(*s.tolist())
                    return expr._func(row, *expr._func_args, **expr._func_kwargs)

                return input.apply(func, axis=1, reduce=True,
                                   args=expr._func_args, **expr._func_kwargs)

        self._add_node(expr, handle)

    def visit_apply_collection(self, expr):
        def conv(l):
            if isinstance(l, tuple):
                l = list(l)
            elif not isinstance(l, list):
                l = [l, ]
            return l

        def handle(kw):
            input = pd.concat([kw.get(field) for field in expr.fields],
                              axis=1, keys=[f.name for f in expr.fields])

            names = [f.name for f in expr.fields]
            t = namedtuple('NamedArgs', names)

            func = expr._func
            if inspect.isfunction(func):
                is_generator_function = inspect.isgeneratorfunction(func)
                close_func = None
                is_close_generator_function = False
            elif callable(func):
                func = func()
                is_generator_function = inspect.isgeneratorfunction(func.__call__)
                close_func = func.close
                is_close_generator_function = inspect.isgeneratorfunction(close_func)
            else:
                raise NotImplementedError

            if hasattr(expr, '_sort_fields') and expr._sort_fields is not None:
                input = input.sort_values([f.name for f in expr._sort_fields])

            rows = []
            for s in input.iterrows():
                row = t(*s[1])
                res = func(row, *expr._func_args, **expr._func_kwargs)
                if is_generator_function:
                    for l in res:
                        rows.append(conv(l))
                else:
                    rows.append(conv(res))
            if close_func:
                if is_close_generator_function:
                    for l in close_func(*expr._func_args, **expr._func_kwargs):
                        rows.append(conv(l))
                else:
                    rows.append(close_func(*expr._func_args, **expr._func_kwargs))
            return pd.DataFrame(rows, columns=expr.schema.names)

        self._add_node(expr, handle)

    def visit_sequence(self, expr):
        raise NotImplementedError

    def visit_cum_window(self, expr):
        raise NotImplementedError

    def visit_rank_window(self, expr):
        raise NotImplementedError

    def visit_shift_window(self, expr):
        raise NotImplementedError

    def visit_scalar(self, expr):
        def handle(_):
            if isinstance(expr, DTScalar):
                arg_name = type(expr).__name__.lower()[:-6] + 's'
                return pd.DateOffset(**{arg_name: expr.value})
            if expr.value is not None:
                return expr.value

            raise NotImplementedError

        self._add_node(expr, handle)

    def visit_cast(self, expr):
        def handle(kw):
            dtype = types.df_type_to_np_type(expr.dtype)
            input = self._get_children_vals(kw, expr)[0]
            return input.astype(dtype)

        self._add_node(expr, handle)

    @classmethod
    def _find_all_equalizations(cls, predicate, lhs, rhs):
        return [eq for eq in predicate.traverse(top_down=True, unique=True)
                if isinstance(eq, arithmetic.Equal) and
                eq.is_ancestor(lhs) and eq.is_ancestor(rhs)]

    def visit_join(self, expr):
        def handle(kw):
            left = kw.get(expr._lhs)
            right = kw.get(expr._rhs)

            eqs = self._find_all_equalizations(expr._predicate, expr._lhs, expr._rhs)

            left_ons = []
            right_ons = []
            for eq in eqs:
                left_name = str(uuid.uuid4())
                left[left_name] = kw.get(eq._lhs)
                left_ons.append(left_name)

                right_name = str(uuid.uuid4())
                right[right_name] = kw.get(eq._rhs)
                right_ons.append(right_name)

            merged = left.merge(right, how=expr._how.lower(), left_on=left_ons,
                                right_on=right_ons,
                                suffixes=(expr._left_suffix, expr._right_suffix))
            return merged[expr.schema.names]

        # Just add node, shouldn't add edge here
        node = (expr, handle)
        self._dag.add_node(node)
        self._expr_to_dag_node[expr] = node

    def visit_union(self, expr):
        if expr._distinct:
            raise CompileError("Distinct union is not supported here.")

        def handle(kw):
            left = kw.get(expr._lhs)
            right = kw.get(expr._rhs)

            merged = pd.concat([left, right])
            return merged[expr.schema.names]

        self._add_node(expr, handle)
