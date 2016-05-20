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
import uuid
import itertools
from datetime import datetime

from ..core import Backend
from ...expr.expressions import *
from ...expr.reduction import GroupedSequenceReduction, GroupedCount, Count
from ...expr.merge import JoinCollectionExpr
from ...expr.datetimes import DTScalar
from ...expr import element
from ...expr import arithmetic
from ....dag import DAG
from ..errors import CompileError
from . import types
from ... import types as df_types
from ....models import FileResource, TableResource
from .... import compat

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pd = None
    np = None


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

if pd:
    SORT_CUM_WINDOW_OP_TO_PANDAS = {
        'CumSum': pd.expanding_sum,
        'CumMean': pd.expanding_mean,
        'CumMedian': pd.expanding_median,
        'CumStd': pd.expanding_std,
        'CumMin': pd.expanding_min,
        'CumMax': pd.expanding_max,
        'CumCount': pd.expanding_count,
    }

if np:
    CUM_WINDOW_OP_TO_PANDAS = {
        'CumSum': np.sum,
        'CumMean': np.mean,
        'CumMedian': np.median,
        'CumStd': np.std,
        'CumMin': np.min,
        'CumMax': np.max,
        'CumCount': lambda x: len(x)
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

        # make a copy to avoid modify
        handle = lambda _: df.rename(columns=dict(zip(df.columns, expr.schema.names)))
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
            elif op == 'strptime':
                res = input.map(lambda x: datetime.strptime(x, children_vals[1]))
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

            fields = [[kw.get(field), ] if isinstance(field, Scalar) else kw.get(field)
                      for field in fields_exprs]
            length = max(len(it) for it in fields)
            for i in range(len(fields)):
                bys = None
                if isinstance(fields_exprs[i], SequenceExpr):
                    is_reduction = False
                    for n in itertools.chain(*(fields_exprs[i].all_path(expr.input))):
                        if isinstance(n, GroupedSequenceReduction):
                            is_reduction = True
                            break
                    if not is_reduction:
                        if not bys:
                            bys = self._get_compiled_bys(kw, expr._by, length)
                        fields[i] = fields[i].groupby(bys).first()
                if len(fields[i]) == 1:
                    fields[i] = fields[i] * length

            df = pd.concat(fields, axis=1)
            if expr._having is not None:
                df = df[kw.get(expr._having)]
            return pd.DataFrame(
                df.values, columns=[f.name for f in fields_exprs])[expr.schema.names]

        self._add_node(expr, handle)

    def visit_mutate(self, expr):
        def handle(kw):
            bys = self._get_compiled_bys(kw, expr._by, len(kw.get(expr.input)))
            bys = pd.concat(bys)
            bys.sort_values(inplace=True)

            wins = [kw.get(f) for f in expr._window_fields]
            return pd.DataFrame(pd.concat([bys] + wins, axis=1).values,
                                columns=expr.schema.names)

        self._add_node(expr, handle)

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

    def visit_sample(self, expr):
        def handle(kw):
            input = kw.get(expr.input)
            parts = kw.get(expr._parts)
            i = kw.get(expr._i)
            n = kw.get(expr._n)
            frac = kw.get(expr._frac)
            if expr._sampled_fields:
                collection = pd.DataFrame(
                    pd.concat([kw.get(e) for e in expr._sampled_fields], axis=1).values,
                    columns=[str(uuid.uuid4()) for _ in expr._sampled_fields])
            else:
                collection = input

            if parts is not None and frac is None:
                frac = 1 / float(parts)
            if i is not None and (len(i) != 1 or i[0] > 0):
                raise NotImplementedError
            sampled = collection.sample(n=n, frac=frac)
            if expr._sampled_fields:
                return pd.concat([input, sampled], axis=1, join='inner')[
                    [n for n in input.columns.tolist()]]
            return sampled

        self._add_node(expr, handle)

    def _get_compiled_bys(self, kw, by_exprs, length):
        bys = [[kw.get(by), ] if isinstance(by, Scalar) else kw.get(by)
               for by in by_exprs]
        if any(isinstance(e, SequenceExpr) for e in by_exprs):
            size = max(len(by) for by, e in zip(bys, by_exprs)
                       if isinstance(e, SequenceExpr))
        else:
            size = length
        return [(by * size if len(by) == 1 else by) for by in bys]

    def _compile_grouped_reduction(self, kw, expr):
        if isinstance(expr, GroupedCount) and isinstance(expr._input, CollectionExpr):
            df = kw.get(expr.input)
            grouped = expr._grouped

            bys = [[kw.get(by), ] if isinstance(by, Scalar) else kw.get(by)
                   for by in grouped._by]
            if any(isinstance(e, SequenceExpr) for e in grouped._by):
                size = max(len(by) for by, e in zip(bys, grouped._by)
                           if isinstance(e, SequenceExpr))
            else:
                size = len(df)
            bys = [(by * size if len(by) == 1 else by) for by in bys]
            return df.groupby(bys).size()

        grouped = expr._grouped
        series = kw.get(expr.input)
        bys = self._get_compiled_bys(kw, grouped._by, len(series))

        kv = dict()
        if hasattr(expr, '_ddof'):
            kv['ddof'] = expr._ddof
        op = expr.node_name.lower()
        op = 'size' if op == 'count' else op

        return getattr(series.groupby(bys), op)(**kv)

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

    def visit_user_defined_aggregator(self, expr):
        def handle(kw):
            resources = self._get_resources(expr, kw)

            input = kw.get(expr.input)

            func = expr._aggregator
            args = expr._func_args
            kwargs = expr._func_kwargs or dict()

            if resources:
                if not args and not kwargs:
                    agg = func(resources)
                else:
                    kwargs['resources'] = resources
                    agg = func(*args, **kwargs)
            else:
                agg = func(*args, **kwargs)

            if isinstance(expr, GroupedSequenceReduction):
                bys = [[kw.get(by), ] if isinstance(by, Scalar) else kw.get(by)
                       for by in expr._grouped._by]
                grouped = expr._grouped
            else:
                bys = [[1, ]]
                grouped = None
            if grouped and any(isinstance(e, SequenceExpr) for e in grouped._by):
                size = max(len(by) for by, e in zip(bys, grouped._by)
                           if isinstance(e, SequenceExpr))
            else:
                size = len(input)
            bys = [(by * size if len(by) == 1 else by) for by in bys]

            def f(x):
                buffer = agg.buffer()
                for it in x:
                    agg(buffer, it)
                ret = agg.getvalue(buffer)
                np_type = types.df_type_to_np_type(expr.dtype)
                return np.array([ret,], dtype=np_type)[0]

            res = input.groupby(bys).agg(f)
            if isinstance(expr, Scalar):
                return res.iloc[0]
            return res

        self._add_node(expr, handle)

    def visit_column(self, expr):
        def handle(kw):
            chidren_vals = self._get_children_vals(kw, expr)
            # FIXME: consider the name which is unicode
            return chidren_vals[0][expr._source_name]

        self._add_node(expr, handle)

    def _get_resources(self, expr, kw):
        if not expr._resources:
            return

        res = []
        collection_idx = 0
        for resource in expr._resources:
            if isinstance(resource, FileResource):
                res.append(resource.open())
            elif isinstance(resource, TableResource):
                def gen():
                    table = resource.get_source_table()
                    named_args = namedtuple('NamedArgs', table.schema.names)
                    partition = resource.get_source_table_partition()
                    with table.open_reader(partition=partition) as reader:
                        for r in reader:
                            yield named_args(*r.values)
                res.append(gen())
            else:
                resource = expr._collection_resources[collection_idx]
                collection_idx += 1

                df = kw.get(resource)

                def gen():
                    named_args = namedtuple('NamedArgs', resource.schema.names)
                    for r in df.iterrows():
                        yield named_args(*r[1])
                res.append(gen())

        return res

    def visit_function(self, expr):
        def handle(kw):
            resources = self._get_resources(expr, kw)
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

                if not inspect.isfunction(func):
                    if resources:
                        func = func(resources)
                    else:
                        func = func()
                else:
                    if resources:
                        func = func(resources)

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
                    if not inspect.isfunction(expr._func):
                        if resources:
                            f = expr._func(resources)
                        else:
                            f = expr._func()
                    else:
                        if resources:
                            f = expr._func(resources)
                        f = expr._func
                    return f(row, *expr._func_args, **expr._func_kwargs)

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
            resources = self._get_resources(expr, kw)

            input = pd.concat([kw.get(field) for field in expr.fields],
                              axis=1, keys=[f.name for f in expr.fields])

            names = [f.name for f in expr.fields]
            t = namedtuple('NamedArgs', names)

            func = expr._func
            if inspect.isfunction(func):
                if resources:
                    func = func(resources)

                is_generator_function = inspect.isgeneratorfunction(func)
                close_func = None
                is_close_generator_function = False
            elif hasattr(func, '__call__'):
                if resources:
                    func = func(resources)
                else:
                    func = func()
                is_generator_function = inspect.isgeneratorfunction(func.__call__)
                close_func = getattr(func, 'close', None)
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
                    if res:
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
        if expr.preceding is not None or expr.following is not None:
            raise NotImplementedError

        def handle(kw):
            input = kw.get(expr.input)
            bys = self._get_compiled_bys(kw, expr.partition_by, len(input))
            grouped = input.groupby(bys)
            if expr.order_by:
                sort = [kw.get(e) for e in expr.order_by]
                ascendings = [e._ascending for e in expr.order_by]
                for s in sort:
                    sort_name = str(uuid.uuid4())
                    s.name = sort_name
            else:
                sort = None
                ascendings = None

            def f(x):
                if sort:
                    df = pd.concat([x] + sort, join='inner', axis=1)
                    df.sort_values([s.name for s in sort], ascending=ascendings, inplace=True)
                    series = df[x.name]

                    return SORT_CUM_WINDOW_OP_TO_PANDAS[expr.node_name](series)
                else:
                    if expr.distinct:
                        new_x = x.drop_duplicates()
                    else:
                        new_x = x
                    val = CUM_WINDOW_OP_TO_PANDAS[expr.node_name](new_x)
                    return pd.Series([val] * len(x), index=x.index)

            res = grouped.apply(f)
            if sort:
                for _ in bys:
                    res = res.reset_index(level=0, drop=True)
            return res

        self._add_node(expr, handle)

    def visit_rank_window(self, expr):
        def handle(kw):
            input = kw.get(expr.input)
            sort = [kw.get(e) * (1 if e._ascending else -1)
                    for e in expr.order_by]
            bys = self._get_compiled_bys(kw, expr.partition_by, len(input))

            sort_names = [str(uuid.uuid4()) for _ in sort]
            by_names = [str(uuid.uuid4()) for _ in bys]
            input_names = [input.name] if isinstance(input, pd.Series) else input.columns.tolist()
            df = pd.DataFrame(pd.concat([input] + sort + bys, axis=1).values,
                              columns=input_names + sort_names + by_names,
                              index=input.index)
            df.sort_values(sort_names, inplace=True)
            grouped = df.groupby(by_names)

            def f(x):
                s_df = pd.Series(pd.lib.fast_zip([x[s].values for s in sort_names]), index=x.index)
                if expr.node_name == 'Rank':
                    return s_df.rank(method='min')
                elif expr.node_name == 'DenseRank':
                    return s_df.rank(method='dense')
                elif expr.node_name == 'RowNumber':
                    return pd.Series(compat.lrange(1, len(s_df)+1), index=s_df.index)
                elif expr.node_name == 'PercentRank':
                    if len(s_df) == 1:
                        return pd.Series([0.0, ], index=s_df.index)
                    return (s_df.rank(method='min') - 1) / (len(s_df) - 1)
                else:
                    raise NotImplementedError

            res = grouped.apply(f)
            for _ in bys:
                res = res.reset_index(level=0, drop=True)
            return res

        self._add_node(expr, handle)

    def visit_shift_window(self, expr):
        def handle(kw):
            input = kw.get(expr.input)

            bys = self._get_compiled_bys(kw, expr.partition_by, len(input))
            grouped = input.groupby(bys)
            if expr.order_by:
                sort = [kw.get(e) for e in expr.order_by]
                ascendings = [e._ascending for e in expr.order_by]
                for s in sort:
                    sort_name = str(uuid.uuid4())
                    s.name = sort_name
            else:
                sort = None
                ascendings = None

            if expr.node_name == 'Lag':
                shift = kw.get(expr.offset)
            else:
                assert expr.node_name == 'Lead'
                shift = -kw.get(expr.offset)
            default = kw.get(expr.default)

            def f(x):
                if sort:
                    df = pd.concat([x] + sort, join='inner', axis=1)
                    df.sort_values([s.name for s in sort], ascending=ascendings, inplace=True)
                    series = df[x.name]
                else:
                    series = x

                res = series.shift(shift)
                if default is not None:
                    return res.fillna(default)
                return res

            res = grouped.apply(f)
            if sort:
                for _ in bys:
                    res = res.reset_index(level=0, drop=True)
            return res

        self._add_node(expr, handle)

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
                if eq._lhs.name == eq._rhs.name:
                    left_ons.append(eq._lhs.name)
                    right_ons.append(eq._rhs.name)
                    continue

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
