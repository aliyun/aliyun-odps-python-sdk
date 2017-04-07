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

import itertools
import json
from datetime import datetime
import re
import time

from ..core import Backend
from ...expr.expressions import *
from ...expr.arithmetic import Power
from ...expr.reduction import GroupedSequenceReduction, GroupedCount, Count, GroupedCat, Cat
from ...expr.merge import JoinCollectionExpr
from ...expr.datetimes import DTScalar
from ...expr.collections import PivotCollectionExpr
from ...expr import arithmetic, element
from ...utils import traverse_until_source
from ....dag import DAG
from ..errors import CompileError
from ..utils import refresh_dynamic
from . import types
from ... import types as df_types
from ....models import FileResource, TableResource, Schema
from .... import compat
from ....lib.xnamedtuple import xnamedtuple

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
    'Mod': operator.mod,
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
        'CumSum': lambda s: s.expanding(min_periods=1).sum(),
        'CumMean': lambda s: s.expanding(min_periods=1).mean(),
        'CumMedian': pd.expanding_median,
        'CumStd': pd.expanding_std,
        'CumMin': lambda s: s.expanding(min_periods=1).min(),
        'CumMax': lambda s: s.expanding(min_periods=1).max(),
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

JOIN_DICT = {
    'INNER': 'inner',
    'LEFT OUTER': 'left',
    'RIGHT OUTER': 'right',
    'FULL OUTER': 'outer'
}


class PandasCompiler(Backend):
    """
    PandasCompiler will compile an Expr into a DAG
    in which each node is a pair of <expr, function>.
    """

    def __init__(self, expr_dag):
        self._dag = DAG()
        self._expr_to_dag_node = dict()
        self._expr_dag = expr_dag
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

        for node in traverse_until_source(expr):
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
        for node in expr._predicate:
            nodes.append(node._lhs)
            self._compile(node._lhs, traversed)
            nodes.append(node._rhs)
            self._compile(node._rhs, traversed)

        expr.accept(self)
        for node in nodes:
            self._dag.add_edge(self._expr_to_dag_node[node], self._expr_to_dag_node[expr])

        cached_args = expr.args

        def cb():
            for arg_name, arg in zip(expr._args, cached_args):
                setattr(expr, arg_name, arg)
        self._callbacks.append(cb)

        for arg_name in expr._args:
            setattr(expr, arg_name, None)

    @classmethod
    def _retrieve_until_find_root(cls, expr):
        for node in traverse_until_source(expr, top_down=True, unique=True):
            if isinstance(node, JoinCollectionExpr):
                return node

    def _add_node(self, expr, handle):
        children = expr.children()

        node = (expr, handle)
        self._dag.add_node(node)
        self._expr_to_dag_node[expr] = node

        # the dependencies do not exist in self._expr_to_dag_node
        predecessors = [self._expr_to_dag_node[child] for child in children
                        if child in self._expr_to_dag_node]
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

    @classmethod
    def _merge_values(cls, exprs, kw):
        fields = [kw.get(expr) for expr in exprs]
        size = max(len(f) for f, e in zip(fields, exprs) if isinstance(e, SequenceExpr))
        fields = [pd.Series([f] * size) if isinstance(e, Scalar) else f
                  for f, e in zip(fields, exprs)]

        return pd.concat(fields, axis=1, keys=[e.name for e in exprs])

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
                if not isinstance(fields[i], pd.Series):
                    fields[i] = pd.Series([fields[i]] * size)

            return pd.concat(fields, axis=1, keys=names)

        self._add_node(expr, handle)

    def visit_filter_partition_collection(self, expr):
        def handle(kw):
            children_vals = self._get_children_vals(kw, expr)
            df, predicate = children_vals[0:1]
            return df[predicate][expr.schema.names]

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
                    return pd.Series(np.where(input, args[0], args[1]), name=expr.name)
                elif isinstance(expr, element.Switch):
                    case = None if expr.case is None else kw.get(expr.case)
                    default = None if expr.default is None else kw.get(expr.default)

                    conditions = [kw.get(it) for it in expr.conditions]
                    thens = [kw.get(it) for it in expr.thens]
                    if case is not None:
                        conditions = [case == condition for condition in conditions]
                        condition_exprs = [expr.case == cond for cond in expr.conditions]
                    else:
                        condition_exprs = expr.conditions

                    size = max(len(val) for e, val in zip(condition_exprs + expr.thens, conditions + thens)
                               if isinstance(e, SequenceExpr))

                    curr = pd.Series([None] * size)
                    for condition, then in zip(conditions, thens):
                        curr = curr.where(-condition, then)
                    if default is not None:
                        return curr.fillna(default)
                    return curr
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
                return ((pd.to_datetime(children_vals[0]) - pd.to_datetime(children_vals[1])) /
                        np.timedelta64(1, 'ms')).astype(np.int64)

            op = BINARY_OP_TO_PANDAS[expr.node_name]
            if isinstance(expr, Power) and isinstance(expr.dtype, df_types.Integer):
                return op(*children_vals).astype(types.df_type_to_np_type(expr.dtype))
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
            elif isinstance(expr, math.Trunc):
                decimals = expr._decimals.value
                order = 10 ** decimals
                return np.trunc(children_vals[0] * order) / order
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
            kv = dict((name.lstrip('_'), self._get(arg, kw))
                      for name, arg in zip(expr._args[1:], expr.args[1:]))
            op = expr.node_name.lower()

            if op == 'get':
                res = getattr(getattr(input, 'str'), op)(children_vals[1])
            elif op == 'strptime':
                res = input.map(lambda x: datetime.strptime(x, children_vals[1]))
            elif op == 'extract':
                def extract(x, pat, flags, group):
                    regex = re.compile(pat, flags=flags)
                    m = regex.match(x)
                    if m:
                        return m.group(group)
                df = self._merge_values([expr.input, expr._pat, expr._flags, expr._group], kw)
                return pd.Series([extract(*r[1]) for r in df.iterrows()])
            else:
                if op == 'slice':
                    kv['stop'] = kv.pop('end', None)
                elif op == 'replace':
                    assert 'regex' in kv
                    if kv['regex']:
                        kv.pop('regex')
                    else:
                        kv['pat'] = re.escape(kv['pat'])
                        kv.pop('regex')
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
                bys = self._get_compiled_bys(kw, expr._by, length)
                if isinstance(fields_exprs[i], SequenceExpr):
                    is_reduction = False
                    for n in itertools.chain(*(fields_exprs[i].all_path(expr.input))):
                        if isinstance(n, GroupedSequenceReduction):
                            is_reduction = True
                            break
                    if not is_reduction:
                        fields[i] = fields[i].groupby(bys).first()
                elif len(fields[i]) == 1:
                    fields[i] = pd.Series(fields[i] * length,
                                          name=fields_exprs[i].name).groupby(bys).first()

            df = pd.concat(fields, axis=1)
            if expr._having is not None:
                having = kw.get(expr._having)
                if all(not isinstance(e, GroupedSequenceReduction)
                       for e in itertools.chain(*expr._having.all_path(expr.input))):
                    # the having comes from the by fields, we need to do Series.groupby explicitly.
                    bys = self._get_compiled_bys(kw, expr._by, len(having))
                    having = having.groupby(bys).first()
                df = df[having]
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
            sort = kw.get(expr._sort)
            ascending = kw.get(expr._ascending)
            dropna = kw.get(expr._dropna)

            df = by.value_counts(sort=sort, ascending=ascending, dropna=dropna).to_frame()
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

    def _get(self, item, kw):
        if item is None:
            return
        if isinstance(item, (list, tuple, set)):
            return type(item)(kw.get(it) for it in item)
        return kw.get(item)

    def visit_sample(self, expr):
        def handle(kw):
            input = self._get(expr.input, kw)
            parts = self._get(expr._parts, kw)
            i = self._get(expr._i, kw)
            n = self._get(expr._n, kw)
            frac = self._get(expr._frac, kw)
            replace = self._get(expr._replace, kw)
            weights = self._get(expr._weights, kw)
            strata = self._get(expr._strata, kw)
            random_state = self._get(expr._random_state, kw)
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
            if not strata:
                sampled = collection.sample(n=n, frac=frac, replace=replace, weights=weights,
                                            random_state=random_state)
            else:
                frames = []
                frac = json.loads(frac) if expr._frac else dict()
                n = json.loads(n) if expr._n else dict()
                for val in itertools.chain(six.iterkeys(frac), six.iterkeys(n)):
                    v_frac = frac.get(val)
                    v_n = n.get(val)
                    filtered = collection[collection[strata].astype(str) == val]
                    sampled = filtered.sample(n=v_n, frac=v_frac, replace=replace, random_state=random_state)
                    frames.append(sampled)
                if frames:
                    sampled = pd.concat(frames)
                else:
                    sampled = pd.DataFrame(columns=collection.columns)
            if expr._sampled_fields:
                return pd.concat([input, sampled], axis=1, join='inner')[
                    [n for n in input.columns.tolist()]]
            return sampled

        self._add_node(expr, handle)

    def _get_names(self, x, force_list=False):
        if x is None:
            return x
        res = [it.name for it in x]
        if not force_list and len(res) == 1:
            return res[0]
        return res

    def _get_pivot_handler(self, expr):
        def handle(kw):
            df = self._merge_values(expr._group + expr._columns + expr._values, kw)

            pivoted = df.pivot(index=self._get_names(expr._group),
                               columns=self._get_names(expr._columns))
            columns = pivoted.columns.levels
            pivoted.reset_index(inplace=True)

            names = self._get_names(expr._group, True)
            tps = [g.dtype for g in expr._group]
            if len(columns[0]) == 1:
                tp = expr._values[0].dtype
                for name in columns[1]:
                    names.append(name)
                    tps.append(tp)
            else:
                for value_name, value_col in zip(columns[0], expr._values):
                    for name in columns[1]:
                        names.append('{0}_{1}'.format(name, value_name))
                        tps.append(value_col.dtype)
            expr._schema = Schema.from_lists(names, tps)

            res = pd.DataFrame(pivoted.values, columns=names)
            to_sub = CollectionExpr(_source_data=res, _schema=expr._schema)
            self._expr_dag.substitute(expr, to_sub)

            # trigger refresh of dynamic operations
            def func(expr):
                for c in traverse_until_source(expr, unique=True):
                    if c not in self._expr_to_dag_node:
                        c.accept(self)
            refresh_dynamic(to_sub, self._expr_dag, func=func)

            return to_sub, res

        return handle

    def _get_pivot_table_handler(self, expr):
        def get_real_aggfunc(aggfunc):
            if isinstance(aggfunc, six.string_types):
                if aggfunc == 'count':
                    return getattr(np, 'size')
                return getattr(np, aggfunc)
            if inspect.isclass(aggfunc):
                aggfunc = aggfunc()

                def func(x):
                    buffer = aggfunc.buffer()
                    for it in x:
                        aggfunc(buffer, it)
                    return aggfunc.getvalue(buffer)

                return func
            return aggfunc

        def handle(kw):
            columns = expr._columns if expr._columns else []
            df = self._merge_values(expr._group + columns + expr._values, kw)
            pivoted = df.pivot_table(index=self._get_names(expr._group),
                                     columns=self._get_names(expr._columns),
                                     values=self._get_names(expr._values),
                                     aggfunc=[get_real_aggfunc(f) for f in expr._agg_func],
                                     fill_value=expr.fill_value)
            levels = pivoted.columns.levels if isinstance(pivoted.columns, pd.MultiIndex) \
                else [pivoted.columns]
            pivoted.reset_index(inplace=True)

            names = self._get_names(expr._group, True)
            tps = [g.dtype for g in expr._group]
            columns_values = levels[-1] if expr._columns else [None, ]
            for agg_func_name in expr._agg_func_names:
                for value_col in expr._values:
                    for col in columns_values:
                        base = '{0}_'.format(col) if col is not None else ''
                        name = '{0}{1}_{2}'.format(base, value_col.name, agg_func_name)
                        names.append(name)
                        tps.append(value_col.dtype)
            if expr._columns:
                expr._schema = Schema.from_lists(names, tps)

            res = pd.DataFrame(pivoted.values, columns=names)
            to_sub = CollectionExpr(_source_data=res, _schema=expr._schema)
            self._expr_dag.substitute(expr, to_sub)

            # trigger refresh of dynamic operations
            def func(expr):
                for c in traverse_until_source(expr, unique=True):
                    if c not in self._expr_to_dag_node:
                        c.accept(self)

            refresh_dynamic(to_sub, self._expr_dag, func=func)

            return to_sub, res

        return handle

    def visit_pivot(self, expr):
        if isinstance(expr, PivotCollectionExpr):
            handle = self._get_pivot_handler(expr)
        else:
            handle = self._get_pivot_table_handler(expr)
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

            bys = [[kw.get(by), ] if isinstance(by, Scalar) else kw.get(by)
                   for by in expr._by]
            if any(isinstance(e, SequenceExpr) for e in expr._by):
                size = max(len(by) for by, e in zip(bys, expr._by)
                           if isinstance(e, SequenceExpr))
            else:
                size = len(df)
            bys = [(by * size if len(by) == 1 else by) for by in bys]
            return df.groupby(bys).size()

        series = kw.get(expr.input) if isinstance(expr.input, SequenceExpr) \
            else pd.Series([kw.get(expr.input)], name=expr.input.name)
        bys = self._get_compiled_bys(kw, expr._by, len(series))
        if isinstance(expr.input, Scalar):
            series = pd.Series(series.repeat(len(bys[0])).values, index=bys[0].index)

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
            if isinstance(expr, Count) and isinstance(expr.input, (CollectionExpr, SequenceExpr)):
                return len(input)
            elif isinstance(expr, (Cat, GroupedCat)):
                kv['sep'] = expr._sep.value if isinstance(expr._sep, Scalar) else expr._sep
                kv['na_rep'] = expr._na_rep.value \
                    if isinstance(expr._na_rep, Scalar) else expr._na_rep
                return getattr(getattr(input, 'str'), 'cat')(**kv)
            return getattr(input, op)(**kv)

        self._add_node(expr, handle)

    def visit_user_defined_aggregator(self, expr):
        def handle(kw):
            resources = self._get_resources(expr, kw)

            input = self._merge_values(expr._inputs, kw)

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
                       for by in expr._by]
            else:
                bys = [[1, ]]
            if expr._by and any(isinstance(e, SequenceExpr) for e in expr._by):
                size = max(len(by) for by, e in zip(bys, expr._by)
                           if isinstance(e, SequenceExpr))
            else:
                size = len(input)
            bys = [(by * size if len(by) == 1 else by) for by in bys]

            def f(x):
                buffer = agg.buffer()
                for it in x.iterrows():
                    agg(buffer, *it[1])
                ret = agg.getvalue(buffer)
                np_type = types.df_type_to_np_type(expr.dtype)
                return np.array([ret,], dtype=np_type)[0]

            res = input.groupby(bys).apply(f)
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
                    named_args = xnamedtuple('NamedArgs', table.schema.names)
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
                    named_args = xnamedtuple('NamedArgs', resource.schema.names)
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

                if inspect.isclass(func):
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
                input = self._merge_values(expr.inputs, kw)

                def func(s):
                    names = [f.name for f in expr.inputs]
                    t = xnamedtuple('NamedArgs', names)
                    row = t(*s.tolist())
                    if not inspect.isfunction(expr._func):
                        if resources:
                            f = expr._func(resources)
                        else:
                            f = expr._func()
                    else:
                        if resources:
                            f = expr._func(resources)
                        else:
                            f = expr._func

                    res = f(row, *expr._func_args, **expr._func_kwargs)
                    if not inspect.isgeneratorfunction(f):
                        return res
                    return next(res)

                return input.apply(func, axis=1, reduce=True,
                                   args=expr._func_args, **expr._func_kwargs)

        self._add_node(expr, handle)

    def visit_reshuffle(self, expr):
        def handle(kw):
            if expr._sort_fields is not None:
                input = kw.get(expr._input)
                names = []
                for sort in expr._sort_fields:
                    name = str(uuid.uuid4())
                    input[name] = kw.get(sort)
                    names.append(name)
                input = input.sort_values(
                    names, ascending=[f._ascending for f in expr._sort_fields])
                return input[expr.schema.names]
            return kw.get(expr._input)

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

            input = self._merge_values(expr.fields, kw)

            names = [f.name for f in expr.fields]
            t = xnamedtuple('NamedArgs', names)

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

            return None

        self._add_node(expr, handle)

    def visit_cast(self, expr):
        def handle(kw):
            dtype = types.df_type_to_np_type(expr.dtype)
            input = self._get_children_vals(kw, expr)[0]
            if isinstance(expr._input, Scalar):
                return pd.Series([input]).astype(dtype)[0]
            return input.astype(dtype)

        self._add_node(expr, handle)

    @classmethod
    def _find_all_equalizations(cls, predicate, lhs, rhs):
        return [eq for eq in traverse_until_source(predicate, top_down=True, unique=True)
                if isinstance(eq, arithmetic.Equal) and
                eq.is_ancestor(lhs) and eq.is_ancestor(rhs)]

    def visit_join(self, expr):
        def handle(kw):
            left = kw.get(expr._lhs)
            right = kw.get(expr._rhs)

            eqs = expr._predicate

            left_ons = []
            right_ons = []
            on_same_names = set()
            for eq in eqs:
                if isinstance(eq._lhs, Column) and isinstance(eq._rhs, Column) and \
                        eq._lhs.source_name == eq._rhs.source_name:
                    left_ons.append(eq._lhs.source_name)
                    right_ons.append(eq._rhs.source_name)
                    on_same_names.add(eq._lhs.source_name)
                    continue

                left_name = str(uuid.uuid4())
                left[left_name] = kw.get(eq._lhs)
                left_ons.append(left_name)

                right_name = str(uuid.uuid4())
                right[right_name] = kw.get(eq._rhs)
                right_ons.append(right_name)

            for idx, collection in enumerate([left, right]):
                collection_expr = (expr._lhs, expr._rhs)[idx]
                for field_name in collection_expr.schema.names:
                    if field_name in expr._renamed_columns and field_name in on_same_names:
                        new_name = expr._renamed_columns[field_name][idx]
                        collection[new_name] = collection[field_name]

            merged = left.merge(right, how=JOIN_DICT[expr._how], left_on=left_ons,
                                right_on=right_ons,
                                suffixes=(expr._left_suffix, expr._right_suffix))
            cols = []
            for name in expr.schema.names:
                if name in merged:
                    cols.append(merged[name])
                else:
                    cols.append(merged[expr._column_origins[name][1]])

            return pd.concat(cols, axis=1, keys=expr.schema.names)

        # Just add node, shouldn't add edge here
        node = (expr, handle)
        self._dag.add_node(node)
        self._expr_to_dag_node[expr] = node

    def visit_extract_kv(self, expr):
        def handle(kw):
            from ... import types
            _input = kw.get(expr._input)
            columns = [getattr(_input, c.name) for c in expr._columns]
            kv_delim = kw.get(expr._kv_delimiter)
            item_delim = kw.get(expr._item_delimiter)
            default = kw.get(expr._default)

            kv_slot_map = dict()
            app_col_names = []

            def validate_kv(v):
                parts = v.split(kv_delim)
                if len(parts) != 2:
                    raise ValueError('Malformed KV pair: %s' % v)
                return parts[0]

            for col in columns:
                kv_slot_map[col.name] = dict()

                keys = col.apply(lambda s: [validate_kv(kv) for kv in s.split(item_delim)])
                for k in sorted(compat.reduce(lambda a, b: set(a) | set(b), keys, set())):
                    app_col_names.append('%s_%s' % (col.name, k))
                    kv_slot_map[col.name][k] = len(app_col_names) - 1

            type_adapter = None
            if isinstance(expr._column_type, types.Float):
                type_adapter = float
            elif isinstance(expr._column_type, types.Integer):
                type_adapter = int

            append_grid = [[default] * len(app_col_names) for _ in compat.irange(len(_input))]
            for col in columns:
                series = getattr(_input, col.name)
                for idx, v in enumerate(series):
                    for kv_item in v.split(item_delim):
                        k, v = kv_item.split(kv_delim)
                        if type_adapter:
                            v = type_adapter(v)
                        append_grid[idx][kv_slot_map[col.name][k]] = v

            intact_names = [c.name for c in expr._intact]
            intact_types = [c.dtype for c in expr._intact]
            intact_df = _input[intact_names]
            append_df = pd.DataFrame(append_grid, columns=app_col_names)
            expr._schema = Schema.from_lists(
                intact_names + app_col_names,
                intact_types + [expr._column_type] * len(app_col_names),
            )

            res = pd.concat([intact_df, append_df], axis=1)
            to_sub = CollectionExpr(_source_data=res, _schema=expr._schema)
            self._expr_dag.substitute(expr, to_sub)

            # trigger refresh of dynamic operations
            def func(expr):
                for c in traverse_until_source(expr, unique=True):
                    if c not in self._expr_to_dag_node:
                        c.accept(self)

            refresh_dynamic(to_sub, self._expr_dag, func=func)

            return to_sub, res

        self._add_node(expr, handle)

    def visit_union(self, expr):
        if expr._distinct:
            raise CompileError("Distinct union is not supported here.")

        def handle(kw):
            left = kw.get(expr._lhs)
            right = kw.get(expr._rhs)

            merged = pd.concat([left, right])
            return merged[expr.schema.names]

        self._add_node(expr, handle)

    def visit_concat(self, expr):
        def handle(kw):
            left = kw.get(expr._lhs)
            right = kw.get(expr._rhs)

            merged = pd.concat([left, right], axis=1)
            return merged[expr.schema.names]

        self._add_node(expr, handle)

    def visit_append_id(self, expr):
        def handle(kw):
            _input = kw.get(expr._input)
            id_col = kw.get(expr._id_col)

            id_seq = pd.DataFrame(compat.lrange(len(_input)), columns=[id_col])
            return pd.concat([id_seq, _input], axis=1)

        self._add_node(expr, handle)

    def visit_split(self, expr):
        def handle(kw):
            _input = kw.get(expr._input)
            frac = kw.get(expr._frac)
            seed = kw.get(expr._seed) if expr._seed else None
            split_id = kw.get(expr._split_id)

            if seed is not None:
                np.random.seed(seed)

            cols = list(_input.columns)
            factor_col = 'rand_factor_%d' % int(time.time())
            factor_df = pd.DataFrame(np.random.rand(len(_input)), columns=[factor_col])
            concated_df = pd.concat([factor_df, _input], axis=1)

            if split_id == 0:
                return concated_df[concated_df[factor_col] <= frac][cols]
            else:
                return concated_df[concated_df[factor_col] > frac][cols]

        self._add_node(expr, handle)
