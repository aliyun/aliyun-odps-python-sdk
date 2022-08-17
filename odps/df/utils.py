#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import inspect


class FunctionWrapper(object):
    def __init__(self, func):
        self._func = func
        self.output_names = None
        self.output_types = None

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __reduce__(self):
        def wrapper_restorer(func, names, types):
            try:
                from odps.df import output
            except ImportError:
                return func

            return output(names, types)(func)

        return (
            wrapper_restorer,
            (self._func, self.output_names, self.output_types),
        )


def output_names(*names):
    if len(names) == 1 and isinstance(names[0], (tuple, list)):
        names = tuple(names[0])

    def inner(func):
        if isinstance(func, FunctionWrapper):
            wrapper = func
        else:
            wrapper = FunctionWrapper(func)
        wrapper.output_names = names
        return wrapper
    return inner


def output_types(*types):
    if len(types) == 1 and isinstance(types[0], (tuple, list)):
        types = tuple(types[0])

    def inner(func):
        if isinstance(func, FunctionWrapper):
            wrapper = func
        else:
            wrapper = FunctionWrapper(func)
        wrapper.output_types = types
        return wrapper
    return inner


def output(names, types):
    if isinstance(names, tuple):
        names = list(names)
    if not isinstance(names, list):
        names = [names, ]

    if isinstance(types, tuple):
        types = list(types)
    if not isinstance(types, list):
        types = [types, ]

    def inner(func):
        if isinstance(func, FunctionWrapper):
            wrapper = func
        else:
            wrapper = FunctionWrapper(func)
        wrapper.output_names = names
        wrapper.output_types = types
        return wrapper
    return inner


def make_copy(f):
    if inspect.isfunction(f):
        if not inspect.isgeneratorfunction(f):
            return lambda *args, **kwargs: f(*args, **kwargs)
        else:
            def new_f(*args, **kwargs):
                for it in f(*args, **kwargs):
                    yield it
            return new_f
    elif inspect.isclass(f):
        class NewCls(f):
            pass
        return NewCls
    else:
        return f


def is_source_collection(expr):
    from .expr.expressions import CollectionExpr

    return (isinstance(expr, CollectionExpr) and expr._source_data is not None) or \
           (type(expr) is CollectionExpr and expr._deps is not None)


def is_constant_scalar(expr):
    from .expr.expressions import Scalar

    return isinstance(expr, Scalar) and expr._value is not None


def is_source_partition(expr, table):
    from .expr.expressions import Column

    if not isinstance(expr, Column):
        return False

    odps_schema = table.schema
    if not odps_schema.is_partition(expr.source_name):
        return False

    return True


def to_collection(seq_or_scalar):
    from .expr.expressions import CollectionExpr, Column, SequenceExpr
    from .expr.reduction import GroupedSequenceReduction, Count

    if seq_or_scalar._non_table:
        return seq_or_scalar
    if isinstance(seq_or_scalar, CollectionExpr):
        return seq_or_scalar

    expr = seq_or_scalar
    for node in expr.traverse(top_down=True, unique=True):
        if isinstance(node, GroupedSequenceReduction):
            return node._grouped.agg(expr)[[expr.name, ]]
        elif isinstance(node, Column):
            return node.input[[expr, ]]
        elif isinstance(node, Count) and isinstance(node.input, CollectionExpr):
            return node.input[[expr, ]]
        elif isinstance(node, SequenceExpr) and hasattr(node, '_input') and \
                isinstance(node._input, CollectionExpr):
            return node.input[[expr, ]]

    raise NotImplementedError


def is_project_expr(collection):
    from .expr.expressions import ProjectCollectionExpr, FilterPartitionCollectionExpr
    from .expr.groupby import GroupByCollectionExpr
    from .expr.window import MutateCollectionExpr
    from .expr.collections import DistinctCollectionExpr

    if isinstance(collection, (ProjectCollectionExpr, FilterPartitionCollectionExpr,
                               GroupByCollectionExpr, MutateCollectionExpr,
                               DistinctCollectionExpr)):
        return True
    return False


def traverse_until_source(expr_or_dag, *args, **kwargs):
    if 'stop_cond' not in kwargs:
        kwargs['stop_cond'] = lambda e: is_source_collection(e) or is_constant_scalar(e)
    return expr_or_dag.traverse(*args, **kwargs)
