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

from __future__ import absolute_import

import itertools
import inspect
import traceback
import collections
import threading
from datetime import datetime
from decimal import Decimal

from .. import types
from ..utils import is_source_collection
from ... import compat
from ...models import FileResource, TableResource
from ...compat import six


def add_method(expr, methods):
    for k, v in six.iteritems(methods):
        setattr(expr, k, v)


def same_data_source(*exprs):
    curr_data_source = None
    for expr in exprs:
        data_source = sorted(list(expr.data_source()))
        if curr_data_source is None:
            curr_data_source = data_source
        else:
            if curr_data_source != data_source:
                return False

    return True


def highest_precedence_data_type(*data_types):
    data_types = set(data_types)
    if len(data_types) == 1:
        return data_types.pop()

    precedences = dict((t, idx) for idx, t in enumerate(
        [types.string, types.boolean, types.int8, types.int16, types.int32, types.int64,
         types.datetime, types.decimal, types.float32, types.float64]))

    type_precedences = [(precedences[data_type], data_type) for data_type in data_types]
    highest_data_type = sorted(type_precedences)[-1][1]
    for data_type in data_types:
        if data_type != highest_data_type and not highest_data_type.can_implicit_cast(data_type):
            raise TypeError(
                'Type cast error: %s cannot implicitly cast to %s' % (data_type, highest_data_type))

    return highest_data_type


def get_attrs(node):
    from .core import Node

    tp = type(node) if not inspect.isclass(node) else node

    if inspect.getmro(tp) is None:
        tp = type(tp)

    return tuple(compat.OrderedDict.fromkeys(
        it for it in
        itertools.chain(*(cls.__slots__ for cls in inspect.getmro(tp) if issubclass(cls, Node)))
        if not it.startswith('__')))


def get_collection_resources(resources):
    from .expressions import CollectionExpr

    if resources:
        for res in resources:
            if not isinstance(res, (TableResource, FileResource, CollectionExpr)):
                raise ValueError('resources must be ODPS file or table Resources or collections')
    if resources is not None and len(resources) > 0:
        ret = [res for res in resources if isinstance(res, CollectionExpr)]
        [r.cache() for r in ret]  # we should execute the expressions by setting cache=True
        return ret


def get_executed_collection_project_table_name(collection):
    from .expressions import CollectionExpr
    from ...models import Table
    from ..backends.context import context

    if not isinstance(collection, CollectionExpr):
        return

    if collection._source_data is not None and \
            isinstance(collection._source_data, Table):
        source_data = collection._source_data
        return source_data.project.name + '.' + source_data.name

    if context.is_cached(collection) and \
            isinstance(context.get_cached(collection), Table):
        source_data = context.get_cached(collection)
        return source_data.project.name + '.' + source_data.name


def is_called_by_inspector():
    return any(1 for v in traceback.extract_stack() if 'oinspect' in v[0].lower() and 'ipython' in v[0].lower())


def to_list(field):
    if isinstance(field, six.string_types):
        return [field, ]
    if isinstance(field, collections.Iterable):
        return list(field)
    return [field, ]


_lock = threading.Lock()
_index = itertools.count(1)


def new_id():
    with _lock:
        return next(_index)


def select_fields(collection):
    from .expressions import ProjectCollectionExpr, Summary
    from .collections import DistinctCollectionExpr, RowAppliedCollectionExpr
    from .groupby import GroupByCollectionExpr, MutateCollectionExpr

    if isinstance(collection, (ProjectCollectionExpr, Summary)):
        return collection.fields
    elif isinstance(collection, DistinctCollectionExpr):
        return collection.unique_fields
    elif isinstance(collection, (GroupByCollectionExpr, MutateCollectionExpr)):
        return collection.fields
    elif isinstance(collection, RowAppliedCollectionExpr):
        return collection.fields


def is_changed(collection, column):
    # if the column is changed before the generated collection
    from .expressions import CollectionExpr, Column

    column_name = column.source_name
    src_collection = column.input

    if src_collection is collection:
        return False

    dag = collection.to_dag(copy=False, validate=False)
    coll = src_collection
    colls = []
    while coll is not collection:
        try:
            parents = [p for p in dag.successors(coll) if isinstance(p, CollectionExpr)]
        except KeyError:
            return
        assert len(parents) == 1
        coll = parents[0]
        colls.append(coll)

    name = column_name
    for coll in colls:
        fields = select_fields(coll)
        if fields:
            col_names = dict((field.source_name, field) for field in fields if isinstance(field, Column))
            if name in col_names:
                name = col_names[name].name
            else:
                return True

    return False


annotation_rtypes = {
    int: types.int64,
    str: types.string,
    float: types.float64,
    bool: types.boolean,
    datetime: types.datetime,
    Decimal: types.decimal,
}


def get_annotation_rtype(func):
    if hasattr(func, '__annotations__'):
        try:
            from typing import Union
        except ImportError:
            Union = None

        ret_type = func.__annotations__.get('return')
        if ret_type in annotation_rtypes:
            return annotation_rtypes.get(ret_type)
        elif hasattr(ret_type, '__origin__') and ret_type.__origin__ is Union:
            actual_types = [typo for typo in ret_type.__args__
                            if typo is not type(None)]
            if len(actual_types) == 1:
                return annotation_rtypes.get(actual_types[0])
        elif Union is not None and type(ret_type) is type(Union):
            actual_types = [typo for typo in ret_type.__args__
                            if typo is not type(None)]
            if len(actual_types) == 1:
                return annotation_rtypes.get(actual_types[0])
    return None


def get_proxied_expr(expr):
    try:
        obj = object.__getattribute__(expr, '_proxy')
        return obj if obj is not None else expr
    except AttributeError:
        return expr
