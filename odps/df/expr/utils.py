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

import itertools
import inspect
import traceback

from .. import types
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
    for data_type in data_types :
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


def get_executed_collection_table_name(collection):
    from .expressions import CollectionExpr
    from ...models import Table

    if not isinstance(collection, CollectionExpr):
        return

    if collection._source_data is not None and \
            isinstance(collection._source_data, Table):
        return collection._source_data.name

    if collection._cache_data is not None and \
            isinstance(collection._cache_data, Table):
        return collection._cache_data.name


def is_called_by_inspector():
    return any(1 for v in traceback.extract_stack() if 'oinspect' in v[0].lower() and 'ipython' in v[0].lower())