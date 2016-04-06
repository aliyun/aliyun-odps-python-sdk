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

from __future__ import absolute_import
from collections import namedtuple

from ...models import Schema
from ..expr.arithmetic import Negate
from ..utils import FunctionWrapper, output
from ...compat import OrderedDict
from .expressions import *
from . import utils


class SortedColumn(SequenceExpr):
    __slots__ = '_ascending',
    _args = '_input',

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_sort_column(self)


class SortedExpr(Expr):
    def _init(self, *args, **kwargs):
        super(SortedExpr, self)._init(*args, **kwargs)

        if isinstance(self._ascending, bool):
            self._ascending = tuple([self._ascending] * len(self._sorted_fields))
        if len(self._sorted_fields) != len(self._ascending):
            raise ValueError('Length of ascending must be 1 or as many as the length of fields')

        sorted_fields = list()
        for i, field in enumerate(self._sorted_fields):
            if isinstance(field, Negate) and isinstance(field._input, SequenceExpr):
                field = field._input
                ascending = list(self._ascending)
                ascending[i] = False
                self._ascending = tuple(ascending)

                attr_values = dict((attr, getattr(field, attr, None)) for attr in utils.get_attrs(field))
                attr_values['_ascending'] = False
                sorted_fields.append(SortedColumn(**attr_values))
            elif isinstance(field, SequenceExpr):
                column = SortedColumn(_input=field, _name=field.name, _source_name=field.source_name,
                                      _data_type=field._data_type,
                                      _source_data_type=field._source_data_type,
                                      _ascending=self._ascending[i])
                sorted_fields.append(column)
            else:
                from .groupby import SequenceGroupBy

                if isinstance(field, SequenceGroupBy):
                    field = field.name

                assert isinstance(field, six.string_types)
                sorted_fields.append(
                    SortedColumn(self._input[field], _name=field,
                                 _data_type=self._input._schema[field].type, _ascending=self._ascending[i]))
        self._sorted_fields = sorted_fields


class SortedCollectionExpr(SortedExpr, CollectionExpr):
    __slots__ = '_ascending',
    _args = '_input', '_sorted_fields'
    node_name = 'SortBy'

    def iter_args(self):
        for it in zip(['collection', 'keys'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        visitor.visit_sort(self)


def sort_values(expr, by, ascending=True):
    """
    Sort the collection by values. `sort` is an alias name for `sort_values`

    :param expr: collection
    :param by: the sequence or sequences to sort
    :param ascending: Sort ascending vs. descending. Sepecify list for multiple sort orders.
                      If this is a list of bools, must match the length of the by
    :return: Sorted collection

    :Example:

    >>> df.sort_values(['name', 'id'])  # 1
    >>> df.sort(['name', 'id'], ascending=False)  # 2
    >>> df.sort(['name', 'id'], ascending=[False, True])  # 3
    >>> df.sort([-df.name, df.id])  # 4, equal to #3
    """
    if not isinstance(by, list):
        by = [by, ]
    by = [it(expr) if inspect.isfunction(it) else it for it in by]
    return SortedCollectionExpr(expr, _sorted_fields=by, _ascending=ascending,
                                _schema=expr._schema)


class DistinctCollectionExpr(CollectionExpr):
    _args = '_input', '_unique_fields'
    node_name = 'Distinct'

    def _init(self, *args, **kwargs):
        super(DistinctCollectionExpr, self)._init(*args, **kwargs)

        get_field = lambda field: \
            Column(self._input, _name=field, _data_type=self._input._schema[field].type)

        if self._unique_fields:
            self._unique_fields = list(get_field(field) if isinstance(field, six.string_types) else field
                                       for field in self._unique_fields)

            names = [field.name for field in self._unique_fields]
            types = [field._data_type for field in self._unique_fields]
            self._schema = Schema.from_lists(names, types)
        else:
            self._unique_fields = list(get_field(field) for field in self._input._schema.names)
            self._schema = self._input._schema

    def iter_args(self):
        for it in zip(['collection', 'distinct'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    @property
    def fields(self):
        return self._unique_fields

    def accept(self, visitor):
        return visitor.visit_distinct(self)


def distinct(expr, on=None, *ons):
    """
    Get collection with duplicate rows removed, optionally only considering certain columns

    :param expr: collection
    :param on: sequence or sequences
    :return: dinstinct collection

    :Example:

    >>> df.distinct(['name', 'id'])
    >>> df['name', 'id'].distinct()
    """
    # TODO: on sequence? maybe call it `unique` to keep compatible with Pandas
    on = on or list()
    if not isinstance(on, list):
        on = [on, ]
    on = on + list(ons)

    on = [it(expr) if inspect.isfunction(it) else it for it in on]

    return DistinctCollectionExpr(expr, _unique_fields=on)


def unique(expr):
    if isinstance(expr, SequenceExpr):
        collection = next(it for it in expr.traverse(top_down=True, unique=True)
                          if isinstance(it, CollectionExpr))
        return collection.distinct(expr)[expr.name]


class RowAppliedCollectionExpr(CollectionExpr):
    __slots__ = '_func', '_func_args', '_func_kwargs', '_close_func'
    _args = '_input', '_fields'
    _node_name = 'Apply'

    @property
    def input(self):
        return self._input

    @property
    def input_types(self):
        return [f.dtype for f in self._fields]

    def accept(self, visitor):
        return visitor.visit_apply_collection(self)


def apply(expr, func, axis=0, names=None, types=None, reduce=False,
          args=(), **kwargs):
    if not isinstance(expr, CollectionExpr):
        return

    if isinstance(func, FunctionWrapper):
        names = names or func.output_names
        types = types or func.output_types
        func = func._func

    if axis == 0:
        raise NotImplementedError
    else:
        if names is not None:
            if isinstance(names, list):
                names = tuple(names)
            elif isinstance(names, six.string_types):
                names = (names, )

        from ..types import validate_data_type, string
        if types is not None:
            if isinstance(types, list):
                types = tuple(types)
            elif isinstance(types, six.string_types):
                types = (types, )

            types = tuple(validate_data_type(t) for t in types)
        if reduce:
            from .element import MappedExpr

            if names is not None and len(names) > 1:
                raise ValueError('When reduce, at most one name can be specified')
            name = names[0] if names is not None else None
            tp = types[0] if types is not None else string
            inputs = expr._fields if hasattr(expr, '_fields') and expr._fields is not None \
                else [expr[n] for n in expr.schema.names]
            return MappedExpr(_func=func, _func_args=args, _func_kwargs=kwargs,
                              _name=name, _data_type=tp,
                              _inputs=inputs, _multiple=True)
        else:
            if names is None:
                raise ValueError('Apply on rows should provide column names')
            tps = (string, ) * len(names) if types is None else (validate_data_type(t) for t in types)
            schema = Schema.from_lists(names, tps)
            return RowAppliedCollectionExpr(_func=func, _func_args=args,
                                            _func_kwargs=kwargs, _schema=schema,
                                            _input=expr,
                                            _fields=[expr[n] for n in expr.schema.names])


def map_reduce(expr, mapper, reducer, group=None, sort=None, ascending=True,
               mapper_output_names=None, mapper_output_types=None,
               reducer_output_names=None, reducer_output_types=None):
    def conv(l, collection=None):
        if l is None:
            return
        if isinstance(l, tuple):
            l = list(l)
        elif not isinstance(l, list):
            l = [l, ]

        if collection is None:
            return l
        return [it if not inspect.isfunction(it) else it(collection)
                for it in l]

    def gen_name():
        return 'pyodps_field_%s' % str(uuid.uuid4()).replace('-', '_')

    if isinstance(mapper, FunctionWrapper):
        mapper_output_names = mapper_output_names or mapper.output_names
        mapper_output_types = mapper_output_types or mapper.output_types

    mapper_output_names = conv(mapper_output_names)
    mapper_output_types = conv(mapper_output_types)

    if mapper_output_types is not None and mapper_output_names is None:
        mapper_output_names = [gen_name() for _ in range(len(mapper_output_types))]

    mapped = expr.apply(mapper, axis=1, names=mapper_output_names,
                        types=mapper_output_types)
    group = conv(group, collection=mapped) or mapper_output_names
    sort = sort or tuple()
    sort = list(OrderedDict.fromkeys(group + conv(sort, collection=mapped)))

    if len(sort) > len(group):
        ascending = [ascending, ] * (len(sort) - len(group)) \
            if isinstance(ascending, bool) else list(ascending)
        ascending = [True] * len(group) + ascending

    clustered = mapped.groupby(group).sort(sort, ascending=ascending)

    if isinstance(reducer, FunctionWrapper):
        reducer_output_names = reducer_output_names or reducer.output_names
        reducer_output_types = reducer_output_types or reducer.output_types
        reducer = reducer._func

    @output(reducer_output_names, reducer_output_types)
    class ActualReducer(object):
        def __init__(self):
            self._func = reducer
            self._curr = None
            self._prev_rows = None
            self._names = mapper_output_names
            self._key_named_tuple = namedtuple('NamedKeys', group)

            self._f = None

        def _is_generator_function(self, f):
            if inspect.isgeneratorfunction(f):
                return True
            elif callable(f) and inspect.isgeneratorfunction(f.__call__):
                return True
            return False

        def __call__(self, row):
            key = tuple(getattr(row, n) for n in group)
            k = self._key_named_tuple(*key)

            if self._prev_rows is not None:
                key_consumed = self._curr != key
                if self._is_generator_function(self._f):
                    for it in self._f(self._prev_rows, key_consumed):
                        yield it
                else:
                    res = self._f(self._prev_rows, key_consumed)
                    if res:
                        yield res

            self._prev_rows = row

            if self._curr is None or self._curr != key:
                self._curr = key
                self._f = self._func(k)

        def close(self):
            if self._prev_rows and self._curr:
                if self._is_generator_function(self._f):
                    for it in self._f(self._prev_rows, True):
                        yield it
                else:
                    res = self._f(self._prev_rows, True)
                    if res:
                        yield res
            self._prev_rows = None

    return clustered.apply(ActualReducer)


_collection_methods = dict(
    sort_values=sort_values,
    sort=sort_values,
    distinct=distinct,
    apply=apply,
    map_reduce=map_reduce
)

_sequence_methods = dict(
    unique=unique
)

utils.add_method(CollectionExpr, _collection_methods)
utils.add_method(SequenceExpr, _sequence_methods)
