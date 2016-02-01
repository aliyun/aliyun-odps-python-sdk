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

from odps.df.expr.arithmetic import Negate
from ...models import Schema
from .expressions import *
from . import utils


class SortedColumn(SequenceExpr):
    __slots__ = '_ascending',
    _args = '_input',

    @property
    def input(self):
        return self._input


class SortedExpr(Expr):
    def __init__(self, *args, **kwargs):
        super(SortedExpr, self).__init__(*args, **kwargs)

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
                assert isinstance(field, six.string_types)
                sorted_fields.append(
                    SortedColumn(self._input, _name=field,
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

    def __init__(self, *args, **kwargs):
        super(DistinctCollectionExpr, self).__init__(*args, **kwargs)

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


class DistinctSequenceExpr(SequenceExpr):
    _args = '_input',

    @property
    def name(self):
        return self._name or self._input.name

    @property
    def source_name(self):
        return self._source_name or self._input.name


def unique(expr):
    if isinstance(expr, SequenceExpr):
        return DistinctSequenceExpr(_input=expr, _name=expr._name,
                                    _data_type=expr._data_type)


_collection_methods = dict(
    sort_values=sort_values,
    sort=sort_values,
    distinct=distinct
)

_sequence_methods = dict(
    unique=unique
)

utils.add_method(CollectionExpr, _collection_methods)
utils.add_method(SequenceExpr, _sequence_methods)
