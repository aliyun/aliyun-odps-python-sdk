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

import json
import re
import time
import uuid
from collections import namedtuple

from .expressions import *
from .dynamic import DynamicCollectionExpr
from .arithmetic import Negate
from . import utils
from ..types import validate_data_type, string, DynamicSchema
from ..utils import FunctionWrapper, output
from ...models import Schema
from ...compat import OrderedDict, six, lkeys, lvalues, reduce
from ...utils import str_to_kv


class SortedColumn(SequenceExpr):
    """
    Notice: we do not inherit from the Column
    """

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
            elif isinstance(field, SortedColumn):
                sorted_fields.append(field)
            elif isinstance(field, SequenceExpr):
                column = SortedColumn(_input=field, _name=field.name, _source_name=field.source_name,
                                      _data_type=field._data_type,
                                      _source_data_type=field._source_data_type,
                                      _ascending=self._ascending[i])
                sorted_fields.append(column)
            elif isinstance(field, Scalar):
                column = SortedColumn(_input=field, _name=field.name, _source_name=field.source_name,
                                      _data_type=field._value_type,
                                      _source_data_type=field._source_value_type,
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

    def rebuild(self):
        rebuilt = super(SortedCollectionExpr, self).rebuild()
        rebuilt._schema = self.input.schema
        return rebuilt

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
    __slots__ = '_all',
    _args = '_input', '_unique_fields'
    node_name = 'Distinct'

    def _init(self, *args, **kwargs):
        super(DistinctCollectionExpr, self)._init(*args, **kwargs)

        if self._unique_fields:
            self._unique_fields = list(self._input._get_field(field)
                                       for field in self._unique_fields)

            if not hasattr(self, '_schema'):
                names = [field.name for field in self._unique_fields]
                types = [field._data_type for field in self._unique_fields]
                self._schema = Schema.from_lists(names, types)
        else:
            self._unique_fields = list(self._input._get_field(field)
                                       for field in self._input._schema.names)
            self._schema = self._input._schema

    def iter_args(self):
        for it in zip(['collection', 'distinct'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    def rebuild(self):
        return self._input.distinct(self._unique_fields if not self._all else [])

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

    on = on or list()
    if not isinstance(on, list):
        on = [on, ]
    on = on + list(ons)

    on = [it(expr) if inspect.isfunction(it) else it for it in on]

    return DistinctCollectionExpr(expr, _unique_fields=on, _all=(len(on) == 0))


def unique(expr):
    if isinstance(expr, SequenceExpr):
        collection = next(it for it in expr.traverse(top_down=True, unique=True)
                          if isinstance(it, CollectionExpr))
        return collection.distinct(expr)[expr.name]


class SampledCollectionExpr(CollectionExpr):
    _args = '_input', '_n', '_frac', '_parts', '_i', '_sampled_fields', '_replace', \
            '_weights', '_strata', '_random_state'
    node_name = 'Sample'

    def _init(self, *args, **kwargs):
        for attr in self._args[1:]:
            self._init_attr(attr, None)
        super(SampledCollectionExpr, self)._init(*args, **kwargs)

        if not isinstance(self._n, dict):
            self._n = self._scalar(self._n)
        else:
            self._n = self._scalar(json.dumps(self._n))
        if not isinstance(self._frac, dict):
            self._frac = self._scalar(self._frac)
        else:
            self._frac = self._scalar(json.dumps(self._frac))
        self._parts = self._scalar(self._parts)
        self._i = self._scalar(self._i)
        self._replace = self._scalar(self._replace)
        self._weights = self._scalar(self._weights)
        self._strata = self._scalar(self._strata)
        self._random_state = self._scalar(self._random_state)

    def _scalar(self, val):
        if val is None:
            return
        if isinstance(val, Scalar):
            return val
        if isinstance(val, tuple):
            return tuple(self._scalar(it) for it in val)
        else:
            return Scalar(_value=val)

    @property
    def input(self):
        return self._input

    def rebuild(self):
        rebuilt = super(SampledCollectionExpr, self).rebuild()
        rebuilt._schema = self.input.schema
        return rebuilt

    def accept(self, visitor):
        return visitor.visit_sample(self)


def __df_sample(expr, parts=None, columns=None, i=None, n=None, frac=None, replace=False,
                 weights=None, strata=None, random_state=None):
    if columns:
        columns = expr.select(columns)._fields

    return SampledCollectionExpr(_input=expr, _parts=parts, _i=i, _sampled_fields=columns, _n=n,
                                 _frac=frac, _weights=weights, _strata=strata, _random_state=random_state,
                                 _replace=replace, _schema=expr.schema)


def sample(expr, parts=None, columns=None, i=None, n=None, frac=None, replace=False,
           weights=None, strata=None, random_state=None):
    """
    Sample collection.

    :param expr: collection
    :param parts: how many parts to hash
    :param columns: the columns to sample
    :param i: the part to sample out, can be a list of parts, must be from 0 to parts-1
    :param n: how many rows to sample. If `strata` is specified, `n` should be a dict with values in the strata column as dictionary keys and corresponding sample size as values
    :param frac: how many fraction to sample. If `strata` is specified, `n` should be a dict with values in the strata column as dictionary keys and corresponding sample weight as values
    :param replace: whether to perform replace sampling
    :param weights: the column name of weights
    :param strata: the name of strata column
    :param random_state: the random seed when performing sampling
    :return: collection

    Note that n, frac, replace, weights, strata and random_state can only be used under Pandas DataFrames or
    XFlow.

    :Example:

    Sampling with parts:

    >>> df.sample(parts=1)
    >>> df.sample(parts=5, i=0)
    >>> df.sample(parts=10, columns=['name'])

    Sampling with fraction or weights, replacement option can be specified:

    >>> df.sample(n=100)
    >>> df.sample(frac=100)
    >>> df.sample(frac=100, replace=True)

    Sampling with weight column:

    >>> df.sample(n=100, weights='weight_col')
    >>> df.sample(n=100, weights='weight_col', replace=True)

    Stratified sampling. Note that currently we do not support stratified sampling with replacement.

    >>> df.sample(strata='category', frac={'Iris Setosa': 0.5, 'Iris Versicolour': 0.4})
    """
    if isinstance(expr, CollectionExpr):
        if n is None and frac is None and parts is None:
            raise ExpressionError('Either n or frac or parts should be provided')
        if i is not None and parts is None:
            raise ExpressionError('`parts` arg is required when `i` arg is specified')
        if len([arg for arg in (n, frac, parts) if arg is not None]) > 1:
            raise ExpressionError('You cannot specify `n` or `frac` or `parts` at the same time')
        if strata is None and n is not None and frac is not None:
            # strata can specify different types of strategies on different columns
            raise ExpressionError('You cannot specify `n` and `frac` at the same time.')
        if weights is not None and strata is not None:
            raise ExpressionError('You cannot specify `weights` and `strata` at the same time.')
        if strata is not None:
            if frac is not None and not isinstance(frac, (six.string_types, dict)):
                raise ExpressionError('`frac` should be a k-v string or a dictionary object.')
            if isinstance(frac, six.string_types):
                frac = str_to_kv(frac, float)

            if n is not None and not isinstance(n, (six.string_types, dict)):
                raise ExpressionError('`n` should be a k-v string or a dictionary object.')
            if isinstance(n, six.string_types):
                n = str_to_kv(n, int)

            for val in six.itervalues(frac or dict()):
                if val < 0 or val > 1:
                    raise ExpressionError('Values in `frac` must be between 0 and 1')
            if n is not None and frac is not None:
                collides = set(six.iterkeys(n)).intersection(set(six.iterkeys(frac)))
                if collides:
                    raise ExpressionError('Values in `frac` and `n` collides with each other.')
        else:
            if frac is not None and (not isinstance(frac, (six.integer_types, float)) or frac < 0 or frac > 1):
                raise ExpressionError('`frac` must be between 0 and 1')

        if parts is not None:
            if i is None:
                i = (0, )
            elif isinstance(i, list):
                i = tuple(i)
            elif not isinstance(i, tuple):
                i = (i, )

            for it in i:
                if it >= parts or it < 0:
                    raise ExpressionError('`i` should be positive numbers that less than `parts`')
        elif hasattr(expr, '_xflow_sample'):
            return expr._xflow_sample(n=n, frac=frac, replace=replace, weights=weights, strata=strata,
                                      random_state=random_state)

        return expr.__sample(parts=parts, columns=columns, i=i, n=n, frac=frac, replace=replace,
                             weights=weights, strata=strata, random_state=random_state)


class RowAppliedCollectionExpr(CollectionExpr):
    __slots__ = '_func', '_func_args', '_func_kwargs', '_close_func', \
                '_resources', '_raw_inputs', '_lateral_view', '_keep_nulls'
    _args = '_input', '_fields', '_collection_resources'
    node_name = 'Apply'

    def _init(self, *args, **kwargs):
        self._init_attr('_raw_inputs', None)
        self._init_attr('_lateral_view', False)
        super(RowAppliedCollectionExpr, self)._init(*args, **kwargs)

    @property
    def input(self):
        return self._input

    @property
    def fields(self):
        return self._fields

    @property
    def input_types(self):
        return [f.dtype for f in self._fields]

    @property
    def raw_input_types(self):
        if self._raw_inputs:
            return [f.dtype for f in self._raw_inputs]
        return self.input_types

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, f):
        self._func = f

    def accept(self, visitor):
        return visitor.visit_apply_collection(self)


def _apply_horizontal(expr, func, names=None, types=None, resources=None,
                      collection_resources=None, keep_nulls=False,
                      args=(), **kwargs):
    if isinstance(func, FunctionWrapper):
        names = names or func.output_names
        types = types or func.output_types
        func = func._func

    if names is not None:
        if isinstance(names, list):
            names = tuple(names)
        elif isinstance(names, six.string_types):
            names = (names,)

    if names is None:
        raise ValueError(
            'Apply on rows to provide multiple values should provide all column names, '
            'for instance, df.apply(func, axis=1, names=["A", "B"], types=["float", "float"]). '
            'See https://pyodps.readthedocs.io/zh_CN/latest/df-sort-distinct-apply.html#dfudtfapp '
            'for more information.'
        )
    tps = (string,) * len(names) if types is None else tuple(validate_data_type(t) for t in types)
    schema = Schema.from_lists(names, tps)

    collection_resources = collection_resources or \
                           utils.get_collection_resources(resources)
    return RowAppliedCollectionExpr(_input=expr, _func=func, _func_args=args,
                                    _func_kwargs=kwargs, _schema=schema,
                                    _fields=[expr[n] for n in expr.schema.names],
                                    _keep_nulls=keep_nulls, _resources=resources,
                                    _collection_resources=collection_resources)


def apply(expr, func, axis=0, names=None, types=None, reduce=False,
          resources=None, keep_nulls=False, args=(), **kwargs):
    """
    Apply a function to a row when axis=1 or column when axis=0.

    :param expr:
    :param func: function to apply
    :param axis: row when axis=1 else column
    :param names: output names
    :param types: output types
    :param reduce: if True will return a sequence else return a collection
    :param resources: resources to read
    :param keep_nulls: if True, keep rows producing empty results, only work in lateral views
    :param args: args for function
    :param kwargs: kwargs for function
    :return:

    :Example:

    Apply a function to a row:

    >>> from odps.df import output
    >>>
    >>> @output(['iris_add', 'iris_sub'], ['float', 'float'])
    >>> def handle(row):
    >>>     yield row.sepallength - row.sepalwidth, row.sepallength + row.sepalwidth
    >>>     yield row.petallength - row.petalwidth, row.petallength + row.petalwidth
    >>>
    >>> iris.apply(handle, axis=1).count()


    Apply a function to a column:

    >>> class Agg(object):
    >>>
    >>>     def buffer(self):
    >>>         return [0.0, 0]
    >>>
    >>>     def __call__(self, buffer, val):
    >>>         buffer[0] += val
    >>>         buffer[1] += 1
    >>>
    >>>     def merge(self, buffer, pbuffer):
    >>>         buffer[0] += pbuffer[0]
    >>>         buffer[1] += pbuffer[1]
    >>>
    >>>     def getvalue(self, buffer):
    >>>         if buffer[1] == 0:
    >>>             return 0.0
    >>>         return buffer[0] / buffer[1]
    >>>
    >>> iris.exclude('name').apply(Agg)
    """

    if not isinstance(expr, CollectionExpr):
        return

    if isinstance(func, FunctionWrapper):
        names = names or func.output_names
        types = types or func.output_types
        func = func._func

    if axis == 0:
        types = types or expr.schema.types
        types = [validate_data_type(t) for t in types]

        fields = [expr[n].agg(func, rtype=t, resources=resources)
                  for n, t in zip(expr.schema.names, types)]
        if names:
            fields = [f.rename(n) for f, n in zip(fields, names)]
        else:
            names = [f.name for f in fields]
        return Summary(_input=expr, _fields=fields, _schema=Schema.from_lists(names, types))
    else:
        collection_resources = utils.get_collection_resources(resources)

        if types is not None:
            if isinstance(types, list):
                types = tuple(types)
            elif isinstance(types, six.string_types):
                types = (types,)

            types = tuple(validate_data_type(t) for t in types)
        if reduce:
            from .element import MappedExpr
            from ..backends.context import context

            if names is not None and len(names) > 1:
                raise ValueError('When reduce, at most one name can be specified')
            name = names[0] if names is not None else None
            if not types and kwargs.get('rtype', None) is not None:
                types = [kwargs.pop('rtype')]
            tp = types[0] if types is not None else (utils.get_annotation_rtype(func) or string)
            if not context.is_cached(expr) and (hasattr(expr, '_fields') and expr._fields is not None):
                inputs = [e.copy_tree(stop_cond=lambda x: any(i is expr.input for i in x.children()))
                          for e in expr._fields]
            else:
                inputs = [expr[n] for n in expr.schema.names]
            return MappedExpr(_func=func, _func_args=args, _func_kwargs=kwargs,
                              _name=name, _data_type=tp,
                              _inputs=inputs, _multiple=True,
                              _resources=resources, _collection_resources=collection_resources)
        else:
            return _apply_horizontal(expr, func, names=names, types=types, resources=resources,
                                     collection_resources=collection_resources, keep_nulls=keep_nulls,
                                     args=args, **kwargs)


class ReshuffledCollectionExpr(CollectionExpr):
    _args = '_input', '_by', '_sort_fields'
    node_name = 'Reshuffle'

    def _init(self, *args, **kwargs):
        from .groupby import BaseGroupBy, SortedGroupBy

        self._init_attr('_sort_fields', None)

        super(ReshuffledCollectionExpr, self)._init(*args, **kwargs)

        if isinstance(self._input, BaseGroupBy):
            if isinstance(self._input, SortedGroupBy):
                self._sort_fields = self._input._sorted_fields
            self._by = self._input._by
            self._input = self._input._input

    @property
    def fields(self):
        return self._by + (self._sort_fields or list())

    @property
    def input(self):
        return self._input

    def iter_args(self):
        arg_names = ['collection', 'bys', 'sort']
        for it in zip(arg_names, self.args):
            yield it

    def accept(self, visitor):
        return visitor.visit_reshuffle(self)


def reshuffle(expr, by=None, sort=None, ascending=True):
    """
    Reshuffle data.

    :param expr:
    :param by: the sequence or scalar to shuffle by. RandomScalar as default
    :param sort: the sequence or scalar to sort.
    :param ascending: True if ascending else False
    :return: collection
    """

    by = by or RandomScalar()

    grouped = expr.groupby(by)
    if sort:
        grouped = grouped.sort_values(sort, ascending=ascending)

    return ReshuffledCollectionExpr(_input=grouped, _schema=expr._schema)


def map_reduce(expr, mapper=None, reducer=None, group=None, sort=None, ascending=True,
               combiner=None, combiner_buffer_size=1024,
               mapper_output_names=None, mapper_output_types=None, mapper_resources=None,
               reducer_output_names=None, reducer_output_types=None, reducer_resources=None):
    """
    MapReduce API, mapper or reducer should be provided.

    :param expr:
    :param mapper: mapper function or class
    :param reducer: reducer function or class
    :param group: the keys to group after mapper
    :param sort: the keys to sort after mapper
    :param ascending: True if ascending else False
    :param combiner: combiner function or class, combiner's output should be equal to mapper
    :param combiner_buffer_size: combiner's buffer size, 1024 as default
    :param mapper_output_names: mapper's output names
    :param mapper_output_types: mapper's output types
    :param mapper_resources: the resources for mapper
    :param reducer_output_names: reducer's output names
    :param reducer_output_types: reducer's output types
    :param reducer_resources: the resources for reducer
    :return:

    :Example:

    >>> from odps.df import output
    >>>
    >>> @output(['word', 'cnt'], ['string', 'int'])
    >>> def mapper(row):
    >>>     for word in row[0].split():
    >>>         yield word.lower(), 1
    >>>
    >>> @output(['word', 'cnt'], ['string', 'int'])
    >>> def reducer(keys):
    >>>     cnt = [0]
    >>>     def h(row, done):  # done illustrates that all the rows of the keys are processed
    >>>         cnt[0] += row.cnt
    >>>         if done:
    >>>             yield keys.word, cnt[0]
    >>>     return h
    >>>
    >>> words_df.map_reduce(mapper, reducer, group='word')
    """

    def _adjust_partial(fun):
        if isinstance(fun, functools.partial) and isinstance(fun.func, FunctionWrapper):
            wrapped_fun = fun.func
            partial_fun = functools.partial(wrapped_fun._func, *fun.args, **fun.keywords)
            ret_fun = FunctionWrapper(partial_fun)
            ret_fun.output_names = wrapped_fun.output_names
            ret_fun.output_types = wrapped_fun.output_types
            return ret_fun
        else:
            return fun

    def conv(l):
        if l is None:
            return
        if isinstance(l, tuple):
            l = list(l)
        elif not isinstance(l, list):
            l = [l, ]

        return l

    def gen_name():
        return 'pyodps_field_%s' % str(uuid.uuid4()).replace('-', '_')

    def _gen_actual_reducer(reducer, group):
        class ActualReducer(object):
            def __init__(self, resources=None):
                self._func = reducer
                self._curr = None
                self._prev_rows = None
                self._key_named_tuple = namedtuple('NamedKeys', group)

                self._resources = resources
                self._f = None

            def _is_generator_function(self, f):
                if inspect.isgeneratorfunction(f):
                    return True
                elif hasattr(f, '__call__') and inspect.isgeneratorfunction(f.__call__):
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
                    if self._resources and self._f is None:
                        self._func = self._func(self._resources)
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

        return ActualReducer

    def _gen_combined_mapper(mapper, combiner, names, group, sort, ascending,
                             buffer_size, mapper_resources=None):
        mapper = mapper if not isinstance(mapper, FunctionWrapper) else mapper._func
        sort_indexes = [names.index(s) for s in sort]
        if isinstance(ascending, bool):
            ascending = [ascending] * len(sort)

        class CombinedMapper(object):
            def __init__(self, resources=None):
                if mapper_resources:
                    self.f = mapper(resources)
                elif inspect.isclass(mapper):
                    self.f = mapper()
                else:
                    self.f = mapper
                self.buffer = list()
                if inspect.isfunction(self.f):
                    self.is_generator = inspect.isgeneratorfunction(self.f)
                else:
                    self.is_generator = inspect.isgeneratorfunction(self.f.__call__)

            def _cmp_to_key(self, cmp):
                """Convert a cmp= function into a key= function"""

                class K(object):
                    def __init__(self, obj):
                        self.obj = obj

                    def __lt__(self, other):
                        return cmp(self.obj, other.obj) < 0

                    def __gt__(self, other):
                        return cmp(self.obj, other.obj) > 0

                    def __eq__(self, other):
                        return cmp(self.obj, other.obj) == 0

                    def __le__(self, other):
                        return cmp(self.obj, other.obj) <= 0

                    def __ge__(self, other):
                        return cmp(self.obj, other.obj) >= 0

                    def __ne__(self, other):
                        return cmp(self.obj, other.obj) != 0

                return K

            def _combine(self):
                def cmp(x, y):
                    for asc, sort_idx in zip(ascending, sort_indexes):
                        indict = 1 if asc else -1
                        if x[sort_idx] > y[sort_idx]:
                            return indict * 1
                        elif x[sort_idx] < y[sort_idx]:
                            return indict * -1
                        else:
                            continue

                    return 0

                self.buffer.sort(key=self._cmp_to_key(cmp))

                ActualCombiner = _gen_actual_reducer(combiner, group)
                ac = ActualCombiner()
                named_row = namedtuple('NamedRow', names)
                for r in self.buffer:
                    row = named_row(*r)
                    for l in ac(row):
                        yield l
                for l in ac.close():
                    yield l

                self.buffer = []

            def _handle_output_line(self, line):
                if len(self.buffer) >= buffer_size:
                    for l in self._combine():
                        yield l

                self.buffer.append(line)

            def __call__(self, row):
                if self.is_generator:
                    for it in self.f(row):
                        for l in self._handle_output_line(it):
                            yield l
                else:
                    for l in self._handle_output_line(self.f(row)):
                        yield l

            def close(self):
                if len(self.buffer) > 0:
                    for l in self._combine():
                        yield l

        return CombinedMapper

    mapper = _adjust_partial(mapper)
    reducer = _adjust_partial(reducer)
    combiner = _adjust_partial(combiner)

    if isinstance(mapper, FunctionWrapper):
        mapper_output_names = mapper_output_names or mapper.output_names
        mapper_output_types = mapper_output_types or mapper.output_types

    mapper_output_names = conv(mapper_output_names)
    mapper_output_types = conv(mapper_output_types)

    if mapper_output_types is not None and mapper_output_names is None:
        mapper_output_names = [gen_name() for _ in range(len(mapper_output_types))]

    if mapper is None and mapper_output_names is None:
        mapper_output_names = expr.schema.names

    group = conv(group) or mapper_output_names
    sort = sort or tuple()
    sort = list(OrderedDict.fromkeys(group + conv(sort)))
    if len(sort) > len(group):
        ascending = [ascending, ] * (len(sort) - len(group)) \
            if isinstance(ascending, bool) else list(ascending)
        if len(ascending) != len(sort):
            ascending = [True] * len(group) + ascending

    if not set(group + sort).issubset(mapper_output_names):
        raise ValueError('group and sort have to be the column names of mapper')

    if mapper is None:
        if mapper_output_names and mapper_output_names != expr.schema.names:
            raise ExpressionError(
                'Null mapper cannot have mapper output names: %s' % mapper_output_names)
        if mapper_output_types and mapper_output_types != expr.schema.types:
            raise ExpressionError(
                'Null mapper cannot have mapper output types: %s' % mapper_output_types)
        mapped = expr
        if combiner is not None:
            raise ValueError('Combiner is not null when mapper is null')
    else:
        if combiner is not None:
            if isinstance(combiner, FunctionWrapper):
                if combiner.output_names and \
                        combiner.output_names != mapper_output_names:
                    raise ExpressionError(
                        'Combiner must have the same output names with mapper')
                if combiner.output_types and \
                        combiner.output_types != mapper_output_types:
                    raise ExpressionError(
                        'Combiner must have the same output types with mapper')
                combiner = combiner._func
            mapper = _gen_combined_mapper(mapper, combiner, mapper_output_names,
                                          group, sort, ascending, combiner_buffer_size,
                                          mapper_resources=mapper_resources)
        mapped = expr.apply(mapper, axis=1, names=mapper_output_names,
                            types=mapper_output_types, resources=mapper_resources)

    clustered = mapped.groupby(group).sort(sort, ascending=ascending)

    if isinstance(reducer, FunctionWrapper):
        reducer_output_names = reducer_output_names or reducer.output_names
        reducer_output_types = reducer_output_types or reducer.output_types
        reducer = reducer._func

    if reducer is None:
        if reducer_output_names and reducer_output_names != mapped.schema.names:
            raise ExpressionError(
                'Null reducer cannot have reducer output names %s' % reducer_output_names)
        if reducer_output_types and reducer_output_types != mapped.schema.types:
            raise ExpressionError(
                'Null reducer cannot have reducer output types %s' % reducer_output_types)
        return mapped

    ActualReducer = _gen_actual_reducer(reducer, group)
    return clustered.apply(ActualReducer, resources=reducer_resources,
                           names=reducer_output_names, types=reducer_output_types)


class PivotCollectionExpr(DynamicCollectionExpr):
    _args = '_input', '_group', '_columns', '_values'
    node_name = 'Pivot'

    def _init(self, *args, **kwargs):
        self._init_attr('_group', None)
        self._init_attr('_columns', None)
        self._init_attr('_values', None)

        super(PivotCollectionExpr, self)._init(*args, **kwargs)

        if not hasattr(self, '_schema'):
            self._schema = DynamicSchema.from_lists(
                [f.name for f in self._group], [f.dtype for f in self._group]
            )

    def iter_args(self):
        for it in zip(['collection', 'group', 'columns', 'values'], self.args):
            yield it

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_pivot(self)


def pivot(expr, rows, columns, values=None):
    """
    Produce ‘pivot’ table based on 3 columns of this DataFrame.
    Uses unique values from rows / columns and fills with values.

    :param expr: collection
    :param rows: use to make new collection's grouped rows
    :param columns: use to make new collection's columns
    :param values: values to use for populating new collection's values
    :return: collection

    :Example:

    >>> df.pivot(rows='id', columns='category')
    >>> df.pivot(rows='id', columns='category', values='sale')
    >>> df.pivot(rows=['id', 'id2'], columns='category', values='sale')
    """

    rows = [expr._get_field(r) for r in utils.to_list(rows)]
    columns = [expr._get_field(c) for c in utils.to_list(columns)]
    if values:
        values = utils.to_list(values)
    else:
        names = set(c.name for c in rows + columns)
        values = [n for n in expr.schema.names if n not in names]
        if not values:
            raise ValueError('No values found for pivot')
    values = [expr._get_field(v) for v in values]

    if len(columns) > 1:
        raise ValueError('More than one `columns` are not supported yet')

    return PivotCollectionExpr(_input=expr, _group=rows,
                               _columns=columns, _values=values)


def melt(expr, id_vars=None, value_vars=None, var_name='variable', value_name='value', ignore_nan=False):
    """
    “Unpivots” a DataFrame from wide format to long format, optionally leaving identifier variables set.

    This function is useful to massage a DataFrame into a format where one or more columns are identifier
    variables (id_vars), while all other columns, considered measured variables (value_vars), are “unpivoted”
    to the row axis, leaving just two non-identifier columns, ‘variable’ and ‘value’.

    :param expr: collection
    :param id_vars: column(s) to use as identifier variables.
    :param value_vars: column(s) to unpivot. If not specified, uses all columns that are not set as id_vars.
    :param var_name: name to use for the ‘variable’ column. If None it uses frame.columns.name or ‘variable’.
    :param value_name: name to use for the ‘value’ column.
    :param ignore_nan: whether to ignore NaN values in data.
    :return: collection

    :Example:

    >>> df.melt(id_vars='id', value_vars=['col1', 'col2'])
    >>> df.melt(id_vars=['id', 'id2'], value_vars=['col1', 'col2'], var_name='variable')
    """
    id_vars = id_vars or []
    id_vars = [expr._get_field(r) for r in utils.to_list(id_vars)]
    if not value_vars:
        id_names = set([c.name for c in id_vars])
        value_vars = [expr._get_field(c) for c in expr.schema.names if c not in id_names]
    else:
        value_vars = [expr._get_field(c) for c in value_vars]

    col_type = utils.highest_precedence_data_type(*[c.dtype for c in value_vars])

    col_names = [c.name for c in value_vars]
    id_names = [r.name for r in id_vars]

    names = id_names + [var_name, value_name]
    dtypes = [r.dtype for r in id_vars] + [types.string, col_type]

    @output(names, dtypes)
    def mapper(row):
        for cn in col_names:
            col_value = getattr(row, cn)
            if ignore_nan and col_value is None:
                continue
            vals = [getattr(row, rn) for rn in id_names]
            yield tuple(vals + [cn, col_value])

    return expr.map_reduce(mapper)


class PivotTableCollectionExpr(CollectionExpr):
    __slots__ = '_agg_func', '_agg_func_names'
    _args = '_input', '_group', '_columns', '_values', '_fill_value'
    node_name = 'PivotTable'

    def _init(self, *args, **kwargs):
        for arg in self._args:
            self._init_attr(arg, None)

        super(PivotTableCollectionExpr, self)._init(*args, **kwargs)

        for attr in ('_fill_value', ):
            val = getattr(self, attr, None)
            if val is not None and not isinstance(val, Scalar):
                setattr(self, attr, Scalar(_value=val))

    @property
    def input(self):
        return self._input

    @property
    def fill_value(self):
        if self._fill_value:
            return self._fill_value.value

    @property
    def margins(self):
        return self._margins.value

    @property
    def margins_name(self):
        return self._margins_name.value

    def accept(self, visitor):
        return visitor.visit_pivot(self)


def pivot_table(expr, values=None, rows=None, columns=None, aggfunc='mean',
                fill_value=None):
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    :param expr: collection
    :param values (optional): column to aggregate
    :param rows: rows to group
    :param columns: keys to group by on the pivot table column
    :param aggfunc: aggregate function or functions
    :param fill_value (optional): value to replace missing value with, default None
    :return: collection

    :Example:
    >>> df
        A    B      C   D
    0  foo  one  small  1
    1  foo  one  large  2
    2  foo  one  large  2
    3  foo  two  small  3
    4  foo  two  small  3
    5  bar  one  large  4
    6  bar  one  small  5
    7  bar  two  small  6
    8  bar  two  large  7
    >>> table = df.pivot_table(values='D', rows=['A', 'B'], columns='C', aggfunc='sum')
    >>> table
         A    B  large_D_sum   small_D_sum
    0  bar  one          4.0           5.0
    1  bar  two          7.0           6.0
    2  foo  one          4.0           1.0
    3  foo  two          NaN           6.0
    """

    def get_names(iters):
        return [r if isinstance(r, six.string_types) else r.name
                for r in iters]

    def get_aggfunc_name(f):
        if isinstance(f, six.string_types):
            if '(' in f:
                f = re.sub(r' *\( *', '_', f)
                f = re.sub(r' *[+\-\*/,] *', '_', f)
                f = re.sub(r' *\) *', '', f)
                f = f.replace('.', '_')
            return f
        if isinstance(f, FunctionWrapper):
            return f.output_names[0]
        return 'aggregation'

    if not rows:
        raise ValueError('No group keys passed')
    rows = utils.to_list(rows)
    rows_names = get_names(rows)
    rows = [expr._get_field(r) for r in rows]

    if isinstance(aggfunc, dict):
        agg_func_names = lkeys(aggfunc)
        aggfunc = lvalues(aggfunc)
    else:
        aggfunc = utils.to_list(aggfunc)
        agg_func_names = [get_aggfunc_name(af) for af in aggfunc]

    if not columns:
        if values is None:
            values = [n for n in expr.schema.names if n not in rows_names]
        else:
            values = utils.to_list(values)
        values = [expr._get_field(v) for v in values]

        names = rows_names
        types = [r.dtype for r in rows]
        for func, func_name in zip(aggfunc, agg_func_names):
            for value in values:
                if isinstance(func, six.string_types):
                    seq = value.eval(func, rewrite=False)
                    if isinstance(seq, ReprWrapper):
                        seq = seq()
                else:
                    seq = value.agg(func)
                seq = seq.rename('{0}_{1}'.format(value.name, func_name))
                names.append(seq.name)
                types.append(seq.dtype)
        schema = Schema.from_lists(names, types)

        return PivotTableCollectionExpr(_input=expr, _group=rows, _values=values,
                                        _fill_value=fill_value, _schema=schema,
                                        _agg_func=aggfunc, _agg_func_names=agg_func_names)
    else:
        columns = [expr._get_field(c) for c in utils.to_list(columns)]

        if values:
            values = utils.to_list(values)
        else:
            names = set(c.name for c in rows + columns)
            values = [n for n in expr.schema.names if n not in names]
            if not values:
                raise ValueError('No values found for pivot_table')
        values = [expr._get_field(v) for v in values]

        if len(columns) > 1:
            raise ValueError('More than one `columns` are not supported yet')

        schema = DynamicSchema.from_lists(rows_names, [r.dtype for r in rows])
        base_tp = PivotTableCollectionExpr
        tp = type(base_tp.__name__, (DynamicCollectionExpr, base_tp), dict())
        return tp(_input=expr, _group=rows, _values=values,
                  _columns=columns, _agg_func=aggfunc,
                  _fill_value=fill_value, _schema=schema,
                  _agg_func_names=agg_func_names)


def _scale_values(expr, columns, agg_fun, scale_fun, preserve=False, suffix='_scaled', group=None):
    from ..types import Float, Integer
    time_suffix = str(int(time.time()))

    if group is not None:
        group = utils.to_list(group)
        group = [expr._get_field(c).name if isinstance(c, Column) else c for c in group]

    if columns is None:
        if group is None:
            columns = expr.schema.names
        else:
            columns = [n for n in expr.schema.names if n not in group]
    else:
        columns = utils.to_list(columns)
    columns = [expr._get_field(v) for v in columns]

    numerical_cols = [col.name for col in columns if isinstance(col.data_type, (Float, Integer))]

    agg_cols = []
    for col_name in numerical_cols:
        agg_cols.extend(agg_fun(expr, col_name))

    if group is None:
        # make a fake constant column to join
        extra_col = 'idx_col_' + time_suffix
        join_cols = [extra_col]
        stats_df = expr.__getitem__([Scalar(1).rename(extra_col)] + agg_cols)
        mapped = expr[expr, Scalar(1).rename(extra_col)]
    else:
        extra_col = None
        join_cols = group
        stats_df = expr.groupby(join_cols).agg(*agg_cols)
        mapped = expr

    joined = mapped.join(stats_df, on=join_cols, mapjoin=True)
    if extra_col is not None:
        joined = joined.exclude(extra_col)

    if preserve:
        norm_cols = [dt.name for dt in expr.dtypes]
        norm_cols.extend([scale_fun(joined, dt.name).rename(dt.name + suffix)
                          for dt in expr.dtypes if dt.name in numerical_cols])
    else:
        norm_cols = [scale_fun(joined, dt.name).rename(dt.name)
                     if dt.name in numerical_cols else getattr(joined, dt.name)
                     for dt in expr.dtypes]
    return joined.__getitem__(norm_cols)


def min_max_scale(expr, columns=None, feature_range=(0, 1), preserve=False, suffix='_scaled', group=None):
    """
    Resize a data frame by max / min values, i.e., (X - min(X)) / (max(X) - min(X))

    :param DataFrame expr: input DataFrame
    :param feature_range: the target range to resize the value into, i.e., v * (b - a) + a
    :param bool preserve: determine whether input data should be kept. If True, scaled input data will be appended to the data frame with `suffix`
    :param columns: columns names to resize. If set to None, float or int-typed columns will be normalized if the column is not specified as a group column.
    :param group: determine scale groups. Scaling will be done in each group separately.
    :param str suffix: column suffix to be appended to the scaled columns.

    :return: resized data frame
    :rtype: DataFrame
    """
    time_suffix = str(int(time.time()))

    def calc_agg(expr, col):
        return [
            getattr(expr, col).min().rename(col + '_min_' + time_suffix),
            getattr(expr, col).max().rename(col + '_max_' + time_suffix),
        ]

    def do_scale(expr, col):
        f_min, f_max = feature_range
        r = getattr(expr, col + '_max_' + time_suffix) - getattr(expr, col + '_min_' + time_suffix)
        scaled = (r == 0).ifelse(Scalar(0), (getattr(expr, col) - getattr(expr, col + '_min_' + time_suffix)) / r)
        return scaled * (f_max - f_min) + f_min

    return _scale_values(expr, columns, calc_agg, do_scale, preserve=preserve, suffix=suffix, group=group)


def std_scale(expr, columns=None, with_means=True, with_std=True, preserve=False, suffix='_scaled', group=None):
    """
    Resize a data frame by mean and standard error.

    :param DataFrame expr: Input DataFrame
    :param bool with_means: Determine whether the output will be subtracted by means
    :param bool with_std: Determine whether the output will be divided by standard deviations
    :param bool preserve: Determine whether input data should be kept. If True, scaled input data will be appended to the data frame with `suffix`
    :param columns: Columns names to resize. If set to None, float or int-typed columns will be normalized if the column is not specified as a group column.
    :param group: determine scale groups. Scaling will be done in each group separately.
    :param str suffix: column suffix to be appended to the scaled columns.

    :return: resized data frame
    :rtype: DataFrame
    """
    time_suffix = str(int(time.time()))

    def calc_agg(expr, col):
        return [
            getattr(expr, col).mean().rename(col + '_mean_' + time_suffix),
            getattr(expr, col).std(ddof=0).rename(col + '_std_' + time_suffix),
        ]

    def do_scale(expr, col):
        c = getattr(expr, col)
        mean_expr = getattr(expr, col + '_mean_' + time_suffix)
        if with_means:
            c = c - mean_expr
            mean_expr = Scalar(0)
        if with_std:
            std_expr = getattr(expr, col + '_std_' + time_suffix)
            c = (std_expr == 0).ifelse(mean_expr, c / getattr(expr, col + '_std_' + time_suffix))
        return c

    return _scale_values(expr, columns, calc_agg, do_scale, preserve=preserve, suffix=suffix, group=group)


class ExtractKVCollectionExpr(DynamicCollectionExpr):
    __slots__ = '_column_type',
    _args = '_input', '_columns', '_intact', '_kv_delimiter', '_item_delimiter', '_default'
    node_name = 'ExtractKV'

    def _init(self, *args, **kwargs):
        from .element import _scalar
        for attr in self._args[1:]:
            self._init_attr(attr, None)
        super(ExtractKVCollectionExpr, self)._init(*args, **kwargs)
        self._kv_delimiter = _scalar(self._kv_delimiter)
        self._item_delimiter = _scalar(self._item_delimiter)
        self._default = _scalar(self._default)

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        visitor.visit_extract_kv(self)


def extract_kv(expr, columns=None, kv_delim=':', item_delim=',', dtype='float', fill_value=None):
    """
    Extract values in key-value represented columns into standalone columns. New column names will
    be the name of the key-value column followed by an underscore and the key.

    :param DataFrame expr: input DataFrame
    :param columns: the key-value columns to be extracted.
    :param str kv_delim: delimiter between key and value.
    :param str item_delim: delimiter between key-value pairs.
    :param str dtype: type of value columns to generate.
    :param fill_value: default value for missing key-value pairs.

    :return: extracted data frame
    :rtype: DataFrame

    :Example:
    >>> df
        name   kv
    0  name1  k1=1.0,k2=3.0,k5=10.0
    1  name2  k2=3.0,k3=5.1
    2  name3  k1=7.1,k7=8.2
    3  name4  k2=1.2,k3=1.5
    4  name5  k2=1.0,k9=1.1
    >>> table = df.extract_kv(columns=['A', 'B'], kv_delim='=')
    >>> table
        name   kv_k1   kv_k2   kv_k3   kv_k5   kv_k7   kv_k9
    0  name1  1.0     3.0     Nan     10.0    Nan     Nan
    1  name2  Nan     3.0     5.1     Nan     Nan     Nan
    2  name3  7.1     Nan     Nan     Nan     8.2     Nan
    3  name4  Nan     1.2     1.5     Nan     Nan     Nan
    4  name5  Nan     1.0     Nan     Nan     Nan     1.1
    """
    if columns is None:
        columns = [expr._get_field(c) for c in expr.schema.names]
        intact_cols = []
    else:
        columns = [expr._get_field(c) for c in utils.to_list(columns)]
        name_set = set([c.name for c in columns])
        intact_cols = [expr._get_field(c) for c in expr.schema.names if c not in name_set]

    column_type = types.validate_data_type(dtype)
    if any(not isinstance(c.dtype, types.String) for c in columns):
        raise ExpressionError('Key-value columns must be strings.')

    schema = DynamicSchema.from_lists([c.name for c in intact_cols], [c.dtype for c in intact_cols])
    return ExtractKVCollectionExpr(_input=expr, _columns=columns, _intact=intact_cols, _schema=schema,
                                   _column_type=column_type, _default=fill_value,
                                   _kv_delimiter=kv_delim, _item_delimiter=item_delim)


def to_kv(expr, columns=None, kv_delim=':', item_delim=',', kv_name='kv_col'):
    """
    Merge values in specified columns into a key-value represented column.

    :param DataFrame expr: input DataFrame
    :param columns: the columns to be merged.
    :param str kv_delim: delimiter between key and value.
    :param str item_delim: delimiter between key-value pairs.
    :param str kv_col: name of the new key-value column

    :return: converted data frame
    :rtype: DataFrame

    :Example:
    >>> df
        name   k1   k2   k3   k5    k7   k9
    0  name1  1.0  3.0  Nan  10.0  Nan  Nan
    1  name2  Nan  3.0  5.1  Nan   Nan  Nan
    2  name3  7.1  Nan  Nan  Nan   8.2  Nan
    3  name4  Nan  1.2  1.5  Nan   Nan  Nan
    4  name5  Nan  1.0  Nan  Nan   Nan  1.1
    >>> table = df.to_kv(columns=['A', 'B'], kv_delim='=')
    >>> table
        name   kv_col
    0  name1  k1=1.0,k2=3.0,k5=10.0
    1  name2  k2=3.0,k3=5.1
    2  name3  k1=7.1,k7=8.2
    3  name4  k2=1.2,k3=1.5
    4  name5  k2=1.0,k9=1.1
    """
    if columns is None:
        columns = [expr._get_field(c) for c in expr.schema.names]
        intact_cols = []
    else:
        columns = [expr._get_field(c) for c in utils.to_list(columns)]
        name_set = set([c.name for c in columns])
        intact_cols = [expr._get_field(c) for c in expr.schema.names if c not in name_set]

    mapped_cols = [c.isnull().ifelse(Scalar(''), c.name + kv_delim + c.astype('string')) for c in columns]
    reduced_col = reduce(lambda a, b: (b == '').ifelse(a, (a == '').ifelse(b, a + item_delim + b)), mapped_cols)
    return expr.__getitem__(intact_cols + [reduced_col.rename(kv_name)])


def dropna(expr, how='any', thresh=None, subset=None):
    """
    Return object with labels on given axis omitted where alternately any or all of the data are missing

    :param DataFrame expr: input DataFrame
    :param how: can be ‘any’ or ‘all’. If 'any' is specified any NA values are present, drop that label. If 'all' is specified and all values are NA, drop that label.
    :param thresh: require that many non-NA values
    :param subset: Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include
    :return: DataFrame
    """
    if subset is None:
        subset = [expr._get_field(c) for c in expr.schema.names]
    else:
        subset = [expr._get_field(c) for c in utils.to_list(subset)]

    if not subset:
        raise ValueError('Illegal subset is provided.')

    if thresh is None:
        thresh = len(subset) if how == 'any' else 1

    sum_exprs = reduce(operator.add, (s.notnull().ifelse(1, 0) for s in subset))
    return expr.filter(sum_exprs >= thresh)


def fillna(expr, value=None, method=None, subset=None):
    """
    Fill NA/NaN values using the specified method

    :param DataFrame expr: input DataFrame
    :param method: can be ‘backfill’, ‘bfill’, ‘pad’, ‘ffill’ or None
    :param value: value to fill into
    :param subset: Labels along other axis to consider.
    :return: DataFrame
    """
    col_dict = OrderedDict([(c, expr._get_field(c)) for c in expr.schema.names])
    if subset is None:
        sel_col_names = expr.schema.names
    else:
        # when c is in expr._fields, _get_field may do substitution which will cause error
        subset = (c.copy() if isinstance(c, Expr) else c for c in utils.to_list(subset))
        sel_col_names = [expr._get_field(c).name for c in subset]

    if method is not None and value is not None:
        raise ValueError('The argument `method` is not compatible with `value`.')
    if method is None and value is None:
        raise ValueError('You should supply at least one argument in `method` and `value`.')
    if method is not None and method not in ('backfill', 'bfill', 'pad', 'ffill'):
        raise ValueError('Method value %s is illegal.' % str(method))

    if method in ('backfill', 'bfill'):
        sel_cols = list(reversed(sel_col_names))
    else:
        sel_cols = sel_col_names

    if method is None:
        for n in sel_col_names:
            e = col_dict[n]
            col_dict[n] = e.isnull().ifelse(value, e).rename(n)
        return expr.select(list(col_dict.values()))

    else:
        names = list(col_dict.keys())
        typs = list(c.dtype.name for c in col_dict.values())

        @output(names, typs)
        def mapper(row):
            last_valid = None
            update_dict = dict()

            try:
                import numpy as np

                def isnan(v):
                    try:
                        return np.isnan(v)
                    except TypeError:
                        return False

            except ImportError:
                isnan = lambda v: False

            for n in sel_cols:
                old_val = getattr(row, n)
                if old_val is None or isnan(old_val):
                    if last_valid is not None:
                        update_dict[n] = last_valid
                else:
                    last_valid = old_val

            yield row.replace(**update_dict)

        return expr.map_reduce(mapper)


def ffill(expr, subset=None):
    """
    Fill NA/NaN values with the forward method. Equivalent to fillna(method='ffill').

    :param DataFrame expr: input DataFrame.
    :param subset: Labels along other axis to consider.
    :return: DataFrame
    """
    return expr.fillna(method='ffill', subset=subset)


def bfill(expr, subset=None):
    """
    Fill NA/NaN values with the backward method. Equivalent to fillna(method='bfill').

    :param DataFrame expr: input DataFrame.
    :param subset: Labels along other axis to consider.
    :return: DataFrame
    """
    return expr.fillna(method='bfill', subset=subset)


class AppendIDCollectionExpr(CollectionExpr):
    _args = '_input', '_id_col'
    node_name = 'AppendID'

    def _init(self, *args, **kwargs):
        from .element import _scalar
        for attr in self._args[1:]:
            self._init_attr(attr, None)
        super(AppendIDCollectionExpr, self)._init(*args, **kwargs)
        self._validate()
        self._id_col = _scalar(self._id_col)
        self._schema = Schema.from_lists(self._input.schema.names + [self._id_col.value],
                                         self._input.schema.types + [types.int64])

    def _validate(self):
        if self._id_col in self._input.schema:
            raise ExpressionError('ID column already exists in current data frame.')

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_append_id(self)


def _append_id(expr, id_col='append_id'):
    return AppendIDCollectionExpr(_input=expr, _id_col=id_col)


def append_id(expr, id_col='append_id'):
    """
    Append an ID column to current column to form a new DataFrame.

    :param str id_col: name of appended ID field.

    :return: DataFrame with ID field
    :rtype: DataFrame
    """
    if hasattr(expr, '_xflow_append_id'):
        return expr._xflow_append_id(id_col)
    else:
        return _append_id(expr, id_col)


class SplitCollectionExpr(CollectionExpr):
    _args = '_input', '_frac', '_seed', '_split_id'
    node_name = 'Split'

    def _init(self, *args, **kwargs):
        from .element import _scalar
        for attr in self._args[1:]:
            self._init_attr(attr, None)
        super(SplitCollectionExpr, self)._init(*args, **kwargs)
        self._frac = _scalar(self._frac)
        self._seed = _scalar(self._seed, types.int32)
        self._split_id = _scalar(self._split_id, types.int32)
        self._schema = self._input.schema

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_split(self)


def _split(expr, frac, seed=None):
    seed = seed or int(time.time())
    return (
        SplitCollectionExpr(_input=expr, _frac=frac, _seed=seed, _split_id=0),
        SplitCollectionExpr(_input=expr, _frac=frac, _seed=seed, _split_id=1),
    )


def split(expr, frac, seed=None):
    """
    Split the current column into two column objects with certain ratio.

    :param float frac: Split ratio

    :return: two split DataFrame objects
    """
    if hasattr(expr, '_xflow_split'):
        return expr._xflow_split(frac, seed=seed)
    else:
        return _split(expr, frac, seed=seed)


def applymap(expr, func, rtype=None, resources=None, columns=None, excludes=None, args=(), **kwargs):
    """
    Call func on each element of this collection.

    :param func: lambda, function, :class:`odps.models.Function`,
                 or str which is the name of :class:`odps.models.Funtion`
    :param rtype: if not provided, will be the dtype of this sequence
    :param columns: columns to apply this function on
    :param excludes: columns to skip when applying the function
    :return: a new collection

    :Example:

    >>> df.applymap(lambda x: x + 1)
    """
    if columns is not None and excludes is not None:
        raise ValueError('`columns` and `excludes` cannot be provided at the same time.')
    if not columns:
        excludes = excludes or []
        if isinstance(excludes, six.string_types):
            excludes = [excludes]
        excludes = set([c if isinstance(c, six.string_types) else c.name for c in excludes])
        columns = set([c for c in expr.schema.names if c not in excludes])
    else:
        if isinstance(columns, six.string_types):
            columns = [columns]
        columns = set([c if isinstance(c, six.string_types) else c.name for c in columns])
    mapping = [expr[c] if c not in columns
               else expr[c].map(func, rtype=rtype, resources=resources, args=args, **kwargs)
               for c in expr.schema.names]
    return expr.select(*mapping)


_collection_methods = dict(
    sort_values=sort_values,
    sort=sort_values,
    distinct=distinct,
    apply=apply,
    reshuffle=reshuffle,
    map_reduce=map_reduce,
    sample=sample,
    __sample=__df_sample,
    pivot=pivot,
    melt=melt,
    pivot_table=pivot_table,
    extract_kv=extract_kv,
    to_kv=to_kv,
    dropna=dropna,
    fillna=fillna,
    ffill=ffill,
    bfill=bfill,
    min_max_scale=min_max_scale,
    std_scale=std_scale,
    _append_id=_append_id,
    append_id=append_id,
    _split=_split,
    split=split,
    applymap=applymap,
)

_sequence_methods = dict(
    unique=unique
)

utils.add_method(CollectionExpr, _collection_methods)
utils.add_method(SequenceExpr, _sequence_methods)
