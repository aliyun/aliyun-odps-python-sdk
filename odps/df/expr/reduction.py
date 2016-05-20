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

from .expressions import *
from .groupby import *
from . import utils
from .. import types


class SequenceReduction(Scalar):
    _args = '_input',

    @property
    def node_name(self):
        return self.__class__.__name__

    @property
    def source_name(self):
        return self._input._name

    @property
    def name(self):
        if self._name:
            return self._name
        source_name = self.source_name
        if source_name:
            return '%s_%s' % (source_name, self.node_name.lower())

    def rename(self, new_name):
        if new_name == self._name:
            return self

        attr_values = dict((attr, getattr(self, attr, None))
                           for attr in utils.get_attrs(self))
        attr_values['_name'] = new_name
        new_reduction = type(self)(**attr_values)

        return new_reduction

    def output_type(self):
        return repr(self._value_type)

    def to_grouped_reduction(self, grouped):
        collection = next(n for n in self._input.traverse(top_down=True, unique=True)
                          if isinstance(n, CollectionExpr))
        if collection is not grouped._input:
            raise ExpressionError('Aggregation should be applied to %s, %s instead' % (
                grouped._input, collection))

        cls_name = 'Grouped%s' % self.__class__.__name__
        clz = globals()[cls_name]

        kwargs = dict((arg, getattr(self, arg, None)) for arg in utils.get_attrs(self)
                       if arg != '_cache_args')
        kwargs['_data_type'] = kwargs.pop('_value_type')
        if '_source_value_type' in kwargs:
            kwargs['_source_data_type'] = kwargs.pop('_source_value_type')
        kwargs['_grouped'] = grouped
        del kwargs['_value']

        return clz(**kwargs)

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_reduction(self)


class GroupedSequenceReduction(SequenceExpr):
    __slots__ = '_grouped',
    _args = '_input',

    def _init(self, *args, **kwargs):
        self._init_attr('_grouped', None)
        super(GroupedSequenceReduction, self)._init(*args, **kwargs)
        assert self._grouped is not None

    @property
    def source_name(self):
        return self._source_name or self._input.name

    @property
    def name(self):
        if self._name:
            return self._name
        source_name = self.source_name
        if source_name:
            return '%s_%s' % (source_name, self.node_name.lower())

    def _repr(self):
        expr = self._grouped.agg(self)[self.name]

        return expr._repr()

    @property
    def input(self):
        return self._input

    def accept(self, visitor):
        return visitor.visit_reduction(self)


class Min(SequenceReduction):
    __slots__ = ()


class GroupedMin(GroupedSequenceReduction):
    node_name = 'min'


class Max(SequenceReduction):
    __slots__ = ()


class GroupedMax(GroupedSequenceReduction):
    node_name = 'max'


class Count(SequenceReduction):
    __slots__ = ()


class GroupedCount(GroupedSequenceReduction):
    node_name = 'count'


class Sum(SequenceReduction):
    __slots__ = ()


class GroupedSum(GroupedSequenceReduction):
    node_name = 'sum'


class Var(SequenceReduction):
    __slots__ = '_ddof',


class GroupedVar(GroupedSequenceReduction):
    __slots__ = '_ddof',
    node_name = 'var'


class Std(SequenceReduction):
    __slots__ = '_ddof',


class GroupedStd(GroupedSequenceReduction):
    __slots__ = '_ddof',
    node_name = 'std'


class Mean(SequenceReduction):
    __slots__ = ()


class GroupedMean(GroupedSequenceReduction):
    node_name = 'mean'


class Median(SequenceReduction):
    __slots__ = ()


class GroupedMedian(GroupedSequenceReduction):
    node_name = 'median'


class Any(SequenceReduction):
    __slots__ = ()


class GroupedAny(GroupedSequenceReduction):
    node_name = 'any'


class All(SequenceReduction):
    __slots__ = ()


class GroupedAll(GroupedSequenceReduction):
    node_name = 'all'


class NUnique(SequenceReduction):
    __slots__ = ()


class GroupedNUnique(GroupedSequenceReduction):
    node_name = 'nunique'


class Aggregation(SequenceReduction):
    __slots__ = '_aggregator', '_func_args', '_func_kwargs', '_resources'
    _args = '_input', '_collection_resources'
    node_name = 'Aggregation'

    @property
    def raw_input(self):
        return self._get_attr('_input')

    @property
    def func(self):
        return self._aggregator

    @func.setter
    def func(self, f):
        self._aggregator = f

    def accept(self, visitor):
        visitor.visit_user_defined_aggregator(self)


class GroupedAggregation(GroupedSequenceReduction):
    __slots__ = '_aggregator', '_func_args', '_func_kwargs', '_resources'
    _args = '_input', '_collection_resources'
    node_name = 'Aggregation'

    @property
    def raw_input(self):
        return self._get_attr('_input')

    @property
    def func(self):
        return self._aggregator

    @func.setter
    def func(self, f):
        self._aggregator = f

    def accept(self, visitor):
        visitor.visit_user_defined_aggregator(self)


def _reduction(expr, output_cls, output_type=None, **kw):
    grouped_output_cls = globals()['Grouped%s' % output_cls.__name__]
    method_name = output_cls.__name__.lower()

    if isinstance(expr, CollectionExpr):
        columns = []

        for name in expr.schema.names:
            column = expr[name]
            if hasattr(column, method_name):
                columns.append(getattr(column, method_name)(**kw))

        return expr[columns]
    elif isinstance(expr, GroupBy):
        aggs = []

        for name in expr._to_agg:
            agg = expr[name]
            if hasattr(agg, method_name):
                aggs.append(getattr(agg, method_name)(**kw))

        return expr.agg(aggs)

    if output_type is None:
        output_type = expr._data_type

    if isinstance(expr, SequenceExpr):
        return output_cls(_value_type=output_type, _input=expr, **kw)
    elif isinstance(expr, SequenceGroupBy):
        collection = expr.input.input
        return grouped_output_cls(_data_type=output_type, _input=collection[expr.name],
                                  _grouped=expr.input, **kw)


def min_(expr):
    return _reduction(expr, Min)


def max_(expr):
    return _reduction(expr, Max)


def count(expr):
    if isinstance(expr, SequenceExpr):
        return Count(_value_type=types.int64, _input=expr)
    elif isinstance(expr, SequenceGroupBy):
        collection = expr.input.input
        return GroupedCount(_data_type=types.int64, _input=collection[expr.name])
    elif isinstance(expr, CollectionExpr):
        return Count(_value_type=types.int64, _input=expr).rename('count')
    elif isinstance(expr, GroupBy):
        return GroupedCount(_data_type=types.int64, _input=expr.input,
                            _grouped=expr).rename('count')


def _stats_type(expr):
    if isinstance(expr, (SequenceExpr, SequenceGroupBy)):
        if expr._data_type == types.decimal:
            output_type = types.decimal
        else:
            output_type = types.float64
        return output_type


def var(expr, **kw):
    ddof = kw.get('ddof', kw.get('_ddof', 1))

    output_type = _stats_type(expr)
    return _reduction(expr, Var, output_type, _ddof=ddof)


def sum_(expr):
    output_type = None
    if isinstance(expr, (SequenceExpr, SequenceGroupBy)):
        if expr._data_type == types.boolean:
            output_type = types.int64
        else:
            output_type = expr._data_type
    return _reduction(expr, Sum, output_type)


def std(expr, **kw):
    ddof = kw.get('ddof', kw.get('_ddof', 1))

    output_type = _stats_type(expr)
    return _reduction(expr, Std, output_type, _ddof=ddof)


def mean(expr):
    output_type = _stats_type(expr)
    return _reduction(expr, Mean, output_type)


def median(expr):
    output_type = _stats_type(expr)
    return _reduction(expr, Median, output_type)


def any_(expr):
    output_type = types.boolean
    return _reduction(expr, Any, output_type)


def all_(expr):
    output_type = types.boolean
    return _reduction(expr, All, output_type)


def nunique(expr):
    output_type = types.int64
    return _reduction(expr, NUnique, output_type)


def describe(expr):
    methods = ['min', 'max', 'mean', 'std']

    if isinstance(expr, CollectionExpr):
        fields = []

        for col in expr.schema.columns:
            if types.is_number(col.type):
                fields.append(expr[col.name].notnull().sum().rename(col.name + '_count'))
                for method in methods:
                    kwargs = dict()
                    if method == 'std':
                        kwargs['ddof'] = 0
                    fields.append(getattr(expr[col.name], method)(**kwargs))

        return expr[fields]


def aggregate(expr, aggregator, rtype=None, resources=None, args=(), **kwargs):
    name = None
    if isinstance(aggregator, FunctionWrapper):
        if aggregator.output_names:
            if len(aggregator.output_names) > 1:
                raise ValueError('Aggregate column has more than one name')
            name = aggregator.output_names[0]
        if aggregator.output_types:
            rtype = rtype or aggregator.output_types[0]
        aggregator = aggregator._func

    rtype = rtype or expr.dtype
    output_type = types.validate_data_type(rtype)

    collection_resources = utils.get_collection_resources(resources)
    if isinstance(expr, SequenceGroupBy):
        collection = expr.input.input
        input = collection[expr.name]
        return GroupedAggregation(_input=input, _aggregator=aggregator,
                                  _data_type=output_type, _name=name,
                                  _func_args=args, _func_kwargs=kwargs, _resources=resources,
                                  _collection_resources=collection_resources,
                                  _grouped=expr.input)
    else:
        input = expr
        return Aggregation(_input=input, _aggregator=aggregator,
                           _value_type=output_type, _name=name,
                           _func_args=args, _func_kwargs=kwargs, _resources=resources,
                           _collection_resources=collection_resources)


_number_sequence_methods = dict(
    var=var,
    std=std,
    mean=mean,
    median=median,
    sum=sum_,
)

_sequence_methods = dict(
    min=min_,
    max=max_,
    count=count,
    size=count,
    nunique=nunique,
)

number_sequences = [globals().get(repr(t).capitalize() + SequenceExpr.__name__)
                    for t in types.number_types()]

for number_sequence in number_sequences:
    utils.add_method(number_sequence, _number_sequence_methods)

utils.add_method(SequenceExpr, _sequence_methods)

StringSequenceExpr.sum = sum_
BooleanSequenceExpr.sum = sum_
BooleanSequenceExpr.any = any_
BooleanSequenceExpr.all = all_
SequenceExpr.aggregate = aggregate
SequenceExpr.agg = aggregate

number_sequences_groupby = [globals().get(repr(t).capitalize() + SequenceGroupBy.__name__)
                            for t in types.number_types()]

for number_sequence_groupby in number_sequences_groupby:
    utils.add_method(number_sequence_groupby, _number_sequence_methods)

utils.add_method(SequenceGroupBy, _sequence_methods)

StringSequenceGroupBy.sum = sum_
BooleanSequenceGroupBy.sum = sum_
BooleanSequenceGroupBy.any = any_
BooleanSequenceGroupBy.all = all_
SequenceGroupBy.aggregate = aggregate
SequenceGroupBy.agg = aggregate

# add method to collection expression
utils.add_method(CollectionExpr, _number_sequence_methods)
utils.add_method(CollectionExpr, _sequence_methods)
CollectionExpr.size = count
CollectionExpr.any = any_
CollectionExpr.all = all_

# add method to GroupBy
utils.add_method(GroupBy, _number_sequence_methods)
utils.add_method(GroupBy, _sequence_methods)
GroupBy.size = count
GroupBy.any = any_
GroupBy.all = all_

CollectionExpr.describe = describe