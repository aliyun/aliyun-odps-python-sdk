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

import os
import uuid

from ..expr.expressions import SequenceExpr, Column, CollectionExpr
from ..expr.groupby import SequenceGroupBy
from ..utils import output
from .. import types
from ...compat import six


path = os.path.dirname(os.path.abspath(__file__))


def hll_count(expr, error_rate=0.01, splitter=None):
    """
    Calculate HyperLogLog count

    :param expr:
    :param error_rate: error rate
    :type error_rate: float
    :param splitter: the splitter to split the column value
    :return: sequence or scalar

    :Example:

    >>> df = DataFrame(pd.DataFrame({'a': np.random.randint(100000, size=100000)}))
    >>> df.a.hll_count()
    63270
    >>> df.a.nunique()
    63250
    """

    # to make the class pickled right by the cloudpickle
    with open(os.path.join(path, 'lib', 'hll.py')) as hll_file:
        local = {}
        six.exec_(hll_file.read(), local)
        HyperLogLog = local['HyperLogLog']

        return expr.agg(HyperLogLog, rtype=types.int64, args=(error_rate, splitter))


def bloomfilter(collection, on, column, capacity=3000, error_rate=0.01):
    """
    Filter collection on the `on` sequence by BloomFilter built by `column`

    :param collection:
    :param on: sequence or column name
    :param column: instance of Column
    :param capacity: numbers of capacity
    :type capacity: int
    :param error_rate: error rate
    :type error_rate: float
    :return: collection

    :Example:

    >>> df1 = DataFrame(pd.DataFrame({'a': ['name1', 'name2', 'name3', 'name1'], 'b': [1, 2, 3, 4]}))
    >>> df2 = DataFrame(pd.DataFrame({'a': ['name1']}))
    >>> df1.bloom_filter('a', df2.a)
           a  b
    0  name1  1
    1  name1  4
    """


    if not isinstance(column, Column):
        raise TypeError('bloomfilter can only filter on the column of a collection')

    # to make the class pickled right by the cloudpickle
    with open(os.path.join(path, 'lib', 'bloomfilter.py')) as bloomfilter_file:
        local = {}
        six.exec_(bloomfilter_file.read(), local)
        BloomFilter = local['BloomFilter']

        col_name = column.source_name or column.name

        on_name = on.name if isinstance(on, SequenceExpr) else on
        rand_name = '%s_%s'% (on_name, str(uuid.uuid4()).replace('-', '_'))
        on_col = collection._get_field(on).rename(rand_name)
        src_collection = collection
        collection = collection[collection, on_col]

        @output(src_collection.schema.names, src_collection.schema.types)
        class Filter(object):
            def __init__(self, resources):
                table = resources[0]

                bloom = BloomFilter(capacity, error_rate)
                for row in table:
                    bloom.add(str(getattr(row, col_name)))

                self.bloom = bloom

            def __call__(self, row):
                if str(getattr(row, rand_name)) not in self.bloom:
                    return
                return row[:-1]

        return collection.apply(Filter, axis=1, resources=[column.input, ])


SequenceExpr.hll_count = hll_count
SequenceGroupBy.hll_count = hll_count
CollectionExpr.bloom_filter = bloomfilter

__all__ = ('bloomfilter', 'hll_count')
