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

from ..expr.expressions import *
from ..expr.groupby import GroupBy, SequenceGroupBy
from ..expr.reduction import Count, GroupedSequenceReduction


class Backend(object):

    def visit_source_collection(self, expr):
        raise NotImplementedError

    def visit_project_collection(self, expr):
        raise NotImplementedError

    def visit_apply_collection(self, expr):
        raise NotImplementedError

    def visit_filter_collection(self, expr):
        raise NotImplementedError

    def visit_slice_collection(self, expr):
        raise NotImplementedError

    def visit_element_op(self, expr):
        raise NotImplementedError

    def visit_binary_op(self, expr):
        raise NotImplementedError

    def visit_unary_op(self, expr):
        raise NotImplementedError

    def visit_math(self, expr):
        raise NotImplementedError

    def visit_string_op(self, expr):
        raise NotImplementedError

    def visit_datetime_op(self, expr):
        raise NotImplementedError

    def visit_groupby(self, expr):
        raise NotImplementedError

    def visit_mutate(self, expr):
        raise NotImplementedError

    def visit_value_counts(self, expr):
        raise NotImplementedError

    def visit_sort(self, expr):
        raise NotImplementedError

    def visit_sort_column(self, expr):
        raise NotImplementedError

    def visit_distinct(self, expr):
        raise NotImplementedError

    def visit_sample(self, expr):
        raise NotImplementedError

    def visit_reduction(self, expr):
        raise NotImplementedError

    def visit_user_defined_aggregator(self, expr):
        raise NotImplementedError

    def visit_column(self, expr):
        raise NotImplementedError

    def visit_function(self, expr):
        raise NotImplementedError

    def visit_builtin_function(self, expr):
        raise NotImplementedError

    def visit_sequence(self, expr):
        raise NotImplementedError

    def visit_cum_window(self, expr):
        raise NotImplementedError

    def visit_rank_window(self, expr):
        raise NotImplementedError

    def visit_shift_window(self, expr):
        raise NotImplementedError

    def visit_scalar(self, expr):
        raise NotImplementedError

    def visit_join(self, expr):
        raise NotImplementedError

    def visit_cast(self, expr):
        raise NotImplementedError

    def visit_union(self, expr):
        raise NotImplementedError


class Engine(object):
    def _convert_table(self, expr):
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

    def compile(self, expr):
        raise NotImplementedError

    def execute(self, expr):
        raise NotImplementedError

    def persist(self, expr, name, partitions=None, bar=None, **kwargs):
        raise NotImplementedError
