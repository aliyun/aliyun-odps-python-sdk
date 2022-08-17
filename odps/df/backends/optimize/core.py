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

import itertools

from ..core import Backend
from ...expr.core import ExprDictionary
from ...expr.expressions import *
from ...expr.groupby import GroupByCollectionExpr
from ...expr.reduction import SequenceReduction, GroupedSequenceReduction
from ...expr.merge import JoinCollectionExpr
from ...expr.window import Window
from ...expr.utils import select_fields
from ...utils import traverse_until_source
from .... import utils
from .columnpruning import ColumnPruning
from .predicatepushdown import PredicatePushdown
from .utils import change_input


class Optimizer(Backend):
    def __init__(self, dag):
        self._dag = dag

    def optimize(self):
        if options.df.optimize:
            if options.df.optimizes.cp:
                ColumnPruning(self._dag).prune()
            if options.df.optimizes.pp:
                PredicatePushdown(self._dag).pushdown()

        for node in traverse_until_source(self._dag, top_down=True):
            try:
                node.accept(self)
            except NotImplementedError:
                continue

        # from down up do again
        for node in traverse_until_source(self._dag):
            try:
                node.accept(self)
            except NotImplementedError:
                continue

        return self._dag.root

    def _sub(self, expr, to_sub, parents=None):
        self._dag.substitute(expr, to_sub, parents=parents)

    def visit_filter_collection(self, expr):
        if not options.df.optimize:
            return

        if isinstance(expr.input, GroupByCollectionExpr) and \
                not expr.input.optimize_banned:
            # move filter on GroupBy to GroupBy's having
            grouped = expr.input
            predicate = self._do_compact(expr.predicate, expr.input)
            if predicate is None:
                predicate = expr.predicate

            having = grouped.having
            if having is not None:
                predicates = having & predicate
            else:
                predicates = predicate

            grouped._having = predicates
            self._sub(expr, grouped)
        elif isinstance(expr.input, FilterCollectionExpr):
            filters = [expr]
            node = expr.input
            while isinstance(node, FilterCollectionExpr):
                filters.append(node)
                node = node.input
            self._compact_filters(*filters)

        raise NotImplementedError

    @classmethod
    def get_compact_filters(cls, dag, *filters):
        input = filters[-1].input

        get_field = lambda n, col: input[col]
        for filter in filters:
            change_input(filter, filter.input, input, get_field, dag)

        predicate = reduce(operator.and_, [f.predicate for f in filters[::-1]])
        return FilterCollectionExpr(input, predicate, _schema=input.schema)

    def _compact_filters(self, *filters):
        new_filter = self.get_compact_filters(self._dag, *filters)
        self._sub(filters[0], new_filter)

    def visit_project_collection(self, expr):
        # Summary does not attend here
        if not options.df.optimize:
            return

        compacted = self._visit_need_compact_collection(expr)
        if compacted:
            expr = compacted

        if isinstance(expr, ProjectCollectionExpr) and \
                isinstance(expr.input, GroupByCollectionExpr) and \
                not expr.input.optimize_banned:
            # compact projection into Groupby
            grouped = expr.input

            selects = []

            for n in expr.traverse(top_down=True, unique=True,
                                   stop_cond=lambda x: x is grouped):
                # stop compact
                if isinstance(n, (Window, SequenceReduction)):
                    return

            for field in expr._fields:
                selects.append(self._do_compact(field, grouped) or field)

            grouped._aggregations = grouped._fields = selects
            grouped._schema = Schema.from_lists([f.name for f in selects],
                                                [f.dtype for f in selects])
            self._sub(expr, grouped)

            return

    def visit_groupby(self, expr):
        if not options.df.optimize:
            return

        # we do not do compact on the projections from Join
        input = expr.input
        while isinstance(input, ProjectCollectionExpr):
            input = input._input
        if isinstance(input, JoinCollectionExpr):
            return

        if len(expr._aggregations) == 1 and \
                isinstance(expr._aggregations[0], GroupedSequenceReduction) and \
                isinstance(expr._aggregations[0]._input, CollectionExpr):
            # we just skip the case: df.groupby(***).count()
            return

        self._visit_need_compact_collection(expr)

    def visit_distinct(self, expr):
        if not options.df.optimize:
            return
        self._visit_need_compact_collection(expr)

    def visit_apply_collection(self, expr):
        if not options.df.optimize:
            return
        if expr._lateral_view:
            return
        if (
            isinstance(expr.input, JoinCollectionExpr)
            and (expr.input._mapjoin or expr.input._skewjoin)
        ):
            return
        self._visit_need_compact_collection(expr)

    def _visit_need_compact_collection(self, expr):
        compacted = self._compact(expr)
        if compacted is None:
            return

        if expr is not compacted:
            self._sub(expr, compacted)
            return compacted

    def _compact(self, expr):
        to_compact = [expr, ]

        for node in traverse_until_source(expr, top_down=True, unique=True):
            if node is expr:
                continue
            if not isinstance(node, CollectionExpr):
                continue

            # We do not handle collection with Scalar column or window function here
            # TODO think way to compact in this situation
            if isinstance(node, ProjectCollectionExpr) and \
                    not node.optimize_banned and \
                    not any(isinstance(n, Window) for n in node._fields):
                valid = True
                for it in itertools.chain(*(node.all_path(to_compact[-1]))):
                    if isinstance(it, CollectionExpr) and \
                            any(isinstance(n.input, LateralViewCollectionExpr) for n in it.columns):
                        valid = False
                        break
                    if isinstance(it, SequenceReduction):
                        valid = False
                        break
                if not valid:
                    break
                to_compact.append(node)
            else:
                break

        if len(to_compact) <= 1:
            return

        return self._do_compact(expr, *to_compact[1:][::-1])

    def _do_compact(self, expr, *to_compact):
        retval = expr

        collection_dict = ExprDictionary()
        for coll in to_compact:
            collection_dict[coll] = True

        for node in expr.traverse(top_down=True, unique=True,
                                  stop_cond=lambda x: x is to_compact[-1]):
            if node in collection_dict:
                parents = self._dag.successors(node)
                for parent in parents:
                    if isinstance(parent, Column):
                        col = parent

                        col_name = col.source_name or col.name
                        field = self._get_field(node, col_name)
                        if col.is_renamed():
                            field = field.rename(col.name)
                        else:
                            field = field.copy()

                        self._sub(col, field)

                        if col is retval:
                            retval = field
                    else:
                        parent.substitute(node, node.input, dag=self._dag)

        return retval

    def _get_fields(self, collection):
        fields = select_fields(collection)
        if isinstance(collection, GroupByCollectionExpr) and \
                collection._having is not None:
            # add GroupbyCollectionExpr.having to broadcast fields
            fields.append(collection._having)
        return fields

    def _get_field(self, collection, name):
        # FIXME: consider name with upper letters
        name = utils.to_str(name)

        if isinstance(collection, GroupByCollectionExpr):
            return collection._name_to_exprs()[name]

        name_idxes = collection.schema._name_indexes
        if name.lower() in name_idxes:
            idx = name_idxes[name.lower()]
        else:
            idx = name_idxes[name]
        return self._get_fields(collection)[idx]