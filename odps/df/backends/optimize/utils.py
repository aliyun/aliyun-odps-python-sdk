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


from ...expr.expressions import Column, CollectionExpr


def change_input(expr, src_input, new_input, get_field, dag):
    traversed = set()
    list(expr.traverse(traversed=traversed, stop_cond=lambda x: x is src_input, unique=True))
    for p in filter(lambda x: x._node_id in traversed, dag.successors(src_input)):
        if isinstance(p, Column):
            col_name = p.source_name or p.name
            field = get_field(new_input, col_name)
            if p.is_renamed():
                field = field.rename(p.name)
            else:
                field = field.copy()
            if p is expr:
                p.substitute(src_input, new_input, dag=dag)
            else:
                parents = [it for it in dag.successors(p)
                           if it._node_id in traversed]
                dag.substitute(p, field, parents=parents)
        else:
            assert isinstance(p, CollectionExpr)
            p.substitute(src_input, new_input, dag=dag)


def copy_sequence(sequence, collection, dag=None):
    if dag is None:
        dag = sequence.to_dag(copy=False)

    traversed = set()
    copies = dict()

    def copy(node):
        if node._node_id not in copies:
            copies[node._node_id] = node.copy()
        copied = copies[node._node_id]
        if not dag.contains_node(copied):
            dag.add_node(copied)
        return copied

    for n in sequence.traverse(top_down=True, traversed=traversed,
                               stop_cond=lambda x: x is collection, unique=True):
        if n is sequence:
            copy(n)
            continue
        if n is collection:
            continue

        parents = [p for p in dag.successors(n) if p._node_id in traversed]
        for parent in parents:
            copy(parent).substitute(n, copy(n), dag=dag)

    return copies[sequence._node_id]