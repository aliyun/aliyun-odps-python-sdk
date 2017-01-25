#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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
    for path in expr.all_path(src_input, strict=True):
        cols = [it for it in path if isinstance(it, Column)]
        assert len(cols) <= 1
        collection_len = len([it for it in path if isinstance(it, CollectionExpr)])
        if isinstance(expr, CollectionExpr):
            assert collection_len == 2
        else:
            assert collection_len == 1
        if len(cols) == 1:
            col = cols[0]

            col_name = col.source_name or col.name
            field = get_field(new_input, col_name)
            if col.is_renamed():
                field = field.rename(col.name)
            else:
                field = field.copy()
            path[-3].substitute(col, field, dag=dag)
        else:
            path[-2].substitute(src_input, new_input, dag=dag)


def copy_sequence(sequence, collection, dag=None):
    copied = sequence.copy()
    if dag:
        dag.add_node(copied)
    is_copied = set()
    for path in sequence.all_path(collection, strict=True):
        curr = copied
        for seq in path[1:-1]:
            if id(seq) in is_copied:
                continue
            is_copied.add(id(seq))
            copied_seq = seq.copy()
            curr.substitute(seq, copied_seq, dag=dag)
            curr = copied_seq

    return copied