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

from ..core import Backend
from ...expr.expressions import *
from ...expr.merge import JoinCollectionExpr


class Analyzer(Backend):
    def __init__(self, dag, traversed=None, on_sub=None):
        self._expr = dag.root
        self._dag = dag
        self._indexer = itertools.count(0)
        self._traversed = traversed or set()
        self._on_sub = on_sub

    def analyze(self):
        for node in self._iter():
            self._visit_node(node)
            self._traversed.add(id(node))

        return self._expr

    def _iter(self):
        for node in self._expr.traverse(top_down=True, unique=True,
                                        traversed=self._traversed):
            yield node

        while True:
            all_traversed = True
            for node in self._expr.traverse(top_down=True, unique=True):
                if id(node) not in self._traversed:
                    all_traversed = False
                    yield node
            if all_traversed:
                break

    def _visit_node(self, node):
        try:
            node.accept(self)
        except NotImplementedError:
            return
