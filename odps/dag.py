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
import functools
from collections import Iterable
from copy import deepcopy

from .compat import Queue, six


class DAGValidationError(Exception):
    pass


class DAG(object):
    """Directed acyclic graph implementation."""

    def __init__(self):
        self._graph = dict()
        self._map = dict()

    def nodes(self):
        return [self._map[n] for n in self._graph]

    def contains_node(self, node):
        return id(node) in self._graph

    def add_node(self, node, graph=None):
        graph = graph or self._graph

        graph[id(node)] = set()
        self._map[id(node)] = node

    def remove_node(self, node, graph=None):
        graph = graph or self._graph

        if id(node) not in graph:
            raise KeyError('Node does not exist')

        graph.pop(id(node))
        self._map.pop(id(node))

        for edges in six.itervalues(self._graph):
            if id(node) in edges:
                edges.remove(id(node))

    def contains_edge(self, predecessor_node, successor_node):
        if id(predecessor_node) not in self._graph or \
                id(successor_node) not in self._graph:
            return False

        return id(successor_node) in self._graph[id(predecessor_node)]

    def add_edge(self, predecessor_node, successor_node, graph=None):
        graph = graph or self._graph

        if id(predecessor_node) not in self._graph or \
                id(successor_node) not in self._graph:
            raise KeyError('Node does not exist')

        test_graph = deepcopy(graph)
        test_graph[id(predecessor_node)].add(id(successor_node))

        valid, msg = self._validate(test_graph)
        if valid:
            graph[id(predecessor_node)].add(id(successor_node))
        else:
            raise DAGValidationError(msg)

    def remove_edge(self, predecessor_node, successor_node, graph=None):
        graph = graph or self._graph

        if id(successor_node) not in graph.get(id(predecessor_node), []):
            raise KeyError('Edge does not exist in the graph')

        graph[id(predecessor_node)].remove(id(successor_node))

    def indep_nodes(self, graph=None):
        graph = graph or self._graph

        all_nodes = set(graph.keys())
        ids = all_nodes - set(itertools.chain(*graph.values()))
        return [self._map.get(i) for i in ids]

    def predecessors(self, node, graph=None):
        graph = graph or self._graph
        return [self._map.get(node_id) for node_id, deps in six.iteritems(graph)
                if id(node) in deps]

    def successors(self, node, graph=None):
        graph = graph or self._graph

        if id(node) not in graph:
            raise KeyError('Node does not exist: %s' % node)

        return [self._map.get(node_id) for node_id in graph[id(node)]]

    def _validate(self, graph=None):
        graph = graph or self._graph
        if len(self.indep_nodes(graph)) == 0:
            return False, 'No independent nodes detected'

        try:
            self.topological_sort(graph)
        except ValueError:
            return False, 'Fail to topological sort'
        return True, 'Valid'

    def bfs(self, start_nodes, successor=None, cond=None, graph=None):
        graph = graph or self._graph
        cond = cond or (lambda v: True)
        successor = successor or functools.partial(self.successors, graph=graph)
        start_nodes = [start_nodes, ] if not isinstance(start_nodes, Iterable) else start_nodes
        start_nodes = [n for n in start_nodes if cond(n)]
        assert all(id(node) in graph for node in start_nodes)

        visited = set(id(node) for node in start_nodes)
        node_queue = Queue()
        [node_queue.put(node) for node in start_nodes]
        while not node_queue.empty():
            cur_node = node_queue.get()
            for up_node in (n for n in successor(cur_node) if cond(n)):
                if id(up_node) not in visited:
                    visited.add(id(up_node))
                    yield up_node
                    node_queue.put(up_node)

    def ancestors(self, start_nodes, cond=None, graph=None):
        return list(self.bfs(start_nodes, functools.partial(self.predecessors, graph=graph), cond, graph))

    def descendants(self, start_nodes, cond=None, graph=None):
        return list(self.bfs(start_nodes, cond=cond, graph=graph))

    def topological_sort(self, graph=None):
        graph = graph or self._graph
        graph = deepcopy(graph)

        nodes = []

        indep_nodes = self.indep_nodes(graph)
        while len(indep_nodes) != 0:
            n = indep_nodes.pop(0)
            nodes.append(n)
            for dep_id in deepcopy(graph[id(n)]):
                graph[id(n)].remove(dep_id)
                dep = self._map.get(dep_id)
                if len(self.predecessors(dep, graph)) == 0:
                    indep_nodes.append(dep)

        if len(nodes) != len(graph):
            raise ValueError('Graph is not acyclic')

        return nodes

    def reset_graph(self):
        self._graph = dict()
        self._map = dict()
