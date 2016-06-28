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
import weakref
from collections import Iterable
from copy import deepcopy

from .compat import Queue, six


class DAGValidationError(Exception):
    pass


class DAG(object):
    """Directed acyclic graph implementation."""
    _dict_type = dict

    def __init__(self):
        self._graph = dict()
        self._map = self._dict_type()

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

    def add_edge(self, predecessor_node, successor_node, graph=None, validate=True):
        graph = graph or self._graph

        if id(predecessor_node) not in self._graph or \
                id(successor_node) not in self._graph:
            raise KeyError('Node does not exist')

        if validate:
            test_graph = deepcopy(graph)
            test_graph[id(predecessor_node)].add(id(successor_node))
            valid, msg = self._validate(test_graph)
        else:
            valid, msg = True, ''
        if valid:
            graph[id(predecessor_node)].add(id(successor_node))
        else:
            raise DAGValidationError(msg)

    def remove_edge(self, predecessor_node, successor_node, graph=None):
        graph = graph or self._graph

        if id(successor_node) not in graph.get(id(predecessor_node), []):
            raise KeyError('Edge does not exist in the graph')

        graph[id(predecessor_node)].remove(id(successor_node))

    def _indep_ids(self, graph=None):
        graph = graph or self._graph

        all_nodes = set(graph.keys())
        return list(all_nodes - set(itertools.chain(*graph.values())))

    def indep_nodes(self, graph=None):
        return [self._map.get(i) for i in self._indep_ids(graph=graph)]

    def _predecessor_ids(self, node_id, graph=None):
        graph = graph or self._graph
        return [nid for nid, deps in six.iteritems(graph) if node_id in deps]

    def predecessors(self, node, graph=None):
        graph = graph or self._graph
        return [self._map.get(node_id) for node_id in self._predecessor_ids(id(node), graph=graph)]

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

        node_ids = []

        indep_ids = self._indep_ids(graph)
        while len(indep_ids) != 0:
            n = indep_ids.pop(0)
            node_ids.append(n)
            for dep_id in deepcopy(graph[n]):
                graph[n].remove(dep_id)
                if len(self._predecessor_ids(dep_id, graph)) == 0:
                    indep_ids.append(dep_id)

        if len(node_ids) != len(graph):
            raise ValueError('Graph is not acyclic')

        return [self._map.get(nid) for nid in node_ids]

    def reset_graph(self):
        self._graph = dict()
        self._map = self._dict_type()


class WeakNodeDAG(DAG):
    _dict_type = weakref.WeakValueDictionary

    def _sync_graph(self):
        removal = set(n for n in self._graph if n not in self._map)
        if not removal:
            return
        for n in removal:
            del self._graph[n]
        for n in self._graph:
            self._graph[n] -= removal

    def nodes(self):
        self._sync_graph()
        return [self._map[n] for n in self._graph if n in self._map]

    def contains_node(self, node):
        return node in self._map and id(node) in self._graph

    def contains_edge(self, predecessor_node, successor_node):
        if id(predecessor_node) not in self._map or \
                        id(successor_node) not in self._map:
            return False

        return id(successor_node) in self._graph[id(predecessor_node)]

    def indep_nodes(self, graph=None):
        return [n for n in super(WeakNodeDAG, self).indep_nodes(graph=graph) if n is not None]

    def predecessors(self, node, graph=None):
        return [n for n in super(WeakNodeDAG, self).predecessors(node, graph=graph) if n is not None]

    def successors(self, node, graph=None):
        return [n for n in super(WeakNodeDAG, self).successors(node, graph=graph) if n is not None]

    def topological_sort(self, graph=None):
        self._sync_graph()
        return [n for n in super(WeakNodeDAG, self).topological_sort(graph=graph) if n is not None]

    def reset_graph(self):
        self._graph = dict()
        self._map = self._dict_type()
