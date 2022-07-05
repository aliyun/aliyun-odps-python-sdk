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
from copy import deepcopy

from .compat import Queue, six, Iterable


class DAGValidationError(Exception):
    pass


class DAG(object):
    """Directed acyclic graph implementation."""
    _graph_dict_type = dict
    _dict_type = dict

    def __init__(self, reverse=False):
        self._graph = self._graph_dict_type()
        self._map = self._dict_type()
        if reverse:
            self._reversed_graph = self._graph_dict_type()
        else:
            self._reversed_graph = None

    def nodes(self):
        return [self._map[n] for n in self._graph]

    def contains_node(self, node):
        return id(node) in self._graph

    def add_node(self, node):
        self._graph[id(node)] = set()
        self._map[id(node)] = node
        if self._reversed_graph is not None:
            self._reversed_graph[id(node)] = set()

    def remove_node(self, node):
        if id(node) not in self._graph:
            raise KeyError('Node does not exist')

        self._graph.pop(id(node))
        self._map.pop(id(node))

        for edges in six.itervalues(self._graph):
            if id(node) in edges:
                edges.remove(id(node))

        if self._reversed_graph is not None:
            self._reversed_graph.pop(id(node))

            for edges in six.itervalues(self._reversed_graph):
                if id(node) in edges:
                    edges.remove(id(node))

    def contains_edge(self, predecessor_node, successor_node):
        if id(predecessor_node) not in self._graph or \
                id(successor_node) not in self._graph:
            return False

        return id(successor_node) in self._graph[id(predecessor_node)]

    def add_edge(self, predecessor_node, successor_node, validate=True):
        if id(predecessor_node) not in self._graph or \
                id(successor_node) not in self._graph:
            raise KeyError('Node does not exist')

        if validate:
            test_graph = deepcopy(self._graph)
            test_graph[id(predecessor_node)].add(id(successor_node))
            test_reversed_graph = None
            if self._reversed_graph is not None:
                test_reversed_graph = deepcopy(self._reversed_graph)
                test_reversed_graph[id(successor_node)].add(id(predecessor_node))
            valid, msg = self._validate(test_graph, test_reversed_graph)
        else:
            valid, msg = True, ''
        if valid:
            self._graph[id(predecessor_node)].add(id(successor_node))
            if self._reversed_graph is not None:
                self._reversed_graph[id(successor_node)].add(id(predecessor_node))
        else:
            raise DAGValidationError(msg)

    def remove_edge(self, predecessor_node, successor_node):
        if id(successor_node) not in self._graph.get(id(predecessor_node), []):
            raise KeyError('Edge does not exist in the graph')

        self._graph[id(predecessor_node)].remove(id(successor_node))
        if self._reversed_graph is not None:
            self._reversed_graph[id(successor_node)].remove(id(predecessor_node))

    def _indep_ids(self, graph=None, reversed_graph=None):
        graph = graph or self._graph
        reversed_graph = reversed_graph or self._reversed_graph

        if reversed_graph is not None:
            return [node for node, precessors in six.iteritems(reversed_graph)
                    if len(precessors) == 0]

        all_nodes = set(graph.keys())
        return list(all_nodes - set(itertools.chain(*graph.values())))

    def indep_nodes(self, graph=None, reversed_graph=None):
        return [self._map.get(i) for i in self._indep_ids(graph=graph,
                                                          reversed_graph=reversed_graph)]

    def _predecessor_ids(self, node_id, graph=None, reversed_graph=None):
        graph = graph or self._graph
        reversed_graph = reversed_graph or self._reversed_graph
        if reversed_graph is not None:
            return reversed_graph[node_id]
        return [nid for nid, deps in six.iteritems(graph) if node_id in deps]

    def predecessors(self, node):
        if id(node) not in self._graph:
            raise KeyError('Node does not exist: %s' % node)

        return [self._map.get(node_id) for node_id in self._predecessor_ids(id(node))]

    def successors(self, node):
        if id(node) not in self._graph:
            raise KeyError('Node does not exist: %r' % node)

        return [self._map.get(node_id) for node_id in self._graph[id(node)]]

    def _validate(self, graph=None, reversed_graph=None):
        graph = graph or self._graph
        reversed_graph = reversed_graph or self._reversed_graph
        if len(self.indep_nodes(graph, reversed_graph)) == 0:
            return False, 'No independent nodes detected'

        try:
            self.topological_sort(graph, reversed_graph)
        except ValueError:
            return False, 'Fail to topological sort'
        return True, 'Valid'

    def bfs(self, start_nodes, successor=None, cond=None):
        cond = cond or (lambda v: True)
        successor = successor or self.successors
        start_nodes = [start_nodes, ] if not isinstance(start_nodes, Iterable) else start_nodes
        start_nodes = [n for n in start_nodes if cond(n)]
        assert all(id(node) in self._graph for node in start_nodes)

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

    def ancestors(self, start_nodes, cond=None):
        return list(self.bfs(start_nodes, self.predecessors, cond))

    def iter_ancestors(self, start_nodes, cond=None):
        for it in self.bfs(start_nodes, self.predecessors, cond=cond):
            yield it

    def descendants(self, start_nodes, cond=None):
        return list(self.bfs(start_nodes, cond=cond))

    def iter_descendants(self, start_nodes, cond=None):
        for it in self.bfs(start_nodes, cond=cond):
            yield it

    def topological_sort(self, graph=None, reversed_graph=None):
        graph = graph or self._graph
        graph = deepcopy(graph)
        reversed_graph = reversed_graph or self._reversed_graph
        reversed_graph = deepcopy(reversed_graph)

        node_ids = []

        indep_ids = self._indep_ids(graph, reversed_graph)
        while len(indep_ids) != 0:
            n = indep_ids.pop(0)
            node_ids.append(n)
            for dep_id in deepcopy(graph[n]):
                graph[n].remove(dep_id)
                if reversed_graph is not None:
                    reversed_graph[dep_id].remove(n)
                if len(self._predecessor_ids(dep_id, graph, reversed_graph)) == 0:
                    indep_ids.append(dep_id)

        if len(node_ids) != len(graph):
            raise ValueError('Graph is not acyclic')

        return [self._map.get(nid) for nid in node_ids]

    def reset_graph(self):
        self._graph = self._graph_dict_type()
        self._map = self._dict_type()
        if self._reversed_graph is not None:
            self._reversed_graph = self._graph_dict_type()
