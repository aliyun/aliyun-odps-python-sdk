#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import pytest

from ..dag import DAG, DAGValidationError


def test_dag():
    dag = DAG()

    labels = tuple("abcde")

    node1, node2, node3, node4, node5 = labels

    for i in range(1, 6):
        dag.add_node(locals()["node%d" % i])

    dag.add_edge(node1, node3)
    dag.add_edge(node2, node4)
    dag.add_edge(node3, node4)
    dag.add_edge(node4, node5)

    loc_vars = locals()
    assert sorted([loc_vars["node%d" % i] for i in range(1, 6)]) == sorted(dag.nodes())

    assert dag.contains_node(node1) is True
    assert dag.contains_edge(node2, node4) is True
    assert dag.contains_edge(node2, node5) is False

    try:
        assert list("bacde") == dag.topological_sort()
    except AssertionError:
        assert list("abcde") == dag.topological_sort()
    assert list("bca") == dag.ancestors([node4])
    assert list("de") == dag.descendants([node3])

    dag.add_edge(node2, node1)
    assert list("bacde") == dag.topological_sort()
    assert list("ab") == dag.ancestors([node3])
    assert list("daec") == dag.descendants([node2])

    pytest.raises(DAGValidationError, lambda: dag.add_edge(node4, node2))
    pytest.raises(DAGValidationError, lambda: dag.add_edge(node4, node1))

    assert dag.successors(node2) == list("da")
    assert dag.predecessors(node4) == list("bc")

    dag.remove_node(node4)
    assert "".join(dag.topological_sort()) in set(["beac", "ebac"])
    assert dag.contains_node(node4) is False
    pytest.raises(KeyError, lambda: dag.remove_node(node4))
    assert dag.contains_edge(node4, node5) is False
    pytest.raises(KeyError, lambda: dag.add_edge(node4, node5))
    pytest.raises(KeyError, lambda: dag.remove_edge(node4, node5))
    pytest.raises(KeyError, lambda: dag.successors(node4))

    assert list("ab") == dag.ancestors([node3])
    assert list("") == dag.ancestors([node5])
    assert list("ac") == dag.descendants([node2])
    assert list("") == dag.descendants([node5])

    dag.remove_edge(node2, node1)
    assert dag.contains_edge(node2, node1) is False
    assert list("a") == dag.ancestors([node3])
    assert list("c") == dag.descendants([node1])
    assert set("abe") == set(dag.indep_nodes())

    dag.reset_graph()
    assert len(dag.nodes()) == 0


def test_reversed_dag():
    dag = DAG(reverse=True)

    labels = tuple("abcde")

    node1, node2, node3, node4, node5 = labels

    for i in range(1, 6):
        dag.add_node(locals()["node%d" % i])

    dag.add_edge(node1, node3)
    dag.add_edge(node2, node4)
    dag.add_edge(node3, node4)
    dag.add_edge(node4, node5)

    loc_vars = locals()
    assert sorted([loc_vars["node%d" % i] for i in range(1, 6)]) == sorted(dag.nodes())

    assert dag.contains_node(node1) is True
    assert dag.contains_edge(node2, node4) is True
    assert dag.contains_edge(node2, node5) is False

    assert list(labels) == dag.topological_sort()
    assert list("bca") == dag.ancestors([node4])
    assert list("de") == dag.descendants([node3])

    dag.add_edge(node2, node1)
    assert list("bacde") == dag.topological_sort()
    assert list("ab") == dag.ancestors([node3])
    assert list("daec") == dag.descendants([node2])

    pytest.raises(DAGValidationError, lambda: dag.add_edge(node4, node2))
    pytest.raises(DAGValidationError, lambda: dag.add_edge(node4, node1))

    assert dag.successors(node2) == list("da")
    assert dag.predecessors(node4) == list("bc")

    dag.remove_node(node4)
    assert "".join(dag.topological_sort()) in set(["beac", "ebac"])
    assert dag.contains_node(node4) is False
    pytest.raises(KeyError, lambda: dag.remove_node(node4))
    assert dag.contains_edge(node4, node5) is False
    pytest.raises(KeyError, lambda: dag.add_edge(node4, node5))
    pytest.raises(KeyError, lambda: dag.remove_edge(node4, node5))
    pytest.raises(KeyError, lambda: dag.successors(node4))

    assert list("ab") == dag.ancestors([node3])
    assert list("") == dag.ancestors([node5])
    assert list("ac") == dag.descendants([node2])
    assert list("") == dag.descendants([node5])

    dag.remove_edge(node2, node1)
    assert dag.contains_edge(node2, node1) is False
    assert list("a") == dag.ancestors([node3])
    assert list("c") == dag.descendants([node1])
    assert set("abe") == set(dag.indep_nodes())

    dag.reset_graph()
    assert len(dag.nodes()) == 0
