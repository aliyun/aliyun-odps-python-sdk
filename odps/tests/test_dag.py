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

from odps.tests.core import TestBase
from odps.compat import unittest
from odps.dag import DAG, DAGValidationError


class Test(TestBase):
    def testDAG(self):
        dag = DAG()

        labels = tuple('abcde')

        node1, node2, node3, node4, node5 = labels

        for i in range(1, 6):
            dag.add_node(locals()['node%d' % i])

        dag.add_edge(node1, node3)
        dag.add_edge(node2, node4)
        dag.add_edge(node3, node4)
        dag.add_edge(node4, node5)

        loc_vars = locals()
        self.assertEqual(sorted([loc_vars['node%d' % i] for i in range(1, 6)]),
                         sorted(dag.nodes()))

        self.assertTrue(dag.contains_node(node1))
        self.assertTrue(dag.contains_edge(node2, node4))
        self.assertFalse(dag.contains_edge(node2, node5))

        try:
            self.assertEqual(list('bacde'), dag.topological_sort())
        except AssertionError:
            self.assertEqual(list('abcde'), dag.topological_sort())
        self.assertEqual(list('bca'), dag.ancestors([node4, ]))
        self.assertEqual(list('de'), dag.descendants([node3, ]))

        dag.add_edge(node2, node1)
        self.assertEqual(list('bacde'), dag.topological_sort())
        self.assertEqual(list('ab'), dag.ancestors([node3, ]))
        self.assertEqual(list('daec'), dag.descendants([node2, ]))

        self.assertRaises(DAGValidationError, lambda: dag.add_edge(node4, node2))
        self.assertRaises(DAGValidationError, lambda: dag.add_edge(node4, node1))

        self.assertEqual(dag.successors(node2), list('da'))
        self.assertEqual(dag.predecessors(node4), list('bc'))

        dag.remove_node(node4)
        self.assertIn(''.join(dag.topological_sort()), set(['beac', 'ebac']))
        self.assertFalse(dag.contains_node(node4))
        self.assertRaises(KeyError, lambda: dag.remove_node(node4))
        self.assertFalse(dag.contains_edge(node4, node5))
        self.assertRaises(KeyError, lambda: dag.add_edge(node4, node5))
        self.assertRaises(KeyError, lambda: dag.remove_edge(node4, node5))
        self.assertRaises(KeyError, lambda: dag.successors(node4))

        self.assertEqual(list('ab'), dag.ancestors([node3, ]))
        self.assertEqual(list(''), dag.ancestors([node5, ]))
        self.assertEqual(list('ac'), dag.descendants([node2, ]))
        self.assertEqual(list(''), dag.descendants([node5, ]))

        dag.remove_edge(node2, node1)
        self.assertFalse(dag.contains_edge(node2, node1))
        self.assertEqual(list('a'), dag.ancestors([node3, ]))
        self.assertEqual(list('c'), dag.descendants([node1, ]))
        self.assertSetEqual(set('abe'), set(dag.indep_nodes()))

        dag.reset_graph()
        self.assertEqual(len(dag.nodes()), 0)

    def testReversedDAG(self):
        dag = DAG(reverse=True)

        labels = tuple('abcde')

        node1, node2, node3, node4, node5 = labels

        for i in range(1, 6):
            dag.add_node(locals()['node%d' % i])

        dag.add_edge(node1, node3)
        dag.add_edge(node2, node4)
        dag.add_edge(node3, node4)
        dag.add_edge(node4, node5)

        loc_vars = locals()
        self.assertEqual(sorted([loc_vars['node%d' % i] for i in range(1, 6)]),
                         sorted(dag.nodes()))

        self.assertTrue(dag.contains_node(node1))
        self.assertTrue(dag.contains_edge(node2, node4))
        self.assertFalse(dag.contains_edge(node2, node5))

        self.assertEqual(list(labels), dag.topological_sort())
        self.assertEqual(list('bca'), dag.ancestors([node4, ]))
        self.assertEqual(list('de'), dag.descendants([node3, ]))

        dag.add_edge(node2, node1)
        self.assertEqual(list('bacde'), dag.topological_sort())
        self.assertEqual(list('ab'), dag.ancestors([node3, ]))
        self.assertEqual(list('daec'), dag.descendants([node2, ]))

        self.assertRaises(DAGValidationError, lambda: dag.add_edge(node4, node2))
        self.assertRaises(DAGValidationError, lambda: dag.add_edge(node4, node1))

        self.assertEqual(dag.successors(node2), list('da'))
        self.assertEqual(dag.predecessors(node4), list('bc'))

        dag.remove_node(node4)
        self.assertIn(''.join(dag.topological_sort()), set(['beac', 'ebac']))
        self.assertFalse(dag.contains_node(node4))
        self.assertRaises(KeyError, lambda: dag.remove_node(node4))
        self.assertFalse(dag.contains_edge(node4, node5))
        self.assertRaises(KeyError, lambda: dag.add_edge(node4, node5))
        self.assertRaises(KeyError, lambda: dag.remove_edge(node4, node5))
        self.assertRaises(KeyError, lambda: dag.successors(node4))

        self.assertEqual(list('ab'), dag.ancestors([node3, ]))
        self.assertEqual(list(''), dag.ancestors([node5, ]))
        self.assertEqual(list('ac'), dag.descendants([node2, ]))
        self.assertEqual(list(''), dag.descendants([node5, ]))

        dag.remove_edge(node2, node1)
        self.assertFalse(dag.contains_edge(node2, node1))
        self.assertEqual(list('a'), dag.ancestors([node3, ]))
        self.assertEqual(list('c'), dag.descendants([node1, ]))
        self.assertSetEqual(set('abe'), set(dag.indep_nodes()))

        dag.reset_graph()
        self.assertEqual(len(dag.nodes()), 0)


if __name__ == '__main__':
    unittest.main()
