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


from odps.tests.core import TestBase
from odps.compat import unittest
from odps.df.expr.core import Node


class FakeNode(Node):
    __slots__ = 'name',
    _args = 'child1', 'child2', 'child3'

    def __str__(self):
        return self.name


class Test(TestBase):
    def testNodes(self):
        node1 = FakeNode(name='1')
        node2 = FakeNode(name='2')
        node3 = FakeNode(node1, name='3')
        node4 = FakeNode(node3, node2, name='4')
        node5 = FakeNode(node1, node4, name='5')

        self.assertSequenceEqual(list(node5.traverse()),
                                 [node1, node1, node3, node2, node4, node5])
        self.assertSequenceEqual(list(node5.traverse(top_down=True)),
                                 [node5, node1, node4, node3, node1, node2])
        self.assertSequenceEqual(list(node5.traverse(unique=True)),
                                 [node1, node3, node2, node4, node5])
        self.assertSequenceEqual(list(node5.traverse(top_down=True, unique=True)),
                                 [node5, node1, node4, node3, node2])

        self.assertSequenceEqual(list(node5.leaves()), [node1, node1, node2])

        node1_copy = FakeNode(name='1')
        self.assertEqual(node1, node1_copy)
        self.assertEqual(hash(node1), hash(node1_copy))

        node3_copy = FakeNode(node1_copy, name='3')
        self.assertEqual(node3, node3_copy)
        self.assertEqual(hash(node3), hash(node3_copy))

        self.assertTrue(node5.is_ancestor(node1))
        self.assertTrue(node5.is_ancestor(node2))
        self.assertFalse(node1.is_ancestor(node2))

        self.assertSequenceEqual([n.name for n in node5.path(node1)], ['5', '1'])
        self.assertSequenceEqual([n.name for n in node5.path(node2)], ['5', '4', '2'])

        paths_node_5_1 = list(node5.all_path(node1))
        self.assertEqual(len(paths_node_5_1), 2)
        self.assertEqual([int(n.name) for n in paths_node_5_1[0]], [5, 1])
        self.assertEqual([int(n.name) for n in paths_node_5_1[1]], [5, 4, 3, 1])

        node6 = FakeNode(name='6')
        node3.substitute(node1, node6)
        self.assertSequenceEqual(list(node5.traverse()),
                                 [node1, node6, node3, node2, node4, node5])

        self.assertIs(node4.parents[0], node5)
        self.assertIs(node2.parents[0], node4)


if __name__ == '__main__':
    unittest.main()
