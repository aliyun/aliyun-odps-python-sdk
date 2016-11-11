# encoding: utf-8
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

import textwrap
import uuid

from odps.compat import six
from odps.df.core import DataFrame, CollectionExpr
from odps.runner import BaseRunnerNode, RunnerContext, RunnerEdge, ObjectContainer, PortType, adapter_from_df
from odps.runner.tests.base import RunnerTestBase, tn

IONOSPHERE_TABLE = tn('pyodps_test_ml_ionosphere')


def _get_bind_node(obj):
    if isinstance(obj, CollectionExpr):
        return adapter_from_df(obj)._bind_node
    else:
        return obj._bind_node


def _get_bind_port(obj):
    if isinstance(obj, CollectionExpr):
        return adapter_from_df(obj)._bind_port
    else:
        return obj._bind_port


class TestCore(RunnerTestBase):
    def setUp(self):
        super(TestCore, self).setUp()
        self.ml_context = RunnerContext.instance()

    def test_base_dag_node(self):
        self.maxDiff = None

        self.create_ionosphere(IONOSPHERE_TABLE)
        df1 = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))
        node1 = _get_bind_node(df1)
        df2 = self.mock_action(df1, msg='Node2')
        node2 = _get_bind_node(df2)

        df31, df32, _ = self.mock_action([df1, df2], 3, msg='Node3')
        node3 = _get_bind_node(df31)

        df41, model42 = self.mock_action(df31, 'dm', msg='Node4')
        self.assertIn('DFAdapter', repr(_get_bind_port(df41)))
        self.assertIn('Model', repr(_get_bind_port(model42)))
        node4 = _get_bind_node(df41)

        model5 = self.mock_action([model42, df32], 'm', msg='Node5')
        node5 = _get_bind_node(model5)

        df6 = self.mock_action([df41, model5], 1, msg='Node6')
        node6 = _get_bind_node(df6)

        # test params
        self.assertDictEqual(node2.parameters, dict(message='Node2'))
        self.assertDictEqual(node3.parameters, dict(message='Node3'))
        self.assertDictEqual(node4.parameters, dict(message='Node4'))
        self.assertDictEqual(node5.parameters, dict(message='Node5'))
        self.assertDictEqual(node6.parameters, dict(message='Node6'))

        # test node inputs and outputs
        gen_type_dict = lambda eps: dict((nm, ep.type) for nm, ep in six.iteritems(eps))
        self.assertDictEqual(gen_type_dict(node2.inputs), dict(input1=PortType.DATA))
        self.assertDictEqual(gen_type_dict(node2.outputs), dict(output1=PortType.DATA))
        self.assertDictEqual(gen_type_dict(node3.inputs), dict(input1=PortType.DATA, input2=PortType.DATA))
        self.assertDictEqual(gen_type_dict(node3.outputs), dict(output1=PortType.DATA, output2=PortType.DATA, output3=PortType.DATA))
        self.assertDictEqual(gen_type_dict(node4.inputs), dict(input1=PortType.DATA))
        self.assertDictEqual(gen_type_dict(node4.outputs), dict(output1=PortType.DATA, output2=PortType.MODEL))
        self.assertDictEqual(gen_type_dict(node5.inputs), dict(input1=PortType.MODEL, input2=PortType.DATA))
        self.assertDictEqual(gen_type_dict(node5.outputs), dict(output1=PortType.MODEL))
        self.assertDictEqual(gen_type_dict(node6.inputs), dict(input1=PortType.DATA, input2=PortType.MODEL))
        self.assertDictEqual(gen_type_dict(node6.outputs), dict(output1=PortType.DATA))

        # test links
        def assertInEdge(dest_node, dest_ep, *sources):
            edges = [RunnerEdge(src_node, src_ep, dest_node, dest_ep) for src_node, src_ep in sources]
            self.assertListEqual(dest_node.input_edges[dest_ep], edges)

        def assertOutEdge(src_node, src_ep, *targets):
            edges = [RunnerEdge(src_node, src_ep, dest_node, dest_ep) for dest_node, dest_ep in targets]
            self.assertListEqual(src_node.output_edges[src_ep], edges)

        assertOutEdge(node1, 'output', (node2, 'input1'), (node3, 'input1'))

        assertInEdge(node2, 'input1', (node1, 'output'))
        assertOutEdge(node2, 'output1', (node3, 'input2'))

        assertInEdge(node3, 'input1', (node1, 'output'))
        assertInEdge(node3, 'input2', (node2, 'output1'))
        assertOutEdge(node3, 'output1', (node4, 'input1'))
        assertOutEdge(node3, 'output2', (node5, 'input2'))

        assertInEdge(node4, 'input1', (node3, 'output1'))
        assertOutEdge(node4, 'output1', (node6, 'input1'))
        assertOutEdge(node4, 'output2', (node5, 'input1'))

        assertInEdge(node5, 'input1', (node4, 'output2'))
        assertInEdge(node5, 'input2', (node3, 'output2'))

        assertInEdge(node6, 'input1', (node4, 'output1'))
        assertInEdge(node6, 'input2', (node5, 'output1'))

        steps_text = textwrap.dedent("""
        DataFrame_1 -> output
        MockNode_2(input1=DataFrame_1:output) -> output1
        MockNode_3(input1=DataFrame_1:output, input2=MockNode_2:output1) -> output1, output2, output3
        MockNode_4(input1=MockNode_3:output1) -> output1, output2
        MockNode_5(input1=MockNode_4:output2, input2=MockNode_3:output2) -> output1
        MockNode_6(input1=MockNode_4:output1, input2=MockNode_5:output1) -> output1(*)
        """).strip()

        steps_obj = df6.show_steps()
        self.assertEqual(steps_obj.text.strip(), steps_text)

    def test_convert_params(self):
        self.create_ionosphere(IONOSPHERE_TABLE)
        df1 = DataFrame(self.odps.get_table(IONOSPHERE_TABLE))
        df2 = self.mock_action(df1, msg='Node2')
        node2 = _get_bind_node(df2)
        self.assertDictEqual(node2.convert_params(), dict(message='Node2'))

        node2.add_exporter('message', lambda: [])
        self.assertEqual(node2.convert_params(), dict())

        node2.add_exporter('message', lambda: 'ConvValue')
        self.assertEqual(node2.convert_params(), dict(message='ConvValue'))

        node2.add_exporter('message', lambda: 2 / 0)  # which raises div by zero error
        self.assertRaises(ValueError, lambda: node2.convert_params())

    def test_object_container(self):
        class TestObj(object):
            def __init__(self, msg):
                self._obj_uuid = uuid.uuid4()
                self._msg = msg

        oc = ObjectContainer()
        self.assertListEqual([], list(oc.items()))
        obj = TestObj('a')
        oc.register(obj)
        self.assertEqual(obj, oc.get(obj._obj_uuid))
        self.assertEqual(obj, oc[obj._obj_uuid])
        self.assertRaises(KeyError, lambda: oc[None])
        self.assertRaises(KeyError, lambda: oc.remove(None))

        oc.remove(obj._obj_uuid)
        self.assertRaises(KeyError, lambda: oc[obj._obj_uuid])
        self.assertEqual(None, oc.get(obj._obj_uuid))
