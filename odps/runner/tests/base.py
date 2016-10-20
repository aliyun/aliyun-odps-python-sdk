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

from __future__ import print_function

import os
import warnings
from collections import Iterable

from odps.config import options
from odps.compat import six
from odps.df.core import CollectionExpr
from odps.runner import BaseRunnerNode, RunnerContext, node_engine, BaseNodeEngine, EngineType, PortType, \
    adapter_from_df, DFAdapter
from odps.tests.core import TestBase, tn
from odps.examples.tables import TestDataMixIn


class MockNode(BaseRunnerNode):
    def __init__(self, msg, action, input_types, output_types):
        super(MockNode, self).__init__('MockNode', EngineType.MOCK)
        self.marshal({
            'parameters': {
                'message': msg,
            },
            'inputs': [(idx + 1, 'input%d' % (idx + 1), typ) for idx, typ in enumerate(input_types)],
            'outputs': [(idx + 1, 'output%d' % (idx + 1), typ) for idx, typ in enumerate(output_types)],
        })
        self.message = msg
        self.action = action


@node_engine(EngineType.MOCK)
class MockNodeEngine(BaseNodeEngine):
    def actual_exec(self):
        in_tables = []
        for in_name, in_port in six.iteritems(self._node.inputs):
            if in_port.type != PortType.DATA:
                continue
            ep = RunnerContext.instance()._obj_container.get(in_port.obj_uuid)
            if ep:
                in_tables.append((in_name, ep.table))
        msg = 'Message: %s Input tables: %s' % (self._node.message, ', '.join('%s<-%s' % ti for ti in in_tables))
        if self._node.action is not None:
            self._node.action(self._node)
        else:
            print(msg)


class RunnerTestBase(TestDataMixIn, TestBase):
    def setUp(self):
        super(RunnerTestBase, self).setUp()

        RunnerContext.reset()
        from odps.runner.df.adapter import _df_endpoint_dict, _df_link_maintainer
        _df_endpoint_dict.clear()
        _df_link_maintainer.clear()

        # Force to false
        options.runner.dry_run = False
        options.lifecycle = 3
        options.verbose = 'CI_MODE' not in os.environ
        options.interactive = False
        # Disable warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning, module=r'.*widget\.py.*')

    def mock_action(self, sources, output_desc=1, msg='', action=None):
        try:
            from odps.ml import PmmlModel
        except ImportError:
            PmmlModel = None

        if not isinstance(sources, Iterable):
            sources = [sources, ]

        input_types = [PortType.DATA if isinstance(o, CollectionExpr) else PortType.MODEL for o in sources]

        source_objs = [adapter_from_df(s) if isinstance(s, CollectionExpr) else s for s in sources]
        uplinks = [adapter for adapter in source_objs if isinstance(adapter, DFAdapter)]

        if isinstance(output_desc, six.integer_types):
            output_types = [PortType.DATA for _ in range(output_desc)]
        else:
            output_types = [PortType.DATA if ch == 'd' else PortType.MODEL for ch in output_desc]

        merge_node = MockNode(msg, action, input_types, output_types)
        odps = None
        for idx, o in enumerate(source_objs):
            o._link_node(merge_node, 'input%d' % (1 + idx))
            odps = o._odps
        outputs = []
        for idx, out_type in enumerate(output_types):
            if out_type == PortType.DATA or PmmlModel is None:
                new_df = six.next(s for s in sources if isinstance(s, CollectionExpr)).copy()
                DFAdapter(odps, merge_node.outputs['output%d' % (1 + idx)], new_df, uplink=uplinks)
                outputs.append(new_df)
            else:
                outputs.append(PmmlModel(odps, port=merge_node.outputs['output%d' % (1 + idx)]))
        if len(output_types) == 1:
            return outputs[0]
        else:
            return outputs

    def after_create_test_data(self, table_name):
        if options.lifecycle:
            self.odps.run_sql('alter table %s set lifecycle %d' % (table_name, options.lifecycle))
