# encoding: utf-8
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

from __future__ import print_function

import logging

from odps.df import DataFrame
from odps.config import options
from odps.ml.utils import TEMP_TABLE_PREFIX
from odps.ml.network import *
from odps.ml.tests.base import MLTestBase, tn

logger = logging.getLogger(__name__)

WEIGHTED_GRAPH_EDGE_TABLE = tn('pyodps_test_ml_weighted_graph_edge')
WEIGHTED_GRAPH_VERTEX_TABLE = tn('pyodps_test_ml_weighted_graph_node')
TREE_GRAPH_EDGE_TABLE = tn('pyodps_test_ml_tree_graph_edge')

NODE_DENSITY_TABLE = tn('pyodps_test_ml_node_density')
EDGE_DENSITY_TABLE = tn('pyodps_test_ml_edge_density')
MAXIMAL_CONNECTED_TABLE = tn('pyodps_test_ml_maximal_connected')
TRIANGLE_COUNT_TABLE = tn('pyodps_test_ml_triangle_count')
PAGE_RANK_TABLE = tn('pyodps_test_ml_page_rank')
LABEL_PROPAGATION_TABLE = tn('pyodps_test_ml_label_prop')
K_CORE_TABLE = tn('pyodps_test_ml__k_core')
SSSP_TABLE = tn('pyodps_test_ml_sssp')
TREE_DEPTH_TABLE = tn('pyodps_test_ml_tree_depth')


class Test(MLTestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.create_weighted_graph_edges(WEIGHTED_GRAPH_EDGE_TABLE)
        self.create_weighted_graph_vertices(WEIGHTED_GRAPH_VERTEX_TABLE)

        self.vertex_df = DataFrame(self.odps.get_table(WEIGHTED_GRAPH_VERTEX_TABLE)) \
            .roles(vertex_label='label', vertex_weight='node_weight').vertex_id_field('node')
        self.edge_df = DataFrame(self.odps.get_table(WEIGHTED_GRAPH_EDGE_TABLE)) \
            .roles(from_vertex='flow_out_id', to_vertex='flow_in_id', edge_weight='edge_weight')

        options.ml.dry_run = True

    def test_node_density(self):
        df = DataFrame(self.odps.get_table(WEIGHTED_GRAPH_EDGE_TABLE))
        output = NodeDensity(from_vertex_col='flow_out_id', to_vertex_col='flow_in_id') \
            .transform(df)._add_case(self.gen_check_params_case(
            {'splitSize': '64', 'workerMem': '4096', 'maxEdgeCnt': '500', 'fromVertexCol': 'flow_out_id',
             'toVertexCol': 'flow_in_id',
             'outputTableName': NODE_DENSITY_TABLE, 'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE}))
        output.persist(NODE_DENSITY_TABLE)

    def test_maximal_connected(self):
        df = DataFrame(self.odps.get_table(WEIGHTED_GRAPH_EDGE_TABLE))
        output = MaximalConnectedComponent(from_vertex_col='flow_out_id', to_vertex_col='flow_in_id') \
            .transform(df)._add_case(self.gen_check_params_case(
            {'splitSize': '64', 'workerMem': '4096', 'fromVertexCol': 'flow_out_id', 'toVertexCol': 'flow_in_id',
             'outputTableName': MAXIMAL_CONNECTED_TABLE, 'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE}))
        output.persist(MAXIMAL_CONNECTED_TABLE)

    def test_triangle_count(self):
        output = TriangleCount().transform(self.edge_df)._add_case(self.gen_check_params_case(
            {'splitSize': '64', 'workerMem': '4096', 'maxEdgeCnt': '500', 'fromVertexCol': 'flow_out_id',
             'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE, 'outputTableName': TRIANGLE_COUNT_TABLE,
             'toVertexCol': 'flow_in_id'}))
        output.persist(TRIANGLE_COUNT_TABLE)

    def test_edge_density(self):
        df = DataFrame(self.odps.get_table(WEIGHTED_GRAPH_EDGE_TABLE))
        output = EdgeDensity(from_vertex_col='flow_out_id', to_vertex_col='flow_in_id') \
            .transform(df)._add_case(self.gen_check_params_case(
            {'splitSize': '64', 'workerMem': '4096', 'fromVertexCol': 'flow_out_id', 'toVertexCol': 'flow_in_id',
             'outputTableName': EDGE_DENSITY_TABLE, 'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE}))
        output.persist(EDGE_DENSITY_TABLE)

    def test_page_rank(self):
        output = PageRank().transform(self.edge_df)._add_case(self.gen_check_params_case(
            {'splitSize': '64', 'hasEdgeWeight': 'true', 'fromVertexCol': 'flow_out_id',
             'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE, 'edgeWeightCol': 'edge_weight',
             'workerMem': '4096', 'toVertexCol': 'flow_in_id', 'outputTableName': PAGE_RANK_TABLE,
             'maxIter': '30'}))
        output.persist(PAGE_RANK_TABLE)

    def test_label_prop_cluster(self):
        output = LabelPropagationClustering() \
            .transform(self.edge_df, self.vertex_df)._add_case(self.gen_check_params_case(
            {'splitSize': '64', 'vertexWeightCol': 'node_weight', 'hasVertexWeight': 'true',
             'hasEdgeWeight': 'true', 'fromVertexCol': 'flow_out_id',
             'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE, 'edgeWeightCol': 'edge_weight',
             'vertexCol': 'node', 'workerMem': '4096', 'toVertexCol': 'flow_in_id',
             'outputTableName': LABEL_PROPAGATION_TABLE, 'maxIter': '30',
             'inputVertexTableName': WEIGHTED_GRAPH_VERTEX_TABLE, 'randSelect': 'false'}))
        output.persist(LABEL_PROPAGATION_TABLE)

    def test_label_prop_cls(self):
        edge_df = DataFrame(self.odps.get_table(WEIGHTED_GRAPH_EDGE_TABLE))
        vertex_df = DataFrame(self.odps.get_table(WEIGHTED_GRAPH_VERTEX_TABLE))
        output = LabelPropagationClassification(from_vertex_col='flow_out_id', to_vertex_col='flow_in_id',
                                                vertex_col='node', vertex_label_col='label',
                                                vertex_weight_col='node_weight', edge_weight_col='edge_weight') \
            .transform(edge_df, vertex_df)._add_case(self.gen_check_params_case(
            {'splitSize': '64', 'edgeWeightCol': 'edge_weight', 'vertexWeightCol': 'node_weight',
             'hasVertexWeight': 'true', 'hasEdgeWeight': 'true', 'fromVertexCol': 'flow_out_id',
             'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE, 'vertexCol': 'node',
             'epsilon': '0.000001', 'workerMem': '4096', 'toVertexCol': 'flow_in_id',
             'outputTableName': LABEL_PROPAGATION_TABLE, 'maxIter': '30', 'alpha': '0.8',
             'vertexLabelCol': 'label', 'inputVertexTableName': WEIGHTED_GRAPH_VERTEX_TABLE}))
        output.persist(LABEL_PROPAGATION_TABLE)

    def test_k_core(self):
        output = KCore(k=2).transform(self.edge_df)._add_case(self.gen_check_params_case(
            {'splitSize': '64', 'fromVertexCol': 'flow_out_id', 'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE,
             'workerMem': '4096', 'toVertexCol': 'flow_in_id', 'outputTableName': K_CORE_TABLE, 'k': '2'}))
        output.persist(K_CORE_TABLE)

    def test_sssp(self):
        output = SSSP(start_vertex='1').transform(self.edge_df)._add_case(self.gen_check_params_case(
            {'splitSize': '64', 'startVertex': '1', 'hasEdgeWeight': 'true', 'fromVertexCol': 'flow_out_id',
             'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE, 'edgeWeightCol': 'edge_weight',
             'workerMem': '4096', 'toVertexCol': 'flow_in_id', 'outputTableName': SSSP_TABLE}))
        output.persist(SSSP_TABLE)

    def test_tree_depth(self):
        self.create_tree_graph(TREE_GRAPH_EDGE_TABLE)
        tree_ds = DataFrame(self.odps.get_table(TREE_GRAPH_EDGE_TABLE)) \
            .roles(from_vertex='flow_out_id', to_vertex='flow_in_id')
        output = TreeDepth().transform(tree_ds)._add_case(self.gen_check_params_case(
            {'outputTableName': tn('pyodps_test_ml_tree_depth'), 'fromVertexCol': 'flow_out_id', 'workerMem': '4096',
             'inputEdgeTableName': tn('pyodps_test_ml_tree_graph_edge'), 'toVertexCol': 'flow_in_id',
             'splitSize': '64'}))
        output.persist(TREE_DEPTH_TABLE)

    def test_modularity(self):
        ds = self.edge_df.from_vertex_label_field('group_out_id').to_vertex_label_field('group_in_id')
        logger.info('Modularity: ' + str(modularity(ds, _cases=self.gen_check_params_case(
            {'splitSize': '64', 'fromVertexCol': 'flow_out_id', 'inputEdgeTableName': WEIGHTED_GRAPH_EDGE_TABLE,
             'fromGroupCol': 'group_out_id', 'workerMem': '4096', 'toVertexCol': 'flow_in_id',
             'outputTableName': TEMP_TABLE_PREFIX + '_modularity', 'toGroupCol': 'group_in_id'}))))
