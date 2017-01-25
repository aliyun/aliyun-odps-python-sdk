# encoding: utf-8
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from functools import partial

from ...df import DataFrame
from ..adapter import ml_collection_mixin
from ..nodes.exporters import get_input_field_names
from ...compat import six, Enum


class NetworkFieldRole(Enum):
    FROM_VERTEX = 'FROM_VERTEX'
    TO_VERTEX = 'TO_VERTEX'
    VERTEX_ID = 'VERTEX_ID'
    VERTEX_LABEL = 'VERTEX_LABEL'
    FROM_VERTEX_LABEL = 'FROM_VERTEX_LABEL'
    TO_VERTEX_LABEL = 'TO_VERTEX_LABEL'
    VERTEX_WEIGHT = 'VERTEX_WEIGHT'
    EDGE_WEIGHT = 'EDGE_WEIGHT'


@ml_collection_mixin
class NetworkDFMixIn(object):
    field_role_enum = NetworkFieldRole
    non_feature_roles = set((NetworkFieldRole.FROM_VERTEX, NetworkFieldRole.TO_VERTEX, NetworkFieldRole.VERTEX_ID,
                             NetworkFieldRole.FROM_VERTEX_LABEL, NetworkFieldRole.TO_VERTEX_LABEL))

"""
Common graph exporters
"""
get_from_vertex_column = partial(get_input_field_names, field_role=NetworkFieldRole.FROM_VERTEX)
get_to_vertex_column = partial(get_input_field_names, field_role=NetworkFieldRole.TO_VERTEX)
get_vertex_id_column = partial(get_input_field_names, field_role=NetworkFieldRole.VERTEX_ID)
get_vertex_label_column = partial(get_input_field_names, field_role=NetworkFieldRole.VERTEX_LABEL)
get_from_vertex_label_column = partial(get_input_field_names, field_role=NetworkFieldRole.FROM_VERTEX_LABEL)
get_to_vertex_label_column = partial(get_input_field_names, field_role=NetworkFieldRole.TO_VERTEX_LABEL)
get_vertex_weight_column = partial(get_input_field_names, field_role=NetworkFieldRole.VERTEX_WEIGHT)
get_edge_weight_column = partial(get_input_field_names, field_role=NetworkFieldRole.EDGE_WEIGHT)


def graph_has_vertex_weight(node, input_name):
    vert_col = get_vertex_weight_column(node, 'vertexWeightCol', input_name)
    return 'true' if vert_col else 'false'


def graph_has_edge_weight(node, input_name):
    edge_col = get_edge_weight_column(node, 'edgeWeightCol', input_name)
    return 'true' if edge_col else 'false'


"""
Metrics
"""


def get_modularity_result(odps, node):
    return DataFrame(odps.get_table(node.table_names)).execute()[0][0]
