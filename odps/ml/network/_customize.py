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

from ..expr.exporters import get_input_field_names
from ..expr.mixin import ml_collection_mixin
from ...compat import Enum
from ...df import DataFrame


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
    __slots__ = ()

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


def graph_has_vertex_weight(expr, input_name):
    vert_col = get_vertex_weight_column(expr, 'vertexWeightCol', input_name)
    return 'true' if vert_col else 'false'


def graph_has_edge_weight(expr, input_name):
    edge_col = get_edge_weight_column(expr, 'edgeWeightCol', input_name)
    return 'true' if edge_col else 'false'


"""
Metrics
"""


def get_modularity_result(expr, odps):
    return DataFrame(odps.get_table(expr.tables[0])).execute()[0][0]
