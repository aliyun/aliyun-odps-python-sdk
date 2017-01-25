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

from ..adapter import ml_collection_mixin
from ..nodes.exporters import get_enable_sparse, get_input_field_names
from ...compat import Enum


class RecommendFieldRole(Enum):
    REC_USER_ID = 'USER_ID'
    REC_ITEM = 'REC_ITEM'
    REC_SEQUENCE = 'REC_SEQUENCE'
    REC_PAYLOAD = 'REC_PAYLOAD'


@ml_collection_mixin
class RecommendMLMixIn(object):
    field_role_enum = RecommendFieldRole
    non_feature_roles = set([RecommendFieldRole.REC_ITEM, ])


"""
Common recommend exporters
"""
get_rec_user_id_column = partial(get_input_field_names, field_role=RecommendFieldRole.REC_USER_ID)
get_rec_item_column = partial(get_input_field_names, field_role=RecommendFieldRole.REC_ITEM)
get_rec_sequence_column = partial(get_input_field_names, field_role=RecommendFieldRole.REC_SEQUENCE)
get_rec_payload_column = partial(get_input_field_names, field_role=RecommendFieldRole.REC_PAYLOAD)


def get_etrec_table_format(node, param_name, input_name):
    if node.parameters.get(param_name, None):
        return node.parameters[param_name]
    return 'user-item' if not get_enable_sparse(node, param_name, input_name) else 'items'


def get_rec_triple_selected_col_names(node, param_name, input_name):
    if node.parameters.get(param_name, None):
        return node.parameters[param_name]
    cols = [get_rec_user_id_column(node, param_name, input_name)[0],
            get_rec_item_column(node, param_name, input_name)[0]]
    payload_col = get_rec_payload_column(node, param_name, input_name)[0]
    if payload_col:
        cols.append(payload_col)
    return ','.join(cols)
