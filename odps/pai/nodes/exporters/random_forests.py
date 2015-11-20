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

import math

from ...utils.odps_utils import FieldRole


def random_forest_algo_types(context, node):
    algo_type = node.parameters["algorithmType"]
    tree_num = int(node.parameters["treeNum"])
    tree_num = tree_num - 1 if tree_num != 0 else 0
    if algo_type == "mix":
        return None
    elif algo_type == "id3":
        return '%d,%d' % (tree_num, tree_num)
    elif algo_type == "c45":
        return "0,0"
    elif "cart":
        return "0,%d" % tree_num
    return None


def random_forest_min_num_per(context, node):
    ret = '-1'
    if node.parameters['minNumPerInt'] is None:
        return ret

    min_num_per_int = node.parameters['minNumPerInt']
    if min_num_per_int.strip() != '':
        if int(min_num_per_int) in {-1, 0}:
            return ret
        else:
            return int(min_num_per_int) / 100.0
    return str(ret)


def random_forest_random_column_number(context, node, input_name):
    random_attr_type = node.parameters['randomAttrType']

    def render(v):
        return str(int(math.ceil(v)))

    if random_attr_type.replace('.', '', 1).isdigit():
        return render(int(random_attr_type))
    else:
        data_uuid = node.inputs[input_name].data_set_uuid
        fields = context._ds_container[data_uuid]._fields
        feature_count = len(filter(lambda f: f.role == FieldRole.FEATURE, fields))
        if random_attr_type == 'logN':
            return render(math.log(feature_count, 2))
        elif random_attr_type == 'N/3':
            return render(feature_count / 3.0)
        elif random_attr_type == 'sqrtN':
            return render(math.sqrt(feature_count))
        elif random_attr_type == 'N':
            return render(feature_count)
    return 0
