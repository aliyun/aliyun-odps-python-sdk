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


def linear_svm_set_cost(algo, param_name, param_value):
    algo._parameters['positiveCost'].value = param_value
    algo._parameters['negativeCost'].value = param_value


"""
Random forests exporters
"""


def random_forest_algo_types(expr):
    algo_type = expr._params["algorithmType"]
    tree_num = int(expr._params["treeNum"])
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


def random_forest_min_num_per(expr):
    ret = '-1'
    if expr._params['minNumPerInt'] is None:
        return ret

    min_num_per_int = expr._params['minNumPerInt']
    if min_num_per_int.strip() != '':
        if int(min_num_per_int) in (-1, 0):
            return ret
        else:
            return int(min_num_per_int) / 100.0
    return str(ret)
