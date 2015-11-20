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

from six import iterkeys


class DagException(Exception):
    pass


def topological_sort(adj_list, rev_adj_list):
    """
    :type adj_list: dict
    :type rev_adj_list: dict
    """
    zero_income_nodes = set(key for key in iterkeys(adj_list) if key not in rev_adj_list)
    out_list = []

    while len(zero_income_nodes) > 0:
        append_nodes = set()
        for node in zero_income_nodes:
            out_list.append(node)
            remove_nodes = []
            for linked_node in adj_list[node]:
                if linked_node in rev_adj_list:
                    rev_adj_list[linked_node].remove(node)
                    remove_nodes.append(linked_node)
                    if len(rev_adj_list[linked_node]) == 0:
                        del rev_adj_list[linked_node]
                        append_nodes.add(linked_node)
            for remove_node in remove_nodes:
                adj_list[node].remove(remove_node)
        zero_income_nodes = append_nodes
    if len(rev_adj_list) > 0:
        raise DagException()
    return out_list
