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

from xml.etree.ElementTree import Element
from six import itervalues

from ..utils.dag_utils import topological_sort, DagException


class DataSetContainer(object):
    def __init__(self):
        self._container = dict()

    def register(self, data_set):
        self._container[str(data_set._data_uuid)] = data_set

    def __getitem__(self, uuid):
        return self._container[str(uuid)]

    def items(self):
        return itervalues(self._container)

    def remove(self, uuid):
        del self._container[str(uuid)]

    def topological_sort(self):
        adj_list = dict()
        rev_adj_list = dict()

        for node in itervalues(self._container):
            if len(node._uplink) != 0:
                rev_adj_list[node] = set(node._uplink)
                if node in rev_adj_list[node]:
                    raise DagException()
            if node not in adj_list:
                adj_list[node] = set()
            for up_node in node._uplink:
                if up_node not in adj_list:
                    adj_list[up_node] = set()
                adj_list[up_node].add(node)

        return topological_sort(adj_list, rev_adj_list)

    def to_xml(self):
        datasets = Element('datasets')
        for node in self.topological_sort():
            datasets.append(node.to_xml())
        return datasets
