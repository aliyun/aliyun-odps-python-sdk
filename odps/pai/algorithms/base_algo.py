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

from six import itervalues

from ..utils.odps_utils import FieldParam, FieldRole
from ..core.dag import PAIDag
from .algorithm_nodes import IndependentAlgorithmNode, ProcessorNode, TrainNode
from ..datasets import DataSet
from ..models import TrainedModel


class BaseSupervisedAlgorithm(object):
    def __init__(self, name, parameters, ports, metas=None):
        self._name = name
        self._node = None
        self._parameters = parameters
        self._ports = ports
        self._metas = metas if metas is not None else {}

    def _create_node(self):
        if self._context().get_config("algorithm.behavior") == "independent":
            node = IndependentAlgorithmNode(self._name, self._parameters)
            self._dag.add_node(node)
            return node
        else:
            return ProcessorNode(self._name, self._parameters, self._ports, self._metas)

    @staticmethod
    def _fill_param(param_obj, def_vals, arg_name):
        if arg_name in def_vals:
            param_obj.value = def_vals[arg_name]
            return param_obj
        else:
            return param_obj

    def train(self, train_data):
        """
        :type train_data: DataSet
        """
        if not isinstance(train_data, DataSet):
            raise TypeError("Can only train model on data sets.")

        self._context = train_data._context
        self._dag = train_data._context()._dag
        node = self._create_node()

        if self._context().get_config("algorithm.behavior") == "independent":
            train_node = TrainNode(self._name)

            self._dag.add_node(train_node)
            self._dag.add_link(node, "model", train_node, "model")
            self._dag.add_data_input(train_data, train_node, "input")
            return TrainedModel(self._context(), train_node.get_output_endpoint("output"))
        else:
            self._dag.add_node(node)
            self._dag.add_data_input(train_data, node, "input")
            return TrainedModel(self._context(), node.get_output_endpoint("output"))
