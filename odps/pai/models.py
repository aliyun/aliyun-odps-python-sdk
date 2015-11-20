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

import logging
import uuid
import weakref
from xml.etree.ElementTree import Element
from six import itervalues

from .algorithms.algorithm_nodes import PredictNode
from .datasets import DataSet
from .nodes.io_nodes import ModelTargetNode
from .utils.odps_utils import FieldParam, FieldRole, FieldContinuity

logger = logging.getLogger(__name__)


class ModelContainer(object):
    def __init__(self):
        self._container = dict()

    def register(self, model):
        self._container[str(model._model_uuid)] = model

    def __getitem__(self, uuid):
        return self._container[str(uuid)]

    def items(self):
        return itervalues(self._container)

    def remove(self, uuid):
        del self._container[str(uuid)]

    def to_xml(self):
        models = Element('models')
        for node in itervalues(self._container):
            models.append(node.to_xml())
        return models


class TrainedModel(object):
    def __init__(self, context, endpoint):
        self._context = weakref.ref(context)
        self._model_uuid = uuid.uuid4()
        self._bind_endpoint = endpoint
        self._bind_node = endpoint.bind_node
        self._bind_output = endpoint.name

        self._model_name = None

        context._model_container.register(self)
        endpoint.model_uuid = self._model_uuid

    def store_odps(self, model_name):
        logger.debug('Operation step TrainedModel.store_odps(\'%s\') called.' % model_name)

        model_node = ModelTargetNode(model_name)
        self._context()._dag.add_node(model_node)
        self._context()._dag.add_link(self._bind_node, self._bind_output, model_node, "model")

        self._context()._run(self._bind_node)

    def predict(self, data_set):
        if not isinstance(data_set, DataSet):
            raise TypeError("Cannot predict on objects other than a data set.")

        predict_node = PredictNode()

        self._context()._dag.add_node(predict_node)
        self._context()._dag.add_model_input(self, predict_node, "model")
        self._context()._dag.add_link(data_set._bind_node, data_set._bind_output, predict_node, "input")

        output_data_set = DataSet(self._context(), predict_node.get_output_endpoint("output"), uplink=[data_set],
                                  fields=data_set._fields)
        output_data_set = output_data_set._append_fields([
            FieldParam('predict_result', 'string', None, FieldContinuity.DISCRETE, True),
            FieldParam('predict_score', 'double', FieldRole.FEATURE, FieldContinuity.CONTINUOUS, True),
            FieldParam('predict_detail', 'string', None, None, True),
        ])
        return output_data_set

    def rebuild(self):
        new_model = self._context().odps_model(self._model_name)
        self._bind_endpoint = new_model._bind_endpoint
        self._bind_node = new_model._bind_node
        self._bind_output = new_model._bind_output

        self._context()._model_container.remove(new_model._model_uuid)

    def to_xml(self):
        return Element('model', {
            'uuid': str(self._model_uuid)
        })
