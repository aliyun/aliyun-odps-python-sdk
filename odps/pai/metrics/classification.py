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

from .nodes.classification import ConfusionMatrixNode, ROCCurveNode
from ..sink import Sink

logger = logging.getLogger(__name__)


def confusion_matrix(dataset, col_true, col_pred='prediction_result'):
    logger.debug('Operation step confusion_matrix called.')

    mat_sink, col_sink = Sink(), Sink()

    cm_node = ConfusionMatrixNode(col_true, col_pred, mat_sink, col_sink)
    dataset._context()._dag.add_node(cm_node)
    dataset._context()._dag.add_link(dataset._bind_node, dataset._bind_output, cm_node, "input")

    dataset._context()._run(cm_node)

    return mat_sink(), col_sink()


def roc_curve(dataset, pos_label, col_true, col_pred='prediction_result', col_scores='prediction_score'):
    logger.debug('Operation step roc_curve called.')

    fpr_sink, tpr_sink, thresholds_sink = Sink(), Sink(), Sink()

    roc_node = ROCCurveNode(col_true, col_pred, col_scores, pos_label, fpr_sink, tpr_sink, thresholds_sink)
    dataset._context()._dag.add_node(roc_node)
    dataset._context()._dag.add_link(dataset._bind_node, dataset._bind_output, roc_node, "input")

    dataset._context()._run(roc_node)

    return fpr_sink(), tpr_sink(), thresholds_sink()
