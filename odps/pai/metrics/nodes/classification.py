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

import json
from time import time
from six import iteritems

from ....tunnel import TableTunnel
from .metrics_base import MetricNode
from ...nodes.exporters import get_input_table_name, get_input_partitions
from ...core.dag import DagEndpointType


class ConfusionMatrixNode(MetricNode):
    def __init__(self, col_true, col_pred, mat_sink, col_sink):
        super(ConfusionMatrixNode, self).__init__("confusionmatrix")

        def gen_cm_output_table_name(context):
            self._data_table_name = 'tmp_p_cm_%d' % int(time())
            return self._data_table_name

        self.marshal({
            "parameters": {
                "labelColName": col_true,
                "predictionColName": col_pred
            },
            "inputs": [(1, "input", DagEndpointType.DATA)]
        })

        self.add_exporter("inputTableName", lambda context: get_input_table_name(context, self, "input"))
        self.add_exporter("inputTablePartitions", lambda context: get_input_partitions(context, self, "input"))
        self.add_exporter("outputTableName", lambda context: gen_cm_output_table_name(context))

        self._mat_sink = mat_sink
        self._col_sink = col_sink

    def calc_metrics(self, context):
        tunnel = TableTunnel(context._odps)
        down_session = tunnel.create_download_session(self._data_table_name)
        # skip the first row
        reader = down_session.open_record_reader(0, 100)
        col_data = json.loads(reader.read().values[0])
        col_data = map(lambda p: p[1], sorted(iteritems(col_data), key=lambda a: int(a[0][1:])))
        self._col_sink.put(col_data)

        import numpy as np
        mat = np.matrix([rec[2:] for rec in reader.reads()])
        self._mat_sink.put(mat)

        super(ConfusionMatrixNode, self).calc_metrics(context)


class ROCCurveNode(MetricNode):
    def __init__(self, col_true, col_pred, col_score, pos_label, fpr_sink, tpr_sink, threshold_sink):
        super(ROCCurveNode, self).__init__("roc")

        def gen_roc_output_table_name(context):
            self._data_table_name = 'tmp_p_roc_%d' % int(time())
            return self._data_table_name

        self.marshal({
            "parameters": {
                "labelColName": col_true,
                "predictionColName": col_pred,
                "predictionScoreName": col_score,
                "goodValue": pos_label
            },
            "inputs": [(1, "input", DagEndpointType.DATA)]
        })

        self.add_exporter("inputTableName", lambda context: get_input_table_name(context, self, "input"))
        self.add_exporter("inputTablePartitions", lambda context: get_input_partitions(context, self, "input"))
        self.add_exporter("outputTableName", lambda context: gen_roc_output_table_name(context))

        self._fpr_sink = fpr_sink
        self._tpr_sink = tpr_sink
        self._thresh_sink = threshold_sink

    def calc_metrics(self, context):
        tunnel = TableTunnel(context._odps)
        down_session = tunnel.create_download_session(self._data_table_name)

        reader = down_session.open_record_reader(0, 1000)
        import numpy as np
        mat = np.matrix([rec.values for rec in reader.reads()])
        thresh = mat[:, 0]
        tp, fn, tn, fp = mat[:, 1], mat[:, 2], mat[:, 3], mat[:, 4]

        self._tpr_sink.put(np.squeeze(np.asarray(tp * 1.0 / (tp + fn))))
        self._fpr_sink.put(np.squeeze(np.asarray(fp * 1.0 / (fp + tn))))
        self._thresh_sink.put(np.squeeze(np.asarray(thresh)))

        super(ROCCurveNode, self).calc_metrics(context)

