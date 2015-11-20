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

from ..core.dag import BaseDagNode, DagEndpointType
from ..nodes.exporters import get_input_table_name, get_input_partitions, get_output_table_name, \
    get_output_table_partitions


class SplitNode(BaseDagNode):
    def __init__(self, percentage):
        super(SplitNode, self).__init__("split")
        self.marshal({
            "parameters": {
                "fraction": percentage,
            },
            "inputs": [(1, "input", DagEndpointType.DATA)],
            "outputs": [(1, "output1", DagEndpointType.DATA), (2, "output2", DagEndpointType.DATA)]
        })

        self.add_exporter("inputTableName", lambda context: get_input_table_name(context, self, "input"))
        self.add_exporter("inputTablePartitions", lambda context: get_input_partitions(context, self, "input"))
        self.add_exporter("output1TableName", lambda context: get_output_table_name(context, self, "output1"))
        self.add_exporter("output1TablePartition", lambda context: get_output_table_partitions(context, self, "output1"))
        self.add_exporter("output2TableName", lambda context: get_output_table_name(context, self, "output2"))
        self.add_exporter("output2TablePartition", lambda context: get_output_table_partitions(context, self, "output2"))
