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

from ...utils.odps_utils import drop_table
from ...core.dag import BaseDagNode

logger = logging.getLogger(__name__)


class MetricNode(BaseDagNode):
    def __init__(self, name):
        super(MetricNode, self).__init__(name)
        self._data_table_name = None

    def after_exec(self, context, is_success):
        super(MetricNode, self).after_exec(context, is_success)
        self.calc_metrics(context)

    def calc_metrics(self, context):
        try:
            drop_table(context._odps, self._data_table_name)
        except Exception as ex:
            logger.warn(ex)
