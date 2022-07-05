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

from ..expr.exporters import get_ml_input


"""
K-Means related exporters
"""


def get_kmeans_input_append_col_idx(expr, input_name):
    data_obj = get_ml_input(expr, input_name)
    if data_obj is None:
        return None
    return ','.join([str(idx) for idx, f in enumerate(data_obj._ml_fields) if not f.is_partition])
