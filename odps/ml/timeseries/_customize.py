# encoding: utf-8
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from functools import partial

from ..expr.exporters import get_input_field_names
from ..expr.mixin import ml_collection_mixin
from ...compat import Enum


class TimeSeriesFieldRole(Enum):
    TS_GROUP = 'TS_GROUP'
    TS_SEQ = 'TS_SEQ'
    TS_VALUE = 'TS_VALUE'


@ml_collection_mixin
class TimeSeriesMLMixIn(object):
    __slots__ = ()

    field_role_enum = TimeSeriesFieldRole
    non_feature_roles = set((TimeSeriesFieldRole.TS_GROUP, TimeSeriesFieldRole.TS_SEQ))


"""
Common time series exporters
"""
get_ts_group_column = partial(get_input_field_names, field_role=TimeSeriesFieldRole.TS_GROUP)
get_ts_seq_column = partial(get_input_field_names, field_role=TimeSeriesFieldRole.TS_SEQ)
get_ts_value_column = partial(get_input_field_names, field_role=TimeSeriesFieldRole.TS_VALUE)
