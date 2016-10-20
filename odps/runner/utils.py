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

import time
from json import JSONEncoder

from ..config import options
from ..utils import TEMP_TABLE_PREFIX as GLOBAL_TEMP_PREFIX, camel_to_underline

TABLE_MODEL_PREFIX = 'otm_'
TABLE_MODEL_SEPARATOR = '__'
TEMP_TABLE_PREFIX = GLOBAL_TEMP_PREFIX + 'ml_'
TEMP_MODEL_PREFIX = 'pm_'


class JSONSerialClassEncoder(JSONEncoder):
    def default(self, o):
        if hasattr(o, 'serial'):
            return o.serial()
        else:
            super(JSONSerialClassEncoder, self).default(o)


def is_temp_table(table_name):
    return table_name.startswith(TEMP_TABLE_PREFIX)


def is_temp_model(model_name):
    return model_name.startswith(TEMP_MODEL_PREFIX)


def split_project_name(name):
    if '.' in name:
        project, name = name.split('.', 1)
    else:
        project = None
    return project, name


def gen_table_name(code_name, ts=None, node_id=None, seq=None, suffix=None):
    if ts is None:
        ts = int(time.time()) if not options.runner.dry_run else 0
    table_name = TEMP_TABLE_PREFIX + '%d_%s' % (ts, camel_to_underline(code_name))
    if node_id:
        table_name += '_' + str(node_id)
    if seq:
        table_name += '_' + str(int(seq))
    if suffix:
        table_name += '_' + suffix
    return table_name
