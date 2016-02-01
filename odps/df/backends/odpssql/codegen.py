#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import os
import base64

from .types import df_type_to_odps_type
from .cloudpickle import dumps
from ....compat import OrderedDict
from ....utils import to_str

dirname = os.path.dirname(os.path.abspath(__file__))
CLOUD_PICKLE_FILE = os.path.join(dirname, 'cloudpickle.py')
with open(CLOUD_PICKLE_FILE) as f:
    CLOUD_PICKLE = f.read()

UDF_TMPL = '''\
%(cloudpickle)s

import base64

from odps.udf import annotate

@annotate('%(from_type)s->%(to_type)s')
class %(func_cls_name)s(object):

    def __init__(self):
        encoded = '%(func_str)s'
        f_str = base64.b64decode(encoded)
        self.f = loads(f_str)

    def evaluate(self, arg):
        return self.f(arg)
'''


def gen_udf(expr, func_cls_name=None):
    func_to_udfs = OrderedDict()

    for node in expr.traverse(unique=True):
        func = getattr(node, '_func', None)
        if func is not None:
            func_to_udfs[func] = UDF_TMPL % {
                'cloudpickle': CLOUD_PICKLE,
                'from_type':  df_type_to_odps_type(node.input_type).name,
                'to_type': df_type_to_odps_type(node.data_type).name,
                'func_cls_name': func_cls_name,
                'func_str': to_str(base64.b64encode(dumps(func)))
            }

    return func_to_udfs
