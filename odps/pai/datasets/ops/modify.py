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

from copy import deepcopy
from xml.etree.ElementTree import Element

from ...utils.odps_utils import FieldParam
from .base_ops import DataSetOperation


class StaticFieldChangeOperation(DataSetOperation):
    def __init__(self, fields, is_append=False):
        self.fields = fields
        self.is_append = is_append

    def execute(self, existing_fields):
        dup_fields = [FieldParam.copy(f, None) for f in self.fields]
        if self.is_append:
            src = list([deepcopy(f) for f in existing_fields[0]])
            src.extend(dup_fields)
            return src
        else:
            return self.fields

    def to_xml(self):
        node = Element('fields')
        for f in self.fields:
            node.append(f.to_xml())
        return node


class DynamicFieldChangeOperation(DataSetOperation):
    def __init__(self, evaluator, params, is_append=False):
        def gen_field(field_str):
            parts = field_str.split(':')
            return FieldParam(parts[0].strip(), parts[1].strip(), None)
        self.fields = [gen_field(fstr) for fstr in evaluator(params).split(',')]
        self.is_append = is_append

    def execute(self, existing_fields):
        dup_fields = [FieldParam.copy(f, None) for f in self.fields]
        if self.is_append:
            src = list([deepcopy(f) for f in existing_fields[0]])
            src.extend(dup_fields)
            return src
        else:
            return self.fields
