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

from xml.etree.ElementTree import Element, SubElement

from six import iteritems
from ...utils.odps_utils import FieldRole
from .base_ops import DataSetOperation


class SelectFeatureDataSetOperation(DataSetOperation):
    def __init__(self, fields):
        self.feature_names = fields

    def execute(self, fields):
        feature_set = set(self.feature_names)
        return list(self._transform_fields(fields[0], feature_set, FieldRole.FEATURE))

    def to_xml(self):
        node = Element('select_feature')
        node.text = ','.join(self.feature_names)
        return node


class ExcludeFeatureDataSetOperation(DataSetOperation):
    def __init__(self, fields):
        self.exclude_names = fields

    def execute(self, fields):
        feature_set = set(self.exclude_names)
        return list(self._transform_fields(fields[0], feature_set, None))

    def to_xml(self):
        node = Element('exclude_feature')
        node.text = ','.join(self.exclude_names)
        return node


class LabelDataSetOperation(DataSetOperation):
    def __init__(self, field):
        self.label_name = field

    def execute(self, fields):
        feature_set = {self.label_name}
        return list(self._transform_fields(fields[0], feature_set, FieldRole.LABEL))

    def to_xml(self):
        node = Element('label')
        node.text = self.label_name
        return node


class WeightDataSetOperation(DataSetOperation):
    def __init__(self, field):
        self.weight_name = field

    def execute(self, fields):
        feature_set = {self.weight_name}
        return list(self._transform_fields(fields[0], feature_set, FieldRole.WEIGHT))

    def to_xml(self):
        node = Element('weight')
        node.text = self.weight_name
        return node


class FieldContinuityDataSetOperation(DataSetOperation):
    def __init__(self, continuity):
        self.continuity = continuity

    def execute(self, fields):
        for f in fields[0]:
            if f.name in self.continuity:
                f.continuity = self.continuity[f.name]

    def to_xml(self):
        node = Element('continuity')
        for k, v in iteritems(self.continuity):
            SubElement(node, 'field', {'field': k, 'continuity': v})
        return node
