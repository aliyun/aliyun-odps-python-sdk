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

from ...utils.odps_utils import FieldParam


class DataSetOperation(object):
    def execute(self, fields):
        pass

    def to_xml(self):
        return None

    @staticmethod
    def _transform_fields(fields, name_set, role):
        for f in fields:
            if f.name in name_set:
                yield FieldParam.copy(f, role)
            elif f.role == role:
                yield FieldParam.copy(f, None)
            else:
                yield deepcopy(f)
