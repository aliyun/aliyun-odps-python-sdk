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
from pkg_resources import resource_string
from six import PY3

from .base_loader import BaseContentLoader


class FileContentLoader(BaseContentLoader):
    def load(self, env, name):
        cats = json.loads(self.load_resource_string('odps.pai.algorithms.defs', 'algo_cats.json'))
        cat_nodes = json.loads(self.load_resource_string('odps.pai.algorithms.defs', name + '.json'))
        self._load_algorithms(cat_nodes, cats[name], env)

    @staticmethod
    def load_resource_string(path, file):
        res_str = resource_string(path, file)
        if PY3:
            res_str = res_str.decode('UTF-8')
        return res_str
