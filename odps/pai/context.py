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
from xml.etree.ElementTree import Element

from ..core import ODPS
from .core.dag import PAIDag
from .datasets import DataSet, DataSetContainer
from .models import TrainedModel, ModelContainer
from .nodes import OdpsSourceNode, ModelSourceNode
from .runner import Runner
from .utils.odps_utils import fetch_table_fields, drop_table, drop_offline_model

logger = logging.getLogger(__name__)


class PAIConf(object):
    """
    Provide interface for api configuration
    """
    def __init__(self, file_name=None):

        def process_file_lines(file):
            for line in file:
                parts = line.split('#')
                if len(parts) == 0:
                    continue
                uncommented = parts[0].strip()
                if '=' in uncommented:
                    splited = uncommented.split('=', 2)
                    yield (splited[0].strip(), splited[1].strip())

        self._config_dict = {
            "temp.table.lifecycle": 14,
            "pai.algorithm.project": "algo_public"
        }
        if file_name is None:
            return
        conf_file = open(file_name, 'r')
        conf = {p[0].strip(): p[1].strip() for p in process_file_lines(conf_file)}
        conf_file.close()
        self._config_dict.update(conf)

    def __getitem__(self, item):
        if item not in self._config_dict:
            return None
        return self._config_dict[item]

    def __setitem__(self, item, value):
        self._config_dict[item] = value
        return self

    def get_odps_access_id(self):
        return self['odps.access.id']

    def set_odps_access_id(self, value):
        self['odps.access.id'] = value

    def get_odps_access_key(self):
        return self['odps.access.key']

    def set_odps_access_key(self, value):
        self['odps.access.key'] = value

    def get_odps_project_name(self):
        return self['odps.project.name']

    def set_odps_project_name(self, value):
        self['odps.project.name'] = value

    def get_odps_endpoint(self):
        return self['odps.endpoint']

    def set_odps_endpoint(self, value):
        self['odps.endpoint'] = value

    def get_pai_algorithm_project(self):
        return self['pai.algorithm.project']

    def set_pai_algorithm_project(self, value):
        self['pai.algorithm.project'] = value

    def get_mode(self):
        return self['mode']

    def set_mode(self, value):
        self['mode'] = value

    def update(self, conf):
        self._config_dict.update(conf)


class PAIContext(object):
    def __init__(self, config=None):
        if isinstance(config, PAIConf):
            self._config = config
        elif isinstance(config, ODPS):
            self._config = PAIConf()
            self._config.set_odps_access_id(config.account.access_id)
            self._config.set_odps_access_key(config.account.secret_access_key)
            self._config.set_odps_endpoint(config.endpoint)
            self._config.set_odps_project_name(config.project)

        self._dag = PAIDag()
        self._ds_container = DataSetContainer()
        self._model_container = ModelContainer()

        self._access_id = self._config.get_odps_access_id()
        self._access_key = self._config.get_odps_access_key()
        self._project_name = self._config.get_odps_project_name()
        self._endpoint = self._config.get_odps_endpoint()

        self._managed_tables = set()
        self._managed_models = set()

        # init odps
        self._odps = ODPS(self._access_id, self._access_key, project=self._project_name, endpoint=self._endpoint)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        for table_name in self._managed_tables:
            logger.debug('Cleaning up ' + table_name)
            drop_table(self._odps, table_name)

        for model_name in self._managed_models:
            logger.debug('Cleaning up ' + model_name)
            drop_offline_model(self._odps, model_name)

    def _to_xml(self):
        experiment = Element("experiment")
        experiment.append(self._dag.to_xml())
        experiment.append(self._ds_container.to_xml())
        experiment.append(self._model_container.to_xml())
        return experiment

    def get_config(self, item):
        return self._config[item]

    def set_config(self, item, value):
        self._config[item] = value

    def odps_data(self, table_name, partition=None):
        source_node = OdpsSourceNode(table_name, partition)
        self._dag.add_node(source_node)

        return DataSet(self, source_node.get_output_endpoint("output"),
                       fields=fetch_table_fields(self._odps, table_name))

    def odps_model(self, model_name):
        source_node = ModelSourceNode(model_name)
        self._dag.add_node(source_node)

        return TrainedModel(self, source_node.get_output_endpoint("model"))

    def _run(self, target_node=None):
        runner = Runner(self)
        runner.run(target_node)
