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
import time
import weakref
from six import iteritems, itervalues

from .core.dag import DagEndpointType
from .utils.odps_utils import fetch_table_fields, is_table_exists, drop_table, drop_table_partition, set_table_lifecycle, \
    drop_offline_model

TEMP_TABLE_PREFIX = "tmp_p_"
TEMP_MODEL_PREFIX = "pm_"
logger = logging.getLogger(__name__)


class Runner(object):
    def __init__(self, context):
        self._context = weakref.ref(context)
        self._odps = context._odps
        self._managed_tables = []
        self._managed_parts = []
        self._managed_models = []

    def run(self, target_node=None):
        dry_run = self._context().get_config('execution.mock')
        self._managed_tables = []
        self._managed_parts = []
        self._managed_models = []

        nodes = self._context()._dag.topological_sort()
        self._fill_source_fields(nodes)
        self._fill_table_names(nodes)

        if target_node is not None:
            restrict = self._context()._dag.get_ancestor_node_set(target_node)
            nodes = [n for n in nodes if n in restrict]

        nodes = [n for n in nodes if not n.executed]

        if not dry_run:
            self._context()._managed_tables.update(filter(lambda tn: tn.startswith(TEMP_TABLE_PREFIX),
                                                          self._managed_tables))
            self._context()._managed_models.update(filter(lambda tn: tn.startswith(TEMP_MODEL_PREFIX),
                                                          self._managed_models))
            # drop existing tables
            for table_name in self._managed_tables:
                if is_table_exists(self._odps, table_name):
                    drop_table(self._odps, table_name, async=False)

            for table_name, part_name in self._managed_parts:
                if is_table_exists(self._odps, table_name):
                    drop_table_partition(self._odps, table_name, part_name, async=False)

            for model_name in self._managed_models:
                drop_offline_model(self._odps, model_name)

        for node in nodes:
            code_name, pai_project, params = self._gen_xflow_instance(node)
            if code_name is None:
                continue

            node.before_exec(self._context())

            logger.debug('Generated PAI command:\nPAI -name %s -project %s %s;\n' %
                         (code_name, pai_project,
                          ' '.join(['-D%s="%s"' % (k, v) for k, v in iteritems(params)])))
            if not dry_run:
                inst = self._odps.run_xflow(code_name, pai_project, params)
                inst.wait_for_success()

            node.after_exec(self._context(), True)

        lifecycle = self._context().get_config('temp.table.lifecycle')
        if not dry_run:
            for table_name in self._managed_tables:
                if table_name.startswith(TEMP_TABLE_PREFIX) and is_table_exists(self._odps, table_name):
                    set_table_lifecycle(self._odps, table_name, lifecycle)
        return True

    def _gen_model_name(self, code_name):
        ts = int(time.time()) % 99991
        m_name = TEMP_MODEL_PREFIX + code_name + '_' + str(ts)
        exc_len = len(code_name) - 32
        if len(code_name) >= exc_len > 0:
            truncated = code_name[0, len(code_name) - exc_len]
            m_name = TEMP_MODEL_PREFIX + truncated + '_' + str(ts)
        self._managed_models.append(m_name)
        return m_name

    def _gen_table_name(self, code_name, ts, seq):
        table_name = TEMP_TABLE_PREFIX + '%d_%s_%d' % (ts, code_name, seq)
        self._managed_tables.append(table_name)
        return table_name

    def _gen_upstream_params(self, initial_uuid):
        ds = self._context()._ds_container[initial_uuid]
        if ds._fields is not None:
            return ds._fields

        source_fields = [self._gen_upstream_params(ul._data_uuid) for ul in ds._uplink]
        for op in ds._operations:
            source_fields = [op.execute(source_fields)]

        ds._fields = source_fields[0]
        return ds

    def _fill_source_fields(self, nodes):
        ds_container = self._context()._ds_container
        for node in nodes:
            if not node.output_edges:
                continue
            for name, output in iteritems(node.outputs):
                if output.type == DagEndpointType.MODEL:
                    continue
                if node.code_name == "odps_source":
                    if "inputTableName" in node.parameters:
                        table_name = node.parameters["inputTableName"]
                        data_uuid = output.data_set_uuid
                        fields = fetch_table_fields(self._odps, table_name)
                        ds_container[data_uuid]._fields = fields

    def _fill_table_names(self, nodes):
        # first pass: table names for sources and dests
        for node in nodes:
            if node.code_name == "odps_source":
                table_name = node.parameters["inputTableName"] if "inputTableName" in node.parameters else None
                table_part = node.parameters["inputTablePartitions"] \
                    if "inputTablePartitions" in node.parameters else None
                for outputs in itervalues(node.output_edges):
                    for edge in outputs:
                        data_uuid = edge.to_node.inputs[edge.to_arg].data_set_uuid
                        ds = self._context()._ds_container[data_uuid]
                        ds._table, ds._partition = table_name, table_part
            elif node.code_name == "odps_target":
                table_name = node.parameters["outputTableName"] if "outputTableName" in node.parameters else None
                table_part = node.parameters["outputTablePartitions"] \
                    if "outputTablePartitions" in node.parameters else None
                executed = False
                for inputs in itervalues(node.input_edges):
                    for edge in inputs:
                        if edge.from_node.executed:
                            executed = True
                        data_uuid = edge.from_node.outputs[edge.from_arg].data_set_uuid
                        ds = self._context()._ds_container[data_uuid]
                        ds._table, ds._partition = table_name, table_part
                if not executed:
                    if table_part is None:
                        self._managed_tables.append(table_name)
                    else:
                        self._managed_parts.append((table_name, table_part))
            elif node.code_name == "model_source":
                model_name = node.parameters["modelName"]
                for outputs in itervalues(node.output_edges):
                    for edge in outputs:
                        model_uuid = edge.to_node.inputs[edge.to_arg].model_uuid
                        model = self._context()._model_container[model_uuid]
                        model._model_name = model_name
            elif node.code_name == "model_target":
                model_name = node.parameters["modelName"]
                executed = False
                for inputs in itervalues(node.input_edges):
                    for edge in inputs:
                        if edge.from_node.executed:
                            executed = True
                        model_uuid = edge.from_node.outputs[edge.from_arg].model_uuid
                        model = self._context()._model_container[model_uuid]
                        model._model_name = model_name
                if not executed:
                    self._managed_models.append(model_name)

        # second pass: assign nodes for other nodes
        for node in nodes:
            if node.code_name in {"odps_source", "odps_target", "model_source", "model_target"}:
                continue
            for output in itervalues(node.outputs):
                if output.type == DagEndpointType.MODEL:
                    model = self._context()._model_container[output.model_uuid]
                    if model._model_name is None:
                        model._model_name = self._gen_model_name(node.code_name)
                elif output.type == DagEndpointType.DATA:
                    ds = self._context()._ds_container[output.data_set_uuid]
                    if ds._table is None:
                        ts = int(time.time())
                        ds._table = self._gen_table_name(node.code_name, ts, output.seq)
                    if output.name not in node.output_edges:
                        continue
                    for edge in node.output_edges[output.name]:
                        target_ds_uuid = edge.to_node.inputs[edge.to_arg].data_set_uuid
                        self._context()._ds_container[target_ds_uuid]._table = ds._table

    def _gen_xflow_instance(self, node):
        pai_project = self._context()._config.get_pai_algorithm_project()
        if node.virtual:
            return None, None, None
        for ep in itervalues(node.outputs):
            if ep.type != DagEndpointType.DATA:
                continue
            self._gen_upstream_params(ep.data_set_uuid)
        return node.code_name, pai_project, node.gen_command_params(self._context())
