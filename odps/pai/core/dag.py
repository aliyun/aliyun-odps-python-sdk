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
from six import iteritems, itervalues, iterkeys
from six.moves import queue

from ..utils.dag_utils import DagException, topological_sort
from ..utils.odps_utils import fetch_table_fields


class DagEndpointType(object):
    DATA = 'DATA'
    MODEL = 'MODEL'


class BaseDagNode(object):
    reload_fields = False

    def __init__(self, code_name):
        self.parameters = dict()
        self.metas = dict()
        self.inputs = dict()
        self.outputs = dict()
        self.input_edges = dict()
        self.output_edges = dict()
        self.node_id = -1
        self.code_name = code_name
        self.virtual = False

        self.exported = set()
        self.exporters = dict()

        self.scheduled = False
        self.executed = False

    def get_parameter(self, name):
        return self.parameters[name]

    def set_parameter(self, name, value):
        self.parameters[name] = value

    def add_input_edge(self, edge, arg_name):
        if arg_name not in self.inputs:
            raise DagException()
        if arg_name not in self.input_edges:
            self.input_edges[arg_name] = []
        cur_sig = '%d-%s' % (edge.from_node.node_id, edge.from_arg)
        signatures = set('%d-%s' % (edge.from_node.node_id, edge.from_arg) for edge in self.input_edges[arg_name])
        if cur_sig not in signatures:
            self.input_edges[arg_name].append(edge)
            output_ep = edge.from_node.get_output_endpoint(edge.from_arg)
            self.inputs[arg_name].data_set_uuid = output_ep.data_set_uuid
            self.inputs[arg_name].model_uuid = output_ep.model_uuid

    def add_output_edge(self, edge, arg_name):
        if arg_name not in self.outputs:
            raise DagException()
        if arg_name not in self.output_edges:
            self.output_edges[arg_name] = []
        cur_sig = '%d-%s' % (edge.to_node.node_id, edge.to_arg)
        signatures = set('%d-%s' % (edge.to_node.node_id, edge.to_arg) for edge in self.output_edges[arg_name])
        if cur_sig not in signatures:
            self.output_edges[arg_name].append(edge)

    def remove_input_edge(self, edge, arg_name):
        if arg_name not in self.inputs or arg_name not in self.input_edges:
            raise DagException()
        self.input_edges[arg_name] = [v for v in self.input_edges[arg_name] if v != edge]

    def remove_output_edge(self, edge, arg_name):
        if arg_name not in self.outputs or arg_name not in self.output_edges:
            raise DagException()
        self.output_edges[arg_name] = [v for v in self.output_edges[arg_name] if v != edge]

    def get_input_endpoint(self, name):
        if name in self.inputs:
            return self.inputs[name]
        return None

    def get_output_endpoint(self, name):
        if name in self.outputs:
            return self.outputs[name]
        return None

    def marshal(self, def_dict):
        if "parameters" in def_dict:
            self.parameters.update(def_dict["parameters"])
        if "metas" in def_dict:
            self.metas.update(def_dict["metas"])
        if "inputs" in def_dict:
            self.inputs.update({obj[1]: DagEndpoint(obj[1], obj[0], obj[2], self) for obj in def_dict["inputs"]})
        if "outputs" in def_dict:
            self.outputs.update({obj[1]: DagEndpoint(obj[1], obj[0], obj[2], self) for obj in def_dict["outputs"]})
        if "exported" in def_dict:
            self.exported = def_dict["exported"]
        else:
            self.exported = set(iterkeys(self.parameters))

    def add_exporter(self, param, exporter):
        self.exporters[param] = exporter

    def gen_command_params(self, context):
        params = {k: self._format_parameter_value(v) for k, v in iteritems(self.parameters)
                  if k in self.exported and v is not None and v != ''}
        for name, exporter in iteritems(self.exporters):
            val = exporter(context)
            if bool(val):
                params[name] = self._format_parameter_value(val)
        return params

    def before_exec(self, context):
        self.scheduled = True
        self.executed = False

    def after_exec(self, context, is_success):
        if is_success:
            self.executed = True
            if self.reload_fields:
                for ep in itervalues(self.outputs):
                    if ep.type != DagEndpointType.DATA:
                        continue
                    ds = context._ds_container[ep.data_set_uuid]

                    old_field_defs = {f.name: f for f in ds._fields}
                    ds._fields = fetch_table_fields(context._odps, ds._table)
                    for f in ds._fields:
                        if f.name not in old_field_defs:
                            continue
                        f.role, f.continuity = old_field_defs[f.name].role, old_field_defs[f.name].continuity

    @staticmethod
    def _format_parameter_value(value):
        if isinstance(value, bool):
            if value:
                return 'true'
            else:
                return 'false'
        elif isinstance(value, (list, set)):
            return ','.join([BaseDagNode._format_parameter_value(v) for v in value])
        else:
            return str(value)

    def to_xml(self):
        node = Element("node", {"id": str(self.node_id), "codeName": self.code_name})
        params = SubElement(node, "parameters")
        for name, value in iteritems(self.parameters):
            param = SubElement(params, "parameter", {"name": name})
            param.text = self._format_parameter_value(value)
        inputs = SubElement(node, "inputs")
        for name, value in iteritems(self.input_edges):
            input = SubElement(inputs, "input", {
                "name": name,
                "seq": str(self.inputs[name].seq),
                "type": self.inputs[name].type,
            })
            if self.inputs[name].data_set_uuid is not None:
                input.attrib['data'] = str(self.inputs[name].data_set_uuid)
            if self.inputs[name].model_uuid is not None:
                input.attrib['model'] = str(self.inputs[name].model_uuid)
            for link in value:
                SubElement(input, "link", {
                    "node": str(link.from_node.node_id),
                    "arg": link.from_arg,
                })
        outputs = SubElement(node, "outputs")
        for name, value in iteritems(self.output_edges):
            output = SubElement(outputs, "output", {
                "name": name,
                "seq": str(self.outputs[name].seq),
                "type": self.outputs[name].type,
            })
            if self.outputs[name].data_set_uuid is not None:
                output.attrib['data'] = str(self.outputs[name].data_set_uuid)
            if self.outputs[name].model_uuid is not None:
                output.attrib['model'] = str(self.outputs[name].model_uuid)
            for link in value:
                SubElement(output, "link", {
                    "node": str(link.to_node.node_id),
                    "arg": link.to_arg,
                })
        self.after_create_xml(node)
        return node

    def after_create_xml(self, node):
        pass


class DagEdge(object):
    def __init__(self, from_node, from_arg, to_node, to_arg):
        self.from_node = from_node
        self.from_arg = from_arg
        self.to_node = to_node
        self.to_arg = to_arg


class DagEndpoint(object):
    def __init__(self, name, seq, endpoint_type, bind_node):
        self.name = name
        self.seq = seq
        # type should be values within DagEndpointType
        self.type = endpoint_type
        self.bind_node = bind_node
        self.data_set_uuid = None
        self.model_uuid = None


class PAIDag(object):
    def __init__(self):
        self._nodes = []
        self._node_seq_id = 1

    def add_node(self, dag_node):
        self._nodes.append(dag_node)
        dag_node.node_id = self._node_seq_id
        self._node_seq_id += 1

    @staticmethod
    def add_link(from_node, from_arg, to_node, to_arg):
        edge = DagEdge(from_node, from_arg, to_node, to_arg)
        from_node.add_output_edge(edge, from_arg)
        to_node.add_input_edge(edge, to_arg)

    @staticmethod
    def add_data_input(data_set, to_node, to_arg):
        PAIDag.add_link(data_set._bind_node, data_set._bind_output, to_node, to_arg)
        to_node.inputs[to_arg].data_set_uuid = data_set._data_uuid

    @staticmethod
    def add_model_input(model, to_node, to_arg):
        PAIDag.add_link(model._bind_node, model._bind_output, to_node, to_arg)
        to_node.inputs[to_arg].model_uuid = model._model_uuid

    def topological_sort(self):
        adj_list = dict()
        rev_adj_list = dict()

        for node in self._nodes:
            if len(node.input_edges) != 0:
                rev_adj_list[node] = set(in_edge.from_node for in_edge in self._extract_edges_from_dict(node.input_edges))
            adj_list[node] = set(out_edge.to_node for out_edge in self._extract_edges_from_dict(node.output_edges))
            if node in adj_list[node]:
                raise DagException()

        return topological_sort(adj_list, rev_adj_list)

    def get_ancestor_node_set(self, start_node):
        rev_adj_list = dict()
        for node in self._nodes:
            rev_adj_list[node] = [in_edge.from_node for in_edge in self._extract_edges_from_dict(node.input_edges)]
        visited = {start_node}
        node_queue = queue.Queue()
        node_queue.put(start_node)
        while not node_queue.empty():
            cur_node = node_queue.get()
            for up_node in rev_adj_list[cur_node]:
                if up_node not in visited:
                    visited.add(up_node)
                    node_queue.put(up_node)
        return visited

    @staticmethod
    def _extract_edges_from_dict(edge_dict):
        for edges in itervalues(edge_dict):
            for edge in edges:
                yield edge

    def to_xml(self):
        nodes = Element("nodes")
        for node in self.topological_sort():
            nodes.append(node.to_xml())
        return nodes
