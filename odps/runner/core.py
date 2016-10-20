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

import fractions
import uuid
import json
import logging
from copy import deepcopy
from itertools import chain
from weakref import WeakValueDictionary, ref

from ..dag import DAG
from ..utils import gen_repr_object, underline_to_capitalized
from ..errors import NoSuchObject
from ..compat import StringIO, six
from .enums import EngineType, PortType
from .utils import JSONSerialClassEncoder
from .user_tb import get_user_stack

logger = logging.getLogger(__name__)


class BaseRunnerNode(object):
    """
    :type inputs: dict[str, RunnerPort]
    :type outputs: dict[str, RunnerPort]
    :type input_edges: dict[str, list[RunnerEdge]]
    :type output_edges: dict[str, list[RunnerEdge]]
    """

    def __init__(self, code_name, engine=EngineType.XFLOW):
        self.parameters = dict()
        self.metas = dict()
        self.inputs = dict()
        self.outputs = dict()
        self.input_edges = dict()
        self.output_edges = dict()
        self.node_id = -1
        self.code_name = code_name
        self.virtual = False

        self.engine = engine

        self.exported = set()
        self.exporters = dict()

        self.scheduled = False
        self.executed = False
        self.skipped = False

        self.reload_on_finish = False

        # store execution trace for troubleshooting
        self.traceback = get_user_stack()

        # test cases bind on nodes
        self.cases = []

        from .context import RunnerContext
        RunnerContext.instance()._dag.add_node(self)

    def add_input_edge(self, edge, arg_name):
        if arg_name not in self.input_edges:
            self.input_edges[arg_name] = []
        cur_sig = '%d-%s' % (edge.from_node.node_id, edge.from_arg)
        signatures = set('%d-%s' % (edge.from_node.node_id, edge.from_arg) for edge in self.input_edges[arg_name])
        if cur_sig not in signatures:
            self.input_edges[arg_name].append(edge)
            output_ep = edge.from_node.outputs.get(edge.from_arg)
            self.inputs[arg_name].obj_uuid = output_ep.obj_uuid
            self.inputs[arg_name].obj_uuid = output_ep.obj_uuid

    def add_output_edge(self, edge, arg_name):
        if arg_name not in self.output_edges:
            self.output_edges[arg_name] = []
        cur_sig = '%d-%s' % (edge.to_node.node_id, edge.to_arg)
        signatures = set('%d-%s' % (edge.to_node.node_id, edge.to_arg) for edge in self.output_edges[arg_name])
        if cur_sig not in signatures:
            self.output_edges[arg_name].append(edge)

    def marshal(self, def_dict):
        """
        :type def_dict:dict
        """
        def assemble_io(defs, ports):
            for obj in defs:
                if isinstance(obj, RunnerPort):
                    obj.bind_node = self
                    ports[obj.name] = obj
                else:
                    ports[obj[1]] = RunnerPort(self, obj[1], obj[0], obj[2], obj[3] if len(obj) >= 4 else True)

        if "parameters" in def_dict:
            self.parameters.update(def_dict["parameters"])
        if "metas" in def_dict:
            self.metas.update(def_dict["metas"])
        if "inputs" in def_dict:
            assemble_io(def_dict["inputs"], self.inputs)
        if "outputs" in def_dict:
            assemble_io(def_dict["outputs"], self.outputs)
        if "exported" in def_dict:
            self.exported = def_dict["exported"]
        else:
            self.exported = set(six.iterkeys(self.parameters))

    def add_exporter(self, param, exporter):
        self.exporters[param] = exporter

    def gen_command_params(self):
        params = dict((k, self._format_parameter_value(v)) for k, v in six.iteritems(self.parameters)
                      if k in self.exported and v is not None)

        for name, exporter in six.iteritems(self.exporters):
            try:
                val = self._format_parameter_value(exporter())
            except Exception as ex:
                raise ValueError('Failed to convert param %s: %s.' % (name, str(ex)))
            if val is not None:
                params[name] = val
            elif name in params:
                del params[name]
        return params

    def calc_node_hash(self, odps):
        param_hash = hash(json.dumps(self.parameters, cls=JSONSerialClassEncoder))

        input_dict = dict()
        for pn, p in six.iteritems(self.inputs):
            ds = p.obj
            if p.type == PortType.DATA and ds is not None:
                if ds.table is not None:
                    try:
                        time_str = odps.get_table(ds.table).last_modified_time.strftime('%Y-%m-%dT%H:%M:%S')
                        role_str = '%X' % hash(frozenset([(f.name, ','.join(r.name for r in f.role))
                                                          for f in ds._fields]))
                        input_dict[pn] = ds.table + '####' + time_str + '####' + role_str
                    except NoSuchObject:
                        input_dict[pn] = 'None'
                else:
                    input_dict[pn] = 'None'
            else:
                input_dict[pn] = 'None'
        port_hash = hash(frozenset(six.iteritems(input_dict)))
        return self.code_name + '_' + ('%X' % (hash((param_hash, port_hash)) & ((1 << 32) - 1)))

    def before_exec(self, odps, conv_params):
        self.executed = False

    def after_exec(self, odps, is_success):
        self.scheduled = False
        if not is_success:
            return
        if not self.skipped:
            self.executed = True
        else:
            self.skipped = False

    def optimize(self):
        return False, None

    @staticmethod
    def _format_parameter_value(value):
        if value is None:
            return None
        elif isinstance(value, bool):
            if value:
                return 'true'
            else:
                return 'false'
        elif isinstance(value, (list, set)):
            if not value:
                return None
            return ','.join([BaseRunnerNode._format_parameter_value(v) for v in value])
        else:
            return str(value)

    def _get_text(self, sel_ports=None):
        name_template = '{0}_{1}'
        sio = StringIO()

        if sel_ports is None:
            sel_ports = set()
        elif isinstance(sel_ports, six.string_types):
            sel_ports = set([sel_ports, ])
        else:
            sel_ports = set(sel_ports)

        sio.write(name_template.format(underline_to_capitalized(self.code_name), self.node_id))
        if self.inputs:
            sio.write('(' + ', '.join(p + '=' + name_template.format(underline_to_capitalized(e[0].from_node.code_name),
                                                                     e[0].from_node.node_id) + ':' + e[0].from_arg
                                      for p, e in sorted(list(six.iteritems(self.input_edges)), key=lambda v: v[0])) + ')')
        if self.outputs:
            sio.write(' -> ')
            sio.write(', '.join(n if n not in sel_ports else n + '(*)' for n in sorted(six.iterkeys(self.outputs))))
        return sio.getvalue()

    def _get_gv(self, sel_ports=None):
        node_template ='node{0} [shape=none,label=<\n<TABLE cellborder="0" cellpadding="0" cellspacing="1" bgcolor="{1}">\n{2}\n</TABLE>\n>];'
        io_template = '<TD align="center" colspan="{1}"><font point-size="8" color="{2}" port="{0}">{0}</font></TD>'
        text_template = '<TR><TD colspan="{0}">{1}</TD></TR>\n'

        if sel_ports is None:
            sel_ports = set()
        elif isinstance(sel_ports, six.string_types):
            sel_ports = set([sel_ports, ])
        else:
            sel_ports = set(sel_ports)
        sel_port_color = lambda c: 'red' if c in sel_ports else 'gray'

        sio = StringIO()

        # calculate cols needed
        inp_cols = len(self.inputs) if self.inputs else 1
        outp_cols = len(self.outputs) if self.outputs else 1
        tot_cols = inp_cols * outp_cols / fractions.gcd(inp_cols, outp_cols)

        # write input row
        if self.inputs:
            sio.write('<TR>')
            for input_ep in sorted(list(six.itervalues(self.inputs)), key=lambda p: p.seq):
                sio.write(io_template.format(input_ep.name, tot_cols / inp_cols, sel_port_color(input_ep.name)))
            sio.write('</TR>\n')
        # write code name row
        sio.write(text_template.format(tot_cols, underline_to_capitalized(self.code_name)))

        # write output row
        if self.outputs:
            sio.write('<TR>')
            for output_ep in sorted(list(six.itervalues(self.outputs)), key=lambda p: p.seq):
                sio.write(io_template.format(output_ep.name, tot_cols / outp_cols, sel_port_color(output_ep.name)))
            sio.write('</TR>\n')

        return node_template.format(self.node_id, 'azure' if self.executed or self.virtual else 'white', sio.getvalue())


class InputOutputNode(BaseRunnerNode):
    def propagate_names(self):
        raise NotImplementedError


class RunnerEdge(object):
    def __init__(self, from_node, from_arg, to_node, to_arg):
        """
        :type from_node: BaseRunnerNode
        :type to_node: BaseRunnerNode
        :type from_arg: str
        :type to_arg: str
        """
        self.from_node = from_node
        self.from_arg = from_arg
        self.to_node = to_node
        self.to_arg = to_arg

    def __eq__(self, other):
        return (self.from_node, self.from_arg, self.to_node, self.to_arg) == \
               (other.from_node, other.from_arg, other.to_node, other.to_arg)

    def __repr__(self):
        return '({0},{1},{2},{3})'.format(self.from_node, self.to_node, self.from_arg, self.to_arg)

    def _get_gv(self):
        link_template = 'node{0}:{1} -> node{2}:{3};'
        return link_template.format(self.from_node.node_id, self.from_arg, self.to_node.node_id, self.to_arg)


class RunnerPort(object):
    def __init__(self, bind_node=None, name=None, seq=None, port_type=None, required=None):
        """
        :type bind_node: BaseRunnerNode
        :type name: str
        :type seq: int
        :type port_type: PortType
        """
        self.name = name
        self.seq = seq
        self.type = port_type
        self.bind_node = bind_node
        self.obj_uuid = None
        self.required = required

    def __repr__(self):
        r = "Port('%s', %s, " % (self.name, self.type.name)
        if self.type == PortType.DATA:
            r += 'DFAdapter(' + str(self.obj_uuid) + ')'
        else:
            r += 'TrainedModel(' + str(self.obj_uuid) + ')'
        r += ')'
        return r

    @property
    def obj(self):
        from ..runner import RunnerContext
        return RunnerContext.instance()._obj_container.get(self.obj_uuid)


class RunnerDAG(DAG):
    """
    :type _nodes: list[BaseRunnerNode]
    """
    def __init__(self):
        super(RunnerDAG, self).__init__()
        self._node_seq_id = 1

    def add_node(self, node):
        super(RunnerDAG, self).add_node(node)
        node.node_id = self._node_seq_id
        self._node_seq_id += 1

    def add_link(self, from_node, from_arg, to_node, to_arg):
        self.add_edge(from_node, to_node, validate=False)
        edge = RunnerEdge(from_node, from_arg, to_node, to_arg)
        from_node.add_output_edge(edge, from_arg)
        to_node.add_input_edge(edge, to_arg)

    def add_obj_input(self, obj, to_node, to_arg):
        self.add_link(obj._bind_node, obj._bind_output, to_node, to_arg)
        to_node.inputs[to_arg].obj_uuid = obj._obj_uuid


_required_sink = dict()


class ObjectContainer(DAG):
    def __init__(self):
        super(ObjectContainer, self).__init__()
        self._container = WeakValueDictionary()
        self._map = WeakValueDictionary()

    def __getitem__(self, uuid):
        if uuid is None:
            raise KeyError()
        return self._container[str(uuid)]

    def __contains__(self, item):
        if item is None:
            return False
        return str(item) in self._container

    def get(self, item, default=None):
        return default if item not in self else self[item]

    def items(self):
        return six.iteritems(self._container)

    def remove(self, uuid):
        if uuid is None:
            raise KeyError()
        del self._container[str(uuid)]

    def register(self, obj):
        self._container[str(obj._obj_uuid)] = obj
        if hasattr(obj, '_uplink'):
            self.add_node(obj)
            [self.add_edge(pred, obj) for pred in obj._uplink]


class ObjectDescription(object):
    def __init__(self, tables=None, offline_models=None, **kwargs):
        self.tables = self.process_obj(tables)
        self.offline_models = self.process_obj(offline_models)
        for k, v in six.iteritems(kwargs):
            setattr(self, k, deepcopy(v))

    @staticmethod
    def process_obj(obj):
        if obj is None:
            return dict()
        elif isinstance(obj, dict):
            return obj
        elif isinstance(obj, (list, set)):
            return dict(enumerate(obj))
        else:
            return {0: obj}


class RunnerObject(object):
    """
    Base class for data objects in PyODPS ML, including models and data sets.
    """
    def __new__(cls, *args, **kwargs):
        cls_path = cls.__module__ + '.' + cls.__name__
        obj = object.__new__(cls)
        obj._cls_path = cls_path
        return obj

    def __init__(self, odps, port, *args, **kwargs):
        self._odps = odps
        self._obj_uuid = uuid.uuid4()
        self._bind_port = port
        self._bind_node = port.bind_node
        self._bind_output = port.name

        from .context import RunnerContext
        self._container = ref(RunnerContext.instance()._obj_container)

        # include related objects to prevent inputs being collected by gc.
        self._dependencies = list(self._fetch_upstream_depends())
        if port.required:
            _required_sink[self._obj_uuid] = self

    def _fetch_upstream_depends(self):
        from .context import RunnerContext
        context = RunnerContext.instance()
        for port in six.itervalues(self._bind_node.inputs):
            # Some nodes, like SQL, might have null inputs
            if not port.obj_uuid:
                continue
            if port.obj_uuid in context._obj_container:
                for o in context._obj_container[port.obj_uuid]._iter_linked_objs():
                    yield o

    def _iter_linked_objs(self):
        yield self

    def gen_temp_names(self):
        raise NotImplementedError

    def describe(self):
        raise NotImplementedError

    def fill(self, desc):
        raise NotImplementedError

    def reload(self):
        pass

    def show_steps(self):
        from .context import RunnerContext
        ancestors = list(reversed([self._bind_node, ] + RunnerContext.instance()._dag.ancestors(self._bind_node)))
        ancestors.sort(key=lambda n: n.node_id)
        node_set = set(ancestors)
        edge_list = []

        sio = StringIO()
        for node in ancestors:
            sio.write(node._get_text(self._bind_output if node == self._bind_node else None))
            sio.write('\n')
        text = sio.getvalue()

        sio = StringIO()
        sio.write('digraph {\n')
        for node in ancestors:
            sio.write(node._get_gv(self._bind_output if node == self._bind_node else None))
            for e in chain(*list(six.itervalues(node.input_edges))):
                if e.from_node in node_set and e.to_node in node_set:
                    edge_list.append(e)
            sio.write('\n')
        for edge in edge_list:
            sio.write(edge._get_gv())
            sio.write('\n')
        sio.write('}')
        gv = sio.getvalue()

        return gen_repr_object(gv=gv, text=text)

    def _add_case(self, case):
        self._bind_node.cases.append(case)
        return self

    def _link_node(self, node, input_port):
        from .context import RunnerContext
        dag = RunnerContext.instance()._dag
        dag.add_obj_input(self, node, input_port)
