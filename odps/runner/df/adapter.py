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

import copy
import itertools
import threading
import weakref

from ...models import Table
from ...df.core import DataFrame, CollectionExpr
from ...df.utils import to_collection as to_df_collection
from ...df.expr.core import ExprDictionary
from ...df.expr.expressions import SequenceExpr, Scalar
from ...df.expr.expressions import FilterPartitionCollectionExpr
from ...compat import six, reduce
from ..core import BaseRunnerNode, RunnerObject, ObjectDescription, EngineType, PortType
from ..utils import gen_table_name

_df_endpoint_dict = ExprDictionary()
_df_link_maintainer = ExprDictionary()


class DFMockProject(object):
    def __init__(self):
        self.name = 'mocked_project'


class DFMockTable(Table):
    def __init__(self, **kwargs):
        super(DFMockTable, self).__init__(**kwargs)
        self._loaded = True

    @property
    def project(self):
        return DFMockProject()


class DFNode(BaseRunnerNode):
    def __init__(self, input_num=1):
        super(DFNode, self).__init__("DataFrame", engine=EngineType.DF)
        self.marshal({
            'inputs': [(idx, 'input%d' % idx, PortType.DATA) for idx in range(1, input_num + 1)],
            'outputs': [(1, 'output', PortType.DATA)]
        })

    def optimize(self):
        # feed cache_data backwards
        for nm, inp in six.iteritems(self.inputs):
            obj = inp.obj
            if not hasattr(obj, 'df'):
                continue
            if obj.df is None or obj.df._cache_data is None:
                continue
            for edge in self.input_edges[nm]:
                src_output = edge.from_node.outputs[edge.from_arg]
                if src_output.obj and src_output.obj.df._cache_data is None:
                    src_output.obj.df._cache_data = obj.df._cache_data

        if len(self.outputs) != 1:
            return False, None

        out_port = six.next(six.itervalues(self.outputs))
        ep = out_port.obj
        if ep is None:
            return False, None
        df = ep.df
        if not self.inputs and isinstance(df, DataFrame):
            # direct table input
            data_sources = list(df.data_source())
            if len(data_sources) == 1 and hasattr(data_sources[0], '_client'):
                ep.table = data_sources[0].name
                return True, None
        elif not self.inputs and isinstance(df, FilterPartitionCollectionExpr) and isinstance(ep.df.input, DataFrame):
            # direct partitioned table input
            data_sources = list(df.input.data_source())
            if len(data_sources) == 1 and hasattr(data_sources[0], '_client'):
                ep.table = data_sources[0].name
                ep.partitions = ep.df.predicate_string
                return True, None
        elif isinstance(df._cache_data, Table):
            # cached data input
            ep.table = df._cache_data.name
            return True, None
        return False, None


class DFExecNode(DFNode):
    def __init__(self, bind_df=None, input_num=1, func=None, args=None, kwargs=None):
        super(DFExecNode, self).__init__(input_num)
        self.parameters.update(dict(args_hash=hash(frozenset(args)), kwargs_hash=hash(frozenset(six.iteritems(kwargs)))))

        self.bind_df = bind_df
        self.args = args
        self.kwargs = kwargs
        self.func = func
        self.sink = None

    def optimize(self):
        if len(self.args) != 2 or len(self.inputs) != 1:
            return False, None
        ep = self.inputs['input1'].obj
        if ep is not None:
            # direct df output
            if self.func.__name__ == 'persist' and self.bind_df == ep.df:
                tables = []
                for pep in ep._get_adapter_chain():
                    pep.table = self.args[1]
                    tables.append(pep.table)
                return True, ObjectDescription(tables=tables, node_id=id(ep._bind_node))
        return False, None


def adapter_from_df(df, odps=None, skip_orphan=False):
    if df in _df_endpoint_dict:
        return _df_endpoint_dict[df]
    else:
        closest_links = list(df.to_dag(False).closest_ancestors(df, lambda d: d in _df_endpoint_dict))
        DFAdapter._add_df_link(df, *closest_links)
        input_eps = [_df_endpoint_dict.get(f) for f in closest_links]

        if skip_orphan and not input_eps:
            return None

        node = DFNode(len(input_eps))
        for idx, inp_ep in enumerate(input_eps):
            inp_ep._link_node(node, 'input%d' % (idx + 1))

        try:
            odps = odps or six.next(df.data_source()).odps
        except StopIteration:
            from ...inter import enter, InteractiveError
            try:
                odps = enter().odps
            except (InteractiveError, AttributeError):
                import warnings
                warnings.warn('No ODPS object available in rooms. Further actions might lead to errors.',
                              RuntimeWarning)
        return DFAdapter(odps, node.outputs['output'], df, uplink=input_eps)


def convert_df_args(arg):
    if arg is None:
        return None
    if isinstance(arg, (CollectionExpr, SequenceExpr)):
        return adapter_from_df(arg)
    if isinstance(arg, dict):
        return dict((k, convert_df_args(v)) for k, v in six.iteritems(arg))
    elif isinstance(arg, list):
        return [convert_df_args(v) for v in arg]
    elif isinstance(arg, tuple):
        return tuple(convert_df_args(v) for v in arg)
    elif isinstance(arg, set):
        return set(convert_df_args(v) for v in arg)
    else:
        return arg


def extract_df_inputs(o):
    if isinstance(o, (CollectionExpr, SequenceExpr, Scalar)):
        yield o
    elif isinstance(o, dict):
        for v in itertools.chain(*(extract_df_inputs(dv) for dv in six.itervalues(o))):
            if v is not None:
                yield v
    elif isinstance(o, (list, set, tuple)):
        for v in itertools.chain(*(extract_df_inputs(dv) for dv in o)):
            if v is not None:
                yield v
    else:
        yield None


class PartitionSelection(object):
    def __init__(self, part_def):

        if isinstance(part_def, six.string_types):
            self.parts = [[self._parse_sub_part(part) for part in one_part.split('/')] for one_part in part_def.split(',')]
        else:
            def parse_single_part(part_repr):
                if isinstance(part_repr, six.string_types):
                    for sub_part in part_repr.split('/'):
                        yield self._parse_sub_part(sub_part)
                else:
                    for sub_part in part_repr:
                        if isinstance(sub_part, six.string_types):
                            yield self._parse_sub_part(sub_part)
                        else:
                            yield sub_part

            self.parts = [list(parse_single_part(part_repr)) for part_repr in part_def]

    @staticmethod
    def _parse_sub_part(p):
        parts = p.strip().split('=', 1)
        if parts[1].startswith('\'') or parts[1].startswith('\"'):
            parts[1] = parts[1].strip('"\'').decode('string-escape')
        else:
            parts[1] = int(parts[1])
        return parts

    @staticmethod
    def _repr_sub_part(p):
        parts = copy.deepcopy(p)
        if isinstance(parts[1], six.string_types):
            parts[1] = '\"{0}\"'.format(str(parts[1]))
        else:
            parts[1] = str(parts[1])
        return '='.join(parts)

    def __iter__(self):
        return iter(self.parts)

    def __getitem__(self, item):
        return self.parts[item]

    def __repr__(self):
        return ','.join('/'.join(self._repr_sub_part(part) for part in one_part) for one_part in self.parts)

    def to_sql_condition(self):
        return '(' + ') or ('.join(' and '.join(self._repr_sub_part(part) for part in one_part) for one_part in self.parts) + ')'

    def to_partition_fields(self):
        return list(reduce(lambda a, b: a + b, map(lambda a: [a[0], ], self.parts), []))

    def to_partition_spec(self, pid):
        return ','.join('='.join(a) for a in self.parts[pid])


class DFAdapter(RunnerObject):
    def __init__(self, odps, port, df, **kw):
        super(DFAdapter, self).__init__(odps, port)
        self._df_ref = weakref.ref(df) if df is not None else None
        self._uplink = kw.pop('uplink', [])
        self._operations = []
        self._table = None
        self._partitions = None

        if df is not None:
            _df_endpoint_dict[df] = self

        from ..context import RunnerContext
        RunnerContext.instance()._obj_container.register(self)

        if port.obj_uuid is None:
            port.obj_uuid = self._obj_uuid

        if hasattr(self, 'init_df'):
            self.init_df(self.df, **kw)

    @staticmethod
    def _add_df_link(df, *depends):
        if not depends:
            return
        if df not in _df_link_maintainer:
            _df_link_maintainer[df] = set()
        _df_link_maintainer[df] |= set(depends)

    @staticmethod
    def _build_mock_table(table_name, schema):
        return DFMockTable(name=table_name, schema=schema)

    def gen_temp_names(self):
        if not self.table:
            self.table = gen_table_name(self._bind_node.code_name, node_id=self._bind_node.node_id,
                                        seq=self._bind_port.seq)
            return ObjectDescription(tables=[self.table, ])
        else:
            return None

    def _get_adapter_chain(self):
        if len(self._uplink) == 1:
            upds = self._uplink[0]
            if (upds._bind_node, upds._bind_port) == (self._bind_node, self._bind_port):
                chain = upds._get_adapter_chain() + [self, ]
            else:
                chain = [self, ]
        else:
            chain = [self, ]
        return chain

    def describe(self):
        if self._partitions is None:
            table_desc = self.table
        else:
            table_desc = (self.table, self.partitions)
        return ObjectDescription(tables=table_desc, fields=self._fields)

    def fill(self, desc):
        if desc.tables:
            if isinstance(desc.tables[0], tuple):
                self.table, self.partitions = desc.tables[0]
            else:
                self.table, self.partitions = desc.tables[0], None
        if desc.fields:
            self._fields = desc.fields
        self.df_from_fields(force_create=True)

    @property
    def table(self):
        if self._table is not None:
            return self._table
        elif len(self._uplink) == 1:
            upds = self._uplink[0]
            if (upds._bind_node, upds._bind_port) == (self._bind_node, self._bind_port):
                return upds.table
            else:
                return None
        else:
            return None

    @table.setter
    def table(self, value):
        self._table = value

    @property
    def partitions(self):
        if self._partitions is not None:
            return self._partitions
        elif len(self._uplink) == 1:
            upds = self._uplink[0]
            if (upds._bind_node, upds._bind_port) == (self._bind_node, self._bind_port):
                return upds.partitions
            else:
                return None
        else:
            return None

    @partitions.setter
    def partitions(self, value):
        if value:
            self._partitions = value if isinstance(value, PartitionSelection) else PartitionSelection(value)
        else:
            self._partitions = None

    @property
    def df(self):
        df_obj = self._df_ref() if self._df_ref is not None else None
        return to_df_collection(df_obj) if df_obj is not None else None

    @df.setter
    def df(self, value):
        if value is None or self._df_ref is None or id(value) != id(self._df_ref()):
            if self._df_ref is not None and self._df_ref() in _df_endpoint_dict:
                del _df_endpoint_dict[self._df_ref()]
            if value is None:
                self._df_ref = None
            else:
                self._add_df_link(value, *(adapter.df for adapter in self._uplink if adapter.df is not None))
                _df_endpoint_dict[value] = self
                self._df_ref = weakref.ref(value)
                if hasattr(self, 'update_df'):
                    self.update_df(value)

    @property
    def fields(self):
        if self.df is not None:
            fields = set(c.name for c in self.df.schema.columns)
            return [f for f in self._fields if f.name in fields]
        return self._fields

    def _link_incoming_dfs(self):
        if self.df is None:
            return
        for p in six.itervalues(self._bind_node.inputs):
            obj = p.obj
            if obj is not None and isinstance(obj, DFAdapter):
                self._add_df_link(self.df, obj.df)

    def _duplicate_df_adapter(self, port, df=None):
        if df is None:
            df = self.df.copy()
        elif self.df is not None:
            self._add_df_link(df, self.df)
        ep = DFAdapter(self._odps, port, df=df)
        ep._link_incoming_dfs()
        for p in six.itervalues(ep._bind_node.inputs):
            obj = p.obj
            if obj is not None and isinstance(obj, DFAdapter):
                self._add_df_link(df, obj.df)
        for attr, value in six.iteritems(vars(self)):
            if not hasattr(ep, attr):
                setattr(ep, attr, value)
        ep._uplink.append(self)
        return ep

    def _iter_linked_objs(self):
        yield self
        if self.df is not None:
            yield self.df

    def perform_operation(self, op):
        if self._uplink:
            op.execute(self._uplink, self)
        self._operations.append(op)


def df_run_hook(*args, **kwargs):
    self = args[0]
    func = kwargs.pop('_df_call')
    if threading.current_thread().name.startswith('PyODPS'):
        return func(*args, **kwargs)

    def _fetch_upspring(df):
        df_iter = df.to_dag(False).closest_ancestors(df, lambda d: d in _df_endpoint_dict)
        if df in _df_endpoint_dict:
            df_iter = itertools.chain(df_iter, (df, ))
        return df_iter

    dfs = itertools.chain(*(_fetch_upspring(f) for f in itertools.chain(extract_df_inputs(args), extract_df_inputs(kwargs)) if f is not None))
    input_eps = [_df_endpoint_dict.get(f) for f in dfs]
    if not input_eps:
        return func(*args, **kwargs)

    node = DFExecNode(self, len(input_eps), func, args, kwargs)
    for idx, input_ep in enumerate(input_eps):
        input_ep._link_node(node, 'input%d' % (1 + idx))

    from ..context import RunnerContext
    RunnerContext.instance()._run(node)
    return node.sink


def install_hook():
    from ...df.expr.expressions import register_exec_hook as register_df_exec_hook
    register_df_exec_hook(df_run_hook)


install_hook()
