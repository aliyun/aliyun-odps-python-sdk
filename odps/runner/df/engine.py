# encoding: utf-8
# Copyright 1999-2017 Alibaba Group Holding Ltd.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ... import DataFrame, Scalar
from ...compat import six
from ...df.backends.engine import MixedEngine
from ...df.backends.utils import refresh_dynamic
from .adapter import DFAdapter, DFExecNode
from ..enums import EngineType
from ..context import RunnerContext
from ..engine import node_engine, BaseNodeEngine
from ..utils import is_temp_table, split_project_name


@node_engine(EngineType.DF)
class DFNodeEngine(BaseNodeEngine):
    def before_exec(self):
        super(DFNodeEngine, self).before_exec()
        self._input_dfs = []

        for inp in six.itervalues(self._node.inputs):
            ep = inp.obj
            if isinstance(ep, DFAdapter) and isinstance(ep.df, DataFrame):
                ep.reload()
                self._input_dfs.append(ep.df)

    def actual_exec(self):
        def _after_run_sql(instance, sql):
            self._instances.append(instance)
            self._last_cmd = sql

        to_nodes = set(e.to_node for edges in six.itervalues(self._node.output_edges) for e in edges
                       if e.to_node.scheduled)
        out_engines = set(e.to_node.engine for edges in six.itervalues(self._node.output_edges) for e in edges
                          if e.to_node.scheduled) - set([EngineType.DF, ])
        if out_engines or not to_nodes:
            try:
                self._odps.after_run_sql = _after_run_sql
                self._run_df()
            finally:
                self._odps.after_run_sql = None
        else:
            self._node.skipped = True

    def _run_df(self):
        context = RunnerContext.instance()
        engine = MixedEngine(self._odps)
        dag = context._dag
        ds_container = context._obj_container
        ancestors = dag.ancestors(self._node, lambda n: n.engine == EngineType.DF)
        [ds_container[inp.obj_uuid].reload()
         for ancestor in ancestors for inp in six.itervalues(ancestor.inputs)]
        output_ds_list = [ds_container[outp.obj_uuid] for outp in six.itervalues(self._node.outputs)
                          if outp.obj_uuid in ds_container]

        if isinstance(self._node, DFExecNode):
            [refresh_dynamic(df, self._node.bind_df.to_dag(copy=False)) for df in self._input_dfs]
            self._node.bind_df._engine = engine
            ret_val = self._node.func(*self._node.args, **self._node.kwargs)
            self._node.sink = ret_val
        else:
            for adapter in output_ds_list:
                # generate output df
                self._runner._fix_upstream_ports(adapter)
                [refresh_dynamic(df, adapter.df.to_dag(copy=False)) for df in self._input_dfs]

                proj, table = split_project_name(adapter.table)
                lifecycle = self._temp_lifecycle if is_temp_table(table) else self._global_lifecycle
                if not adapter.partitions:
                    try:
                        self._last_cmd = repr(engine.compile(adapter.df))
                    except:
                        self._last_cmd = None

                    engine.persist(adapter.df, table, project=proj, lifecycle=lifecycle)
                else:
                    new_fields = [adapter.df, ] + [Scalar(v).rename(k) for k, v in adapter.partitions[0]]
                    part_names = [k for k, _ in adapter.partitions[0]]
                    target_df = adapter.df.__getitem__(tuple(new_fields))

                    try:
                        self._last_cmd = repr(engine.compile(target_df))
                    except:
                        self._last_cmd = None

                    engine.persist(target_df, adapter.table, partitions=part_names, project=proj, lifecycle=lifecycle)
