#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import itertools

from .compiler import PandasCompiler
from ..core import Engine, ExecuteNode, ExprDAG
from ..frame import ResultFrame
from ... import DataFrame
from ...expr.core import ExprDictionary
from ...expr.expressions import *
from ...backends.odpssql.types import df_schema_to_odps_schema, df_type_to_odps_type
from ..errors import CompileError
from ...utils import is_source_collection, is_constant_scalar
from ....models import Schema, Partition
from ....errors import ODPSError
from ....types import PartitionSpec
from .... import compat
from ..context import context
from . import analyzer as ana


class PandasExecuteNode(ExecuteNode):
    def __repr__(self):
        return 'Local execution by pandas backend'

    def _repr_html_(self):
        return '<p>Local execution by pandas backend</p>'


class PandasEngine(Engine):
    def __init__(self, odps=None):
        self._odps = odps

    def _new_execute_node(self, expr_dag):
        return PandasExecuteNode(expr_dag)

    def _run(self, expr_dag, pd_dag, ui=None, progress_proportion=1, **_):
        ui.status('Try to execute by local pandas...')

        results = ExprDictionary()
        while True:
            topos = pd_dag.topological_sort()

            no_sub = True
            for node in topos:
                expr, func = node
                if expr in results:
                    continue
                res = func(results)
                if isinstance(res, tuple):
                    src = expr
                    expr = res[0]
                    res = res[1]
                    results[src] = res
                    results[expr] = res

                    # break cuz the dag has changed
                    no_sub = False
                    break
                results[expr] = res
            if no_sub:
                break

        ui.inc(progress_proportion)

        try:
            return results[expr_dag.root]
        except KeyError as e:
            if len(results) == 1:
                return compat.lvalues(results)[0]
            raise e

    def _new_analyzer(self, expr_dag, on_sub=None):
        return ana.Analyzer(expr_dag)

    def _compile(self, expr_dag):
        backend = PandasCompiler(expr_dag)
        return backend.compile(expr_dag.root)

    def _cache(self, expr_dag, dag, expr, **kwargs):
        import pandas as pd

        if is_source_collection(expr_dag.root) or \
                is_constant_scalar(expr_dag.root):
            return

        execute_dag = ExprDAG(expr_dag.root, dag=expr_dag)

        if isinstance(expr, CollectionExpr):
            root = expr_dag.root
            sub = CollectionExpr(_source_data=pd.DataFrame(), _schema=expr.schema)
            sub.add_deps(root)
            expr_dag.substitute(root, sub)

            execute_node = self._execute(execute_dag, dag, expr, ret_df=True, **kwargs)

            def callback(res):
                sub._source_data = res

            execute_node.callback = callback
        else:
            assert isinstance(expr, Scalar)  # sequence is not cache-able

            class ValueHolder(object): pass
            sub = Scalar(_value_type=expr.dtype)
            sub._value = ValueHolder()

            execute_node = self._execute(execute_dag, dag, expr, **kwargs)

            def callback(res):
                sub._value = res

            execute_node.callback = callback

        return sub

    def _do_execute(self, expr_dag, expr, ui=None, progress_proportion=1,
                    head=None, tail=None, **kw):
        expr_dag = self._convert_table(expr_dag)
        self._rewrite(expr_dag)

        ret_df = kw.pop('ret_df', False)

        src_expr = expr

        pd_dag = self._compile(expr_dag)
        df = self._run(expr_dag, pd_dag, ui=ui, progress_proportion=progress_proportion,
                       **kw)

        if not isinstance(src_expr, Scalar):
            context.cache(src_expr, df)
            if head:
                df = df[:head]
            elif tail:
                df = df[-tail:]
            if ret_df:
                return df
            return ResultFrame(df.values, schema=expr_dag.root.schema)
        else:
            res = df.values[0][0]
            context.cache(src_expr, res)
            return res

    @classmethod
    def _convert_pd_type(cls, values, table):
        import pandas as pd

        retvals = []
        for val, t in compat.izip(values, table.schema.types):
            if pd.isnull(val):
                retvals.append(None)
            else:
                retvals.append(val)

        return retvals

    def _write_table_no_partitions(self, frame, table, ui, partition=None,
                                   progress_proportion=1):
        def gen():
            df = frame.values
            size = len(df)

            last_percent = 0
            for i, row in zip(itertools.count(), df.values):
                if i % 50 == 0:
                    percent = float(i) / size * progress_proportion
                    ui.inc(percent - last_percent)
                    last_percent = percent

                yield table.new_record(self._convert_pd_type(row, table))

            if last_percent < progress_proportion:
                ui.inc(progress_proportion - last_percent)

        with table.open_writer(partition=partition) as writer:
            writer.write(gen())

    def _write_table_with_partitions(self, frame, table, partitions, ui,
                                     progress_proportion=1):
        df = frame.values
        vals_to_partitions = dict()
        for ps in df[partitions].drop_duplicates().values:
            p = ','.join('='.join([str(n), str(v)]) for n, v in zip(partitions, ps))
            table.create_partition(p)
            vals_to_partitions[tuple(ps)] = p

        size = len(df)
        curr = [0]
        last_percent = [0]
        for name, group in df.groupby(partitions):
            name = name if isinstance(name, tuple) else (name, )
            group = group[[it for it in group.columns.tolist() if it not in partitions]]

            def gen():
                for i, row in zip(itertools.count(), group.values):
                    curr[0] += i
                    if curr[0] % 50 == 0:
                        percent = float(curr[0]) / size * progress_proportion
                        ui.inc(percent - last_percent[0])
                        last_percent[0] = percent

                    yield table.new_record(self._convert_pd_type(row, table))

            with table.open_writer(partition=vals_to_partitions[name]) as writer:
                writer.write(gen())

        if last_percent[0] < progress_proportion:
            ui.inc(progress_proportion - last_percent[0])

    def _write_table(self, frame, table, ui, partitions=None, partition=None,
                     progress_proportion=1):
        ui.status('Try to upload to ODPS with tunnel...')
        if partitions is None:
            self._write_table_no_partitions(frame, table, ui, partition=partition,
                                            progress_proportion=progress_proportion)
        else:
            self._write_table_with_partitions(frame, table, partitions, ui,
                                              progress_proportion=progress_proportion)

    def _do_persist(self, expr_dag, expr, name, ui=None, project=None,
                    partitions=None, partition=None, odps=None, lifecycle=None,
                    progress_proportion=1, execute_percent=0.5,
                    drop_table=False, create_table=True,
                    drop_partition=False, create_partition=False, **kwargs):
        expr_dag = self._convert_table(expr_dag)
        self._rewrite(expr_dag)

        src_expr = expr
        expr = expr_dag.root
        odps = odps or self._odps
        if odps is None:
            raise ODPSError('ODPS entrance should be provided')

        df = self._do_execute(expr_dag, src_expr, ui=ui,
                              progress_proportion=progress_proportion*execute_percent, **kwargs)
        schema = Schema(columns=df.columns)

        if partitions is not None:
            if isinstance(partitions, tuple):
                partitions = list(partitions)
            if not isinstance(partitions, list):
                partitions = [partitions, ]

            for p in partitions:
                if p not in schema:
                    raise ValueError(
                            'Partition field(%s) does not exist in DataFrame schema' % p)

            columns = [c for c in schema.columns if c.name not in partitions]
            ps = [Partition(name=t, type=schema.get_type(t)) for t in partitions]
            schema = Schema(columns=columns, partitions=ps)
        elif partition is not None:
            t = self._odps.get_table(name, project=project)
            for col in expr.schema.columns:
                if col.name.lower() not in t.schema:
                    raise CompileError('Column %s does not exist in table' % col.name)
                t_col = t.schema[col.name.lower()]
                if df_type_to_odps_type(col.type) != t_col.type:
                    raise CompileError('Column %s\'s type does not match, expect %s, got %s' % (
                        col.name, t_col.type, col.type))

            if drop_partition:
                t.delete_partition(partition, if_exists=True)
            if create_partition:
                t.create_partition(partition, if_not_exists=True)

        if partition is None:
            if drop_table:
                odps.delete_table(name, project=project, if_exists=True)
            if create_table:
                schema = df_schema_to_odps_schema(schema)
                table = odps.create_table(name, schema, project=project, lifecycle=lifecycle)
            else:
                table = odps.get_table(name, project=project)
        else:
            table = odps.get_table(name, project=project)
        self._write_table(df, table, ui=ui, partitions=partitions, partition=partition,
                          progress_proportion=progress_proportion*(1-execute_percent))

        if partition:
            partition = PartitionSpec(partition)
            filters = []
            for k in partition.keys:
                filters.append(lambda x: x[k] == partition[k])
            return DataFrame(odps.get_table(name, project=project)).filter(*filters)
        return DataFrame(odps.get_table(name, project=project))
