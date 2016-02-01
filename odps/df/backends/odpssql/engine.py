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

import time
import sys

from ....errors import ODPSError
from ....utils import init_progress_bar
from ....models import Schema, Partition
from ...core import DataFrame
from ...expr.reduction import *
from ...expr.arithmetic import And, Equal
from ..core import Engine
from ..frame import ResultFrame
from . import types
from . import analyzer as ana
from .context import ODPSContext, UDF_CLASS_NAME
from .compiler import OdpsSQLCompiler
from .codegen import gen_udf


class ODPSEngine(Engine):
    def __init__(self, odps):
        self._odps = odps
        self._ctx = ODPSContext(self._odps)

    def _get_task_percent(self, instance, task_name):
        progress = instance.get_task_progress(task_name)

        if len(progress.stages) > 0:
            all_percent = sum((float(stage.terminated_workers) / stage.total_workers)
                              for stage in progress.stages if stage.total_workers > 0)
            return all_percent / len(progress.stages)
        else:
            return 0

    @classmethod
    def _to_stdout(cls, msg):
        print(msg)

    def _log(self, msg):
        if options.verbose:
            (options.verbose_log or self._to_stdout)(msg)

    @classmethod
    def _is_source_table(cls, expr):
        return isinstance(expr, CollectionExpr) and \
               expr._source_data is not None

    def _run(self, sql, bar, max_progress=1, async=False):
        instance = self._odps.run_sql(sql)

        self._log('logview:')
        self._log(self._odps.get_logview_address(instance.id, 24))
        try:
            percent = 0
            while not instance.is_terminated():
                task_names = instance.get_task_names()
                last_percent = percent
                if len(task_names) > 0:
                    percent = sum(self._get_task_percent(instance, name)
                                  for name in task_names) / len(task_names)
                else:
                    percent = 0
                percent = min(1, max(percent, last_percent))
                bar.update(percent * max_progress)

                time.sleep(1)

            instance.wait_for_success()
            bar.update(max_progress)
        except KeyboardInterrupt:
            instance.stop()
            sys.exit(1)

        return instance

    @classmethod
    def _is_source_column(cls, expr, table):
        if not isinstance(expr, Column):
            return False

        odps_schema = table.schema
        if odps_schema.is_partition(expr.source_name):
            return False

        return True

    @classmethod
    def _is_source_partition(cls, expr, table):
        if not isinstance(expr, Column):
            return False

        odps_schema = table.schema
        if not odps_schema.is_partition(expr.source_name):
            return False

        return True

    @classmethod
    def _can_propagate(cls, collection, no_filter=False, no_projection=False):
        if not isinstance(collection, CollectionExpr):
            return False

        if cls._is_source_table(collection):
            return True
        if not no_filter and cls._filter_on_partition(collection):
            return True
        if not no_projection and cls._projection_on_source(collection):
            return True
        return False

    @classmethod
    def _projection_on_source(cls, expr):
        cols = []

        if isinstance(expr, ProjectCollectionExpr) and \
                cls._can_propagate(expr.input, no_projection=True):
            for col in expr.fields:
                source = next(expr.data_source())
                if not cls._is_source_column(col, source):
                    return False
                cols.append(col.source_name)
            return cols
        return False

    @classmethod
    def _filter_on_partition(cls, expr):
        if not isinstance(expr, FilterCollectionExpr) or \
                not cls._can_propagate(expr.input, no_filter=True):
            return False

        cols = []
        values = []

        def extract(expr):
            if isinstance(expr, Column):
                if cls._is_source_partition(expr, next(expr.data_source())):
                    cols.append(expr.source_name)
                    return True
                else:
                    return False

            if isinstance(expr, And):
                for child in expr.args:
                    if not extract(child):
                        return False
            elif isinstance(expr, Equal) and isinstance(expr._rhs, Scalar):
                if extract(expr._lhs):
                    values.append(expr._rhs.value)
                    return True
            else:
                return False

        if not extract(expr.predicate):
            return False

        if len(cols) == len(values):
            return list(zip(cols, values))
        return False

    def _handle_cases(self, expr, bar=None, tail=None):
        if bar is None:
            bar = init_progress_bar()

        if isinstance(expr, (ProjectCollectionExpr, Summary)) and \
                len(expr.fields) == 1 and \
                isinstance(expr.fields[0], Count) and \
                self._is_source_table(expr.input):
            expr = expr.fields[0]

        columns, partition, count = (None, ) * 3
        if isinstance(expr, Count):
            if isinstance(expr.input, Column):
                input = expr.input.input
            else:
                input = expr.input
        elif isinstance(expr, SliceCollectionExpr):
            input = expr.input
        else:
            input = expr

        if not self._can_propagate(input):
            return

        while True:
            ret = self._filter_on_partition(input)
            if ret:
                partition = ','.join(['='.join(it) for it in ret])
                input = input.input
                continue

            ret = self._projection_on_source(input)
            if ret:
                columns = ret
                input = input.input
                continue
            break

        table = next(expr.data_source())
        partition_size = len(partition.split(',')) if partition is not None else 0
        if table.schema._partitions is not None and \
                len(table.schema._partitions) != partition_size:
            return

        if isinstance(expr, Count):
            try:
                with table.open_reader(reopen=True, partition=partition) as reader:
                    bar.update(1)
                    return reader.count
            except ODPSError:
                return
        else:
            self._log('Try to fetch data from tunnel')
            if isinstance(expr, SliceCollectionExpr):
                count = expr.stop
            try:
                with table.open_reader(reopen=True, partition=partition) as reader:
                    if tail is not None:
                        start = max(reader.count - tail, 0)
                    else:
                        start = None

                    data = []
                    curr = itertools.count(0)
                    size = count or reader.count
                    for r in reader.read(start=start, count=count, columns=columns):
                        i = next(curr)
                        if i % 50 == 0:
                            bar.update(min(float(i) / size, 1))
                        data.append(r.values)
                    bar.update(1)

                    schema = types.df_schema_to_odps_schema(expr._schema, ignorecase=True)
                    return ResultFrame(data, schema=schema)
            except ODPSError:
                return

    def execute(self, expr, async=False):
        bar = init_progress_bar()

        if isinstance(expr, Scalar) and expr.value is not None:
            bar.update(1)
            return expr.value

        src_expr = expr
        expr = self._pre_process(expr)

        try:
            result = self._handle_cases(expr, bar)
        except KeyboardInterrupt:
            sys.exit(1)
        if result is not None:
            try:
                return result
            finally:
                bar.close()

        sql = self._compile(expr)
        self._log('Sql compiled:')
        self._log(sql)

        self._ctx.create_udfs()
        instance = self._run(sql, bar, max_progress=0.9, async=async)
        self._ctx.close()  # clear udfs and resources generated

        if isinstance(expr, (CollectionExpr, Summary)):
            df_schema = expr._schema
            schema = types.df_schema_to_odps_schema(expr._schema, ignorecase=True)
        elif isinstance(expr, SequenceExpr):
            df_schema = Schema.from_lists([expr.name], [expr._data_type])
            schema = types.df_schema_to_odps_schema(df_schema, ignorecase=True)
        else:
            df_schema = None
            schema = None
        try:
            with instance.open_reader(schema=schema) as reader:
                if not isinstance(src_expr, Scalar):
                    try:
                        return ResultFrame([r.values for r in reader], schema=df_schema)
                    finally:
                        bar.update(1)
                else:
                    bar.update(1)
                    odps_type = types.df_type_to_odps_type(src_expr._value_type)
                    return types.odps_types.validate_value(reader[0][0], odps_type)
        finally:
            bar.close()

    def _convert_table(self, expr):
        for node in expr.traverse(top_down=True, unique=True):
            if hasattr(node, 'raw_input') and \
                    isinstance(node.raw_input, (SequenceGroupBy, GroupBy)):
                grouped = node.raw_input.input
                return grouped.agg(expr)[[expr.name, ]]
            elif isinstance(node, Column):
                return node.input[[expr, ]]
            elif isinstance(node, Count) and isinstance(node.input, CollectionExpr):
                return node.input[[expr, ]]

        raise NotImplementedError

    def _optimize(self, expr):
        # We just traverse first to get all the parents
        # TODO think a better way
        memo = dict()
        list(expr.traverse(parent_cache=memo, unique=True))

        # TODO optimizer should be lifted to planner, before the compile action
        from ..optimize import Optimizer
        expr = Optimizer(expr, memo).optimize()

        expr = ana.Analyzer(expr, memo).analyze()
        return expr

    def _pre_process(self, expr):
        src_expr = expr

        if isinstance(expr, Scalar) and expr.value is not None:
            return expr.value
        elif isinstance(expr, (Scalar, SequenceExpr)):
            expr = self._convert_table(expr)

        replaced = self._ctx.get_replaced_expr(src_expr)
        if replaced is None:
            expr = self._optimize(expr)
            self._ctx.add_replaced_expr(src_expr, expr)
        else:
            expr = replaced

        return expr

    def compile(self, expr, prettify=True):
        expr = self._pre_process(expr)

        return self._compile(expr, prettify=prettify)

    def _compile(self, expr, prettify=False):
        backend = OdpsSQLCompiler(self._ctx, beautify=prettify)

        self._ctx.register_udfs(gen_udf(expr, UDF_CLASS_NAME))

        return backend.compile(expr)

    def persist(self, expr, name, partitions=None):
        bar = init_progress_bar()

        if partitions is None:
            sql = self.compile(expr, prettify=False)
            sql = 'CREATE TABLE {0} AS \n{1}'.format(name, sql)
        else:
            if isinstance(partitions, tuple):
                partitions = list(partitions)
            if not isinstance(partitions, list):
                partitions = [partitions, ]

            if isinstance(expr, CollectionExpr):
                schema = types.df_schema_to_odps_schema(expr.schema, ignorecase=True)
            else:
                col_name = expr.name
                tp = types.df_type_to_odps_type(expr.dtype)
                schema = Schema.from_lists([col_name, ], [tp, ])

            for p in partitions:
                if p not in schema:
                    raise ValueError(
                            'Partition field(%s) does not exist in DataFrame schema' % p)

            columns = [c for c in schema.columns if c.name not in partitions]
            ps = [Partition(name=t, type=schema.get_type(t)) for t in partitions]
            self._odps.create_table(name, Schema(columns=columns, partitions=ps))

            expr = expr[[c.name for c in expr.schema if c.name not in partitions] + partitions]

            sql = self.compile(expr, prettify=False)

            sql = 'INSERT OVERWRITE TABLE {0} PARTITION({1}) \n{2}'.format(
                name, ', '.join(partitions), sql
            )

        try:
            self._run(sql, bar)
            return DataFrame(self._odps.get_table(name))
        finally:
            bar.close()

