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
import itertools
import uuid  # don't remove
from contextlib import contextmanager
import warnings

from ....errors import ODPSError

from ....utils import init_progress_ui, write_log as log
from ....models import Partition, Resource
from ....tempobj import register_temp_table
from ....types import PartitionSpec
from ....compat import izip
from ....tunnel.tabletunnel.downloadsession import TableDownloadSession
from ....ui import reload_instance_status, fetch_instance_group
from ...core import DataFrame
from ...expr.reduction import *
from ...expr.arithmetic import And, Equal
from ...utils import is_source_collection
from ..core import Engine
from ..frame import ResultFrame
from ..context import context
from . import types
from . import analyzer as ana
from . import rewriter as rwr
from ..errors import CompileError
from .context import ODPSContext, UDF_CLASS_NAME
from .compiler import OdpsSQLCompiler
from .codegen import gen_udf


class ODPSEngine(Engine):
    def __init__(self, odps, global_optimize=True):
        self._odps = odps
        self._ctx = ODPSContext(self._odps)

        self._global_optimize = global_optimize

    @staticmethod
    def _get_task_percent(task_progress):
        if len(task_progress.stages) > 0:
            all_percent = sum((float(stage.terminated_workers) / stage.total_workers)
                              for stage in task_progress.stages if stage.total_workers > 0)
            return all_percent / len(task_progress.stages)
        else:
            return 0

    def _reload_ui(self, group, instance, ui):
        if group:
            reload_instance_status(self._odps, group, instance.id)
            ui.update_group()
            return fetch_instance_group(group).instances.get(instance.id)

    @classmethod
    def _get_libraries(cls, libraries):
        def conv(libs):
            if isinstance(libs, (six.binary_type, six.text_type, Resource)):
                return [libs, ]
            return libs

        libraries = conv(libraries) or []
        if options.df.libraries is not None:
            libraries.extend(conv(options.df.libraries))
        if len(libraries) == 0:
            return
        return list(set(libraries))

    def _run(self, sql, ui, start_progress=0, max_progress=1, async=False, hints=None,
             group=None, libraries=None):
        self._ctx.create_udfs(libraries=self._get_libraries(libraries))
        instance = self._odps.run_sql(sql, hints=hints)

        log('Instance ID: ' + instance.id)
        log('  Log view: ' + instance.get_logview_address())
        ui.status('Executing', 'execution details')

        if async:
            return instance

        try:
            percent = 0
            while not instance.is_terminated():
                inst_progress = self._reload_ui(group, instance, ui)

                if inst_progress:
                    last_percent = percent
                    if inst_progress is not None and len(inst_progress.tasks) > 0:
                        percent = sum(self._get_task_percent(task)
                                      for task in six.itervalues(inst_progress.tasks)) / len(inst_progress.tasks)
                    else:
                        percent = 0
                    percent = min(1, max(percent, last_percent))
                    ui.update(start_progress + percent * max_progress)

                time.sleep(1)

            instance.wait_for_success()
            self._reload_ui(group, instance, ui)
            ui.update(start_progress + max_progress)
        except KeyboardInterrupt:
            instance.stop()
            self._ctx.close()
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

        if is_source_collection(collection):
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
            else:
                return False

            return True

        if not extract(expr.predicate):
            return False

        if len(cols) == len(values):
            return list(zip(cols, values))
        return False

    @contextmanager
    def _open_reader(self, t, **kwargs):
        try:
            with t.open_reader(**kwargs) as reader:
                if reader.status == TableDownloadSession.Status.Normal:
                    yield reader
                    return
        except ODPSError:
            # ignore the error when reusing the tunnel before
            pass

        # reopen
        with t.open_reader(reopen=True, **kwargs) as reader:
            yield reader

    @classmethod
    def _to_partition_spec(cls, kv):
        spec = PartitionSpec()
        for k, v in kv:
            spec[k] = v
        return spec

    @classmethod
    def _partition_prefix(cls, all_partitions, filtered_partitions):
        filtered_partitions = sorted(six.iteritems(filtered_partitions.kv),
                                     key=lambda x: all_partitions.index(x[0]))
        if len(filtered_partitions) > len(all_partitions):
            return
        if not all(zip(l == r for l, r in zip(all_partitions, filtered_partitions))):
            return

        return cls._to_partition_spec(filtered_partitions)

    def _handle_cases(self, expr, ui=None, start_progress=0, max_progress=1,
                      head=None, tail=None):
        if ui is None:
            ui = init_progress_ui()

        if isinstance(expr, (ProjectCollectionExpr, Summary)) and \
                len(expr.fields) == 1 and \
                isinstance(expr.fields[0], Count):
            expr = expr.fields[0]

        columns, partitions, count = (None, ) * 3
        pkv = None
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
                pkv = dict((k, v) for k, v in ret)
                partitions = self._to_partition_spec(ret)
                input = input.input
                continue

            ret = self._projection_on_source(input)
            if ret:
                columns = ret
                input = input.input
                continue
            break

        table = next(expr.data_source())
        partition, filter_all_partitions = None, True
        if table.schema._partitions:
            if partitions is not None:
                partition = self._partition_prefix(
                    [p.name for p in table.schema._partitions], partitions)
                if partition is None:
                    return
                if len(table.schema._partitions) != len(partitions):
                    filter_all_partitions = False
            else:
                filter_all_partitions = False

        if isinstance(expr, Count):
            if not filter_all_partitions:
                # if not filter all partitions, fall back to ODPS SQL to calculate count
                return
            try:
                with self._open_reader(table, partition=partition) as reader:
                    ui.update(start_progress + max_progress)
                    return reader.count
            except ODPSError:
                return
        else:
            log('Try to fetch data from tunnel')
            ui.status('Try to download data with tunnel...')
            if isinstance(expr, SliceCollectionExpr):
                if expr.start:
                    raise ExpressionError('For ODPS backend, slice\'s start cannot be specified')
                count = expr.stop
            try:
                fetch_partitions = [partition] if filter_all_partitions else \
                    (p.name for p in table.iterate_partitions(partition))
                if tail is not None:
                    fetch_partitions = list(fetch_partitions)[::-1]

                data = []

                start, size, step = None, None, None
                if head is not None:
                    size = min(head, count) if count is not None else head
                elif tail is not None:
                    if filter_all_partitions:
                        start = None if count is None else max(count - tail, 0)
                        size = tail if count is None else min(count, tail)
                    else:
                        # tail on multi partitions, just fall back to SQL
                        return
                else:
                    size = count

                if size is None and tail is not None:
                    fetch_partitions = list(fetch_partitions)

                cum = 0
                for curr_part, partition in izip(itertools.count(1), fetch_partitions):
                    rest = size - cum if size is not None else None
                    finished = False

                    with self._open_reader(table, partition=partition) as reader:
                        if tail is not None and start is None:
                            s = max(reader.count - tail, 0)
                            start = s if start is None else max(s, start)

                        for i, r in izip(itertools.count(1),
                                         reader.read(start=start, count=rest, columns=columns)):
                            if size is not None and cum > size - 1:
                                finished = True
                                break
                            cum += 1
                            if cum % 50 == 0:
                                if size is not None:
                                    ui.update(start_progress + float(cum) / size * max_progress)
                                else:
                                    p = float(i) / reader.count * curr_part / len(fetch_partitions)
                                    ui.update(start_progress + p * max_progress)

                            if pkv:
                                self._fill_back_partition_values(r, table, pkv)
                            data.append(r.values)

                    if finished:
                        break

                ui.update(start_progress + max_progress)
                return ResultFrame(data, schema=expr._schema)
            except ODPSError:
                return

    @classmethod
    def _fill_back_partition_values(cls, record, table, pkv):
        if pkv:
            for k, v in six.iteritems(pkv):
                if k in record and record[k] is None:
                    # fill back the partition data which is lost in the tunnel
                    record[k] = types.odps_types.validate_value(v, table.schema.get_type(k))

    def execute(self, expr, ui=None, async=False, start_progress=0,
                max_progress=1, lifecycle=None, head=None, tail=None, hints=None, **kw):
        close_ui = ui is None
        ui = ui or init_progress_ui()
        lifecycle = lifecycle or options.temp_lifecycle
        group = kw.get('group')
        libraries = kw.pop('libraries', None)

        if isinstance(expr, Scalar) and expr.value is not None:
            ui.update(start_progress+max_progress)
            return expr.value

        src_expr = kw.get('src_expr', expr)
        expr = self._pre_process(expr, src_expr, dag=kw.get('dag'))

        try:
            result = self._handle_cases(expr, ui, head=head, tail=tail)
        except KeyboardInterrupt:
            ui.status('Halt by interruption')
            sys.exit(1)
        if result is not None:
            try:
                return result
            finally:
                if close_ui:
                    ui.close()

        sql = self._compile(expr, libraries=libraries)

        cache_data = None
        if isinstance(expr, CollectionExpr):
            tmp_table_name = '%s%s' % (TEMP_TABLE_PREFIX, str(uuid.uuid4()).replace('-', '_'))
            register_temp_table(self._odps, tmp_table_name)
            cache_data = self._odps.get_table(tmp_table_name)

            lifecycle_str = 'LIFECYCLE {0} '.format(lifecycle) if lifecycle is not None else ''
            sql = 'CREATE TABLE {0} {1}AS \n{2}'.format(tmp_table_name, lifecycle_str, sql)

        log('Sql compiled:')
        log(sql)

        instance = self._run(sql, ui, start_progress=start_progress,
                             max_progress=0.9*max_progress, async=async, hints=hints,
                             group=group, libraries=libraries)

        use_tunnel = kw.get('use_tunnel', True)
        if async:
            engine = self

            global_head, global_tail = head, tail

            class AsyncResult(object):
                @property
                def instance(self):
                    return instance

                def wait(self):
                    instance.wait_for_completion()

                def is_done(self):
                    return instance.is_terminated()

                def fetch(self, head=None, tail=None):
                    head = head or global_head
                    tail = tail or global_tail
                    return engine._fetch(expr, src_expr, instance, ui,
                                         finish_progress=start_progress+max_progress,
                                         close_ui=close_ui, cache_data=cache_data,
                                         head=head, tail=tail, use_tunnel=use_tunnel,
                                         group=group)

            return AsyncResult()
        else:
            self._ctx.close()  # clear udfs and resources generated
            return self._fetch(expr, src_expr, instance, ui, close_ui=close_ui,
                               cache_data=cache_data, head=head, tail=tail,
                               use_tunnel=use_tunnel, group=group, finish_progress=max_progress,
                               finish=kw.get('finish', True))

    def _fetch(self, expr, src_expr, instance, ui, finish_progress=1,
               close_ui=True, cache_data=None, head=None, tail=None, use_tunnel=True,
               group=None, finish=True):
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
            if cache_data is not None:
                if group and finish:
                    ui.remove_keys(group)
                if use_tunnel:
                    try:
                        if finish:
                            ui.status('Start to use tunnel to download results...')
                        with cache_data.open_reader(reopen=True) as reader:
                            if head:
                                reader = reader[:head]
                            elif tail:
                                start = max(reader.count - tail, 0)
                                reader = reader[start: ]
                            try:
                                return ResultFrame([r.values for r in reader], schema=df_schema)
                            finally:
                                src_expr._cache_data = cache_data
                                ui.update(finish_progress)
                    except ODPSError:
                        # some project has closed the tunnel download
                        # we just ignore the error
                        warnings.warn('Fail to download data by tunnel, 10000 records will be limited')
                        pass

                if tail:
                    raise NotImplementedError

                try:
                    if finish:
                        ui.status('Start to use head to download results...')
                    return ResultFrame(cache_data.head(head or 10000), schema=df_schema)
                finally:
                    src_expr._cache_data = cache_data
                    ui.update(finish_progress)

            with instance.open_reader(schema=schema) as reader:
                ui.status('Start to read instance results...')
                if not isinstance(src_expr, Scalar):
                    if head:
                        reader = reader[:head]
                    elif tail:
                        start = max(reader.count - tail, 0)
                        reader = reader[start: ]
                    try:
                        return ResultFrame([r.values for r in reader], schema=df_schema)
                    finally:
                        src_expr._cache_data = cache_data
                        ui.update(finish_progress)
                else:
                    ui.update(finish_progress)
                    odps_type = types.df_type_to_odps_type(src_expr._value_type)
                    res = types.odps_types.validate_value(reader[0][0], odps_type)
                    src_expr._cache_data = res
                    return res
        finally:
            if close_ui:
                ui.close()

    def _optimize(self, expr, src_expr, dag=None):
        if self._global_optimize:
            # we copy the entire ast,
            # and all the modification will be applied to the copied ast.
            if dag is None:
                # the expr is already copied one
                expr = context.register_to_copy_expr(src_expr, expr)
            dag = context.build_dag(src_expr, expr, dag=dag)

            # analyze first
            ana.Analyzer(dag).analyze()

            from ..optimize import Optimizer
            Optimizer(dag).optimize()
        else:
            dag = context.build_dag(src_expr, expr, dag=dag)

        expr = rwr.Rewriter(dag).rewrite()
        return expr

    def _pre_process(self, expr, src_expr=None, dag=None):
        src_expr = src_expr or expr

        if isinstance(expr, Scalar) and expr.value is not None:
            return expr.value

        replaced = self._ctx.get_replaced_expr(src_expr)
        if replaced is not None:
            return replaced

        if isinstance(expr, (Scalar, SequenceExpr)):
            expr = self._convert_table(expr)

        expr = self._optimize(expr, src_expr, dag=dag)
        self._ctx.add_replaced_expr(src_expr, expr)
        return expr

    def compile(self, expr, prettify=True, libraries=None):
        expr = self._pre_process(expr)

        return self._compile(expr, prettify=prettify, libraries=libraries)

    def _compile(self, expr, prettify=False, libraries=None):
        backend = OdpsSQLCompiler(self._ctx, beautify=prettify)

        self._ctx.register_udfs(*gen_udf(expr, UDF_CLASS_NAME,
                                         libraries=self._get_libraries(libraries)))

        return backend.compile(expr)

    def persist(self, expr, name, partitions=None, partition=None, project=None, ui=None,
                start_progress=0, max_progress=1, lifecycle=None, hints=None,
                create_table=True, drop_partition=False, create_partition=False, **kw):
        close_ui = ui is None
        ui = ui or init_progress_ui()
        group = kw.get('group')
        libraries = kw.pop('libraries', None)

        src_expr = kw.pop('src_expr', expr)

        should_cache = False

        table_name = name if project is None else '%s.`%s`' % (project, name)
        if partitions is None and partition is None:
            sql = self.compile(expr, prettify=False)
            lifecycle_str = 'LIFECYCLE {0} '.format(lifecycle) if lifecycle is not None else ''
            sql = 'CREATE TABLE {0} {1}AS \n{2}'.format(table_name, lifecycle_str, sql)

            should_cache = True
        elif partition is not None:
            sql = self.compile(expr, prettify=False)
            t = self._odps.get_table(name, project=project)

            for col in expr.schema.columns:
                if col.name.lower() not in t.schema:
                    raise CompileError('Column %s does not exist in table' % col.name)
                t_col = t.schema[col.name.lower()]
                if types.df_type_to_odps_type(col.type) != t_col.type:
                    raise CompileError('Column %s\'s type does not match, expect %s, got %s' % (
                        col.name, t_col.type, col.type))

            if drop_partition:
                t.delete_partition(partition, if_exists=True)
            if create_partition:
                t.create_partition(partition, if_not_exists=True)

            partition = PartitionSpec(partition)
            sql = 'INSERT OVERWRITE TABLE {0} PARTITION({1}) \n{2}'.format(
                name, partition, sql
            )
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
            if create_table:
                self._odps.create_table(name, Schema(columns=columns, partitions=ps), project=project)
            expr = expr[[c.name for c in expr.schema if c.name not in partitions] + partitions]

            sql = self.compile(expr, prettify=False)

            sql = 'INSERT OVERWRITE TABLE {0} PARTITION({1}) \n{2}'.format(
                name, ', '.join(partitions), sql
            )

        log('Sql compiled:')
        log(sql)

        try:
            self._run(sql, ui, start_progress=start_progress, max_progress=max_progress,
                      hints=hints, group=group, libraries=libraries)
            t = self._odps.get_table(name, project=project)
            if should_cache:
                src_expr._cache_data = t
            if partition:
                filters = []
                for k in partition.keys:
                    filters.append(lambda x: x[k] == partition[k])
                return DataFrame(t).filter(*filters)
            return DataFrame(t)
        finally:
            if close_ui:
                ui.close()

