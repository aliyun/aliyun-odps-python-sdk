#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

from __future__ import absolute_import

import time
import sys
import uuid  # don't remove
import threading
import warnings
import types as tps

from ....errors import ODPSError, NoPermission, ConnectTimeout
from ....utils import write_log as log
from ....models.table import TableSchema
from ....tempobj import register_temp_table
from ....ui import reload_instance_status, fetch_instance_group
from ...core import DataFrame
from ...expr.reduction import *
from ...expr.core import ExprDAG
from ...expr.dynamic import DynamicMixin
from ...utils import is_source_collection, is_constant_scalar
from ...types import DynamicSchema, Unknown
from ..utils import refresh_dynamic, process_persist_kwargs
from ..core import Engine, ExecuteNode
from ..frame import ResultFrame
from ..context import context
from . import types
from . import analyzer as ana
from . import rewriter as rwr
from ..errors import CompileError
from .context import ODPSContext, UDF_CLASS_NAME
from .compiler import OdpsSQLCompiler
from .codegen import gen_udf
from .tunnel import TunnelEngine
from .models import MemCacheReference


class SQLExecuteNode(ExecuteNode):
    def _sql(self):
        raise NotImplementedError

    def __repr__(self):
        buf = six.StringIO()

        sql = self._sql()

        if sql:
            if isinstance(sql, list):
                sql = '\n'.join(sql)
            buf.write('SQL compiled: \n\n')
            buf.write(sql)
        else:
            buf.write('Use tunnel to download data')

        return buf.getvalue()

    def _repr_html_(self):
        buf = six.StringIO()

        sql = self._sql()

        if sql:
            if isinstance(sql, list):
                sql = '\n'.join(sql)
            buf.write('<h4>SQL compiled</h4>')
            buf.write('<code>%s</code>' % sql)
        else:
            buf.write('<p>Use tunnel to download data</p>')

        return buf.getvalue()


class ODPSSQLEngine(Engine):
    def __init__(self, odps):
        self._odps = odps
        self._ctx_local = threading.local()
        self._instances = []

    @property
    def _ctx(self):
        if not hasattr(self._ctx_local, '_ctx'):
            self._ctx_local._ctx = ODPSContext(self._odps)
        return self._ctx_local._ctx

    def stop(self):
        for inst_id in self._instances:
            try:
                self._odps.stop_instance(inst_id)
            except ODPSError:
                pass
        self._ctx.close()

    def _new_execute_node(self, expr_dag):
        node = SQLExecuteNode(expr_dag)

        def _sql(*_):
            return self._compile_sql(node.expr_dag)

        node._sql = tps.MethodType(_sql, node)
        return node

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

    def _run(self, sql, ui, progress_proportion=1, hints=None, priority=None,
             running_cluster=None, group=None, libraries=None):
        libraries = self._ctx.prepare_resources(self._get_libraries(libraries))
        self._ctx.create_udfs(libraries=libraries)

        if self._ctx.get_udf_count() > 0 and sys.version_info[0] > 2:
            hints = hints or dict()
            hints['odps.sql.jobconf.odps2'] = True
            hints['odps.sql.python.version'] = 'cp37'
        instance = self._odps.run_sql(sql, hints=hints, priority=priority, name='PyODPSDataFrameTask',
                                      running_cluster=running_cluster)

        self._instances.append(instance.id)
        log('Instance ID: ' + instance.id)
        log('  Log view: ' + instance.get_logview_address())
        ui.status('Executing', 'execution details')

        percent = 0
        while not instance.is_terminated(retry=True):
            inst_progress = self._reload_ui(group, instance, ui)

            if inst_progress:
                last_percent = percent
                if inst_progress is not None and len(inst_progress.tasks) > 0:
                    percent = sum(self._get_task_percent(task)
                                  for task in six.itervalues(inst_progress.tasks)) \
                              / len(inst_progress.tasks)
                else:
                    percent = 0
                percent = min(1, max(percent, last_percent))
                ui.inc((percent - last_percent) * progress_proportion)

            time.sleep(1)

        instance.wait_for_success()

        self._reload_ui(group, instance, ui)
        if percent < 1:
            ui.inc((1 - percent) * progress_proportion)

        return instance

    def _handle_cases(self, *args, **kwargs):
        tunnel_engine = TunnelEngine(self._odps)
        return tunnel_engine.execute(*args, **kwargs)

    def _gen_table_name(self):
        table_name = '%s%s_%s' % (TEMP_TABLE_PREFIX, int(time.time()),
                                  str(uuid.uuid4()).replace('-', '_'))
        register_temp_table(self._odps, table_name)
        return table_name

    def _new_analyzer(self, expr_dag, on_sub=None):
        return ana.Analyzer(expr_dag, on_sub=on_sub)

    def _new_rewriter(self, expr_dag):
        return rwr.Rewriter(expr_dag)

    def _compile_sql(self, expr_dag, prettify=True):
        self._rewrite(expr_dag)

        return self._compile(expr_dag.root, prettify=prettify)

    def _compile(self, expr, prettify=False, libraries=None):
        backend = OdpsSQLCompiler(self._ctx, beautify=prettify)

        libraries = self._ctx.prepare_resources(self._get_libraries(libraries))
        self._ctx.register_udfs(*gen_udf(expr, UDF_CLASS_NAME, libraries=libraries))

        return backend.compile(expr)

    def _mem_cache(self, expr_dag, expr):
        engine = self
        root = expr

        def _sub():
            ref_name = self._ctx.get_mem_cache_ref_name(root)
            sub = CollectionExpr(_source_data=MemCacheReference(root._id, ref_name),
                                 _schema=expr_dag.root.schema, _id=expr_dag.root._id)
            expr_dag.substitute(expr_dag.root, sub)
            return sub

        if self._ctx.is_expr_mem_cached(expr):
            if is_source_collection(expr) and \
                    isinstance(expr._source_data, MemCacheReference):
                return
            _sub()
            return

        class MemCacheCompiler(OdpsSQLCompiler):
            def visit_source_collection(self, expr):
                if isinstance(expr._source_data, MemCacheReference):
                    alias = self._ctx.register_collection(expr)
                    from_clause = '{0} {1}'.format(expr._source_data.ref_name, alias)
                    self.add_from_clause(expr, from_clause)
                    self._ctx.add_expr_compiled(expr, from_clause)
                    engine._ctx.register_mem_cache_dep(root, expr)
                else:
                    super(MemCacheCompiler, self).visit_source_collection(expr)

        compiler = MemCacheCompiler(self._ctx, indent_size=0)
        sql = compiler.compile(expr_dag.root).replace('\n', '')
        self._ctx.register_mem_cache_sql(root, sql)
        sub = _sub()

        return sub, None

    def _cache(self, expr_dag, dag, expr, **kwargs):
        if isinstance(expr, CollectionExpr) and expr._mem_cache:
            return self._mem_cache(expr_dag, expr)

        # prevent the kwargs come from `persist`
        process_persist_kwargs(kwargs)

        if is_source_collection(expr_dag.root) or \
                is_constant_scalar(expr_dag.root):
            return

        execute_dag = ExprDAG(expr_dag.root, dag=expr_dag)

        if isinstance(expr, CollectionExpr):
            table_name = self._gen_table_name()
            table = self._odps.get_table(table_name)
            root = expr_dag.root
            sub = CollectionExpr(_source_data=table, _schema=expr.schema)
            sub.add_deps(root)
            expr_dag.substitute(root, sub)

            kw = dict(kwargs)
            kw['lifecycle'] = options.temp_lifecycle

            execute_node = self._persist(table_name, execute_dag, dag, expr, **kw)

            def callback(_):
                if isinstance(expr, DynamicMixin):
                    sub._schema = types.odps_schema_to_df_schema(table.schema)
                    refresh_dynamic(sub, expr_dag)

            execute_node.callback = callback
        else:
            assert isinstance(expr, Scalar)  # sequence is not cache-able

            class ValueHolder(object): pass
            sub = Scalar(_value_type=expr.dtype)
            sub._value = ValueHolder()
            root = expr_dag.root
            sub.add_deps(root)
            expr_dag.substitute(root, sub)

            execute_node = self._execute(execute_dag, dag, expr, **kwargs)

            def callback(res):
                sub._value = res
            execute_node.callback = callback

        return sub, execute_node

    @classmethod
    def _join_sql(cls, sql):
        if isinstance(sql, list):
            return '\n'.join(sql)
        return sql

    def _do_execute(self, expr_dag, expr, ui=None, progress_proportion=1,
                    lifecycle=None, head=None, tail=None,
                    hints=None, priority=None, running_cluster=None, **kw):
        lifecycle = lifecycle or options.temp_lifecycle
        group = kw.get('group')
        libraries = kw.pop('libraries', None)
        use_tunnel = kw.get('use_tunnel', True)

        expr_dag = self._convert_table(expr_dag)
        self._rewrite(expr_dag)

        src_expr = expr
        expr = expr_dag.root

        if isinstance(expr, Scalar) and expr.value is not None:
            ui.inc(progress_proportion)
            return expr.value

        no_permission = False
        if options.df.optimizes.tunnel:
            force_tunnel = kw.get('_force_tunnel', False)
            try:
                result = self._handle_cases(expr, ui, progress_proportion=progress_proportion,
                                            head=head, tail=tail)
            except KeyboardInterrupt:
                ui.status('Halt by interruption')
                sys.exit(1)
            except (NoPermission, ConnectTimeout) as ex:
                result = None
                no_permission = True
                if head:
                    expr = expr[:head]
                warnings.warn('Failed to download data by table tunnel, 10000 records will be limited.\n' +
                              'Cause: ' + str(ex))
            if force_tunnel or result is not None:
                return result

        old_odps2_extension_cfg = options.sql.use_odps2_extension
        try:
            if old_odps2_extension_cfg is None:
                project_obj = self._odps.get_project()
                project_prop = project_obj.properties.get("odps.sql.type.system.odps2")
                options.sql.use_odps2_extension = ("true" == (project_prop or "false").lower())
            sql = self._compile(expr, libraries=libraries)
        finally:
            options.sql.use_odps2_extension = old_odps2_extension_cfg

        cache_data = None
        if not no_permission and isinstance(expr, CollectionExpr) and not isinstance(expr, Summary):
            # When tunnel cannot handle, we will try to create a table
            tmp_table_name = '%s%s' % (TEMP_TABLE_PREFIX, str(uuid.uuid4()).replace('-', '_'))
            register_temp_table(self._odps, tmp_table_name)
            cache_data = self._odps.get_table(tmp_table_name)

            lifecycle_str = 'LIFECYCLE {0} '.format(lifecycle) if lifecycle is not None else ''
            format_sql = lambda s: 'CREATE TABLE {0} {1}AS \n{2}'.format(tmp_table_name, lifecycle_str, s)
            if isinstance(sql, list):
                sql[-1] = format_sql(sql[-1])
            else:
                sql = format_sql(sql)

        sql = self._join_sql(sql)

        log('Sql compiled:')
        log(sql)

        instance = self._run(sql, ui, progress_proportion=progress_proportion*0.9,
                             hints=hints, priority=priority, running_cluster=running_cluster,
                             group=group, libraries=libraries)

        self._ctx.close()  # clear udfs and resources generated
        res = self._fetch(expr, src_expr, instance, ui,
                          cache_data=cache_data, head=head, tail=tail,
                          use_tunnel=use_tunnel, group=group,
                          progress_proportion=progress_proportion*.1,
                          finish=kw.get('finish', True))
        if kw.get('ret_instance', False) is True:
            return instance, res
        return res

    def _fetch(self, expr, src_expr, instance, ui, progress_proportion=1,
               cache_data=None, head=None, tail=None, use_tunnel=True,
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
                            context.cache(src_expr, cache_data)
                            # reset schema
                            if isinstance(src_expr, CollectionExpr) and \
                                    (isinstance(src_expr._schema, DynamicSchema) or
                                     any(isinstance(col.type, Unknown) for col in src_expr._schema.columns)):
                                src_expr._schema = df_schema
                            ui.inc(progress_proportion)
                except ODPSError as ex:
                    # some project has closed the tunnel download
                    # we just ignore the error
                    warnings.warn('Failed to download data by table tunnel, 10000 records will be limited.\n' +
                                  'Cause: ' + str(ex))
                    pass

            if tail:
                raise NotImplementedError

            try:
                if finish:
                    ui.status('Start to use head to download results...')
                return ResultFrame(cache_data.head(head or 10000), schema=df_schema)
            finally:
                context.cache(src_expr, cache_data)
                ui.inc(progress_proportion)

        with instance.open_reader(schema=schema, use_tunnel=False) as reader:
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
                    context.cache(src_expr, cache_data)
                    ui.inc(progress_proportion)
            else:
                ui.inc(progress_proportion)
                odps_type = types.df_type_to_odps_type(src_expr._value_type, project=instance.project)
                res = types.odps_types.validate_value(reader[0][0], odps_type)
                context.cache(src_expr, res)
                return res

    def _do_persist(self, expr_dag, expr, name, partitions=None, partition=None, project=None, ui=None,
                    progress_proportion=1, lifecycle=None, hints=None, priority=None,
                    running_cluster=None, overwrite=True, drop_table=False, create_table=True,
                    drop_partition=False, create_partition=None, cast=False, **kw):
        group = kw.get('group')
        libraries = kw.pop('libraries', None)

        expr_dag = self._convert_table(expr_dag)
        self._rewrite(expr_dag)

        src_expr = expr
        expr = expr_dag.root

        should_cache = False

        if drop_table:
            self._odps.delete_table(name, project=project, if_exists=True)

        table_name = name if project is None else '%s.`%s`' % (project, name)
        project_obj = self._odps.get_project(project)
        if partitions is None and partition is None:
            # the non-partitioned table
            if drop_partition:
                raise ValueError('Cannot drop partition for non-partition table')
            if create_partition:
                raise ValueError('Cannot create partition for non-partition table')

            if self._odps.exist_table(name, project=project) or not create_table:
                t = self._odps.get_table(name, project=project)
                if t.schema.partitions:
                    raise CompileError('Cannot insert into partition table %s without specifying '
                                       '`partition` or `partitions`.')
                expr = self._reorder(expr, t, cast=cast)
            else:
                # We don't use `CREATE TABLE ... AS` because it will report `table already exists`
                # when service retries.
                if isinstance(expr, CollectionExpr):
                    schema = types.df_schema_to_odps_schema(expr.schema, ignorecase=True)
                else:
                    col_name = expr.name
                    tp = types.df_type_to_odps_type(expr.dtype, project=project_obj)
                    schema = Schema.from_lists([col_name, ], [tp, ])
                self._odps.create_table(name, Schema(columns=schema.columns),
                                        project=project, lifecycle=lifecycle)

            sql = self._compile(expr, prettify=False, libraries=libraries)
            action_str = 'OVERWRITE' if overwrite else 'INTO'
            format_sql = lambda s: 'INSERT {0} TABLE {1} \n{2}'.format(action_str, table_name, s)
            if isinstance(sql, list):
                sql[-1] = format_sql(sql[-1])
            else:
                sql = format_sql(sql)

            should_cache = True
        elif partition is not None:
            if self._odps.exist_table(name, project=project) or not create_table:
                t = self._odps.get_table(name, project=project)
                partition = self._get_partition(partition, t)

                if drop_partition:
                    t.delete_partition(partition, if_exists=True)
                if create_partition:
                    t.create_partition(partition, if_not_exists=True)
            else:
                partition = self._get_partition(partition)
                column_names = [n for n in expr.schema.names if n not in partition]
                column_types = [
                    types.df_type_to_odps_type(expr.schema[n].type, project=project_obj)
                    for n in column_names
                ]
                partition_names = [n for n in partition.keys]
                partition_types = ['string'] * len(partition_names)
                t = self._odps.create_table(
                    name, TableSchema.from_lists(column_names, column_types,
                                                 partition_names, partition_types),
                    project=project, lifecycle=lifecycle)
                if create_partition is None or create_partition is True:
                    t.create_partition(partition)

            expr = self._reorder(expr, t, cast=cast)
            sql = self._compile(expr, prettify=False, libraries=libraries)

            action_str = 'OVERWRITE' if overwrite else 'INTO'
            format_sql = lambda s: 'INSERT {0} TABLE {1} PARTITION({2}) \n{3}'.format(
                action_str, table_name, partition, s
            )
            if isinstance(sql, list):
                sql[-1] = format_sql(sql[-1])
            else:
                sql = format_sql(sql)
        else:
            if isinstance(partitions, tuple):
                partitions = list(partitions)
            if not isinstance(partitions, list):
                partitions = [partitions, ]

            if isinstance(expr, CollectionExpr):
                schema = types.df_schema_to_odps_schema(expr.schema, ignorecase=True)
            else:
                col_name = expr.name
                tp = types.df_type_to_odps_type(expr.dtype, project=project_obj)
                schema = Schema.from_lists([col_name, ], [tp, ])

            for p in partitions:
                if p not in schema:
                    raise ValueError(
                        'Partition field(%s) does not exist in DataFrame schema' % p)

            columns = [c for c in schema.columns if c.name not in partitions]
            ps = [TableSchema.TablePartition(name=pt, type=schema.get_type(pt)) for pt in partitions]
            if self._odps.exist_table(name, project=project) or not create_table:
                t = self._odps.get_table(name, project=project)
            else:
                t = self._odps.create_table(name, Schema(columns=columns, partitions=ps),
                                            project=project, lifecycle=lifecycle)
            if drop_partition:
                raise ValueError('Cannot drop partitions when specify `partitions`')
            if create_partition:
                raise ValueError('Cannot create partitions when specify `partitions`')
            expr = expr[[c.name for c in expr.schema if c.name not in partitions] + partitions]

            expr = self._reorder(expr, t, cast=cast, with_partitions=True)
            sql = self._compile(expr, prettify=False, libraries=libraries)

            action_str = 'OVERWRITE' if overwrite else 'INTO'
            format_sql = lambda s: 'INSERT {0} TABLE {1} PARTITION({2}) \n{3}'.format(
                action_str, table_name, ', '.join(partitions), s
            )
            if isinstance(sql, list):
                sql[-1] = format_sql(sql[-1])
            else:
                sql = format_sql(sql)

        sql = self._join_sql(sql)

        log('Sql compiled:')
        log(sql)

        instance = self._run(sql, ui, progress_proportion=progress_proportion,
                             hints=hints, priority=priority, running_cluster=running_cluster,
                             group=group, libraries=libraries)
        self._ctx.close()  # clear udfs and resources generated
        t = self._odps.get_table(name, project=project)
        if should_cache and not is_source_collection(src_expr):
            # TODO: support cache partition
            context.cache(src_expr, t)
        if partition:
            filters = []
            df = DataFrame(t)
            for k in partition.keys:
                # actual type of partition and column type may mismatch
                filters.append(df[k] == Scalar(partition[k]).cast(df[k].dtype))
            res = df.filter(*filters)
        else:
            res = DataFrame(t)
        if kw.get('ret_instance', False) is True:
            return instance, res
        return res
