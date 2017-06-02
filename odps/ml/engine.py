#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import hashlib
import json
import time
import uuid

from .runners import create_node_runner
from .expr import AlgoCollectionExpr, ODPSModelExpr, ModelDataCollectionExpr, MetricsResultExpr
from .utils import is_temp_table
from ..df.backends.context import context
from ..df.backends.analyzer import BaseAnalyzer
from ..df.backends.engine import Engine
from ..df.backends.odpssql import types
from ..df.backends.odpssql.types import df_schema_to_odps_schema
from ..df.backends.errors import CompileError
from ..df.backends.utils import refresh_dynamic
from ..df import DataFrame
from ..df.expr.collections import Node, CollectionExpr, Scalar
from ..df.expr.core import ExprDAG
from ..df.expr.dynamic import DynamicMixin
from ..df.utils import is_source_collection, is_constant_scalar
from .. import options, tempobj, utils
from ..compat import six, futures
from ..errors import ODPSError
from ..models import Partition, Schema
from ..ui import fetch_instance_group, reload_instance_status


class OdpsAlgoContext(object):
    def __init__(self, odps):
        self._odps = odps
        self._node_caches = dict()

    def register_exec(self, idx, parameters):
        pass


class OdpsAlgoAnalyzer(BaseAnalyzer):
    def visit_algo(self, expr):
        pass


class OdpsAlgoEngine(Engine):
    def __init__(self, odps):
        self._odps = odps
        self._ctx = OdpsAlgoContext(odps)
        self._instances = []

    def _dispatch(self, expr_dag, expr, ctx):
        if expr._need_cache and not ctx.is_cached(expr):
            # when the expr should be disk-persisted, skip
            if expr is expr_dag.root and not expr._mem_cache:
                return None
        return super(OdpsAlgoEngine, self)._dispatch(expr_dag, expr, ctx)

    def stop(self):
        for inst in self._instances:
            try:
                self._odps.stop_instance(inst.id)
            except ODPSError:
                pass

    def _gen_table_name(self, expr):
        if options.ml.dry_run:
            if isinstance(expr, Node):
                node_name = expr.node_name
            else:
                node_name = str(expr)
            return '%s_%s' % (utils.TEMP_TABLE_PREFIX, utils.camel_to_underline(node_name))

        table_name = '%s%s_%s' % (utils.TEMP_TABLE_PREFIX, int(time.time()),
                                  str(uuid.uuid4()).replace('-', '_'))
        tempobj.register_temp_table(self._odps, table_name)
        return table_name

    def _gen_model_name(self, expr):
        from .utils import TEMP_MODEL_PREFIX
        if options.ml.dry_run:
            if isinstance(expr, Node):
                node_name = expr.node_name
            else:
                node_name = str(expr)
            return '%s%s' % (utils.TEMP_TABLE_PREFIX, utils.camel_to_underline(node_name))

        model_id_str = utils.to_binary(str(int(time.time())) + '_' + str(uuid.uuid4()).replace('-', '_'))
        digest = hashlib.md5(model_id_str).hexdigest()

        model_name = TEMP_MODEL_PREFIX + digest[-(32 - len(TEMP_MODEL_PREFIX)):]
        tempobj.register_temp_model(self._odps, model_name)
        return model_name

    def _reload_ui(self, group, instance, ui):
        if group:
            reload_instance_status(self._odps, group, instance.id)
            ui.update_group()
            return fetch_instance_group(group).instances.get(instance.id)

    def _run(self, algo_name, params, metas, engine_kw, ui, **kw):
        runner = create_node_runner(self, algo_name, params, metas, engine_kw, ui, **kw)
        runner.execute()

    def _new_analyzer(self, expr_dag, on_sub=None):
        return OdpsAlgoAnalyzer(expr_dag, on_sub=on_sub)

    def _build_model(self, expr, model_name):
        if expr._is_offline_model:
            model = self._odps.get_offline_model(model_name)
            return ODPSModelExpr(_source_data=model, _is_offline_model=True)

        model_params = expr._model_params.copy()
        for meta in ['predictor', 'recommender']:
            meta_val = getattr(expr, '_' + meta, None)
            if meta_val:
                model_params[meta] = meta_val

        model = self._odps.get_tables_model(model_name, tables=list(six.iterkeys(expr._model_collections)))
        model._params = model_params

        sub = ODPSModelExpr(_source_data=model, _is_offline_model=False,
                            _model_params=expr._model_params.copy(), _predictor=expr._predictor)
        data_exprs = dict()
        for k, v in six.iteritems(expr._model_collections):
            data_exprs[k] = ModelDataCollectionExpr(_mlattr_model=sub, _data_item=k)
            data_exprs[k]._source_data = self._odps.get_table(data_exprs[k].table_name())
        sub._model_collections = data_exprs
        return sub

    def _cache(self, expr_dag, dag, expr, **kwargs):

        is_source_model = isinstance(expr, ODPSModelExpr) and expr_dag.root._source_data is not None

        # prevent the `partition` and `partitions` kwargs come from `persist`
        kwargs.pop('partition', None)
        kwargs.pop('partitions', None)

        if is_source_collection(expr_dag.root) or \
                is_constant_scalar(expr_dag.root) or \
                is_source_model:
            return

        execute_dag = ExprDAG(expr_dag.root, dag=expr_dag)

        if isinstance(expr, CollectionExpr):
            table_name = self._gen_table_name(expr)
            table = self._odps.get_table(table_name)
            root = expr_dag.root
            sub = CollectionExpr(_source_data=table, _schema=expr.schema)
            sub.add_deps(root)
            expr_dag.substitute(root, sub)

            kw = dict(kwargs)
            kw['lifecycle'] = options.temp_lifecycle

            execute_node = self._persist(table_name, execute_dag, dag, expr, **kw)

            def callback(result):
                if getattr(expr, 'is_extra_expr', False):
                    sub._source_data = result._source_data
                if isinstance(expr, DynamicMixin):
                    sub._schema = types.odps_schema_to_df_schema(table.schema)
                    refresh_dynamic(sub, expr_dag)

            execute_node.callback = callback
        elif isinstance(expr, ODPSModelExpr):
            model_name = self._gen_model_name(expr)
            sub = self._build_model(expr, model_name)
            root = expr_dag.root
            sub.add_deps(root)
            expr_dag.substitute(root, sub)

            kw = dict(kwargs)
            if 'lifecycle' in kw:
                del kw['lifecycle']

            execute_node = self._persist(model_name, execute_dag, dag, expr, **kw)
        else:
            assert isinstance(expr, Scalar)  # sequence is not cache-able

            class ValueHolder(object): pass
            sub = Scalar(_value_type=expr.dtype)
            sub._value = ValueHolder()

            execute_node = self._execute(execute_dag, dag, expr, **kwargs)

            def callback(res):
                sub._value = res
            execute_node.callback = callback

        return sub, execute_node

    def _write_persist_kw(self, name, expr, **kwargs):
        if isinstance(expr, CollectionExpr):
            persist_kw = kwargs.copy()
            persist_kw['_table'] = name
            project = persist_kw.pop('project', None)
            if self._odps.project != project:
                persist_kw['_project'] = project
            expr.persist_kw = persist_kw
        elif isinstance(expr, ODPSModelExpr):
            persist_kw = kwargs.copy()
            persist_kw['_model'] = name
            project = persist_kw.pop('project', None)
            if project is not None and self._odps.project != project:
                persist_kw['_project'] = project
            expr.persist_kw = persist_kw

    def _persist(self, name, expr_dag, dag, expr, **kwargs):
        self._write_persist_kw(name, expr, **kwargs)
        return super(OdpsAlgoEngine, self)._persist(name, expr_dag, dag, expr, **kwargs)

    @staticmethod
    def _is_output_model_only(src_expr):
        if isinstance(src_expr, MetricsResultExpr):
            return False
        output_exprs = src_expr.outputs()
        return not any(1 for out_expr in six.itervalues(output_exprs) if isinstance(out_expr, CollectionExpr))

    def _build_output_tables(self, expr):
        from .expr.exporters import get_output_table_name

        if not utils.str_to_bool(expr.algo_meta.get('buildTables', False)):
            return

        def create_output_table(table_name, table_schema):
            lifecycle = options.temp_lifecycle if is_temp_table(table_name) else options.lifecycle
            self._odps.create_table(table_name, table_schema, lifecycle=lifecycle)

        table_names, table_schemas = [], []
        for out_name, out_expr in six.iteritems(expr.outputs()):
            if getattr(out_expr, '_algo', None) is None:
                continue
            tn = get_output_table_name(expr, out_name)
            if tn:
                ts = getattr(out_expr, '_algo_schema', None) or out_expr._schema
                table_names.append(tn)
                table_schemas.append(df_schema_to_odps_schema(ts))

        executor = futures.ThreadPoolExecutor(10)
        list(executor.map(create_output_table, table_names, table_schemas))

    def _do_execute(self, expr_dag, src_expr, **kwargs):
        expr = expr_dag.root

        kwargs['_output_models_only'] = self._is_output_model_only(src_expr)

        kw = kwargs.copy()
        if isinstance(src_expr, ODPSModelExpr):
            ui = kw.pop('ui')
            progress_proportion = kw.pop('progress_proportion', 1)
            download_progress = progress_proportion
            ui_group = kw.pop('group', None)

            if hasattr(src_expr, '_source_data'):
                result_expr = src_expr
            else:
                if not context.is_cached(src_expr):
                    temp_name = self._gen_model_name(src_expr)
                    download_progress = 0.1 * progress_proportion
                    self._do_persist(expr_dag, src_expr, temp_name, ui=ui,
                                     progress_proportion=0.9 * progress_proportion, group=ui_group, **kw)

                result_expr = src_expr.get_cached(context.get_cached(src_expr))

            if result_expr._is_offline_model:
                from .expr.models.pmml import PmmlResult
                from .runners import XFlowNodeRunner

                model = result_expr._source_data
                pmml = model.get_model()
                if not options.ml.use_model_transfer:
                    return PmmlResult(pmml)
                else:
                    volume_name = options.ml.model_volume
                    if not self._odps.exist_volume(volume_name):
                        self._odps.create_parted_volume(volume_name)

                    vol_part = hashlib.md5(utils.to_binary(model.name)).hexdigest()
                    tempobj.register_temp_volume_partition(self._odps, (volume_name, vol_part))
                    algo_params = {
                        'modelName': model.name,
                        'volumeName': volume_name,
                        'partition': vol_part,
                        'format': 'pmml'
                    }
                    runner = XFlowNodeRunner(self, 'modeltransfer', algo_params, {}, {},
                                             ui=ui, progress_proportion=download_progress, group=ui_group)
                    runner.execute()
                    pmml = self._odps.open_volume_reader(volume_name, vol_part, model.name + '.xml').read()
                    self._odps.delete_volume_partition(volume_name, vol_part)
                    return PmmlResult(utils.to_str(pmml))
            else:
                from .expr.models.base import TablesModelResult
                results = dict()
                frac = 1.0 / len(result_expr._model_collections)
                for key, item in six.iteritems(result_expr._model_collections):
                    result = item.execute(ui=ui, progress_proportion=frac * 0.1 * progress_proportion,
                                          group=ui_group)
                    results[key] = result
                return TablesModelResult(result_expr._model_params, results)
        elif isinstance(src_expr, MetricsResultExpr):
            if not src_expr.executed:
                expr.tables = dict((pt.name, self._gen_table_name(src_expr)) for pt in src_expr.output_ports)
                gen_params = expr.convert_params(src_expr)

                ui = kw.pop('ui')
                progress_proportion = kw.pop('progress_proportion', 1)
                ui_group = kw.pop('group', None)
                engine_kw = getattr(src_expr, '_engine_kw', {})

                engine_kw['lifecycle'] = options.temp_lifecycle
                if hasattr(src_expr, '_cases'):
                    kw['_cases'] = src_expr._cases

                self._run(src_expr._algo, gen_params, src_expr.algo_meta, engine_kw, ui,
                          progress_proportion=progress_proportion, group=ui_group, **kw)

                src_expr.executed = True

            if options.ml.dry_run:
                return None
            else:
                if hasattr(src_expr, '_result_callback'):
                    callback = src_expr._result_callback
                else:
                    callback = lambda v: v
                return callback(expr.calculator(self._odps))
        else:
            temp_name = self._gen_table_name(src_expr)
            persist_kw = kwargs.copy()
            persist_kw['_table'] = temp_name
            expr.persist_kw = persist_kw

            ui = kw.pop('ui')
            progress_proportion = kw.pop('progress_proportion', 1)
            ui_group = kw.pop('group', None)

            kw['lifecycle'] = options.temp_lifecycle
            df = self._do_persist(expr_dag, src_expr, temp_name, ui=ui,
                                  progress_proportion=0.9 * progress_proportion, group=ui_group, **kw)
            return df.execute(ui=ui, progress_proportion=0.1 * progress_proportion, group=ui_group)

    def _handle_expr_persist(self, out_expr):
        from ..df.backends.engine import ODPSSQLEngine

        class ODPSEngine(ODPSSQLEngine):
            def compile(self, expr, prettify=True, libraries=None):
                expr = self._convert_table(expr)
                expr_dag = expr.to_dag()
                self._analyze(expr_dag, expr)
                new_expr = self._rewrite(expr_dag)
                sql = self._compile(new_expr, prettify=prettify, libraries=libraries)
                if isinstance(sql, list):
                    return '\n'.join(sql)
                return sql

        if isinstance(out_expr, CollectionExpr):
            partition = out_expr.persist_kw.get('partition')
            partitions = out_expr.persist_kw.get('partitions')
            drop_table = out_expr.persist_kw.get('drop_table', False)
            create_table = out_expr.persist_kw.get('create_table', True)
            drop_partition = out_expr.persist_kw.get('drop_partition', False)
            create_partition = out_expr.persist_kw.get('create_partition', False)
            overwrite = out_expr.persist_kw.get('overwrite', True)
            cast = out_expr.persist_kw.get('cast', False)

            expr_table = out_expr.persist_kw['_table']
            expr_project = out_expr.persist_kw.get('_project')
            expr_table_path = expr_table if expr_project is None else expr_project + '.' + expr_table

            if partitions is None and partition is None:
                if drop_table:
                    self._odps.delete_table(expr_table, project=expr_project, if_exists=True)

                if self._odps.exist_table(expr_table):
                    temp_table_name = self._gen_table_name(out_expr)
                    out_expr.persist_kw['_table'] = temp_table_name
                    out_expr.persist_kw['_project'] = None

                    def callback():
                        t = self._odps.get_table(expr_table)
                        if t.schema.partitions:
                            raise CompileError('Cannot insert into partition table %s without specifying '
                                               '`partition` or `partitions`.')
                        expr = self._odps.get_table(temp_table_name).to_df()
                        expr = self._reorder(expr, t, cast=cast)

                        sql = ODPSEngine(self._odps).compile(expr, prettify=False)
                        action_str = 'OVERWRITE' if overwrite else 'INTO'
                        return 'INSERT {0} TABLE {1} \n{2}'.format(action_str, expr_table_path, sql)

                    return callback
                else:
                    return None
            elif partition is not None:
                temp_table_name = self._gen_table_name(out_expr)

                out_expr.persist_kw['_table'] = temp_table_name
                out_expr.persist_kw['_project'] = None

                def callback():
                    t = self._odps.get_table(temp_table_name)

                    for col in out_expr.schema.columns:
                        if col.name.lower() not in t.schema:
                            raise CompileError('Column %s does not exist in table' % col.name)

                    if drop_partition:
                        t.delete_partition(partition, if_exists=True)
                    if create_partition:
                        t.create_partition(partition, if_not_exists=True)

                    expr = t.to_df()
                    expr = self._reorder(expr, t, cast=cast)
                    sql = ODPSEngine(self._odps).compile(expr, prettify=False)
                    action_str = 'OVERWRITE' if overwrite else 'INTO'

                    return 'INSERT {0} TABLE {1} PARTITION({2}) {3}'.format(
                        action_str, expr_table_path, partition, sql,
                    )
                return callback
            else:
                temp_table_name = self._gen_table_name(out_expr)

                out_expr.persist_kw['_table'] = temp_table_name
                out_expr.persist_kw['_project'] = None

                if isinstance(partitions, tuple):
                    partitions = list(partitions)
                if not isinstance(partitions, list):
                    partitions = [partitions, ]

                def callback():
                    t = self._odps.get_table(temp_table_name)
                    schema = t.schema

                    columns = [c for c in schema.columns if c.name not in partitions]
                    ps = [Partition(name=pt, type=schema.get_type(pt)) for pt in partitions]
                    if drop_table:
                        self._odps.delete_table(expr_table, project=expr_project, if_exists=True)
                    if create_table:
                        self._odps.create_table(expr_table, Schema(columns=columns, partitions=ps),
                                                project=expr_project)

                    expr = t.to_df()
                    expr = self._reorder(expr, t, cast=cast, with_partitions=True)
                    sql = ODPSEngine(self._odps).compile(expr, prettify=False)
                    action_str = 'OVERWRITE' if overwrite else 'INTO'

                    return 'INSERT {0} TABLE {1} PARTITION({2}) {3}'.format(
                        action_str, expr_table_path, ', '.join(partitions), sql,
                    )
                return callback
        elif isinstance(out_expr, ODPSModelExpr):
            drop_model = out_expr.persist_kw.get('drop_model', False)
            expr_model = out_expr.persist_kw['_model']

            if drop_model:
                if out_expr._is_offline_model:
                    self._odps.delete_offline_model(expr_model, if_exists=True)
                else:
                    self._odps.delete_tables_model(expr_model, if_exists=True)

    def _do_persist(self, expr_dag, src_expr, name, partitions=None, partition=None, project=None,
                    drop_table=False, create_table=True, drop_partition=False, create_partition=False,
                    **kwargs):
        from .runners import SQLNodeRunner
        from .enums import PortType

        expr = expr_dag.root
        kwargs['_output_models_only'] = self._is_output_model_only(src_expr)

        output_exprs = src_expr.outputs()
        shared_kw = src_expr.shared_kw
        shared_kw['required_outputs'] = dict()
        if hasattr(src_expr, 'output_ports'):
            for out_port in src_expr.output_ports:
                if not out_port.required and out_port.name not in output_exprs:
                    continue

                if out_port.name in output_exprs:
                    out_expr = output_exprs[out_port.name]
                    if not getattr(out_expr, 'persist_kw', None):
                        expr_name = self._gen_table_name(out_expr) if isinstance(out_expr, CollectionExpr) \
                            else self._gen_model_name(expr)
                        self._write_persist_kw(expr_name, out_expr, **kwargs)
                else:
                    expr_name = self._gen_table_name(src_expr.node_name) if out_port.type == PortType.DATA \
                        else self._gen_model_name(src_expr.node_name)
                    shared_kw['required_outputs'][out_port.name] = expr_name
        src_expr.shared_kw = shared_kw

        kw = kwargs.copy()
        ui = kw.pop('ui')
        progress_proportion = kw.pop('progress_proportion', 1)
        ui_group = kw.pop('group', None)
        engine_kw = getattr(src_expr, '_engine_kw', None)

        if kw.get('lifecycle'):
            engine_kw['lifecycle'] = kw['lifecycle']
        elif options.lifecycle:
            engine_kw['lifecycle'] = options.lifecycle

        if hasattr(src_expr, '_cases'):
            kw['_cases'] = src_expr._cases

        if not options.ml.dry_run:
            self._build_output_tables(src_expr)

        sql_callbacks = []

        expr.wait_execution()

        if not src_expr.executed:
            for out_expr in six.itervalues(output_exprs):
                callback = self._handle_expr_persist(out_expr)
                if callback is not None:
                    sql_callbacks.append(callback)

        gen_params = expr.convert_params(src_expr)

        if not src_expr.executed:
            prog_ratio = 1
            sub_ratio = 0
            if sql_callbacks:
                prog_ratio = 0.8
                sub_ratio = (1 - prog_ratio) * progress_proportion / len(sql_callbacks)
            try:
                self._run(src_expr._algo, gen_params, src_expr.algo_meta, engine_kw, ui,
                          progress_proportion=prog_ratio * progress_proportion, group=ui_group, **kw)

                for cb in sql_callbacks:
                    sql = cb()
                    runner = SQLNodeRunner(self, 'SQL', dict(sql=sql), dict(), dict(), ui,
                                           progress_proportion=sub_ratio, group=ui_group)
                    runner.execute()
            finally:
                src_expr.executed = True

        if getattr(src_expr, 'is_extra_expr', False):
            t = src_expr._table_callback(self._odps, src_expr)
            context.cache(src_expr, t)

            if options.ml.dry_run:
                df = CollectionExpr(_source_data=t, _schema=src_expr._schema)
            else:
                df = DataFrame(t)
            df._ml_fields = src_expr._ml_fields
            return df

        ret = None
        for out_name, out_expr in six.iteritems(output_exprs):
            r = self._cache_expr_result(out_expr)
            if out_name == src_expr._output_name:
                ret = r

        return ret

    def _cache_expr_result(self, src_expr):
        if isinstance(src_expr, ODPSModelExpr):
            name = src_expr.persist_kw['_model']
            model_expr = self._build_model(src_expr, name)
            context.cache(src_expr, model_expr._source_data)

            if not model_expr._is_offline_model:
                params_str = utils.escape_odps_string(json.dumps(model_expr._source_data.params))
                for k, v in six.iteritems(model_expr._model_collections):
                    if not options.ml.dry_run:
                        self._odps.run_sql("alter table %s set comment '%s'" % (v._source_data.name, params_str))
                    context.cache(src_expr._model_collections[k], v._source_data)

            return model_expr
        else:
            name = src_expr.persist_kw['_table']
            project = src_expr.persist_kw.get('_project')
            t = self._odps.get_table(name, project=project)
            context.cache(src_expr, t)
            if options.ml.dry_run:
                df = CollectionExpr(_source_data=t, _schema=src_expr._schema)
            else:
                df = DataFrame(t)
            df._ml_fields = src_expr._ml_fields
            return df
