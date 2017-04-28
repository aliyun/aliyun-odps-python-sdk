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

from __future__ import absolute_import

from decimal import Decimal
import types as tps

from .compiler import SQLAlchemyCompiler
from . import analyzer as ana
from . import rewriter as rwr
from .types import df_schema_to_sqlalchemy_columns
from ..core import ExecuteNode, Engine
from ..errors import CompileError
from ..utils import write_table, refresh_dynamic
from ..frame import ResultFrame
from ..context import context
from ... import DataFrame
from ...utils import is_source_collection, is_constant_scalar
from ...expr.expressions import *
from ...expr.core import ExprDAG
from ...expr.dynamic import DynamicMixin
from ...types import DynamicSchema, Unknown
from ...backends.odpssql.types import df_schema_to_odps_schema, df_type_to_odps_type
from ....types import PartitionSpec
from ....utils import write_log as log, gen_temp_table
from ....models import Schema, Partition


class SQLExecuteNode(ExecuteNode):
    def _sql(self):
        raise NotImplementedError

    def __repr__(self):
        buf = six.StringIO()

        sql = self._sql()
        buf.write('MPP SQL compiled: \n\n')
        buf.write(sql)

        return buf.getvalue()

    def _repr_html_(self):
        buf = six.StringIO()

        sql = self._sql()
        buf.write('<h4>MPP SQL compiled</h4>')
        buf.write('<code>%s</code>' % sql)

        return buf.getvalue()


_engine_to_connections = {}


class SQLAlchemyEngine(Engine):
    def __init__(self, odps=None):
        self._odps = odps

    def _new_execute_node(self, expr_dag):
        node = SQLExecuteNode(expr_dag)

        def _sql(*_):
            return self._compile_sql(expr_dag)

        def verify(*_):
            try:
                _sql(*_)
                return True
            except NotImplementedError:
                return False

        node._sql = tps.MethodType(_sql, node)
        node.verify = tps.MethodType(verify, node)
        return node

    def _new_analyzer(self, expr_dag, on_sub=None):
        return ana.Analyzer(expr_dag, on_sub=on_sub)

    def _new_rewriter(self, expr_dag):
        return rwr.Rewriter(expr_dag)

    def _compile_sql(self, expr_dag):
        self._rewrite(expr_dag)

        sa = self._compile(expr_dag.root, expr_dag)
        return self._sa_to_sql(sa)

    def _sa_to_sql(self, sa):
        try:
            return sa.compile(compile_kwargs={"literal_binds": True})
        except NotImplementedError:
            return sa.compile()

    def _status_ui(self, ui):
        ui.status('Try to execute by sqlalchemy...', clear_keys=True)

    @classmethod
    def _get_or_create_conn(cls, engine):
        if engine in _engine_to_connections:
            return _engine_to_connections[engine]
        conn = engine.connect()
        _engine_to_connections[engine] = conn
        return conn

    def _get_cast_func(self, tp):
        df_types_to_builtin_types = {
            types.Integer: int,
            types.Float: float
        }
        for df_type, builtin_type in six.iteritems(df_types_to_builtin_types):
            if isinstance(tp, df_type):
                return lambda value: builtin_type(value)
        return lambda value: value

    def _get_sa_table(self, table_name, engine, schema):
        import sqlalchemy

        metadata = sqlalchemy.MetaData(bind=engine)
        columns = df_schema_to_sqlalchemy_columns(schema, engine=engine)
        table = sqlalchemy.Table(table_name, metadata, *columns, extend_existing=True)

        return table

    def _get_result(self, table_name, engine, schema, head, tail):
        import sqlalchemy

        table = self._get_sa_table(table_name, engine, schema)
        sa = sqlalchemy.select([table.alias('t1')])
        if head:
            sa = sa.limit(head)
        elif tail:
            count = sa.alias('t2').count().execute().scalar()
            skip = max(count - tail, 0)
            if skip:
                sa = sa.offset(skip)

        conn = self._get_or_create_conn(engine)
        return conn.execute(sa)

    @classmethod
    def _create_table(cls, table_name, sa, expr):
        from .ext import SACreateTempTableAs

        return SACreateTempTableAs(table_name, sa)

    def _convert_result(self, result, schema):
        res = [list(r) for r in result]
        if len(res) > 0:
            record = res[0]
            for i, r in enumerate(record):
                if schema[i].type != types.decimal and isinstance(r, Decimal):
                    cast = self._get_cast_func(schema[i].type)
                    for r in res:
                        r[i] = cast(r[i])

        return res

    def _run(self, sa, ui, expr_dag, src_expr, progress_proportion=1,
             head=None, tail=None, fetch=True, tmp_table_name=None,
             execution_options=None):
        self._status_ui(ui)

        schema = expr_dag.root.schema
        execution_options = dict() if execution_options is None else execution_options
        try:
            if isinstance(src_expr, (CollectionExpr, SequenceExpr)):
                if tmp_table_name is None:
                    tmp_table_name = gen_temp_table()
                to_execute = self._create_table(tmp_table_name, sa, expr_dag.root)
                log('Sql compiled:')
                log(self._sa_to_sql(to_execute))
                conn = self._get_or_create_conn(sa.bind)
                conn.execution_options(**execution_options).execute(to_execute)

                if fetch:
                    result = self._get_result(tmp_table_name, sa.bind, schema, head, tail)
                    res = self._convert_result(result, schema)
                    return tmp_table_name, res
                else:
                    return tmp_table_name, None
            else:
                log('Sql compiled:')
                log(self._sa_to_sql(sa))

                conn = self._get_or_create_conn(sa.bind)
                res = conn.execution_options(**execution_options).execute(sa).scalar()
                if src_expr.dtype != types.decimal and isinstance(res, Decimal):
                    return self._get_cast_func(src_expr.dtype)(res)
                return res
        finally:
            ui.inc(progress_proportion)

    def _cache(self, expr_dag, dag, expr, **kwargs):
        if is_source_collection(expr_dag.root) or \
                is_constant_scalar(expr_dag.root):
            return

        execute_dag = ExprDAG(expr_dag.root, dag=expr_dag)

        if isinstance(expr, CollectionExpr):
            table_name = gen_temp_table()
            table = self._get_table(table_name, expr_dag)
            root = expr_dag.root
            sub = CollectionExpr(_source_data=table, _schema=expr.schema)
            sub.add_deps(root)
            expr_dag.substitute(root, sub)

            kw = dict(kwargs)

            execute_node = self._execute(execute_dag, dag, expr,
                                         execute_kw={'fetch': False, 'temp_table_name': table_name},
                                         **kw)

            def callback(res):
                if isinstance(expr, DynamicMixin):
                    sub._schema = res.schema
                    refresh_dynamic(sub, expr_dag)

            execute_node.callback = callback
        else:
            assert isinstance(expr, Scalar)  # sequence is not cache-able

            class ValueHolder(object):
                pass

            sub = Scalar(_value_type=expr.dtype)
            sub._value = ValueHolder()

            execute_node = self._execute(execute_dag, dag, expr, **kwargs)

            def callback(res):
                sub._value = res

            execute_node.callback = callback

        return sub, execute_node

    def _compile(self, expr, dag):
        compiler = SQLAlchemyCompiler(dag)
        return compiler.compile(expr)

    def _get_table(self, table_name, expr_dag, bind=None):
        if bind is None:
            bind = next(e for e in expr_dag.traverse()
                        if is_source_collection(e) and e._source_data.bind)._source_data.bind

        return self._get_sa_table(table_name, bind, expr_dag.root.schema)

    def _do_execute(self, expr_dag, expr, ui=None, progress_proportion=1,
                    head=None, tail=None, **kwargs):
        expr_dag = self._convert_table(expr_dag)
        self._rewrite(expr_dag)

        src_expr = expr
        expr = expr_dag.root

        if isinstance(expr, Scalar) and expr.value is not None:
            ui.inc(progress_proportion)
            return expr.value

        sqlalchemy_expr = self._compile(expr, expr_dag)

        fetch = kwargs.pop('fetch', True)
        temp_table_name = kwargs.pop('temp_table_name', None)
        execution_options = kwargs.pop('execution_options',
                                       options.df.sqlalchemy.execution_options)
        result = self._run(sqlalchemy_expr, ui, expr_dag, src_expr,
                           progress_proportion=progress_proportion,
                           head=head, tail=tail, fetch=fetch,
                           tmp_table_name=temp_table_name,
                           execution_options=execution_options)

        if not isinstance(src_expr, Scalar):
            # reset schema
            if isinstance(src_expr, CollectionExpr) and \
                    (isinstance(src_expr._schema, DynamicSchema) or
                         any(isinstance(col.type, Unknown) for col in src_expr._schema.columns)):
                src_expr._schema = expr_dag.root.schema
            table_name, result = result
            table = self._get_table(table_name, expr_dag, sqlalchemy_expr.bind)
            context.cache(src_expr, table)
            if fetch:
                return ResultFrame(result, schema=expr_dag.root.schema)
            else:
                return table
        else:
            context.cache(src_expr, result)
            return result

    def _do_persist(self, expr_dag, expr, name, partitions=None, partition=None,
                    odps=None, project=None, ui=None,
                    progress_proportion=1, execute_percent=0.5, lifecycle=None,
                    overwrite=True, drop_table=False, create_table=True,
                    drop_partition=False, create_partition=False, cast=False, **kwargs):
        expr_dag = self._convert_table(expr_dag)
        self._rewrite(expr_dag)

        src_expr = expr
        expr = expr_dag.root
        odps = odps or self._odps

        try:
            import pandas
        except ImportError:
            raise DependencyNotInstalledError('persist requires for pandas')

        df = self._do_execute(expr_dag, src_expr, ui=ui,
                              progress_proportion=progress_proportion * execute_percent, **kwargs)
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
        write_table(df, table, ui=ui, cast=cast, overwrite=overwrite, partitions=partitions, partition=partition,
                    progress_proportion=progress_proportion * (1 - execute_percent))

        if partition:
            partition = PartitionSpec(partition)
            filters = []
            for k in partition.keys:
                filters.append(lambda x: x[k] == partition[k])
            return DataFrame(odps.get_table(name, project=project)).filter(*filters)
        return DataFrame(odps.get_table(name, project=project))
