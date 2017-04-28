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

import sys
import tarfile
import zipfile

from .compiler import PandasCompiler
from .types import pd_to_df_schema
from ..core import Engine, ExecuteNode, ExprDAG
from ..frame import ResultFrame
from ... import DataFrame
from ...expr.core import ExprDictionary
from ...expr.expressions import *
from ...expr.dynamic import DynamicMixin
from ...backends.odpssql.types import df_schema_to_odps_schema, df_type_to_odps_type
from ..errors import CompileError
from ..utils import refresh_dynamic, write_table
from ...utils import is_source_collection, is_constant_scalar
from ...types import DynamicSchema, Unknown
from ....models import Schema, Partition
from ....errors import ODPSError
from ....types import PartitionSpec
from ....lib.importer import CompressImporter
from .... import compat
from ..context import context
from . import analyzer as ana


class PandasExecuteNode(ExecuteNode):
    def __repr__(self):
        return 'Local execution by pandas backend'

    def _repr_html_(self):
        return '<p>Local execution by pandas backend</p>'


def with_thirdparty_libs(fun):
    def wrapped(self, *args, **kwargs):
        libraries = self._get_libraries(kwargs.get('libraries'))
        importer = self._build_library_importer(libraries)
        if importer is not None:
            sys.meta_path.append(importer)

        try:
            return fun(self, *args, **kwargs)
        finally:
            if importer is not None:
                sys.meta_path = [p for p in sys.meta_path if p is not importer]

    wrapped.__name__ = fun.__name__
    wrapped.__doc__ = fun.__doc__
    return wrapped


class PandasEngine(Engine):
    def __init__(self, odps=None):
        self._odps = odps

    def _new_execute_node(self, expr_dag):
        return PandasExecuteNode(expr_dag)

    def _run(self, expr_dag, pd_dag, ui=None, progress_proportion=1, **_):
        ui.status('Try to execute by local pandas...', clear_keys=True)

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
                for col in res.columns:
                    sub._source_data[col] = res[col]
                if isinstance(expr, DynamicMixin):
                    sub._schema = pd_to_df_schema(res)
                    refresh_dynamic(sub, expr_dag)

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

        return sub, execute_node

    def _build_library_importer(self, libraries):
        if libraries is None:
            return None

        readers = []

        for library in libraries:
            if isinstance(library, six.string_types):
                library = self._odps.get_resource(library)

            lib_name = library.name
            if lib_name.endswith('.zip') or lib_name.endswith('.egg') or lib_name.endswith('.whl'):
                readers.append(zipfile.ZipFile(library.open(mode='rb')))
            elif lib_name.endswith('.tar') or lib_name.endswith('.tar.gz') or lib_name.endswith('.tar.bz2'):
                from io import BytesIO
                if lib_name.endswith('.tar'):
                    mode = 'r'
                else:
                    mode = 'r:gz' if lib_name.endswith('.tar.gz') else 'r:bz2'
                readers.append(tarfile.open(fileobj=BytesIO(library.open(mode='rb').read()), mode=mode))
            else:
                raise ValueError(
                    'Unknown library type which should be one of zip(egg, wheel), tar, or tar.gz')

        return CompressImporter(*readers)

    @with_thirdparty_libs
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
            # reset schema
            if isinstance(src_expr, CollectionExpr) and \
                    (isinstance(src_expr._schema, DynamicSchema) or
                         any(isinstance(col.type, Unknown) for col in src_expr._schema.columns)):
                src_expr._schema = expr_dag.root.schema
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

    @with_thirdparty_libs
    def _do_persist(self, expr_dag, expr, name, ui=None, project=None,
                    partitions=None, partition=None, odps=None, lifecycle=None,
                    progress_proportion=1, execute_percent=0.5,
                    overwrite=True, drop_table=False, create_table=True,
                    drop_partition=False, create_partition=False, cast=False, **kwargs):
        expr_dag = self._convert_table(expr_dag)
        self._rewrite(expr_dag)

        src_expr = expr
        expr = expr_dag.root
        odps = odps or self._odps
        if odps is None:
            raise ODPSError('ODPS entrance should be provided')

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
            t = odps.get_table(name, project=project)
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
            if create_table and not odps.exist_table(name):
                schema = df_schema_to_odps_schema(schema)
                odps.create_table(name, schema, project=project, lifecycle=lifecycle)

        table = odps.get_table(name, project=project)
        write_table(df, table, ui=ui, cast=cast, partitions=partitions, partition=partition,
                    progress_proportion=progress_proportion*(1-execute_percent))

        if partition:
            partition = PartitionSpec(partition)
            filters = []
            for k in partition.keys:
                filters.append(lambda x: x[k] == partition[k])
            return DataFrame(odps.get_table(name, project=project)).filter(*filters)
        return DataFrame(odps.get_table(name, project=project))
