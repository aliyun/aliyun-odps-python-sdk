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

import itertools
import os
import tarfile
import uuid
import time

from .... import tempobj
from ....compat import OrderedDict, BytesIO, six
from ....utils import TEMP_TABLE_PREFIX
from ....errors import ODPSError


UDF_CLASS_NAME = 'PyOdpsFunc'

expr_to_sql = dict()
expr_deps = dict()
expr_ref_name = dict()


class ODPSContext(object):
    def __init__(self, odps, indent_size=2):
        self._odps = odps

        self._index = itertools.count(1)
        self._expr_alias = dict()
        self._compiled_exprs = dict()
        self._expr_raw_args = dict()

        self._col_index = itertools.count(1)
        self._need_alias_columns = dict()
        self._need_alias_column_indexes = dict()

        self._select_index = itertools.count(1)
        self._expr_ref_index = itertools.count(1)

        self._func_to_udfs = OrderedDict()
        self._registered_funcs = OrderedDict()
        self._func_to_functions = OrderedDict()
        self._func_to_resources = OrderedDict()

        self._indent_size = indent_size
        self._mapjoin_hints = []
        self._skewjoin_hints = []

        self._path_to_resources = dict()

        self._to_drops = []

    def next_select_id(self):
        return next(self._select_index)

    def new_alias_table_name(self):
        return 't%s' % next(self._index)

    def new_ref_name(self):
        return '@c%d' % next(self._expr_ref_index)

    def register_collection(self, expr):
        expr_id = id(expr)
        if expr_id in self._expr_alias:
            return self._expr_alias[expr_id]

        self._expr_alias[expr_id] = self.new_alias_table_name()
        return self._expr_alias[expr_id]

    def get_collection_alias(self, expr, create=False, silent=False):
        try:
            return self._expr_alias[id(expr)], False
        except KeyError as e:
            if create:
                return self.register_collection(expr), True
            if not silent:
                raise e

    def remove_collection_alias(self, expr):
        if id(expr) in self._expr_alias:
            del self._expr_alias[id(expr)]

    def add_expr_compiled(self, expr, compiled):
        self._compiled_exprs[id(expr)] = compiled

    def get_expr_compiled(self, expr):
        return self._compiled_exprs[id(expr)]

    def _gen_udf_name(self):
        return 'pyodps_udf_%s_%s' % (int(time.time()), str(uuid.uuid4()).replace('-', '_'))

    def _gen_resource_name(self):
        return 'pyodps_res_%s_%s' % (int(time.time()), str(uuid.uuid4()).replace('-', '_'))

    def _gen_table_name(self):
        return '%s_%s_%s' % (TEMP_TABLE_PREFIX, int(time.time()),
                             str(uuid.uuid4()).replace('-', '_'))

    def register_udfs(self, func_to_udfs, func_to_resources):
        self._func_to_udfs = func_to_udfs
        for func in func_to_udfs.keys():
            self._registered_funcs[func] = self._gen_udf_name()
        self._func_to_resources = func_to_resources

    def get_udf(self, func):
        return self._registered_funcs[func]

    def get_udf_count(self):
        return len(self._registered_funcs)

    def prepare_resources(self, libraries):
        from ....models import Resource

        if libraries is None:
            return None

        ret_libs = []
        for lib in libraries:
            if isinstance(lib, Resource):
                ret_libs.append(lib)
                continue
            elif lib in self._path_to_resources:
                ret_libs.append(self._path_to_resources[lib])
                continue

            tarbinary = BytesIO()
            tar = tarfile.open(fileobj=tarbinary, mode='w:gz')
            if os.path.isfile(lib):
                with open(lib, 'rb') as fo:
                    finfo = tarfile.TarInfo('pyodps_files/' + os.path.basename(lib))
                    finfo.size = os.path.getsize(lib)
                    tar.addfile(finfo, fo)
            else:
                base_dir = os.path.dirname(os.path.abspath(lib))
                for root, dirs, files in os.walk(lib):
                    for f in files:
                        fpath = os.path.join(root, f)
                        rpath = os.path.relpath(fpath, base_dir).replace(os.path.sep, '/')
                        with open(fpath, 'rb') as fo:
                            finfo = tarfile.TarInfo(rpath)
                            finfo.size = os.path.getsize(fpath)
                            tar.addfile(finfo, fo)
            tar.close()

            res_name = self._gen_resource_name() + '.tar.gz'
            res = self._odps.create_resource(res_name, 'archive', file_obj=tarbinary.getvalue(), temp=True)
            tempobj.register_temp_resource(self._odps, res_name)
            self._path_to_resources[lib] = res
            self._to_drops.append(res)
            ret_libs.append(res)
        return ret_libs

    def create_udfs(self, libraries=None):
        self._func_to_functions.clear()

        for func, udf in six.iteritems(self._func_to_udfs):
            udf_name = self._registered_funcs[func]
            py_resource = self._odps.create_resource(udf_name + '.py', 'py', file_obj=udf, temp=True)
            tempobj.register_temp_resource(self._odps, udf_name + '.py')
            self._to_drops.append(py_resource)

            resources = [py_resource, ]
            if func in self._func_to_resources:
                for _, name, _, create, table_name in self._func_to_resources[func]:
                    if not create:
                        resources.append(name)
                    else:
                        res = self._odps.create_resource(name, 'table', table_name=table_name, temp=True)
                        tempobj.register_temp_resource(self._odps, name)
                        resources.append(res)
                        self._to_drops.append(res)
            if libraries is not None:
                resources.extend(libraries)

            function = self._odps.create_function(
                    udf_name, class_type='{0}.{1}'.format(udf_name, UDF_CLASS_NAME),
                    resources=resources)
            tempobj.register_temp_function(self._odps, udf_name)

            self._func_to_functions[func] = function
            self._to_drops.append(function)

    def add_need_alias_column(self, column):
        if id(column) in self._need_alias_column_indexes:
            return self._need_alias_column_indexes[id(column)]
        symbol = 'col_%s' % next(self._col_index)
        self._need_alias_columns[id(column)] = column
        self._need_alias_column_indexes[id(column)] = symbol
        return symbol

    def get_all_need_alias_column_symbols(self):
        for col_id, column in six.iteritems(self._need_alias_columns):
            symbol = self._need_alias_column_indexes[col_id]
            yield symbol, column


    #####################################
    # mem cache relatives
    #####################################

    def register_mem_cache_sql(self, expr, sql):
        expr_to_sql[expr._id] = sql

    def is_expr_mem_cached(self, expr):
        return expr._id in expr_to_sql

    def get_mem_cached_sql(self, expr):
        return expr_to_sql[expr._id]

    def register_mem_cache_dep(self, expr, dep):
        if expr._id not in expr_deps:
            expr_deps[expr._id] = set()
        expr_deps[expr._id].add(dep._id)

    def get_mem_cache_ref_name(self, expr, create=True):
        if not create:
            return expr_ref_name[expr._id]
        ref_name = expr_ref_name.get(expr._id, self.new_ref_name())
        if expr._id not in expr_ref_name:
            expr_ref_name[expr._id] = ref_name
        return ref_name

    def get_mem_cache_dep_sqls(self, *expr_ids):
        sqls = []
        fetched = dict()

        def h(eid):
            if eid in expr_deps:
                for dep_id in expr_deps[eid]:
                    if dep_id not in fetched:
                        h(dep_id)

            if eid not in fetched:
                origin_sql = expr_to_sql[eid]
                sql = '{0} := CACHE ON {1}'.format(
                    expr_ref_name[eid], origin_sql
                )
                sqls.append(sql)
                fetched[eid] = True

        for expr_id in expr_ids:
            h(expr_id)
        return sqls

    def _drop_silent(self, obj):
        try:
            obj.drop()
        except ODPSError:
            pass

    def close(self):
        [self._drop_silent(it) for it in self._to_drops]
