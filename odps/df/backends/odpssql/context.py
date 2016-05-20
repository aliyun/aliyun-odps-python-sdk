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
import uuid
import time
import weakref

from ....compat import OrderedDict, six
from ....utils import TEMP_TABLE_PREFIX
from ....errors import ODPSError


UDF_CLASS_NAME = 'PyOdpsFunc'

_replaced_exprs = weakref.WeakKeyDictionary()


class ODPSContext(object):
    def __init__(self, odps, indent_size=2):
        self._odps = odps

        self._index = itertools.count(1)
        self._expr_alias = dict()
        self._compiled_exprs = dict()

        self._col_index = itertools.count(1)
        self._need_alias_columns = dict()
        self._need_alias_column_indexes = dict()

        self._select_index = itertools.count(1)

        self._func_to_udfs = OrderedDict()
        self._registered_funcs = OrderedDict()
        self._func_to_functions = OrderedDict()
        self._func_to_resources = OrderedDict()

        self._indent_size = indent_size
        self._mapjoin_hints = []

        self._to_drops = []

    def next_select_id(self):
        return next(self._select_index)

    def new_alias_table_name(self):
        return 't%s' % next(self._index)

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

    def add_expr_compiled(self, expr, compiled):
        if self._compiled_exprs.get(id(expr)):
            pass
        self._compiled_exprs[id(expr)] = compiled

    def get_expr_compiled(self, expr):
        if id(expr) not in self._compiled_exprs:
            pass
        return self._compiled_exprs[id(expr)]

    def _gen_udf_name(self):
        return 'pyodps_udf_%s_%s' % (int(time.time()), str(uuid.uuid4()).replace('-', '_'))

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

    def create_udfs(self):
        self._func_to_functions.clear()

        for func, udf in six.iteritems(self._func_to_udfs):
            udf_name = self._registered_funcs[func]
            py_resource = self._odps.create_resource(udf_name + '.py', 'py', file_obj=udf)
            self._to_drops.append(py_resource)

            resources = [py_resource, ]
            if func in self._func_to_resources:
                for _, name, _, create, table_name in self._func_to_resources[func]:
                    if not create:
                        resources.append(name)
                    else:
                        res = self._odps.create_resource(name, 'table',
                                                         table_name=table_name)
                        resources.append(res)
                        self._to_drops.append(res)

            function = self._odps.create_function(
                    udf_name, class_type='{0}.{1}'.format(udf_name, UDF_CLASS_NAME),
                    resources=resources)

            self._func_to_functions[func] = function
            self._to_drops.append(function)

    def add_replaced_expr(self, expr, to_replace):
        _replaced_exprs[expr] = to_replace

    def get_replaced_expr(self, expr):
        return _replaced_exprs.get(expr)

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

    def _drop_silent(self, obj):
        try:
            obj.drop()
        except ODPSError:
            pass

    def close(self):
        [self._drop_silent(it) for it in self._to_drops]
