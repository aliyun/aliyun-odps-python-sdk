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

try:
    import sqlalchemy
    from sqlalchemy import create_engine, Column, Integer, MetaData

    from .ext import *

    has_sqlalchemy = True
except ImportError:
    has_sqlalchemy = False

from ....models import Table
from ....config import options
from ....types import Decimal
from ... import types as df_types
from ...expr.reduction import Max, GroupedMax, Min, GroupedMin
from ..sqlalchemy.compiler import SQLAlchemyCompiler
from ..sqlalchemy.types import df_schema_to_sqlalchemy_columns
from .models import SeahawksTable


_url_to_sqlalchemy_engine_and_metadatas = {}


class SeahawksCompiler(SQLAlchemyCompiler):

    def __init__(self, expr_dag, odps):
        super(SeahawksCompiler, self).__init__(expr_dag)
        self._odps = odps

        seahawks_url = odps._seahawks_url or options.seahawks_url
        if seahawks_url in _url_to_sqlalchemy_engine_and_metadatas:
            self._sqlalchemy_engine, self._metadata = \
                _url_to_sqlalchemy_engine_and_metadatas[seahawks_url]
        else:
            _sqlalchemy_engine = self._sqlalchemy_engine = \
                create_engine(seahawks_url, isolation_level='AUTOCOMMIT')
            _sqlalchemy_engine.dialect.identifier_preparer.initial_quote = ''
            _sqlalchemy_engine.dialect.identifier_preparer.final_quote = ''
            self._metadata = MetaData(bind=_sqlalchemy_engine)
            _url_to_sqlalchemy_engine_and_metadatas[seahawks_url] = \
                _sqlalchemy_engine, self._metadata

    def _mapping_odps_table(self, t, df_schema):
        if isinstance(t, SeahawksTable):  # from cache, actually heap table
            table_name = t.name
        else:
            table_name = 'odps.{0}.{1}'.format(t.project.name, t.name)
        columns = df_schema_to_sqlalchemy_columns(df_schema)
        return sqlalchemy.Table(table_name, self._metadata, *columns, extend_existing=True)

    def visit_source_collection(self, expr):
        table = next(expr.data_source())

        if isinstance(table, Table):
            if any(isinstance(col.type, Decimal) for col in table.schema.columns):
                # FIXME: decimal and datetime are not supported by seahawks by now
                raise NotImplementedError
            table = self._mapping_odps_table(table, expr.schema)

        self._add(expr, table.alias(self._new_alias()))

    def visit_reduction(self, expr):
        if isinstance(expr, (Max, Min, GroupedMax, GroupedMin)) and \
                expr._input.dtype == df_types.string:
            # string max or min act different
            raise NotImplementedError

        return super(SeahawksCompiler, self).visit_reduction(expr)
