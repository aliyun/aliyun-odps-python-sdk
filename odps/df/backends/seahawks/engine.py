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

from ..sqlalchemy.engine import SQLAlchemyEngine
from ..odpssql.types import df_schema_to_odps_schema
from .compiler import SeahawksCompiler
from .models import SeahawksTable


class SeahawksEngine(SQLAlchemyEngine):

    def __init__(self, odps):
        super(SeahawksEngine, self).__init__(odps=odps)

    def _compile(self, expr, dag):
        compiler = SeahawksCompiler(dag, self._odps)
        return compiler.compile(expr)

    def _status_ui(self, ui):
        ui.status('Try to execute by seahawks...')

    @classmethod
    def _create_table(cls, table_name, sa, expr):
        from .ext import CreateTempTableAs

        return CreateTempTableAs(table_name, sa, expr)

    def _get_table(self, table_name, expr_dag, bind=None):
        return SeahawksTable(name=table_name,
                             schema=expr_dag.root.schema)
