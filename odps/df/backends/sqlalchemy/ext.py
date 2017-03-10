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

from sqlalchemy.sql.expression import FunctionElement, Executable, ClauseElement
from sqlalchemy.ext.compiler import compiles


class SALog(FunctionElement):
    def __init__(self, name, base, *args, **kwargs):
        self.name = name
        self.base = base
        super(SALog, self).__init__(*args, **kwargs)


@compiles(SALog)
def visit_log(log, compiler, **_):
    if log.base is None:
        return 'ln(%s)' % compiler.process(log.clauses)
    else:
        return 'log(%s)' % compiler.process(log.clauses)


@compiles(SALog, 'mssql')
def visit_log(log, compiler, **_):
    if log.base is None:
        return 'log(%s)' % compiler.process(log.clauses)
    else:
        return 'log(%s)' % compiler.process(
            log.clauses._constructor(log.clauses.clauses[::-1]))  # in mssql, log(num, base)


class SATruncate(FunctionElement):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super(SATruncate, self).__init__(*args, **kwargs)


@compiles(SATruncate, 'postgresql')
@compiles(SATruncate, 'oracle')
def visit_truncate(trunc, compiler, **_):
    return 'trunc(%s)' % compiler.process(trunc.clauses)


@compiles(SATruncate, 'mysql')
def visit_truncate(trunc, compiler, **_):
    return 'truncate(%s)' % compiler.process(trunc.clauses)


@compiles(SATruncate, 'mssql')
def visit_truncate(trunc, compiler, **_):
    return 'round(%s, 1)' % compiler.process(trunc.clauses)


class SACreateTempTableAs(Executable, ClauseElement):

    def __init__(self, name, query):
        self.name = name
        self.query = query
        if hasattr(self.query, 'bind') and self.query.bind is not None \
                and self.bind is None:
            self._bind = query.bind


@compiles(SACreateTempTableAs, 'mysql')
def create_temp_table_as(element, compiler, **kw):
    return 'CREATE TEMPORARY TABLE %s AS %s' % (
        element.name,
        compiler.process(element.query),
    )


@compiles(SACreateTempTableAs)
def create_temp_table_as(element, compiler, **kw):
    return 'CREATE TEMP TABLE %s AS %s' % (
        element.name,
        compiler.process(element.query),
    )
