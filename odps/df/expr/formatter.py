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

from operator import itemgetter

from ... import utils, models, compat
from .expressions import *
from .collections import SortedColumn
from .reduction import Count, GroupedCount
from .window import Window
from .merge import JoinCollectionExpr


class ExprFormatter(object):
    def __init__(self, expr, indent_size=2):
        self._expr = expr

        self._ref_id = 0
        self._cache = dict()
        self._is_referred = dict()

        self._indent_size = 2

    def _next_ref_id(self):
        curr = self._ref_id
        self._ref_id += 1
        return 'ref_%s' % curr

    def _format_source_collection(self, expr, buf=None, indent=0):
        expr_id = id(expr)
        if expr_id in self._cache:
            return self._cache[expr_id]

        ref_id = self._next_ref_id()

        need_cache = buf is None
        if need_cache:
            buf = six.StringIO()
            buf.write('Collection: {0}\n'.format(ref_id))

            indent = self._indent_size

        if isinstance(expr._source_data, models.Table):
            buf.write(utils.indent(repr(expr._source_data), indent))

        if need_cache:
            self._cache[expr_id] = ref_id, buf.getvalue()
            return self._cache[expr_id]
        else:
            return buf.getvalue()

    def _format_transformed_collection(self, expr, buf=None, indent=0):
        expr_id = id(expr)
        if expr_id in self._cache:
            return self._cache[expr_id]

        table_aliases = dict()
        if isinstance(expr, JoinCollectionExpr):
            lhs, rhs = expr.lhs, expr.rhs
            for s in (lhs, rhs):
                ref_id, _ = self._format_collection(s)
                table_aliases[s] = ref_id
        else:
            input = expr.input
            input_ref_id, _ = self._format_collection(input)
            table_aliases[input] = input_ref_id

        ref_id = self._next_ref_id()

        need_cache = buf is None
        if need_cache:
            buf = six.StringIO()
            buf.write('Collection: {0}\n'.format(ref_id))

            indent = self._indent_size

        buf.write(utils.indent('{0}[collection]\n'.format(expr.node_name), indent))
        for name, node in expr.iter_args():
            if name.startswith('collection'):
                content = '{0}: {1}\n'.format(name, table_aliases[node])
            else:
                if node is None:
                    continue
                content = '{0}:\n'.format(name)
            buf.write(utils.indent(content, indent+self._indent_size))
            if isinstance(node, list):
                for n in node:
                    self._format_node(n, buf, indent=indent+self._indent_size*2)
            else:
                self._format_node(node, buf, indent=indent+self._indent_size*2)

        if need_cache:
            self._cache[expr_id] = ref_id, buf.getvalue()
            return self._cache[expr_id]
        else:
            return buf.getvalue()

    def _format_collection(self, expr, buf=None, indent=0):
        if isinstance(expr, CollectionExpr) and expr._source_data is not None:
            return self._format_source_collection(expr, buf=buf, indent=indent)
        else:
            return self._format_transformed_collection(expr, buf=buf, indent=indent)

    def _format_column_content(self, expr):
        input_ref_id, _ = self._format_collection(expr._input)

        return "{0} = Column[{1}] '{2}' from collection {3}\n".format(
            expr.name, expr.output_type(), expr.source_name, input_ref_id)

    def _format_column(self, expr, buf, indent=0):
        content = self._format_column_content(expr)
        buf.write(utils.indent(content, indent))

    def _format_count_table(self, expr, buf, indent=0):
        input_ref_id, _ = self._format_collection(expr.input)

        buf.write(utils.indent("{0} = {1}[{2}]\n".format(
                expr.name, expr.node_name, expr.output_type()), indent))
        buf.write(utils.indent("collection: {0}\n".format(input_ref_id), indent+self._indent_size))

    def _format_sorted_column(self, expr, buf, indent=0):
        if isinstance(expr.input, Column):
            col = expr.input
            input_ref_id, _ = self._format_collection(expr.input.input)
        else:
            col = expr
            input_ref_id, _ = self._format_collection(expr._input)

        contents = ['SortKey', utils.indent('by', self._indent_size),
                    utils.indent(self._format_column_content(col), self._indent_size*2).rstrip('\n'),
                    utils.indent('ascending', self._indent_size),
                    utils.indent(str(expr._ascending), self._indent_size*2)
                    ]
        buf.write(utils.indent('\n'.join(contents+['']), indent))

    def _format_window(self, expr, buf, indent=0):
        if self._need_output_seq_name(expr, indent=indent):
            buf.write(utils.indent('{0} = '.format(expr.name), indent))
        buf.write(utils.indent('{0}[{1}]\n'.format(expr.node_name, expr.output_type()), indent))

        kw = list(expr.iter_args())

        arg = kw[0][1]
        if isinstance(arg, CollectionExpr):
            self._format_collection(arg)
        else:
            self._format_node(arg, buf, indent=indent+self._indent_size)

        for name, node in kw[1:]:
            if node is None:
                continue
            content = '{0}:\n'.format(name)

            buf.write(utils.indent(content, indent+self._indent_size))
            if isinstance(node, (list, tuple)):
                for n in node:
                    self._format_node(n, buf, indent=indent+self._indent_size*2)
            else:
                self._format_node(node, buf, indent=indent+self._indent_size*2)

    @classmethod
    def _need_output_seq_name(cls, expr, indent=0):
        if indent == 0 and expr.name:
            return True

        if expr.source_name is not None and expr.name is not None and \
                expr.source_name != expr.name:
            return True

    def _format_sub_node(self, expr, buf, indent=0):
        for name, node in expr.iter_args():
            if name.startswith('_'):
                self._format_node(node, buf, indent=indent+self._indent_size)
                continue
            if node is None:
                continue
            content = '{0}:\n'.format(name)

            buf.write(utils.indent(content, indent+self._indent_size))
            if isinstance(node, (list, tuple)):
                for n in node:
                    self._format_node(n, buf, indent=indent+self._indent_size*2)
            else:
                self._format_node(node, buf, indent=indent+self._indent_size*2)

    def _format_node(self, expr, buf, indent=0):
        if isinstance(expr, SortedColumn):
            self._format_sorted_column(expr, buf, indent=indent)
        elif isinstance(expr, Column):
            self._format_column(expr, buf, indent=indent)
        elif isinstance(expr, Window):
            self._format_window(expr, buf, indent=indent)
        elif isinstance(expr, CollectionExpr):
            self._format_collection(expr, buf, indent)
        elif isinstance(expr, (Count, GroupedCount)) and isinstance(expr.input, CollectionExpr):
            self._format_count_table(expr, buf, indent=indent)
        elif isinstance(expr, SequenceExpr):
            txt = '{0}[{1}]\n'.format(expr.node_name, expr.output_type())
            if self._need_output_seq_name(expr, indent=indent):  # TODO: need to check
                txt = '{0} = {1}'.format(expr.name, txt)
            buf.write(utils.indent(txt, indent))
            self._format_sub_node(expr, buf, indent=indent)
        elif isinstance(expr, Scalar) and getattr(expr, '_input', None) is not None:
            buf.write(utils.indent(
                '{0} = {1}[{2}]\n'.format(
                    expr.node_name.lower(), expr.node_name, expr.output_type()), indent))
            self._format_sub_node(expr, buf, indent=indent)
        elif isinstance(expr, Scalar):
            buf.write(utils.indent(expr.output_type(), indent))
            buf.write('\n')
            buf.write(utils.indent(repr(expr._value)+'\n', indent+self._indent_size))

    def format(self):
        buf = six.StringIO()
        self._format_node(self._expr, buf)

        result = six.StringIO()
        collections = sorted(compat.lvalues(self._cache), key=itemgetter(0))
        result.write('\n'.join([it[1] for it in collections]))
        result.write('\n')
        result.write(buf.getvalue())

        return result.getvalue()

    __call__ = format
