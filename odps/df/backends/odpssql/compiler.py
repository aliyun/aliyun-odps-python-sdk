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

import re
from datetime import datetime
from decimal import Decimal

from ...expr.reduction import *
from ...expr.arithmetic import BinOp, Add, Substract, Power, Invert, Negate, Abs
from ...expr.merge import JoinCollectionExpr, UnionCollectionExpr
from ...expr.element import MappedExpr, Func
from ...expr.window import CumSum
from ...expr.datetimes import DTScalar
from ...expr.collections import RowAppliedCollectionExpr
from ...expr import element
from ...expr import strings
from ...expr import datetimes
from ...utils import is_source_collection, traverse_until_source
from ... import types as df_types
from . import types
from .models import MemCacheReference
from ..core import Backend
from .... import utils
from ....models import Function
from ..errors import CompileError


BINARY_OP_COMPILE_DIC = {
    'Add': '+',
    'Substract': '-',
    'Multiply': '*',
    'Divide': '/',
    'Greater': '>',
    'GreaterEqual': '>=',
    'Less': '<',
    'LessEqual': '<=',
    'Equal': '==',
    'NotEqual': '!=',
    'And': 'and',
    'Or': 'or'
}

UNARY_OP_COMPILE_DIC = {
    'Negate': '-'
}

WINDOW_COMPILE_DIC = {
    'CumSum': 'sum',
    'CumMean': 'avg',
    'CumMedian': 'median',
    'CumStd': 'stddev',
    'CumMax': 'max',
    'CumMin': 'min',
    'CumCount': 'count',
    'Lag': 'lag',
    'Lead': 'lead',
    'Rank': 'rank',
    'DenseRank': 'dense_rank',
    'PercentRank': 'percent_rank',
    'RowNumber': 'row_number'
}

MATH_COMPILE_DIC = {
    'Abs': 'abs',
    'Sqrt': 'sqrt',
    'Sin': 'sin',
    'Sinh': 'sinh',
    'Cos': 'cos',
    'Cosh': 'cosh',
    'Tan': 'tan',
    'Tanh': 'tanh',
    'Exp': 'exp',
    'Arccos': 'acos',
    'Arcsin': 'asin',
    'Arctan': 'atan',
    'Ceil': 'ceil',
    'Floor': 'floor'
}

DATE_PARTS_DIC = {
    'Year': 'yyyy',
    'Month': 'mm',
    'Day': 'dd',
    'Hour': 'hh',
    'Minute': 'mi',
    'Second': 'ss'
}


class OdpsSQLCompiler(Backend):
    """
    OdpsSQLCompiler will compile an Expr into an ODPS SQL.
    """

    def __init__(self, ctx, indent_size=2, beautify=False):
        self._ctx = ctx
        self._indent_size = indent_size
        self._beautify = beautify

        # use for `join` or `union` operations etc.
        self._sub_compiles = defaultdict(lambda: list())
        # When encountering `join` or `union`, we will try to compile all child branches,
        # for each nodes of these branches, we should not check the uniqueness,
        # when compilation finishes, we substitute the children of `join` or `union` with None,
        # so the upcoming compilation will not visit its children.
        # When everything are done, we use the callbacks to substitute back the original children
        # of the `join` or `union` node.
        self._callbacks = list()

        # store the expr ids to do mem cache
        self._mem_ref_caches = set()

        self._re_init()

    def _re_init(self):
        self._select_clause = None
        self._from_clause = None
        self._where_clause = None
        self._group_by_clause = None
        self._having_clause = None
        self._order_by_clause = None
        self._limit = None

    def _cleanup(self):
        self._sub_compiles = defaultdict(lambda: list())
        for callback in self._callbacks:
            callback()
        self._callbacks = list()

    @classmethod
    def _need_recursive_handle_in_expr(cls, node):
        return isinstance(node, (element.IsIn, element.NotIn)) and \
               not all(n is None for n in node.args) and \
               isinstance(node.values[0], SequenceExpr)

    @classmethod
    def _retrieve_until_find_root(cls, expr):
        for node in traverse_until_source(expr, top_down=True, unique=True):
            if isinstance(node, (JoinCollectionExpr, UnionCollectionExpr)) and \
                    not all(n is None for n in node.args):
                yield node
            else:
                ins = [n for n in node.children() if cls._need_recursive_handle_in_expr(n)]
                if len(ins) > 0:
                    for n in ins:
                        yield n

    def _compile_union_node(self, expr, traversed):
        compiled = self._compile(expr.lhs)

        self._sub_compiles[expr].append(compiled)

        compiled = self._compile(expr.rhs)
        self._sub_compiles[expr].append(compiled)

        args = expr.args
        self._ctx._expr_raw_args[id(expr)] = args
        for arg_name in expr._args:
            setattr(expr, arg_name, None)

        def cb():
            for arg_name, arg in zip(expr._args, args):
                setattr(expr, arg_name, arg)

        self._callbacks.append(cb)

    def _compile_join_node(self, expr, traversed):
        travs = set()

        compiled, trav = self._compile(expr.lhs, return_traversed=True)
        travs.update(trav)
        if not is_source_collection(expr.lhs) and not isinstance(expr.lhs, JoinCollectionExpr):
            self._sub_compiles[expr].append(
                '(\n{0}\n) {1}'.format(utils.indent(compiled, self._indent_size),
                                     self._ctx.get_collection_alias(expr.lhs, create=True)[0])
            )
        else:
            self._sub_compiles[expr].append(self._ctx.get_expr_compiled(expr.lhs))

        compiled, trav = self._compile(expr.rhs, return_traversed=True)
        travs.update(trav)
        if not is_source_collection(expr.rhs):
            self._sub_compiles[expr].append(
                '(\n{0}\n) {1}'.format(utils.indent(compiled, self._indent_size),
                                     self._ctx.get_collection_alias(expr.rhs, create=True)[0])
            )
        else:
            self._sub_compiles[expr].append(self._ctx.get_expr_compiled(expr.rhs))

        if expr.predicate is None:
            self._sub_compiles[expr].append(None)
            traversed.update(travs)
        else:
            self._compile(expr.predicate, traversed)
            self._sub_compiles[expr].append(self._ctx.get_expr_compiled(expr.predicate))

        if expr._mapjoin:
            self._ctx._mapjoin_hints.append(self._ctx.get_collection_alias(expr.rhs)[0])

        args = expr.args
        self._ctx._expr_raw_args[id(expr)] = args
        for arg_name in expr._args:
            setattr(expr, arg_name, None)

        def cb():
            for arg_name, arg in zip(expr._args, args):
                setattr(expr, arg_name, arg)

        self._callbacks.append(cb)

    @classmethod
    def _find_table(cls, expr):
        return next(it for it in expr.traverse(top_down=True, unique=True)
                    if isinstance(it, CollectionExpr))

    def _compile_in_node(self, expr, traversed):
        self._compile(expr.input)
        self._sub_compiles[expr].append(self._ctx.get_expr_compiled(expr.input))

        to_sub = self._find_table(expr.values[0])[[expr.values[0], ]]
        compiled = self._compile(to_sub)
        self._sub_compiles[expr].append(compiled)

        args = expr.args
        self._ctx._expr_raw_args[id(expr)] = args
        for arg_name in expr._args:
            setattr(expr, arg_name, None)

        def cb():
            for arg_name, arg in zip(expr._args, args):
                setattr(expr, arg_name, arg)

        self._callbacks.append(cb)

    def _compile(self, expr, traversed=None,
                 return_traversed=False, root_expr=None):
        roots = self._retrieve_until_find_root(expr)

        if traversed is None:
            traversed = set()

        for root in roots:
            if root is not None:
                if isinstance(root, JoinCollectionExpr):
                    self._compile_join_node(root, traversed)
                elif isinstance(root, UnionCollectionExpr):
                    self._compile_union_node(root, traversed)
                elif isinstance(root, (element.IsIn, element.NotIn)):
                    self._compile_in_node(root, traversed)
                root.accept(self)
                traversed.add(id(root))

        for node in traverse_until_source(expr):
            if id(node) not in traversed:
                node.accept(self)
                traversed.add(id(node))

        if expr is root_expr and self._select_clause is None and \
                self._ctx._mapjoin_hints:
            self.add_select_clause(expr, '* ')

        sql = self.to_sql().strip()
        if not return_traversed:
            return sql
        return sql, traversed

    def compile(self, expr):
        try:
            sql = self._compile(expr, root_expr=expr)
            symbol_columns = dict(self._ctx.get_all_need_alias_column_symbols())
            sql = self._fill_back_columns(sql, symbol_columns)
            sql = self._re_join_select_fields(sql, symbol_columns)

            if self._mem_ref_caches:
                dep_sqls = self._ctx.get_mem_cache_dep_sqls(*self._mem_ref_caches)
                dep_sqls = [dep_sql if dep_sql.endswith(';') else dep_sql + ';'
                            for dep_sql in dep_sqls]
                return dep_sqls + [sql,]

            return sql
        finally:
            self._cleanup()

    def to_sql(self):
        lines = [
            'SELECT {0} '.format(self._select_clause or '*'),
            'FROM {0} '.format(self._from_clause),
        ]

        if self._where_clause:
            lines.append('WHERE {0} '.format(self._where_clause))
        if self._group_by_clause:
            lines.append(self._group_by_clause)
            if self._having_clause:
                lines.append('HAVING {0} '.format(self._having_clause))
        if self._order_by_clause:
            if self._order_by_clause.startswith('ORDER BY') and not self._limit and \
                    options.df.odps.sort.limit:
                # for `order by`, limit is required
                # for `sort by`, limit is unnecessary
                self._limit = options.df.odps.sort.limit
            lines.append(self._order_by_clause)
        if self._limit is not None:
            lines.append('LIMIT {0}'.format(self._limit))

        self._re_init()
        return '\n'.join(lines)

    def _fill_back_columns(self, sql, symbols_to_columns):
        symbol_compiled = dict()
        for symbol, column in six.iteritems(symbols_to_columns):
            collection, name = self._retrieve_column_alias_collection(column)
            symbol_compiled[symbol] = '{0}.{1}'.format(
                self._ctx.get_collection_alias(collection)[0],
                self._quote(name))
        sql = sql % symbol_compiled

        reg = re.compile('###\[(col_\d+)\]##')

        def repl(matched):
            symbol = matched.group(1)
            compiled = symbol_compiled[symbol]
            compiled_name = self._unquote(compiled)
            column = symbols_to_columns[symbol]
            if compiled_name != column.name:
                return ' AS {0}'.format(self._quote(column.name))
            else:
                return ''

        return reg.sub(repl, sql)

    def _re_join_select_fields(self, sql, symbols_to_columns):
        if not self._beautify:
            return sql

        reg = re.compile('/{_i(\d+)}')

        for select_idx in reg.findall(sql):
            s_regex_str = '//{{_i{0}}}(.+?){{_i{0}}}//'.format(select_idx)
            regex_str = '\n?( *)/{{_i{0}}}({1})+/'.format(select_idx, s_regex_str)
            regex = re.compile(regex_str, (re.M | re.DOTALL))
            s_regex = re.compile(s_regex_str, (re.M | re.DOTALL))

            def repl(matched):
                space = len(matched.group(1))
                joined = matched.group()

                fields = s_regex.findall(joined)
                sub = self._join_compiled_fields(fields)
                if not joined.startswith('\n'):
                    sub = sub.lstrip('\n')
                if space > self._indent_size:
                    return utils.indent(sub, space-self._indent_size)
                return sub

            sql = regex.sub(repl, sql)

        return sql

    def _retrieve_column_alias_collection(self, expr):
        column_name = expr.source_name

        collection = expr.input
        while True:
            if isinstance(collection, JoinCollectionExpr):
                idx, column_name = collection._column_origins[column_name]
                args = self._ctx._expr_raw_args[id(collection)]  # get the args which are substituted out
                lhs, rhs = args[0], args[1]
                collection = (lhs, rhs)[idx]
            elif self._ctx.get_collection_alias(collection, silent=True):
                return collection, column_name
            elif isinstance(getattr(collection, 'input', None), CollectionExpr):
                collection = collection.input

        raise CompileError('Cannot find table alias for column: \n%s' % repr_obj(expr))

    def sub_sql_to_from_clause(self, expr):
        sql = self.to_sql()

        alias, _ = self._ctx.get_collection_alias(expr, create=True)
        from_clause = '(\n{0}\n) {1}'.format(
            utils.indent(sql, self._indent_size), alias
        )

        self._re_init()
        self._from_clause = from_clause

    def add_select_clause(self, expr, select_clause):
        if self._select_clause is not None:
            self.sub_sql_to_from_clause(expr.input)
        elif self._order_by_clause is not None:
            self.sub_sql_to_from_clause(expr.input)
        elif isinstance(expr, Summary) and self._limit is not None:
            self.sub_sql_to_from_clause(expr.input)
        elif isinstance(expr, (GroupByCollectionExpr, MutateCollectionExpr)) and \
                self._limit is not None:
            self.sub_sql_to_from_clause(expr.input)
        elif isinstance(expr.input, ReshuffledCollectionExpr):
            self.sub_sql_to_from_clause(expr.input)

        self._select_clause = select_clause
        if self._ctx._mapjoin_hints:
            self._select_clause = '/*+mapjoin({0})*/ {1}'.format(
                ','.join(self._ctx._mapjoin_hints), self._select_clause)
            self._ctx._mapjoin_hints = []

    def add_from_clause(self, expr, from_clause):
        if self._from_clause is None:
            self._from_clause = from_clause

    def add_where_clause(self, expr, where_clause):
        if any(clause is not None for clause in
               (self._where_clause, self._select_clause, self._limit)):
            self.sub_sql_to_from_clause(expr.input)

        self._where_clause = where_clause

    def add_group_by_clause(self, expr, group_by_clause):
        if self._group_by_clause is not None:
            self.sub_sql_to_from_clause(expr.input)
        elif isinstance(expr, ReshuffledCollectionExpr) and \
                self._select_clause is not None:
            self.sub_sql_to_from_clause(expr.input)

        self._group_by_clause = group_by_clause

    def add_having_clause(self, expr, having_clause):
        if self._having_clause is None:
            self._having_clause = having_clause

        assert having_clause == self._having_clause

    def add_order_by_clause(self, expr, order_by_clause):
        if self._order_by_clause is not None:
            self.sub_sql_to_from_clause(expr.input)

        self._order_by_clause = order_by_clause

    def set_limit(self, expr, limit):
        if self._limit is not None:
            self.sub_sql_to_from_clause(expr.input)

        self._limit = limit

    def visit_source_collection(self, expr):
        source_data = expr._source_data
        alias = self._ctx.register_collection(expr)
        if isinstance(source_data, MemCacheReference):
            from_clause = '{0} {1}'.format(source_data.ref_name, alias)
            self.add_from_clause(expr, from_clause)
            self._ctx.add_expr_compiled(expr, from_clause)
            self._mem_ref_caches.add(source_data.expr_id)
        else:
            if options.df.quote:
                name = '%s.`%s`' % (source_data.project.name, source_data.name)
            else:
                name = '.'.join((source_data.project.name, source_data.name))

            from_clause = '{0} {1}'.format(name, alias)
            self.add_from_clause(expr, from_clause)
            self._ctx.add_expr_compiled(expr, from_clause)

    def _compile_select_field(self, field):
        compiled = self._ctx.get_expr_compiled(field)

        if not isinstance(field, Column) or field._source_data_type != field._data_type:
            compiled = '{0} AS {1}'.format(compiled, self._quote(field.name))
        else:
            if compiled.startswith('%(') and compiled.endswith(')s'):
                symbol = compiled[2:-2]
                compiled = '{0}###[{1}]##'.format(compiled, symbol)
            else:
                compiled_name = self._unquote(compiled)
                if field.name != compiled_name:
                    compiled = '{0} AS {1}'.format(compiled, self._quote(field.name))

        return compiled

    def _join_select_fields(self, fields):
        if not self._beautify:
            return ', '.join(fields)
        else:
            select_id = self._ctx.next_select_id()

            def h(field):
                return '//{{_i{0}}}{1}{{_i{0}}}//'.format(select_id, field)

            return utils.indent('\n/{{_i{0}}}{1}/'.format(select_id, ''.join([h(f) for f in fields])),
                                self._indent_size)

    def _join_compiled_fields(self, fields):
        if not self._beautify:
            return ', '.join(fields)
        else:
            buf = six.StringIO()
            buf.write('\n')

            split_fields = [field.rsplit(' AS ', 1) for field in fields]
            get = lambda s: s if '\n' not in s else s.rsplit('\n', 1)[1]
            max_length = max(len(get(f[0])) for f in split_fields)

            for f in split_fields:
                if len(f) > 1:
                    buf.write(f[0].ljust(max_length))
                    buf.write(' AS ')
                    buf.write(f[1])
                else:
                    buf.write(f[0])
                buf.write(',\n')

            return utils.indent(buf.getvalue()[:-2], self._indent_size)

    def visit_project_collection(self, expr):
        fields = expr._fields
        compiled_fields = [self._compile_select_field(field)
                           for field in fields]

        compiled = self._join_select_fields(compiled_fields)

        self._ctx.add_expr_compiled(expr, compiled)
        self.add_select_clause(expr, compiled)

    def visit_filter_collection(self, expr):
        predicate = expr.args[1]
        compiled = self._ctx.get_expr_compiled(predicate)

        self._ctx.add_expr_compiled(expr, compiled)
        self.add_where_clause(expr, compiled)

    def visit_filter_partition_collection(self, expr):
        compiled = self._ctx.get_expr_compiled(expr._predicate)

        self._ctx.add_expr_compiled(expr, compiled)
        self.add_where_clause(expr, compiled)

        compiled_fields = [self._compile_select_field(field)
                           for field in expr.fields]

        compiled = self._join_select_fields(compiled_fields)

        self._ctx.add_expr_compiled(expr, compiled)
        self.add_select_clause(expr, compiled)

    def visit_slice_collection(self, expr):
        sliced = expr._indexes
        if sliced[0] is not None:
            raise NotImplementedError
        if sliced[2] is not None:
            raise NotImplementedError
        if sliced[1].value <= 0:
            raise CompileError('limit number must be greater than 0')

        self.set_limit(expr, sliced[1].value)

    def visit_element_op(self, expr):
        if isinstance(expr, element.IsNull):
            compiled = '{0} IS NULL'.format(
                self._ctx.get_expr_compiled(expr.input))
        elif isinstance(expr, element.NotNull):
            compiled = '{0} IS NOT NULL'.format(
                self._ctx.get_expr_compiled(expr.input))
        elif isinstance(expr, element.FillNa):
            compiled = 'IF(%(input)s IS NULL, %(value)s, %(input)s)' % {
                'input': self._ctx.get_expr_compiled(expr.input),
                'value': self._ctx.get_expr_compiled(expr._fill_value),
            }
        elif isinstance(expr, element.IsIn):
            if expr.values is not None:
                compiled = '{0} IN ({1})'.format(
                    self._ctx.get_expr_compiled(expr.input),
                    ', '.join(self._ctx.get_expr_compiled(it) for it in expr.values)
                )
            else:
                subs = self._sub_compiles[expr]
                compiled = '{0} IN ({1})'.format(
                    subs[0], subs[1].replace('\n', '')
                )
        elif isinstance(expr, element.NotIn):
            if expr.values is not None:
                compiled = '{0} NOT IN ({1})'.format(
                    self._ctx.get_expr_compiled(expr.input),
                    ', '.join(self._ctx.get_expr_compiled(it) for it in expr.values)
                )
            else:
                subs = self._sub_compiles[expr]
                compiled = '{0} NOT IN ({1})'.format(
                    subs[0], subs[1].replace('\n', '')
                )
        elif isinstance(expr, element.IfElse):
            compiled = 'IF({0}, {1}, {2})'.format(
                self._ctx.get_expr_compiled(expr._input),
                self._ctx.get_expr_compiled(expr._then),
                self._ctx.get_expr_compiled(expr._else),
            )
        elif isinstance(expr, element.Switch):
            case = self._ctx.get_expr_compiled(expr.case) + ' ' \
                if expr.case is not None else ''
            lines = ['CASE {0}'.format(case)]
            for pair in zip(expr.conditions, expr.thens):
                args = [self._ctx.get_expr_compiled(p) for p in pair]
                lines.append('WHEN {0} THEN {1} '.format(*args))
            if expr.default is not None:
                lines.append('ELSE {0} '.format(self._ctx.get_expr_compiled(expr.default)))
            lines.append('END')
            if self._beautify:
                for i in range(1, len(lines) - 1):
                    lines[i] = utils.indent(lines[i], self._indent_size)
                compiled = '\n'.join(lines)
            else:
                compiled = ''.join(lines)
        elif isinstance(expr, element.IntToDatetime):
            compiled = 'FROM_UNIXTIME({0})'.format(
                self._ctx.get_expr_compiled(expr._input),
            )
        else:
            raise NotImplementedError

        self._ctx.add_expr_compiled(expr, compiled)

    def _parenthesis(self, child):
        if isinstance(child, BinOp):
            return '(%s)' % self._ctx.get_expr_compiled(child)
        elif isinstance(child, (element.IsNull, element.NotNull, element.IsIn, element.NotIn,
                                element.Between, element.Switch, element.Cut)):
            return '(%s)' % self._ctx.get_expr_compiled(child)
        else:
            return self._ctx.get_expr_compiled(child)

    def visit_binary_op(self, expr):
        if isinstance(expr, Add) and expr.dtype == df_types.string:
            compiled = 'CONCAT({0}, {1})'.format(
                self._ctx.get_expr_compiled(expr.lhs),
                self._ctx.get_expr_compiled(expr.rhs),
            )
        elif isinstance(expr, (Add, Substract)) and expr.dtype == df_types.datetime:
            if isinstance(expr, Add) and \
                    all(child.dtype == df_types.datetime for child in (expr.lhs, expr.rhs)):
                raise CompileError('Cannot add two datetimes')
            if isinstance(expr.rhs, DTScalar) or (isinstance(expr, Add) and expr.lhs, DTScalar):
                if isinstance(expr.rhs, DTScalar):
                    dt, scalar = expr.lhs, expr.rhs
                else:
                    dt, scalar = expr.rhs, expr.lhs

                class_name = type(scalar).__name__[:-6]
                date_part = DATE_PARTS_DIC[class_name]
                val = scalar.value
                if isinstance(expr, Substract):
                    val = -val
                compiled = 'DATEADD({0}, {1}, {2})'.format(
                    self._ctx.get_expr_compiled(dt), repr(val), repr(date_part)
                )
        else:
            compiled, op = None, None
            try:
                op = BINARY_OP_COMPILE_DIC[expr.node_name].upper()
            except KeyError:
                if isinstance(expr, Power):
                    compiled = 'POW({0}, {1})'.format(
                        self._ctx.get_expr_compiled(expr.lhs),
                        self._ctx.get_expr_compiled(expr.rhs)
                    )
                    if not isinstance(expr.dtype, df_types.Float):
                        compiled = self._cast(compiled, df_types.float64, expr.dtype)
                else:
                    raise NotImplementedError

            if compiled is None:
                lhs, rhs = expr.args
                if op:
                    compiled = '{0} {1} {2}'.format(
                        self._parenthesis(lhs), op, self._parenthesis(rhs)
                    )
                else:
                    raise NotImplementedError

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_unary_op(self, expr):
        try:
            if isinstance(expr, Negate) and expr.input.dtype == df_types.boolean:
                compiled = 'NOT {0}'.format(self._parenthesis(expr.input))
            else:
                op = UNARY_OP_COMPILE_DIC[expr.node_name]

                compiled = '{0}{1}'.format(
                        op, self._parenthesis(expr.input))
        except KeyError:
            if isinstance(expr, Abs):
                compiled = 'ABS({0})'.format(
                    self._ctx.get_expr_compiled(expr.input))
            elif isinstance(expr, (Invert, Negate)) and \
                    expr.input.dtype == df_types.boolean:
                compiled = 'NOT {0}'.format(self._parenthesis(expr.input))
            else:
                raise NotImplementedError

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_math(self, expr):
        compiled = None
        try:
            op = MATH_COMPILE_DIC[expr.node_name]
        except KeyError:
            if expr.node_name == 'Log':
                if expr._base is None:
                    op = 'ln'
                else:
                    compiled = 'LOG({0}, {1})'.format(
                        self._ctx.get_expr_compiled(expr._base),
                        self._ctx.get_expr_compiled(expr.input)
                    )
            elif expr.node_name == 'Log2':
                compiled = 'LOG(2, {0})'.format(
                    self._ctx.get_expr_compiled(expr.input)
                )
            elif expr.node_name == 'Log10':
                compiled = 'LOG(10, {0})'.format(
                    self._ctx.get_expr_compiled(expr.input)
                )
            elif expr.node_name == 'Log1p':
                compiled = 'LN(1 + {0})'.format(
                    self._ctx.get_expr_compiled(expr.input)
                )
            elif expr.node_name == 'Expm1':
                compiled = 'EXP({0}) - 1'.format(
                    self._ctx.get_expr_compiled(expr.input)
                )
            elif expr.node_name == 'Trunc':
                if expr._decimals is None:
                    op = 'TRUNC'
                else:
                    compiled = 'TRUNC({0}, {1})'.format(
                        self._ctx.get_expr_compiled(expr.input),
                        self._ctx.get_expr_compiled(expr._decimals)
                    )
            else:
                raise NotImplementedError

        if compiled is None:
            compiled = '{0}({1})'.format(
                op.upper(), self._ctx.get_expr_compiled(expr.input))

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_string_op(self, expr):
        # FIXME quite a few operations cannot support by internal function
        compiled = None

        input = self._ctx.get_expr_compiled(expr.input)
        if isinstance(expr, strings.Capitalize):
            compiled = 'CONCAT(TOUPPER(SUBSTR(%(input)s, 1, 1)), TOLOWER(SUBSTR(%(input)s, 2)))' % {
                'input': input
            }
        elif isinstance(expr, strings.CatStr):
            nodes = [expr._input]
            if expr._others is not None:
                others = (expr._others, ) if not isinstance(expr._others, Iterable) else expr._others
                for other in others:
                    if expr._sep is not None:
                        nodes.extend([expr._sep, other])
                    else:
                        nodes.append(other)
            compiled = 'CONCAT(%s)' % ', '.join(self._ctx.get_expr_compiled(e) for e in nodes)
        elif isinstance(expr, strings.Contains):
            if expr.regex:
                raise NotImplementedError
            compiled = 'INSTR(%s, %s) > 0' % (input, self._ctx.get_expr_compiled(expr._pat))
        elif isinstance(expr, strings.Endswith):
            # TODO: any better solution?
            compiled = 'INSTR(REVERSE(%s), REVERSE(%s)) == 1' % (
                input, self._ctx.get_expr_compiled(expr._pat))
        elif isinstance(expr, strings.Startswith):
            compiled = 'INSTR(%s, %s) == 1' % (input, self._ctx.get_expr_compiled(expr._pat))
        elif isinstance(expr, strings.Find):
            if isinstance(expr.start, six.integer_types):
                start = expr.start + 1 if expr.start >= 0 else expr.start
            else:
                start = 'IF(%(start)s >= 0, %(start)s + 1, %(start)s)' % {
                    'start': self._ctx.get_expr_compiled(expr._start)
                }
            if expr.end is not None:
                raise NotImplementedError
            else:
                compiled = 'INSTR(%s, %s, %s) - 1' % (
                    input, self._ctx.get_expr_compiled(expr._sub), start)
        elif isinstance(expr, strings.Get):
            compiled = 'SUBSTR(%s, %s, 1)' % (input, expr.index + 1)
        elif isinstance(expr, strings.Len):
            compiled = 'LENGTH(%s)' % input
        elif isinstance(expr, strings.Lower):
            compiled = 'TOLOWER(%s)' % input
        elif isinstance(expr, strings.Upper):
            compiled = 'TOUPPER(%s)' % input
        elif isinstance(expr, (strings.Lstrip, strings.Rstrip, strings.Strip)):
            if expr.to_strip != ' ':
                raise NotImplementedError
            func = {
                'Lstrip': 'LTRIM',
                'Rstrip': 'RTRIM',
                'Strip': 'TRIM'
            }
            compiled = '%s(%s)' % (func[type(expr).__name__], input)
        elif isinstance(expr, strings.Slice):
            # internal function will be compiled in two cases:
            # 1) start is not None
            # 2) positive start and end
            if expr.end is None and expr.step is None:
                compiled = 'SUBSTR(%s, %s)' % (input, expr.start + 1)
            else:
                # expr.start and expr.end
                length = expr.end - expr.start
                compiled = 'SUBSTR(%s, %s, %s)' % (input, expr.start + 1, length)
        elif isinstance(expr, strings.Repeat):
            compiled = 'REPEAT(%s, %s)' % (
                input, self._ctx.get_expr_compiled(expr._repeats))

        if compiled is not None:
            self._ctx.add_expr_compiled(expr, compiled)
        else:
            raise NotImplementedError

    def visit_datetime_op(self, expr):
        # FIXME quite a few operations cannot support by internal function
        class_name = type(expr).__name__
        input = self._ctx.get_expr_compiled(expr.input)

        compiled = None
        if class_name in DATE_PARTS_DIC:
            compiled = 'DATEPART(%s, %r)' % (input, DATE_PARTS_DIC[class_name])
        elif isinstance(expr, datetimes.WeekOfYear):
            compiled = 'WEEKOFYEAR(%s)' % input
        elif isinstance(expr, datetimes.WeekDay):
            compiled = 'WEEKDAY(%s)' % input
        elif isinstance(expr, datetimes.Date):
            compiled = 'DATETRUNC(%s, %r)' % (input, 'dd')
        elif isinstance(expr, datetimes.UnixTimestamp):
            compiled = 'UNIX_TIMESTAMP(%s)' % input

        if compiled is not None:
            self._ctx.add_expr_compiled(expr, compiled)
        else:
            raise NotImplementedError

    def visit_groupby(self, expr):
        bys, having, aggs, fields = tuple(expr.args[1:])
        if fields is None:
            fields = bys + aggs

        by_fields = [self._ctx.get_expr_compiled(by) for by in bys]
        group_by_clause = 'GROUP BY {0} '.format(self._join_compiled_fields(by_fields))

        select_fields = [self._compile_select_field(field) for field in fields]
        select_clause = self._join_select_fields(select_fields)

        self.add_select_clause(expr, select_clause)
        self.add_group_by_clause(expr, group_by_clause)

        if having:
            self.add_having_clause(expr, self._ctx.get_expr_compiled(having))

    def visit_mutate(self, expr):
        bys, mutates, fields = tuple(expr.args[1:])
        if fields is None:
            fields = bys + mutates

        select_fields = [self._compile_select_field(field) for field in fields]
        select_clause = self._join_select_fields(select_fields)

        self.add_select_clause(expr, select_clause)

    def visit_sort_column(self, expr):
        def get_field(field):
            if isinstance(field.input, CollectionExpr):
                return field._source_name
            elif isinstance(field.input, Column):
                return field.input.source_name
            else:
                return self._ctx.get_expr_compiled(field.input)
        compiled = '{0} DESC'.format(get_field(expr)) \
            if not expr._ascending else get_field(expr)
        self._ctx.add_expr_compiled(expr, compiled)

    def visit_sort(self, expr):
        keys_fields = expr.args[1]

        order_by_clause = 'ORDER BY {0} '.format(self._join_compiled_fields(
            [self._ctx.get_expr_compiled(field) for field in keys_fields]))

        self.add_order_by_clause(expr, order_by_clause)

    def visit_distinct(self, expr):
        distinct_fields = expr.args[1]

        fields_clause = self._join_select_fields(
            [self._compile_select_field(field) for field in distinct_fields])
        select_clause = 'DISTINCT {0}'.format(fields_clause)

        self.add_select_clause(expr, select_clause)

    def visit_reduction(self, expr):
        if isinstance(expr, (Count, GroupedCount)) and isinstance(expr.input, CollectionExpr):
            compiled = 'COUNT(1)'
            self._ctx.add_expr_compiled(expr, compiled)
            return

        if isinstance(expr, (Std, GroupedStd)):
            if expr._ddof not in (0, 1):
                raise CompileError('Does not support %s with ddof=%s' % (
                    expr.node_name, expr._ddof))

        compiled = None

        if isinstance(expr, (Mean, GroupedMean)):
            node_name = 'avg'
        elif isinstance(expr, (Std, GroupedStd)):
            node_name = 'stddev' if expr._ddof == 0 else 'stddev_samp'
        elif isinstance(expr, (Sum, GroupedSum)) and expr.input.dtype == df_types.string:
            compiled = 'WM_CONCAT(\'\', %s)' % self._ctx.get_expr_compiled(expr.input)
        elif isinstance(expr, (Sum, GroupedSum)) and expr.input.dtype == df_types.boolean:
            compiled = 'SUM(IF(%s, 1, 0))' % self._ctx.get_expr_compiled(expr.input)
        elif isinstance(expr, (Max, GroupedMax, Min, GroupedMin)) and \
                expr.input.dtype == df_types.boolean:
            compiled = '%s(IF(%s, 1, 0)) == 1' % (
                expr.node_name, self._ctx.get_expr_compiled(expr.input))
        elif isinstance(expr, (Any, GroupedAny)):
            compiled = 'MAX(IF(%s, 1, 0)) == 1' % self._ctx.get_expr_compiled(expr.args[0])
        elif isinstance(expr, (All, GroupedAll)):
            compiled = 'MIN(IF(%s, 1, 0)) == 1' % self._ctx.get_expr_compiled(expr.args[0])
        elif isinstance(expr, (NUnique, GroupedNUnique)):
            compiled = 'COUNT(DISTINCT %s)' % self._ctx.get_expr_compiled(expr.args[0])
        elif isinstance(expr, (Cat, GroupedCat)):
            compiled = 'WM_CONCAT(%s, %s)' % (self._ctx.get_expr_compiled(expr._sep),
                                              self._ctx.get_expr_compiled(expr.input))
        else:
            node_name = expr.node_name

        if compiled is None:
            compiled = '{0}({1})'.format(
                node_name.upper(), self._ctx.get_expr_compiled(expr.args[0]))

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_user_defined_aggregator(self, expr):
        is_func_created = False
        if isinstance(expr._aggregator, six.string_types):
            func_name = expr._aggregator
        elif isinstance(expr._aggregator, Function):
            func_name = expr._aggregator.name
        else:
            func_name = self._ctx.get_udf(expr._aggregator)
            is_func_created = True

        args = [self._ctx.get_expr_compiled(i) for i in expr.inputs]
        if hasattr(expr, '_func_args') and expr._func_args is not None \
                and not is_func_created:
            func_args = [repr(arg) for arg in expr._func_args]
            args.extend(func_args)
        compiled = '{0}({1})'.format(func_name, ', '.join(args))

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_column(self, expr):
        alias = self._ctx.get_collection_alias(expr.input, silent=True)

        if alias:
            alias = alias[0]
            compiled = '{0}.{1}'.format(alias, self._quote(expr.source_name))
        else:
            symbol = self._ctx.add_need_alias_column(expr)
            compiled = '%({0})s'.format(symbol)

        if expr._source_data_type != expr._data_type:
            compiled = 'CAST({0} AS {1})'.format(
                compiled, types.df_type_to_odps_type(expr.dtype))

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_function(self, expr):
        is_func_created = False
        if isinstance(expr, Func):
            func_name = expr._func_name
        else:
            if isinstance(expr._func, six.string_types):
                func_name = expr._func
            elif isinstance(expr._func, Function):
                func_name = expr._func.name
            else:
                func_name = self._ctx.get_udf(expr._func)
                is_func_created = True

        if isinstance(expr, (MappedExpr, Func)):
            args = [self._ctx.get_expr_compiled(f) for f in expr.inputs]
        else:
            raise NotImplementedError
        if hasattr(expr, '_func_args') and expr._func_args is not None \
                and not is_func_created:
            func_args = [repr(arg) for arg in expr._func_args]
            args.extend(func_args)
        compiled = '{0}({1})'.format(func_name, ', '.join(args))

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_builtin_function(self, expr):
        compiled = '{0}({1})'.format(expr._func_name,
                                     ', '.join(repr(arg) for arg in expr._func_args))
        self._ctx.add_expr_compiled(expr, compiled)

    def _quote(self, compiled):
        if options.df.quote:
            return '`{0}`'.format(compiled)
        else:
            return compiled

    def _unquote(self, compiled):
        if options.df.quote:
            reg = re.compile('`([^`]+)`')
        else:
            reg = re.compile('\.(\w+)')
        matched = reg.search(compiled)
        if matched:
            return matched.group(1)
        return compiled

    def visit_reshuffle(self, expr):
        bys, sorts = expr._by, expr._sort_fields

        by_fields = [self._unquote(self._ctx.get_expr_compiled(by)) for by in bys]
        distribute_by_clause = 'DISTRIBUTE BY {0} '.format(
            self._join_compiled_fields(by_fields))

        self.add_group_by_clause(expr, distribute_by_clause)

        if sorts:
            sort_fields = [self._ctx.get_expr_compiled(sort) for sort in sorts]
            sort_by_clause = 'SORT BY {0}'.format(self._join_compiled_fields(sort_fields))

            self.add_order_by_clause(expr, sort_by_clause)

    def visit_apply_collection(self, expr):
        is_func_created = False
        if isinstance(expr._func, six.string_types):
            func_name = expr._func
        elif isinstance(expr._func, Function):
            func_name = expr._func.name
        else:
            func_name = self._ctx.get_udf(expr._func)
            is_func_created = True

        args = [self._ctx.get_expr_compiled(f) for f in expr._fields]
        if hasattr(expr, '_func_args') and expr._func_args is not None \
                and not is_func_created:
            func_args = [repr(arg) for arg in expr._func_args]
            args.extend(func_args)
        compiled = '{0}({1}) AS ({2})'.format(
            func_name, ', '.join(args),
            ', '.join(self._quote(n) for n in expr.schema.names))

        self._ctx.add_expr_compiled(expr, compiled)
        self.add_select_clause(expr, compiled)

    def _wrap_typed(self, expr, compiled):
        if expr._source_data_type != expr._data_type:
            compiled = 'cast({0} AS {1})'.format(
                compiled, types.df_type_to_odps_type(expr._data_type))

        return compiled

    def visit_sequence(self, expr):
        compiled = expr._source_name
        compiled = self._wrap_typed(expr, compiled)

        self._ctx.add_expr_compiled(expr, compiled)

    def _compile_window_order_by(self, expr):
        if isinstance(expr.input, SequenceExpr):
            compiled = self._ctx.get_expr_compiled(expr.input)
            return '%s DESC' % compiled if not expr._ascending else compiled
        else:
            return self._ctx.get_expr_compiled(expr)

    def _compile_window_function(self, func, args, partition_by=None,
                                 order_by=None, preceding=None, following=None):
        partition_by = 'PARTITION BY {0}'.format(partition_by or '1')
        order_by = 'ORDER BY {0}'.format(order_by) if order_by is not None else ''

        if isinstance(preceding, tuple):
            window_clause = 'ROWS BETWEEN {0} PRECEDING AND {1} PRECEDING' \
                .format(*preceding)
        elif isinstance(following, tuple):
            window_clause = 'ROWS BETWEEN {0} FOLLOWING AND {1} FOLLOWING' \
                .format(*following)
        elif preceding is not None and following is not None:
            window_clause = 'ROWS BETWEEN {0} PRECEDING AND {1} FOLLOWING' \
                .format(preceding, following)
        elif preceding is not None:
            window_clause = 'ROWS {0} PRECEDING'.format(preceding)
        elif following is not None:
            window_clause = 'ROWS {0} FOLLOWING'.format(following)
        else:
            window_clause = ''

        over = ' '.join(sub for sub in (partition_by, order_by, window_clause)
                        if len(sub) > 0)

        return '{0}({1}) OVER ({2})'.format(func, args, over)

    def visit_cum_window(self, expr):
        col_compiled = self._ctx.get_expr_compiled(expr.input)
        if isinstance(expr, CumSum) and expr.input.dtype == df_types.boolean:
            col_compiled = 'IF({0}, 1, 0)'.format(col_compiled)
        if expr.distinct:
            col_compiled = 'DISTINCT {0}'.format(col_compiled)

        partition_by = ', '.join(self._ctx.get_expr_compiled(by)
                                 for by in expr._partition_by) if expr._partition_by else None
        order_by = ', '.join(self._compile_window_order_by(by)
                             for by in expr._order_by) if expr._order_by else None

        func_name = WINDOW_COMPILE_DIC[expr.node_name].upper()
        compiled = self._compile_window_function(func_name, col_compiled, partition_by=partition_by,
                                                 order_by=order_by, preceding=expr._preceding,
                                                 following=expr._following)

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_rank_window(self, expr):
        func_name = WINDOW_COMPILE_DIC[expr.node_name].upper()

        partition_by = ', '.join(self._ctx.get_expr_compiled(by)
                                 for by in expr._partition_by) if expr._partition_by else None
        order_by = ', '.join(self._compile_window_order_by(by)
                             for by in expr._order_by) if expr._order_by else None

        compiled = self._compile_window_function(func_name, '', partition_by=partition_by,
                                                 order_by=order_by)

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_shift_window(self, expr):
        func_name = WINDOW_COMPILE_DIC[expr.node_name].upper()

        compiled_fields = [self._ctx.get_expr_compiled(expr.input), ]
        if expr._offset:
            compiled_fields.append(self._ctx.get_expr_compiled(expr._offset))
        if expr._default:
            compiled_fields.append(self._ctx.get_expr_compiled(expr._default))

        col_compiled = self._join_compiled_fields(compiled_fields)

        partition_by = ', '.join(self._ctx.get_expr_compiled(by)
                                 for by in expr._partition_by) if expr._partition_by else None
        order_by = ', '.join(self._compile_window_order_by(by)
                             for by in expr._order_by) if expr._order_by else None

        compiled = self._compile_window_function(func_name, col_compiled, partition_by=partition_by,
                                                 order_by=order_by)

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_scalar(self, expr):
        compiled = None
        if expr._value is not None:
            if expr.dtype == df_types.string:
                val = utils.to_str(expr.value) \
                    if isinstance(expr.value, six.text_type) else expr.value
                compiled = "'{0}'".format(val.replace("'", "\\'"))
            elif isinstance(expr._value, bool):
                compiled = 'true' if expr._value else 'false'
            elif isinstance(expr._value, datetime):
                # FIXME: just ignore shorter than second
                compiled = 'FROM_UNIXTIME({0})'.format(utils.to_timestamp(expr._value))
            elif isinstance(expr._value, Decimal):
                compiled = 'CAST({0} AS DECIMAL)'.format(repr(str(expr._value)))
        else:
            compiled = 'CAST(NULL AS {0})'.format(types.df_type_to_odps_type(expr._value_type))

        if compiled is None:
            compiled = repr(expr._value)
        self._ctx.add_expr_compiled(expr, compiled)

    @classmethod
    def _cast(cls, compiled, source_type, to_type):
        source_odps_type = types.df_type_to_odps_type(source_type)
        to_type = types.df_type_to_odps_type(to_type)

        if not to_type.can_explicit_cast(source_odps_type):
            raise CompileError(
                    'Cannot cast from %s to %s' % (source_odps_type, to_type))

        return 'CAST({0} AS {1})'.format(compiled, to_type)

    def visit_cast(self, expr):
        compiled = self._ctx.get_expr_compiled(expr._input)

        if isinstance(expr.source_type, df_types.Integer) and expr.dtype == df_types.datetime:
            compiled = 'FROM_UNIXTIME({0})'.format(self._ctx.get_expr_compiled(expr.input))
        elif expr.dtype != expr.source_type:
            compiled = self._cast(compiled, expr.source_type, expr.dtype)

        self._ctx.add_expr_compiled(expr, compiled)

    def visit_join(self, expr):
        left_compiled, right_compiled, predicate_compiled = tuple(self._sub_compiles[expr])

        from_clause = '{0} \n{1} JOIN \n{2}'.format(
            left_compiled, expr._how, utils.indent(right_compiled, self._indent_size)
        )
        if predicate_compiled:
            from_clause += '\nON {0}'.format(predicate_compiled)

        self.add_from_clause(expr, from_clause)
        self._ctx.add_expr_compiled(expr, from_clause)

    def visit_union(self, expr):
        if expr._distinct:
            raise CompileError("Distinct union is not supported here.")

        left_compiled, right_compiled = tuple(self._sub_compiles[expr])

        from_clause = '{0} \nUNION ALL\n{1}'.format(left_compiled, utils.indent(right_compiled, self._indent_size))

        compiled = '(\n{0}\n) {1}'.format(utils.indent(from_clause, self._indent_size),
                                          self._ctx.get_collection_alias(expr, create=True)[0])

        self.add_from_clause(expr, compiled)
        self._ctx.add_expr_compiled(expr, compiled)