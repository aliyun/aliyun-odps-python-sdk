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
from datetime import timedelta

from ..core import Backend
from ..errors import CompileError
from ...expr.reduction import *
from ...expr.arithmetic import *
from ...expr.datetimes import *
from ...expr.window import *
from ...expr.merge import *
from ...expr.utils import highest_precedence_data_type
from ... import types as df_types
from ...utils import is_constant_scalar, traverse_until_source, is_source_collection
from ....compat import lzip
from .... import utils
from . import types

try:
    from sqlalchemy import Table as SATable, select, func, case, \
        extract, desc, distinct, literal, join, union, union_all
    from sqlalchemy.sql.expression import Alias, text

    # define the compile ext
    from .ext import *

    has_sqlalchemy = True
except ImportError:
    has_sqlalchemy = False


BINARY_OP = {
    'Add': operator.add,
    'Substract': operator.sub,
    'Multiply': operator.mul,
    'Divide': operator.div if six.PY2 else operator.truediv,
    'Mod': operator.mod,
    'FloorDivide': operator.floordiv,
    'Greater': operator.gt,
    'GreaterEqual': operator.ge,
    'Less': operator.lt,
    'LessEqual': operator.le,
    'Equal': operator.eq,
    'NotEqual': operator.ne,
    'And': operator.and_,
    'Or': operator.or_
}

UNARY_OP = {
    'Negate': operator.neg,
    'Invert': operator.inv,
}

DATE_KEY_DIC = {
    'Day': 'days',
    'Hour': 'hours',
    'Minute': 'minutes',
    'Second': 'seconds',
    'MilliSecond': 'milliseconds',
    'MicroSecond': 'microseconds',
}

MATH_COMPILE_DIC = {
    'Abs': 'abs',
    'Sqrt': 'sqrt',
    'Sin': 'sin',
    'Cos': 'cos',
    'Tan': 'tan',
    'Exp': 'exp',
    'Arccos': 'acos',
    'Arcsin': 'asin',
    'Arctan': 'atan',
    'Ceil': 'ceil',
    'Floor': 'floor'
}

WINDOW_COMPILE_DIC = {
    'CumSum': 'sum',
    'CumMean': 'avg',
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

DATE_PARTS_DIC = {
    'Year': 'year',
    'Month': 'month',
    'Day': 'day',
    'Hour': 'hour',
    'Minute': 'minute',
    'Second': 'second',
    'WeekOfYear': 'week',
    'DayOfYear': 'doy',
    'MicroSecond': 'microseconds',
    'UnixTimestamp': 'epoch',
}


class SQLAlchemyCompiler(Backend):
    def __init__(self, expr_dag):
        self._expr_dag = expr_dag
        self._expr_to_sqlalchemy = ExprDictionary()
        self._id_gen = itertools.count(1)

        self._sa_engine = None

    def _new_alias(self):
        return 't%s' % next(self._id_gen)

    def compile(self, expr, traversed=None):
        if traversed is None:
            traversed = set()

        for node in traverse_until_source(expr):
            if id(node) not in traversed:
                node.accept(self)
                traversed.add(id(node))

        sa_expr = self._expr_to_sqlalchemy[self._expr_dag.root]
        if is_source_collection(expr):
            sa_expr = select([sa_expr])
        return sa_expr

    def _add(self, expr, op):
        self._expr_to_sqlalchemy[expr] = op

    def _gen_select_columns(self, fields):
        sa_exprs = []
        for field in fields:
            if not isinstance(field, Scalar) or field._value is None:
                sa_exprs.append(self._expr_to_sqlalchemy[field].label(field.name))
            else:
                sa_exprs.append(
                    literal(self._expr_to_sqlalchemy[field]).label(field.name))
        return sa_exprs

    def visit_source_collection(self, expr):
        table = next(expr.data_source())

        if not isinstance(table, SATable):
            raise ValueError('Source data must be a sqlalchemy table')

        if table.bind and self._sa_engine is None:
            self._sa_engine = table.bind

        self._add(expr, table.alias(self._new_alias()))

    def visit_project_collection(self, expr):
        selects = select(self._gen_select_columns(expr._fields))\
            .select_from(self._expr_to_sqlalchemy[expr.input])

        if expr is not self._expr_dag.root:
            selects = selects.alias(self._new_alias())
        self._add(expr, selects)

    def visit_apply_collection(self, expr):
        raise NotImplementedError

    def visit_filter_collection(self, expr):
        input = self._expr_to_sqlalchemy[expr.input]
        predicate = self._expr_to_sqlalchemy[expr._predicate]
        filtered = input.select(predicate)

        if expr is not self._expr_dag.root:
            filtered = filtered.alias(self._new_alias())
        self._add(expr, filtered)

    def visit_slice_collection(self, expr):
        input = self._expr_to_sqlalchemy[expr.input]
        sliced = expr._indexes
        if sliced[2] is not None:
            raise NotImplementedError
        if sliced[0] is not None and sliced[0].value < 0:
            raise CompileError('start number must be greater than 0')
        if sliced[1] is not None and sliced[1].value <= 0:
            raise CompileError('end number must be greater than 0')

        kw = dict()
        if sliced[0] is not None and sliced[0].value > 0:
            kw['offset'] = sliced[0].value
        if sliced[1] is not None and sliced[1].value > 0:
            kw['limit'] = sliced[1].value
        input = input.select(**kw)

        if expr is not self._expr_dag.root:
            input = input.alias(self._new_alias())

        self._add(expr, input)

    def visit_element_op(self, expr):
        input = self._expr_to_sqlalchemy.get(expr.input)
        if isinstance(expr, element.IsNull):
            sa_expr = input.is_(None)
        elif isinstance(expr, element.NotNull):
            sa_expr = input.isnot(None)
        elif isinstance(expr, element.FillNa):
            sa_expr = case([(input.is_(None), expr.fill_value)], else_=input)
        elif isinstance(expr, (element.IsIn, element.NotIn)):
            op = input.in_ if isinstance(expr, element.IsIn) else input.notin_
            if expr._values is None:
                sa_expr = op([None])
            elif len(expr._values) == 1 and isinstance(expr._values[0], SequenceExpr):
                right = select([self._expr_to_sqlalchemy[expr._values[0]]])
                sa_expr = op(right)
            else:
                sa_expr = op(tuple(self._expr_to_sqlalchemy[it] for it in expr._values))
        elif isinstance(expr, element.Between):
            if not expr.inclusive:
                raise NotImplementedError
            sa_expr = input.between(
                self._expr_to_sqlalchemy[expr._left],
                self._expr_to_sqlalchemy[expr._right]
            )
        elif isinstance(expr, element.IfElse):
            sa_expr = case([(input, self._expr_to_sqlalchemy[expr._then])],
                           else_=self._expr_to_sqlalchemy[expr._else])
        elif isinstance(expr, element.Switch):
            conditions = [self._expr_to_sqlalchemy[cond] for cond in expr._conditions]
            thens = [self._expr_to_sqlalchemy[then] for then in expr._thens]
            sa_else = self._expr_to_sqlalchemy[expr._default] \
                if expr._default is not None else expr._default
            if expr._input is None:
                sa_expr = case(lzip(conditions, thens), else_=sa_else)
            else:
                sa_expr = case(dict(lzip(conditions, thens)),
                               value=input, else_=sa_else)
        else:
            raise NotImplementedError

        self._add(expr, sa_expr)

    def visit_binary_op(self, expr):
        if isinstance(expr, Power):
            op = func.pow
        elif isinstance(expr, FloorDivide):
            op = operator.div if six.PY2 else operator.truediv
        elif isinstance(expr, (Add, Substract)) and expr.dtype == df_types.datetime:
            if isinstance(expr, Add) and \
                    all(child.dtype == df_types.datetime for child in (expr.lhs, expr.rhs)):
                raise CompileError('Cannot add two datetimes')
            if isinstance(expr.rhs, DTScalar) or (isinstance(expr, Add) and expr.lhs, DTScalar):
                if isinstance(expr.rhs, DTScalar):
                    dt, scalar = expr.lhs, expr.rhs
                else:
                    dt, scalar = expr.rhs, expr.lhs
                val = scalar.value
                if isinstance(expr, Substract):
                    val = -val

                dt_type = type(scalar).__name__[:-6]
                sa_dt = self._expr_to_sqlalchemy[dt]
                try:
                    key = DATE_KEY_DIC[dt_type]
                except KeyError:
                    raise NotImplementedError
                if self._sa_engine and self._sa_engine.name == 'mysql':
                    if dt_type == 'MilliSecond':
                        val, dt_type = val * 1000, 'MicroSecond'
                    sa_expr = func.date_add(sa_dt, text('interval %d %s' % (val, dt_type.lower())))
                else:
                    sa_expr = sa_dt + timedelta(**{key: val})
                self._add(expr, sa_expr)
                return
            else:
                raise NotImplementedError
        elif isinstance(expr, Substract) and expr._lhs.dtype == df_types.datetime and \
                expr._rhs.dtype == df_types.datetime:
            sa_expr = self._expr_to_sqlalchemy[expr._lhs] - self._expr_to_sqlalchemy[expr._rhs]
            if self._sa_engine and self._sa_engine.name == 'mysql':
                sa_expr = func.abs(func.microsecond(sa_expr)
                                   .cast(types.df_type_to_sqlalchemy_type(expr.dtype))) / 1000
            else:
                sa_expr = func.abs(extract('MICROSECONDS', sa_expr)
                                   .cast(types.df_type_to_sqlalchemy_type(expr.dtype))) / 1000
            self._add(expr, sa_expr)
            return
        elif isinstance(expr, Mod):
            lhs, rhs = self._expr_to_sqlalchemy[expr._lhs], self._expr_to_sqlalchemy[expr._rhs]
            sa_expr = BINARY_OP[expr.node_name](lhs, rhs)
            if not is_constant_scalar(expr._rhs):
                sa_expr = case([(rhs > 0, func.abs(sa_expr))], else_=sa_expr)
            elif expr._rhs.value > 0:
                sa_expr = func.abs(sa_expr)
            self._add(expr, sa_expr)
            return
        else:
            op = BINARY_OP[expr.node_name]
        lhs, rhs = self._expr_to_sqlalchemy[expr._lhs], self._expr_to_sqlalchemy[expr._rhs]
        sa_expr = op(lhs, rhs)
        self._add(expr, sa_expr)

    def visit_unary_op(self, expr):
        if isinstance(expr, Abs):
            op = func.abs
        else:
            op = UNARY_OP[expr.node_name]
        self._add(expr, op(self._expr_to_sqlalchemy[expr._input]))

    def visit_math(self, expr):
        try:
            op = getattr(func, MATH_COMPILE_DIC[expr.node_name])
            sa_expr = op(self._expr_to_sqlalchemy[expr._input])
        except KeyError:
            if expr.node_name == 'Log':
                if expr._base is not None:
                    sa_expr = SALog('log', self._expr_to_sqlalchemy[expr._base],
                                    self._expr_to_sqlalchemy[expr._base],
                                    self._expr_to_sqlalchemy[expr._input])
                else:
                    sa_expr = SALog('log', None, self._expr_to_sqlalchemy[expr._input])
            elif expr.node_name == 'Log2':
                sa_expr = SALog('log', 2, 2, self._expr_to_sqlalchemy[expr._input])
                sa_expr = sa_expr.cast(types.df_type_to_sqlalchemy_type(expr.dtype))
            elif expr.node_name == 'Log10':
                sa_expr = SALog('log', 10, 10, self._expr_to_sqlalchemy[expr._input])
                sa_expr = sa_expr.cast(types.df_type_to_sqlalchemy_type(expr.dtype))
            elif expr.node_name == 'Trunc':
                input = self._expr_to_sqlalchemy[expr._input]
                decimals = 0 if expr._decimals is None else self._expr_to_sqlalchemy[expr._decimals]
                sa_expr = SATruncate('trunc', input, decimals)
            else:
                raise NotImplementedError

        self._add(expr, sa_expr)

    def visit_string_op(self, expr):
        if isinstance(expr, strings.Capitalize):
            input = self._expr_to_sqlalchemy[expr._input]
            tp = types.df_type_to_sqlalchemy_type(expr.dtype)
            sa_expr = func.upper(func.substr(input, 1, 1)).cast(tp) + \
                      func.lower(func.substr(input, 2)).cast(tp)
        elif isinstance(expr, strings.Contains) and not expr.regex:
            sa_expr = self._expr_to_sqlalchemy[expr._input].contains(
                self._expr_to_sqlalchemy[expr._pat])
        elif isinstance(expr, strings.Endswith):
            sa_expr = self._expr_to_sqlalchemy[expr._input].endswith(
                self._expr_to_sqlalchemy[expr._pat])
        elif isinstance(expr, strings.Startswith):
            sa_expr = self._expr_to_sqlalchemy[expr._input].startswith(
                self._expr_to_sqlalchemy[expr._pat])
        elif isinstance(expr, strings.Replace) and not expr.regex:
            sa_expr = func.replace(self._expr_to_sqlalchemy[expr._input],
                                   self._expr_to_sqlalchemy[expr._pat],
                                   self._expr_to_sqlalchemy[expr._repl])
        elif isinstance(expr, strings.Get):
            sa_expr = func.substr(self._expr_to_sqlalchemy[expr._input],
                                  self._expr_to_sqlalchemy[expr._index] + 1, 1)
        elif isinstance(expr, strings.Len):
            sa_expr = func.length(self._expr_to_sqlalchemy[expr._input])
        elif isinstance(expr, (strings.Ljust, strings.Rjust, strings.Pad)):
            if isinstance(expr, strings.Pad):
                if expr.side == 'both':
                    raise NotImplementedError
                op = func.lpad if expr.side == 'left' else func.rpad
            else:
                op = func.lpad if isinstance(expr, strings.Ljust) else func.rpad
            sa_expr = op(self._expr_to_sqlalchemy[expr._input],
                         self._expr_to_sqlalchemy[expr._width],
                         self._expr_to_sqlalchemy[expr._fillchar])
        elif isinstance(expr, (strings.Lower, strings.Upper)):
            op = func.lower if isinstance(expr, strings.Lower) else func.upper
            sa_expr = op(self._expr_to_sqlalchemy[expr._input])
        elif isinstance(expr, (strings.Lstrip, strings.Rstrip, strings.Strip)):
            if expr._to_strip is None:
                raise NotImplementedError
            op = func.ltrim if isinstance(expr, strings.Lstrip) else (
                func.rtrim if isinstance(expr, strings.Rstrip) else func.btrim
            )
            sa_expr = op(self._expr_to_sqlalchemy[expr._input],
                         self._expr_to_sqlalchemy[expr._to_strip])
        elif isinstance(expr, strings.Repeat):
            sa_expr = func.repeat(self._expr_to_sqlalchemy[expr._input],
                                  self._expr_to_sqlalchemy[expr._repeats])
        elif isinstance(expr, strings.Slice):
            if expr.end is None and expr.step is None:
                sa_expr = func.substr(self._expr_to_sqlalchemy[expr._input],
                                      self._expr_to_sqlalchemy[expr._start] + 1)
            elif isinstance(expr.start, six.integer_types) and \
                    isinstance(expr.end, six.integer_types) and \
                    expr.step is None and expr.start > 0 and expr.end > 0:
                length = expr.end - expr.start
                sa_expr = func.substr(self._expr_to_sqlalchemy[expr._input],
                                      expr.start + 1, length)
            else:
                raise NotImplementedError
        elif isinstance(expr, strings.Title):
            sa_expr = func.initcap(self._expr_to_sqlalchemy[expr._input])
        else:
            raise NotImplementedError

        self._add(expr, sa_expr)

    def visit_datetime_op(self, expr):
        class_name = type(expr).__name__
        input = self._expr_to_sqlalchemy[expr._input]

        if class_name in DATE_PARTS_DIC:
            if self._sa_engine and self._sa_engine.name == 'mysql':
                if class_name == 'UnixTimestamp':
                    fun = func.unix_timestamp
                else:
                    fun = getattr(func, class_name.lower())
                sa_expr = fun(input).cast(types.df_type_to_sqlalchemy_type(expr.dtype))
            else:
                sa_expr = func.date_part(DATE_PARTS_DIC[class_name], input)\
                    .cast(types.df_type_to_sqlalchemy_type(expr.dtype))
        elif isinstance(expr, Date):
            if self._sa_engine and self._sa_engine.name == 'mysql':
                sa_expr = func.date(input).cast(types.df_type_to_sqlalchemy_type(expr.dtype))
            else:
                sa_expr = func.date_trunc('day', input)
        elif isinstance(expr, WeekDay):
            if self._sa_engine and self._sa_engine.name == 'mysql':
                sa_expr = (func.dayofweek(input).cast(types.df_type_to_sqlalchemy_type(expr.dtype)) + 5) % 7
            else:
                sa_expr = (func.date_part('dow', input).cast(types.df_type_to_sqlalchemy_type(expr.dtype)) + 6) % 7
        else:
            raise NotImplementedError

        self._add(expr, sa_expr)

    def visit_groupby(self, expr):
        bys, having, aggs, fields = tuple(expr.args[1:])
        if fields is None:
            fields = bys + aggs

        selects = select(self._gen_select_columns(fields))
        if len(fields) == 1 and isinstance(fields[0], (Count, GroupedCount)):
            selects = selects.select_from(self._expr_to_sqlalchemy[fields[0].input])
        grouped = selects.group_by(*self._gen_select_columns(bys))
        if having:
            grouped = grouped.having(self._expr_to_sqlalchemy[having])

        if expr is not self._expr_dag.root:
            grouped = grouped.alias(self._new_alias())

        self._add(expr, grouped)

    def visit_mutate(self, expr):
        bys, mutates, fields = tuple(expr.args[1:])
        if fields is None:
            fields = bys + mutates

        selects = select(self._gen_select_columns(fields))
        if expr is not self._expr_dag.root:
            selects = selects.alias(self._new_alias())

        self._add(expr, selects)

    def visit_sort_column(self, expr):
        if isinstance(expr.input, CollectionExpr):
            sa_expr = self._expr_to_sqlalchemy[expr.input].c[expr.source_name]
        else:
            sa_expr = self._expr_to_sqlalchemy[expr.input]
        if not expr._ascending:
            sa_expr = desc(sa_expr)

        self._add(expr, sa_expr)

    def visit_sort(self, expr):
        input = self._expr_to_sqlalchemy[expr.input]
        sa_expr = input.select(order_by=[self._expr_to_sqlalchemy[e]
                                         for e in expr._sorted_fields])
        if expr is not self._expr_dag.root:
            sa_expr = sa_expr.alias(self._new_alias())
        self._add(expr, sa_expr)

    def visit_distinct(self, expr):
        sa_expr = select(self._gen_select_columns(expr._unique_fields), distinct=True)

        if expr is not self._expr_dag.root:
            sa_expr = sa_expr.alias(self._new_alias())
        self._add(expr, sa_expr)

    def visit_column(self, expr):
        table = self._expr_to_sqlalchemy[expr.input]
        col = table.c[expr.source_name]

        if expr._source_data_type != expr._data_type:
            col = col.cast(types.df_type_to_sqlalchemy_type(expr._data_type))

        self._add(expr, col)

    def visit_reduction(self, expr):
        input = self._expr_to_sqlalchemy[expr.input]

        # TODO: MEDIAN does not support
        if isinstance(expr, (Max, GroupedMax)):
            f = func.max
        elif isinstance(expr, (Min, GroupedMin)):
            f = func.min
        elif isinstance(expr, (Count, GroupedCount)):
            f = func.count
        elif isinstance(expr, (Sum, GroupedSum)):
            f = func.sum
        elif isinstance(expr, (Var, GroupedVar)) and expr._ddof in (0, 1):
            f = func.var_pop if expr._ddof == 0 else func.var_samp
        elif isinstance(expr, (Std, GroupedStd)) and expr._ddof in (0, 1):
            f = func.stddev_pop if expr._ddof == 0 else func.stddev_samp
        elif isinstance(expr, (Mean, GroupedMean)):
            f = func.avg
        elif isinstance(expr, (NUnique, GroupedNUnique)):
            f = lambda x: func.count(distinct(x))
        elif isinstance(expr, (Cat, GroupedCat)):
            f = lambda x: func.array_to_string(func.array_agg(x),
                                               self._expr_to_sqlalchemy[expr._sep])
        else:
            raise NotImplementedError

        if isinstance(expr, (Count, GroupedCount)) and \
                isinstance(expr.input, CollectionExpr):
            reduced = f()
        else:
            reduced = f(input)
        self._add(expr, reduced)

    def visit_cum_window(self, expr):
        input = self._expr_to_sqlalchemy[expr._input]
        if expr._distinct.value is True:
            raise NotImplementedError
        try:
            func_name = WINDOW_COMPILE_DIC[expr.node_name]
        except KeyError:
            raise NotImplementedError
        f = getattr(func, func_name)
        partition_by = self._gen_select_columns(expr._partition_by) \
            if expr._partition_by else None
        order_by = self._gen_select_columns(expr._order_by) \
            if expr._order_by else None
        rows = (self._expr_to_sqlalchemy[expr._preceding] if expr._preceding else None,
                self._expr_to_sqlalchemy[expr._following] if expr._following else None)
        rows = None if all(r is None for r in rows) else rows

        sa_expr = f(input).over(partition_by=partition_by, order_by=order_by, rows=rows)
        self._add(expr, sa_expr)

    def visit_rank_window(self, expr):
        try:
            func_name = WINDOW_COMPILE_DIC[expr.node_name]
        except KeyError:
            raise NotImplementedError
        f = getattr(func, func_name)
        partition_by = self._gen_select_columns(expr._partition_by) \
            if expr._partition_by else None
        order_by = self._gen_select_columns(expr._order_by) \
            if expr._order_by else None

        sa_expr = f().over(partition_by=partition_by, order_by=order_by)
        if isinstance(expr, PercentRank):
            sa_expr = sa_expr.cast(types.df_type_to_sqlalchemy_type(expr.dtype))
        self._add(expr, sa_expr)

    def visit_shift_window(self, expr):
        input = self._expr_to_sqlalchemy[expr._input]
        try:
            func_name = WINDOW_COMPILE_DIC[expr.node_name]
        except KeyError:
            raise NotImplementedError
        f = getattr(func, func_name)
        partition_by = self._gen_select_columns(expr._partition_by) \
            if expr._partition_by else None
        order_by = self._gen_select_columns(expr._order_by) \
            if expr._order_by else None

        args = (input, self._expr_to_sqlalchemy[expr._offset])
        if expr._default:
            args += (literal(self._expr_to_sqlalchemy[expr._default]).cast(
                types.df_type_to_sqlalchemy_type(expr._input.dtype)),)

        sa_expr = f(*args).over(partition_by=partition_by, order_by=order_by)
        self._add(expr, sa_expr)

    def visit_scalar(self, expr):
        if expr._value is not None:
            if expr.dtype == df_types.string:
                val = utils.to_str(expr.value) \
                    if isinstance(expr.value, six.text_type) else expr.value
                self._add(expr, val)
                return
            else:
                self._add(expr, expr._value)
        else:
            self._add(expr, None)

    def visit_cast(self, expr):
        to_type = types.df_type_to_sqlalchemy_type(expr.dtype)
        self._add(expr, self._expr_to_sqlalchemy[expr.input].cast(to_type))

    def visit_join(self, expr):
        lhs, rhs = self._expr_to_sqlalchemy[expr._lhs], self._expr_to_sqlalchemy[expr._rhs]
        if isinstance(expr, RightJoin):
            lhs, rhs = rhs, lhs
        on = self._expr_to_sqlalchemy[expr._predicate]
        kw = dict()
        if isinstance(expr, OuterJoin):
            kw['full'] = True
        elif isinstance(expr, (LeftJoin, RightJoin)):
            kw['isouter'] = True
        joined = join(lhs, rhs, onclause=on, **kw)

        self._add(expr, joined)

    def visit_union(self, expr):
        lhs, rhs = self._expr_to_sqlalchemy[expr._lhs], self._expr_to_sqlalchemy[expr._rhs]
        if is_source_collection(expr._lhs):
            lhs = select([lhs])
        elif isinstance(lhs, Alias):
            lhs = lhs.element
        if is_source_collection(expr._rhs):
            rhs = select([rhs])
        elif isinstance(rhs, Alias):
            rhs = rhs.element
        method = union if expr._distinct else union_all
        unioned = method(lhs, rhs)

        if expr is not self._expr_dag.root:
            unioned = unioned.alias(self._new_alias())

        self._add(expr, unioned)