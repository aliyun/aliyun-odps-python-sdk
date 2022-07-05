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

from datetime import datetime

from .expressions import *
from .element import AnyOp, ElementWise
from . import utils
from .. import types


class BinOp(AnyOp):
    __slots__ = ()
    _args = '_lhs', '_rhs'

    @property
    def node_name(self):
        return self.__class__.__name__

    @property
    def name(self):
        if self._name:
            return self._name

        seq_hs = [hs for hs in self.args if isinstance(hs, SequenceExpr)]
        if len(seq_hs) == 1:
            return seq_hs[0].name

    def accept(self, visitor):
        visitor.visit_binary_op(self)


class UnaryOp(ElementWise):
    __slots__ = ()

    def accept(self, visitor):
        visitor.visit_unary_op(self)


class Arithmetic(BinOp):
    __slots__ = ()


class Comparison(BinOp):
    __slots__ = ()


class LogicalBinOp(BinOp):
    __slots__ = ()


class Negate(UnaryOp):
    __slots__ = ()


class Invert(UnaryOp):
    __slots__ = ()


class Abs(UnaryOp):
    __slots__ = ()


class Add(Arithmetic):
    __slots__ = ()


class Substract(Arithmetic):
    __slots__ = ()


class Multiply(Arithmetic):
    __slots__ = ()


class Divide(Arithmetic):
    __slots__ = ()


class FloorDivide(Arithmetic):
    __slots__ = ()


class Mod(Arithmetic):
    __slots__ = ()


class Power(Arithmetic):
    __slots__ = ()


class Greater(Comparison):
    __slots__ = ()


class Less(Comparison):
    __slots__ = ()


class Equal(Comparison):
    __slots__ = ()


class NotEqual(Comparison):
    __slots__ = ()


class GreaterEqual(Comparison):
    __slots__ = ()


class LessEqual(Comparison):
    __slots__ = ()


class Or(LogicalBinOp):
    __slots__ = ()


class And(LogicalBinOp):
    __slots__ = ()


def _get_type(other):
    if isinstance(other, SequenceExpr):
        other_type = other._data_type
    elif isinstance(other, Scalar):
        other_type = other._value_type
    else:
        other = Scalar(_value=other)
        other_type = other._value_type

    return other_type, other


def _arithmetic(expr, other, output_expr_cls, reverse=False, output_type=None):
    if isinstance(expr, (SequenceExpr, Scalar)):
        other_type, other = _get_type(other)
        is_sequence = isinstance(expr, SequenceExpr) or isinstance(other, SequenceExpr)
        if output_type is None:
            output_type = utils.highest_precedence_data_type(expr.dtype, other_type)

        if reverse:
            expr, other = other, expr
        if is_sequence:
            return output_expr_cls(_data_type=output_type, _lhs=expr, _rhs=other)
        else:
            return output_expr_cls(_value_type=output_type, _lhs=expr, _rhs=other)


def _reversed_arithmetic(expr, other, output_expr_cls, output_type=None):
    return _arithmetic(expr, other, output_expr_cls, reverse=True, output_type=output_type)


def _cmp(expr, other, output_expr_cls):
    if isinstance(expr, (SequenceExpr, Scalar)):
        other_type, other = _get_type(other)
        is_sequence = isinstance(expr, SequenceExpr) or isinstance(other, SequenceExpr)
        utils.highest_precedence_data_type(expr.dtype, other_type)
        # operand cast to data_type to compare
        output_type = types.boolean

        if is_sequence:
            return output_expr_cls(_data_type=output_type, _lhs=expr, _rhs=other)
        else:
            return output_expr_cls(_value_type=output_type, _lhs=expr, _rhs=other)


def _unary(expr, output_expr_cls):
    if isinstance(expr, (SequenceExpr, Scalar)):
        is_sequence = isinstance(expr, SequenceExpr)

        if is_sequence:
            return output_expr_cls(_data_type=expr.dtype, _input=expr)
        else:
            return output_expr_cls(_value_type=expr.dtype, _input=expr)


def _logic(expr, other, output_expr_cls):
    if isinstance(expr, (SequenceExpr, Scalar)):
        other_type, other = _get_type(other)
        is_sequence = isinstance(expr, SequenceExpr) or isinstance(other, SequenceExpr)

        if expr.dtype == types.boolean and other.dtype == types.boolean:
            output_type = types.boolean
            if is_sequence:
                return output_expr_cls(_data_type=output_type, _lhs=expr, _rhs=other)
            else:
                return output_expr_cls(_value_type=output_type, _lhs=expr, _rhs=other)

        raise TypeError('Logic operation needs boolean operand')


def _is_datetime(expr):
    if isinstance(expr, Expr):
        return expr.dtype == types.datetime
    else:
        return isinstance(expr, datetime)


def _add(expr, other):
    if _is_datetime(expr) and _is_datetime(other):
        raise ExpressionError('Cannot add two datetimes')
    return _arithmetic(expr, other, Add)


def _radd(expr, other):
    if _is_datetime(expr) and _is_datetime(other):
        raise ExpressionError('Cannot add two datetimes')
    return _reversed_arithmetic(expr, other, Add)


def _sub(expr, other):
    rtype = None
    if _is_datetime(expr) and _is_datetime(other):
        rtype = types.int64
    return _arithmetic(expr, other, Substract, output_type=rtype)


def _rsub(expr, other):
    rtype = None
    if _is_datetime(expr) and _is_datetime(other):
        rtype = types.int64
    return _reversed_arithmetic(expr, other, Substract, output_type=rtype)


def _eq(expr, other):
    return _cmp(expr, other, Equal)


def _ne(expr, other):
    return _cmp(expr, other, NotEqual)


def _gt(expr, other):
    return _cmp(expr, other, Greater)


def _lt(expr, other):
    return _cmp(expr, other, Less)


def _le(expr, other):
    return _cmp(expr, other, LessEqual)


def _ge(expr, other):
    return _cmp(expr, other, GreaterEqual)


def _mul(expr, other):
    return _arithmetic(expr, other, Multiply)


def _rmul(expr, other):
    return _reversed_arithmetic(expr, other, Multiply)


def _div(expr, other):
    if isinstance(expr.dtype, types.Integer) and isinstance(_get_type(other)[0], types.Integer):
        output_type = types.float64
    else:
        output_type = None
    return _arithmetic(expr, other, Divide, output_type=output_type)


def _rdiv(expr, other):
    if isinstance(expr.dtype, types.Integer) and isinstance(_get_type(other)[0], types.Integer):
        output_type = types.float64
    else:
        output_type = None
    return _reversed_arithmetic(expr, other, Divide, output_type=output_type)


def _mod(expr, other):
    return _arithmetic(expr, other, Mod)


def _rmod(expr, other):
    return _reversed_arithmetic(expr, other, Mod)


def _floordiv(expr, other):
    return _arithmetic(expr, other, FloorDivide)


def _rfloordiv(expr, other):
    return _reversed_arithmetic(expr, other, FloorDivide)


def _pow(expr, other):
    return _arithmetic(expr, other, Power)


def _rpow(expr, other):
    return _reversed_arithmetic(expr, other, Power)


def _or(expr, other):
    return _logic(expr, other, Or)


def _ror(expr, other):
    return _or(expr, other)


def _and(expr, other):
    return _logic(expr, other, And)


def _rand(expr, other):
    return _and(expr, other)


def _neg(expr):
    if isinstance(expr, Negate):
        return expr.input

    return _unary(expr, Negate)


def _invert(expr):
    if isinstance(expr, Invert):
        return expr.input

    return _unary(expr, Invert)


def _abs(expr):
    if isinstance(expr, Abs):
        return expr

    return _unary(expr, Abs)


_number_methods = dict(
    _add=_add,
    _radd=_radd,
    _sub=_sub,
    _rsub=_rsub,
    _mul=_mul,
    _rmul=_rmul,
    _div=_div,
    _rdiv=_rdiv,
    _floordiv=_floordiv,
    _rfloordiv=_rfloordiv,
    _mod=_mod,
    _rmod=_rmod,
    _pow=_pow,
    _rpow=_rpow,
    _neg=_neg,
    _abs=_abs,
    _eq=_eq,
    _ne=_ne,
    _gt=_gt,
    _lt=_lt,
    _le=_le,
    _ge=_ge,
)

_int_number_methods = dict(
    _invert=_invert
)

_string_methods = dict(
    _add=_add,
    _radd=_radd,
    _eq=_eq,
    _ne=_ne,
    _gt=_gt,
    _lt=_lt,
    _le=_le,
    _ge=_ge,
)

_boolean_methods = dict(
    _or=_or,
    _ror=_ror,
    _and=_and,
    _rand=_rand,
    _eq=_eq,
    _ne=_ne,
    _invert=_invert,
    _neg=_neg
)

_datetime_methods = dict(
    _add=_add,  # TODO, to check
    _radd=_radd,
    _sub=_sub,
    _rsub=_rsub,
    _eq=_eq,
    _ne=_ne,
    _gt=_gt,
    _lt=_lt,
    _le=_le,
    _ge=_ge
)

utils.add_method(StringSequenceExpr, _string_methods)
utils.add_method(StringScalar, _string_methods)

utils.add_method(BooleanSequenceExpr, _boolean_methods)
utils.add_method(BooleanScalar, _boolean_methods)

utils.add_method(DatetimeSequenceExpr, _datetime_methods)
utils.add_method(DatetimeScalar, _datetime_methods)

for number_sequence in number_sequences + number_scalars:
    utils.add_method(number_sequence, _number_methods)

for int_number_sequence in int_number_sequences + int_number_scalars:
    utils.add_method(int_number_sequence, _int_number_methods)
