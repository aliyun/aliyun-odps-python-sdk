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

import tokenize
import ast
import operator
from collections import OrderedDict

from . import element as ele
from .expressions import Expr, ExpressionError
from ...compat import reduce, StringIO

LOCAL_TAG = '_local_var_'


def _tokenize_str(reader):
    for toknum, tokval, _, _, _ in tokenize.generate_tokens(reader):
        if toknum == tokenize.OP:
            if tokval == '@':
                tokval = LOCAL_TAG
            if tokval == '&':
                toknum = tokenize.NAME
                tokval = 'and'
            elif tokval == '|':
                toknum = tokenize.NAME
                tokval = 'or'
        yield toknum, '==' if tokval == '=' else tokval


class ExprVisitor(ast.NodeVisitor):
    _op_handlers = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.mod: operator.mod,
        ast.Pow: operator.pow,

        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: ele._isin,
        ast.NotIn: ele._notin,

        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Invert: operator.invert,

        ast.And: operator.and_,
        ast.Or: operator.or_
    }

    def __init__(self, env):
        self.env = env

    def _preparse(self, expr):
        reader = StringIO(expr).readline
        return tokenize.untokenize(list(_tokenize_str(reader)))

    def eval(self, expr, rewrite=True):
        if rewrite:
            expr = self._preparse(expr)
        node = ast.fix_missing_locations(ast.parse(expr))
        return self.visit(node)

    def get_named_object(self, obj_name):
        raise NotImplementedError

    def visit(self, node):
        if isinstance(node, Expr):
            return node
        node_name = node.__class__.__name__
        method = 'visit_' + node_name
        try:
            visitor = getattr(self, method)
        except AttributeError:
            raise ExpressionError('Query string contains unsupported syntax: {}'.format(node_name))
        return visitor(node)

    def visit_Module(self, node):
        if len(node.body) != 1:
            raise SyntaxError('Only a single expression is allowed')
        expr = node.body[0]
        return self.visit(expr)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._op_handlers[type(node.op)](left, right)

    def visit_Call(self, node):
        from .expressions import ReprWrapper
        func = self.visit(node.func)
        if not isinstance(func, ReprWrapper):
            raise SyntaxError('Calling none-expression methods is not allowed.')
        args = [self.visit(n) for n in node.args]
        kwargs = OrderedDict([(kw.arg, self.visit(kw.value)) for kw in node.keywords])
        return func(*args, **kwargs)

    def visit_Compare(self, node):
        ops = node.ops
        comps = node.comparators

        if len(comps) == 1:
            binop = ast.BinOp(op=ops[0], left=node.left, right=comps[0])
            return self.visit(binop)

        left = node.left
        values = []
        for op, comp in zip(ops, comps):
            new_node = ast.Compare(comparators=[comp], left=left, ops=[op])
            left = comp
            values.append(new_node)
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    def visit_BoolOp(self, node):
        def func(lhs, rhs):
            binop = ast.BinOp(op=node.op, left=lhs, right=rhs)
            return self.visit(binop)
        return reduce(func, node.values)

    def visit_UnaryOp(self, node):
        oprand = self.visit(node.operand)
        return self._op_handlers[type(node.op)](oprand)

    def visit_Name(self, node):
        if node.id.startswith(LOCAL_TAG):
            local_name = node.id.replace(LOCAL_TAG, '')
            return self.env[local_name]
        if node.id in ['True', 'False', 'None']:
            return eval(node.id)
        return self.get_named_object(node.id)

    def visit_NameConstant(self, node):
        return node.value

    def visit_Num(self, node):
        return node.n

    def visit_Str(self, node):
        return node.s

    def visit_Constant(self, node):
        return node.value

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    visit_Tuple = visit_List

    def visit_Attribute(self, node):
        attr = node.attr
        value = node.value

        ctx = node.ctx
        if isinstance(ctx, ast.Load):
            resolved = self.visit(value)
            return getattr(resolved, attr)

        raise ValueError("Invalid Attribute context {0}".format(ctx.__name__))

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        sub = self.visit(node.slice)
        return value[sub]

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Slice(self, node):
        lower = node.lower
        if lower is not None:
            lower = self.visit(lower)
        upper = node.upper
        if upper is not None:
            upper = self.visit(upper)
        step = node.step
        if step is not None:
            step = self.visit(step)

        return slice(lower, upper, step)


class CollectionVisitor(ExprVisitor):
    def __init__(self, collection, env):
        super(CollectionVisitor, self).__init__(env)
        self.collection = collection

    def get_named_object(self, obj_name):
        return self.collection._get_field(obj_name)


class SequenceVisitor(ExprVisitor):
    def __init__(self, sequence, env):
        super(SequenceVisitor, self).__init__(env)
        self.sequence = sequence

    def get_named_object(self, obj_name):
        return getattr(self.sequence, obj_name)
