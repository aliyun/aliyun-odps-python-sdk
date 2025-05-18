# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import json

import requests

try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import pandas as pd
except ImportError:
    pd = None

from ..compat import six
from ..serializers import (
    JSONNodeField,
    JSONNodeReferenceField,
    JSONNodesReferencesField,
    JSONSerializableModel,
)
from ..types import is_record, validate_data_type, validate_value
from ..utils import backquote_string, to_odps_scalar
from .functions import ExprFunction

_name_to_expr_clses = {}


class Expression(JSONSerializableModel):
    __slots__ = ("final",)
    _type = JSONNodeField("type")

    @classmethod
    def _load_expr_classes(cls):
        if not _name_to_expr_clses:
            for val in globals().values():
                if (
                    not isinstance(val, type)
                    or not issubclass(val, Expression)
                    or val is Expression
                ):
                    continue
                cls_name = val.__name__[0].lower() + val.__name__[1:]
                _name_to_expr_clses[cls_name] = val
        return _name_to_expr_clses

    @classmethod
    def _get_expr_class(cls, expr):
        expr_name = next(iter(expr.keys()))
        return cls._load_expr_classes()[expr_name]

    @classmethod
    def deserial(cls, content, obj=None, **kw):
        if obj is None:
            inst_cls = cls._get_expr_class(content)
            obj = inst_cls(_parent=kw.get("_parent"))
            content = next(iter(content.values()))
        return super(Expression, cls).deserial(content, obj=obj, **kw)

    @property
    def type(self):
        return validate_data_type(self._type)

    def eval(self, data):
        raise NotImplementedError

    def to_str(self, ref_to_str=None):
        raise NotImplementedError

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, str(self))

    def _make_final_result(self, data, res):
        from ..tunnel.io.types import odps_type_to_arrow_type

        if not self.final:
            return res

        if is_record(data):
            return res
        elif pd and isinstance(data, pd.DataFrame):
            if not isinstance(res, pd.Series):
                return pd.Series([res] * len(data))
            return res
        elif pa and isinstance(data, (pa.RecordBatch, pa.Table)):
            if isinstance(res, (pa.Array, pa.ChunkedArray)):
                return res
            return pa.array(
                [res] * data.num_rows, type=odps_type_to_arrow_type(self.type)
            )


class FunctionCall(Expression):
    __slots__ = ("args", "references")

    name = JSONNodeField("name")

    def to_str(self, ref_to_str=None):
        ref_to_str = ref_to_str or {}
        if not hasattr(self, "args"):
            return super(FunctionCall, self).to_str(ref_to_str)

        func_cls = self.get_function_cls()
        return func_cls.to_str([s.to_str(ref_to_str) for s in self.args])

    def get_function_cls(self):
        return ExprFunction.get_cls(self.name)

    def eval(self, data):
        func_cls = self.get_function_cls()
        args = [a.eval(data) for a in self.args]
        res = func_cls.call(*args)
        return self._make_final_result(data, res)


class LeafExprDesc(Expression):
    class Reference(JSONSerializableModel):
        name = JSONNodeField("name")

    constant = JSONNodeField("constant", default=None)
    reference = JSONNodeReferenceField(Reference, "reference", default=None)

    def eval(self, data):
        if self.constant:
            val = validate_value(self.constant, self.type)
        elif self.reference:
            if pa and isinstance(data, (pa.RecordBatch, pa.Table)):
                name_to_idx = {
                    c.lower(): idx for idx, c in enumerate(data.schema.names)
                }
                val = data.column(name_to_idx[self.reference.name.lower()])
            elif pd and isinstance(data, pd.DataFrame):
                lower_to_name = {c.lower(): c for c in data.columns}
                val = data[lower_to_name[self.reference.name.lower()]]
            else:
                val = data[self.reference.name]
        else:
            raise NotImplementedError("Expression cannot be called")
        return self._make_final_result(data, val)

    def to_str(self, ref_to_str=None):
        ref_to_str = ref_to_str or {}
        if self.constant:
            val = validate_value(self.constant, self.type)
            return to_odps_scalar(val)
        elif self.reference:
            default_str = backquote_string(self.reference.name)
            return ref_to_str.get(self.reference.name, default_str)
        else:
            raise NotImplementedError("Expression cannot be accepted")


class VisitedExpressions(JSONSerializableModel):
    expressions = JSONNodesReferencesField(Expression, "expressions")

    @classmethod
    def parse(cls, response, obj=None, **kw):
        if isinstance(response, Expression):
            return response

        if isinstance(response, requests.Response):
            # PY2 prefer bytes, while PY3 prefer str
            response = response.content.decode() if six.PY3 else response.content
        if isinstance(response, six.string_types):
            response = json.loads(response)

        if isinstance(response, list):
            response = {"expressions": response}
        parsed = super(VisitedExpressions, cls).parse(response, obj=obj, **kw)

        res_stack = []
        for expr in parsed.expressions:
            expr.final = False
            if isinstance(expr, FunctionCall):
                arg_count = expr.get_function_cls().arg_count
                expr.args = res_stack[-arg_count:]
                res_stack = res_stack[:-arg_count]
            res_stack.append(expr)
        assert len(res_stack) == 1
        res_stack[0].final = True
        res_stack[0].references = [
            t.reference.name
            for t in parsed.expressions
            if isinstance(t, LeafExprDesc) and t.reference is not None
        ]
        return res_stack[0]


parse = VisitedExpressions.parse
