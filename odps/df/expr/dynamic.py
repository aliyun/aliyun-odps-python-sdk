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

import inspect

from .expressions import CollectionExpr, SequenceExpr, Scalar, ReprWrapper
from .utils import get_attrs
from ..types import DynamicSchema
from .. import types
from ..utils import is_source_collection


def _need_to_dynamic(res):
    if isinstance(res, CollectionExpr) and \
            not is_source_collection(res) and \
            (isinstance(res._schema, DynamicSchema) or
             any(isinstance(t, types.Unknown) for t in res.schema.types)):
        return True
    if isinstance(res, SequenceExpr) and \
            (isinstance(res.dtype, types.Unknown) or
             isinstance(res._source_data_type, types.Unknown)):
        return True
    if isinstance(res, Scalar) and res._value is not None and \
            (isinstance(res.dtype, types.Unknown) or
             isinstance(res._source_value_type, types.Unknown)):
        return True
    return False


def _get_base_type(res):
    if isinstance(res, CollectionExpr):
        return DynamicCollectionExpr
    if isinstance(res, SequenceExpr):
        return DynamicSequenceExpr
    if isinstance(res, Scalar):
        return DynamicScalar

    raise TypeError('Expr should be provided')


def to_dynamic(func):
    def inner(*args, **kwargs):
        res = func(*args, **kwargs)

        if res is None:
            return
        if isinstance(res, DynamicMixin):
            return res

        tp = type(res)
        new_tp = None
        if _need_to_dynamic(res):
            base_tp = _get_base_type(res)
            new_tp = type(tp.__name__, (base_tp, tp), dict())

        if new_tp is not None:
            attr_vals = dict((attr, getattr(res, attr)) for attr in get_attrs(res))
            if isinstance(res, SequenceExpr) and isinstance(res.dtype, types.Unknown) and \
                    res.dtype.type is not None:
                attr_vals['_data_type'] = res.dtype.type
            elif isinstance(res, Scalar) and isinstance(res.dtype, types.Unknown) and \
                    res.dtype.type is not None:
                attr_vals['_value_type'] = res.dtype.type

            new_res = new_tp(**attr_vals)
            if hasattr(new_res, '_schema'):
                for i, col in enumerate(new_res._schema._columns):
                    if isinstance(col.type, types.Unknown) and col.type.type is not None:
                        new_res._schema._columns[i] = type(col)(col.name, col.type.type)
            return new_res
        return res

    inner.__name__ = func.__name__
    inner.__doc__ = func.__doc__

    return inner


class DynamicMixin(object):
    __slots__ = ()

    def __getattribute__(self, item):
        res = super(DynamicMixin, self).__getattribute__(item)
        if inspect.ismethod(res) or isinstance(res, ReprWrapper):  # method may be wrapped as ReprWrapper
            return to_dynamic(res)
        return res

    def _copy_type(self):
        # the type which is not inherited from DynamicMixin
        tp = type(self)
        return [t for t in inspect.getmro(tp)
                if t.__name__ == tp.__name__ and not issubclass(t, DynamicMixin)][0]

    def copy(self):
        attr_dict = self._attr_dict()
        static_tp = self._copy_type()
        return static_tp(**attr_dict)

    def to_static(self):
        return self.rebuild()


class DynamicCollectionExpr(DynamicMixin, CollectionExpr):
    def __init__(self, *args, **kwargs):
        DynamicMixin.__init__(self)
        CollectionExpr.__init__(self, *args, **kwargs)

    def _project(self, fields):
        # when the collection is dynamic, and select all the field,
        # the projected collection should be dynamic
        is_dynamic = False
        for field in fields:
            field = self._get_field(field)
            if isinstance(field, CollectionExpr) and isinstance(field, DynamicMixin):
                is_dynamic = True
                break

        if is_dynamic:
            _, raw_fields = self._get_fields(fields, ret_raw_fields=True)
            def func(fs):
                default_type = self._schema.default_type \
                    if isinstance(self._schema, DynamicSchema) else None
                expr = super(DynamicCollectionExpr, self)._project(fs)
                expr._raw_fields = raw_fields
                if not isinstance(expr._schema, DynamicSchema):
                    expr._schema = DynamicSchema.from_schema(expr._schema,
                                                             default_type=default_type)
                return expr

            res = to_dynamic(func)(fields)

            return res
        return super(DynamicCollectionExpr, self)._project(fields)


class DynamicSequenceExpr(DynamicMixin, SequenceExpr):
    def __init__(self, *args, **kwargs):
        DynamicMixin.__init__(self)
        SequenceExpr.__init__(self, *args, **kwargs)


class DynamicScalar(DynamicMixin, Scalar):
    def __init__(self, *args, **kwargs):
        DynamicMixin.__init__(self)
        Scalar.__init__(self, *args, **kwargs)

