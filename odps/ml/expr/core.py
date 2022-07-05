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

import threading
import weakref
from collections import namedtuple

from ..utils import import_class_member
from ...compat import six
from ...utils import hashable
from ...df import DataFrame
from ...df.expr.collections import CollectionExpr, Expr, FilterPartitionCollectionExpr
from ...df.expr.dynamic import DynamicMixin, DynamicCollectionExpr
from ...utils import get_id

expr_output_registry = dict()
shared_props_registry = dict()


class AlgoExprProxy(object):
    def __init__(self, expr):
        def _finalizer(_):
            if self._exec_id not in expr_output_registry:
                return
            if self._register_name in expr_output_registry[self._exec_id]:
                del expr_output_registry[self._exec_id][self._register_name]
            if len(expr_output_registry[self._exec_id]) == 0:
                del expr_output_registry[self._exec_id]
                if self._exec_id in shared_props_registry:
                    del shared_props_registry[self._exec_id]

        self._ref = weakref.ref(expr, _finalizer)
        self._exec_id = expr._exec_id
        self._register_name = expr.register_name

    def __call__(self):
        return self._ref()

    def __getattr__(self, item):
        return getattr(self._ref(), item)


class AlgoExprMixin(Expr):
    _mixin_slots = '_exec_id', '_params', '_engine_kw', '_output_name', '_persist_kw', '_cases', \
                   '_table_callback', '_is_extra'
    _add_args_slots = False
    _algo = ''

    @staticmethod
    def cache_input(expr):
        if not isinstance(expr, Expr):
            return expr
        if isinstance(expr, FilterPartitionCollectionExpr) and isinstance(expr.input, DataFrame):
            return expr
        if isinstance(expr, AlgoExprMixin) and expr.executed:
            return expr
        return expr.cache(mem=False)

    def _register_name(self):
        return self._output_name

    register_name = property(fget=_register_name)

    def register_expr(self):
        if self._exec_id not in expr_output_registry:
            expr_output_registry[self._exec_id] = dict()
        expr_output_registry[self._exec_id][self.register_name] = AlgoExprProxy(self)
        shared_props_registry[self._exec_id] = dict()
        shared_props_registry[self._exec_id]['executed'] = False

    def uncache(self):
        pass

    def _get_executed(self):
        if self._exec_id not in shared_props_registry:
            return False
        return shared_props_registry[self._exec_id].get('executed', False)

    def _set_executed(self, value):
        if self._exec_id not in shared_props_registry:
            shared_props_registry[self._exec_id] = dict()
        shared_props_registry[self._exec_id]['executed'] = value
        if value and 'exec_lock' in shared_props_registry[self._exec_id]:
            exec_lock = shared_props_registry[self._exec_id]['exec_lock']
            exec_lock.release()

    executed = property(fget=lambda self: self._get_executed(),
                        fset=lambda self, value: self._set_executed(value))

    def wait_execution(self):
        if self._exec_id not in shared_props_registry:
            shared_props_registry[self._exec_id] = dict()
        if 'exec_lock' not in shared_props_registry[self._exec_id]:
            exec_lock = threading.Lock()
            shared_props_registry[self._exec_id]['exec_lock'] = exec_lock
        else:
            exec_lock = shared_props_registry[self._exec_id]['exec_lock']
        if not self.executed:
            exec_lock.acquire()

    def convert_params(self, src_expr=None):
        params = dict((k, v) for k, v in six.iteritems(self._params) if k in self._exported)

        for name, exporter in six.iteritems(self._exporters):
            if src_expr is not None:
                params[name] = exporter(src_expr)
            if not params[name]:
                params[name] = exporter(self)

        for k, v in list(six.iteritems(params)):
            if v is None:
                params.pop(k)
            if isinstance(v, (list, tuple, set)) and len(v) == 0:
                params.pop(k)
        return params

    @property
    def persist_kw(self):
        if self._exec_id not in expr_output_registry:
            raise AttributeError
        if self.register_name not in expr_output_registry[self._exec_id]:
            raise AttributeError
        return expr_output_registry[self._exec_id][self.register_name]._persist_kw

    @persist_kw.setter
    def persist_kw(self, value):
        if self._exec_id not in expr_output_registry:
            raise AttributeError
        if self.register_name not in expr_output_registry[self._exec_id]:
            raise AttributeError
        expr_output_registry[self._exec_id][self.register_name]._persist_kw = value

    @property
    def shared_kw(self):
        if self._exec_id not in shared_props_registry:
            raise AttributeError
        return shared_props_registry[self._exec_id].get('shared_kw', dict())

    @shared_kw.setter
    def shared_kw(self, value):
        if self._exec_id not in shared_props_registry:
            raise AttributeError
        shared_props_registry[self._exec_id]['shared_kw'] = value

    def outputs(self):
        if getattr(self, '_exec_id', None) is None:
            return dict()
        out_dict = expr_output_registry.get(self._exec_id, dict())
        out_dict = dict((k, v()) for k, v in six.iteritems(out_dict))
        return dict((k, v) for k, v in six.iteritems(out_dict) if v is not None and not v.is_extra_expr)

    @property
    def is_extra_expr(self):
        return getattr(self, '_is_extra', False)


class AlgoCollectionExpr(AlgoExprMixin, CollectionExpr):
    _suffix = 'CollectionExpr'
    node_name = 'Algo'

    def _init(self, *args, **kwargs):
        register_expr = kwargs.pop('register_expr', False)

        p_args = [a.cache() if isinstance(a, Expr) else a for a in args]
        p_kw = dict((k, self.cache_input(v)) for k, v in six.iteritems(kwargs))

        absent_args = (a for a in self._args[len(args):] if a not in p_kw)
        for a in absent_args:
            p_kw[a] = None

        super(AlgoCollectionExpr, self)._init(*p_args, **p_kw)
        self.cache(False)

        if register_expr:
            self.register_expr()

    def accept(self, visitor):
        visitor.visit_algo(self)


class DynamicAlgoCollectionExpr(DynamicMixin, AlgoCollectionExpr):
    def __init__(self, *args, **kwargs):
        DynamicMixin.__init__(*args, **kwargs)
        AlgoCollectionExpr.__init__(*args, **kwargs)

    _project = DynamicCollectionExpr._project


metrics_tables = dict()
metrics_executed = dict()


class MetricsResultExpr(AlgoExprMixin, Expr):
    __slots__ = '_metrics_hash', '_result_callback'
    _suffix = 'ResultExpr'
    _non_table = True
    node_name = 'Metrics'

    def _init(self, *args, **kwargs):
        kwargs.pop('register_expr', False)

        p_args = [a.cache() if isinstance(a, Expr) else a for a in args]
        p_kw = dict((k, self.cache_input(v)) for k, v in six.iteritems(kwargs))

        absent_args = (a for a in self._args[len(args):] if a not in p_kw)
        for a in absent_args:
            p_kw[a] = None

        super(MetricsResultExpr, self)._init(*p_args, **p_kw)

        if getattr(self, '_metrics_hash', None) is None:
            param_hash = hash(frozenset(six.iteritems(hashable(self._params))))
            port_hash = hash(frozenset(self.get_input_hash(pt) for pt in self.input_ports))
            self._metrics_hash = hash((self.node_name, param_hash, port_hash))

    def get_input_hash(self, input_port):
        from .exporters import get_ml_input
        input_expr = get_ml_input(self, input_port.name)
        pid = get_id(input_expr) if input_expr else None
        return hash((input_port.name, pid))

    @property
    def tables_tuple(self):
        cls = type(self)
        if not hasattr(cls, '_tables_tuple'):
            cls._tables_tuple = namedtuple('_tables_tuple', [pt.name for pt in cls.output_ports])
        return cls._tables_tuple

    @property
    def calculator(self):
        cls = type(self)
        if not hasattr(cls, '_calculator'):
            cls._calculator = import_class_member(self.algo_meta['calculator'])
        return self._calculator

    def accept(self, visitor):
        visitor.visit_algo(self)

    def _get_executed(self):
        return metrics_executed.get(self._metrics_hash, False)

    def _set_executed(self, value):
        metrics_executed[self._metrics_hash] = value

    @property
    def tables(self):
        return metrics_tables.get(self._metrics_hash)

    @tables.setter
    def tables(self, value):
        if isinstance(value, dict):
            value = self.tables_tuple(**value)
        metrics_tables[self._metrics_hash] = value


def make_extra_expr(expr, name, schema, **kw):
    new_expr = expr.copy(clear_keys=['_id'], register_expr=True, _is_extra=True, _schema=schema,
                         _output_name=name, **kw)
    new_expr.cache(False)
    return new_expr
