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

from __future__ import absolute_import

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

try:
    import sqlalchemy
    has_sqlalchemy = True
except ImportError:
    has_sqlalchemy = False

from ...models import Table
from ...models.ml import OfflineModel
from ..expr.expressions import CollectionExpr
from ..expr.core import ExprDictionary
from .odpssql.models import MemCacheReference
from .seahawks.models import SeahawksTable
from .core import EngineTypes as Engines
from .errors import NoBackendFound
from .utils import fetch_data_source_size
from ...config import options


def available_engines(sources):
    from ...ml.models import TablesModelObject
    engines = set()

    for src in sources:
        if isinstance(src, SeahawksTable):
            engines.add(Engines.SEAHAWKS)
        elif isinstance(src, (Table, MemCacheReference)):
            engines.add(Engines.ODPS)
        elif isinstance(src, (OfflineModel, TablesModelObject)):
            engines.add(Engines.ALGO)
        elif has_pandas and isinstance(src, pd.DataFrame):
            engines.add(Engines.PANDAS)
        elif has_sqlalchemy and isinstance(src, sqlalchemy.Table):
            engines.add(Engines.SQLALCHEMY)
        else:
            raise NoBackendFound('No backend found for data source: %s' % src)

    return engines


class EngineSelecter(object):
    def __init__(self):
        self._node_engines = ExprDictionary()
        self.force_odps = False

    def _choose_backend(self, node):
        if node in self._node_engines:
            return self._node_engines[node]

        available_engine_size = len(Engines)

        engines = set()
        children = node.children()
        if len(children) == 0:
            if not isinstance(node, CollectionExpr):
                self._node_engines[node] = None
                return
            engines.update(available_engines(node._data_source()))
        else:
            for n in children:
                e = self._choose_backend(n)
                if e and e not in engines:
                    engines.add(e)
                if len(engines) == available_engine_size:
                    break

        if len(engines) == 1:
            self._node_engines[node] = engines.pop()
        else:
            self._node_engines[node] = self._choose(node)

        return self._node_engines[node]

    def has_diff_data_sources(self, node, no_cache=False):
        if no_cache:
            return len(available_engines(node.data_source())) > 1

        children = node.children()
        if len(children) == 0:
            return False
        return len(set(filter(lambda x: x is not None,
                              (self._choose_backend(c) for c in children)))) > 1

    def _has_data_source(self, node, src):
        children = node.children()
        if len(children) == 0:
            return src == self._choose_backend(node)
        return src in set(self._choose_backend(c) for c in node.children())

    def has_pandas_data_source(self, node):
        return self._has_data_source(node, Engines.PANDAS)

    def has_odps_data_source(self, node):
        return self._has_data_source(node, Engines.ODPS)

    def _choose(self, node):
        # Now if we do an operation like join
        # which comes from different data sources
        # we just use ODPS to caculate.
        # In the future, we will choose according to the scale of data

        return Engines.ODPS

    def _choose_odps_backend(self, node_dag):
        node = node_dag.root

        if not has_sqlalchemy or not options.seahawks_url:
            return Engines.ODPS

        sizes = 0
        for n in node.traverse(top_down=True, unique=True):
            for ds in n._data_source():
                if isinstance(ds, MemCacheReference):
                    return Engines.ODPS  # if use mem cache, directly return ODPSSQL backend
                if not isinstance(ds, Table) or isinstance(ds, SeahawksTable):
                    continue
                if isinstance(ds, Table) and n._deps is not None:
                    # the cached collection, the table has not been calculated yet
                    continue
                size = fetch_data_source_size(node_dag, n, ds)
                if size is None:
                    return Engines.ODPS
                sizes += size

        if sizes <= options.df.seahawks.max_size:
            return Engines.SEAHAWKS

        return Engines.ODPS

    def select(self, expr_dag):
        node = expr_dag.root

        if getattr(node, '_algo', None) is not None:
            return Engines.ALGO

        if not self.has_diff_data_sources(node):
            engine = self._choose_backend(node)
            if engine == Engines.ODPS and not self.force_odps:
                return self._choose_odps_backend(expr_dag)
            return engine

        engines = available_engines(node.data_source())
        if len(engines) > 1 or (len(engines) == 1 and list(engines)[0] == Engines.ODPS):
            if not self.force_odps:
                return self._choose_odps_backend(expr_dag)
            else:
                return Engines.ODPS
        return engines.pop()

