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

from enum import Enum
import six

from odps.errors import DependencyNotInstalledError
from odps.df.expr.expressions import CollectionExpr, SequenceExpr, run_at_once
from odps.df.types import is_number


class PlottingCore(Enum):
    PANDAS = 'pandas'


def _plot_pandas(df, kind='line', **kwargs):
    x_label, y_label = kwargs.pop('xlabel', None), kwargs.pop('ylabel', None)

    fig = df.plot(kind=kind, **kwargs)
    if x_label:
        fig.set_xlabel(x_label)
    if y_label:
        fig.set_ylabel(y_label)

    return fig


def _hist_pandas(df, **kwargs):
    x_label, y_label = kwargs.pop('xlabel', None), kwargs.pop('ylabel', None)
    title = kwargs.pop('title', None)

    fig = df.hist(**kwargs)
    if x_label:
        fig.set_xlabel(x_label)
    if y_label:
        fig.set_ylabel(y_label)
    if title:
        fig.set_title(title)

    return fig


def _boxplot_pandas(df, **kwargs):
    x_label, y_label = kwargs.pop('xlabel', None), kwargs.pop('ylabel', None)
    title = kwargs.pop('title', None)

    fig = df.boxplot(**kwargs)
    if x_label:
        fig.set_xlabel(x_label)
    if y_label:
        fig.set_ylabel(y_label)
    if title:
        fig.set_title(title)

    return fig


@run_at_once
def _plot_sequence(expr, kind='line', **kwargs):
    try:
        import pandas as pd
    except ImportError:
        raise DependencyNotInstalledError('plot requires for pandas')

    series = expr.to_pandas()
    xerr = kwargs.get('xerr', None)
    if xerr is not None and isinstance(xerr, (CollectionExpr, SequenceExpr)):
        kwargs['xerr'] = xerr.to_pandas()
    yerr = kwargs.get('yerr', None)
    if yerr is not None and isinstance(yerr, (CollectionExpr, SequenceExpr)):
        kwargs['yerr'] = yerr.to_pandas()
    return _plot_pandas(series, kind=kind, **kwargs)


@run_at_once
def _hist_sequence(expr, **kwargs):
    try:
        import pandas as pd
    except ImportError:
        raise DependencyNotInstalledError('plot requires for pandas')

    series = expr.to_pandas()
    return _hist_pandas(series, **kwargs)


@run_at_once
def _plot_collection(expr, x=None, y=None, kind='line', **kwargs):
    try:
        import pandas as pd
    except ImportError:
        raise DependencyNotInstalledError('plot requires for pandas')

    fields = []
    x_name = None
    y_name = None

    if x is not None:
        fields.append(x)
        if isinstance(x, six.string_types):
            x_name = x
        else:
            x_name = x.name
    if y is not None:
        fields.append(y)
        if isinstance(y, six.string_types):
            y_name = y
        else:
            y_name = y.name

    if x_name is None or y_name is None:
        for col in expr.dtypes.columns:
            if col.name == x_name or col.name == y_name:
                continue
            elif is_number(col.type):
                fields.append(col.name)

    df = expr[fields].to_pandas()
    if x_name is not None:
        kwargs['x'] = x_name
    if y is not None:
        kwargs['y'] = y_name

    xerr = kwargs.get('xerr', None)
    if xerr is not None and isinstance(xerr, (CollectionExpr, SequenceExpr)):
        kwargs['xerr'] = xerr.to_pandas()
    yerr = kwargs.get('yerr', None)
    if yerr is not None and isinstance(yerr, (CollectionExpr, SequenceExpr)):
        kwargs['yerr'] = yerr.to_pandas()

    return _plot_pandas(df, kind=kind, **kwargs)


@run_at_once
def _hist_collection(expr, **kwargs):
    try:
        import pandas as pd
    except ImportError:
        raise DependencyNotInstalledError('plot requires for pandas')

    column = kwargs.get('column')
    if isinstance(column, six.string_types):
        column = [column, ]
    if column is not None:
        expr = expr[column]

    df = expr.to_pandas()

    return _hist_pandas(df, **kwargs)


@run_at_once
def _boxplot_collection(expr, **kwargs):
    try:
        import pandas as pd
    except ImportError:
        raise DependencyNotInstalledError('plot requires for pandas')

    fields = set()

    column = kwargs.get('column')
    if isinstance(column, six.string_types):
        fields.add(column)
    elif column is not None:
        fields = fields.union(column)

    by = kwargs.get('by')
    if isinstance(by, six.string_types):
        fields.add(by)
    elif by is not None:
        fields = fields.union(by)

    if fields:
        expr = expr[list(fields)]
    df = expr.to_pandas()

    return _boxplot_pandas(df, **kwargs)


CollectionExpr.plot = _plot_collection
CollectionExpr.hist = _hist_collection
CollectionExpr.boxplot = _boxplot_collection
SequenceExpr.plot = _plot_sequence
SequenceExpr.hist = _hist_sequence

