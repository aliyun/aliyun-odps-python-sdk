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

from ...compat import Enum, six
from ...errors import DependencyNotInstalledError
from ..expr.expressions import CollectionExpr, SequenceExpr, Expr
from ..expr.dynamic import DynamicMixin
from ..types import is_number


class PlottingCore(Enum):
    PANDAS = 'pandas'


def _plot_pandas(df, method='plot', **kwargs):
    x_label, y_label = kwargs.pop('xlabel', None), kwargs.pop('ylabel', None)

    x_label_size, y_label_size = kwargs.pop('xlabelsize', None), kwargs.pop('ylabelsize', None)
    label_size = kwargs.pop('labelsize', None)
    x_label_size = x_label_size or label_size
    y_label_size = y_label_size or label_size

    title_size = kwargs.pop('titlesize', None)
    title = kwargs.pop('title', None)

    annotate = kwargs.pop('annotate', None)
    x_annotate_scale = kwargs.pop('xannotatescale', 1.0)
    y_annotate_scale = kwargs.pop('yannotatescale', 1.0)

    fig = getattr(df, method)(**kwargs)

    import numpy as np
    if isinstance(fig, np.ndarray):
        figs = fig
        fig = fig[0]
    else:
        figs = [fig, ]

    if x_label:
        if x_label_size:
            fig.set_xlabel(x_label, fontsize=x_label_size)
        else:
            fig.set_xlabel(x_label)
    if y_label:
        if y_label_size:
            fig.set_ylabel(y_label, fontsize=y_label_size)
        else:
            fig.set_ylabel(y_label)

    if title_size:
        fig.title.set_fontsize(title_size)
    if title:
        fig.set_title(title)

    if annotate:
        for ax in figs:
            for p in ax.patches:
                ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),
                            xytext=(p.get_x() * (x_annotate_scale - 1),
                                    p.get_height() * (y_annotate_scale - 1)),
                            textcoords='offset points')
            for l in ax.lines:
                xs, ys = l.get_data()
                for x, y in zip(xs, ys):
                    ax.annotate(str(y), (x, y),
                                xytext=(x * (x_annotate_scale - 1), y * (y_annotate_scale - 1)),
                                textcoords='offset points')

    return fig


def _hist_pandas(df, **kwargs):
    return _plot_pandas(df, method='hist', **kwargs)


def _boxplot_pandas(df, **kwargs):
    return _plot_pandas(df, method='boxplot', **kwargs)


def _plot_sequence(expr, kind='line', use_cache=None, **kwargs):
    try:
        import pandas as pd
    except ImportError:
        raise DependencyNotInstalledError('plot requires for pandas')

    series = expr.to_pandas(use_cache=use_cache)
    for k in list(kwargs.keys()):
        v = kwargs[k]
        if v is not None and isinstance(v, (CollectionExpr, SequenceExpr)):
            kwargs[k] = v.to_pandas()
    return _plot_pandas(series, kind=kind, **kwargs)


def _hist_sequence(expr, use_cache=None, **kwargs):
    try:
        import pandas as pd
    except ImportError:
        raise DependencyNotInstalledError('plot requires for pandas')

    series = expr.to_pandas(use_cache=use_cache)
    return _hist_pandas(series, **kwargs)


def _plot_collection(expr, x=None, y=None, kind='line', **kwargs):
    try:
        import pandas as pd
    except ImportError:
        raise DependencyNotInstalledError('plot requires for pandas')

    if isinstance(expr, DynamicMixin):
        rf = expr.head(1)
        columns = rf.columns
    else:
        columns = expr.dtypes

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
        for col in columns:
            if col.name == x_name or col.name == y_name:
                continue
            elif is_number(col.type):
                fields.append(col.name)

    to_replace = dict()
    for k, v in six.iteritems(kwargs):
        if isinstance(v, Expr):
            fields.append(v)
            to_replace[k] = v.name

    df = expr[fields].to_pandas()

    if x_name is not None:
        kwargs['x'] = x_name
    if y is not None:
        kwargs['y'] = y_name

    for k, v in six.iteritems(to_replace):
        kwargs[k] = df[v]

    _plot_func = kwargs.pop('plot_func', _plot_pandas)
    return _plot_func(df, kind=kind, **kwargs)


def _hist_collection(expr, **kwargs):
    try:
        import pandas as pd
    except ImportError:
        raise DependencyNotInstalledError('plot requires for pandas')

    fields = []
    column = kwargs.get('column')
    if isinstance(column, six.string_types):
        fields.append(column)

    to_replace = dict()
    for k, v in six.iteritems(kwargs):
        if isinstance(v, Expr):
            fields.append(v)
            to_replace[k] = v.name

    if fields:
        expr = expr[fields]

    df = expr.to_pandas()

    for k, v in six.iteritems(to_replace):
        kwargs[k] = df[v]

    _plot_func = kwargs.pop('plot_func', _hist_pandas)
    return _plot_func(df, **kwargs)


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
        fields = fields.union([column])

    by = kwargs.get('by')
    if isinstance(by, six.string_types):
        fields.add(by)
    elif by is not None:
        fields = fields.union([by])

    fields = list(fields)
    to_replace = dict()
    for k, v in six.iteritems(kwargs):
        if isinstance(v, Expr):
            fields.append(v)
            to_replace[k] = v.name

    if fields:
        expr = expr[fields]

    df = expr.to_pandas()

    for k, v in six.iteritems(to_replace):
        kwargs[k] = df[v]

    _plot_func = kwargs.pop('plot_func', _boxplot_pandas)
    return _plot_func(df, **kwargs)


CollectionExpr.plot = _plot_collection
CollectionExpr.hist = _hist_collection
CollectionExpr.boxplot = _boxplot_collection
SequenceExpr.plot = _plot_sequence
SequenceExpr.hist = _hist_sequence

try:
    from pandas.tools.plotting import plot_frame, hist_frame, boxplot, \
        plot_series, hist_series

    _plot_collection.__doc__ = plot_frame.__doc__
    _hist_collection.__doc__ = hist_frame.__doc__
    _boxplot_collection.__doc__ = boxplot.__doc__
    _plot_sequence.__doc__ = plot_series.__doc__
    _hist_sequence.__doc__ = hist_series.__doc__
except ImportError:
    pass

__all__ = ()
