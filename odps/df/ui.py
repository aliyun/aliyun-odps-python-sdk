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

import math
import decimal
import os

from . import DataFrame, Scalar
from ..ui.common import build_trait
from ..console import get_widgets, in_ipython_frontend, is_widgets_available
from ..utils import to_text, init_progress_ui

MAX_TABLE_FETCH_SIZE = 10

try:
    # None unless IPython was already imported and ipywidgets is installed;
    # skip the classes below instead of failing on `widgets.DOMWidget`.
    widgets = get_widgets()
    if widgets is None:
        raise ImportError
    from IPython.display import display
    from traitlets import Unicode, Dict, Int, Bool
except ImportError:
    DFViewWidget = None
    DFViewMixin = None
    _rv = None
    _rv_list = None
    _rv_table = None
else:
    def _rv(v):
        if isinstance(v, decimal.Decimal):
            return float(v)
        if isinstance(v, (float, int)):
            return v
        # numpy scalars: emit native Python numbers so charts get numbers,
        # not their string rendering.
        dtype = getattr(v, "dtype", None)
        # numpy temporal scalars `.item()` to datetime/timedelta, which the
        # synchronized Dict trait cannot serialize; render them as text.
        if dtype is not None and getattr(dtype, "kind", "") in ("M", "m"):
            return to_text(v)
        if hasattr(v, "item") and dtype is not None:
            try:
                item = v.item()
            except (ValueError, TypeError):
                pass
            else:
                if item is None or isinstance(item, (bool, int, float, str)):
                    return item
        return to_text(v)

    def _rv_list(l):
        return [_rv(v) for v in l]

    def _rv_table(tb):
        return [_rv_list(r) for r in tb]

    _NUMERIC_TYPE_NAMES = frozenset(
        [
            # odps-layer names
            "bigint", "int", "smallint", "tinyint", "double", "float",
            # df-layer names
            "int8", "int16", "int32", "int64", "float32", "float64",
        ]
    )
    # Aggregations that are only defined for numeric columns; the names are the
    # ones registered by odps.df.expr.reduction. min/max/count/nunique work for
    # every type, so they are never rewritten.
    _NUMERIC_ONLY_AGG_METHODS = frozenset(["sum", "mean", "std", "var", "median"])

    # Traits a chart request may write its records into.
    _CHART_RECORD_TRAITS = frozenset(
        ["bar_chart_records", "pie_chart_records", "line_chart_records",
         "scatter_chart_records"]
    )

    def _is_numeric_type(typo):
        name = typo.name
        return name in _NUMERIC_TYPE_NAMES or name.startswith("decimal")

    class DFViewMixin:
        @property
        def mock_ui(self):
            cls = type(self)
            if not hasattr(cls, '_mock_ui'):
                cls._mock_ui = init_progress_ui(mock=True)
            return cls._mock_ui

        @staticmethod
        def _render_results(df, **kw):

            return dict(columns=[to_text(c) for c in df.columns], data=_rv_table(df.values), **kw)

        @staticmethod
        def _render_graph(pd_df, group_cols, key_cols, **kw):
            def _to_columnar_dict(pd_df):
                vdict = dict((_rv(c), _rv_list(pd_df[c].values)) for c in pd_df.columns)
                vdict['$count'] = len(pd_df)
                return vdict

            if key_cols:
                distinct_df = pd_df[list(key_cols)].drop_duplicates()
                # Mixed byte/str key values raise TypeError when sorted; the old
                # implementation normalized them through a `to_text` UDF before
                # this point, so do it here on the materialized frame.
                for col in distinct_df.columns:
                    if distinct_df[col].dtype == object:
                        distinct_df[col] = distinct_df[col].map(
                            lambda v: to_text(v) if isinstance(v, (bytes, bytearray)) else v
                        )
                key_dfs = _rv_table(distinct_df.sort_values(list(key_cols)).values)
            else:
                key_dfs = None

            groups, sub_dfs = [], []
            # Key columns stay in the payload even when a column plays both
            # roles: the front end aligns each group's series by key value.
            drop_cols = [col for col in (group_cols or []) if col not in set(key_cols or [])]
            if group_cols:
                for gp, cols in pd_df.groupby(group_cols):
                    g = [gp] if not isinstance(gp, (list, tuple, set, slice)) else gp
                    groups.append(_rv_list(list(g)))
                    sub_dfs.append(_to_columnar_dict(cols.drop(drop_cols, axis=1)))
            else:
                groups = None
                sub_dfs.append(_to_columnar_dict(pd_df))
            return dict(groups=groups, data=sub_dfs, keys=key_dfs,
                        group_cols=list(group_cols or []), key_cols=list(key_cols or []), **kw)

        def _handle_fetch_table(self, content, _):
            df = self.df.to_pandas(ui=self.mock_ui) if isinstance(self.df, DataFrame) else self.df
            start_pos, stop_pos, page = 0, len(df), 0
            if len(df) > MAX_TABLE_FETCH_SIZE:
                page = content.get('page', 0)
                start_pos, stop_pos = page * MAX_TABLE_FETCH_SIZE, min((page + 1) * MAX_TABLE_FETCH_SIZE, len(df))
            result_dict = self._render_results(df[start_pos:stop_pos], page=page, size=len(df))
            if page is not None:
                result_dict['pages'] = math.ceil(len(df) * 1.0 / MAX_TABLE_FETCH_SIZE)
            self.table_records = result_dict

        def _handle_aggregate_graph(self, content, _):
            # Roles of the three field families, as in the classic notebook
            # front end:
            #   groups: columns whose values split the result into series
            #   keys:   columns used as the x-axis categories
            #   values: measures; a dict maps column -> [methods] and aggregates
            #           with those methods, a list selects the columns as-is
            #           (no aggregation, used by the line and scatter charts)
            groups = content.get('groups') or list()
            keys = content.get('keys') or list()
            if isinstance(keys, str):
                keys = [keys]

            schema = self.df.schema if isinstance(self.df, DataFrame) else []
            numeric_cols = [col.name for col in schema if _is_numeric_type(col.type)]
            string_cols = [col.name for col in schema if col.type.name == 'string']

            values = content.get('values')
            if not values:
                # No explicit config: pick type-safe defaults.
                if numeric_cols:
                    values = dict((name, ['sum']) for name in numeric_cols)
                elif schema:
                    values = dict((col.name, ['count']) for col in schema)
                else:
                    return
                if not keys and string_cols:
                    keys = [string_cols[0]]

            if isinstance(values, dict):
                group_by_keys = list(dict.fromkeys(groups + keys))

                def _gen_agg_func(df):
                    try:
                        available = set(df.schema.names)
                    except AttributeError:
                        available = None
                    agg_funcs = dict()
                    for field, methods in values.items():
                        if available is not None and field not in available:
                            continue
                        if isinstance(methods, str):
                            methods = [methods]
                        col_type = None
                        for col in (schema or []):
                            if col.name == field:
                                col_type = col.type
                                break
                        for method in methods:
                            if not hasattr(df[field], method):
                                raise ValueError(
                                    'Unsupported aggregation {} for column {}'.format(
                                        method, field))
                            # sum/mean/std/var/median are undefined for non-numeric
                            # columns, fall back to count so the chart still renders.
                            if (col_type is not None and method in _NUMERIC_ONLY_AGG_METHODS
                                    and not _is_numeric_type(col_type)):
                                method = 'count'
                            func_key = '{}__{}'.format(field, method)
                            agg_funcs[func_key] = getattr(df[field], method)()
                    return agg_funcs

                if group_by_keys:
                    # Build the projection explicitly so grouping keeps the
                    # schema order. A `.map(to_text)` UDF is not used here:
                    # these columns are already strings and the lambda cannot
                    # be shipped to the cluster.
                    key_set = set(group_by_keys)
                    col_list = [self.df[col] for col in group_by_keys] + [
                        self.df[col.name]
                        for col in self.df.schema
                        if col.name not in key_set
                    ]
                    trans_df = self.df.select(col_list)
                    agg_funcs = _gen_agg_func(trans_df)
                    if not agg_funcs:
                        raise ValueError(
                            'No aggregatable value column in {}'.format(sorted(values)))
                    pd_results = trans_df.groupby(group_by_keys).agg(**agg_funcs) \
                        .to_pandas(ui=self.mock_ui)
                else:
                    group_col = '__group_col__'
                    augment_df = self.df[Scalar(1).rename(group_col), self.df]
                    agg_funcs = _gen_agg_func(augment_df)
                    if not agg_funcs:
                        raise ValueError(
                            'No aggregatable value column in {}'.format(sorted(values)))
                    pd_results = augment_df.groupby([group_col]).agg(**agg_funcs) \
                        .exclude(group_col).to_pandas(ui=self.mock_ui)
            else:
                pd_results = self.df.select(list(dict.fromkeys(groups + keys + values))).to_pandas(ui=self.mock_ui)

            result_dict = self._render_graph(pd_results, groups, keys)
            target = content.get('target') or ''
            if target not in _CHART_RECORD_TRAITS:
                raise ValueError('Unknown chart target {}'.format(target))
            setattr(self, target, result_dict)

    class DFViewWidget(DFViewMixin, widgets.DOMWidget):
        _view_name = build_trait(Unicode, 'DFView', sync=True)
        _view_module = build_trait(Unicode, 'pyodps/df-view', sync=True)
        # Must match WIDGET_MODULE_VERSION in odps/static/ui/src/extension.js,
        # otherwise the browser cannot resolve this view class.
        _view_module_version = build_trait(Unicode, '2.0.0', sync=True)
        start_sign = build_trait(Int, False, sync=True)
        error_sign = build_trait(Int, False, sync=True)
        # Message of the last failed chart request, shown in place of the chart.
        last_error = build_trait(Unicode, '', sync=True)
        closed = build_trait(Bool, False, sync=True)
        table_records = build_trait(Dict, sync=True)
        bar_chart_records = build_trait(Dict, sync=True)
        pie_chart_records = build_trait(Dict, sync=True)
        line_chart_records = build_trait(Dict, sync=True)
        scatter_chart_records = build_trait(Dict, sync=True)

        def __init__(self, df, **kwargs):
            super(DFViewWidget, self).__init__(**kwargs)
            self.df = df
            self.on_msg(self._handle_msgs)

        def _handle_msgs(self, _, content, buffers):
            try:
                if isinstance(content, str):
                    import json as _json
                    content = _json.loads(content)
                action = content.get('action', '')
                if action == 'start_widget':
                    self.start_sign = (self.start_sign + 1) % 2
                    self._handle_fetch_table(content, buffers)
                elif action == 'fetch_table':
                    self._handle_fetch_table(content, buffers)
                elif action == 'aggregate_graph':
                    self._handle_aggregate_graph(content, buffers)
                elif action == 'close_widget':
                    self.closed = True
                self.last_error = ''
            except Exception as e:
                self.last_error = str(e)
                self.error_sign = (self.error_sign + 1) % 2
                raise


def show_df_widget(df, **kwargs):
    # Bail out before touching widget state: outside an IPython front end there
    # is no view to display, and constructing a DOMWidget would register a
    # kernel comm even when only the HTML fallback is shown.
    if not in_ipython_frontend() or DFViewWidget is None:
        return
    if not is_widgets_available():
        return
    display(DFViewWidget(df, **kwargs))
