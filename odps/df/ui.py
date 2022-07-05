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

from __future__ import print_function

import math
import decimal
import os

from . import DataFrame, Scalar
from ..ui.common import build_trait
from ..console import in_ipython_frontend, is_widgets_available
from ..compat import six, long_type
from ..utils import to_text, init_progress_ui

MAX_TABLE_FETCH_SIZE = 10

try:
    import ipywidgets as widgets
    from IPython.display import display
    from traitlets import Unicode, Dict, Int
except ImportError:
    DFViewWidget = None
    DFViewMixin = None
    _rv = None
    _rv_list = None
    _rv_table = None
else:
    def _rv(v):
        if isinstance(v, (float, int, long_type)):
            return v
        elif isinstance(v, decimal.Decimal):
            return float(v)
        else:
            return to_text(v)

    def _rv_list(l):
        return [_rv(v) for v in l]

    def _rv_table(tb):
        return [_rv_list(r) for r in tb]

    class DFViewMixin(object):
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
                key_dfs = _rv_table(distinct_df.sort_values(key_cols).values)
            else:
                key_dfs = None

            groups, sub_dfs = [], []
            if group_cols:
                for gp, cols in pd_df.groupby(group_cols):
                    g = [gp] if not isinstance(gp, (list, tuple, set, slice)) else gp
                    groups.append(_rv_list(list(g)))
                    sub_dfs.append(_to_columnar_dict(cols.drop(group_cols, axis=1)))
            else:
                groups = None
                sub_dfs.append(_to_columnar_dict(pd_df))
            return dict(groups=groups, data=sub_dfs, keys=key_dfs, **kw)

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
            groups = content.get('groups') or list()
            keys = content.get('keys') or list()
            if isinstance(keys, six.string_types):
                keys = [keys]

            values = content.get('values')
            if not values:
                return

            if isinstance(values, dict):
                group_by_keys = list(set(groups + keys))

                def _gen_agg_func(df):
                    agg_funcs = dict()
                    for field, methods in six.iteritems(values):
                        for method in methods:
                            func_key = u'{}__{}'.format(field, method)
                            agg_funcs[func_key] = getattr(df[field], method)()
                    return agg_funcs

                if group_by_keys:
                    string_keys = [col.name for col in self.df.schema
                                   if col.name in group_by_keys and col.type.name == 'string']
                    other_keys = [col.name for col in self.df.schema
                                  if col.name not in group_by_keys or col.name not in string_keys]
                    col_list = [self.df[col].map(lambda v: to_text(v)) for col in string_keys] +\
                               [self.df[col] for col in other_keys]
                    trans_df = self.df.select(col_list)
                    pd_results = trans_df.groupby(group_by_keys).agg(**_gen_agg_func(trans_df)) \
                        .to_pandas(ui=self.mock_ui)
                else:
                    group_col = '__group_col__'
                    augment_df = self.df[Scalar(1).rename(group_col), self.df]
                    pd_results = augment_df.groupby([group_col]).agg(**_gen_agg_func(augment_df)) \
                        .exclude(group_col).to_pandas(ui=self.mock_ui)
            else:
                pd_results = self.df.select(list(set(groups + keys + values))).to_pandas(ui=self.mock_ui)

            result_dict = self._render_graph(pd_results, groups, keys)
            setattr(self, content.get('target'), result_dict)

    class DFViewWidget(DFViewMixin, widgets.DOMWidget):
        _view_name = build_trait(Unicode, 'DFView', sync=True)
        _view_module = build_trait(Unicode, 'pyodps/df-view', sync=True)
        start_sign = build_trait(Int, False, sync=True)
        error_sign = build_trait(Int, False, sync=True)
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
                action = content.get('action', '')
                if action == 'start_widget':
                    self.start_sign = (self.start_sign + 1) % 2
                elif action == 'fetch_table':
                    self._handle_fetch_table(content, buffers)
                elif action == 'aggregate_graph':
                    self._handle_aggregate_graph(content, buffers)
            except:
                self.error_sign = (self.error_sign + 1) % 2
                raise


def show_df_widget(df, **kwargs):
    if in_ipython_frontend() and DFViewWidget:
        widget = DFViewWidget(df, **kwargs)
        if is_widgets_available():
            display(widget)
