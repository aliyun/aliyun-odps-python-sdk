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

from __future__ import print_function

import math

from . import DataFrame, Scalar
from ..ui.common import build_trait
from ..console import in_ipython_frontend, is_widgets_available
from ..compat import six

MAX_TABLE_FETCH_SIZE = 10

try:
    import ipywidgets as widgets
    from IPython.display import display
    from traitlets import Unicode, Dict, Int
except ImportError:
    DFViewWidget = None
else:
    class DFViewWidget(widgets.DOMWidget):
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

        @staticmethod
        def _render_results(df, **kw):
            return dict(columns=[c for c in df.columns], data=[[v for v in d] for d in df.values], **kw)

        @staticmethod
        def _render_graph(pd_df, group_cols, key_cols, **kw):
            def _to_columnar_dict(pd_df):
                vdict = dict((c, pd_df[c].values.tolist()) for c in pd_df.columns)
                vdict['$count'] = len(pd_df)
                return vdict

            if key_cols:
                distinct_df = pd_df[list(key_cols)].drop_duplicates()
                key_dfs = distinct_df.sort_values(key_cols).values.tolist()
            else:
                key_dfs = None

            groups, sub_dfs = [], []
            if group_cols:
                for gp, cols in pd_df.groupby(group_cols):
                    g = [gp] if not isinstance(gp, (list, tuple, set, slice)) else gp
                    groups.append(list(g))
                    sub_dfs.append(_to_columnar_dict(cols.drop(group_cols, axis=1)))
            else:
                groups = None
                sub_dfs.append(_to_columnar_dict(pd_df))
            return dict(groups=groups, data=sub_dfs, keys=key_dfs, **kw)

        def _handle_msgs(self, _, content, buffers):
            try:
                self._actual_handle_msgs(content, buffers)
            except:
                self.error_sign = (self.error_sign + 1) % 2
                raise

        def _actual_handle_msgs(self, content, buffers):
            action = content.get('action', '')
            if action == 'start_widget':
                self.start_sign = (self.start_sign + 1) % 2
            elif action == 'fetch_table':
                df = self.df.to_pandas() if isinstance(self.df, DataFrame) else self.df
                start_pos, stop_pos, page = 0, len(df), 0
                if len(df) > MAX_TABLE_FETCH_SIZE:
                    page = content.get('page', 0)
                    start_pos, stop_pos = page * MAX_TABLE_FETCH_SIZE, min((page + 1) * MAX_TABLE_FETCH_SIZE, len(df))
                result_dict = self._render_results(df[start_pos:stop_pos], page=page, size=len(df))
                if page is not None:
                    result_dict['pages'] = math.ceil(len(df) * 1.0 / MAX_TABLE_FETCH_SIZE)
                self.table_records = result_dict
            elif action == 'aggregate_graph':
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
                                func_key = '{}__{}'.format(field, method)
                                agg_funcs[func_key] = getattr(df[field], method)()
                        return agg_funcs

                    if group_by_keys:
                        pd_results = self.df.groupby(group_by_keys).agg(**_gen_agg_func(self.df)).to_pandas()
                    else:
                        group_col = '__group_col__'
                        augment_df = self.df[Scalar(1).rename(group_col), self.df]
                        pd_results = augment_df.groupby([group_col]).agg(**_gen_agg_func(augment_df))\
                            .exclude(group_col).to_pandas()
                else:
                    pd_results = self.df.select(list(set(groups + keys + values))).to_pandas()

                result_dict = self._render_graph(pd_results, groups, keys)
                setattr(self, content.get('target'), result_dict)


def show_df_widget(df, **kwargs):
    if in_ipython_frontend() and DFViewWidget:
        widget = DFViewWidget(df, **kwargs)
        if is_widgets_available():
            display(widget)
