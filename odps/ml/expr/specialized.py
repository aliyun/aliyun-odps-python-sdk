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

from . import exporters
from .core import AlgoCollectionExpr
from ..utils import ML_ARG_PREFIX


_MERGE_DATA_ARGS = """
inputTableNames outputTableName selectedColNamesList inputPartitionsInfoList outputPartition
autoRenameCol
""".split()
_MERGE_DATA_EXPORTERS = {
    "outputTableName": lambda e: exporters.get_output_table_name(e, "output"),
    "inputTableNames": lambda e: e._export_input_table_names(),
    "inputPartitionsInfoList": lambda e: e._export_input_table_partitions(),
    "selectedColNamesList": lambda e: e._export_selected_cols(),
}
_MERGE_DATA_META = {
    'xflowName': 'AppendColumns',
    'xflowProjectName': 'algo_public',
}


class BaseMergeDataCollectionExpr(AlgoCollectionExpr):
    __slots__ = '_selected_merge_cols', '_excluded_merge_cols'
    node_name = 'MergeData'
    _algo = 'MergeData'
    _exported = _MERGE_DATA_ARGS
    _exporters = _MERGE_DATA_EXPORTERS
    algo_meta = _MERGE_DATA_META

    def _init(self, *args, **kwargs):
        self._init_attr('_selected_merge_cols', kwargs.pop('selected_cols', dict()))
        self._init_attr('_excluded_merge_cols', kwargs.pop('excluded_cols', dict()))

        super(BaseMergeDataCollectionExpr, self)._init(*args, **kwargs)

        self._init_attr('_params', dict())
        self._init_attr('_engine_kw', dict())

    def _iter_inputs(self):
        for name, df in self.iter_args():
            if name.startswith(ML_ARG_PREFIX):
                yield name[len(ML_ARG_PREFIX):], df

    def _export_input_table_names(self):
        tables = []
        for name, expr in self._iter_inputs():
            tn = exporters.get_input_table_name(self, name)
            if expr is not None and not tn:
                return None
            tables.append(tn if tn else '')
        return ','.join(tables)

    def _export_input_table_partitions(self):
        parts = []
        for name, _ in self._iter_inputs():
            tn = exporters.get_input_table_partitions(self, name)
            parts.append(tn if tn else '')
        if all(not tn for tn in parts):
            return None
        return ','.join(parts)

    def _export_selected_cols(self):
        def fetch_ds_cols(seq, df):
            if seq in self._selected_merge_cols:
                return self._selected_merge_cols[seq]
            elif seq in self._excluded_merge_cols:
                return [f for f in df._ml_fields if f.role and f.name not in self._excluded_merge_cols[seq]]
            else:
                return [f for f in df._ml_fields if f.role]

        return ';'.join(','.join(f.name for f in fetch_ds_cols(seq, df_tuple[1]))
                        for seq, df_tuple in enumerate(self._iter_inputs()))


def build_merge_expr(num_inputs):
    inputs = [ML_ARG_PREFIX + 'input%d' % (idx + 1) for idx in range(num_inputs)]

    class MergeDataCollectionExpr(BaseMergeDataCollectionExpr):
        _args = tuple(inputs)

    return MergeDataCollectionExpr
