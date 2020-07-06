#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import os
import logging

from cupid.errors import CupidError
from mars.utils import parse_readable_size
from mars.dataframe.operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from mars.serialize import StringField, Int64Field, SeriesField, DictField, BoolField, ListField
from mars.dataframe.utils import parse_index, standardize_range_index
try:
    from mars.optimizes.runtime.optimizers.dataframe import DataFrameRuntimeOptimizeRule
except ImportError:
    DataFrameRuntimeOptimizeRule = object

from ...df.backends.odpssql.types import odps_type_to_df_type
from ...df.backends.pd.types import df_type_to_np_type
from ...utils import to_str

logger = logging.getLogger('mars.worker')

CHUNK_LIMIT = 32 * 1024 ** 2
MAX_CHUNK_SIZE = 512 * 1024 ** 2


class DataFrameReadTable(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123450

    _odps_params = DictField('odps_params')
    _table_name = StringField('table_name')
    _partition_spec = StringField('partition_spec')
    _dtypes = SeriesField('dtypes')
    _add_offset = BoolField('add_offset')
    _columns = ListField('columns')

    def __init__(self, odps_params=None, table_name=None, partition_spec=None, columns=None,
                 dtypes=None, sparse=None, add_offset=True, **kw):
        super(DataFrameReadTable, self).__init__(_odps_params=odps_params, _table_name=table_name,
                                                 _partition_spec=partition_spec, _columns=columns,
                                                 _dtypes=dtypes, _sparse=sparse, _add_offset=add_offset,
                                                 _object_type=ObjectType.dataframe, **kw)

    @property
    def retryable(self):
        return False

    @property
    def odps_params(self):
        return self._odps_params

    @property
    def table_name(self):
        return self._table_name

    @property
    def partition(self):
        return getattr(self, '_partition_spec', None)

    @property
    def columns(self):
        return self._columns

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def add_offset(self):
        return self._add_offset

    def __call__(self, shape, chunk_bytes=None):
        import numpy as np
        import pandas as pd

        if np.isnan(shape[0]):
            index_value = parse_index(pd.RangeIndex(0))
        else:
            index_value = parse_index(pd.RangeIndex(shape[0]))
        columns_value = parse_index(self.dtypes.index, store_data=True)
        return self.new_dataframe(None, shape, dtypes=self.dtypes, index_value=index_value,
                                  columns_value=columns_value, chunk_bytes=chunk_bytes)

    @classmethod
    def tile(cls, op):
        import numpy as np
        import pandas as pd
        from odps import ODPS
        from odps.accounts import BearerTokenAccount
        from cupid import CupidSession, context

        bearer_token = context().get_bearer_token()
        account = BearerTokenAccount(bearer_token)
        project = os.environ.get('ODPS_PROJECT_NAME', None)
        odps_params = op.odps_params.copy()
        if project:
            odps_params['project'] = project
        o = ODPS(None, None, account=account, **odps_params)
        cupid_session = CupidSession(o)

        df = op.outputs[0]
        split_size = df.extra_params.chunk_bytes or CHUNK_LIMIT

        data_src = o.get_table(op.table_name)
        if op.partition is not None:
            data_src = data_src.get_partition(op.partition)

        logger.debug('Start creating download session from cupid.')
        while True:
            try:
                download_session = cupid_session.create_download_session(
                    data_src, split_size=split_size, columns=op.columns)
                break
            except CupidError:
                logger.debug('The number of splits exceeds 100000, split_size is {}'.format(split_size))
                if split_size >= MAX_CHUNK_SIZE:
                    raise
                else:
                    split_size *= 2

        logger.debug('%s table splits have been created.', str(len(download_session.splits)))

        out_chunks = []
        # Ignore add_offset at this time.
        op._add_offset = False

        for idx, split in enumerate(download_session.splits):
            chunk_op = DataFrameReadTableSplit(cupid_handle=to_str(split.handle),
                                               split_index=split.split_index,
                                               split_file_start=split.split_file_start,
                                               split_file_end=split.split_file_end,
                                               schema_file_start=split.schema_file_start,
                                               schema_file_end=split.schema_file_end,
                                               add_offset=op.add_offset, dtypes=op.dtypes,
                                               sparse=op.sparse)
            # the chunk shape is unknown
            index_value = parse_index(pd.RangeIndex(0))
            columns_value = parse_index(df.dtypes.index, store_data=True)
            out_chunk = chunk_op.new_chunk(None, shape=(np.nan, df.shape[1]), dtypes=op.dtypes,
                                               index_value=index_value, columns_value=columns_value,
                                               index=(idx, 0))
            out_chunks.append(out_chunk)

        if op.add_offset:
            out_chunks = standardize_range_index(out_chunks)

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (df.shape[1],))
        return new_op.new_dataframes(None, shape=df.shape, dtypes=op.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)


class DataFrameReadTableSplit(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123451

    _cupid_handle = StringField('cupid_handle')
    _split_index = Int64Field('split_index')
    _split_file_start = Int64Field('split_file_start')
    _split_file_end = Int64Field('split_file_end')
    _schema_file_start = Int64Field('schema_file_start')
    _schema_file_end = Int64Field('schema_file_end')
    _dtypes = SeriesField('dtypes')
    _nrows = Int64Field('nrows')

    def __init__(self, cupid_handle=None, split_index=None, split_file_start=None, split_file_end=None,
                 schema_file_start=None, schema_file_end=None, nrows=None, dtypes=None,
                 sparse=None, **kw):
        super(DataFrameReadTableSplit, self).__init__(_cupid_handle=cupid_handle, _split_index=split_index,
                                                      _split_file_start=split_file_start,
                                                      _split_file_end=split_file_end,
                                                      _schema_file_start=schema_file_start,
                                                      _schema_file_end=schema_file_end,
                                                      _nrows=nrows, _dtypes=dtypes, _sparse=sparse,
                                                      _object_type=ObjectType.dataframe, **kw)

    @property
    def retryable(self):
        return False

    @property
    def output_limit(self):
        return 1

    @property
    def cupid_handle(self):
        return self._cupid_handle

    @property
    def split_index(self):
        return self._split_index

    @property
    def split_file_start(self):
        return self._split_file_start

    @property
    def split_file_end(self):
        return self._split_file_end

    @property
    def schema_file_start(self):
        return self._schema_file_start

    @property
    def schema_file_end(self):
        return self._schema_file_end

    @property
    def nrows(self):
        return self._nrows

    @property
    def dtypes(self):
        return self._dtypes

    @classmethod
    def execute(cls, ctx, op):
        import pyarrow as pa
        from cupid.io.table import TableSplit

        tsp = TableSplit(
            _handle=op.cupid_handle,
            _split_index=op.split_index,
            _split_file_start=op.split_file_start,
            _split_file_end=op.split_file_end,
            _schema_file_start=op.schema_file_start,
            _schema_file_end=op.schema_file_end,
        )
        logger.debug('Read split table, split index: %s', op.split_index)
        reader = tsp.open_arrow_reader()
        if op.nrows is not None:
            nrows = 0
            batches = []
            while nrows < op.nrows:
                try:
                    batch = reader.read_next_batch()
                    nrows += batch.num_rows
                    batches.append(batch)
                except StopIteration:
                    break
            logger.debug('Read %s rows of this split.', op.nrows)
            data = pa.Table.from_batches(batches).to_pandas()[:op.nrows]
        else:
            arrow_table = reader.read_all()
            data = arrow_table.to_pandas()
        logger.debug("Read data size is %s", data.memory_usage(deep=True).sum())
        logger.debug('Read split table finished, split index: %s', op.split_index)
        ctx[op.outputs[0].key] = data


def read_odps_table(table, shape, partition=None, sparse=False, chunk_bytes=None,
                    columns=None, odps_params=None, add_offset=False):
    import pandas as pd

    if chunk_bytes is not None:
        chunk_bytes = int(parse_readable_size(chunk_bytes)[0])
    table_name = '%s.%s' % (table.project.name, table.name)
    table_columns = table.schema.names
    table_types = table.schema.types
    df_types = [df_type_to_np_type(odps_type_to_df_type(type)) for type in table_types]
    dtypes = pd.Series(df_types, index=table_columns)

    op = DataFrameReadTable(odps_params=odps_params, table_name=table_name, partition_spec=partition,
                            dtypes=dtypes, sparse=sparse, add_offset=add_offset, columns=columns)
    return op(shape, chunk_bytes=chunk_bytes)


class ReadODPSTableRule(DataFrameRuntimeOptimizeRule):
    @staticmethod
    def match(chunk, graph, keys):
        from mars.dataframe.indexing.iloc import DataFrameIlocGetItem

        op = chunk.op
        inputs = graph.predecessors(chunk)
        if len(inputs) == 1 and isinstance(op, DataFrameIlocGetItem) and \
                op.is_head() and isinstance(inputs[0].op, DataFrameReadTableSplit) and \
                inputs[0].key not in keys:
            return True
        return False

    @staticmethod
    def apply(chunk, graph, keys):
        read_table_chunk = graph.predecessors(chunk)[0]
        nrows = read_table_chunk.op.nrows or 0
        head = chunk.op.indexes[0].stop
        # delete read_table from graph
        graph.remove_node(read_table_chunk)

        head_read_table_chunk_op = read_table_chunk.op.copy().reset_key()
        head_read_table_chunk_op._nrows = max(nrows, head)
        head_read_table_chunk_params = read_table_chunk.params
        head_read_table_chunk_params['_key'] = chunk.key
        head_read_table_chunk = head_read_table_chunk_op.new_chunk(
            read_table_chunk.inputs, kws=[head_read_table_chunk_params]).data
        graph.add_node(head_read_table_chunk)

        for succ in list(graph.iter_successors(chunk)):
            succ_inputs = succ.inputs
            new_succ_inputs = []
            for succ_input in succ_inputs:
                if succ_input is chunk:
                    new_succ_inputs.append(head_read_table_chunk)
                else:
                    new_succ_inputs.append(succ_input)
            succ.inputs = new_succ_inputs
            graph.add_edge(head_read_table_chunk, succ)

        graph.remove_node(chunk)
