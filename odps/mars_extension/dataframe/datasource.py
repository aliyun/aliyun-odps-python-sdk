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

import logging

from mars.config import options
from mars.dataframe.operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from mars.serialize import StringField, Int64Field, SeriesField, DictField, BoolField
from mars.dataframe.utils import parse_index

from ...df.backends.odpssql.types import odps_type_to_df_type
from ...df.backends.pd.types import df_type_to_np_type
from ...utils import to_str

logger = logging.getLogger('mars.worker')


class DataFrameReadTable(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123450

    _odps_params = DictField('odps_params')
    _table_name = StringField('table_name')
    _partition_spec = StringField('partition_spec')
    _dtypes = SeriesField('dtypes')
    _add_offset = BoolField('add_offset')

    def __init__(self, odps_params=None, table_name=None, partition_spec=None, dtypes=None,
                 sparse=None, add_offset=True, **kw):
        super(DataFrameReadTable, self).__init__(_odps_params=odps_params, _table_name=table_name,
                                                 _partition_spec=partition_spec,
                                                 _dtypes=dtypes, _sparse=sparse, _add_offset=add_offset,
                                                 _object_type=ObjectType.dataframe, **kw)

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
    def dtypes(self):
        return self._dtypes

    @property
    def add_offset(self):
        return self._add_offset

    def __call__(self, shape, chunk_store_limit=None):
        import numpy as np
        import pandas as pd

        if np.isnan(shape[0]):
            index_value = parse_index(pd.RangeIndex(0))
        else:
            index_value = parse_index(pd.RangeIndex(shape[0]))
        columns_value = parse_index(self.dtypes.index, store_data=True)
        return self.new_dataframe(None, shape, dtypes=self.dtypes, index_value=index_value,
                                  columns_value=columns_value, chunk_store_limit=chunk_store_limit)

    @classmethod
    def tile(cls, op):
        import numpy as np
        import pandas as pd
        from odps import ODPS
        from odps.accounts import BearerTokenAccount
        from cupid import CupidSession, context

        bearer_token = context().get_bearer_token()
        account = BearerTokenAccount(bearer_token)
        o = ODPS(None, None, account=account, **op.odps_params)
        cupid_session = CupidSession(o)

        df = op.outputs[0]
        split_size = df.extra_params.chunk_store_limit or options.tensor.chunk_store_limit

        data_src = o.get_table(op.table_name)
        if op.partition is not None:
            data_src = data_src.get_partition(op.partition)

        logger.debug('Start creating download session from cupid.')
        download_session = cupid_session.create_download_session(data_src, split_size=split_size)
        logger.debug('%s table splits have been created.', str(len(download_session.splits)))

        out_chunks = []
        out_count_chunks = []
        for idx, split in enumerate(download_session.splits):
            chunk_op = DataFrameReadTableSplit(cupid_handle=to_str(split.handle),
                                               split_index=split.split_index,
                                               split_file_start=split.split_file_start,
                                               split_file_end=split.split_file_end,
                                               schema_file_start=split.schema_file_start,
                                               schema_file_end=split.schema_file_end,
                                               dtypes=op.dtypes, sparse=op.sparse)
            # the chunk shape is unknown
            index_value = parse_index(pd.RangeIndex(0))
            columns_value = parse_index(df.dtypes.index, store_data=True)
            out_chunk, out_count_chunk = chunk_op.new_chunks(None,
                                                             kws=[
                                                                 {'shape': (np.nan, df.shape[1]),
                                                                  'dtypes': op.dtypes,
                                                                  'index_value': index_value,
                                                                  'columns_value': columns_value,
                                                                  'index': (idx,)},
                                                                 {'shape': (1,),
                                                                  'index': (idx,)}
                                                             ])
            out_chunks.append(out_chunk)
            out_count_chunks.append(out_count_chunk)

        if op.add_offset:
            output_chunks = []
            for i, chunk in enumerate(out_chunks):
                if i == 0:
                    output_chunks.append(chunk)
                    continue
                counts = out_count_chunks[:i]
                inputs = [chunk] + counts
                output_chunk = DataFrameReadTableWithOffset(dtypes=chunk.dtypes).new_chunk(
                    inputs, shape=chunk.shape, index=chunk.index, dtypes=chunk.dtypes,
                    index_value=chunk.index_value, columns_value=chunk.columns_value)
                output_chunks.append(output_chunk)
        else:
            output_chunks = out_chunks

        new_op = op.copy()
        nsplits = ((np.nan,) * len(output_chunks), (df.shape[1],))
        return new_op.new_dataframes(None, shape=df.shape, dtypes=op.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns_value,
                                     chunks=output_chunks, nsplits=nsplits)


class DataFrameReadTableSplit(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123451

    _cupid_handle = StringField('cupid_handle')
    _split_index = Int64Field('split_index')
    _split_file_start = Int64Field('split_file_start')
    _split_file_end = Int64Field('split_file_end')
    _schema_file_start = Int64Field('schema_file_start')
    _schema_file_end = Int64Field('schema_file_end')
    _dtypes = SeriesField('dtypes')

    def __init__(self, cupid_handle=None, split_index=None, split_file_start=None, split_file_end=None,
                 schema_file_start=None, schema_file_end=None, dtypes=None, sparse=None, **kw):
        super(DataFrameReadTableSplit, self).__init__(_cupid_handle=cupid_handle, _split_index=split_index,
                                                      _split_file_start=split_file_start,
                                                      _split_file_end=split_file_end,
                                                      _schema_file_start=schema_file_start,
                                                      _schema_file_end=schema_file_end,
                                                      _dtypes=dtypes, _sparse=sparse,
                                                      _object_type=ObjectType.dataframe, **kw)

    @property
    def output_limit(self):
        return 2

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
    def dtypes(self):
        return self._dtypes

    @classmethod
    def execute(cls, ctx, op):
        import numpy as np
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
        data = reader.read_all().to_pandas()
        logger.debug('Read split table finished, split index: %s', op.split_index)
        count = np.array([data.shape[0]])
        data_chunk, count_chunk = op.outputs
        ctx[data_chunk.key] = data
        ctx[count_chunk.key] = count


class DataFrameReadTableWithOffset(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123453

    _dtypes = SeriesField('dtypes')

    def __init__(self, dtypes=None, **kw):
        super(DataFrameReadTableWithOffset, self).__init__(_dtypes=dtypes, _object_type=ObjectType.dataframe, **kw)

    @property
    def dtypes(self):
        return self._dtypes

    @classmethod
    def execute(cls, ctx, op):
        import numpy as np

        inputs = [ctx[inp.key] for inp in op.inputs]
        df = inputs[0]
        if len(inputs) > 1:
            offset = np.sum(inputs[1:])
        else:
            offset = 0
        new_df = df.copy()
        new_df.index = new_df.index + offset
        ctx[op.outputs[0].key] = new_df


def read_odps_table(table, shape, partition=None, sparse=False, chunk_store_limit=None,
                    odps_params=None, add_offset=True):
    import pandas as pd

    table_name = '%s.%s' % (table.project.name, table.name)
    table_columns = table.schema.names
    table_types = table.schema.types
    df_types = [df_type_to_np_type(odps_type_to_df_type(type)) for type in table_types]
    dtypes = pd.Series(df_types, index=table_columns)

    op = DataFrameReadTable(odps_params=odps_params, table_name=table_name, partition_spec=partition,
                            dtypes=dtypes, sparse=sparse, add_offset=add_offset)
    return op(shape, chunk_store_limit=chunk_store_limit)
