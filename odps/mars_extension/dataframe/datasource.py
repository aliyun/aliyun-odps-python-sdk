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

import numpy as np
import pandas as pd

from cupid.errors import CupidError
from mars.utils import parse_readable_size
from mars.serialize import StringField, Int64Field, SeriesField, DictField, BoolField, ListField
from mars.dataframe import ArrowStringDtype
from mars.dataframe.operands import DataFrameOperandMixin, DataFrameOperand
from mars.dataframe.utils import parse_index, standardize_range_index, arrow_table_to_pandas_dataframe
from mars.optimizes.runtime.dataframe import DataSourceHeadRule

from ...df.backends.odpssql.types import odps_type_to_df_type
from ...errors import ODPSError
from ...utils import to_str

logger = logging.getLogger(__name__)

READ_CHUNK_LIMIT = 16 * 1024 ** 2
MAX_CHUNK_SIZE = 512 * 1024 ** 2

ORC_COMPRESSION_RATIO = 5
STRING_FIELD_OVERHEAD = 50

try:
    from mars.core import OutputType
    _output_type_kw = dict(_output_types=[OutputType.dataframe])
except ImportError:
    from mars.dataframe.operands import ObjectType
    _output_type_kw = dict(_object_type=ObjectType.dataframe)


class DataFrameReadTable(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123450

    _odps_params = DictField('odps_params')
    _table_name = StringField('table_name')
    _partition_spec = StringField('partition_spec')
    _dtypes = SeriesField('dtypes')
    _add_offset = BoolField('add_offset')
    _columns = ListField('columns')
    _nrows = Int64Field('nrows')
    _use_arrow_dtype = BoolField('use_arrow_dtype')

    def __init__(self, odps_params=None, table_name=None, partition_spec=None,
                 columns=None, dtypes=None, nrows=None, sparse=None,
                 add_offset=True, use_arrow_dtype=None, **kw):
        kw.update(_output_type_kw)
        super(DataFrameReadTable, self).__init__(_odps_params=odps_params, _table_name=table_name,
                                                 _partition_spec=partition_spec, _columns=columns,
                                                 _dtypes=dtypes, _nrows=nrows, _sparse=sparse,
                                                 _use_arrow_dtype=use_arrow_dtype,
                                                 _add_offset=add_offset, **kw)

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
    def nrows(self):
        return self._nrows

    @property
    def use_arrow_dtype(self):
        return self._use_arrow_dtype

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
        from mars.context import get_context

        cupid_ctx = context()
        if cupid_ctx is None:
            raise SystemError('No Mars cluster found, please create via `o.create_mars_cluster`.')

        bearer_token = cupid_ctx.get_bearer_token()
        account = BearerTokenAccount(bearer_token)
        project = os.environ.get('ODPS_PROJECT_NAME', None)
        odps_params = op.odps_params.copy()
        if project:
            odps_params['project'] = project
        o = ODPS(None, None, account=account, **odps_params)
        cupid_session = CupidSession(o)

        mars_context = get_context()

        df = op.outputs[0]
        split_size = df.extra_params.chunk_bytes or READ_CHUNK_LIMIT

        data_src = o.get_table(op.table_name)
        if op.partition is not None:
            data_src = data_src.get_partition(op.partition)

        try:
            data_store_size = data_src.size
        except ODPSError:
            # fail to get data size, just ignore
            pass
        else:
            if data_store_size < split_size and mars_context is not None:
                # get worker counts
                worker_count = max(len(mars_context.get_worker_addresses()), 1)
                # data is too small, split as many as number of cores
                split_size = data_store_size // worker_count
                # at least 1M
                split_size = max(split_size, 1 * 1024 ** 2)
                logger.debug('Input data size is too small, split_size is {}'.format(split_size))

        logger.debug('Start creating download session of table {} from cupid.'.format(op.table_name))
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

        if np.isnan(df.shape[0]):
            est_chunk_rows = [None] * len(download_session.splits)
        else:
            sp_file_sizes = np.array([sp.split_file_end - sp.split_file_start
                                      for sp in download_session.splits])
            total_size = sp_file_sizes.sum()
            est_chunk_rows = sp_file_sizes * df.shape[0] // total_size

        logger.warning('Estimated chunk rows: %r', est_chunk_rows)

        out_chunks = []
        # Ignore add_offset at this time.
        op._add_offset = False

        if len(download_session.splits) == 0:
            logger.debug('Table {} has no data'.format(op.table_name))
            chunk_op = DataFrameReadTableSplit()
            index_value = parse_index(pd.RangeIndex(0))
            columns_value = parse_index(df.dtypes.index, store_data=True)
            out_chunk = chunk_op.new_chunk(None, shape=(np.nan, df.shape[1]), dtypes=op.dtypes,
                                           index_value=index_value, columns_value=columns_value,
                                           index=(0, 0))
            out_chunks = [out_chunk]
        else:
            for idx, split in enumerate(download_session.splits):
                chunk_op = DataFrameReadTableSplit(cupid_handle=to_str(split.handle),
                                                   split_index=split.split_index,
                                                   split_file_start=split.split_file_start,
                                                   split_file_end=split.split_file_end,
                                                   schema_file_start=split.schema_file_start,
                                                   schema_file_end=split.schema_file_end,
                                                   add_offset=op.add_offset, dtypes=op.dtypes,
                                                   sparse=op.sparse, split_size=split_size,
                                                   use_arrow_dtype=op.use_arrow_dtype,
                                                   estimate_rows=est_chunk_rows[idx])
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
    _use_arrow_dtype = BoolField('use_arrow_dtype')
    _dtypes = SeriesField('dtypes')
    _nrows = Int64Field('nrows')

    _split_size = Int64Field('split_size')
    _estimate_rows = Int64Field('estimate_rows')

    def __init__(self, cupid_handle=None, split_index=None, split_file_start=None, split_file_end=None,
                 schema_file_start=None, schema_file_end=None, nrows=None, dtypes=None,
                 split_size=None, use_arrow_dtype=None, estimate_rows=None, sparse=None, **kw):
        kw.update(_output_type_kw)
        super(DataFrameReadTableSplit, self).__init__(_cupid_handle=cupid_handle, _split_index=split_index,
                                                      _split_file_start=split_file_start,
                                                      _split_file_end=split_file_end,
                                                      _schema_file_start=schema_file_start,
                                                      _schema_file_end=schema_file_end,
                                                      _use_arrow_dtype=use_arrow_dtype,
                                                      _nrows=nrows, _estimate_rows=estimate_rows,
                                                      _split_size=split_size, _dtypes=dtypes,
                                                      _sparse=sparse, **kw)

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

    @property
    def split_size(self):
        return self._split_size

    @property
    def estimate_rows(self):
        return self._estimate_rows

    @property
    def use_arrow_dtype(self):
        return self._use_arrow_dtype

    @classmethod
    def estimate_size(cls, ctx, op):
        import numpy as np

        def is_object_dtype(dtype):
            try:
                return np.issubdtype(dtype, np.object_) \
                       or np.issubdtype(dtype, np.unicode_) \
                       or np.issubdtype(dtype, np.bytes_)
            except TypeError:  # pragma: no cover
                return False

        if op.split_size is None:
            ctx[op.outputs[0].key] = (0, 0)
            return

        arrow_size = ORC_COMPRESSION_RATIO * op.split_size
        n_strings = len([dt for dt in op.dtypes if is_object_dtype(dt)])
        if op.estimate_rows or op.nrows:
            rows = op.nrows if op.nrows is not None else op.estimate_rows
            pd_size = arrow_size + n_strings * rows * STRING_FIELD_OVERHEAD
            logger.debug('Estimate pandas memory cost: %r', pd_size)
        else:
            pd_size = arrow_size * 10 if n_strings else arrow_size

        ctx[op.outputs[0].key] = (pd_size, pd_size + arrow_size)

    @classmethod
    def execute(cls, ctx, op):
        import pyarrow as pa
        from cupid.io.table import TableSplit

        if op.cupid_handle is None:
            empty_df = pd.DataFrame()
            for name, dtype in op.outputs[0].dtypes.items():
                empty_df[name] = pd.Series(dtype=dtype)
            ctx[op.outputs[0].key] = empty_df
            return

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
            data = arrow_table_to_pandas_dataframe(
                pa.Table.from_batches(batches),
                use_arrow_dtype=op.use_arrow_dtype)[:op.nrows]
        else:
            arrow_table = reader.read_all()
            data = arrow_table_to_pandas_dataframe(arrow_table,
                                                   use_arrow_dtype=op.use_arrow_dtype)
        data_columns = data.dtypes.index
        expected_columns = op.outputs[0].dtypes.index
        if not data_columns.equals(expected_columns):
            logger.debug("Data columns differs from output columns, "
                         "data columns: {}, output columns: {}".format(data_columns, expected_columns))
            data.columns = expected_columns

        logger.debug('Read split table finished, split index: %s', op.split_index)
        logger.debug('Split data shape is {}, size is {}'.format(
            data.shape,
            data.memory_usage(deep=True).sum()))
        ctx[op.outputs[0].key] = data


def df_type_to_np_type(df_type, use_arrow_dtype=False):
    from ...df import types
    from ...df.backends.pd.types import _df_to_np_types

    if df_type == types.string:
        if use_arrow_dtype:
            return ArrowStringDtype()
        else:
            return np.dtype('object')
    elif df_type in _df_to_np_types:
        return _df_to_np_types[df_type]
    elif df_type == types.timestamp:
        return np.datetime64(0, 'ns').dtype
    else:
        return np.dtype('object')


def read_odps_table(table, shape, partition=None, sparse=False, chunk_bytes=None,
                    columns=None, odps_params=None, add_offset=False, use_arrow_dtype=False):
    import pandas as pd

    if chunk_bytes is not None:
        chunk_bytes = int(parse_readable_size(chunk_bytes)[0])
    table_name = '%s.%s' % (table.project.name, table.name)
    table_columns = table.schema.names
    table_types = table.schema.types
    df_types = [df_type_to_np_type(odps_type_to_df_type(type), use_arrow_dtype=use_arrow_dtype)
                for type in table_types]

    if columns is not None:
        df_types = [df_types[table_columns.index(col)] for col in columns]
        table_columns = columns

    dtypes = pd.Series(df_types, index=table_columns)

    op = DataFrameReadTable(odps_params=odps_params, table_name=table_name, partition_spec=partition,
                            dtypes=dtypes, sparse=sparse, add_offset=add_offset, columns=columns,
                            use_arrow_dtype=use_arrow_dtype)
    return op(shape, chunk_bytes=chunk_bytes)


class ReadODPSTableHeadRule(DataSourceHeadRule):
    @staticmethod
    def match(chunk, graph, keys):
        from mars.dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

        op = chunk.op
        inputs = graph.predecessors(chunk)
        if len(inputs) == 1 and isinstance(op, (DataFrameIlocGetItem, SeriesIlocGetItem)) and \
                op.is_head() and isinstance(inputs[0].op, DataFrameReadTableSplit) and \
                inputs[0].key not in keys:
            return True
        return False
