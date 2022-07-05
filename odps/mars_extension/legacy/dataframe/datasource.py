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

import logging
import os

import numpy as np
import pandas as pd

from mars.utils import parse_readable_size, ceildiv
from mars.serialize import (
    StringField,
    Int64Field,
    SeriesField,
    DictField,
    BoolField,
    ListField,
)
from mars.dataframe import ArrowStringDtype
from mars.dataframe.operands import DataFrameOperandMixin, DataFrameOperand
from mars.dataframe.utils import (
    parse_index,
    standardize_range_index,
    arrow_table_to_pandas_dataframe,
)
from mars.optimizes.runtime.dataframe import DataSourceHeadRule

try:
    from mars.dataframe.datasource.core import (
        HeadOptimizedDataSource,
        ColumnPruneSupportedDataSourceMixin,
    )

    BASES = (HeadOptimizedDataSource, ColumnPruneSupportedDataSourceMixin)
    head_can_be_opt = True
except ImportError:
    BASES = (DataFrameOperand, DataFrameOperandMixin)
    head_can_be_opt = False
try:
    from mars.core import OutputType

    _output_type_kw = dict(_output_types=[OutputType.dataframe])
except ImportError:
    from mars.dataframe.operands import ObjectType

    _output_type_kw = dict(_object_type=ObjectType.dataframe)

from ....df.backends.odpssql.types import odps_type_to_df_type
from ....errors import ODPSError
from ....utils import to_str, to_timestamp
from ...utils import filter_partitions, check_partition_exist

logger = logging.getLogger(__name__)

READ_CHUNK_LIMIT = 64 * 1024**2
MAX_CHUNK_SIZE = 512 * 1024**2

ORC_COMPRESSION_RATIO = 5
STRING_FIELD_OVERHEAD = 50

_Base = type("_DataSource", BASES, dict())


class DataFrameReadTable(_Base):
    _op_type_ = 123450

    _odps_params = DictField("odps_params")
    _table_name = StringField("table_name")
    _partition_spec = StringField("partition_spec")
    _dtypes = SeriesField("dtypes")
    _add_offset = BoolField("add_offset")
    _columns = ListField("columns")
    _nrows = Int64Field("nrows")
    _use_arrow_dtype = BoolField("use_arrow_dtype")
    _string_as_binary = BoolField("string_as_binary")
    _append_partitions = BoolField("append_partitions")
    _last_modified_time = Int64Field("last_modified_time")
    _with_split_meta_on_tile = BoolField("with_split_meta_on_tile")

    def __init__(
        self,
        odps_params=None,
        table_name=None,
        partition_spec=None,
        columns=None,
        dtypes=None,
        nrows=None,
        sparse=None,
        add_offset=True,
        use_arrow_dtype=None,
        string_as_binary=None,
        memory_scale=None,
        append_partitions=None,
        last_modified_time=None,
        with_split_meta_on_tile=False,
        **kw
    ):
        kw.update(_output_type_kw)
        super(DataFrameReadTable, self).__init__(
            _odps_params=odps_params,
            _table_name=table_name,
            _partition_spec=partition_spec,
            _columns=columns,
            _dtypes=dtypes,
            _nrows=nrows,
            _sparse=sparse,
            _use_arrow_dtype=use_arrow_dtype,
            _string_as_binary=string_as_binary,
            _add_offset=add_offset,
            _append_partitions=append_partitions,
            _last_modified_time=last_modified_time,
            _memory_scale=memory_scale,
            _with_split_meta_on_tile=with_split_meta_on_tile,
            **kw
        )

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
        return getattr(self, "_partition_spec", None)

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
    def string_as_binary(self):
        return self._string_as_binary

    @property
    def add_offset(self):
        return self._add_offset

    @property
    def append_partitions(self):
        return self._append_partitions

    @property
    def with_split_meta_on_tile(self):
        return self._with_split_meta_on_tile

    def get_columns(self):
        return self._columns

    def set_pruned_columns(self, columns):
        self._columns = columns

    def __call__(self, shape, chunk_bytes=None, chunk_size=None):
        import numpy as np
        import pandas as pd

        if np.isnan(shape[0]):
            index_value = parse_index(pd.RangeIndex(0))
        else:
            index_value = parse_index(pd.RangeIndex(shape[0]))
        columns_value = parse_index(self.dtypes.index, store_data=True)
        return self.new_dataframe(
            None,
            shape,
            dtypes=self.dtypes,
            index_value=index_value,
            columns_value=columns_value,
            chunk_bytes=chunk_bytes,
            chunk_size=chunk_size,
        )

    @classmethod
    def _tile_cupid(cls, op):
        from odps import ODPS
        from odps.accounts import BearerTokenAccount
        from cupid import CupidSession, context
        from cupid.errors import CupidError
        from mars.context import get_context

        cupid_ctx = context()

        bearer_token = cupid_ctx.get_bearer_token()
        account = BearerTokenAccount(bearer_token)
        project = os.environ.get("ODPS_PROJECT_NAME", None)
        odps_params = op.odps_params.copy()
        if project:
            odps_params["project"] = project
        endpoint = os.environ.get("ODPS_RUNTIME_ENDPOINT") or odps_params["endpoint"]
        o = ODPS(
            None,
            None,
            account=account,
            project=odps_params["project"],
            endpoint=endpoint,
        )
        cupid_session = CupidSession(o)

        mars_context = get_context()

        df = op.outputs[0]
        split_size = df.extra_params.chunk_bytes or READ_CHUNK_LIMIT

        out_dtypes = df.dtypes
        out_shape = df.shape
        out_columns_value = df.columns_value
        if op.columns is not None:
            out_dtypes = out_dtypes[op.columns]
            out_shape = (df.shape[0], len(op.columns))
            out_columns_value = parse_index(out_dtypes.index, store_data=True)

        table_obj = o.get_table(op.table_name)
        if not table_obj.schema.partitions:
            data_srcs = [table_obj]
        elif op.partition is not None and check_partition_exist(
            table_obj, op.partition
        ):
            data_srcs = [table_obj.get_partition(op.partition)]
        else:
            data_srcs = list(table_obj.partitions)
            if op.partition is not None:
                data_srcs = filter_partitions(o, data_srcs, op.partition)

        out_chunks = []
        chunk_idx = 0

        for data_src in data_srcs:
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
                    split_size = max(split_size, 1 * 1024**2)
                    logger.debug(
                        "Input data size is too small, split_size is %s", split_size
                    )

            logger.debug(
                "Start creating download session of table %s from cupid, "
                "columns: %s",
                op.table_name,
                op.columns,
            )
            while True:
                try:
                    download_session = cupid_session.create_download_session(
                        data_src,
                        split_size=split_size,
                        columns=op.columns,
                        with_split_meta=op.with_split_meta_on_tile,
                    )
                    break
                except CupidError:
                    logger.debug(
                        "The number of splits exceeds 100000, split_size is %s",
                        split_size,
                    )
                    if split_size >= MAX_CHUNK_SIZE:
                        raise
                    else:
                        split_size *= 2

            logger.debug(
                "%s table splits have been created.", str(len(download_session.splits))
            )

            meta_chunk_rows = [
                split.meta_row_count for split in download_session.splits
            ]
            if np.isnan(out_shape[0]):
                est_chunk_rows = meta_chunk_rows
            else:
                sp_file_sizes = np.array(
                    [
                        sp.split_file_end - sp.split_file_start
                        for sp in download_session.splits
                    ]
                )
                total_size = sp_file_sizes.sum()
                ratio_chunk_rows = (sp_file_sizes * out_shape[0] // total_size).tolist()
                est_chunk_rows = [
                    mr if mr is not None else rr
                    for mr, rr in zip(meta_chunk_rows, ratio_chunk_rows)
                ]

            partition_spec = (
                str(data_src.partition_spec)
                if getattr(data_src, "partition_spec", None)
                else None
            )

            logger.warning("Estimated chunk rows: %r", est_chunk_rows)

            if len(download_session.splits) == 0:
                logger.debug("Table %s has no data", op.table_name)
                chunk_op = DataFrameReadTableSplit()
                index_value = parse_index(pd.RangeIndex(0))
                columns_value = parse_index(out_dtypes.index, store_data=True)
                out_chunk = chunk_op.new_chunk(
                    None,
                    shape=(np.nan, out_shape[1]),
                    dtypes=op.dtypes,
                    index_value=index_value,
                    columns_value=columns_value,
                    index=(chunk_idx, 0),
                )
                out_chunks.append(out_chunk)
                chunk_idx += 1
            else:
                for idx, split in enumerate(download_session.splits):
                    chunk_op = DataFrameReadTableSplit(
                        cupid_handle=to_str(split.handle),
                        split_index=split.split_index,
                        split_file_start=split.split_file_start,
                        split_file_end=split.split_file_end,
                        schema_file_start=split.schema_file_start,
                        schema_file_end=split.schema_file_end,
                        add_offset=op.add_offset,
                        dtypes=out_dtypes,
                        sparse=op.sparse,
                        split_size=split_size,
                        string_as_binary=op.string_as_binary,
                        use_arrow_dtype=op.use_arrow_dtype,
                        estimate_rows=est_chunk_rows[idx],
                        partition_spec=partition_spec,
                        append_partitions=op.append_partitions,
                        meta_raw_size=split.meta_raw_size,
                        nrows=meta_chunk_rows[idx] or op.nrows,
                        memory_scale=op.memory_scale,
                    )
                    # the chunk shape is unknown
                    index_value = parse_index(pd.RangeIndex(0))
                    columns_value = parse_index(out_dtypes.index, store_data=True)
                    out_chunk = chunk_op.new_chunk(
                        None,
                        shape=(np.nan, out_shape[1]),
                        dtypes=out_dtypes,
                        index_value=index_value,
                        columns_value=columns_value,
                        index=(chunk_idx, 0),
                    )
                    chunk_idx += 1
                    out_chunks.append(out_chunk)

        if op.add_offset:
            out_chunks = standardize_range_index(out_chunks)

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (out_shape[1],))
        return new_op.new_dataframes(
            None,
            shape=out_shape,
            dtypes=op.dtypes,
            index_value=df.index_value,
            columns_value=out_columns_value,
            chunks=out_chunks,
            nsplits=nsplits,
        )

    @classmethod
    def _tile_tunnel(cls, op):
        from odps import ODPS

        project = os.environ.get("ODPS_PROJECT_NAME", None)
        odps_params = op.odps_params.copy()
        if project:
            odps_params["project"] = project
        endpoint = os.environ.get("ODPS_RUNTIME_ENDPOINT") or odps_params["endpoint"]
        o = ODPS(
            odps_params["access_id"],
            odps_params["secret_access_key"],
            project=odps_params["project"],
            endpoint=endpoint,
        )

        table_obj = o.get_table(op.table_name)
        if not table_obj.schema.partitions:
            data_srcs = [table_obj]
        elif op.partition is not None and check_partition_exist(
            table_obj, op.partition
        ):
            data_srcs = [table_obj.get_partition(op.partition)]
        else:
            data_srcs = list(table_obj.partitions)
            if op.partition is not None:
                data_srcs = filter_partitions(o, data_srcs, op.partition)

        out_chunks = []
        row_nsplits = []
        index_start = 0
        df = op.outputs[0]

        out_dtypes = df.dtypes
        out_shape = df.shape
        out_columns_value = df.columns_value
        if op.columns is not None:
            out_dtypes = out_dtypes[op.columns]
            out_shape = (df.shape[0], len(op.columns))
            out_columns_value = parse_index(out_dtypes.index, store_data=True)

        for data_src in data_srcs:
            data_store_size = data_src.size
            shape = out_shape
            chunk_size = df.extra_params.chunk_size

            partition_spec = (
                str(data_src.partition_spec)
                if getattr(data_src, "partition_spec", None)
                else None
            )

            if chunk_size is None:
                chunk_bytes = df.extra_params.chunk_bytes or READ_CHUNK_LIMIT
                chunk_count = data_store_size // chunk_bytes + (
                    data_store_size % chunk_bytes != 0
                )
                chunk_size = ceildiv(shape[0], chunk_count)
                split_size = chunk_bytes
            else:
                chunk_count = ceildiv(shape[0], chunk_size)
                split_size = data_store_size // chunk_count

            for i in range(chunk_count):
                start_index = chunk_size * i
                end_index = min(chunk_size * (i + 1), shape[0])
                row_size = end_index - start_index
                chunk_op = DataFrameReadTableSplit(
                    table_name=op.table_name,
                    partition_spec=partition_spec,
                    start_index=start_index,
                    end_index=end_index,
                    nrows=op.nrows,
                    odps_params=op.odps_params,
                    columns=op.columns,
                    add_offset=op.add_offset,
                    dtypes=out_dtypes,
                    sparse=op.sparse,
                    split_size=split_size,
                    use_arrow_dtype=op.use_arrow_dtype,
                    estimate_rows=row_size,
                    append_partitions=op.append_partitions,
                    memory_scale=op.memory_scale,
                )
                index_value = parse_index(pd.RangeIndex(start_index, end_index))
                columns_value = parse_index(out_dtypes.index, store_data=True)
                out_chunk = chunk_op.new_chunk(
                    None,
                    shape=(row_size, out_shape[1]),
                    dtypes=out_dtypes,
                    index_value=index_value,
                    columns_value=columns_value,
                    index=(index_start + i, 0),
                )
                row_nsplits.append(row_size)
                out_chunks.append(out_chunk)

            index_start += chunk_count

        if op.add_offset:
            out_chunks = standardize_range_index(out_chunks)

        new_op = op.copy()
        nsplits = (tuple(row_nsplits), (out_shape[1],))
        return new_op.new_dataframes(
            None,
            shape=out_shape,
            dtypes=op.dtypes,
            index_value=df.index_value,
            columns_value=out_columns_value,
            chunks=out_chunks,
            nsplits=nsplits,
        )

    @classmethod
    def _tile(cls, op):
        from cupid.runtime import RuntimeContext

        if RuntimeContext.is_context_ready():
            return cls._tile_cupid(op)
        else:
            return cls._tile_tunnel(op)

    if not head_can_be_opt:
        tile = _tile


class DataFrameReadTableSplit(_Base):
    _op_type_ = 123451

    # for cupid
    _cupid_handle = StringField("cupid_handle")
    _split_index = Int64Field("split_index")
    _split_file_start = Int64Field("split_file_start")
    _split_file_end = Int64Field("split_file_end")
    _schema_file_start = Int64Field("schema_file_start")
    _schema_file_end = Int64Field("schema_file_end")
    _use_arrow_dtype = BoolField("use_arrow_dtype")
    _string_as_binary = BoolField("string_as_binary")
    _dtypes = SeriesField("dtypes")
    _nrows = Int64Field("nrows")

    # for tunnel
    _table_name = StringField("table_name")
    _partition_spec = StringField("partition_spec")
    _start_index = Int64Field("start_index")
    _end_index = Int64Field("end_index")
    _odps_params = DictField("odps_params")
    _columns = ListField("columns")

    _split_size = Int64Field("split_size")
    _append_partitions = BoolField("append_partitions")
    _estimate_rows = Int64Field("estimate_rows")
    _meta_raw_size = Int64Field("meta_raw_size")

    def __init__(
        self,
        cupid_handle=None,
        split_index=None,
        split_file_start=None,
        split_file_end=None,
        schema_file_start=None,
        schema_file_end=None,
        table_name=None,
        partition_spec=None,
        start_index=None,
        end_index=None,
        odps_params=None,
        columns=None,
        nrows=None,
        dtypes=None,
        string_as_binary=None,
        split_size=None,
        use_arrow_dtype=None,
        memory_scale=None,
        estimate_rows=None,
        meta_raw_size=None,
        append_partitions=None,
        sparse=None,
        **kw
    ):
        kw.update(_output_type_kw)
        super(DataFrameReadTableSplit, self).__init__(
            _cupid_handle=cupid_handle,
            _split_index=split_index,
            _split_file_start=split_file_start,
            _split_file_end=split_file_end,
            _schema_file_start=schema_file_start,
            _schema_file_end=schema_file_end,
            _table_name=table_name,
            _partition_spec=partition_spec,
            _columns=columns,
            _start_index=start_index,
            _end_index=end_index,
            _odps_params=odps_params,
            _use_arrow_dtype=use_arrow_dtype,
            _string_as_binary=string_as_binary,
            _nrows=nrows,
            _estimate_rows=estimate_rows,
            _split_size=split_size,
            _dtypes=dtypes,
            _append_partitions=append_partitions,
            _sparse=sparse,
            _meta_raw_size=meta_raw_size,
            _memory_scale=memory_scale,
            **kw
        )

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
    def table_name(self):
        return self._table_name

    @property
    def partition_spec(self):
        return self._partition_spec

    @property
    def start_index(self):
        return self._start_index

    @property
    def end_index(self):
        return self._end_index

    @property
    def odps_params(self):
        return self._odps_params

    @property
    def columns(self):
        return self._columns

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

    @property
    def string_as_binary(self):
        return self._string_as_binary

    @property
    def append_partitions(self):
        return self._append_partitions

    @property
    def meta_raw_size(self):
        return self._meta_raw_size

    @classmethod
    def estimate_size(cls, ctx, op):
        import numpy as np

        def is_object_dtype(dtype):
            try:
                return (
                    np.issubdtype(dtype, np.object_)
                    or np.issubdtype(dtype, np.unicode_)
                    or np.issubdtype(dtype, np.bytes_)
                )
            except TypeError:  # pragma: no cover
                return False

        if op.split_size is None:
            ctx[op.outputs[0].key] = (0, 0)
            return

        arrow_size = (op.memory_scale or ORC_COMPRESSION_RATIO) * op.split_size
        if op.meta_raw_size is not None:
            raw_arrow_size = (op.memory_scale or 1) * op.meta_raw_size
            arrow_size = max(arrow_size, raw_arrow_size)

        n_strings = len([dt for dt in op.dtypes if is_object_dtype(dt)])
        if op.estimate_rows or op.nrows:
            rows = op.nrows if op.nrows is not None else op.estimate_rows
            pd_size = arrow_size + n_strings * rows * STRING_FIELD_OVERHEAD
            logger.debug("Estimate pandas memory cost: %r", pd_size)
        else:
            pd_size = arrow_size * 10 if n_strings else arrow_size

        ctx[op.outputs[0].key] = (pd_size, pd_size + arrow_size)

    @classmethod
    def _cast_string_to_binary(cls, arrow_table):
        import pyarrow as pa

        new_schema = []
        for field in arrow_table.schema:
            if field.type == pa.string():
                new_schema.append(pa.field(field.name, pa.binary()))
            else:
                new_schema.append(field)

        return arrow_table.cast(pa.schema(new_schema))

    @classmethod
    def _append_partition_values(cls, arrow_table, op):
        import pyarrow as pa

        if op.append_partitions and op.partition_spec:
            from odps.types import PartitionSpec

            spec = PartitionSpec(op.partition_spec)

            for col_name, pt_val in spec.items():
                arrow_table = arrow_table.append_column(
                    col_name, pa.array([pt_val] * arrow_table.num_rows, pa.string())
                )

        return arrow_table

    @staticmethod
    def _align_columns(data, expected_dtypes):
        data_columns = data.dtypes.index
        expected_columns = expected_dtypes.index
        if not data_columns.equals(expected_columns):
            logger.debug(
                "Data columns differs from output columns, "
                "data columns: %s, output columns: %s",
                data_columns,
                expected_columns,
            )
            data.columns = expected_columns[: len(data.columns)]
            for extra_col in expected_columns[len(data.columns) :]:
                data[extra_col] = pd.Series([], dtype=expected_dtypes[extra_col])
            if not data.dtypes.index.equals(expected_columns):
                data = data[expected_columns]
        return data

    @classmethod
    def _execute_in_cupid(cls, ctx, op):
        import pyarrow as pa
        from cupid.io.table import TableSplit

        out = op.outputs[0]

        if op.cupid_handle is None:
            empty_df = pd.DataFrame()
            for name, dtype in out.dtypes.items():
                empty_df[name] = pd.Series(dtype=dtype)
            ctx[out.key] = empty_df
            return

        tsp = TableSplit(
            _handle=op.cupid_handle,
            _split_index=op.split_index,
            _split_file_start=op.split_file_start,
            _split_file_end=op.split_file_end,
            _schema_file_start=op.schema_file_start,
            _schema_file_end=op.schema_file_end,
        )
        logger.debug("Read split table, split index: %s", op.split_index)
        reader = tsp.open_arrow_reader()
        if op.nrows is None:
            arrow_table = reader.read_all()
        else:
            nrows = 0
            batches = []
            while nrows < op.nrows:
                try:
                    batch = reader.read_next_batch()
                    nrows += batch.num_rows
                    batches.append(batch)
                except StopIteration:
                    break
            logger.debug("Read %s rows of this split.", op.nrows)
            arrow_table = pa.Table.from_batches(batches)

        arrow_table = cls._append_partition_values(arrow_table, op)

        if op.string_as_binary:
            arrow_table = cls._cast_string_to_binary(arrow_table)
        data = arrow_table_to_pandas_dataframe(
            arrow_table, use_arrow_dtype=op.use_arrow_dtype
        )
        if op.nrows is not None:
            data = data[: op.nrows]

        data = cls._align_columns(data, out.dtypes)

        logger.debug("Read split table finished, split index: %s", op.split_index)
        logger.debug(
            "Split data shape is %s, size is %s",
            data.shape,
            data.memory_usage(deep=True).sum(),
        )
        ctx[out.key] = data

    @classmethod
    def _execute_arrow_tunnel(cls, ctx, op):
        from odps import ODPS
        from odps.tunnel import TableTunnel

        project = os.environ.get("ODPS_PROJECT_NAME", None)
        odps_params = op.odps_params.copy()
        if project:
            odps_params["project"] = project
        endpoint = os.environ.get("ODPS_RUNTIME_ENDPOINT") or odps_params["endpoint"]
        o = ODPS(
            odps_params["access_id"],
            odps_params["secret_access_key"],
            project=odps_params["project"],
            endpoint=endpoint,
        )

        t = o.get_table(op.table_name)
        tunnel = TableTunnel(o, project=t.project)

        if op.partition_spec is not None:
            download_session = tunnel.create_download_session(
                t.name, partition_spec=op.partition_spec
            )
        else:
            download_session = tunnel.create_download_session(t.name)
        logger.debug(
            "Start reading table %s(%s) split from %s to %s",
            op.table_name,
            op.partition_spec,
            op.start_index,
            op.end_index,
        )
        if op.nrows is None:
            count = op.end_index - op.start_index
        else:
            count = op.nrows

        with download_session.open_arrow_reader(
            op.start_index, count, columns=op.columns
        ) as reader:
            table = reader.read()

        table = cls._append_partition_values(table, op)
        if op.string_as_binary:
            table = cls._cast_string_to_binary(table)
        data = arrow_table_to_pandas_dataframe(
            table, use_arrow_dtype=op.use_arrow_dtype
        )

        data = cls._align_columns(data, op.outputs[0].dtypes)

        logger.debug(
            "Finish reading table %s(%s) split from %s to %s",
            op.table_name,
            op.partition_spec,
            op.start_index,
            op.end_index,
        )
        ctx[op.outputs[0].key] = data

    @classmethod
    def execute(cls, ctx, op):
        from cupid.runtime import RuntimeContext

        if RuntimeContext.is_context_ready():
            cls._execute_in_cupid(ctx, op)
        else:
            cls._execute_arrow_tunnel(ctx, op)


def df_type_to_np_type(df_type, use_arrow_dtype=False):
    from ....df import types
    from ....df.backends.pd.types import _df_to_np_types

    if df_type == types.string:
        if use_arrow_dtype:
            return ArrowStringDtype()
        else:
            return np.dtype("object")
    elif df_type in _df_to_np_types:
        return _df_to_np_types[df_type]
    elif df_type == types.timestamp:
        return np.datetime64(0, "ns").dtype
    else:
        return np.dtype("object")


def read_odps_table(
    table,
    shape,
    partition=None,
    sparse=False,
    chunk_bytes=None,
    chunk_size=None,
    columns=None,
    odps_params=None,
    add_offset=False,
    use_arrow_dtype=False,
    string_as_binary=None,
    memory_scale=None,
    append_partitions=False,
    with_split_meta_on_tile=False,
):
    import pandas as pd

    if isinstance(chunk_size, (list, tuple)):
        if len(chunk_size) == 1:
            chunk_size = chunk_size[0]
        if len(chunk_size) > 1:
            raise ValueError("Only support split on rows")

    if chunk_bytes is not None:
        chunk_bytes = int(parse_readable_size(chunk_bytes)[0])
    table_name = "%s.%s" % (table.project.name, table.name)
    cols = table.schema.columns if append_partitions else table.schema.simple_columns
    table_columns = [c.name for c in cols]
    table_types = [c.type for c in cols]
    df_types = [
        df_type_to_np_type(odps_type_to_df_type(type), use_arrow_dtype=use_arrow_dtype)
        for type in table_types
    ]

    if columns is not None:
        # reorder columns
        new_columns = [c for c in table_columns if c in columns]
        df_types = [df_types[table_columns.index(col)] for col in new_columns]
        table_columns = new_columns
        columns = new_columns

    dtypes = pd.Series(df_types, index=table_columns)

    op = DataFrameReadTable(
        odps_params=odps_params,
        table_name=table_name,
        partition_spec=partition,
        dtypes=dtypes,
        sparse=sparse,
        add_offset=add_offset,
        columns=columns,
        use_arrow_dtype=use_arrow_dtype,
        string_as_binary=string_as_binary,
        memory_scale=memory_scale,
        append_partitions=append_partitions,
        last_modified_time=to_timestamp(table.last_modified_time),
        with_split_meta_on_tile=with_split_meta_on_tile,
    )
    return op(shape, chunk_bytes=chunk_bytes, chunk_size=chunk_size)


class ReadODPSTableHeadRule(DataSourceHeadRule):
    @staticmethod
    def match(chunk, graph, keys):
        from mars.dataframe.indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem

        op = chunk.op
        inputs = graph.predecessors(chunk)
        if (
            len(inputs) == 1
            and isinstance(op, (DataFrameIlocGetItem, SeriesIlocGetItem))
            and isinstance(inputs[0].op, DataFrameReadTableSplit)
            and inputs[0].key not in keys
        ):
            try:
                is_head = op.can_be_optimized()
            except AttributeError:
                is_head = op.is_head()
            if is_head:
                return True
            else:
                return False
        return False
