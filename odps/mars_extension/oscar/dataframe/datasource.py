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
import warnings
import time

import numpy as np
import pandas as pd

from mars.utils import parse_readable_size, ceildiv
from mars.serialization.serializables import (
    StringField,
    Int64Field,
    SeriesField,
    DictField,
    BoolField,
    AnyField,
)
from mars.dataframe import ArrowStringDtype
from mars.dataframe.core import DATAFRAME_CHUNK_TYPE
from mars.dataframe.utils import (
    parse_index,
    standardize_range_index,
    arrow_table_to_pandas_dataframe,
)
from mars.dataframe.datasource.core import ColumnPruneSupportedDataSourceMixin
from mars.core import OutputType

from ....config import options
from ....df.backends.odpssql.types import odps_type_to_df_type
from ....utils import to_str, to_timestamp
from ...utils import check_partition_exist, filter_partitions
from ..cupid_service import CupidServiceClient

logger = logging.getLogger(__name__)

CHUNK_BYTES_LIMIT = 64 * 1024**2
MAX_CHUNK_NUM = 512 * 1024**2

ORC_COMPRESSION_RATIO = 5
STRING_FIELD_OVERHEAD = 50

try:
    from mars.dataframe.datasource.core import (
        IncrementalIndexDatasource,
        IncrementalIndexDataSourceMixin,
    )

    _BASE = (
        IncrementalIndexDatasource,
        ColumnPruneSupportedDataSourceMixin,
        IncrementalIndexDataSourceMixin,
    )
    _NEED_STANDARDIZE = False
except ImportError:
    from mars.dataframe.datasource.core import HeadOptimizedDataSource

    _BASE = (HeadOptimizedDataSource, ColumnPruneSupportedDataSourceMixin)
    _NEED_STANDARDIZE = True


class DataFrameReadTable(*_BASE):
    _op_type_ = 123450

    odps_params = DictField("odps_params")
    table_name = StringField("table_name")
    partition_spec = StringField("partition_spec", default=None)
    dtypes = SeriesField("dtypes", default=None)
    index_type = StringField("index_type", default=None)
    columns = AnyField("columns", default=None)
    nrows = Int64Field("nrows", default=None)
    use_arrow_dtype = BoolField("use_arrow_dtype", default=None)
    string_as_binary = BoolField("string_as_binary", default=None)
    append_partitions = BoolField("append_partitions", default=None)
    last_modified_time = Int64Field("last_modified_time", default=None)
    with_split_meta_on_tile = BoolField("with_split_meta_on_tile", default=None)
    retry_times = Int64Field("retry_times", default=None)

    def __init__(self, sparse=None, memory_scale=None, **kw):
        super(DataFrameReadTable, self).__init__(
            sparse=sparse,
            memory_scale=memory_scale,
            _output_types=[OutputType.dataframe],
            **kw
        )

    @property
    def retryable(self):
        if "CUPID_SERVICE_SOCKET" in os.environ:
            return False
        else:
            return True

    @property
    def partition(self):
        return getattr(self, "partition_spec", None)

    @property
    def incremental_index(self):
        return self.index_type == "incremental"

    def get_columns(self):
        return self.columns

    def set_pruned_columns(self, columns, *, keep_order=None):  # pragma: no cover
        self.columns = columns

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
        import numpy as np
        import pandas as pd
        from mars.core.context import get_context

        df = op.outputs[0]
        split_size = df.extra_params.chunk_bytes or CHUNK_BYTES_LIMIT

        out_dtypes = df.dtypes
        out_shape = df.shape
        out_columns_value = df.columns_value
        if op.columns is not None:
            out_dtypes = out_dtypes[op.columns]
            out_shape = (df.shape[0], len(op.columns))
            out_columns_value = parse_index(out_dtypes.index, store_data=True)

        mars_context = get_context()
        if mars_context is not None:
            worker_count = len(mars_context.get_worker_addresses())
        else:
            worker_count = None

        cupid_client = CupidServiceClient()
        try:
            parts = cupid_client.enum_table_partitions(
                op.odps_params, op.table_name, op.partition
            )
            if parts is None:
                parts = [None]

            out_chunks = []
            chunk_idx = 0

            for partition_spec in parts:
                splits, split_size = cupid_client.create_table_download_session(
                    op.odps_params,
                    op.table_name,
                    partition_spec,
                    op.columns,
                    worker_count,
                    split_size,
                    MAX_CHUNK_NUM,
                    op.with_split_meta_on_tile,
                )

                logger.info("%s table splits have been created.", str(len(splits)))
                meta_chunk_rows = [split.meta_row_count for split in splits]
                if np.isnan(out_shape[0]):
                    est_chunk_rows = meta_chunk_rows
                else:
                    sp_file_sizes = np.array(
                        [sp.split_file_end - sp.split_file_start for sp in splits]
                    )
                    total_size = sp_file_sizes.sum()
                    ratio_chunk_rows = (
                        sp_file_sizes * out_shape[0] // total_size
                    ).tolist()
                    est_chunk_rows = [
                        mr if mr is not None else rr
                        for mr, rr in zip(meta_chunk_rows, ratio_chunk_rows)
                    ]

                logger.warning("Estimated chunk rows: %r", est_chunk_rows)

                if len(splits) == 0:
                    logger.info("Table %s has no data", op.table_name)
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
                    for idx, split in enumerate(splits):
                        chunk_op = DataFrameReadTableSplit(
                            cupid_handle=to_str(split.handle),
                            split_index=split.split_index,
                            split_file_start=split.split_file_start,
                            split_file_end=split.split_file_end,
                            schema_file_start=split.schema_file_start,
                            schema_file_end=split.schema_file_end,
                            index_type=op.index_type,
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
                            extra_params=op.extra_params,
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
        finally:
            cupid_client.close()

        if op.index_type == "incremental" and _NEED_STANDARDIZE:
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

        if len(data_srcs) == 0:
            # no partitions are selected
            chunk_op = DataFrameReadTableSplit()
            index_value = parse_index(pd.RangeIndex(0))
            columns_value = parse_index(out_dtypes.index, store_data=True)
            out_chunk = chunk_op.new_chunk(
                None,
                shape=(0, out_shape[1]),
                dtypes=op.dtypes,
                index_value=index_value,
                columns_value=columns_value,
                index=(index_start, 0),
            )
            out_chunks.append(out_chunk)
        else:
            retry_times = op.retry_times or options.retry_times
            for data_src in data_srcs:
                data_store_size = data_src.size

                retries = 0
                while True:
                    try:
                        with data_src.open_reader() as reader:
                            record_count = reader.count
                        break
                    except:
                        if retries >= retry_times:
                            raise
                        retries += 1
                        time.sleep(1)
                if data_store_size == 0:
                    # empty table
                    chunk_op = DataFrameReadTableSplit()
                    index_value = parse_index(pd.RangeIndex(0))
                    columns_value = parse_index(out_dtypes.index, store_data=True)
                    out_chunk = chunk_op.new_chunk(
                        None,
                        shape=(0, out_shape[1]),
                        dtypes=op.dtypes,
                        index_value=index_value,
                        columns_value=columns_value,
                        index=(index_start, 0),
                    )
                    out_chunks.append(out_chunk)
                    index_start += 1
                    continue
                chunk_size = df.extra_params.chunk_size

                partition_spec = (
                    str(data_src.partition_spec)
                    if getattr(data_src, "partition_spec", None)
                    else None
                )

                if chunk_size is None:
                    chunk_bytes = df.extra_params.chunk_bytes or CHUNK_BYTES_LIMIT
                    chunk_count = data_store_size // chunk_bytes + (
                        data_store_size % chunk_bytes != 0
                    )
                    chunk_size = ceildiv(record_count, chunk_count)
                    split_size = chunk_bytes
                else:
                    chunk_count = ceildiv(record_count, chunk_size)
                    split_size = data_store_size // chunk_count

                for i in range(chunk_count):
                    start_index = chunk_size * i
                    end_index = min(chunk_size * (i + 1), record_count)
                    row_size = end_index - start_index
                    chunk_op = DataFrameReadTableSplit(
                        table_name=op.table_name,
                        partition_spec=partition_spec,
                        start_index=start_index,
                        end_index=end_index,
                        nrows=op.nrows,
                        odps_params=op.odps_params,
                        columns=op.columns,
                        index_type=op.index_type,
                        dtypes=out_dtypes,
                        sparse=op.sparse,
                        split_size=split_size,
                        use_arrow_dtype=op.use_arrow_dtype,
                        estimate_rows=row_size,
                        append_partitions=op.append_partitions,
                        memory_scale=op.memory_scale,
                        retry_times=op.retry_times,
                        extra_params=op.extra_params,
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

        if op.index_type == "incremental" and _NEED_STANDARDIZE:
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
        if "CUPID_SERVICE_SOCKET" in os.environ:
            return cls._tile_cupid(op)
        else:
            return cls._tile_tunnel(op)


class DataFrameReadTableSplit(*_BASE):
    _op_type_ = 123451

    # for cupid
    cupid_handle = StringField("cupid_handle", default=None)
    split_index = Int64Field("split_index", default=None)
    split_file_start = Int64Field("split_file_start", default=None)
    split_file_end = Int64Field("split_file_end", default=None)
    schema_file_start = Int64Field("schema_file_start", default=None)
    schema_file_end = Int64Field("schema_file_end", default=None)
    use_arrow_dtype = BoolField("use_arrow_dtype", default=None)
    string_as_binary = BoolField("string_as_binary", default=None)
    dtypes = SeriesField("dtypes", default=None)
    nrows = Int64Field("nrows", default=None)
    index_type = StringField("index_type", default=None)

    # for tunnel
    table_name = StringField("table_name", default=None)
    partition_spec = StringField("partition_spec", default=None)
    start_index = Int64Field("start_index", default=None)
    end_index = Int64Field("end_index", default=None)
    odps_params = DictField("odps_params", default=None)
    columns = AnyField("columns", default=None)

    split_size = Int64Field("split_size", default=None)
    append_partitions = BoolField("append_partitions", default=None)
    estimate_rows = Int64Field("estimate_rows", default=None)
    meta_raw_size = Int64Field("meta_raw_size", default=None)
    retry_times = Int64Field("retry_times", default=None)

    def __init__(self, memory_scale=None, sparse=None, **kw):
        super(DataFrameReadTableSplit, self).__init__(
            sparse=sparse,
            memory_scale=memory_scale,
            _output_types=[OutputType.dataframe],
            **kw
        )

    @property
    def retryable(self):
        if "CUPID_SERVICE_SOCKET" in os.environ:
            return False
        else:
            return True

    @property
    def output_limit(self):
        return 1

    @property
    def incremental_index(self):
        return self.index_type == "incremental"

    def set_pruned_columns(self, columns, *, keep_order=None):  # pragma: no cover
        if isinstance(columns, str):
            columns = [columns]
        self.columns = list(columns)

    def get_columns(self):
        return self.columns

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
            logger.info("Estimate pandas memory cost: %r", pd_size)
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
            logger.info(
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
    def _align_output_data(cls, op, data):
        if isinstance(op.outputs[0], DATAFRAME_CHUNK_TYPE):
            dtypes = op.outputs[0].dtypes
            data = cls._align_columns(data, dtypes)
        else:
            dtypes = pd.Series([op.outputs[0].dtype], index=[op.outputs[0].name])
            data = cls._align_columns(data, dtypes)
            data = data[op.outputs[0].name]
        return data

    @classmethod
    def _build_empty_df(cls, out):
        empty_df = pd.DataFrame()
        for name, dtype in out.dtypes.items():
            empty_df[name] = pd.Series(dtype=dtype)
        return empty_df

    @classmethod
    def _execute_in_cupid(cls, ctx, op):
        out = op.outputs[0]

        if op.cupid_handle is None:
            ctx[out.key] = cls._build_empty_df(out)
            return

        split_config = dict(
            _handle=op.cupid_handle,
            _split_index=op.split_index,
            _split_file_start=op.split_file_start,
            _split_file_end=op.split_file_end,
            _schema_file_start=op.schema_file_start,
            _schema_file_end=op.schema_file_end,
        )
        cupid_client = CupidServiceClient()
        try:
            pa_table = cupid_client.read_table_data(split_config, op.nrows)
        finally:
            cupid_client.close()
            cupid_client = None
        pa_table = cls._append_partition_values(pa_table, op)

        if op.string_as_binary:
            pa_table = cls._cast_string_to_binary(pa_table)
        data = arrow_table_to_pandas_dataframe(
            pa_table, use_arrow_dtype=op.use_arrow_dtype
        )[: op.nrows]

        data = cls._align_output_data(op, data)
        logger.info("Read split table finished, split index: %s", op.split_index)
        logger.info("Split data shape is %s", data.shape)
        ctx[out.key] = data

    @classmethod
    def _execute_arrow_tunnel(cls, ctx, op):
        from odps import ODPS
        from odps.tunnel import TableTunnel

        out = op.outputs[0]

        if op.table_name is None:
            # is empty table
            ctx[out.key] = cls._build_empty_df(out)
            return

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
        retry_times = op.retry_times or options.retry_times
        init_sleep_secs = 1
        logger.info(
            "Start creating download session for table %s(%s) start index %s end index %s retry_times %s.",
            op.table_name,
            op.partition_spec,
            op.start_index,
            op.end_index,
            retry_times,
        )
        retries = 0
        while True:
            try:
                if op.partition_spec is not None:
                    download_session = tunnel.create_download_session(
                        t.name, partition_spec=op.partition_spec
                    )
                else:
                    download_session = tunnel.create_download_session(t.name)
                break
            except:
                if retries >= retry_times:
                    raise
                retries += 1
                sleep_secs = retries * init_sleep_secs
                logger.exception(
                    "Create download session failed, sleep %s seconds and retry it",
                    sleep_secs,
                    exc_info=1,
                )
                time.sleep(sleep_secs)

        logger.info(
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

        retries = 0
        while True:
            try:
                with download_session.open_arrow_reader(
                    op.start_index, count, columns=op.columns
                ) as reader:
                    table = reader.read()
                break
            except:
                if retries >= retry_times:
                    raise
                retries += 1
                sleep_secs = retries * init_sleep_secs
                logger.exception(
                    "Read table failed, sleep %s seconds and retry it",
                    sleep_secs,
                    exc_info=1,
                )
                time.sleep(sleep_secs)

        table = cls._append_partition_values(table, op)
        if op.string_as_binary:
            table = cls._cast_string_to_binary(table)
        data = arrow_table_to_pandas_dataframe(
            table, use_arrow_dtype=op.use_arrow_dtype
        )
        data = cls._align_output_data(op, data)
        logger.info(
            "Finish reading table %s(%s) split from %s to %s",
            op.table_name,
            op.partition_spec,
            op.start_index,
            op.end_index,
        )
        ctx[op.outputs[0].key] = data

    @classmethod
    def execute(cls, ctx, op):
        if "CUPID_SERVICE_SOCKET" in os.environ:
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
    add_offset=None,
    use_arrow_dtype=False,
    string_as_binary=None,
    memory_scale=None,
    append_partitions=False,
    with_split_meta_on_tile=False,
    index_type="incremental",
    extra_params=None,
):
    import pandas as pd

    if add_offset is not None:
        warnings.warn(
            "add_offset is deprecated, please use index_type instead",
            DeprecationWarning,
        )
        if add_offset in (True, False):
            index_type = "incremental" if add_offset else "chunk_incremental"

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
    retry_times = options.retry_times
    op = DataFrameReadTable(
        odps_params=odps_params,
        table_name=table_name,
        partition_spec=partition,
        dtypes=dtypes,
        sparse=sparse,
        index_type=index_type,
        columns=columns,
        use_arrow_dtype=use_arrow_dtype,
        string_as_binary=string_as_binary,
        memory_scale=memory_scale,
        append_partitions=append_partitions,
        last_modified_time=to_timestamp(table.last_modified_time),
        with_split_meta_on_tile=with_split_meta_on_tile,
        retry_times=retry_times,
        extra_params=extra_params or dict(),
    )
    return op(shape, chunk_bytes=chunk_bytes, chunk_size=chunk_size)
