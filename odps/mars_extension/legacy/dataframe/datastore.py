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

import os
import time
import uuid
import logging

from mars.dataframe.operands import DataFrameOperandMixin, DataFrameOperand
from mars.dataframe.utils import parse_index
from mars.serialize import StringField, SeriesField, BoolField, DictField, Int64Field
from mars.dataframe.utils import build_concatenated_rows_frame

from ....utils import to_str
from ...utils import convert_pandas_object_to_string

logger = logging.getLogger(__name__)

try:
    from mars.core import OutputType

    _output_type_kw = dict(_output_types=[OutputType.dataframe])
except ImportError:
    from mars.dataframe.operands import ObjectType

    _output_type_kw = dict(_object_type=ObjectType.dataframe)


class DataFrameWriteTable(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123460

    _dtypes = SeriesField("dtypes")

    _odps_params = DictField("odps_params")
    _table_name = StringField("table_name")
    _partition_spec = StringField("partition_spec")
    _overwrite = BoolField("overwrite")
    _write_batch_size = Int64Field("write_batch_size")
    _unknown_as_string = BoolField("unknown_as_string")

    def __init__(
        self,
        dtypes=None,
        odps_params=None,
        table_name=None,
        partition_spec=None,
        unknown_as_string=None,
        over_write=None,
        write_batch_size=None,
        **kw
    ):
        kw.update(_output_type_kw)
        super(DataFrameWriteTable, self).__init__(
            _dtypes=dtypes,
            _odps_params=odps_params,
            _table_name=table_name,
            _partition_spec=partition_spec,
            _unknown_as_string=unknown_as_string,
            _overwrite=over_write,
            _write_batch_size=write_batch_size,
            **kw
        )

    @property
    def retryable(self):
        return False

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def unknown_as_string(self):
        return self._unknown_as_string

    @property
    def odps_params(self):
        return self._odps_params

    @property
    def table_name(self):
        return self._table_name

    @property
    def partition_spec(self):
        return self._partition_spec

    @property
    def overwrite(self):
        return self._overwrite

    @property
    def write_batch_size(self):
        return self._write_batch_size

    def __call__(self, x):
        shape = (0,) * len(x.shape)
        index_value = parse_index(x.index_value.to_pandas()[:0], x.key, "index")
        columns_value = parse_index(
            x.columns_value.to_pandas()[:0], x.key, "columns", store_data=True
        )
        return self.new_dataframe(
            [x],
            shape=shape,
            dtypes=x.dtypes[:0],
            index_value=index_value,
            columns_value=columns_value,
        )

    @classmethod
    def _tile_cupid(cls, op):
        from odps import ODPS
        from odps.accounts import BearerTokenAccount
        from cupid import CupidSession, context
        from cupid.runtime import RuntimeContext

        if not RuntimeContext.is_context_ready():
            raise SystemError(
                "No Mars cluster found, please create via `o.create_mars_cluster`."
            )
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

        data_src = o.get_table(op.table_name)

        logger.debug("Start creating upload session from cupid.")
        upload_session = cupid_session.create_upload_session(data_src)

        input_df = build_concatenated_rows_frame(op.inputs[0])
        out_df = op.outputs[0]

        out_chunks = []
        out_chunk_shape = (0,) * len(input_df.shape)
        blocks = {}
        for chunk in input_df.chunks:
            block_id = str(int(time.time())) + "_" + str(uuid.uuid4()).replace("-", "")
            chunk_op = DataFrameWriteTableSplit(
                dtypes=op.dtypes,
                table_name=op.table_name,
                unknown_as_string=op.unknown_as_string,
                partition_spec=op.partition_spec,
                cupid_handle=to_str(upload_session.handle),
                block_id=block_id,
                write_batch_size=op.write_batch_size,
            )
            out_chunk = chunk_op.new_chunk(
                [chunk],
                shape=out_chunk_shape,
                index=chunk.index,
                index_value=out_df.index_value,
                dtypes=chunk.dtypes,
            )
            out_chunks.append(out_chunk)
            blocks[block_id] = op.partition_spec

        # build commit tree
        combine_size = 8
        chunks = out_chunks
        while len(chunks) >= combine_size:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i : i + combine_size]
                if len(chks) == 1:
                    chk = chks[0]
                else:
                    chk_op = DataFrameWriteTableCommit(
                        dtypes=op.dtypes, is_terminal=False
                    )
                    chk = chk_op.new_chunk(
                        chks,
                        shape=out_chunk_shape,
                        index_value=out_df.index_value,
                        dtypes=op.dtypes,
                    )
                new_chunks.append(chk)
            chunks = new_chunks

        assert len(chunks) < combine_size

        commit_table_op = DataFrameWriteTableCommit(
            dtypes=op.dtypes,
            table_name=op.table_name,
            blocks=blocks,
            cupid_handle=to_str(upload_session.handle),
            overwrite=op.overwrite,
            odps_params=op.odps_params,
            is_terminal=True,
        )
        commit_table_chunk = commit_table_op.new_chunk(
            chunks,
            shape=out_chunk_shape,
            dtypes=op.dtypes,
            index_value=out_df.index_value,
        )

        new_op = op.copy()
        return new_op.new_dataframes(
            op.inputs,
            shape=out_df.shape,
            index_value=out_df.index_value,
            dtypes=out_df.dtypes,
            columns_value=out_df.columns_value,
            chunks=[commit_table_chunk],
            nsplits=((0,),) * len(out_chunk_shape),
        )

    @classmethod
    def _tile_tunnel(cls, op):
        out_df = op.outputs[0]
        in_df = build_concatenated_rows_frame(op.inputs[0])

        out_chunks = []
        for chunk in in_df.chunks:
            chunk_op = DataFrameWriteTableSplit(
                dtypes=op.dtypes,
                table_name=op.table_name,
                odps_params=op.odps_params,
                partition_spec=op.partition_spec,
            )
            index_value = parse_index(chunk.index_value.to_pandas()[:0], chunk)
            out_chunk = chunk_op.new_chunk(
                [chunk],
                shape=(0, 0),
                index_value=index_value,
                columns_value=out_df.columns_value,
                dtypes=out_df.dtypes,
                index=chunk.index,
            )
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out_df.params.copy()
        params.update(
            dict(chunks=out_chunks, nsplits=((0,) * in_df.chunk_shape[0], (0,)))
        )
        return new_op.new_tileables([in_df], **params)

    @classmethod
    def tile(cls, op):
        from cupid.runtime import RuntimeContext

        if RuntimeContext.is_context_ready():
            return cls._tile_cupid(op)
        else:
            return cls._tile_tunnel(op)


class DataFrameWriteTableSplit(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123461

    _dtypes = SeriesField("dtypes")

    _table_name = StringField("table_name")
    _partition_spec = StringField("partition_spec")
    _cupid_handle = StringField("cupid_handle")
    _block_id = StringField("block_id")
    _write_batch_size = Int64Field("write_batch_size")
    _unknown_as_string = BoolField("unknown_as_string")

    # for tunnel
    _odps_params = DictField("odps_params")

    def __init__(
        self,
        dtypes=None,
        table_name=None,
        odps_params=None,
        partition_spec=None,
        cupid_handle=None,
        unknown_as_string=None,
        block_id=None,
        write_batch_size=None,
        **kw
    ):
        kw.update(_output_type_kw)
        super(DataFrameWriteTableSplit, self).__init__(
            _dtypes=dtypes,
            _table_name=table_name,
            _odps_params=odps_params,
            _partition_spec=partition_spec,
            _unknown_as_string=unknown_as_string,
            _cupid_handle=cupid_handle,
            _block_id=block_id,
            _write_batch_size=write_batch_size,
            **kw
        )

    @property
    def retryable(self):
        return False

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def table_name(self):
        return self._table_name

    @property
    def odps_params(self):
        return self._odps_params

    @property
    def unknown_as_string(self):
        return self._unknown_as_string

    @property
    def partition_spec(self):
        return self._partition_spec

    @property
    def cupid_handle(self):
        return self._cupid_handle

    @property
    def block_id(self):
        return self._block_id

    @property
    def write_batch_size(self):
        return self._write_batch_size

    @classmethod
    def _execute_in_cupid(cls, ctx, op):
        import pyarrow as pa
        import pandas as pd
        from ....df.backends.pd.types import pd_to_df_schema
        from cupid.io.table.core import BlockWriter

        to_store_data = ctx[op.inputs[0].key]

        odps_schema = pd_to_df_schema(
            to_store_data, unknown_as_string=op.unknown_as_string
        )
        project_name, table_name = op.table_name.split(".")
        block_writer = BlockWriter(
            _table_name=table_name,
            _project_name=project_name,
            _table_schema=odps_schema,
            _partition_spec=op.partition_spec,
            _block_id=op.block_id,
            _handle=op.cupid_handle,
        )
        logger.debug("Start writing table block, block id: %s", op.block_id)
        with block_writer.open_arrow_writer() as cupid_writer:

            sink = pa.BufferOutputStream()

            batch_size = op.write_batch_size or 1024
            batch_idx = 0
            batch_data = to_store_data[
                batch_size * batch_idx : batch_size * (batch_idx + 1)
            ]
            batch_data = convert_pandas_object_to_string(batch_data)
            schema = pa.RecordBatch.from_pandas(
                to_store_data[:1], preserve_index=False
            ).schema
            arrow_writer = pa.RecordBatchStreamWriter(sink, schema)
            while len(batch_data) > 0:
                batch = pa.RecordBatch.from_pandas(batch_data, preserve_index=False)
                arrow_writer.write_batch(batch)
                batch_idx += 1
                batch_data = to_store_data[
                    batch_size * batch_idx : batch_size * (batch_idx + 1)
                ]
            arrow_writer.close()
            cupid_writer.write(sink.getvalue())
        logger.debug("Write table block finished, block id: %s", op.block_id)

        block_writer.commit()
        ctx[op.outputs[0].key] = pd.DataFrame()

    @classmethod
    def _execute_arrow_tunnel(cls, ctx, op):
        from odps import ODPS
        from odps.tunnel import TableTunnel
        import pyarrow as pa
        import pandas as pd

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
            upload_session = tunnel.create_upload_session(
                t.name, partition_spec=op.partition_spec
            )
        else:
            upload_session = tunnel.create_upload_session(t.name)

        logger.debug(
            "Start writing table %s split index: %s", op.table_name, op.inputs[0].index
        )
        writer = upload_session.open_arrow_writer(0)
        arrow_rb = pa.RecordBatch.from_pandas(ctx[op.inputs[0].key])
        writer.write(arrow_rb)
        writer.close()
        upload_session.commit([0])
        logger.debug(
            "Finish writing table %s split index: %s", op.table_name, op.inputs[0].index
        )
        ctx[op.outputs[0].key] = pd.DataFrame()

    @classmethod
    def execute(cls, ctx, op):
        if op.cupid_handle is not None:
            cls._execute_in_cupid(ctx, op)
        else:
            cls._execute_arrow_tunnel(ctx, op)


class DataFrameWriteTableCommit(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123462

    _dtypes = SeriesField("dtypes")

    _odps_params = DictField("odps_params")
    _table_name = StringField("table_name")
    _overwrite = BoolField("overwrite")
    _blocks = DictField("blocks")
    _cupid_handle = StringField("cupid_handle")
    _is_terminal = BoolField("is_terminal")

    def __init__(
        self,
        dtypes=None,
        odps_params=None,
        table_name=None,
        blocks=None,
        cupid_handle=None,
        overwrite=False,
        is_terminal=None,
        **kw
    ):
        kw.update(_output_type_kw)
        super(DataFrameWriteTableCommit, self).__init__(
            _dtypes=dtypes,
            _odps_params=odps_params,
            _table_name=table_name,
            _blocks=blocks,
            _overwrite=overwrite,
            _cupid_handle=cupid_handle,
            _is_terminal=is_terminal,
            **kw
        )

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def table_name(self):
        return self._table_name

    @property
    def blocks(self):
        return self._blocks

    @property
    def overwrite(self):
        return self._overwrite

    @property
    def cupid_handle(self):
        return self._cupid_handle

    @property
    def odps_params(self):
        return self._odps_params

    @property
    def is_terminal(self):
        return self._is_terminal

    @classmethod
    def execute(cls, ctx, op):
        import pandas as pd
        from odps import ODPS
        from odps.accounts import BearerTokenAccount
        from cupid import CupidSession, context
        from cupid.io.table import CupidTableUploadSession

        if op.is_terminal:
            bearer_token = context().get_bearer_token()
            account = BearerTokenAccount(bearer_token)
            project = os.environ.get("ODPS_PROJECT_NAME", None)
            odps_params = op.odps_params.copy()
            if project:
                odps_params["project"] = project
            endpoint = (
                os.environ.get("ODPS_RUNTIME_ENDPOINT") or odps_params["endpoint"]
            )
            o = ODPS(
                None,
                None,
                account=account,
                project=odps_params["project"],
                endpoint=endpoint,
            )
            cupid_session = CupidSession(o)

            project_name, table_name = op.table_name.split(".")
            upload_session = CupidTableUploadSession(
                session=cupid_session,
                table_name=table_name,
                project_name=project_name,
                handle=op.cupid_handle,
                blocks=op.blocks,
            )
            upload_session.commit(overwrite=op.overwrite)

        ctx[op.outputs[0].key] = pd.DataFrame()


def write_odps_table(
    df,
    table,
    partition=None,
    overwrite=False,
    unknown_as_string=None,
    odps_params=None,
    write_batch_size=None,
):
    table_name = "%s.%s" % (table.project.name, table.name)
    op = DataFrameWriteTable(
        dtypes=df.dtypes,
        odps_params=odps_params,
        table_name=table_name,
        unknown_as_string=unknown_as_string,
        partition_spec=partition,
        over_write=overwrite,
        write_batch_size=write_batch_size,
    )
    return op(df)
