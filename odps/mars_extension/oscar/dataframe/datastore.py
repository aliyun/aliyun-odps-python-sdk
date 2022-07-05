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
import sys
import time
import uuid
import logging

import requests
from typing import List
from mars.core.context import get_context
from mars.oscar.errors import ActorNotExist
from mars.core import OutputType
from mars.dataframe.operands import DataFrameOperandMixin, DataFrameOperand
from mars.dataframe.utils import build_concatenated_rows_frame, parse_index
from mars.serialization.serializables import (
    StringField,
    SeriesField,
    BoolField,
    DictField,
    Int64Field,
)

from ....config import options
from ....utils import to_str
from ..cupid_service import CupidServiceClient

logger = logging.getLogger(__name__)

_output_type_kw = dict(_output_types=[OutputType.dataframe])


class DataFrameWriteTable(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123460

    dtypes = SeriesField("dtypes")

    odps_params = DictField("odps_params", default=None)
    table_name = StringField("table_name", default=None)
    partition_spec = StringField("partition_spec", default=None)
    overwrite = BoolField("overwrite", default=None)
    write_batch_size = Int64Field("write_batch_size", default=None)
    unknown_as_string = BoolField("unknown_as_string", default=None)

    def __init__(self, **kw):
        kw.update(_output_type_kw)
        super(DataFrameWriteTable, self).__init__(**kw)

    @property
    def retryable(self):
        return "CUPID_SERVICE_SOCKET" not in os.environ

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
        from mars.dataframe.utils import build_concatenated_rows_frame

        cupid_client = CupidServiceClient()
        upload_handle = cupid_client.create_table_upload_session(
            op.odps_params, op.table_name
        )

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
                odps_params=op.odps_params,
                unknown_as_string=op.unknown_as_string,
                partition_spec=op.partition_spec,
                cupid_handle=to_str(upload_handle),
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
            cupid_handle=to_str(upload_handle),
            overwrite=op.overwrite,
            odps_params=op.odps_params,
            is_terminal=True,
        )
        commit_table_chunk = commit_table_op.new_chunk(
            chunks,
            shape=out_chunk_shape,
            dtypes=op.dtypes,
            index_value=out_df.index_value,
            index=(0,) * len(out_chunk_shape),
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
        logger.info("Tile table %s[%s]", op.table_name, op.partition_spec)
        recorder_name = str(uuid.uuid4())
        out_chunks = []
        for chunk in in_df.chunks:
            chunk_op = DataFrameWriteTableSplit(
                dtypes=op.dtypes,
                table_name=op.table_name,
                odps_params=op.odps_params,
                partition_spec=op.partition_spec,
                commit_recorder_name=recorder_name,
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
        ctx = get_context()
        ctx.create_remote_object(recorder_name, _TunnelCommitRecorder, len(out_chunks))
        new_op = op.copy()
        params = out_df.params.copy()
        params.update(
            dict(chunks=out_chunks, nsplits=((0,) * in_df.chunk_shape[0], (0,)))
        )
        return new_op.new_tileables([in_df], **params)

    @classmethod
    def tile(cls, op):
        if "CUPID_SERVICE_SOCKET" in os.environ:
            return cls._tile_cupid(op)
        else:
            return cls._tile_tunnel(op)


class DataFrameWriteTableSplit(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123461

    dtypes = SeriesField("dtypes")
    table_name = StringField("table_name")
    partition_spec = StringField("partition_spec")
    cupid_handle = StringField("cupid_handle")
    block_id = StringField("block_id")
    write_batch_size = Int64Field("write_batch_size")
    unknown_as_string = BoolField("unknown_as_string")
    commit_recorder_name = StringField("commit_recorder_name")

    # for tunnel
    odps_params = DictField("odps_params")

    def __init__(self, **kw):
        kw.update(_output_type_kw)
        super(DataFrameWriteTableSplit, self).__init__(**kw)

    @property
    def retryable(self):
        return "CUPID_SERVICE_SOCKET" not in os.environ

    @classmethod
    def _execute_in_cupid(cls, ctx, op):
        import os

        import pandas as pd
        from odps import ODPS
        from odps.accounts import BearerTokenAccount

        cupid_client = CupidServiceClient()
        to_store_data = ctx[op.inputs[0].key]

        bearer_token = cupid_client.get_bearer_token()
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
        odps_schema = o.get_table(op.table_name).schema
        project_name, table_name = op.table_name.split(".")

        writer_config = dict(
            _table_name=table_name,
            _project_name=project_name,
            _table_schema=odps_schema,
            _partition_spec=op.partition_spec,
            _block_id=op.block_id,
            _handle=op.cupid_handle,
        )
        cupid_client.write_table_data(writer_config, to_store_data, op.write_batch_size)
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
        retry_times = options.retry_times
        init_sleep_secs = 1
        split_index = op.inputs[0].index
        logger.info(
            "Start creating upload session for table %s split index %s retry_times %s.",
            op.table_name,
            split_index,
            retry_times,
        )
        retries = 0
        while True:
            try:
                if op.partition_spec is not None:
                    upload_session = tunnel.create_upload_session(
                        t.name, partition_spec=op.partition_spec
                    )
                else:
                    upload_session = tunnel.create_upload_session(t.name)
                break
            except:
                if retries >= retry_times:
                    raise
                retries += 1
                sleep_secs = retries * init_sleep_secs
                logger.exception(
                    "Create upload session failed, sleep %s seconds and retry it",
                    sleep_secs,
                    exc_info=1,
                )
                time.sleep(sleep_secs)
        logger.info(
            "Start writing table %s split index: %s", op.table_name, split_index
        )
        retries = 0
        while True:
            try:
                writer = upload_session.open_arrow_writer(0)
                arrow_rb = pa.RecordBatch.from_pandas(ctx[op.inputs[0].key])
                writer.write(arrow_rb)
                writer.close()
                break
            except:
                if retries >= retry_times:
                    raise
                retries += 1
                sleep_secs = retries * init_sleep_secs
                logger.exception(
                    "Write data failed, sleep %s seconds and retry it",
                    sleep_secs,
                    exc_info=1,
                )
                time.sleep(sleep_secs)
        recorder_name = op.commit_recorder_name
        try:
            recorder = ctx.get_remote_object(recorder_name)
        except ActorNotExist:
            while True:
                logger.info(
                    "Writing table %s has been finished, waitting to be canceled by speculaitive scheduler",
                    op.table_name,
                )
                time.sleep(3)
        can_commit, can_destroy = recorder.try_commit(split_index)
        if can_commit:
            # FIXME If this commit failed or the process crashed, the whole write will still raise error.
            # But this situation is very rare so we skip the error handling.
            upload_session.commit([0])
            logger.info(
                "Finish writing table %s split index: %s", op.table_name, split_index
            )
        else:
            logger.info(
                "Skip writing table %s split index: %s", op.table_name, split_index
            )
        if can_destroy:
            try:
                ctx.destroy_remote_object(recorder_name)
                logger.info("Delete remote object %s", recorder_name)
            except ActorNotExist:
                pass
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


class _TunnelCommitRecorder:
    _commit_status: List[bool]

    def __init__(self, n_chunk: int):
        self._n_chunk = n_chunk
        self._commit_status = {}

    def try_commit(self, index: tuple):
        if index in self._commit_status:
            return False, len(self._commit_status) == self._n_chunk
        else:
            self._commit_status[index] = True
            return True, len(self._commit_status) == self._n_chunk


class DataFrameWriteTableCommit(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = 123462

    dtypes = SeriesField("dtypes")

    odps_params = DictField("odps_params")
    table_name = StringField("table_name")
    overwrite = BoolField("overwrite")
    blocks = DictField("blocks")
    cupid_handle = StringField("cupid_handle")
    is_terminal = BoolField("is_terminal")

    def __init__(self, **kw):
        kw.update(_output_type_kw)
        super(DataFrameWriteTableCommit, self).__init__(**kw)

    @classmethod
    def execute(cls, ctx, op):
        import pandas as pd
        from ..cupid_service import CupidServiceClient

        if op.is_terminal:
            odps_params = op.odps_params.copy()
            project = os.environ.get("ODPS_PROJECT_NAME", None)
            if project:
                odps_params["project"] = project

            client = CupidServiceClient()
            client.commit_table_upload_session(
                odps_params, op.table_name, op.cupid_handle, op.blocks, op.overwrite
            )

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
