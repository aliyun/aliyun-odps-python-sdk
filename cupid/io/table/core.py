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

import itertools
import json
import logging
import time
import warnings
from types import GeneratorType

from odps import options
from odps.models import Table, Schema
from odps.models.partition import Partition as TablePartition

from cupid.rpc import CupidRpcController, CupidTaskServiceRpcChannel, SandboxRpcChannel
from cupid.errors import CupidError
from cupid.session import CupidSession

try:
    from cupid.proto import cupid_task_service_pb2 as task_service_pb
    from cupid.proto import cupid_subprocess_service_pb2 as subprocess_pb
except TypeError:
    warnings.warn('Cannot import protos from pycupid: '
        'consider upgrading your protobuf python package.', ImportWarning)
    raise ImportError

logger = logging.getLogger(__name__)

ATTEMPT_FILE_PREFIX = 'attempt_'


class TableSplit(object):
    __slots__ = '_handle', '_split_index', '_split_file_start', '_split_file_end', \
                '_schema_file_start', '_schema_file_end', '_meta_row_count', \
                '_meta_raw_size'

    def __init__(self, **kwargs):
        if 'split_proto' in kwargs:
            split_pb = kwargs.pop('split_proto')
            self._split_index = split_pb.splitIndexId
            self._split_file_start = split_pb.splitFileStart
            self._split_file_end = split_pb.splitFileEnd
            self._schema_file_start = split_pb.schemaFileStart
            self._schema_file_end = split_pb.schemaFileEnd

        if kwargs.get('meta_proto'):
            meta_pb = kwargs.pop('meta_proto')
            try:
                self._meta_row_count = meta_pb.rowCount
                self._meta_raw_size = meta_pb.rawSize
            except AttributeError:
                pass

        for k in self.__slots__:
            if k in kwargs:
                setattr(self, k, kwargs[k])
            elif k.lstrip('_') in kwargs:
                setattr(self, k, kwargs[k.lstrip('_')])
            elif not hasattr(self, k):
                setattr(self, k, None)

    @property
    def handle(self):
        return self._handle

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
    def meta_row_count(self):
        return self._meta_row_count

    @property
    def meta_raw_size(self):
        return self._meta_raw_size

    @property
    def split_proto(self):
        return subprocess_pb.InputSplit(
            splitIndexId=self._split_index,
            splitFileStart=self._split_file_start,
            splitFileEnd=self._split_file_end,
            schemaFileStart=self._schema_file_start,
            schemaFileEnd=self._schema_file_end,
        )

    def _register_reader(self):
        channel = SandboxRpcChannel()
        stub = subprocess_pb.CupidSubProcessService_Stub(channel)

        req = subprocess_pb.RegisterTableReaderRequest(inputTableHandle=self._handle,
                                                       inputSplit=self.split_proto)
        controller = CupidRpcController()
        resp = stub.RegisterTableReader(controller, req, None)
        if controller.Failed():
            raise CupidError(controller.ErrorText())

        logger.info("RegisterTableReader response: %s", resp)
        logger.info("RegisterTableReaderResponse protobuf field size = %d", len(resp.ListFields()))

        schema_json = json.loads(resp.schema)
        partition_schema_json = json.loads(resp.partitionSchema) \
            if resp.HasField('partitionSchema') else None

        schema_names = [d['name'] for d in schema_json]
        schema_types = [d['type'] for d in schema_json]
        pt_schema_names = [d['name'] for d in partition_schema_json]
        pt_schema_types = [d['type'] for d in partition_schema_json]
        schema = Schema.from_lists(schema_names, schema_types, pt_schema_names, pt_schema_types)

        return resp.readIterator, schema

    def open_record_reader(self):
        from ...runtime import context
        context = context()

        read_iter, schema = self._register_reader()
        logger.debug('Obtained schema: %s', schema)
        return context.channel_client.create_record_reader(read_iter, schema)

    def open_pandas_reader(self):
        from ...runtime import context
        context = context()

        read_iter, schema = self._register_reader()
        logger.debug('Obtained schema: %s', schema)
        return context.channel_client.create_pandas_reader(read_iter, schema)

    def open_arrow_file_reader(self):
        from ...runtime import context
        import pyarrow as pa

        context = context()

        read_iter, schema = self._register_reader()

        params = dict(type='ReadByLabel', label=read_iter, arrow=True, batch=True)
        return context.channel_client.create_file_reader('createTableInputStream', json.dumps(params).encode())

    def open_arrow_reader(self):
        from ...runtime import context
        import pyarrow as pa

        context = context()

        read_iter, schema = self._register_reader()

        params = dict(type='ReadByLabel', label=read_iter, arrow=True, batch=True)
        stream = context.channel_client.create_file_reader('createTableInputStream', json.dumps(params).encode())
        return pa.RecordBatchStreamReader(stream)


class CupidTableDownloadSession(object):
    __slots__ = '_session', '_handle', '_splits',

    def __init__(self, **kwargs):
        for k in self.__slots__:
            if k in kwargs:
                setattr(self, k, kwargs[k])
            elif k.lstrip('_') in kwargs:
                setattr(self, k, kwargs[k.lstrip('_')])
            elif not hasattr(self, k):
                setattr(self, k, None)

    @property
    def splits(self):
        return self._splits

    def open_record_reader(self, split_id=0):
        return self._splits[split_id].open_record_reader()

    def open_pandas_reader(self, split_id=0):
        return self._splits[split_id].open_pandas_reader()


class BlockWriter(object):
    __slots__ = '_table_name', '_project_name', '_table_schema', '_partition_spec', '_block_id', '_handle'

    def __init__(self, **kwargs):
        for k in self.__slots__:
            if k in kwargs:
                setattr(self, k, kwargs[k])
            elif k.lstrip('_') in kwargs:
                setattr(self, k, kwargs[k.lstrip('_')])
            elif not hasattr(self, k):
                setattr(self, k, None)

    @property
    def table_name(self):
        return self._table_name

    @property
    def project_name(self):
        return self._project_name

    @property
    def block_id(self):
        return self._block_id

    @property
    def handle(self):
        return self._handle

    def new_record(self, values=None):
        from odps.models import Record
        return Record(schema=self._table_schema, values=values)

    def _register_writer(self, partition=None):
        if isinstance(partition, TablePartition):
            partition = str(partition.spec)

        controller = CupidRpcController()
        channel = SandboxRpcChannel()
        stub = subprocess_pb.CupidSubProcessService_Stub(channel)

        table_schema = self._table_schema
        schema_str = '|' + '|'.join(str(col.type) for col in table_schema.simple_columns)
        req = subprocess_pb.RegisterTableWriterRequest(
            outputTableHandle=self._handle,
            projectName=self._project_name,
            tableName=self._table_name,
            attemptFileName=ATTEMPT_FILE_PREFIX + self._block_id,
            partSpec=partition.replace("'", '') if partition else None,
            schema=schema_str,
        )
        resp = stub.RegisterTableWriter(controller, req, None)
        write_label = resp.subprocessWriteTableLabel
        return write_label

    def _open_writer(self, partition=None, create_method=None):
        from ...runtime import context
        context = context()

        write_label = self._register_writer(partition)
        writer = getattr(context.channel_client, create_method)(write_label, self._table_schema)
        writer._block_id = self._block_id
        writer._partition_spec = partition
        return writer

    def open_arrow_writer(self, partition=None):
        from ...runtime import context
        context = context()

        write_label = self._register_writer(partition or self._partition_spec)
        return context.channel_client.create_arrow_writer(write_label)

    def open_record_writer(self, partition=None):
        return self._open_writer(partition=partition or self._partition_spec, create_method='create_record_writer')

    def open_pandas_writer(self, partition=None):
        return self._open_writer(partition=partition or self._partition_spec, create_method='create_pandas_writer')

    def commit(self):
        channel = SandboxRpcChannel()
        stub = subprocess_pb.CupidSubProcessService_Stub(channel)

        commit_actions = [subprocess_pb.CommitActionInfo(
            commitFileName=self._block_id,
            attemptFileName=ATTEMPT_FILE_PREFIX + self._block_id,
            partSpec=self._partition_spec,
        )]

        req = subprocess_pb.CommitTableFilesRequest(
            outputTableHandle=self._handle,
            projectName=self._project_name,
            tableName=self._table_name,
            commitActionInfos=commit_actions,
        )

        controller = CupidRpcController()
        for _ in range(options.retry_times):
            stub.CommitTableFiles(controller, req, None)
            if controller.Failed():
                time.sleep(0.1)
                controller = CupidRpcController()
            else:
                break
        if controller.Failed():
            raise CupidError(controller.ErrorText())


class CupidTableUploadSession(object):
    __slots__ = '_session', '_table_name', '_project_name', '_handle', '_blocks'

    def __init__(self, **kwargs):
        self._blocks = dict()

        if 'blocks' in kwargs:
            blocks = kwargs.pop('blocks')
            if isinstance(blocks, dict):
                self._blocks.update(blocks)
            else:
                if not isinstance(blocks, (list, set, GeneratorType)):
                    blocks = [blocks]
                for bl in blocks:
                    if isinstance(bl, tuple):
                        self._blocks[bl[0]] = bl[1]
                    else:
                        self._blocks[bl] = None
        for k in self.__slots__:
            if k in kwargs:
                setattr(self, k, kwargs[k])
            elif k.lstrip('_') in kwargs:
                setattr(self, k, kwargs[k.lstrip('_')])
            elif not hasattr(self, k):
                setattr(self, k, None)

    @property
    def handle(self):
        return self._handle

    def commit(self, overwrite=False):
        partitions = list(set(p for p in self._blocks.values() if p is not None))
        if not partitions:
            partitions = ['']
        channel = CupidTaskServiceRpcChannel(self._session)
        stub = task_service_pb.CupidTaskService_Stub(channel)

        part_specs = [pt.replace("'", '') for pt in partitions]
        req = task_service_pb.CommitTableRequest(
            outputTableHandle=self._handle,
            projectName=self._project_name,
            tableName=self._table_name,
            isOverWrite=overwrite,
            lookupName=self._session.lookup_name,
            partSpecs=part_specs,
        )

        controller = CupidRpcController()
        resp = None
        for _ in range(options.retry_times):
            resp = stub.CommitTable(controller, req, None)
            if controller.Failed():
                time.sleep(0.1)
                controller = CupidRpcController()
            else:
                break
        if controller.Failed():
            raise CupidError(controller.ErrorText())

        logger.info(
            "[CupidTask] commitTable call, CurrentInstanceId: %s, "
            "request: %s, response: %s", self._session.lookup_name, req, resp,
        )


def create_download_session(session, table_or_parts, split_size=None, split_count=None,
                            columns=None, with_split_meta=False):
    channel = CupidTaskServiceRpcChannel(session)
    stub = task_service_pb.CupidTaskService_Stub(channel)

    if not isinstance(table_or_parts, (list, tuple, set, GeneratorType)):
        table_or_parts = [table_or_parts]

    if split_size is None and split_count is None:
        split_count = 1
    split_count = split_count or 0
    split_size = (split_size or 1024 ** 2) // 1024 ** 2

    table_pbs = []
    for t in table_or_parts:
        if isinstance(t, Table):
            if not columns:
                columns = t.schema.names
            table_kw = dict(
                projectName=t.project.name,
                tableName=t.name,
                columns=','.join(columns),
            )
        elif isinstance(t, TablePartition):
            if not columns:
                columns = t.table.schema.names
            table_kw = dict(
                projectName=t.table.project.name,
                tableName=t.table.name,
                columns=','.join(columns),
                partSpec=str(t.partition_spec).replace("'", '').strip(),
            )
        else:
            raise NotImplementedError
        table_pbs.append(task_service_pb.TableInputInfo(**table_kw))

    request = task_service_pb.SplitTablesRequest(
        lookupName=session.lookup_name,
        splitSize=split_size,
        splitCount=split_count,
        tableInputInfos=table_pbs,
        allowNoColumns=True,
        requireSplitMeta=with_split_meta,
    )

    controller = CupidRpcController()
    resp = stub.SplitTables(controller, request, None)
    if controller.Failed():
        raise CupidError(controller.ErrorText())
    logger.info(
        "[CupidTask] splitTables call, CurrentInstanceId: %s, "
        "request: %s, response: %s" % (
            session.lookup_name, str(request), str(resp),
        )
    )
    handle = resp.inputTableHandle

    channel = SandboxRpcChannel()
    stub = subprocess_pb.CupidSubProcessService_Stub(channel)

    if not with_split_meta:
        split_meta = itertools.repeat(None)
    else:
        req = subprocess_pb.GetSplitsMetaRequest(
            inputTableHandle=handle,
        )
        controller = CupidRpcController()
        resp = stub.GetSplitsMeta(controller, req, None)
        logger.info(
            "[CupidTask] getSplitsMeta call, CurrentInstanceId: %s, "
            "request: %s, response: %s" % (
                session.lookup_name, str(request), str(resp),
            )
        )
        if controller.Failed():
            split_meta = itertools.repeat(None)
            logger.warning('Failed to get results of getSplitsMeta, '
                        'may running on an old service')
        else:
            split_meta = resp.inputSplitsMeta

    req = subprocess_pb.GetSplitsRequest(inputTableHandle=handle)
    controller = CupidRpcController()
    resp = stub.GetSplits(controller, req, None)
    if controller.Failed():
        raise CupidError(controller.ErrorText())

    input_splits = []
    for info, meta in zip(resp.inputSplits, split_meta):
        input_splits.append(TableSplit(
            split_proto=info, meta_proto=meta, handle=handle, columns=columns))
    logger.info(
        "[SubProcess] getSplits call, CurrentInstanceId: %s, "
        "request: %s, response: %s" % (
            session.lookup_name,
            str(req), str(resp),
        )
    )
    return CupidTableDownloadSession(session=session, handle=handle, splits=input_splits)


def create_upload_session(session, table):
    controller = CupidRpcController()
    channel = CupidTaskServiceRpcChannel(session)
    stub = task_service_pb.CupidTaskService_Stub(channel)

    req = task_service_pb.WriteTableRequest(lookupName=session.lookup_name, tableName=table.name,
                                            projectName=table.project.name)
    resp = stub.WriteTable(controller, req, None)
    if controller.Failed():
        raise CupidError(controller.ErrorText())
    logger.info(
        "[CupidTask] writeTable call, CurrentInstanceId: %s, "
        "request: %s, response: %s", session.lookup_name, req, resp,
    )
    return CupidTableUploadSession(
        session=session, table_name=table.name, project_name=table.project.name, handle=resp.outputTableHandle)


def query_table_meta(session, table):
    controller = CupidRpcController()
    channel = CupidTaskServiceRpcChannel(session)
    stub = task_service_pb.CupidTaskService_Stub(channel)

    table_info = task_service_pb.TableInfo(projectName=table.project.name, tableName=table.name)
    req = task_service_pb.GetTableMetaRequest(lookupName=session.lookup_name, tableInfo=table_info,
                                              needContent=True, uploadFile='')
    resp = stub.GetTableMeta(controller, req, None)
    if controller.Failed():
        raise CupidError(controller.ErrorText())
    logger.info(
        "[CupidTask] getTableMeta call, CurrentInstanceId: %s, "
        "request: %s, response: %s", session.lookup_name, req, resp,
    )
    return json.loads(resp.getTableMetaContent)


CupidSession.create_download_session = create_download_session
CupidSession.create_upload_session = create_upload_session
CupidSession.query_table_meta = query_table_meta
