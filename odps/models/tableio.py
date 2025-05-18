# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

import copy
import functools
import itertools
import logging
import multiprocessing
import os
import socket
import struct
import sys
import threading
import uuid
import warnings
from collections import OrderedDict, defaultdict
from types import GeneratorType, MethodType

try:
    import pyarrow as pa
except (AttributeError, ImportError):
    pa = None
try:
    import pandas as pd
except ImportError:
    pd = None

from .. import errors
from .. import types as odps_types
from .. import utils
from ..compat import Iterable, six
from ..config import options
from ..dag import DAG
from ..lib import cloudpickle
from ..lib.tblib import pickling_support
from .readers import TunnelArrowReader, TunnelRecordReader

logger = logging.getLogger(__name__)
pickling_support.install()

_GET_NEXT_BLOCK_CMD = 0x01
_PUT_WRITTEN_BLOCKS_CMD = 0x02
_SERVER_ERROR_CMD = 0xFD
_STOP_SERVER_CMD = 0xFE


if sys.version_info[0] == 2:
    _ord_if_possible = ord

    def _load_classmethod(cls, func_name):
        return getattr(cls, func_name)

    class InstanceMethodWrapper(object):
        """Trick for classmethods under Python 2.7 to be pickleable"""

        def __init__(self, func):
            assert isinstance(func, MethodType)
            assert isinstance(func.im_self, type)
            self._func = func

        def __call__(self, *args, **kw):
            return self._func()

        def __reduce__(self):
            return _load_classmethod, (self._func.im_self, self._func.im_func.__name__)

    _wrap_classmethod = InstanceMethodWrapper
else:

    def _ord_if_possible(x):
        return x

    def _wrap_classmethod(x):
        return x


class SpawnedTableReaderMixin(object):
    @property
    def schema(self):
        return self._parent.table_schema

    @staticmethod
    def _read_table_split(
        conn,
        download_id,
        start,
        count,
        idx,
        rest_client=None,
        project=None,
        table_name=None,
        partition_spec=None,
        tunnel_endpoint=None,
        quota_name=None,
        columns=None,
        arrow=False,
        schema_name=None,
        append_partitions=None,
    ):
        # read part data
        from ..tunnel import TableTunnel

        try:
            tunnel = TableTunnel(
                client=rest_client,
                project=project,
                endpoint=tunnel_endpoint,
                quota_name=quota_name,
            )
            session = utils.call_with_retry(
                tunnel.create_download_session,
                table_name,
                schema=schema_name,
                download_id=download_id,
                partition_spec=partition_spec,
            )

            def _data_to_pandas():
                if not arrow:
                    with session.open_record_reader(
                        start,
                        count,
                        columns=columns,
                        append_partitions=append_partitions,
                    ) as reader:
                        return reader.to_pandas()
                else:
                    with session.open_arrow_reader(
                        start,
                        count,
                        columns=columns,
                        append_partitions=append_partitions,
                    ) as reader:
                        return reader.to_pandas()

            data = utils.call_with_retry(_data_to_pandas)
            conn.send((idx, data, True))
        except:
            try:
                conn.send((idx, sys.exc_info(), False))
            except:
                logger.exception("Failed to write in process %d", idx)
                raise

    def _get_process_split_reader(self, columns=None, append_partitions=None):
        rest_client = self._parent._client
        table_name = self._parent.name
        schema_name = self._parent.get_schema()
        project = self._parent.project.name
        tunnel_endpoint = self._download_session._client.endpoint
        quota_name = self._download_session._quota_name
        partition_spec = self._partition_spec

        return functools.partial(
            self._read_table_split,
            rest_client=rest_client,
            project=project,
            table_name=table_name,
            partition_spec=partition_spec,
            tunnel_endpoint=tunnel_endpoint,
            quota_name=quota_name,
            arrow=isinstance(self, TunnelArrowReader),
            columns=columns or self._column_names,
            schema_name=schema_name,
            append_partitions=append_partitions,
        )


class TableRecordReader(SpawnedTableReaderMixin, TunnelRecordReader):
    def __init__(
        self,
        table,
        download_session,
        partition_spec=None,
        columns=None,
        append_partitions=True,
    ):
        super(TableRecordReader, self).__init__(
            table,
            download_session,
            columns=columns,
            append_partitions=append_partitions,
        )
        self._partition_spec = partition_spec


class TableArrowReader(SpawnedTableReaderMixin, TunnelArrowReader):
    def __init__(
        self,
        table,
        download_session,
        partition_spec=None,
        columns=None,
        append_partitions=False,
    ):
        super(TableArrowReader, self).__init__(
            table,
            download_session,
            columns=columns,
            append_partitions=append_partitions,
        )
        self._partition_spec = partition_spec


class MPBlockServer(object):
    def __init__(self, writer):
        self._writer = writer
        self._sock = None
        self._serve_thread_obj = None
        self._authkey = multiprocessing.current_process().authkey

    @property
    def address(self):
        return self._sock.getsockname() if self._sock else None

    @property
    def authkey(self):
        return self._authkey

    def _serve_thread(self):
        while True:
            data, addr = self._sock.recvfrom(4096)
            try:
                pos = len(self._authkey)
                assert data[:pos] == self._authkey, "Authentication key mismatched!"

                cmd_code = _ord_if_possible(data[pos])
                pos += 1
                if cmd_code == _GET_NEXT_BLOCK_CMD:
                    block_id = self._writer._gen_next_block_id()
                    data = struct.pack("<B", _GET_NEXT_BLOCK_CMD) + struct.pack(
                        "<I", block_id
                    )
                    self._sock.sendto(data, addr)
                elif cmd_code == _PUT_WRITTEN_BLOCKS_CMD:
                    blocks_queue = self._writer._used_block_id_queue
                    (count,) = struct.unpack("<H", data[pos : pos + 2])
                    pos += 2
                    assert 4 * count < len(data), "Data too short for block count!"
                    block_ids = struct.unpack(
                        "<%dI" % count, data[pos : pos + 4 * count]
                    )
                    blocks_queue.put(block_ids)
                    self._sock.sendto(struct.pack("<B", _PUT_WRITTEN_BLOCKS_CMD), addr)
                elif cmd_code == _STOP_SERVER_CMD:
                    assert (
                        addr[0] == self._sock.getsockname()[0]
                    ), "Cannot stop server from other hosts!"
                    break
                else:  # pragma: no cover
                    raise AssertionError("Unrecognized command %x", cmd_code)
            except BaseException:
                pk_exc_info = cloudpickle.dumps(sys.exc_info())
                data = (
                    struct.pack("<B", _SERVER_ERROR_CMD)
                    + struct.pack("<I", len(pk_exc_info))
                    + pk_exc_info
                )
                self._sock.sendto(data, addr)
                logger.exception("Serve thread error.")

        self._sock.close()
        self._sock = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))
        self._serve_thread_obj = threading.Thread(target=self._serve_thread)
        self._serve_thread_obj.daemon = True
        self._serve_thread_obj.start()

    def stop(self):
        stop_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        stop_data = self._authkey + struct.pack("<B", _STOP_SERVER_CMD)
        stop_sock.sendto(stop_data, self._sock.getsockname())
        stop_sock.close()
        self._serve_thread_obj.join()


class MPBlockClient(object):
    _MAX_BLOCK_COUNT = 256

    def __init__(self, address, authkey):
        self._addr = address
        self._authkey = authkey
        self._sock = None

    def __del__(self):
        self.close()

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _get_socket(self):
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._sock

    @staticmethod
    def _reraise_remote_error(recv_data):
        if recv_data[0] != _SERVER_ERROR_CMD:
            return
        (pk_len,) = struct.unpack("<I", recv_data[1:5])
        exc_info = cloudpickle.loads(recv_data[5 : 5 + pk_len])
        six.reraise(*exc_info)

    def get_next_block_id(self):
        sock = self._get_socket()
        data = self._authkey + struct.pack("<B", _GET_NEXT_BLOCK_CMD)
        sock.sendto(data, self._addr)
        recv_data, server_addr = sock.recvfrom(1024)
        self._reraise_remote_error(recv_data)
        assert _ord_if_possible(recv_data[0]) == _GET_NEXT_BLOCK_CMD
        assert self._addr == server_addr
        (count,) = struct.unpack("<I", recv_data[1:5])
        return count

    def put_written_blocks(self, block_ids):
        sock = self._get_socket()
        for pos in range(0, len(block_ids), self._MAX_BLOCK_COUNT):
            sub_block_ids = block_ids[pos : pos + self._MAX_BLOCK_COUNT]
            data = (
                self._authkey
                + struct.pack("<B", _PUT_WRITTEN_BLOCKS_CMD)
                + struct.pack("<H", len(sub_block_ids))
                + struct.pack("<%dI" % len(sub_block_ids), *sub_block_ids)
            )
            sock.sendto(data, self._addr)
            recv_data, server_addr = sock.recvfrom(1024)
            self._reraise_remote_error(recv_data)
            assert _ord_if_possible(recv_data[0]) == _PUT_WRITTEN_BLOCKS_CMD
            assert self._addr == server_addr


class AbstractTableWriter(object):
    def __init__(
        self, table, upload_session, blocks=None, commit=True, on_close=None, **kwargs
    ):
        self._table = table
        self._upload_session = upload_session
        self._commit = commit
        self._closed = False
        self._on_close = on_close

        self._use_buffered_writer = None
        if blocks is not None:
            self._use_buffered_writer = False

        # block writer options
        self._blocks = blocks or upload_session.blocks or [0]
        self._blocks_writes = [False] * len(self._blocks)
        self._blocks_writers = [None] * len(self._blocks)

        for block in upload_session.blocks or ():
            self._blocks_writes[self._blocks.index(block)] = True

        # buffered writer options
        self._thread_to_buffered_writers = dict()

        # objects for cross-process sharings
        self._mp_server = None
        self._main_pid = kwargs.get("main_pid") or os.getpid()
        self._mp_fixed = kwargs.get("mp_fixed")
        if kwargs.get("mp_client"):
            self._mp_client = kwargs["mp_client"]
            self._mp_context = self._block_id_counter = self._used_block_id_queue = None
        else:
            self._mp_client = self._mp_authkey = None
            self._mp_context = kwargs.get("mp_context") or multiprocessing
            self._block_id_counter = kwargs.get(
                "block_id_counter"
            ) or self._mp_context.Value("i", 1 + max(upload_session.blocks or [0]))
            self._used_block_id_queue = (
                kwargs.get("used_block_id_queue") or self._mp_context.Queue()
            )

    @classmethod
    def _restore_subprocess_writer(
        cls,
        mp_server_address,
        mp_server_auth,
        upload_id,
        main_pid=None,
        blocks=None,
        rest_client=None,
        project=None,
        table_name=None,
        partition_spec=None,
        tunnel_endpoint=None,
        quota_name=None,
        schema=None,
    ):
        from ..core import ODPS
        from ..tunnel import TableTunnel

        odps_entry = ODPS(
            account=rest_client.account,
            app_account=rest_client.app_account,
            endpoint=rest_client.endpoint,
            overwrite_global=False,
        )
        tunnel = TableTunnel(
            client=rest_client,
            project=project,
            endpoint=tunnel_endpoint,
            quota_name=quota_name,
        )
        table = odps_entry.get_table(table_name, schema=schema, project=project)
        session = utils.call_with_retry(
            tunnel.create_upload_session,
            table_name,
            schema=schema,
            upload_id=upload_id,
            partition_spec=partition_spec,
        )
        mp_client = MPBlockClient(mp_server_address, mp_server_auth)
        writer = cls(
            table,
            session,
            commit=False,
            blocks=blocks,
            main_pid=main_pid,
            mp_fixed=True,
            mp_client=mp_client,
        )
        return writer

    def _start_mp_server(self):
        if self._mp_server is not None:
            return
        self._mp_server = MPBlockServer(self)
        self._mp_server.start()
        # replace mp queue with ordinary queue
        self._used_block_id_queue = six.moves.queue.Queue()

    def __reduce__(self):
        rest_client = self._table._client
        table_name = self._table.name
        schema_name = self._table.get_schema()
        project = self._table.project.name
        tunnel_endpoint = self._upload_session._client.endpoint
        quota_name = self._upload_session._quota_name
        partition_spec = self._upload_session._partition_spec
        blocks = None if self._use_buffered_writer is not False else self._blocks

        self._start_mp_server()

        return _wrap_classmethod(self._restore_subprocess_writer), (
            self._mp_server.address,
            bytes(self._mp_server.authkey),
            self.upload_id,
            self._main_pid,
            blocks,
            rest_client,
            project,
            table_name,
            partition_spec,
            tunnel_endpoint,
            quota_name,
            schema_name,
        )

    @property
    def upload_id(self):
        return self._upload_session.id

    @property
    def schema(self):
        return self._table.table_schema

    @property
    def status(self):
        return self._upload_session.status

    def _open_writer(self, block_id, compress):
        raise NotImplementedError

    def _write_contents(self, writer, *args):
        raise NotImplementedError

    def _gen_next_block_id(self):
        if self._mp_client is not None:
            return self._mp_client.get_next_block_id()
        with self._block_id_counter.get_lock():
            block_id = self._block_id_counter.value
            self._block_id_counter.value += 1
            return block_id

    def _fix_mp_attributes(self):
        if not self._mp_fixed:
            self._mp_fixed = True
            if os.getpid() == self._main_pid:
                return
            self._commit = False
            self._on_close = None

    def write(self, *args, **kwargs):
        if self._closed:
            raise IOError("Cannot write to a closed writer.")
        self._fix_mp_attributes()

        compress = kwargs.get("compress", False)

        block_id = kwargs.get("block_id")
        if block_id is None:
            if type(args[0]) in six.integer_types:
                block_id = args[0]
                args = args[1:]
            else:
                block_id = 0 if options.tunnel.use_block_writer_by_default else None

        use_buffered_writer = block_id is None
        if self._use_buffered_writer is None:
            self._use_buffered_writer = use_buffered_writer
        elif self._use_buffered_writer is not use_buffered_writer:
            raise ValueError(
                "Cannot mix block writing mode with non-block writing mode within a single writer"
            )

        if use_buffered_writer:
            idx = None
            writer = self._thread_to_buffered_writers.get(
                threading.current_thread().ident
            )
        else:
            idx = self._blocks.index(block_id)
            writer = self._blocks_writers[idx]

        if writer is None:
            writer = self._open_writer(block_id, compress)

        self._write_contents(writer, *args)
        if not use_buffered_writer:
            self._blocks_writes[idx] = True

    def close(self):
        if self._closed:
            return

        written_blocks = []
        if self._use_buffered_writer:
            for writer in self._thread_to_buffered_writers.values():
                writer.close()
                written_blocks.extend(writer.get_blocks_written())
        else:
            for writer in self._blocks_writers:
                if writer is not None:
                    writer.close()
            written_blocks = [
                block
                for block, block_write in zip(self._blocks, self._blocks_writes)
                if block_write
            ]

        if written_blocks:
            if self._mp_client is not None:
                self._mp_client.put_written_blocks(written_blocks)
            else:
                self._used_block_id_queue.put(written_blocks)

        if self._commit:
            collected_blocks = []
            # as queue.empty() not reliable, we need to fill local blocks manually
            collected_blocks.extend(written_blocks)
            while not self._used_block_id_queue.empty():
                collected_blocks.extend(self._used_block_id_queue.get())
            collected_blocks.extend(self._upload_session.blocks or [])
            collected_blocks = sorted(set(collected_blocks))
            self._upload_session.commit(collected_blocks)

        if callable(self._on_close):
            self._on_close()

        if self._mp_client is not None:
            self._mp_client.close()
            self._mp_client = None
        if self._mp_server is not None:
            self._mp_server.stop()
            self._mp_server = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if an error occurs inside the with block, we do not commit
        if exc_val is not None:
            return
        self.close()


class ToRecordsMixin(object):
    def _new_record(self, arg):
        raise NotImplementedError

    def _to_records(self, *args):
        def convert_records(arg, sample_rec):
            if odps_types.is_record(sample_rec):
                return arg
            elif isinstance(sample_rec, (list, tuple)):
                return (self._new_record(vals) for vals in arg)
            else:
                return [self._new_record(arg)]

        if len(args) == 0:
            return
        if len(args) > 1:
            args = [args]

        arg = args[0]
        if odps_types.is_record(arg):
            return [arg]
        elif isinstance(arg, (list, tuple)):
            return convert_records(arg, arg[0])
        elif isinstance(arg, GeneratorType):
            try:
                # peek the first element and then put back
                next_arg = six.next(arg)
                chained = itertools.chain((next_arg,), arg)
                return convert_records(chained, next_arg)
            except StopIteration:
                return ()
        else:
            raise ValueError("Unsupported record type.")


class TableRecordWriter(ToRecordsMixin, AbstractTableWriter):
    def _open_writer(self, block_id, compress):
        if self._use_buffered_writer:
            writer = self._upload_session.open_record_writer(
                compress=compress,
                initial_block_id=self._gen_next_block_id(),
                block_id_gen=self._gen_next_block_id,
            )
            thread_ident = threading.current_thread().ident
            self._thread_to_buffered_writers[thread_ident] = writer
        else:
            writer = self._upload_session.open_record_writer(
                block_id, compress=compress
            )
            self._blocks_writers[block_id] = writer
        return writer

    def _new_record(self, arg):
        return self._upload_session.new_record(arg)

    def _write_contents(self, writer, *args):
        for record in self._to_records(*args):
            writer.write(record)


class TableArrowWriter(AbstractTableWriter):
    def _open_writer(self, block_id, compress):
        if self._use_buffered_writer:
            writer = self._upload_session.open_arrow_writer(
                compress=compress,
                initial_block_id=self._gen_next_block_id(),
                block_id_gen=self._gen_next_block_id,
            )
            thread_ident = threading.current_thread().ident
            self._thread_to_buffered_writers[thread_ident] = writer
        else:
            writer = self._upload_session.open_arrow_writer(block_id, compress=compress)
            self._blocks_writers[block_id] = writer
        return writer

    def _write_contents(self, writer, *args):
        for arg in args:
            writer.write(arg)


class TableUpsertWriter(ToRecordsMixin):
    def __init__(self, table, upsert_session, commit=True, on_close=None, **_):
        self._table = table
        self._upsert_session = upsert_session
        self._closed = False
        self._commit = commit
        self._on_close = on_close

        self._upsert = None

    def _open_upsert(self, compress):
        self._upsert = self._upsert_session.open_upsert_stream(compress=compress)

    def _new_record(self, arg):
        return self._upsert_session.new_record(arg)

    def _write(self, *args, **kw):
        compress = kw.pop("compress", None)
        delete = kw.pop("delete", False)
        if not self._upsert:
            self._open_upsert(compress)

        for record in self._to_records(*args):
            if delete:
                self._upsert.delete(record)
            else:
                self._upsert.upsert(record)

    def write(self, *args, **kw):
        compress = kw.pop("compress", None)
        self._write(*args, compress=compress, delete=False)

    def delete(self, *args, **kw):
        compress = kw.pop("compress", None)
        self._write(*args, compress=compress, delete=True)

    def close(self, success=True, commit=True):
        if not success:
            self._upsert_session.abort()
        else:
            self._upsert.close()
            if commit and self._commit:
                self._upsert_session.commit()

        if callable(self._on_close):
            self._on_close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if an error occurs inside the with block, we do not commit
        if exc_val is not None:
            self.close(success=False, commit=False)
            return
        self.close()


class TableIOMethods(object):
    @classmethod
    def _get_table_obj(cls, odps, name, project=None, schema=None):
        if isinstance(name, six.string_types) and "." in name:
            project, schema, name = odps._split_object_dots(name)

        if not isinstance(name, six.string_types):
            if name.get_schema():
                schema = name.get_schema().name
            project, name = name.project.name, name.name

        parent = odps._get_project_or_schema(project, schema)
        return parent.tables[name]

    @classmethod
    def read_table(
        cls,
        odps,
        name,
        limit=None,
        start=0,
        step=None,
        project=None,
        schema=None,
        partition=None,
        **kw
    ):
        """
        Read table's records.

        :param name: table or table name
        :type name: :class:`odps.models.table.Table` or str
        :param limit:  the records' size, if None will read all records from the table
        :param start:  the record where read starts with
        :param step:  default as 1
        :param project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param partition: the partition of this table to read
        :param list columns: the columns' names which are the parts of table's columns
        :param bool compress: if True, the data will be compressed during downloading
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param endpoint: tunnel service URL
        :param reopen: reading the table will reuse the session which opened last time,
                       if set to True will open a new download session, default as False
        :return: records
        :rtype: generator

        :Example:

        >>> for record in odps.read_table('test_table', 100):
        >>>     # deal with such 100 records
        >>> for record in odps.read_table('test_table', partition='pt=test', start=100, limit=100):
        >>>     # read the `pt=test` partition, skip 100 records and read 100 records

        .. seealso:: :class:`odps.models.Record`
        """

        table = cls._get_table_obj(odps, name, project=project, schema=schema)

        compress = kw.pop("compress", False)
        columns = kw.pop("columns", None)

        with table.open_reader(partition=partition, **kw) as reader:
            for record in reader.read(
                start, limit, step=step, compress=compress, columns=columns
            ):
                yield record

    @classmethod
    def _is_pa_collection(cls, obj):
        return pa is not None and isinstance(obj, (pa.Table, pa.RecordBatch))

    @classmethod
    def _is_pd_df(cls, obj):
        return pd is not None and isinstance(obj, pd.DataFrame)

    @classmethod
    def _resolve_schema(
        cls,
        records_list=None,
        data_schema=None,
        unknown_as_string=False,
        partition=None,
        partition_cols=None,
        type_mapping=None,
        infer_type_with_arrow=False,
    ):
        from ..df.backends.odpssql.types import df_schema_to_odps_schema
        from ..df.backends.pd.types import pd_to_df_schema
        from ..tunnel.io.types import arrow_schema_to_odps_schema

        type_mapping = type_mapping or {}
        type_mapping = {
            k: odps_types.validate_data_type(v) for k, v in type_mapping.items()
        }
        if records_list is not None:
            if cls._is_pa_collection(records_list):
                data_schema = arrow_schema_to_odps_schema(records_list.schema)
            elif cls._is_pd_df(records_list):
                data_schema = df_schema_to_odps_schema(
                    pd_to_df_schema(
                        records_list,
                        unknown_as_string=unknown_as_string,
                        type_mapping=type_mapping,
                        infer_type_with_arrow=infer_type_with_arrow,
                    )
                )
            elif isinstance(records_list, list) and odps_types.is_record(
                records_list[0]
            ):
                data_schema = odps_types.OdpsSchema(records_list[0]._columns)
            else:
                raise TypeError(
                    "Inferring schema from provided data not implemented. "
                    "You need to supply a pandas DataFrame or records."
                )
        assert data_schema is not None

        part_col_names = partition_cols or []
        if partition is not None:
            part_spec = odps_types.PartitionSpec(partition)
            part_col_names.extend(k for k in part_spec.keys())
        if part_col_names:
            part_col_set = set(part_col_names)
            simple_cols = [c for c in data_schema.columns if c.name not in part_col_set]
            part_cols = [
                odps_types.Column(n, odps_types.string) for n in part_col_names
            ]
            data_schema = odps_types.OdpsSchema(simple_cols, part_cols)

        if not type_mapping:
            return data_schema

        simple_cols, part_cols = [], []
        unmapped_cols = set(type_mapping.keys())
        for col in data_schema.columns:
            if col.name not in type_mapping:
                simple_cols.append(col)
            else:
                unmapped_cols.remove(col.name)
                simple_cols.append(col.replace(type=type_mapping[col.name]))
        for col in getattr(data_schema, "partitions", None) or ():
            if col.name not in type_mapping:
                part_cols.append(col)
            else:
                unmapped_cols.remove(col.name)
                part_cols.append(col.replace(type=type_mapping[col.name]))

        for col_name in unmapped_cols:
            simple_cols.append(
                odps_types.Column(name=col_name, type=type_mapping[col_name])
            )
        return odps_types.OdpsSchema(simple_cols, part_cols or None)

    @classmethod
    def _calc_schema_diff(cls, src_schema, dest_schema, partition_cols=None):
        if not src_schema or not dest_schema:
            return [], []
        union_cols, diff_cols = [], []
        part_col_set = set(partition_cols or [])
        # collect union columns in the order of dest schema
        for col in dest_schema.simple_columns:
            if col.name in src_schema:
                union_cols.append(col)
        # collect columns not in dest schema
        for col in src_schema.simple_columns:
            if col.name not in dest_schema and col.name not in part_col_set:
                diff_cols.append(col)
        return union_cols, diff_cols

    @classmethod
    def _check_partition_specified(
        cls, table_name, table_schema, partition_cols=None, partition=None
    ):
        partition_cols = partition_cols or []
        no_eval_set = set(n.lower() for n in partition_cols)
        if partition:
            no_eval_set.update(
                c.lower() for c in odps_types.PartitionSpec(partition).keys()
            )
        expr_cols = [
            c.name.lower()
            for c in table_schema.partitions
            if c.generate_expression and c.name.lower() not in no_eval_set
        ]
        partition_cols += expr_cols
        if not partition_cols and not partition:
            if table_schema.partitions:
                raise ValueError(
                    "Partition spec is required for table %s with partitions, "
                    "please specify a partition with `partition` argument or "
                    "specify a list of columns with `partition_cols` argument "
                    "to enable dynamic partitioning." % table_name
                )
            return partition_cols
        else:
            if not table_schema.partitions:
                raise ValueError(
                    "Cannot store into a non-partitioned table %s when `partition` "
                    "or `partition_cols` is specified." % table_name
                )
            all_parts = (
                [n.lower() for n in odps_types.PartitionSpec(partition).keys()]
                if partition
                else []
            )
            if partition_cols:
                all_parts.extend(partition_cols)
            req_all_parts_set = set(all_parts)
            table_all_parts = [c.name.lower() for c in table_schema.partitions]
            no_exist_parts = req_all_parts_set - set(table_all_parts)
            if no_exist_parts:
                raise ValueError(
                    "Partitions %s are not in table %s whose partitions are (%s)."
                    % (sorted(no_exist_parts), table_name, table_all_parts)
                )
            no_specified_parts = set(table_all_parts) - req_all_parts_set
            if no_specified_parts:
                raise ValueError(
                    "Partitions %s in table %s are not specified in `partition_cols` or "
                    "`partition` argument." % (sorted(no_specified_parts), table_name)
                )
            return partition_cols

    @classmethod
    def _get_ordered_col_expressions(cls, table, partition):
        """
        Get column expressions in topological order
        by variable dependencies
        """
        part_spec = odps_types.PartitionSpec(partition)
        col_to_expr = {
            c.name.lower(): table._get_column_generate_expression(c.name)
            for c in table.table_schema.columns
            if c.name not in part_spec
        }
        col_to_expr = {c: expr for c, expr in col_to_expr.items() if expr}
        if not col_to_expr:
            # no columns with expressions, quit
            return {}
        col_dag = DAG()
        for col in col_to_expr:
            col_dag.add_node(col)
            for ref in col_to_expr[col].references:
                ref_col_name = ref.lower()
                col_dag.add_node(ref_col_name)
                col_dag.add_edge(ref_col_name, col)

        out_col_to_expr = OrderedDict()
        for col in col_dag.topological_sort():
            if col not in col_to_expr:
                continue
            out_col_to_expr[col] = col_to_expr[col]
        return out_col_to_expr

    @classmethod
    def _fill_missing_expressions(cls, data, col_to_expr):
        def handle_recordbatch(batch):
            col_names = list(batch.schema.names)
            col_arrays = list(batch.columns)
            for col in missing_cols:
                col_names.append(col)
                col_arrays.append(col_to_expr[col].eval(batch))
            return pa.RecordBatch.from_arrays(col_arrays, col_names)

        if pa and isinstance(data, (pa.Table, pa.RecordBatch)):
            col_name_set = set(c.lower() for c in data.schema.names)
            missing_cols = [c for c in col_to_expr if c not in col_name_set]
            if not missing_cols:
                return data
            if isinstance(data, pa.Table):
                batches = [handle_recordbatch(b) for b in data.to_batches()]
                return pa.Table.from_batches(batches)
            else:
                return handle_recordbatch(data)
        elif pd and isinstance(data, pd.DataFrame):
            col_name_set = set(c.lower() for c in data.columns)
            missing_cols = [c for c in col_to_expr if c not in col_name_set]
            if not missing_cols:
                return data
            data = data.copy()
            for col in missing_cols:
                data[col] = col_to_expr[col].eval(data)
            return data
        else:
            wrapped = False
            if odps_types.is_record(data):
                data = [data]
                wrapped = True
            for rec in data:
                if not odps_types.is_record(rec):
                    continue
                for c in col_to_expr:
                    if rec[c] is not None:
                        continue
                    rec[c] = col_to_expr[c].eval(rec)
            return data[0] if wrapped else data

    @classmethod
    def _split_block_data_in_partitions(
        cls, table, block_data, partition_cols=None, partition=None
    ):
        from . import Record

        table_schema = table.table_schema
        col_to_expr = cls._get_ordered_col_expressions(table, partition)

        def _fill_cols(data):
            if col_to_expr:
                data = cls._fill_missing_expressions(data, col_to_expr)
            if not pd or not isinstance(data, pd.DataFrame):
                return data
            data.columns = [col.lower() for col in data.columns]
            tb_col_names = [c.name.lower() for c in table_schema.simple_columns]
            tb_col_set = set(tb_col_names)
            extra_cols = [col for col in data.columns if col not in tb_col_set]
            return data.reindex(tb_col_names + extra_cols, axis=1)

        if not partition_cols:
            is_arrow = cls._is_pa_collection(block_data) or cls._is_pd_df(block_data)
            return {(is_arrow, None): [_fill_cols(block_data)]}

        input_cols = list(table_schema.simple_columns) + [
            odps_types.Column(part, odps_types.string) for part in partition_cols
        ]
        input_schema = odps_types.OdpsSchema(input_cols)

        non_generate_idxes = [
            idx
            for idx, c in enumerate(table_schema.simple_columns)
            if not table._get_column_generate_expression(c.name)
        ]
        num_generate_pts = len(
            [
                c
                for c in (table_schema.partitions or [])
                if table._get_column_generate_expression(c.name)
            ]
        )

        parted_data = defaultdict(list)
        if (
            cls._is_pa_collection(block_data)
            or cls._is_pd_df(block_data)
            or odps_types.is_record(block_data)
            or (
                isinstance(block_data, list)
                and block_data
                and not isinstance(block_data[0], list)
            )
        ):
            # pd dataframes, arrow RecordBatch, single record or single record-like array
            block_data = [block_data]
        for data in block_data:
            if cls._is_pa_collection(data):
                data = data.to_pandas()
            elif isinstance(data, list):
                if len(data) != len(input_schema):
                    # fill None columns for generate functions to fill them
                    #  once size of data matches size of non-generative columns
                    if len(data) < len(table_schema.simple_columns) and len(
                        data
                    ) == len(non_generate_idxes):
                        new_data = [None] * len(table_schema.simple_columns)
                        for idx, d in zip(non_generate_idxes, data):
                            new_data[idx] = d
                        data = new_data
                    # fill None partitions for generate functions to fill them
                    data += [None] * (
                        num_generate_pts
                        - (len(data) - len(table_schema.simple_columns))
                    )
                    if len(data) != len(input_schema):
                        raise ValueError(
                            "Need to specify %d values when writing table "
                            "with dynamic partition." % len(input_schema)
                        )
                data = Record(schema=input_schema, values=data)

            if cls._is_pd_df(data):
                data = _fill_cols(data)
                part_set = set(partition_cols)
                for name, group in data.groupby(partition_cols):
                    name = name if isinstance(name, tuple) else (name,)
                    pt_name = ",".join(
                        "=".join([str(n), str(v)]) for n, v in zip(partition_cols, name)
                    )
                    parted_data[(True, pt_name)].append(
                        group.drop(part_set, axis=1, errors="ignore")
                    )
            elif odps_types.is_record(data):
                data = _fill_cols(data)
                pt_name = ",".join(
                    "=".join([str(n), data[str(n)]]) for n in partition_cols
                )
                values = [data[str(c.name)] for c in table_schema.simple_columns]
                if not parted_data[(False, pt_name)]:
                    parted_data[(False, pt_name)].append([])
                parted_data[(False, pt_name)][0].append(
                    Record(schema=table_schema, values=values)
                )
            else:
                raise ValueError(
                    "Cannot accept data with type %s" % type(data).__name__
                )
        return parted_data

    @classmethod
    def write_table(cls, odps, name, *block_data, **kw):
        """
        Write records or pandas DataFrame into given table.

        :param name: table or table name
        :type name: :class:`.models.table.Table` or str
        :param block_data: records / DataFrame, or block ids and records / DataFrame.
            If given records or DataFrame only, the block id will be 0 as default.
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param partition: the partition of this table to write into
        :param list partition_cols: columns representing dynamic partitions
        :param bool append_missing_cols: Whether to append missing columns to the target
            table. False by default.
        :param bool overwrite: if True, will overwrite existing data
        :param bool create_table: if true, the table will be created if not exist
        :param dict table_kwargs: specify other kwargs for :meth:`~odps.ODPS.create_table`
        :param dict type_mapping: specify type mapping for columns when creating tables,
            can be dicts like ``{"column": "bigint"}``. If column does not exist in data,
            it will be added as an empty column.
        :param bool infer_type_with_arrow: whether to infer column types of pandas objects
            with arrow when creating tables. Default as False.
        :param table_schema_callback: a function to accept table schema resolved from data
            and return a new schema for table to create. Only works when target table does
            not exist and ``create_table`` is True.
        :param int lifecycle: specify table lifecycle when creating tables
        :param bool create_partition: if true, the partition will be created if not exist
        :param compress_option: the compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param str endpoint:  tunnel service URL
        :param bool reopen: writing the table will reuse the session which opened last time,
            if set to True will open a new upload session, default as False
        :return: None

        :Example:

        Write records into a specified table.

        >>> odps.write_table('test_table', data)

        Write records into multiple blocks.

        >>> odps.write_table('test_table', 0, records1, 1, records2)

        Write into a given partition.

        >>> odps.write_table('test_table', data, partition='pt=test')

        Write a pandas DataFrame. Create the table if it does not exist.

        >>> import pandas as pd
        >>> df = pd.DataFrame([
        >>>     [111, 'aaa', True],
        >>>     [222, 'bbb', False],
        >>>     [333, 'ccc', True],
        >>>     [444, '中文', False]
        >>> ], columns=['num_col', 'str_col', 'bool_col'])
        >>> o.write_table('test_table', df, partition='pt=test', create_table=True, create_partition=True)

        Passing more arguments when creating table.

        >>> import pandas as pd
        >>> df = pd.DataFrame([
        >>>     [111, 'aaa', True],
        >>>     [222, 'bbb', False],
        >>>     [333, 'ccc', True],
        >>>     [444, '中文', False]
        >>> ], columns=['num_col', 'str_col', 'bool_col'])
        >>> # this dict will be passed to `create_table` as kwargs.
        >>> table_kwargs = {"transactional": True, "primary_key": "num_col"}
        >>> o.write_table('test_table', df, partition='pt=test', create_table=True, create_partition=True,
        >>>               table_kwargs=table_kwargs)

        Write with dynamic partitioning.

        >>> import pandas as pd
        >>> df = pd.DataFrame([
        >>>     [111, 'aaa', True, 'p1'],
        >>>     [222, 'bbb', False, 'p1'],
        >>>     [333, 'ccc', True, 'p2'],
        >>>     [444, '中文', False, 'p2']
        >>> ], columns=['num_col', 'str_col', 'bool_col', 'pt'])
        >>> o.write_table('test_part_table', df, partition_cols=['pt'], create_partition=True)

        :Note:

        ``write_table`` treats object type of Pandas data as strings as it is often hard to determine their
        types when creating a new table for your data. To make sure the column type meet your need, you can
        specify `type_mapping` argument to specify the column types, for instance,
        ``type_mapping={"col1": "array<struct<id:string>>"}``.

        .. seealso:: :class:`odps.models.Record`
        """
        project = kw.pop("project", None)
        schema = kw.pop("schema", None)
        append_missing_cols = kw.pop("append_missing_cols", False)
        infer_type_with_arrow = kw.pop("infer_type_with_arrow", False)
        overwrite = kw.pop("overwrite", False)

        single_block_types = (Iterable,)
        if pa is not None:
            single_block_types += (pa.RecordBatch, pa.Table)

        if len(block_data) == 1 and isinstance(block_data[0], single_block_types):
            blocks = [None]
            data_list = list(block_data)
        else:
            blocks = list(block_data[::2])
            data_list = list(block_data[1::2])

            if len(blocks) != len(data_list):
                raise ValueError(
                    "Should invoke like odps.write_table(block_id, records, "
                    "block_id2, records2, ..., **kw)"
                )

        unknown_as_string = kw.pop("unknown_as_string", False)
        create_table = kw.pop("create_table", False)
        create_partition = kw.pop(
            "create_partition", kw.pop("create_partitions", False)
        )
        partition = kw.pop("partition", None)
        partition_cols = kw.pop("partition_cols", None) or kw.pop("partitions", None)
        lifecycle = kw.pop("lifecycle", None)
        type_mapping = kw.pop("type_mapping", None)
        table_schema_callback = kw.pop("table_schema_callback", None)
        table_kwargs = dict(kw.pop("table_kwargs", None) or {})
        if lifecycle:
            table_kwargs["lifecycle"] = lifecycle

        if isinstance(partition_cols, six.string_types):
            partition_cols = [partition_cols]

        try:
            data_sample = data_list[0]
            if isinstance(data_sample, GeneratorType):
                data_gen = data_sample
                data_sample = [next(data_gen)]
                data_list[0] = utils.chain_generator([data_sample[0]], data_gen)
            table_schema = cls._resolve_schema(
                data_sample,
                unknown_as_string=unknown_as_string,
                partition=partition,
                partition_cols=partition_cols,
                type_mapping=type_mapping,
                infer_type_with_arrow=infer_type_with_arrow,
            )
        except TypeError:
            table_schema = None

        if not odps.exist_table(name, project=project, schema=schema):
            if not create_table:
                raise errors.NoSuchTable(
                    "Target table %s not exist. To create a new table "
                    "you can add an argument `create_table=True`." % name
                )
            if callable(table_schema_callback):
                table_schema = table_schema_callback(table_schema)
            if table_schema is None:
                raise ValueError(
                    "Table schema is required when creating a new table. "
                    "You can pass a dict to `type_mapping` argument."
                )
            target_table = odps.create_table(
                name, table_schema, project=project, schema=schema, **table_kwargs
            )
        else:
            target_table = cls._get_table_obj(
                odps, name, project=project, schema=schema
            )

        union_cols, diff_cols = cls._calc_schema_diff(
            table_schema, target_table.schema, partition_cols=partition_cols
        )
        if table_schema and not union_cols:
            warnings.warn(
                "No columns overlapped between source and target table. If result "
                "is not as expected, please check if your query provides correct "
                "column names."
            )
        if diff_cols:
            if append_missing_cols:
                target_table.add_columns(diff_cols)
            else:
                warnings.warn(
                    "Columns in source data %s are missing in target table %s. "
                    "Specify append_missing_cols=True to append missing columns "
                    "to the target table."
                    % (", ".join(c.name for c in diff_cols), target_table.name)
                )

        partition_cols = cls._check_partition_specified(
            name,
            target_table.table_schema,
            partition_cols=partition_cols,
            partition=partition,
        )

        data_lists = defaultdict(lambda: defaultdict(list))
        for block, data in zip(blocks, data_list):
            for key, parted_data in cls._split_block_data_in_partitions(
                target_table,
                data,
                partition_cols=partition_cols,
                partition=partition,
            ).items():
                data_lists[key][block].extend(parted_data)

        if partition is None or isinstance(partition, six.string_types):
            partition_str = partition
        else:
            partition_str = str(odps_types.PartitionSpec(partition))

        # fixme cover up for overwrite failure on table.format.version=2:
        #  only applicable for transactional table with partitions
        #  with generate expressions
        manual_truncate = (
            overwrite
            and target_table.is_transactional
            and any(
                pt_col.generate_expression
                for pt_col in target_table.table_schema.partitions
            )
        )

        for (is_arrow, pt_name), block_to_data in data_lists.items():
            if not block_to_data:
                continue

            blocks, data_list = [], []
            for block, data in block_to_data.items():
                blocks.append(block)
                data_list.extend(data)

            if len(blocks) == 1 and blocks[0] is None:
                blocks = None

            final_pt = ",".join(p for p in (pt_name, partition_str) if p is not None)
            # fixme cover up for overwrite failure on table.format.version=2
            if overwrite and manual_truncate:
                if not final_pt or target_table.exist_partition(final_pt):
                    target_table.truncate(partition_spec=final_pt or None)
            with target_table.open_writer(
                partition=final_pt or None,
                blocks=blocks,
                arrow=is_arrow,
                create_partition=create_partition,
                reopen=append_missing_cols,
                overwrite=overwrite,
                **kw
            ) as writer:
                if blocks is None:
                    for data in data_list:
                        writer.write(data)
                else:
                    for block, data in zip(blocks, data_list):
                        writer.write(block, data)

    @classmethod
    def write_sql_result_to_table(
        cls,
        odps,
        table_name,
        sql,
        partition=None,
        partition_cols=None,
        create_table=False,
        create_partition=False,
        append_missing_cols=False,
        overwrite=False,
        project=None,
        schema=None,
        lifecycle=None,
        type_mapping=None,
        table_schema_callback=None,
        table_kwargs=None,
        hints=None,
        running_cluster=None,
        unique_identifier_id=None,
        **kwargs
    ):
        """
        Write SQL query results into a specified table and partition. If the target
        table does not exist, you may specify the argument create_table=True. Columns
        are inserted into the target table aligned by column names. Note that column
        order in the target table will NOT be changed.

        :param str table_name: The target table name
        :param str sql: The SQL query to execute
        :param str partition: Target partition in the format "part=value" or
            "part1=value1,part2=value2"
        :param list partition_cols: List of dynamic partition fields. If not provided,
            all partition fields of the target table are used.
        :param bool create_table: Whether to create the target table if it does not exist.
            False by default.
        :param bool create_partition: Whether to create partitions if they do not exist.
            False by default.
        :param bool append_missing_cols: Whether to append missing columns to the target
            table. False by default.
        :param bool overwrite: Whether to overwrite existing data. False by default.
        :param str project: project name, if not provided, will be the default project
        :param str schema: schema name, if not provided, will be the default schema
        :param int lifecycle: specify table lifecycle when creating tables
        :param dict type_mapping: specify type mapping for columns when creating tables,
            can be dicts like ``{"column": "bigint"}``. If column does not exist in data,
            it will be added as an empty column.
        :param table_schema_callback: a function to accept table schema resolved from data
            and return a new schema for table to create. Only works when target table does
            not exist and ``create_table`` is True.
        :param dict table_kwargs: specify other kwargs for :meth:`~odps.ODPS.create_table`
        :param dict hints: specify hints for SQL statements, will be passed through
            to execute_sql method
        :param dict running_cluster: specify running cluster for SQL statements, will
            be passed through to execute_sql method
        """
        partition_cols = partition_cols or kwargs.pop("partitions", None)
        if isinstance(partition_cols, six.string_types):
            partition_cols = [partition_cols]

        temp_table_name = "_".join(
            [utils.TEMP_TABLE_PREFIX, utils.md5_hexdigest(table_name), uuid.uuid4().hex]
        )
        insert_mode = "OVERWRITE" if overwrite else "INTO"

        # move table params in kwargs into table_kwargs
        table_kwargs = dict(table_kwargs or {})
        for extra_table_arg in (
            "table_properties",
            "shard_num",
            "transactional",
            "primary_key",
            "storage_tier",
        ):
            if extra_table_arg in kwargs:
                table_kwargs[extra_table_arg] = kwargs.pop(extra_table_arg)
        # if extra table kwargs are supported, create table ... as ...
        #  may not work, and the table need to be created first
        with_extra_table_kw = bool(table_kwargs)

        table_kwargs.update(
            {"schema": schema, "project": project, "lifecycle": lifecycle}
        )
        sql_kwargs = kwargs.copy()
        sql_kwargs.update(
            {
                "hints": copy.deepcopy(hints or {}),
                "running_cluster": running_cluster,
                "unique_identifier_id": unique_identifier_id,
                "default_schema": schema,
                "project": project,
            }
        )

        def _format_raw_sql(fmt, args):
            """Add DDLs for existing SQL, multiple statements acceptable"""
            args = list(args)
            sql_parts = utils.split_sql_by_semicolon(args[-1])
            if len(sql_parts) == 1:
                return fmt % tuple(args)
            # need script mode for multiple statements
            sql_kwargs["hints"]["odps.sql.submit.mode"] = "script"
            sql_parts[-1] = fmt % tuple(args[:-1] + [sql_parts[-1]])
            return "\n".join(sql_parts)

        # Check if the target table exists
        if not odps.exist_table(table_name, project=project, schema=schema):
            if not create_table:
                raise ValueError(
                    "Table %s does not exist and create_table is set to False."
                    % table_name
                )
            elif (
                not partition
                and not partition_cols
                and not with_extra_table_kw
                and table_schema_callback is None
            ):
                # return directly when creating table without partitions
                #  and special kwargs
                if not lifecycle:
                    lifecycle_clause = ""
                else:
                    lifecycle_clause = "LIFECYCLE %d " % lifecycle
                sql_stmt = _format_raw_sql(
                    "CREATE TABLE %s %sAS %s",
                    (utils.backquote_string(table_name), lifecycle_clause, sql),
                )
                odps.execute_sql(sql_stmt, **sql_kwargs)
                return
            else:
                # create temp table, get result schema and create target table
                sql_stmt = _format_raw_sql(
                    "CREATE TABLE %s LIFECYCLE %d AS %s",
                    (
                        utils.backquote_string(temp_table_name),
                        options.temp_lifecycle,
                        sql,
                    ),
                )
                odps.execute_sql(sql_stmt, **sql_kwargs)
                tmp_schema = odps.get_table(temp_table_name).table_schema
                out_table_schema = cls._resolve_schema(
                    data_schema=tmp_schema,
                    partition=partition,
                    partition_cols=partition_cols,
                    type_mapping=type_mapping,
                )
                if table_schema_callback:
                    out_table_schema = table_schema_callback(out_table_schema)
                target_table = odps.create_table(
                    table_name, table_schema=out_table_schema, **table_kwargs
                )
        else:
            target_table = cls._get_table_obj(
                odps, table_name, project=project, schema=schema
            )
            # for partitioned target, create a temp table and store results
            sql_stmt = _format_raw_sql(
                "CREATE TABLE %s LIFECYCLE %d AS %s",
                (utils.backquote_string(temp_table_name), options.temp_lifecycle, sql),
            )
            odps.execute_sql(sql_stmt, **sql_kwargs)

        try:
            partition_cols = cls._check_partition_specified(
                table_name,
                target_table.table_schema,
                partition_cols=partition_cols,
                partition=partition,
            )

            temp_table = odps.get_table(temp_table_name)
            union_cols, diff_cols = cls._calc_schema_diff(
                temp_table.table_schema,
                target_table.table_schema,
                partition_cols=partition_cols,
            )
            if not union_cols:
                warnings.warn(
                    "No columns overlapped between source and target table. If result "
                    "is not as expected, please check if your query provides correct "
                    "column names."
                )
            if diff_cols:
                if append_missing_cols:
                    target_table.add_columns(diff_cols, hints=hints)
                    union_cols += diff_cols
                else:
                    warnings.warn(
                        "Columns in source query %s are missing in target table %s. "
                        "Specify append_missing_cols=True to append missing columns "
                        "to the target table."
                        % (", ".join(c.name for c in diff_cols), target_table.name)
                    )

            target_columns = [col.name for col in union_cols]

            if partition:
                static_part_spec = odps_types.PartitionSpec(partition)
            else:
                static_part_spec = odps_types.PartitionSpec()

            if (
                target_table.table_schema.partitions
                and len(static_part_spec) == len(target_table.table_schema.partitions)
                and not target_table.exist_partition(static_part_spec)
            ):
                if create_partition:
                    target_table.create_partition(static_part_spec)
                else:
                    raise ValueError(
                        "Partition %s does not exist and create_partition is set to False."
                        % static_part_spec
                    )

            all_parts, part_specs, dyn_parts = [], [], []
            has_dyn_parts = False
            for col in target_table.table_schema.partitions:
                if col.name in static_part_spec:
                    spec = "%s='%s'" % (
                        col.name,
                        utils.escape_odps_string(static_part_spec[col.name]),
                    )
                    part_specs.append(spec)
                    all_parts.append(spec)
                elif col.name not in temp_table.table_schema:
                    if col.generate_expression:
                        all_parts.append(col.name)
                        has_dyn_parts = True
                        continue
                    else:
                        raise ValueError(
                            "Partition column %s does not exist in source query."
                            % col.name
                        )
                else:
                    has_dyn_parts = True
                    all_parts.append(col.name)
                    dyn_parts.append(col.name)
                    part_specs.append(col.name)

            if not part_specs:
                part_clause = ""
            else:
                part_clause = "PARTITION (%s) " % ", ".join(part_specs)

            if overwrite:
                insert_mode = "INTO"
                if not has_dyn_parts:
                    target_table.truncate(partition or None)
                elif any(target_table.partitions):
                    # generate column expressions in topological order
                    col_to_exprs = cls._get_ordered_col_expressions(
                        target_table, partition
                    )
                    part_expr_map = {
                        col: utils.backquote_string(col)
                        for col in partition_cols
                        if col in temp_table.table_schema
                    }
                    generated_cols = set()
                    for col_name, expr in col_to_exprs.items():
                        if col_name in part_expr_map:
                            continue
                        generated_cols.add(col_name)
                        part_expr_map[col_name] = expr.to_str(part_expr_map)
                    # add an alias for generated columns
                    part_expr_map = {
                        col: (
                            v
                            if col not in generated_cols
                            else "%s AS %s" % (v, utils.backquote_string(col))
                        )
                        for col, v in part_expr_map.items()
                    }

                    # query for partitions need to be truncated
                    part_selections = [
                        part_expr_map[col_name] for col_name in partition_cols
                    ]
                    part_distinct_sql = "SELECT DISTINCT %s FROM %s" % (
                        ", ".join(part_selections),
                        utils.backquote_string(temp_table_name),
                    )
                    distinct_inst = odps.execute_sql(part_distinct_sql)
                    trunc_part_specs = []
                    with distinct_inst.open_reader(tunnel=True) as reader:
                        for row in reader:
                            local_part_specs = [
                                "%s='%s'" % (c, utils.escape_odps_string(row[c]))
                                if c in row
                                else c
                                for c in all_parts
                            ]
                            local_part_str = ",".join(local_part_specs)
                            if target_table.exist_partition(local_part_str):
                                trunc_part_specs.append(local_part_str)
                    target_table.truncate(trunc_part_specs)

            col_selection = ", ".join(
                utils.backquote_string(s) for s in (target_columns + dyn_parts)
            )
            sql_stmt = "INSERT %s %s %s (%s) SELECT %s FROM %s" % (
                insert_mode,
                utils.backquote_string(table_name),
                part_clause,
                col_selection,
                col_selection,
                temp_table_name,
            )
            odps.execute_sql(sql_stmt, **sql_kwargs)
        finally:
            odps.delete_table(
                temp_table_name,
                project=project,
                schema=schema,
                if_exists=True,
                async_=True,
            )
