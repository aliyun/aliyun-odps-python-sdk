# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import functools
import itertools
import logging
import multiprocessing
import os
import socket
import struct
import sys
import threading
from types import GeneratorType, MethodType

try:
    import pyarrow as pa
except (AttributeError, ImportError):
    pa = None

from .. import types as odps_types, utils
from ..compat import six
from ..config import options
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
        conn, download_id, start, count, idx,
        rest_client=None, project=None, table_name=None, partition_spec=None,
        tunnel_endpoint=None, quota_name=None, columns=None, arrow=False,
        schema_name=None
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
            if not arrow:
                data = utils.call_with_retry(
                    session.open_record_reader, start, count, columns=columns
                ).to_pandas()
            else:
                data = utils.call_with_retry(
                    session.open_arrow_reader, start, count, columns=columns
                ).to_pandas()
            conn.send((idx, data, True))
        except:
            conn.send((idx, sys.exc_info(), False))

    def _get_process_split_reader(self, columns=None):
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
        )


class TableRecordReader(SpawnedTableReaderMixin, TunnelRecordReader):
    def __init__(self, table, download_session, partition_spec=None, columns=None):
        super(TableRecordReader, self).__init__(
            table, download_session, columns=columns
        )
        self._partition_spec = partition_spec


class TableArrowReader(SpawnedTableReaderMixin, TunnelArrowReader):
    def __init__(self, table, download_session, partition_spec=None, columns=None):
        super(TableArrowReader, self).__init__(
            table, download_session, columns=columns
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
                    count, = struct.unpack("<H", data[pos:pos + 2])
                    pos += 2
                    assert 4 * count < len(data), "Data too short for block count!"
                    block_ids = struct.unpack("<%dI" % count, data[pos:pos + 4 * count])
                    blocks_queue.put(block_ids)
                    self._sock.sendto(struct.pack("<B", _PUT_WRITTEN_BLOCKS_CMD), addr)
                elif cmd_code == _STOP_SERVER_CMD:
                    assert addr[0] == self._sock.getsockname()[0], "Cannot stop server from other hosts!"
                    break
                else:  # pragma: no cover
                    raise AssertionError("Unrecognized command %x", cmd_code)
            except BaseException:
                pk_exc_info = cloudpickle.dumps(sys.exc_info())
                data = struct.pack("<B", _SERVER_ERROR_CMD) + struct.pack(
                    "<I", len(pk_exc_info)
                ) + pk_exc_info
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
        pk_len, = struct.unpack("<I", recv_data[1:5])
        exc_info = cloudpickle.loads(recv_data[5:5 + pk_len])
        six.reraise(*exc_info)

    def get_next_block_id(self):
        sock = self._get_socket()
        data = self._authkey + struct.pack("<B", _GET_NEXT_BLOCK_CMD)
        sock.sendto(data, self._addr)
        recv_data, server_addr = sock.recvfrom(1024)
        self._reraise_remote_error(recv_data)
        assert _ord_if_possible(recv_data[0]) == _GET_NEXT_BLOCK_CMD
        assert self._addr == server_addr
        count, = struct.unpack("<I", recv_data[1:5])
        return count

    def put_written_blocks(self, block_ids):
        sock = self._get_socket()
        for pos in range(0, len(block_ids), self._MAX_BLOCK_COUNT):
            sub_block_ids = block_ids[pos:pos + self._MAX_BLOCK_COUNT]
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
        self,
        table,
        upload_session,
        blocks=None,
        commit=True,
        on_close=None,
        **kwargs
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
        self._blocks = blocks or upload_session.blocks or [0, ]
        self._blocks_writes = [False] * len(self._blocks)
        self._blocks_writers = [None] * len(self._blocks)

        for block in (upload_session.blocks or ()):
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
            self._used_block_id_queue = kwargs.get(
                "used_block_id_queue"
            ) or self._mp_context.Queue()

    @classmethod
    def _restore_subprocess_writer(
        cls, mp_server_address, mp_server_auth, upload_id, main_pid=None, blocks=None,
        rest_client=None, project=None, table_name=None, partition_spec=None,
        tunnel_endpoint=None, quota_name=None, schema=None
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
            table, session, commit=False, blocks=blocks, main_pid=main_pid,
            mp_fixed=True, mp_client=mp_client,
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
            raise IOError('Cannot write to a closed writer.')
        self._fix_mp_attributes()

        compress = kwargs.get('compress', False)

        block_id = kwargs.get('block_id')
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
            writer = self._thread_to_buffered_writers.get(threading.current_thread().ident)
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


class TableRecordWriter(AbstractTableWriter):
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

    def _to_records(self, arg, sample_rec):
        if odps_types.is_record(sample_rec):
            return arg
        elif isinstance(sample_rec, (list, tuple)):
            return (self._table.new_record(vals) for vals in arg)
        else:
            return None

    def _write_contents(self, writer, *args):
        if len(args) == 0:
            return
        if len(args) > 1:
            args = [args]

        arg = args[0]
        if odps_types.is_record(arg):
            records = [arg]
        elif isinstance(arg, (list, tuple)):
            records = self._to_records(arg, arg[0])
            if records is None:
                records = [self._table.new_record(arg), ]
        elif isinstance(arg, GeneratorType):
            try:
                # peek the first element and then put back
                next_arg = six.next(arg)
                chained = itertools.chain((next_arg,), arg)
                records = self._to_records(chained, next_arg)
            except StopIteration:
                records = ()
        else:
            raise ValueError('Unsupported record type.')

        for record in records:
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
            writer = self._upload_session.open_arrow_writer(
                block_id, compress=compress
            )
            self._blocks_writers[block_id] = writer
        return writer

    def _write_contents(self, writer, *args):
        for arg in args:
            writer.write(arg)
