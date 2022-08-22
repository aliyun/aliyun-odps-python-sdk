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

import json
import logging
import os
import pickle
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from mars.lib.tblib import pickling_support
from ...compat import six
from ..utils import filter_partitions, check_partition_exist

pickling_support.install()

logger = logging.getLogger(__name__)

REQUEST_TYPE_READ_TABLE_DATA = 0
REQUEST_TYPE_WRITE_TABLE_DATA = 1
REQUEST_TYPE_ENUM_TABLE_PARTITIONS = 2
REQUEST_TYPE_CREATE_TABLE_DOWNLOAD_SESSION = 3
REQUEST_TYPE_CREATE_TABLE_UPLOAD_SESSION = 4
REQUEST_TYPE_COMMIT_TABLE_UPLOAD_SESSION = 5
REQUEST_TYPE_GET_KV = 6
REQUEST_TYPE_PUT_KV = 7
REQUEST_TYPE_TERMINATE_INSTANCE = 8
REQUEST_TYPE_GET_BEARER_TOKEN = 9
REQUEST_TYPE_REPORT_CONTAINER_STATUS = 10

CHUNK_BYTES_LIMIT = 64 * 1024**2
TRANSFER_BLOCK_SIZE = 64 * 1024**2
MAX_CHUNK_NUM = 512 * 1024**2


def _create_arrow_writer(sink, schema, **kwargs):
    import pyarrow as pa

    try:
        return pa.ipc.new_stream(sink, schema, **kwargs)
    except AttributeError:
        return pa.ipc.RecordBatchStreamWriter(sink, schema, **kwargs)


def _write_request_result(sock, success=True, result=None, exc_info=None):
    try:
        result_dict = {
            "status": success,
            "result": result,
            "exc_info": exc_info,
        }
        pickled = pickle.dumps(dict((k, v) for k, v in result_dict.items()))
        sock_out_file = sock.makefile("wb")
        sock_out_file.write(struct.pack("<I", len(pickled)))
        sock_out_file.write(pickled)
    finally:
        sock_out_file.flush()
        sock_out_file.close()


def _handle_read_table_data(sock):
    from cupid.io.table import TableSplit
    from cupid.errors import SubprocessStreamEOFError

    sock_out_file = sock.makefile("wb")
    ipc_writer = None
    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        read_config = pickle.loads(sock.recv(cmd_len))
        min_rows = read_config.pop("min_rows", None)

        tsp = TableSplit(**read_config)
        logger.debug("Read split table, split index: %s", read_config["_split_index"])
        read_config = None

        if min_rows is None:
            reader = tsp.open_arrow_file_reader()
            while True:
                chunk = reader.read(TRANSFER_BLOCK_SIZE)
                if len(chunk) == 0:
                    break
                try:
                    sock_out_file.write(chunk)
                except (BrokenPipeError, SubprocessStreamEOFError):
                    break
                finally:
                    chunk = None
        else:
            reader = tsp.open_arrow_reader()
            nrows = 0
            while min_rows is None or nrows < min_rows:
                try:
                    batch = reader.read_next_batch()
                    nrows += batch.num_rows
                    if ipc_writer is None:
                        ipc_writer = _create_arrow_writer(sock_out_file, batch.schema)
                    ipc_writer.write_batch(batch)
                except StopIteration:
                    break
                finally:
                    batch = None
            if ipc_writer is not None:  # pragma: no branch
                ipc_writer.close()
                ipc_writer = None
    except:
        logger.exception("Failed to read table")
    finally:
        if ipc_writer is not None:
            ipc_writer.close()
        sock_out_file.flush()
        sock_out_file.close()


def _handle_write_table_data(sock):
    import pyarrow as pa
    from cupid.io.table.core import BlockWriter

    try:
        sock_in_file = sock.makefile("rb")

        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        writer_config = pickle.loads(sock.recv(cmd_len))

        block_writer = BlockWriter(**writer_config)
        logger.debug(
            "Start writing table block, block id: %s", writer_config["_block_id"]
        )
        ipc_reader = pa.ipc.open_stream(sock_in_file)
        with block_writer.open_arrow_writer() as cupid_writer:
            arrow_writer = pa.RecordBatchStreamWriter(cupid_writer, ipc_reader.schema)
            while True:
                try:
                    batch = ipc_reader.read_next_batch()
                    arrow_writer.write_batch(batch)
                except StopIteration:
                    break
            arrow_writer.close()
        logger.debug(
            "Write table block finished, block id: %s", writer_config["_block_id"]
        )
        block_writer.commit()

        _write_request_result(sock, result={"block_id": writer_config["_block_id"]})
    except:
        logger.exception("Failed to read table")
        _write_request_result(sock, False, exc_info=sys.exc_info())


def _handle_enum_table_partitions(sock):
    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        # dict with odps_params, table_name, partition
        task_config = pickle.loads(sock.recv(cmd_len))

        from odps import ODPS
        from odps.accounts import BearerTokenAccount
        from cupid import context

        cupid_ctx = context()

        odps_params = task_config["odps_params"]
        bearer_token = cupid_ctx.get_bearer_token()
        account = BearerTokenAccount(bearer_token)
        project = os.environ.get("ODPS_PROJECT_NAME", None) or odps_params["project"]
        endpoint = os.environ.get("ODPS_RUNTIME_ENDPOINT") or odps_params["endpoint"]
        o = ODPS(None, None, account=account, project=project, endpoint=endpoint)

        table = o.get_table(task_config["table_name"])
        partition_desc = task_config.get("partition")
        if not table.schema.partitions:
            _write_request_result(sock, result=None)
        elif partition_desc:
            if check_partition_exist(table, partition_desc):
                _write_request_result(sock, result=[partition_desc])
            else:
                parts = filter_partitions(o, list(table.partitions), partition_desc)
                _write_request_result(
                    sock, result=[str(pt.partition_spec) for pt in parts]
                )
        else:
            _write_request_result(
                sock, result=[str(pt.partition_spec) for pt in table.partitions]
            )
    except:
        logger.exception("Failed to create download session")
        _write_request_result(sock, False, exc_info=sys.exc_info())


def _handle_create_table_download_session(sock):
    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        # dict with odps_params, table_name, partition, columns, worker_count, split_size, max_chunk_num
        session_config = pickle.loads(sock.recv(cmd_len))

        from odps import ODPS
        from odps.errors import ODPSError
        from odps.accounts import BearerTokenAccount
        from cupid import CupidSession, context
        from cupid.errors import CupidError
        from cupid.runtime import RuntimeContext

        if not RuntimeContext.is_context_ready():
            raise SystemError(
                "No Mars cluster found, please create via `o.create_mars_cluster`."
            )

        cupid_ctx = context()

        odps_params = session_config["odps_params"]
        bearer_token = cupid_ctx.get_bearer_token()
        account = BearerTokenAccount(bearer_token)
        project = os.environ.get("ODPS_PROJECT_NAME", None) or odps_params["project"]
        endpoint = os.environ.get("ODPS_RUNTIME_ENDPOINT") or odps_params["endpoint"]
        o = ODPS(None, None, account=account, project=project, endpoint=endpoint)
        cupid_session = CupidSession(o)

        split_size = session_config["split_size"]
        table_name = session_config["table_name"]
        data_src = o.get_table(table_name)
        if session_config.get("partition") is not None:
            data_src = data_src.get_partition(session_config["partition"])

        try:
            data_store_size = data_src.size
        except ODPSError:
            # fail to get data size, just ignore
            pass
        else:
            worker_count = session_config["worker_count"]
            if data_store_size < split_size and worker_count is not None:
                # data is too small, split as many as number of cores
                split_size = data_store_size // worker_count
                # at least 1M
                split_size = max(split_size, 1 * 1024**2)
                logger.debug(
                    "Input data size is too small, split_size is {}".format(split_size)
                )

        max_chunk_num = session_config["max_chunk_num"]
        columns = session_config["columns"]
        with_split_meta = session_config.get("with_split_meta_on_tile")

        logger.debug(
            "Start creating download session of table %s from cupid, columns %r",
            table_name,
            columns,
        )
        while True:
            try:
                download_session = cupid_session.create_download_session(
                    data_src,
                    split_size=split_size,
                    columns=columns,
                    with_split_meta=with_split_meta,
                )
                break
            except CupidError:
                logger.debug(
                    "The number of splits exceeds 100000, split_size is {}".format(
                        split_size
                    )
                )
                if split_size >= max_chunk_num:
                    raise
                else:
                    split_size *= 2

        ret_data = {
            "splits": download_session.splits,
            "split_size": split_size,
        }
        _write_request_result(sock, result=ret_data)
    except:
        logger.exception("Failed to create download session")
        _write_request_result(sock, False, exc_info=sys.exc_info())


def _handle_create_table_upload_session(sock):
    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        # dict with odps_params, table_name
        session_config = pickle.loads(sock.recv(cmd_len))

        from odps import ODPS
        from odps.accounts import BearerTokenAccount
        from cupid import CupidSession, context
        from cupid.runtime import RuntimeContext

        if not RuntimeContext.is_context_ready():
            raise SystemError(
                "No Mars cluster found, please create via `o.create_mars_cluster`."
            )
        cupid_ctx = context()

        odps_params = session_config["odps_params"]
        bearer_token = cupid_ctx.get_bearer_token()
        account = BearerTokenAccount(bearer_token)
        project = os.environ.get("ODPS_PROJECT_NAME", None) or odps_params["project"]
        endpoint = os.environ.get("ODPS_RUNTIME_ENDPOINT") or odps_params["endpoint"]
        o = ODPS(None, None, account=account, project=project, endpoint=endpoint)
        cupid_session = CupidSession(o)

        data_src = o.get_table(session_config["table_name"])

        logger.debug("Start creating upload session from cupid.")
        upload_session = cupid_session.create_upload_session(data_src)

        ret_data = {
            "handle": upload_session.handle,
        }
        _write_request_result(sock, result=ret_data)
    except:
        logger.exception("Failed to create upload session")
        _write_request_result(sock, False, exc_info=sys.exc_info())


def _handle_commit_table_upload_session(sock):
    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        # dict with odps_params, table_name, cupid_handle, blocks, overwrite
        commit_config = pickle.loads(sock.recv(cmd_len))

        from odps import ODPS
        from odps.accounts import BearerTokenAccount
        from cupid import CupidSession, context
        from cupid.runtime import RuntimeContext
        from cupid.io.table import CupidTableUploadSession

        if not RuntimeContext.is_context_ready():
            raise SystemError(
                "No Mars cluster found, please create via `o.create_mars_cluster`."
            )
        cupid_ctx = context()

        odps_params = commit_config["odps_params"]
        bearer_token = cupid_ctx.get_bearer_token()
        account = BearerTokenAccount(bearer_token)
        project = os.environ.get("ODPS_PROJECT_NAME", None) or odps_params["project"]
        endpoint = os.environ.get("ODPS_RUNTIME_ENDPOINT") or odps_params["endpoint"]
        o = ODPS(None, None, account=account, project=project, endpoint=endpoint)
        cupid_session = CupidSession(o)

        project_name, table_name = commit_config["table_name"].split(".")
        upload_session = CupidTableUploadSession(
            session=cupid_session,
            table_name=table_name,
            project_name=project_name,
            handle=commit_config["cupid_handle"],
            blocks=commit_config["blocks"],
        )
        upload_session.commit(overwrite=commit_config["overwrite"])

        _write_request_result(sock)
    except:
        logger.exception("Failed to commit upload session")
        _write_request_result(sock, False, exc_info=sys.exc_info())


def _handle_get_kv(sock):
    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        # dict with key
        cmd_body = pickle.loads(sock.recv(cmd_len))

        from cupid.runtime import RuntimeContext

        if not RuntimeContext.is_context_ready():
            logger.warning("Cupid context not ready")
            value = None
        else:
            from cupid import context

            cupid_kv = context().kv_store()
            value = cupid_kv.get(cmd_body["key"])

        ret_data = {
            "value": value,
        }
        _write_request_result(sock, result=ret_data)
    except:
        logger.exception("Failed to get kv value")
        _write_request_result(sock, False, exc_info=sys.exc_info())


def _handle_put_kv(sock):
    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        # dict with key
        cmd_body = pickle.loads(sock.recv(cmd_len))

        from cupid.runtime import RuntimeContext

        if not RuntimeContext.is_context_ready():
            logger.warning("Cupid context not ready")
        else:
            from cupid import context

            cupid_kv = context().kv_store()
            cupid_kv[cmd_body["key"]] = cmd_body["value"]

        _write_request_result(sock)
    except:
        logger.exception("Failed to put kv value")
        _write_request_result(sock, False, exc_info=sys.exc_info())


def _handle_terminate_instance(sock):
    from cupid.runtime import context, RuntimeContext
    from odps import ODPS
    from odps.accounts import BearerTokenAccount

    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        # dict with key
        cmd_body = pickle.loads(sock.recv(cmd_len))

        instance_id = cmd_body["instance_id"]

        if not RuntimeContext.is_context_ready():
            logger.warning("Cupid context not ready")
        else:
            bearer_token = context().get_bearer_token()
            account = BearerTokenAccount(bearer_token)
            project = os.environ["ODPS_PROJECT_NAME"]
            endpoint = os.environ["ODPS_RUNTIME_ENDPOINT"]
            o = ODPS(None, None, account=account, project=project, endpoint=endpoint)

            o.stop_instance(instance_id)
    except:
        logger.exception("Failed to put kv value")
        _write_request_result(sock, False, exc_info=sys.exc_info())


def _handle_get_bearer_token(sock):
    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        # dict with odps_params, table_name, cupid_handle, blocks, overwrite
        _commit_config = pickle.loads(sock.recv(cmd_len))  # noqa: F841

        from cupid import context

        bearer_token = context().get_bearer_token()
        _write_request_result(sock, result={"token": bearer_token})
    except:
        logger.exception("Failed to get bearer token")
        _write_request_result(sock, False, exc_info=sys.exc_info())


def _handle_report_container_status(sock):
    try:
        (cmd_len,) = struct.unpack("<I", sock.recv(4))
        # dict with odps_params, table_name, cupid_handle, blocks, overwrite
        cmd_body = pickle.loads(sock.recv(cmd_len))

        status = cmd_body["status"]
        message = cmd_body["message"]
        progress = cmd_body["progress"]
        timeout = cmd_body["timeout"]

        from cupid import context

        context().report_container_status(status, message, progress, timeout=timeout)
    except:
        logger.exception("Failed to get bearer token")
        _write_request_result(sock, False, exc_info=sys.exc_info())


_request_handlers = {
    REQUEST_TYPE_READ_TABLE_DATA: _handle_read_table_data,
    REQUEST_TYPE_WRITE_TABLE_DATA: _handle_write_table_data,
    REQUEST_TYPE_ENUM_TABLE_PARTITIONS: _handle_enum_table_partitions,
    REQUEST_TYPE_CREATE_TABLE_DOWNLOAD_SESSION: _handle_create_table_download_session,
    REQUEST_TYPE_CREATE_TABLE_UPLOAD_SESSION: _handle_create_table_upload_session,
    REQUEST_TYPE_COMMIT_TABLE_UPLOAD_SESSION: _handle_commit_table_upload_session,
    REQUEST_TYPE_GET_KV: _handle_get_kv,
    REQUEST_TYPE_PUT_KV: _handle_put_kv,
    REQUEST_TYPE_TERMINATE_INSTANCE: _handle_terminate_instance,
    REQUEST_TYPE_GET_BEARER_TOKEN: _handle_get_bearer_token,
    REQUEST_TYPE_REPORT_CONTAINER_STATUS: _handle_report_container_status,
}


def _handle_requests(sock):
    while True:
        try:
            (req_type,) = struct.unpack("<I", sock.recv(4))
            if req_type in _request_handlers:
                _request_handlers[req_type](sock)
            else:
                sock.close()
                break
        except ConnectionAbortedError:
            break


def _prepare_channel(channel_file):
    while not os.path.exists(channel_file):
        time.sleep(1)
    try:
        with open(channel_file, "r") as env_file:
            envs = json.loads(env_file.read())
    except:
        time.sleep(1)
        with open(channel_file, "r") as env_file:
            envs = json.loads(env_file.read())

    from cupid import context

    os.environ.update(envs)
    context()
    odps_envs = {
        "ODPS_BEARER_TOKEN": os.environ["BEARER_TOKEN_INITIAL_VALUE"],
        "ODPS_ENDPOINT": os.environ["ODPS_RUNTIME_ENDPOINT"],
    }
    os.environ.update(odps_envs)
    logger.info("Started channel for Cupid Server.")


def run_cupid_service(channel_file, sock_file=None, pool_size=None):
    _prepare_channel(channel_file)

    pool = ThreadPoolExecutor(pool_size)

    sock_file = sock_file or os.environ["CUPID_SERVICE_SOCKET"]
    logger.warning("Starting Cupid Service with socket %s", sock_file)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(sock_file)
    sock.listen(1)

    while True:
        conn, _ = sock.accept()
        pool.submit(_handle_requests, conn)


class CupidServiceClient:
    def __init__(self, sock_file=None):
        self._sock_file = sock_file or os.environ["CUPID_SERVICE_SOCKET"]
        self._sock = None

    def __del__(self):
        self.close()

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    @property
    def sock(self):
        if self._sock is None or self._sock._closed:
            if not os.path.exists(self._sock_file):
                raise OSError("Socket file does not exist")
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(self._sock_file)
        return self._sock

    def _send_cupid_service_request(self, req_type, arg_object):
        ser = pickle.dumps(arg_object)

        sock_out_file = self.sock.makefile("wb")
        sock_out_file.write(struct.pack("<I", req_type))
        sock_out_file.write(struct.pack("<I", len(ser)))
        sock_out_file.write(ser)
        sock_out_file.flush()

    def _fetch_cupid_service_result(self):
        (ret_len,) = struct.unpack("<I", self.sock.recv(4))
        ret = pickle.loads(self.sock.recv(ret_len))

        if not ret["status"]:
            six.reraise(*ret["exc_info"])
        else:
            return ret.get("result")

    def read_table_data(self, split_config, min_rows=None):
        import pyarrow as pa

        split_config = split_config.copy()
        split_config["min_rows"] = min_rows
        self._send_cupid_service_request(REQUEST_TYPE_READ_TABLE_DATA, split_config)

        sock_in_file = self.sock.makefile("rb")
        batches = []
        try:
            if min_rows is None:
                reader = pa.RecordBatchStreamReader(sock_in_file)
                return reader.read_all()
            else:
                ipc_reader = pa.ipc.open_stream(sock_in_file)
                while True:
                    try:
                        batches.append(ipc_reader.read_next_batch())
                    except StopIteration:
                        break
                return pa.Table.from_batches(batches)
        finally:
            batches[:] = []
            sock_in_file.close()

    def write_table_data(self, writer_config, to_store_data, write_batch_size=None):
        import pyarrow as pa
        from odps.mars_extension.utils import convert_pandas_object_to_string
        from odps.tunnel.io.types import odps_schema_to_arrow_schema

        writer_config = writer_config.copy()
        self._send_cupid_service_request(REQUEST_TYPE_WRITE_TABLE_DATA, writer_config)

        sock_out_file = self.sock.makefile("wb")

        batch_size = write_batch_size or 1024
        batch_idx = 0
        batch_data = to_store_data[
            batch_size * batch_idx : batch_size * (batch_idx + 1)
        ]
        batch_data = convert_pandas_object_to_string(batch_data)
        schema = odps_schema_to_arrow_schema((writer_config["_table_schema"]))
        arrow_writer = _create_arrow_writer(sock_out_file, schema)
        while len(batch_data) > 0:
            batch = pa.RecordBatch.from_pandas(
                batch_data, schema=schema, preserve_index=False
            )
            arrow_writer.write_batch(batch)
            batch_idx += 1
            batch_data = to_store_data[
                batch_size * batch_idx : batch_size * (batch_idx + 1)
            ]
        arrow_writer.close()
        sock_out_file.flush()

        return self._fetch_cupid_service_result()

    def enum_table_partitions(self, odps_params, table_name, partition=None):
        cmd_pack = {
            "odps_params": odps_params,
            "table_name": table_name,
            "partition": partition,
        }
        self._send_cupid_service_request(REQUEST_TYPE_ENUM_TABLE_PARTITIONS, cmd_pack)
        return self._fetch_cupid_service_result()

    def create_table_download_session(
        self,
        odps_params,
        table_name,
        partition,
        columns,
        worker_count=None,
        split_size=None,
        max_chunk_num=None,
        with_split_meta_on_tile=False,
    ):
        cmd_pack = {
            "odps_params": odps_params,
            "table_name": table_name,
            "partition": partition,
            "columns": columns,
            "worker_count": worker_count,
            "split_size": split_size or CHUNK_BYTES_LIMIT,
            "max_chunk_num": max_chunk_num or MAX_CHUNK_NUM,
            "with_split_meta_on_tile": with_split_meta_on_tile,
        }
        self._send_cupid_service_request(
            REQUEST_TYPE_CREATE_TABLE_DOWNLOAD_SESSION, cmd_pack
        )
        result = self._fetch_cupid_service_result()
        return result["splits"], result["split_size"]

    def create_table_upload_session(self, odps_params, table_name):
        cmd_pack = {
            "odps_params": odps_params,
            "table_name": table_name,
        }
        self._send_cupid_service_request(
            REQUEST_TYPE_CREATE_TABLE_UPLOAD_SESSION, cmd_pack
        )
        result = self._fetch_cupid_service_result()
        return result["handle"]

    def commit_table_upload_session(
        self, odps_params, table_name, cupid_handle, blocks, overwrite=True
    ):
        cmd_pack = {
            "odps_params": odps_params,
            "table_name": table_name,
            "cupid_handle": cupid_handle,
            "blocks": blocks,
            "overwrite": overwrite,
        }
        self._send_cupid_service_request(
            REQUEST_TYPE_COMMIT_TABLE_UPLOAD_SESSION, cmd_pack
        )
        self._fetch_cupid_service_result()

    def get_kv(self, key):
        cmd_pack = {"key": key}
        self._send_cupid_service_request(REQUEST_TYPE_GET_KV, cmd_pack)
        result = self._fetch_cupid_service_result()
        return result["value"]

    def put_kv(self, key, value):
        cmd_pack = {"key": key, "value": value}
        self._send_cupid_service_request(REQUEST_TYPE_PUT_KV, cmd_pack)
        self._fetch_cupid_service_result()

    def terminate_instance(self, instance_id):
        cmd_pack = {"instance_id": instance_id}
        self._send_cupid_service_request(REQUEST_TYPE_TERMINATE_INSTANCE, cmd_pack)
        self._fetch_cupid_service_result()

    def get_bearer_token(self):
        self._send_cupid_service_request(REQUEST_TYPE_GET_BEARER_TOKEN, {})
        result = self._fetch_cupid_service_result()
        return result["token"]

    def report_container_status(self, status, message, progress, timeout=-1):
        self._send_cupid_service_request(
            REQUEST_TYPE_REPORT_CONTAINER_STATUS,
            {
                "status": status,
                "message": message,
                "progress": progress,
                "timeout": timeout,
            },
        )
        self._fetch_cupid_service_result()
