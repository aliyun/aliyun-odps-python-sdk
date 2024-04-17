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

import io
import os
import select
import shutil
import socket
import sys
import tempfile
import threading

import pytest

from ..compat import futures, quote_plus, urlparse
from ..core import ODPS

try:
    import requests_unixsocket
except ImportError:
    requests_unixsocket = None


class UnixEndpointProxy(object):
    def __init__(self, sock_file, endpoint):
        self._sock_file = sock_file
        self._remote_endpoint = endpoint
        self._unix_server_sock = None
        self._running = True
        self._server_thread = None

    @classmethod
    def _client_handler(cls, conn, endpoint):
        parsed_ep = urlparse(endpoint)
        remote_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        default_port = 80 if parsed_ep.scheme == "http" else 443
        netloc_part = parsed_ep.netloc.split(":") + [default_port]
        remote_conn.connect((netloc_part[0], int(netloc_part[1])))

        loc_buf = io.BytesIO()
        remote_buf = io.BytesIO()
        try:
            while True:
                sel_timeout = 120 if "CI_MODE" in os.environ else None
                ready_to_read, ready_to_write, in_error = select.select(
                    [conn, remote_conn],
                    [conn, remote_conn],
                    [conn, remote_conn],
                    sel_timeout,
                )
                if in_error:
                    raise RuntimeError("in_error not empty")
                for read_sock in ready_to_read:
                    data = read_sock.recv(8192)
                    if not data:
                        raise RuntimeError("connection closed")
                    if read_sock is conn:
                        loc_buf.write(data)
                    else:
                        remote_buf.write(data)
                for write_sock in ready_to_write:
                    if write_sock is conn and remote_buf.tell():
                        write_sock.send(remote_buf.getvalue())
                        remote_buf = io.BytesIO()
                    elif write_sock is remote_conn and loc_buf.tell():
                        write_sock.send(loc_buf.getvalue())
                        loc_buf = io.BytesIO()
        except RuntimeError:
            conn.close()
            remote_conn.close()

    def _server_thread_func(self):
        pool = futures.ThreadPoolExecutor(10)

        self._unix_server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._unix_server_sock.bind(self._sock_file)
        self._unix_server_sock.listen(10)

        conns = []
        try:
            while self._running:
                conn, _ = self._unix_server_sock.accept()
                conns.append(conn)
                if not self._running:
                    break
                conn.setblocking(False)
                pool.submit(self._client_handler, conn, self._remote_endpoint)
        finally:
            for conn in conns:
                conn.close()

    def start(self):
        self._server_thread = threading.Thread(target=self._server_thread_func)
        self._server_thread.daemon = True
        self._server_thread.start()

    def stop(self):
        if self._server_thread is None:
            return
        self._running = False
        stop_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stop_sock.connect(self._sock_file)
        stop_sock.close()
        self._unix_server_sock.close()
        self._server_thread.join()
        self._server_thread = None


@pytest.mark.skipif(
    requests_unixsocket is None or sys.platform == "win32",
    reason="Need requests_unixsocket and unix-like OS to run this test",
)
def test_unixsocket_access(odps):
    odps_endpoint = odps.endpoint
    parsed_endpoint = urlparse(odps_endpoint)

    dir_name = tempfile.mkdtemp(prefix="pyodps-test-")
    sock_name = os.path.join(dir_name, "unix_ep.sock")
    proxy_obj = UnixEndpointProxy(sock_name, odps_endpoint)
    try:
        proxy_obj.start()

        local_endpoint = "http+unix://%s%s" % (
            quote_plus(sock_name), parsed_endpoint.path
        )
        unix_odps = ODPS(
            account=odps.account, project=odps.project, endpoint=local_endpoint
        )
        res = unix_odps.create_resource(
            "test_unix_socket_resource.txt", "file", fileobj="data"
        )
        res.drop()
    finally:
        proxy_obj.stop()
        shutil.rmtree(dir_name)
