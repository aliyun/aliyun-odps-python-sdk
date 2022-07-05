# !/usr/bin/env python
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
import base64
import sys
import json
import socket
import subprocess

from .client import GS_COORDINATOR_NAME


DEFAULT_GS_COORDINATOR_PORT = 63800
DEFAULT_GS_COORDINATOR_GATEWAY_PORT = 64800


def start_coordinator(args, coordinator_port, vineyard_socket):
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gscoordinator",
            "--enable_k8s",
            "false",
            "--hosts",
            ",".join(args["worker_pod_ip_list"]),
            "--port",
            str(coordinator_port),
            "--num_workers",
            str(args["num_workers"]),
            "--vineyard_socket",
            vineyard_socket,
            "--log_level",
            "DEBUG",
            "--timeout_seconds",
            "-1",
        ],
        stdout=sys.__stdout__,
        stderr=sys.__stdout__,
        bufsize=1,
    )


def start_coordinator_gateway(args, coordinator_port, coordinator_gateway_port):
    subprocess.Popen(
        [
            "/usr/local/bin/graphscope-gateway",
            "-grpc-server-endpoint",
            "localhost:%s" % coordinator_port,
            "--gateway-address",
            ":%s" % coordinator_gateway_port,
            "-v",
            "100",
            "-alsologtostderr",
        ],
        stdout=sys.__stdout__,
        stderr=sys.__stdout__,
        bufsize=1,
    )


def _main():
    argv = sys.argv[1]
    args_dict = json.loads(base64.b64decode(argv).decode())
    print("launch graphscope:", args_dict)

    from cupid import context

    cupid_context = context()
    host_addr = socket.gethostbyname(socket.gethostname())

    os.environ.pop("KUBE_API_ADDRESS")

    coordinator_port = args_dict.get("port", None) or DEFAULT_GS_COORDINATOR_PORT
    coordinator_gateway_port = (
        args_dict.get("gateway_port", None) or DEFAULT_GS_COORDINATOR_GATEWAY_PORT
    )

    endpoint = "http://{0}:{1}".format(host_addr, coordinator_gateway_port)
    kvstore = cupid_context.kv_store()
    kvstore[GS_COORDINATOR_NAME] = json.dumps(dict(endpoint=endpoint))

    # start coordinator
    vineyard_socket = os.environ.get("VINEYARD_IPC_SOCKET", "/tmp/vineyard.sock")
    start_coordinator(args_dict, coordinator_port, vineyard_socket)
    start_coordinator_gateway(args_dict, coordinator_port, coordinator_gateway_port)

    # modify in hyper mode
    if os.environ.get("VM_ENGINE_TYPE") == "hyper":
        endpoint = socket.gethostname() + "-{}".format(coordinator_port)
    cupid_context.register_application(GS_COORDINATOR_NAME, endpoint)


if __name__ == "__main__":
    _main()
