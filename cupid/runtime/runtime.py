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
import warnings

from odps.compat import enum

from ..utils import get_environ
from ..rpc import CupidRpcController, SandboxRpcChannel

try:
    from ..proto import cupid_process_service_pb2 as process_pb
except TypeError:
    warnings.warn('Cannot import protos from pycupid: '
        'consider upgrading your protobuf python package.', ImportWarning)
    raise ImportError

logger = logging.getLogger(__name__)
_pid_to_context = dict()


def context():
    try:
        from .ctypes_libs import Subprocess_Container_Init, Subprocess_StartFDReceiver
    except ImportError:
        return

    pid = os.getpid()
    if pid not in _pid_to_context:
        Subprocess_Container_Init()
        Subprocess_StartFDReceiver()
        _pid_to_context[pid] = RuntimeContext()
    return _pid_to_context[pid]


class ContainerStatus(enum.Enum):
    START = 1
    TERMINATED = 2
    RUNNING = 3


class WorkItemProgress(enum.Enum):
    WIP_WAITING = 1      # waiting for enough resource
    WIP_READY = 2        # resource ready, but not running yet
    WIP_RUNNING = 3
    WIP_TERMINATING = 4  # worker has tried to terminate the work item but is not yet confirmed
    WIP_TERMINATED = 5   # work confirms that the work item is terminated normally
    WIP_FAILED = 6       # worker confirms that the work item has failed
    WIP_INTERRUPTED = 7
    WIP_DEAD = 8
    WIP_INTERRUPTING = 9


class RuntimeContext(object):
    def __init__(self):
        from .ctypes_libs import ChannelConf, ChannelSlaveClient
        # init channels
        server_read_pipe_num = int(get_environ('serverReadPipeNum', 2))
        server_write_pipe_num = int(get_environ('serverWritePipeNum', 2))
        client_read_pipe_num = int(get_environ('clientReadPipeNum', 2))
        client_write_pipe_num = int(get_environ('clientWritePipeNum', 2))
        cpu_cores = min(1, int(get_environ('workerCores', 2)))
        channel_conf = ChannelConf()
        channel_conf.set_integer('odps.subprocess.channel.rpc.handler.count',
                                 max(cpu_cores / 2, 1))
        channel_conf.set_integer('odps.subprocess.channel.stream.handler.count', cpu_cores * 3)

        logger.info("serverReadPipeNum: %d, serverWritePipeNum: %d, clientReadPipeNum: %d, "
                    "clientWritePipeNum: %d, cpuCores: %d" %
                    (server_read_pipe_num, server_write_pipe_num, client_read_pipe_num,
                     client_write_pipe_num, cpu_cores))
        self._channel_client = ChannelSlaveClient(client_write_pipe_num, client_read_pipe_num, 'slave_client')
        self._channel_client.start()

    @staticmethod
    def is_context_ready():
        return os.getpid() in _pid_to_context

    @property
    def channel_client(self):
        return self._channel_client

    def register_application(self, app_name, address):
        return self.channel_client.sync_call('report_app_address', json.dumps(
            dict(name=app_name, address=address)
        ))

    def get_bearer_token(self):
        return self.channel_client.sync_call('get_bearer_token', '')

    def kv_store(self):
        """
        Get key-value store of Cupid Service

        :param session: cupid session
        :return: kv-store instance
        """
        from ..io.kvstore import CupidKVStore

        if not hasattr(self, '_cupid_kv_store'):
            self._cupid_kv_store = CupidKVStore()
        return self._cupid_kv_store

    @staticmethod
    def prepare_channel():
        controller = CupidRpcController()
        channel = SandboxRpcChannel()
        stub = process_pb.ProcessService_Stub(channel)
        env = process_pb.EnvEntry()
        resp = stub.Prepare(controller, env, None)
        return resp.entries

    def report_container_status(self, status, message, progress, timeout=-1):
        params = json.dumps(dict(
            status=str(status.value),
            message=json.dumps(dict(
                status=message,
                progress=str(progress.value),
            )),
        ))
        self._channel_client.sync_call("report_container_status", params, timeout=timeout)
