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

from contextlib import contextmanager
from subprocess import Popen, PIPE

from ...compat import Empty
from ...tests.core import ignore_case

try:
    from jupyter_core import paths
    from jupyter_client import BlockingKernelClient
    from ipython_genutils import py3compat

    _has_jupyter = True
except ImportError:
    _has_jupyter = False

SETUP_TIMEOUT = 60
TIMEOUT = 15
DEFAULT_CMD = """
from odps.tests.core import start_coverage
start_coverage()

from IPython import start_kernel
start_kernel()
""".lstrip()


def ui_case(func):
    if _has_jupyter:
        return func
    else:
        return ignore_case(func, "UI case skipped, since no Jupyter installation found.")


def grab_iopub_messages(client, msg_id):
    try:
        iopub_msg = {}

        while (not iopub_msg or
               iopub_msg['parent_header']['msg_id'] != msg_id or
               iopub_msg['msg_type'] != 'status' or
               'execution_state' not in iopub_msg['content'] or
               iopub_msg['content']['execution_state'] != "idle"):

            iopub_msg = client.get_iopub_msg(timeout=TIMEOUT)
            if iopub_msg['parent_header']['msg_id'] != msg_id:
                continue
            yield iopub_msg
    except Empty:
        pass


def grab_iopub_comm(client, msg_id):
    iopub_data = {}
    for iopub_msg in grab_iopub_messages(client, msg_id):
        content = iopub_msg['content']
        if 'comm_id' not in content:
            continue
        comm_id = content['comm_id']
        if iopub_msg['msg_type'] == 'comm_open':
            iopub_data[comm_id] = []
        elif iopub_msg['msg_type'] == 'comm_msg' and comm_id in iopub_data:
            iopub_data[comm_id].append(content['data'])
    return iopub_data


def grab_execute_result(client, msg_id):
    for iopub_msg in grab_iopub_messages(client, msg_id):
        content = iopub_msg['content']
        if iopub_msg['msg_type'] == 'execute_result':
            return content


# from https://github.com/ipython/ipykernel/blob/master/ipykernel/tests/test_embed_kernel.py
@contextmanager
def setup_kernel(cmd=DEFAULT_CMD):
    """start an embedded kernel in a subprocess, and wait for it to be ready
    Returns
    -------
    kernel_manager: connected KernelManager instance
    """
    kernel = Popen([sys.executable, '-c', cmd], stdout=PIPE, stderr=PIPE)
    connection_file = os.path.join(
        paths.jupyter_runtime_dir(),
        'kernel-%i.json' % kernel.pid,
    )
    # wait for connection file to exist, timeout after 5s
    tic = time.time()
    while not os.path.exists(connection_file) \
        and kernel.poll() is None \
        and time.time() < tic + SETUP_TIMEOUT:
        time.sleep(0.1)

    if kernel.poll() is not None:
        o, e = kernel.communicate()
        e = py3compat.cast_unicode(e)
        raise IOError("Kernel failed to start:\n%s" % e)

    if not os.path.exists(connection_file):
        if kernel.poll() is None:
            kernel.terminate()
        raise IOError("Connection file %r never arrived" % connection_file)

    client = BlockingKernelClient(connection_file=connection_file)
    client.load_connection_file()
    client.start_channels()
    client.wait_for_ready()

    try:
        yield client
    finally:
        client.stop_channels()
        kernel.terminate()
