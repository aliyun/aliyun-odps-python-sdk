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

import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import requests

from ....utils import to_str
from .client import CUPID_APP_NAME, NOTEBOOK_NAME


DEFAULT_NOTEBOOK_PORT = 50003
ACTOR_ADDRESS = "127.0.0.1:32123"
ACTOR_UID = "BearerTokenActor"


def start_notebook(port):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jupyter",
            "notebook",
            "--ip",
            "0.0.0.0",
            "--no-browser",
            "--allow-root",
            "--port",
            str(port),
            '--NotebookApp.token=""',
            '--NotebookApp.password=""',
            "--NotebookApp.disable_check_xsrf=True",
        ]
    )
    while True:
        if proc.poll() is not None:
            raise SystemError("Notebook not started.")
        try:
            resp = requests.get("http://localhost:{}".format(port))
            if resp.status_code == 200:
                return
        except:
            pass
        time.sleep(1)


def wait_mars_ready(kvstore, name):
    # wait mars web ready
    while True:
        json_val = to_str(kvstore.get(name))
        if json_val:
            config = json.loads(to_str(json_val))
            mars_endpoint = to_str(config["endpoint"])
            return mars_endpoint


def dump_endpoint(endpoint):
    path = os.path.join(os.path.expanduser("~"), ".mars")
    with open(path, "w") as f:
        f.write(endpoint)


def refresh_bearer_token(cupid_context):
    path = os.path.join(os.path.expanduser("~"), ".bearertoken")
    token = cupid_context.get_bearer_token()

    refresh_time = 60
    while True:
        time.sleep(refresh_time)
        with open(path, "w") as f:
            f.write(token)


start_code = f"""
import asyncio
import os
from odps import options, ODPS
from odps.accounts import BearerTokenAccount

options.verbose = True

from mars.session import new_session
path = os.path.join(os.path.expanduser('~'), '.mars')
with open(path, 'r') as f:
    endpoint = f.read()
mars_session = new_session(endpoint).as_default()

def refresh_bearer_token():
    import mars.oscar as mo

    async def _refresher():
        ref = await mo.actor_ref(uid='{ACTOR_UID}', address='{ACTOR_ADDRESS}')
        return await ref.get_bearer_token()

    loop = mars_session._isolation.loop
    return asyncio.run_coroutine_threadsafe(_refresher(), loop).result()

bearer_token = refresh_bearer_token()
account = BearerTokenAccount(bearer_token, get_bearer_token_fun=refresh_bearer_token)
project = os.environ.get('ODPS_PROJECT_NAME')
endpoint = os.environ.get('ODPS_RUNTIME_ENDPOINT')
o = ODPS(None, None, account=account, project=project, endpoint=endpoint)
"""


def config_startup():
    startup_dir = os.path.join(
        os.path.expanduser("~"), ".ipython/profile_default/startup"
    )
    if not os.path.exists(startup_dir):
        os.makedirs(startup_dir)
    f_path = os.path.join(startup_dir, "set_default_session.py")
    with open(f_path, "w") as f:
        f.write(start_code)


async def create_bearer_token_actor():
    import mars.oscar as mo

    class BearerTokenActor(mo.Actor):
        def get_bearer_token(self):
            from cupid import context

            ctx = context()
            return ctx.get_bearer_token()

    pool = await mo.create_actor_pool(address=ACTOR_ADDRESS, n_process=1)
    await mo.create_actor(
        BearerTokenActor, uid=ACTOR_UID, address=pool.external_address
    )
    await pool.join()


def _main():
    from cupid import context
    from mars.utils import get_next_port

    cupid_context = context()
    mars_endpoint = wait_mars_ready(cupid_context.kv_store(), CUPID_APP_NAME)
    host_addr = socket.gethostbyname(socket.gethostname())

    os.environ.pop("KUBE_API_ADDRESS")

    if os.environ.get("VM_ENGINE_TYPE") == "hyper":
        notebook_port = DEFAULT_NOTEBOOK_PORT
    else:
        notebook_port = str(get_next_port())

    endpoint = "http://{0}:{1}".format(host_addr, notebook_port)

    # dump endpoint to ~/.mars
    dump_endpoint(mars_endpoint)

    # add startup script for notebook
    config_startup()

    # start notebook
    start_notebook(notebook_port)

    # modify in hyper mode
    if os.environ.get("VM_ENGINE_TYPE") == "hyper":
        endpoint = socket.gethostname() + "-{}".format(notebook_port)
    cupid_context.register_application(NOTEBOOK_NAME, endpoint)

    asyncio.run(create_bearer_token_actor())


if __name__ == "__main__":
    _main()
