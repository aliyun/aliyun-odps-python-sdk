# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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
import json
import socket
import subprocess

import requests

from .web import CUPID_APP_NAME
from .client import NOTEBOOK_NAME
from ...utils import to_str


DEFAULT_NOTEBOOK_PORT = 50003


def start_notebook(port):
    proc = subprocess.Popen([sys.executable, '-m', 'jupyter', 'notebook', '--ip', '0.0.0.0',
                             '--no-browser', '--allow-root', '--port', str(port),
                             '--NotebookApp.token=""', '--NotebookApp.password=""',
                             '--NotebookApp.disable_check_xsrf=True'])
    while True:
        if proc.poll() is not None:
            raise SystemError('Notebook not started.')
        try:
            resp = requests.get('http://localhost:{}'.format(port))
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
            mars_endpoint = to_str(config['endpoint'])
            return mars_endpoint


def dump_endpoint(endpoint):
    path = os.path.join(os.path.expanduser('~'), '.mars')
    with open(path, 'w') as f:
        f.write(endpoint)


start_code = """
import os
from odps import options

options.verbose = True

from mars.session import new_session
path = os.path.join(os.path.expanduser('~'), '.mars')
with open(path, 'r') as f:
    endpoint = f.read()
mars_session = new_session(endpoint).as_default()
"""


def config_startup():
    startup_dir = os.path.join(os.path.expanduser('~'), '.ipython/profile_default/startup')
    if not os.path.exists(startup_dir):
        os.makedirs(startup_dir)
    f_path = os.path.join(startup_dir, 'set_default_session.py')
    with open(f_path, 'w') as f:
        f.write(start_code)


def _main():
    from cupid import context
    from mars.utils import get_next_port

    cupid_context = context()
    mars_endpoint = wait_mars_ready(cupid_context.kv_store(), CUPID_APP_NAME)
    host_addr = socket.gethostbyname(socket.gethostname())

    os.environ.pop('KUBE_API_ADDRESS')

    if os.environ.get('VM_ENGINE_TYPE') == 'hyper':
        notebook_port = DEFAULT_NOTEBOOK_PORT
    else:
        notebook_port = str(get_next_port())

    endpoint = 'http://{0}:{1}'.format(host_addr, notebook_port)

    # dump endpoint to ~/.mars
    dump_endpoint(mars_endpoint)

    # add startup script for notebook
    config_startup()

    # start notebook
    start_notebook(notebook_port)

    # modify in hyper mode
    if os.environ.get('VM_ENGINE_TYPE') == 'hyper':
        endpoint = socket.gethostname() + "-{}".format(notebook_port)
    cupid_context.register_application(NOTEBOOK_NAME, endpoint)


if __name__ == '__main__':
    _main()
