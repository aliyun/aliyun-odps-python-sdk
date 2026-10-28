# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
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
import subprocess
import sys
import zipfile

import pytest


@pytest.mark.parametrize("entry_name", ("pyodpswrapper.py", "__main__.py"))
def test_resource_pack_uses_entry_credentials(tmp_path, entry_name):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    wrapper = os.path.join(root, "bin", "pyodpswrapper.py")
    if entry_name == "__main__.py":
        renamed_wrapper = tmp_path / entry_name
        with open(wrapper, "rb") as source:
            renamed_wrapper.write_bytes(source.read())
        wrapper = str(renamed_wrapper)
    archive = tmp_path / "resource_probe.zip"
    with zipfile.ZipFile(str(archive), "w") as zf:
        zf.writestr("dw_resource_probe.py", "VALUE = 42\n")

    env = os.environ.copy()
    for key in ("PYODPS_CONFIG_FILE", "PYODPS_RESOURCE_LOADED", "SKYNET_CONNECTION"):
        env.pop(key, None)
    env.update(
        {
            "PYTHONPATH": root,
            "SKYNET_ACCESSID": "placeholder_id",
            "SKYNET_ACCESSKEY": "placeholder_key",
            "SKYNET_PACKAGEID": "placeholder_project",
            "SKYNET_ENDPOINT": "http://127.0.0.1:1/api",
            "SKYNET_SYSTEM_ENV": "prod",
            "TASK_EXEC_PATH": str(tmp_path),
            "PYODPS_ACESS_KEY_ENCRYPT": "false",
        }
    )

    # Header parsing is covered separately; start at its subprocess boundary.
    driver = (
        "import sys\n"
        "from odps_scripts.pyodpswrapper.resource import load_packages_in_subprocess\n"
        "wrapper, archive = sys.argv[1:]\n"
        "sys.argv = [wrapper, '--code-file', 'unused.py']\n"
        "load_packages_in_subprocess(archive)\n"
        "import dw_resource_probe\n"
        "print('RESOURCE_PROBE=%s' % dw_resource_probe.VALUE)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", driver, wrapper, str(archive)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "RESOURCE_PROBE=42" in result.stdout
