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

import json

from ...compat import six
from ..tests.base import grab_iopub_comm, setup_kernel, ui_case

TIMEOUT = 10


@ui_case
def test_html_notify():
    with setup_kernel() as client:
        client.execute("from odps.ui import html_notify")
        shell_msg = client.get_shell_msg(timeout=TIMEOUT)
        content = shell_msg["content"]
        assert content["status"] == "ok"

        msg_id = client.execute('html_notify("TestMessage")')
        iopub_data = grab_iopub_comm(client, msg_id)
        assert any(u"TestMessage" in json.dumps(l) for l in six.itervalues(iopub_data))

        shell_msg = client.get_shell_msg(timeout=TIMEOUT)
        content = shell_msg["content"]
        assert content["status"] == "ok"
