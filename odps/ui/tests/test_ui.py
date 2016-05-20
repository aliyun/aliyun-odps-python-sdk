# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import json
import os
import sys
import textwrap

from odps.ui.tests.base import UITestBase, setup_kernel, ui_case
from odps.compat import six

TIMEOUT = 10


class TestUI(UITestBase):
    @ui_case
    def test_script_read(self):
        print(os.environ)
        print(sys.argv)
        with setup_kernel() as client:
            test_code = textwrap.dedent("""
            from odps.compat import six
            from odps.ui.common import load_script
            'True' if len(load_script()) > 0 else 'False'
            """).strip()
            msg_id = client.execute(test_code)

            exec_result_data = self.grab_execute_result(client, msg_id)
            self.assertDictEqual(exec_result_data['data'], {'text/plain': "'True'"})

            shell_msg = client.get_shell_msg(block=True, timeout=TIMEOUT)
            content = shell_msg['content']
            self.assertEqual(content['status'], 'ok')

    @ui_case
    def test_html_notify(self):
        with setup_kernel() as client:
            client.execute('from odps.ui import html_notify')
            shell_msg = client.get_shell_msg(block=True, timeout=TIMEOUT)
            content = shell_msg['content']
            self.assertEqual(content['status'], 'ok')

            msg_id = client.execute('html_notify("TestMessage")')
            iopub_data = self.grab_iopub_comm(client, msg_id)
            assert any(u"TestMessage" in json.dumps(l)
                       for l in six.itervalues(iopub_data))

            shell_msg = client.get_shell_msg(block=True, timeout=TIMEOUT)
            content = shell_msg['content']
            self.assertEqual(content['status'], 'ok')
