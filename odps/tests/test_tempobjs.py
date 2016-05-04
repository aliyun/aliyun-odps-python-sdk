#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import os
import sys
import subprocess
import tempfile
import json
from time import sleep

from odps import tempobj
from odps.tests.core import TestBase

TEMP_TABLE_NAME = 'pyodps_test_tempobj_cleanup'

SECONDARY_PROCESS_CODE = """
import os
import json
from time import sleep
from odps import ODPS, tempobj

try:
    os.unlink(os.path.realpath(__file__))
except Exception:
    pass

odps_info = json.loads(\"\"\"
{odps_info}
\"\"\".strip())

odps = ODPS(**tempobj.compat_kwargs(odps_info))
sleep(5)
"""


PLENTY_CREATE_CODE = """
import os
import json
from odps import ODPS, tempobj

try:
    os.unlink(os.path.realpath(__file__))
except Exception:
    pass

odps_info = json.loads(\"\"\"
{odps_info}
\"\"\".strip())

odps = ODPS(**tempobj.compat_kwargs(odps_info))
for tid in range(50):
    odps.execute_sql('create table tmp_pyodps_create_temp_{{0}} (col1 string) lifecycle 1'.format(tid))
    tempobj.register_temp_table(odps, 'tmp_pyodps_create_temp_{{0}}'.format(tid))
"""


class Test(TestBase):
    def _get_odps_json(self):
        return json.dumps(dict(access_id=self.odps.account.access_id,
                               secret_access_key=self.odps.account.secret_access_key,
                               project=self.odps.project,
                               endpoint=self.odps.endpoint))

    def test_cleanup(self):
        self.odps.execute_sql('drop table if exists {0}'.format(TEMP_TABLE_NAME))
        self.odps.execute_sql('create table {0} (col1 string) lifecycle 1'.format(TEMP_TABLE_NAME))
        tempobj.register_temp_table(self.odps, TEMP_TABLE_NAME)
        tempobj._obj_repos._exec_cleanup_script()
        sleep(5)
        assert not self.odps.exist_table(TEMP_TABLE_NAME)

    def test_multi_process(self):
        self.odps.execute_sql('drop table if exists {0}'.format(TEMP_TABLE_NAME))

        self.odps.execute_sql('create table {0} (col1 string) lifecycle 1'.format(TEMP_TABLE_NAME))
        tempobj.register_temp_table(self.odps, TEMP_TABLE_NAME)

        script = SECONDARY_PROCESS_CODE.format(odps_info=self._get_odps_json())

        script_name = tempfile.gettempdir() + os.sep + 'tmp_' + str(os.getpid()) + '_secondary_script.py'
        with open(script_name, 'w') as script_file:
            script_file.write(script)
            script_file.close()
        subprocess.call([sys.executable, script_name], close_fds=True, env={'WAIT_CLEANUP': '1'})

        sleep(5)

        assert self.odps.exist_table(TEMP_TABLE_NAME)

    def test_plenty_create(self):
        del_insts = [self.odps.run_sql('drop table tmp_pyodps_create_temp_%d' % n) for n in range(50)]
        [inst.wait_for_completion() for inst in del_insts]

        script = PLENTY_CREATE_CODE.format(odps_info=self._get_odps_json())

        script_name = tempfile.gettempdir() + os.sep + 'tmp_' + str(os.getpid()) + '_plenty_script.py'
        with open(script_name, 'w') as script_file:
            script_file.write(script)
            script_file.close()
        subprocess.call([sys.executable, script_name], close_fds=True, env={'WAIT_CLEANUP': '1'})

        sleep(10)
        assert all(not self.odps.exist_table('tmp_pyodps_create_temp_%d' % tid) for tid in range(50))
