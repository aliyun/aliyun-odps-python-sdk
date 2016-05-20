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

import copy
import os
import sys
import subprocess
import tempfile
import json
from time import sleep

from odps import tempobj
from odps.tests.core import TestBase, tn, in_coverage_mode

TEMP_TABLE_NAME = tn('pyodps_test_tempobj_cleanup')

SECONDARY_PROCESS_CODE = """
from odps.tests.core import start_coverage
start_coverage()

import os
import sys
import json
from time import sleep

try:
    os.unlink(os.path.realpath(__file__))
except Exception:
    pass

import_paths = json.loads(r\"\"\"
{import_paths}
\"\"\".strip())
odps_info = json.loads(\"\"\"
{odps_info}
\"\"\".strip())

sys.path.extend(import_paths)

from odps import ODPS, tempobj

odps = ODPS(**tempobj.compat_kwargs(odps_info))
sleep(5)
"""


PLENTY_CREATE_CODE = """
from odps.tests.core import start_coverage
start_coverage()

import os
import sys
import json

try:
    os.unlink(os.path.realpath(__file__))
except Exception:
    pass

import_paths = json.loads(r\"\"\"
{import_paths}
\"\"\".strip())
odps_info = json.loads(\"\"\"
{odps_info}
\"\"\".strip())

sys.path.extend(import_paths)

from odps import ODPS, tempobj
from odps.tests.core import tn

odps = ODPS(**tempobj.compat_kwargs(odps_info))
insts = []
for tid in range(10):
    table_name = tn('tmp_pyodps_create_temp_{{}}'.format(tid))
    tempobj.register_temp_table(odps, table_name)
    insts.append(odps.run_sql('create table {{}} (col1 string) lifecycle 1'.format(table_name)))
for inst in insts:
    inst.wait_for_completion()
"""


class TestTempObjs(TestBase):
    def setUp(self):
        super(TestTempObjs, self).setUp()
        if in_coverage_mode():
            # pack cleanup script
            tempobj.CLEANUP_SCRIPT_TMPL = 'from odps.tests.core import start_coverage\nstart_coverage()\n\n' +\
                                          tempobj.CLEANUP_SCRIPT_TMPL
            tempobj._obj_repos = tempobj.ObjectRepositoryLib()

    def _get_odps_json(self):
        return json.dumps(dict(access_id=self.odps.account.access_id,
                               secret_access_key=self.odps.account.secret_access_key,
                               project=self.odps.project,
                               endpoint=self.odps.endpoint))

    def test_temp_object(self):
        class TestTempObject(tempobj.TempObject):
            _type = 'Temp'
            __slots__ = 'param1', 'param2'

        class TestTempObject2(TestTempObject):
            _type = 'Temp2'

        obj1 = TestTempObject('v1', param2='v2')
        assert obj1.param1 == 'v1' and obj1.param2 == 'v2'
        obj2 = TestTempObject('v1', 'v2')

        assert obj1 == obj2
        assert obj1 != 'String'
        assert hash(obj1) == hash(obj2)
        assert obj1 != TestTempObject2('v1', 'v2')

    def test_drop(self):
        tempobj.register_temp_table(self.odps, 'non_exist_table')
        tempobj.register_temp_model(self.odps, 'non_exist_model')
        tempobj.register_temp_volume_partition(self.odps, ('non_exist_vol', 'non_exist_vol_part'))
        tempobj.clean_objects(self.odps)

    def test_cleanup(self):
        self.odps.execute_sql('drop table if exists {0}'.format(TEMP_TABLE_NAME))
        self.odps.execute_sql('create table {0} (col1 string) lifecycle 1'.format(TEMP_TABLE_NAME))
        tempobj.register_temp_table(self.odps, TEMP_TABLE_NAME)
        tempobj._obj_repos._exec_cleanup_script()

        sleep(10)
        assert not self.odps.exist_table(TEMP_TABLE_NAME)

    def test_multi_process(self):
        self.odps.execute_sql('drop table if exists {0}'.format(TEMP_TABLE_NAME))

        self.odps.execute_sql('create table {0} (col1 string) lifecycle 1'.format(TEMP_TABLE_NAME))
        tempobj.register_temp_table(self.odps, TEMP_TABLE_NAME)

        script = SECONDARY_PROCESS_CODE.format(odps_info=self._get_odps_json(), import_paths=json.dumps(sys.path))

        script_name = tempfile.gettempdir() + os.sep + 'tmp_' + str(os.getpid()) + '_secondary_script.py'
        with open(script_name, 'w') as script_file:
            script_file.write(script)
            script_file.close()
        env = copy.deepcopy(os.environ)
        env.update({'WAIT_CLEANUP': '1'})
        subprocess.call([sys.executable, script_name], close_fds=True, env=env)

        sleep(10)

        assert self.odps.exist_table(TEMP_TABLE_NAME)
        self.odps.run_sql('drop table {0}'.format(TEMP_TABLE_NAME))

    def test_plenty_create(self):
        del_insts = [self.odps.run_sql('drop table {0}'.format(tn('tmp_pyodps_create_temp_%d' % n))) for n in range(10)]
        [inst.wait_for_completion() for inst in del_insts]

        script = PLENTY_CREATE_CODE.format(odps_info=self._get_odps_json(), import_paths=json.dumps(sys.path))

        script_name = tempfile.gettempdir() + os.sep + 'tmp_' + str(os.getpid()) + '_plenty_script.py'
        with open(script_name, 'w') as script_file:
            script_file.write(script)
            script_file.close()
        env = copy.deepcopy(os.environ)
        env.update({'WAIT_CLEANUP': '1'})
        subprocess.call([sys.executable, script_name], close_fds=True, env=env)

        sleep(5)
        trial = 4
        case = lambda: all(not self.odps.exist_table(tn('tmp_pyodps_create_temp_%d' % tid)) for tid in range(10))
        while not case():
            trial -= 1
            sleep(5)
            if trial == 0:
                assert case()
