#!/usr/bin/env python
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

import copy
import os
import sys
import subprocess
import tempfile
import json
from time import sleep

from odps import tempobj, utils
from odps.tests.core import TestBase, tn, in_coverage_mode

TEMP_TABLE_NAME = tn('pyodps_test_tempobj_cleanup')

SECONDARY_PROCESS_CODE = """
#-*- coding:utf-8 -*-
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
""".lstrip()


PLENTY_CREATE_CODE = u"""
#-*- coding:utf-8 -*-
from odps.tests.core import start_coverage
start_coverage()

import os
import sys
import json

try:
    os.unlink(os.path.realpath(__file__))
except Exception:
    pass

import_paths = json.loads({import_paths!r})
odps_info = json.loads({odps_info!r})

sys.path.extend(import_paths)

from odps import ODPS, tempobj
from odps.tests.core import tn

odps = ODPS(**tempobj.compat_kwargs(odps_info))
insts = []
for tid in range(10):
    table_name = tn('tmp_pyodps_create_temp_{{0}}'.format(tid))
    tempobj.register_temp_table(odps, table_name)
    insts.append(odps.run_sql('create table {{0}} (col1 string) lifecycle 1'.format(table_name)))
for inst in insts:
    inst.wait_for_completion()
""".lstrip()


TEMP_FUNCTION_CODE = u"""
#-*- coding:utf-8 -*-
from odps.tests.core import start_coverage
start_coverage()

import os
import sys
import json

try:
    os.unlink(os.path.realpath(__file__))
except Exception:
    pass

import_paths = json.loads({import_paths!r})
odps_info = json.loads({odps_info!r})

sys.path.extend(import_paths)

from odps import ODPS, tempobj
from odps.tests.core import tn

odps = ODPS(**tempobj.compat_kwargs(odps_info))

resource_name = {resource_name!r}
function_name = {function_name!r}

tempobj.register_temp_resource(odps, resource_name)
tempobj.register_temp_function(odps, function_name)

res_text = \"\"\"
from odps.udf import annotate
@annotate("bigint->bigint")
class TempFun(object):
   def evaluate(self, arg0):
       return arg0
\"\"\"
if odps.exist_resource(resource_name + '.py'):
    odps.delete_resource(resource_name + '.py')
res = odps.create_resource(resource_name + '.py', 'py', file_obj=res_text)

if odps.exist_function(function_name):
    odps.delete_function(function_name)
fun = odps.create_function(function_name, class_type=resource_name + '.TempFun', resources=[res, ])
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

    def testTempObject(self):
        class TestTempObject(tempobj.TempObject):
            _type = 'Temp'
            __slots__ = 'param1', 'param2'

        class TestTempObject2(TestTempObject):
            _type = 'Temp2'

        obj1 = TestTempObject('v1', param2='v2')
        self.assertEqual((obj1.param1, obj1.param2), ('v1', 'v2'))
        obj2 = TestTempObject('v1', 'v2')

        self.assertEqual(obj1, obj2)
        self.assertNotEqual(obj1, 'String')
        self.assertEqual(hash(obj1), hash(obj2))
        self.assertNotEqual(obj1, TestTempObject2('v1', 'v2'))

    def testDrop(self):
        tempobj.register_temp_table(self.odps, 'non_exist_table')
        tempobj.register_temp_model(self.odps, 'non_exist_model')
        tempobj.register_temp_function(self.odps, 'non_exist_function')
        tempobj.register_temp_resource(self.odps, 'non_exist_resource')
        tempobj.register_temp_volume_partition(self.odps, ('non_exist_vol', 'non_exist_vol_part'))
        tempobj.clean_stored_objects(self.odps)

    def testCleanup(self):
        self.odps.execute_sql('drop table if exists {0}'.format(TEMP_TABLE_NAME))
        self.odps.execute_sql('create table {0} (col1 string) lifecycle 1'.format(TEMP_TABLE_NAME))
        tempobj.register_temp_table(self.odps, TEMP_TABLE_NAME)
        tempobj.clean_objects(self.odps)
        sleep(10)
        assert not self.odps.exist_table(TEMP_TABLE_NAME)

    def testCleanupScript(self):
        self.odps.execute_sql('drop table if exists {0}'.format(TEMP_TABLE_NAME))
        self.odps.execute_sql('create table {0} (col1 string) lifecycle 1'.format(TEMP_TABLE_NAME))
        tempobj.register_temp_table(self.odps, TEMP_TABLE_NAME)
        tempobj._obj_repos._exec_cleanup_script()

        sleep(10)
        assert not self.odps.exist_table(TEMP_TABLE_NAME)

    def testMultiProcess(self):
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

    def testPlentyCreate(self):
        del_insts = [self.odps.run_sql('drop table {0}'.format(tn('tmp_pyodps_create_temp_%d' % n))) for n in range(10)]
        [inst.wait_for_completion() for inst in del_insts]

        script = PLENTY_CREATE_CODE.format(
            odps_info=self._get_odps_json(),
            import_paths=utils.to_text(json.dumps(sys.path)),
        )

        script_name = tempfile.gettempdir() + os.sep + 'tmp_' + str(os.getpid()) + '_plenty_script.py'
        with open(script_name, 'wb') as script_file:
            script_file.write(script.encode())
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
                self.assertTrue(case())

    def testTempFunctions(self):
        resource_name = tn('pyodps_test_tempobj_temp_resource') + '.py'
        function_name = tn('pyodps_test_tempobj_temp_function')

        script = TEMP_FUNCTION_CODE.format(
            odps_info=self._get_odps_json(), import_paths=json.dumps(sys.path),
            resource_name=resource_name, function_name=function_name,
        )

        script_name = tempfile.gettempdir() + os.sep + 'tmp_' + str(os.getpid()) + '_temp_functions.py'
        with open(script_name, 'w') as script_file:
            script_file.write(script)
            script_file.close()
        env = copy.deepcopy(os.environ)
        env.update({'WAIT_CLEANUP': '1'})
        subprocess.call([sys.executable, script_name], close_fds=True, env=env)

        sleep(10)

        self.assertFalse(self.odps.exist_resource(resource_name))
        self.assertFalse(self.odps.exist_function(function_name))
