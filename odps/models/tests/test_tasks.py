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

from odps.tests.core import TestBase, to_str
from odps.errors import ODPSError
from odps.compat import unittest
from odps.config import options
from odps.models import SQLTask, MergeTask, CupidTask, SQLCostTask, Task
from odps.tests.core import tn

try:
    import pytz
except ImportError:
    pytz = None


sql_template = '''<?xml version="1.0" encoding="utf-8"?>
<SQL>
  <Name>AnonymousSQLTask</Name>
  <Config>
    <Property>
      <Name>settings</Name>
      <Value>{"odps.sql.udf.strict.mode": "true"}</Value>
    </Property>
  </Config>
  <Query><![CDATA[%(sql)s;]]></Query>
</SQL>
'''

sql_tz_template = '''<?xml version="1.0" encoding="utf-8"?>
<SQL>
  <Name>AnonymousSQLTask</Name>
  <Config>
    <Property>
      <Name>settings</Name>
      <Value>{"odps.sql.timezone": "%(tz)s"}</Value>
    </Property>
  </Config>
  <Query><![CDATA[%(sql)s;]]></Query>
</SQL>
'''

merge_template = '''<?xml version="1.0" encoding="utf-8"?>
<Merge>
  <Name>%(name)s</Name>
  <Config>
    <Property>
      <Name>settings</Name>
      <Value>{"odps.merge.cross.paths": "true"}</Value>
    </Property>
  </Config>
  <TableName>%(table)s</TableName>
</Merge>
'''

cupid_template = '''<?xml version="1.0" encoding="utf-8"?>
<CUPID>
  <Name>task_1</Name>
  <Config>
    <Property>
      <Name>type</Name>
      <Value>cupid</Value>
    </Property>
    <Property>
      <Name>settings</Name>
      <Value>{"odps.cupid.wait.am.start.time": 600}</Value>
    </Property>
  </Config>
  <Plan><![CDATA[plan_text]]></Plan>
</CUPID>
'''

sql_cost_template = '''<?xml version="1.0" encoding="utf-8"?>
<SQLCost>
  <Name>AnonymousSQLCostTask</Name>
  <Query><![CDATA[%(sql)s;]]></Query>
</SQLCost>
'''


class Test(TestBase):
    def testTaskClassType(self):
        typed = Task(type='SQL', query='select * from dual')
        self.assertIsInstance(typed, SQLTask)

        unknown_typed = Task(type='UnknownType')
        self.assertIs(type(unknown_typed), Task)
        self.assertRaises(ODPSError, lambda: unknown_typed.serialize())

        untyped = Task()
        self.assertIs(type(untyped), Task)
        self.assertRaises(ODPSError, lambda: untyped.serialize())

    def testSQLTaskToXML(self):
        query = 'select * from dual'

        task = SQLTask(query=query)
        to_xml = task.serialize()
        right_xml = sql_template % {'sql': query}

        self.assertEqual(to_str(to_xml), to_str(right_xml))

        task = Task.parse(None, to_xml)
        self.assertIsInstance(task, SQLTask)

    @unittest.skipIf(pytz is None, 'pytz not installed')
    def testSQLTaskToXMLTimezone(self):
        from odps.lib import tzlocal
        query = 'select * from dual'

        try:
            options.local_timezone = True
            local_zone_name = tzlocal.get_localzone().zone
            task = SQLTask(query=query)
            task.update_sql_settings()
            to_xml = task.serialize()
            right_xml = sql_tz_template % {'sql': query, 'tz': local_zone_name}

            self.assertEqual(to_str(to_xml), to_str(right_xml))

            options.local_timezone = False
            task = SQLTask(query=query)
            task.update_sql_settings()
            to_xml = task.serialize()
            right_xml = sql_tz_template % {'sql': query, 'tz': 'Etc/GMT'}

            self.assertEqual(to_str(to_xml), to_str(right_xml))

            options.local_timezone = pytz.timezone('Asia/Shanghai')
            task = SQLTask(query=query)
            task.update_sql_settings()
            to_xml = task.serialize()
            right_xml = sql_tz_template % {'sql': query, 'tz': options.local_timezone.zone}

            self.assertEqual(to_str(to_xml), to_str(right_xml))
        finally:
            options.local_timezone = None

    def testMergeTaskToXML(self):
        task = MergeTask('task_1', table='table_name')
        task.update_settings({'odps.merge.cross.paths': True})
        to_xml = task.serialize()
        right_xml = merge_template % dict(name='task_1', table='table_name')

        self.assertEqual(to_str(to_xml), to_str(right_xml))

        task = Task.parse(None, to_xml)
        self.assertIsInstance(task, MergeTask)

    def testRunMergeTask(self):
        table_name = tn('pyodps_test_merge_task_table')
        if self.odps.exist_table(table_name):
            self.odps.delete_table(table_name)

        table = self.odps.create_table(table_name, ('col string', 'part1 string, part2 string'))
        table.create_partition('part1=1,part2=1', if_not_exists=True)
        self.odps.write_table(table_name, [('col_name', )], partition='part1=1,part2=1')
        inst = self.odps.run_merge_files(table_name, 'part1=1, part2="1"')
        self.waitContainerFilled(lambda: inst.tasks)

        task = inst.tasks[0]
        self.assertIsInstance(task, MergeTask)

        try:
            inst.stop()
        except:
            pass

        self.odps.delete_table(table_name)

    def testCupidTaskToXML(self):
        task = CupidTask('task_1', 'plan_text', {'odps.cupid.wait.am.start.time': 600})
        to_xml = task.serialize()
        right_xml = cupid_template

        self.assertEqual(to_str(to_xml), to_str(right_xml))

        task = Task.parse(None, to_xml)
        self.assertIsInstance(task, CupidTask)

    def testSQLCostTaskToXML(self):
        query = 'select * from dual'
        task = SQLCostTask(query=query)
        to_xml = task.serialize()
        right_xml = sql_cost_template % {'sql': query}

        self.assertEqual(to_str(to_xml), to_str(right_xml))

        task = Task.parse(None, to_xml)
        self.assertIsInstance(task, SQLCostTask)


if __name__ == '__main__':
    unittest.main()
