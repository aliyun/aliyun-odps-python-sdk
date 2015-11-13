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

from odps.tests.core import TestBase, to_str
from odps.compat import unittest
from odps.models import SQLTask, Task


template = '''<?xml version="1.0" ?>
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


class Test(TestBase):
    def testSQLTaskToXML(self):
        query = 'select * from dual'

        task = SQLTask(query=query)
        to_xml = task.serialize()
        right_xml = template % {'sql': query}

        self.assertEqual(to_str(to_xml), to_str(right_xml))

        task = Task.parse(None, to_xml)
        self.assertIsInstance(task, SQLTask)


if __name__ == '__main__':
    unittest.main()
