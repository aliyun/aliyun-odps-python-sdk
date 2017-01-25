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

from datetime import datetime

from odps.tests.core import TestBase
from odps.compat import unittest, six
from odps.models import Projects


class Test(TestBase):

    def testProjectsExists(self):
        not_exists_project_name = 'a_not_exists_project'
        self.assertFalse(self.odps.exist_project(not_exists_project_name))

        self.assertTrue(self.odps.exist_project(self.odps.project))

    def testProject(self):
        self.assertIs(self.odps.get_project(), self.odps.get_project())
        self.assertIs(Projects(client=self.odps.rest), Projects(client=self.odps.rest))

        del self.odps._projects[self.odps.project]
        project = self.odps.get_project()

        self.assertEqual(project.name, self.odps.project)

        self.assertIsNone(project._getattr('owner'))
        self.assertIsNone(project._getattr('comment'))
        self.assertIsNone(project._getattr('creation_time'))
        self.assertIsNone(project._getattr('last_modified_time'))
        self.assertIsNone(project._getattr('project_group_name'))
        self.assertIsNone(project._getattr('properties'))
        self.assertIsNone(project._getattr('extended_properties'))
        self.assertIsNone(project._getattr('state'))
        self.assertIsNone(project._getattr('clusters'))

        self.assertFalse(project.is_loaded)

        self.assertIsInstance(project.extended_properties, dict)
        self.assertIsInstance(project.owner, six.string_types)
        self.assertIsInstance(project.creation_time, datetime)
        self.assertIsInstance(project.last_modified_time, datetime)
        self.assertIsInstance(project.properties, dict)
        self.assertGreater(len(project.properties), 0)
        self.assertGreater(len(project.extended_properties), 0)
        self.assertIsInstance(project.state, six.string_types)

        self.assertTrue(project.is_loaded)


if __name__ == "__main__":
    unittest.main()
