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

from odps.tests.core import TestBase, to_str, tn
from odps.examples import create_iris
from odps.compat import unittest
from odps.models import XFlows

EXPECTED_XFLOW_INSTANCE_XML = '''<?xml version="1.0" encoding="utf-8"?>
<Instance>
  <XflowInstance>
    <Project>algo_project</Project>
    <Xflow>pyodps_t_tmp_xflow_algo_name</Xflow>
    <Parameters>
      <Parameter>
        <Key>key</Key>
        <Value>value</Value>
      </Parameter>
    </Parameters>
    <Config>
      <Property>
        <Name>odps.setting</Name>
        <Value>value</Value>
      </Property>
    </Config>
  </XflowInstance>
</Instance>
'''
EXPECTED_PRIORITY_XFLOW_INSTANCE_XML = '''<?xml version="1.0" encoding="utf-8"?>
<Instance>
  <XflowInstance>
    <Project>algo_project</Project>
    <Xflow>pyodps_t_tmp_xflow_algo_name</Xflow>
    <Parameters>
      <Parameter>
        <Key>key</Key>
        <Value>value</Value>
      </Parameter>
    </Parameters>
    <Priority>1</Priority>
    <Config>
      <Property>
        <Name>odps.setting</Name>
        <Value>value</Value>
      </Property>
    </Config>
  </XflowInstance>
</Instance>
'''


class Test(TestBase):
    def testXFlows(self):
        self.assertIs(self.odps.get_project().xflows, self.odps.get_project().xflows)

        xflows = list(self.odps.list_xflows())
        self.assertGreaterEqual(len(xflows), 0)

    def testXFlowInstanceToXML(self):
        xflow_name = 'pyodps_t_tmp_xflow_algo_name'
        project = 'algo_project'
        parameters = {'key': 'value'}
        properties = {'odps.setting': 'value'}

        got_xml = self.odps.get_project(project).xflows._gen_xflow_instance_xml(
            xflow_name=xflow_name, xflow_project=project, parameters=parameters,
            properties=properties)
        self.assertEqual(to_str(got_xml), to_str(EXPECTED_XFLOW_INSTANCE_XML))

        got_xml = self.odps.get_project(project).xflows._gen_xflow_instance_xml(
            xflow_name=xflow_name, xflow_project=project, parameters=parameters,
            properties=properties, priority=1)
        self.assertEqual(to_str(got_xml), to_str(EXPECTED_PRIORITY_XFLOW_INSTANCE_XML))

    def testRunXFlowInstance(self):
        xflow_name = 'test_xflow'
        if not self.odps.exist_xflow(xflow_name):
            return

        instance = self.odps.execute_xflow(xflow_name, parameters=dict())
        xflow_results = self.odps.get_xflow_results(instance)

        self.assertIsInstance(xflow_results, dict)
        self.assertTrue(all(
            map(lambda x: isinstance(x, XFlows.XFlowResult.XFlowAction), xflow_results.values())))

    def testIterSubInstances(self):
        table = create_iris(self.odps, tn('test_iris_table'))
        model_name = tn('test_xflow_model')
        try:
            xflow_inst = self.odps.run_xflow(
                'LogisticRegression',
                'algo_public',
                dict(
                    featureColNames='sepal_length,sepal_width,petal_length,petal_width',
                    labelColName='category',
                    inputTableName=table.name,
                    modelName=model_name,
                ),
                hints={"settings": "{\"SKYNET_ID\": \"12345\"}"}
            )
            sub_insts = dict()
            for k, v in self.odps.iter_xflow_sub_instances(xflow_inst):
                sub_insts[k] = v
            self.assertTrue(xflow_inst.is_terminated)
            self.assertTrue(len(sub_insts) > 0)
        finally:
            self.odps.delete_offline_model(model_name, if_exists=True)


if __name__ == '__main__':
    unittest.main()