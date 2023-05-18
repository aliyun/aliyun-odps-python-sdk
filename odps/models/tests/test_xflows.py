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

from ...examples import create_iris
from ...tests.core import tn
from ...utils import to_text
from .. import XFlows

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


def test_x_flows(odps):
    assert odps.get_project().xflows is odps.get_project().xflows

    xflows = list(odps.list_xflows())
    assert len(xflows) >= 0


def test_x_flow_instance_to_xml(odps):
    xflow_name = 'pyodps_t_tmp_xflow_algo_name'
    project = 'algo_project'
    parameters = {'key': 'value'}
    properties = {'odps.setting': 'value'}

    got_xml = odps.get_project(project).xflows._gen_xflow_instance_xml(
        xflow_name=xflow_name, xflow_project=project, parameters=parameters,
        properties=properties)
    assert to_text(got_xml) == to_text(EXPECTED_XFLOW_INSTANCE_XML)

    got_xml = odps.get_project(project).xflows._gen_xflow_instance_xml(
        xflow_name=xflow_name, xflow_project=project, parameters=parameters,
        properties=properties, priority=1)
    assert to_text(got_xml) == to_text(EXPECTED_PRIORITY_XFLOW_INSTANCE_XML)


def test_run_x_flow_instance(odps):
    xflow_name = 'test_xflow'
    if not odps.exist_xflow(xflow_name):
        return

    instance = odps.execute_xflow(xflow_name, parameters=dict())
    xflow_results = odps.get_xflow_results(instance)

    assert isinstance(xflow_results, dict)
    assert all(
        map(lambda x: isinstance(x, XFlows.XFlowResult.XFlowAction), xflow_results.values()))


def test_iter_sub_instances(odps):
    table = create_iris(odps, tn('test_iris_table'))
    model_name = tn('test_xflow_model')
    try:
        odps.delete_offline_model(model_name, if_exists=True)
    except:
        pass
    try:
        xflow_inst = odps.run_xflow(
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
        for k, v in odps.iter_xflow_sub_instances(xflow_inst, check=True):
            sub_insts[k] = v
        assert xflow_inst.is_terminated
        assert len(sub_insts) > 0
    finally:
        odps.delete_offline_model(model_name, if_exists=True)
