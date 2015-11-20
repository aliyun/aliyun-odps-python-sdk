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

from datetime import datetime
import time

from odps.tests.core import TestBase, to_str
from odps.compat import unittest
from odps.serializers import *
from odps import utils

expected_xml_template = '''<?xml version="1.0" ?>
<Example type="ex">
  <Name>example 1</Name>
  <Created>%s</Created>
  <Lessons>
    <Lesson>less1</Lesson>
    <Lesson>less2</Lesson>
  </Lessons>
  <Teacher>
    <Name>t1</Name>
  </Teacher>
  <Professors>
    <Professor>
      <Name>p1</Name>
    </Professor>
    <Professor>
      <Name>p2</Name>
    </Professor>
  </Professors>
  <Config>
    <Property>
      <Name>test</Name>
      <Value>true</Value>
    </Property>
  </Config>
  <json>{"label": "json", "tags": [{"tag": "t1"}, {"tag": "t2"}], "nest": {"name": "n"}, "nests": {"nest": [{"name": "n1"}, {"name": "n2"}]}}</json>
</Example>
'''


class Example(XMLSerializableModel):
    __slots__ = 'name', 'type', 'date', 'lessons', 'teacher', \
                'professors', 'properties', 'jsn'

    _root = 'Example'

    class Teacher(XMLSerializableModel):

        name = XMLNodeField('Name')
        tag = XMLTagField('.')

        def __eq__(self, other):
            return isinstance(other, Example.Teacher) and \
                   self.name == other.name

    class Json(JSONSerializableModel):

        __slots__ = 'label', 'tags', 'nest', 'nests'

        class Nest(JSONSerializableModel):
            name = JSONNodeField('name')

            def __eq__(self, other):
                return isinstance(other, Example.Json.Nest) and \
                       self.name == other.name

        label = JSONNodeField('label')
        tags = JSONNodesField('tags', 'tag')
        nest = JSONNodeReferenceField(Nest, 'nest')
        nests = JSONNodesReferencesField(Nest, 'nests', 'nest')

    name = XMLNodeField('Name')
    type = XMLNodeAttributeField('.', attr='type')
    date = XMLNodeField('Created', serialize_callback=lambda t: utils.gen_rfc822(t, localtime=True),
                        parse_callback=utils.parse_rfc822)
    lessons = XMLNodesField('Lessons', 'Lesson')
    teacher = XMLNodeReferenceField(Teacher, 'Teacher')
    professors = XMLNodesReferencesField(Teacher, 'Professors', 'Professor')
    properties = XMLNodePropertiesField('Config', 'Property', key_tag='Name', value_tag='Value')
    jsn = XMLNodeReferenceField(Json, 'json')


class Test(TestBase):

    def testSerializers(self):
        teacher = Example.Teacher(name='t1')
        professors = [Example.Teacher(name='p1'), Example.Teacher(name='p2')]
        jsn = Example.Json(label='json', tags=['t1', 't2'],
                           nest=Example.Json.Nest(name='n'),
                           nests=[Example.Json.Nest(name='n1'), Example.Json.Nest(name='n2')])

        dt = datetime.fromtimestamp(time.mktime(datetime.now().timetuple()))
        example = Example(name='example 1', type='ex', date=dt,
                          lessons=['less1', 'less2'], teacher=teacher, professors=professors,
                          properties={'test': 'true'}, jsn=jsn)
        sel = example.serialize()

        self.assertEqual(
            to_str(expected_xml_template % utils.gen_rfc822(dt, localtime=True)), to_str(sel))

        parsed_example = Example.parse(sel)

        self.assertEqual(example.name, parsed_example.name)
        self.assertEqual(example.type, parsed_example.type)
        self.assertEqual(example.date, parsed_example.date)
        self.assertSequenceEqual(example.lessons, parsed_example.lessons)
        self.assertEqual(example.teacher, parsed_example.teacher)
        self.assertSequenceEqual(example.professors, parsed_example.professors)
        self.assertTrue(len(example.properties) == len(parsed_example.properties) and
                        any(example.properties[it] == parsed_example.properties[it])
                        for it in example.properties)
        self.assertEqual(example.jsn.label, parsed_example.jsn.label)
        self.assertEqual(example.jsn.tags, parsed_example.jsn.tags)
        self.assertEqual(example.jsn.nest, parsed_example.jsn.nest)
        self.assertSequenceEqual(example.jsn.nests, parsed_example.jsn.nests)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
