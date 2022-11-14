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

from datetime import datetime
import time
import email.header
import base64

from odps.tests.core import TestBase, to_str
from odps.compat import unittest
from odps.serializers import *
from odps import utils

expected_xml_template = '''<?xml version="1.0" encoding="utf-8"?>
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
  <Student name="s1">s1_content</Student>
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
  <Disabled>false</Disabled>
  <Enabled>true</Enabled>
</Example>
'''

LIST_OBJ_TMPL = '''<?xml version="1.0" ?>
<objs>
  <marker>%s</marker>
  <obj>%s</obj>
</objs>
'''

LIST_OBJ_LAST_TMPL = '''<?xml version="1.0" ?>
<objs>
  <obj>%s</obj>
</objs>
'''

expected_ascii_json = {
  "default_value":1,
  "name":"test1",
  "nullable":True,
  "type":"bigint",
}

expected_email_header_json = {
  "default_value":1,
  "name":"测试一",
  "nullable":True,
  "type":"bigint",
}

class Example(XMLSerializableModel):
    __slots__ = 'name', 'type', 'date', 'lessons', 'teacher', 'student',\
                'professors', 'properties', 'jsn', 'bool_false', 'bool_true'

    _root = 'Example'

    class Teacher(XMLSerializableModel):

        name = XMLNodeField('Name')
        tag = XMLTagField('.')

        def __eq__(self, other):
            return isinstance(other, Example.Teacher) and \
                   self.name == other.name

    class Student(XMLSerializableModel):
        name = XMLNodeAttributeField(attr='name')
        content = XMLNodeField('.')

        def __eq__(self, other):
            return isinstance(other, Example.Student) and \
                   self.name == other.name and \
                   self.content == other.content

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
    date = XMLNodeField('Created', type='rfc822l')
    bool_true = XMLNodeField('Enabled', type='bool')
    bool_false = XMLNodeField('Disabled', type='bool')
    lessons = XMLNodesField('Lessons', 'Lesson')
    teacher = XMLNodeReferenceField(Teacher, 'Teacher')
    student = XMLNodeReferenceField(Student, 'Student')
    professors = XMLNodesReferencesField(Teacher, 'Professors', 'Professor')
    properties = XMLNodePropertiesField('Config', 'Property', key_tag='Name', value_tag='Value')
    jsn = XMLNodeReferenceField(Json, 'json')


class JsonExample(JSONSerializableModel):
    __slots__ = 'default_value', 'name', 'nullable', 'type'

    default_value = JSONNodeField('default_value')
    name = JSONNodeField('name')
    nullable = JSONNodeField('nullable')
    type = JSONNodeField('type')


class Test(TestBase):

    def testSerializers(self):
        teacher = Example.Teacher(name='t1')
        student = Example.Student(name='s1', content='s1_content')
        professors = [Example.Teacher(name='p1'), Example.Teacher(name='p2')]
        jsn = Example.Json(label='json', tags=['t1', 't2'],
                           nest=Example.Json.Nest(name='n'),
                           nests=[Example.Json.Nest(name='n1'), Example.Json.Nest(name='n2')])

        dt = datetime.fromtimestamp(time.mktime(datetime.now().timetuple()))
        example = Example(name='example 1', type='ex', date=dt, bool_true=True, bool_false=False,
                          lessons=['less1', 'less2'], teacher=teacher, student=student,
                          professors=professors, properties={'test': 'true'}, jsn=jsn)
        sel = example.serialize()

        self.assertEqual(
            to_str(expected_xml_template % utils.gen_rfc822(dt, localtime=True)), to_str(sel))

        parsed_example = Example.parse(sel)

        self.assertEqual(example.name, parsed_example.name)
        self.assertEqual(example.type, parsed_example.type)
        self.assertEqual(example.date, parsed_example.date)
        self.assertEqual(example.bool_true, parsed_example.bool_true)
        self.assertEqual(example.bool_false, parsed_example.bool_false)
        self.assertSequenceEqual(example.lessons, parsed_example.lessons)
        self.assertEqual(example.teacher, parsed_example.teacher)
        self.assertEqual(example.student, parsed_example.student)
        self.assertSequenceEqual(example.professors, parsed_example.professors)
        self.assertTrue(len(example.properties) == len(parsed_example.properties) and
                        any(example.properties[it] == parsed_example.properties[it])
                        for it in example.properties)
        self.assertEqual(example.jsn.label, parsed_example.jsn.label)
        self.assertEqual(example.jsn.tags, parsed_example.jsn.tags)
        self.assertEqual(example.jsn.nest, parsed_example.jsn.nest)
        self.assertSequenceEqual(example.jsn.nests, parsed_example.jsn.nests)

    def testPropertyOverride(self):
        def gen_objs(marker):
            assert marker > 0
            if marker >= 3:
                return LIST_OBJ_LAST_TMPL % 3
            else:
                return LIST_OBJ_TMPL % (marker, marker)

        class Objs(XMLSerializableModel):
            skip_null = False

            marker = XMLNodeField('marker')
            obj = XMLNodeField('obj')

        objs = Objs()
        i = 1
        while True:
            objs.parse(gen_objs(i), obj=objs)
            if objs.marker is None:
                break
            i += 1

        self.assertEqual(i, 3)

    def testJsonSerializeAscii(self):
        jsn = JsonExample(default_value=1, name="test1", nullable=True, type="bigint")
        sel = jsn.serialize()
        self.assertEqual(to_str(json.dumps(expected_ascii_json)), to_str(sel))

        encoded_str = json.dumps(expected_ascii_json)
        parsed = jsn.parse(encoded_str)
        self.assertEqual(expected_ascii_json["default_value"], parsed.default_value)
        self.assertEqual(expected_ascii_json["name"], parsed.name)
        self.assertEqual(expected_ascii_json["nullable"], parsed.nullable)
        self.assertEqual(expected_ascii_json["type"], parsed.type)

    def testJsonSerializeEmailHeader(self):
        jsn = JsonExample(default_value=1, name="测试一", nullable=True, type="bigint")
        sel = jsn.serialize()
        self.assertEqual(to_str(json.dumps(expected_email_header_json)), to_str(sel))

        json_bytes = json.dumps(expected_email_header_json).encode("ascii")
        email_header = email.header.Header(json_bytes, "UTF-8")
        encoded_str = email_header.encode(maxlinelen=0)
        parsed = jsn.parse(encoded_str)
        self.assertEqual(expected_email_header_json["default_value"], parsed.default_value)
        self.assertEqual(expected_email_header_json["name"], parsed.name)
        self.assertEqual(expected_email_header_json["nullable"], parsed.nullable)
        self.assertEqual(expected_email_header_json["type"], parsed.type)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
