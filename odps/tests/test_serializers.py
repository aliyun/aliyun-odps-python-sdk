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

import email.header
import textwrap
import time
from datetime import datetime

from ..serializers import *
from .. import utils

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


def test_serializers():
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

    assert utils.to_str(
        expected_xml_template % utils.gen_rfc822(dt, localtime=True)
    ) == utils.to_str(sel)

    parsed_example = Example.parse(sel)

    assert example.name == parsed_example.name
    assert example.type == parsed_example.type
    assert example.date == parsed_example.date
    assert example.bool_true == parsed_example.bool_true
    assert example.bool_false == parsed_example.bool_false
    assert list(example.lessons) == list(parsed_example.lessons)
    assert example.teacher == parsed_example.teacher
    assert example.student == parsed_example.student
    assert list(example.professors) == list(parsed_example.professors)
    assert len(example.properties) == len(parsed_example.properties) and \
        (any(example.properties[it] == parsed_example.properties[it])
                        for it in example.properties)
    assert example.jsn.label == parsed_example.jsn.label
    assert example.jsn.tags == parsed_example.jsn.tags
    assert example.jsn.nest == parsed_example.jsn.nest
    assert list(example.jsn.nests) == list(parsed_example.jsn.nests)


def test_coded_json():
    parsed_example = Example.parse(expected_xml_template % utils.gen_rfc822(datetime.now(), localtime=True))
    json_bytes = parsed_example.jsn.serialize().encode("iso-8859-1")
    coded_json = email.header.Header(json_bytes, "iso-8859-1").encode()

    coded = textwrap.dedent('''
    <?xml version="1.0" encoding="utf-8"?>
    <Example type="ex">
      <json>{JSON_CODED}</json>
    </Example>
    ''').strip().replace("{JSON_CODED}", coded_json)

    parsed = Example.parse(coded)
    assert list(parsed.jsn.nests) == list(parsed_example.jsn.nests)


def test_property_override():
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

    assert i == 3
