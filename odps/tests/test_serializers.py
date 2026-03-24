#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import datetime
import email.header
import textwrap
import time

from .. import serializers, utils
from ..config import option_context
from ..models.tasks.core import format_cdata

expected_xml_template = """<?xml version="1.0" encoding="utf-8"?>
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
  <Config2>
    <Property name="test2">
      <Value>false</Value>
    </Property>
  </Config2>
  <Config3>
    <Property name="test3">test-val</Property>
  </Config3>
</Example>
"""

LIST_OBJ_TMPL = """<?xml version="1.0" ?>
<objs>
  <marker>%s</marker>
  <obj>%s</obj>
</objs>
"""

LIST_OBJ_LAST_TMPL = """<?xml version="1.0" ?>
<objs>
  <obj>%s</obj>
</objs>
"""

DATA_WITH_BODY_TMPL = """<?xml version="1.0" encoding="utf-8"?>
<DataWithBody>
  <Body><![CDATA[{text_to_repl};]]></Body>
</DataWithBody>
"""
DATA_WITH_BODY_TMPL2 = """<?xml version="1.0" encoding="utf-8"?>
<DataWithBody><Body><![CDATA[{text_to_repl};]]></Body></DataWithBody>
""".strip()

DATA_WITH_BODY_LEGACY = """<?xml version="1.0" encoding="utf-8"?>
<DataWithBody>
  <Body><![CDATA['"content"';]]></Body>
</DataWithBody>
"""


class Example(serializers.XMLSerializableModel):
    __slots__ = (
        "name",
        "type",
        "date",
        "lessons",
        "teacher",
        "student",
        "professors",
        "properties",
        "jsn",
        "bool_false",
        "bool_true",
    )

    _root = "Example"

    class Teacher(serializers.XMLSerializableModel):
        name = serializers.XMLNodeField("Name")
        tag = serializers.XMLTagField(".")

        def __eq__(self, other):
            return isinstance(other, Example.Teacher) and self.name == other.name

    class Student(serializers.XMLSerializableModel):
        name = serializers.XMLNodeAttributeField(attr="name")
        content = serializers.XMLNodeField(".")

        def __eq__(self, other):
            return (
                isinstance(other, Example.Student)
                and self.name == other.name
                and self.content == other.content
            )

    class Json(serializers.JSONSerializableModel):
        __slots__ = "label", "tags", "nest", "nests"

        class Nest(serializers.JSONSerializableModel):
            name = serializers.JSONNodeField("name")

            def __eq__(self, other):
                return isinstance(other, Example.Json.Nest) and self.name == other.name

        label = serializers.JSONNodeField("label")
        tags = serializers.JSONNodesField("tags", "tag")
        nest = serializers.JSONNodeReferenceField(Nest, "nest")
        nests = serializers.JSONNodesReferencesField(Nest, "nests", "nest")

    name = serializers.XMLNodeField("Name")
    type = serializers.XMLNodeAttributeField(".", attr="type")
    date = serializers.XMLNodeField("Created", type="rfc822l")
    bool_true = serializers.XMLNodeField("Enabled", type="bool")
    bool_false = serializers.XMLNodeField("Disabled", type="bool")
    lessons = serializers.XMLNodesField("Lessons", "Lesson")
    teacher = serializers.XMLNodeReferenceField(Teacher, "Teacher")
    student = serializers.XMLNodeReferenceField(Student, "Student")
    professors = serializers.XMLNodesReferencesField(Teacher, "Professors", "Professor")
    properties = serializers.XMLNodePropertiesField(
        "Config", "Property", key_tag="Name", value_tag="Value"
    )
    properties2 = serializers.XMLNodePropertiesField(
        "Config2", "Property", key_attr="name", value_tag="Value"
    )
    properties3 = serializers.XMLNodePropertiesField(
        "Config3", "Property", key_attr="name"
    )
    jsn = serializers.XMLNodeReferenceField(Json, "json")


def test_serializers():
    teacher = Example.Teacher(name="t1")
    student = Example.Student(name="s1", content="s1_content")
    professors = [Example.Teacher(name="p1"), Example.Teacher(name="p2")]
    jsn = Example.Json(
        label="json",
        tags=["t1", "t2"],
        nest=Example.Json.Nest(name="n"),
        nests=[Example.Json.Nest(name="n1"), Example.Json.Nest(name="n2")],
    )

    dt = datetime.datetime.fromtimestamp(
        time.mktime(datetime.datetime.now().timetuple())
    )
    example = Example(
        name="example 1",
        type="ex",
        date=dt,
        bool_true=True,
        bool_false=False,
        lessons=["less1", "less2"],
        teacher=teacher,
        student=student,
        professors=professors,
        properties={"test": "true"},
        properties2={"test2": "false"},
        properties3={"test3": "test-val"},
        jsn=jsn,
    )
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
    assert len(example.properties) == len(parsed_example.properties) and (
        any(example.properties[it] == parsed_example.properties[it])
        for it in example.properties
    )
    assert len(example.properties2) == len(parsed_example.properties2) and (
        any(example.properties2[it] == parsed_example.properties2[it])
        for it in example.properties2
    )
    assert len(example.properties3) == len(parsed_example.properties3) and (
        any(example.properties3[it] == parsed_example.properties3[it])
        for it in example.properties3
    )
    assert example.jsn.label == parsed_example.jsn.label
    assert example.jsn.tags == parsed_example.jsn.tags
    assert example.jsn.nest == parsed_example.jsn.nest
    assert list(example.jsn.nests) == list(parsed_example.jsn.nests)


def test_coded_json():
    parsed_example = Example.parse(
        expected_xml_template
        % utils.gen_rfc822(datetime.datetime.now(), localtime=True)
    )
    json_bytes = parsed_example.jsn.serialize().encode("iso-8859-1")
    coded_json = email.header.Header(json_bytes, "iso-8859-1").encode()

    coded = (
        textwrap.dedent(
            """
    <?xml version="1.0" encoding="utf-8"?>
    <Example type="ex">
      <json>{JSON_CODED}</json>
    </Example>
    """
        )
        .strip()
        .replace("{JSON_CODED}", coded_json)
    )

    parsed = Example.parse(coded)
    assert list(parsed.jsn.nests) == list(parsed_example.jsn.nests)


def test_property_override():
    def gen_objs(marker):
        assert marker > 0
        if marker >= 3:
            return LIST_OBJ_LAST_TMPL % 3
        else:
            return LIST_OBJ_TMPL % (marker, marker)

    class Objs(serializers.XMLSerializableModel):
        skip_null = False

        marker = serializers.XMLNodeField("marker")
        obj = serializers.XMLNodeField("obj")

    objs = Objs()
    i = 1
    while True:
        objs.parse(gen_objs(i), obj=objs)
        if objs.marker is None:
            break
        i += 1

    assert i == 3


def test_cdata_unescape():
    class DataWithBody(serializers.XMLSerializableModel):
        _root = "DataWithBody"

        body = serializers.XMLNodeField(
            "Body", serialize_callback=lambda x: format_cdata(x, True)
        )

    obj = DataWithBody(body="'&quot;content&quot;'")
    assert obj.serialize() == DATA_WITH_BODY_TMPL.replace("{text_to_repl}", obj.body)

    with option_context() as options_ctx:
        options_ctx.use_legacy_xml_unescape = True
        assert obj.serialize() == DATA_WITH_BODY_LEGACY

    # long texts are not prettified
    obj = DataWithBody(body="large_text" * (serializers._MAX_PRETTIFY_SIZE // 4))
    assert obj.serialize().replace("'", '"') == DATA_WITH_BODY_TMPL2.replace(
        "{text_to_repl}", obj.body
    )

    with option_context() as options_ctx:
        # still prettified with legacy option
        options_ctx.use_legacy_xml_unescape = True
        assert obj.serialize() == DATA_WITH_BODY_TMPL.replace(
            "{text_to_repl}", obj.body
        )
