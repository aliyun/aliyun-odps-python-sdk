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

from odps.tests.core import TestBase
from odps.ipython.completer import ObjectCompleter


class Test(TestBase):
    def test_object_completer_call(self):
        completer = ObjectCompleter()
        self.assertIsNone(completer.get_list_call("o3.get_bizarre(abc"))

        self.assertTupleEqual(
            completer.get_list_call("table = o1.get_table("),
            ("o1.list_tables(project=None)", None),
        )
        self.assertTupleEqual(
            completer.get_list_call(", o2_a.get_table("),
            ("o2_a.list_tables(project=None)", None),
        )
        self.assertIsNone(completer.get_list_call("o3.get_table(abc"))
        self.assertIsNone(completer.get_list_call('o3.get_table("abc", project='))
        self.assertTupleEqual(
            completer.get_list_call('(o4.delete_table(" def'),
            ('o4.list_tables(prefix=" def", project=None)', '"'),
        )
        self.assertTupleEqual(
            completer.get_list_call("( o5.get_table( 'ghi"),
            ('o5.list_tables(prefix="ghi", project=None)', "'"),
        )
        self.assertTupleEqual(
            completer.get_list_call("obj.o6.write_table( 'ghi"),
            ('obj.o6.list_tables(prefix="ghi", project=None)', "'"),
        )
        self.assertTupleEqual(
            completer.get_list_call(
                'obj.o7.get_table(project= "another_proj", name= \'ghi'
            ),
            ('obj.o7.list_tables(prefix="ghi", project="another_proj")', "'"),
        )
        self.assertTupleEqual(
            completer.get_list_call(
                "obj.o8.get_table('ghi",
                'obj.o8.get_table(\'ghi, project= "another_proj"',
            ),
            ('obj.o8.list_tables(prefix="ghi", project="another_proj")', "'"),
        )
        self.assertTupleEqual(
            completer.get_list_call(
                "obj.o9.get_table(name  = 'ghi",
                'obj.o9.get_table(name  = \'ghi, project= "another_proj"',
            ),
            ('obj.o9.list_tables(prefix="ghi", project="another_proj")', "'"),
        )
        self.assertIsNone(
            completer.get_list_call('obj.o10.get_table(project= "another_proj", \'ghi')
        )
