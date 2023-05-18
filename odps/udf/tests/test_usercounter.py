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

import json

from ...compat import six
from ...counters import *


def test_counter():
    counter = Counter("test", 12)
    assert "test" == counter.get_name()
    assert 12 == counter.get_value()

    counter = Counter("test2")
    assert "test2" == counter.get_name()
    assert 0 == counter.get_value()


def test_counter_group():
    counter_group = CounterGroup("test_group")
    assert "test_group" == counter_group.get_name()

    counter_group.get_counter("test")
    counter = Counter("test2")
    counter_group.add_counter(counter)

    assert 2 == counter_group.size()


def test_counters():
    def _normalize_counter(json_str):
        obj = json.loads(json_str)
        for v in six.itervalues(obj):
            if 'counters' not in v:
                continue
            v['counters'] = sorted(v['counters'], key=lambda item: item['name'])

        return json.dumps(obj, sort_keys=True)

    result_json = '''
            {
              "group1" : {
                "name" : "group1",
                "counters" : [
                     {
                       "name" : "test1",
                       "value" : 1
                     },
                     {
                       "name" : "test2",
                       "value" : 2
                     }
                ]},
              "group2" : {
                "name" : "group2",
                "counters" : [
                     {
                       "name" : "test3",
                       "value" : 3
                     }
                 ]
               }
            }
            '''

    counters = Counters()
    c1 = counters.get_group("group1").get_counter("test1")
    c1.increment(1)
    c2 = counters.get_group("group1").get_counter("test2")
    c2.increment(2)
    c3 = counters.get_group("group2").get_counter("test3")
    c3.increment(3)

    assert 2 == counters.size()
    assert _normalize_counter(result_json) == _normalize_counter(counters.to_json_string())
