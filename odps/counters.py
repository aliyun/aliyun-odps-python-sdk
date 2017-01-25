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
import json

__all__ = ['Counter', 'CounterGroup', 'Counters']


class Counter(object):
    def __init__(self, name, value=0):
        self.name = name
        self.value = value
    
    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def increment(self, incr):
        self.value += incr

    def _get_data_obj(self):
        data = {
                'name' : self.name,
                'value' : self.value
                }

        return data


class CounterGroup(object):
    def __init__(self, name):
        self.name = name
        self.counters = dict()

    def get_name(self):
        return self.name

    def add_counter(self, counter):
        self.counters[counter.get_name()] = counter

    def get_counter(self, counter_name):
        return self.counters.setdefault(counter_name, Counter(counter_name))

    def size(self):
        return len(self.counters)

    def _get_data_obj(self):
        data = {
                'name' : self.name,
                'counters' : [c._get_data_obj() for c in self.counters.values()]
                }

        return data


class Counters(object):
    def __init__(self):
        self.groups = dict()

    def get_group(self, group_name):
        return self.groups.setdefault(group_name, CounterGroup(group_name))

    def get_counter(self, group_name, counter_name):
        return self.get_group(group_name).get_counter(counter_name)

    def size(self):
        return len(self.groups)

    def to_json_string(self, **json_options):
        data = dict()
        for (k, v) in self.groups.items():
            data[k] = v._get_data_obj()

        return json.dumps(data, **json_options)
