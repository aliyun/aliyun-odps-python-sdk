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
import odps.counters

_origin_int = int
_annotated_classes = {}

__all__ = ['get_execution_context', 'ExecutionContext', 'BaseUDAF', 'BaseUDTF', 'int', 'annotate', 'get_annotation']

class ExecutionContext(object):
    counters = odps.counters.Counters()

    def get_counters(self):
        return self.counters

    def get_counters_as_json_string(self):
        return self.counters.to_json_string()

class BaseUDAF(object):
    pass

class BaseUDTF(object):
    def forward(self, *args):
        pass

    def close(self):
        pass

def get_execution_context():
    return ExecutionContext()

def int(v, silent=True):
    v = _int(v)
    if type(v) is long:
        if silent:
            return None
        else:
            raise OverflowError('Python int too large to convert to bigint: %s' % v)
    return v

def annotate(prototype):
    def ann(clz):
        _annotated_classes[clz] = prototype
        return clz
    return ann

def get_annotation(clz):
    return _annotated_classes.get(clz)
