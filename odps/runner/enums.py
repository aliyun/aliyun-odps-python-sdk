# encoding: utf-8
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

from ..compat import Enum


class EngineType(Enum):
    MOCK = 'MOCK'
    LAMBDA = 'LAMBDA'
    XFLOW = 'XFLOW'
    XLIB = 'XLIB'
    SQL = 'SQL'
    DF = 'DF'
    PS = 'PS'


class PortType(Enum):
    DATA = 'DATA'
    MODEL = 'MODEL'


class PortDirection(Enum):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
