#!/usr/bin/env python
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

from setuptools import setup, find_packages

import sys

version = sys.version_info
if version[0] != 2:
    raise Exception('pyodps does not support python3 yet.')
if version[1] < 7:
    raise Exception('Python version must be >= 2.7 (but not python3)')

requirements = []
with open('requirements.txt') as f:
    requirements.extend(f.read().splitlines())

setup(name='pyodps',
      version='0.1.0',
      description='ODPS Python SDK',
      author='Wu Wei',
      author_email='weiwu@cacheme.net',
      maintainer='Qin Xuye',
      maintainer_email='qin@qinxuye.me',
      url='http://github.com/aliyun/aliyun-odps-python-sdk',
      packages=find_packages(exclude=('*.test.*', '*.test')),
      scripts=['scripts/pyou',],
      install_requires=requirements,
      )
