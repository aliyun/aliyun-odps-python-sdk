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
import os

version = sys.version_info
PY2 = version[0] == 2
PY3 = version[0] == 3
LESS_PY34 = version[:2] < (3, 4)
PY26 = PY2 and version[1] == 6

if PY2 and version[:2] < (2, 6):
    raise Exception('pyodps supports python 2.6+ (including python 3+).')

requirements = []
with open('requirements.txt') as f:
    requirements.extend(f.read().splitlines())

if PY2:
    requirements.append('protobuf>=2.5.0')
else:
    requirements.append('python3-protobuf>=2.5.0')
if LESS_PY34:
    requirements.append('enum34>=1.0.4')
if PY26:
    requirements.append('ordereddict>=1.1')

long_description = None
if os.path.exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()

setup(name='pyodps',
      version='0.2.10',
      description='ODPS Python SDK',
      long_description=long_description,
      author='Wu Wei',
      author_email='weiwu@cacheme.net',
      maintainer='Qin Xuye',
      maintainer_email='qin@qinxuye.me',
      url='http://github.com/aliyun/aliyun-odps-python-sdk',
      license='Apache License 2.0',
      packages=find_packages(exclude=('*.tests.*', '*.tests')),
      include_package_data=True,
      scripts=['scripts/pyou',],
      install_requires=requirements,
      )
