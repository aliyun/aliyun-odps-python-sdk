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

from setuptools import setup, find_packages, Extension

import sys
import os
import warnings

version = sys.version_info
PY2 = version[0] == 2
PY3 = version[0] == 3
PY26 = PY2 and version[1] == 6

if PY2 and version[:2] < (2, 6):
    raise Exception('pyodps supports python 2.6+ (including python 3+).')

requirements = []
with open('requirements.txt') as f:
    requirements.extend(f.read().splitlines())

if PY26:
    requirements.append('ordereddict>=1.1')
    requirements.append('threadpool>=1.3')


long_description = None
if os.path.exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()


setup_options = dict(
    name='pyodps',
    version='0.5.1',
    description='ODPS Python SDK',
    long_description=long_description,
    author='Wu Wei',
    author_email='weiwu@cacheme.net',
    maintainer='Qin Xuye',
    maintainer_email='qin@qinxuye.me',
    url='http://github.com/aliyun/aliyun-odps-python-sdk',
    license='Apache License 2.0',
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries',
    ],
    packages=find_packages(exclude=('*.tests.*', '*.tests')),
    include_package_data=True,
    scripts=['scripts/pyou', ],
    install_requires=requirements,
)

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext

    ext_modules = cythonize([
        Extension('odps.tunnel.pb.encoder_c', ['odps/tunnel/pb/encoder_c.pyx']),
        Extension('odps.tunnel.pb.internal_c', ['odps/tunnel/pb/internal_c.pyx']),
        Extension('odps.tunnel.pb.util_c', ['odps/tunnel/pb/util_c.pyx']),
        Extension('odps.crc32c_c', ['odps/src/crc32c/*.pyx'])
    ])

    setup_options['cmdclass'] = {'build_ext': build_ext}
    setup_options['ext_modules'] = ext_modules
except ImportError:
    pass


setup(**setup_options)
