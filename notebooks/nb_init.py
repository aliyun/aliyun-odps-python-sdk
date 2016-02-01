# encoding: utf-8
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
"""
Notebook initializing code
You can edit your ipython_config.py to detect and add this file to
 exec_lines to help create pre-defined objects.
"""
import sys
from six.moves.configparser import ConfigParser

sys.path.append('../')

config = ConfigParser()
config.read('../odps/tests/test.conf')

access_id = config.get('odps', 'access_id')
secret_access_key = config.get('odps', 'secret_access_key')
project = config.get('odps', 'project')
endpoint = config.get('odps', 'endpoint')
