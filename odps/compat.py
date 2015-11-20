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

import sys
import logging.config
try:
    import xml.etree.cElementTree as ElementTree
except ImportError:
    import xml.etree.ElementTree as ElementTree

import six

PY26 = six.PY2 and sys.version_info[1] == 6
LESS_PY26 = six.PY2 and sys.version_info[1] < 6
LESS_PY32 = six.PY3 and sys.version_info[1] < 2

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

if six.PY3:
    lrange = lambda x: list(range(x))
    lzip = lambda *x: list(zip(*x))
    lkeys = lambda x: list(x.keys())
    lvalues = lambda x: list(x.values())
    litems = lambda x: list(x.items())

    import unittest
    from collections import OrderedDict

    dictconfig = lambda config: logging.config.dictConfig(config)
else:
    lrange = range
    lzip = zip
    lkeys = lambda x: x.keys()
    lvalues = lambda x: x.values()
    litems = lambda x: x.items()

    if PY26:
        import unittest2 as unittest
        try:
            from ordereddict import OrderedDict
        except ImportError:
            raise
    else:
        import unittest
        from collections import OrderedDict

        dictconfig = lambda config: logging.config.dictConfig(config)

if PY26 or LESS_PY32:
    try:
        from .tests.dictconfig import dictConfig
        dictconfig = lambda config: dictConfig(config)
    except ImportError:
        pass

