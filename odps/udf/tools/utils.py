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

import os
import sys
import logging

from odps import (core, accounts)


CONF_FILENAME = '.pyouconfig'

logger = logging.getLogger(__name__)
_odps = None


class UDFToolError(Exception):
    pass


def require_conf(func):
    def f(*args, **kwargs):
        if _odps is None:
            get_conf()
        return func(*args, **kwargs)
    return f
    

def get_conf():
    import ConfigParser
    global _odps

    home_dir = os.environ.get('HOME')
    if not home_dir:
        raise UDFToolError('Cannot find home dir, '
                           'perhaps you are using windowns :(')
    f = os.path.join(home_dir, CONF_FILENAME)
    if not os.path.exists(f):
        access_id = raw_input('access_id:')
        
    config = ConfigParser.RawConfigParser()
    config.read(f)
    access_id = config.get('access_id')
    access_key = config.get('secret_access_key')
    end_point = config.get('endpoint')
    project = config.get('project')
    _odps = core.ODPS(access_id, access_key,
                      project, end_point)


@require_conf
def get_cache_table(name):
    return _odps.read_table(name)
