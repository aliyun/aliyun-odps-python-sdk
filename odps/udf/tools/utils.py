# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import logging
import os

from ...compat import ConfigParser
from ...core import ODPS

CONF_FILENAME = "~/.pyouconfig"

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
    global _odps

    f = os.path.expanduser(CONF_FILENAME)
    if not os.path.exists(f):
        return

    config = ConfigParser.RawConfigParser()
    config.read(f)
    access_id = config.get("access_id")
    access_key = config.get("secret_access_key")
    end_point = config.get("endpoint")
    project = config.get("project")
    _odps = ODPS(access_id, access_key, project, end_point)


@require_conf
def get_cache_table(name):
    odps_entry = _odps or ODPS.from_global()
    return odps_entry.read_table(name)
