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
import platform
import sys


def get_property(name, args=None):
    environ_name = '_'.join(n.upper() for n in name.split('.'))
    args = args or sys.argv[1:]
    for arg in args:
        if arg.startswith('-D%s=' % arg):
            _, value_part = arg.split('=', 1)
            if value_part.startswith('\"') and value_part.endswith('\"'):
                value_part = value_part[1:-1].replace('""', '"')
            return value_part
    return get_environ(environ_name)


# resolve the mdzz environ values
def get_environ(key, default=None):
    val = os.environ.get(key)
    if val:
        if val.startswith('"'):
            val = val.strip('"')
        return val
    return default


def build_image_name(app_name, with_dist_suffix=True):
    from .config import options

    prefix = options.cupid.image_prefix
    version = options.cupid.image_version
    if with_dist_suffix:
        dist_suffix = "-cp" + "".join(str(x) for x in sys.version_info[:2]) + "-" + platform.machine() + "-cpu"
    else:
        dist_suffix = ""

    if prefix is None:
        if version is None:
            return None
        else:
            return app_name + ':' + version + dist_suffix
    else:
        return prefix + app_name + ':' + version + dist_suffix
