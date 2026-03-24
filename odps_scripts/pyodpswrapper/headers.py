# -*- coding: utf-8 -*-
# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import re
import sys

_re_flag = 0 if sys.version_info[0] == 2 else re.ASCII
MARS_VERSION_RE = re.compile(
    r"^[ \t\f]*#.*?mars_version[ \t]*[:=][ \t]*([-\w.]+)", _re_flag
)
DUMP_TRACEBACK_RE = re.compile(
    r"^[ \t\f]*#.*?dump_traceback[ \t]*[:=][ \t]*([-\w.]+)", _re_flag
)
PROFILE_RE = re.compile(r"^[ \t\f]*#.*?profile[ \t]*[:=][ \t]*([-\w.]+)", _re_flag)
RESOURCE_PACK_RE = re.compile(
    r"^[ \t\f]*#.*?resource_pack[ \t]*[:=][ \t]*([^\r\n]+)", _re_flag
)
USE_SPAWN_METHOD_RE = re.compile(
    r"^[ \t\f]*#.*?use_spawn_method[ \t]*[:=][ \t]*([-\w.]+)", _re_flag
)

_run_flags = {}


def _scan_file_comments(code, *regexes):
    if not code:
        return [None] * len(regexes)

    result_dict = dict()
    for line in code.splitlines():
        if sys.version_info[0] >= 3:
            try:
                if isinstance(line, bytes):
                    line = line.decode()
            except ValueError:
                continue
        for idx, regex in enumerate(regexes):
            match = regex.match(line)
            if match and len(match.groups()):
                result_dict[idx] = match.group(1)
    return [result_dict.get(i) for i in range(len(regexes))]


def _config_run_flag(flag, config_str, default=False):
    if config_str is None:
        _run_flags[flag] = default
    elif config_str.strip().lower() in ("0", "false"):
        _run_flags[flag] = False
    else:
        _run_flags[flag] = True


def config_with_headers(code):
    from .mars_support import config_mars_version
    from .resource import load_packages_in_subprocess

    file_comments = _scan_file_comments(
        code,
        MARS_VERSION_RE,
        DUMP_TRACEBACK_RE,
        PROFILE_RE,
        USE_SPAWN_METHOD_RE,
        RESOURCE_PACK_RE,
    )

    # load source headers
    if sys.version_info[0] >= 3:
        config_mars_version(file_comments[0])
        _config_run_flag("dump_traceback", file_comments[1])
        _config_run_flag("profile", file_comments[2])
        _config_run_flag("use_spawn_method", file_comments[3], default=None)
    load_packages_in_subprocess(file_comments[4])


def get_run_flags():
    return _run_flags


def use_spawn_method():
    if _run_flags.get("use_spawn_method") is not None:
        return _run_flags["use_spawn_method"]
    else:
        return sys.version_info[0] >= 3
