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
import glob
from ....compat import urlparse


class LocalFileSystem(object):
    sep = os.sep

    def __init__(self, **__):
        self._work_dir = os.getcwd()

    def _normalize_path(self, path):
        parse_result = urlparse(path)
        path = parse_result.path
        if parse_result.netloc:
            path = os.path.join(parse_result.netloc, parse_result.path)

        if not os.path.isabs(path):
            return os.path.join(self._work_dir, path)
        return os.path.normpath(path)

    def glob(self, path):
        for p in glob.glob(self._normalize_path(path)):
            yield "file://" + p

    def open(self, path, mode="rb"):
        return open(self._normalize_path(path), mode)


from . import core

core._file_systems["file"] = LocalFileSystem
