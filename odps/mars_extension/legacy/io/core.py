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

from ....compat import urlparse

_file_systems = dict()


def _get_scheme(path):
    result = urlparse(path)
    return result.scheme if result.scheme else "file"


def open(path, mode, **kwargs):
    scheme = _get_scheme(path)
    fs = _file_systems[scheme](**kwargs)
    return fs.open(path, mode)


def glob(path, **kwargs):
    scheme = _get_scheme(path)
    fs = _file_systems[scheme](**kwargs)
    return fs.glob(path)
