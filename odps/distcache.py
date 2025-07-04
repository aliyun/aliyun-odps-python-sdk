# Copyright 1999-2025 Alibaba Group Holding Ltd.
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


"""
This module provides stubs for resource references in SQL UDFs.
Actual implementations are provided by MaxCompute SQL Engine
which are NOT available within PyODPS code base. DO NOT USE
these methods in your local environments or environments other
than MaxCompute SQL UDFs as they will NEVER be functional.
"""


def get_cache_file(name):
    return open(name)


def get_cache_table(name):
    pass


def get_cache_archive(name, relative_path="."):
    pass


def get_cache_tabledesc(name):
    pass


def get_cache_tableinfo(name):
    pass
