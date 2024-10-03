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

import sys

from ._version import __version__

__all__ = ["ODPS", "DataFrame", "options"]

if sys.version_info[0] == 2 and sys.version_info[:2] < (2, 7):
    raise Exception("pyodps supports python 2.7+ (including python 3+).")

from .config import options
from .core import ODPS
from .df import DataFrame, NullScalar, RandomScalar, Scalar
from .inter import enter, list_rooms, setup, teardown
from .utils import show_versions
from .utils import write_log as log

try:
    from .ipython import load_ipython_extension
except ImportError:
    pass
try:
    from sqlalchemy.dialects import registry

    registry.register("odps", "odps.sqlalchemy_odps", "ODPSDialect")
except ImportError:
    pass


def install_plugins():
    try:
        from .ml import install_mixin, install_plugin
    except (ImportError, SyntaxError):
        pass


install_plugins()
del install_plugins
