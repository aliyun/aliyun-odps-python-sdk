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

import sys
from pathlib import Path

from ._version import __version__

__all__ = ["ODPS", "DataFrame", "options"]

from .config import options
from .core import ODPS
from .df import DataFrame, NullScalar, RandomScalar, Scalar
from .inter import enter, list_rooms, setup, teardown
from .lib import Monthdelta
from .utils import show_versions
from .utils import write_log as log

try:
    from .ipython import load_ipython_extension
except ImportError:
    pass
# The dialect is discovered through the "sqlalchemy.dialects" entry points
# declared in pyproject.toml. Register it here as well when SQLAlchemy has
# already been imported, which covers source trees that carry no installed
# package metadata. `import odps` itself never imports SQLAlchemy; in a source
# tree where odps is imported first, import `odps.sqlalchemy_odps` (it
# self-registers) before create_engine() is called.
if sys.modules.get("sqlalchemy") is not None:
    try:
        from sqlalchemy.dialects import registry

        registry.register("odps", "odps.sqlalchemy_odps", "ODPSDialect")
    except ImportError:
        pass


def _jupyter_labextension_paths():
    """Return the prebuilt JupyterLab labextension directories.

    JupyterLab discovers prebuilt extensions by calling this function.
    Each entry maps a source directory (shipped in the wheel as
    ``share/jupyter/labextensions/<dest>``) to a labextension name.

    For editable/development installs the built labextension lives at
    ``odps/static/ui/labextension``; for wheel installs it lands at
    ``share/jupyter/labextensions/pyodps``. We resolve the former first,
    then fall back to the latter (matching the pattern in jupyterlab_widgets).
    """
    here = Path(__file__).parent.resolve()
    # Development: built output lives alongside the source tree
    src_prefix = here / "static" / "ui" / "labextension"
    if not src_prefix.exists():
        # Wheel install: data_files landed in sys.prefix
        src_prefix = Path(sys.prefix) / "share" / "jupyter" / "labextensions" / "pyodps"
    return [{"src": str(src_prefix), "dest": "pyodps"}]


def _jupyter_nbextension_paths():
    """Return the classic Notebook nbextension directories.

    Notebook discovers nbextensions by calling this function.
    """
    here = Path(__file__).parent.resolve()
    src_prefix = here / "static" / "ui" / "nbextension"
    if not src_prefix.exists():
        src_prefix = Path(sys.prefix) / "share" / "jupyter" / "nbextensions" / "pyodps"
    return [
        {
            "section": "notebook",
            "src": str(src_prefix),
            "dest": "pyodps",
            "require": "pyodps/main",
        }
    ]


def install_plugins():
    try:
        from .ml import install_mixin, install_plugin
    except (ImportError, SyntaxError):
        pass


install_plugins()
del install_plugins
