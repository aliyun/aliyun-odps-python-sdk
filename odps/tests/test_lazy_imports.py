#!/usr/bin/env python
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

"""`import odps` must not import optional dependencies.

SQLAlchemy, IPython and the widget stack (ipywidgets/traitlets) are optional.
The checks run in a fresh interpreter because imports are process-global.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

OPTIONAL_MODULES = ("sqlalchemy", "IPython", "ipywidgets", "traitlets")

# Run subprocesses against the tree under test, not against an installed pyodps.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _is_installed(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def _run_python(code):
    # cwd alone is not enough (PYTHONSAFEPATH, or an installed pyodps earlier
    # on sys.path), so put the checkout in front explicitly as well.
    prelude = "import sys\nsys.path.insert(0, %r)\n" % _REPO_ROOT
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        cwd=_REPO_ROOT,
        timeout=60,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


@pytest.mark.parametrize("module_name", OPTIONAL_MODULES)
def test_bare_import_does_not_import_optional_dependency(module_name):
    if not _is_installed(module_name):
        pytest.skip("%s is not installed" % module_name)

    proc = _run_python(
        """
        import sys

        import odps

        print("lazy" if %r not in sys.modules else "eager")
        """
        % module_name
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "lazy", "`import odps` imported %s" % module_name


def test_has_sqlalchemy_reports_installation_without_importing():
    if not _is_installed("sqlalchemy"):
        pytest.skip("sqlalchemy is not installed")

    proc = _run_python(
        """
        import sys

        import odps

        print(odps.df.core.has_sqlalchemy, "sqlalchemy" in sys.modules)
        """
    )

    assert proc.returncode == 0, proc.stderr
    # findable is advisory; it does not promise the import would succeed
    assert proc.stdout.split() == ["True", "False"]


def test_dataframe_accepts_sqlalchemy_table():
    if not _is_installed("sqlalchemy"):
        pytest.skip("sqlalchemy is not installed")

    proc = _run_python(
        """
        import odps

        import sqlalchemy
        from sqlalchemy import Column, Integer, MetaData, String, Table

        table = Table(
            "demo", MetaData(), Column("a", Integer), Column("b", String)
        )
        schema = odps.DataFrame(table).schema
        print(schema["a"].type.name, schema["b"].type.name)
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["int32", "string"]


def test_dialect_registers_without_entry_points():
    if not _is_installed("sqlalchemy"):
        pytest.skip("sqlalchemy is not installed")

    proc = _run_python(
        """
        import odps  # first, so odps/__init__.py cannot register the dialect

        from sqlalchemy.dialects import registry
        from sqlalchemy.util import compat

        # Simulate a source tree with no installed package metadata, where
        # SQLAlchemy finds no "sqlalchemy.dialects" entry point.
        compat.importlib_metadata_get = lambda group: []
        registry.auto_fn = None
        assert not registry.impls, registry.impls

        import odps.sqlalchemy_odps
        from odps.sqlalchemy_odps import ODPSDialect

        print(registry.load("odps") is ODPSDialect)
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True"


def test_subprocess_imports_tree_under_test():
    proc = _run_python(
        """
        import odps

        print(odps.__file__)
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith(_REPO_ROOT)


def test_accessors_are_empty_without_ipython():
    if not _is_installed("IPython"):
        pytest.skip("IPython is not installed")

    proc = _run_python(
        """
        import sys

        import odps

        print(
            odps.console.get_widgets() is None,
            odps.console.get_ipython_display() is None,
            "IPython" in sys.modules,
        )
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["True", "True", "False"]


def test_bare_import_keeps_public_api():
    proc = _run_python(
        """
        import odps

        assert odps.ODPS is not None
        assert odps.options is not None

        from odps.df import DataFrame

        print(DataFrame.__name__)
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "DataFrame"
