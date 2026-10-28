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

"""
Smoke test: verify that all custom widget classes have the required
module/version traits for ipywidgets >=7.x frontend resolution.
"""

import contextlib
import importlib

import pytest

from ..tests.base import ui_case

TIMEOUT = 10


def _get_trait_names(cls):
    """Get all trait names that contain 'module' from a widget class."""
    if cls is None:
        return []
    return [t for t in dir(cls) if "module" in t and not t.startswith("__")]


@ui_case
def test_instances_progress_traits():
    # InstancesProgress is only defined inside an IPython kernel
    # (guarded by in_ipython_frontend()). The module may have been imported
    # before the kernel shell was ready, so we reload it inside the kernel.
    from ..tests.base import setup_kernel

    check_code = (
        "import importlib, odps.ui.progress\n"
        "importlib.reload(odps.ui.progress)\n"
        "from odps.ui.progress import InstancesProgress\n"
        "assert InstancesProgress is not None, 'InstancesProgress is None'\n"
        "traits = [t for t in dir(InstancesProgress) if 'module' in t and not t.startswith('__')]\n"
        "assert '_view_module' in traits, f'Missing _view_module: {traits}'\n"
        "assert '_view_module_version' in traits, f'Missing _view_module_version: {traits}'\n"
        "assert '_model_module' in traits, f'Missing _model_module: {traits}'\n"
        "assert '_model_module_version' in traits, f'Missing _model_module_version: {traits}'\n"
    )
    with setup_kernel() as client:
        client.execute(check_code)
        shell_msg = client.get_shell_msg(timeout=TIMEOUT)
        assert (
            shell_msg["content"]["status"] == "ok"
        ), f"Kernel assertion failed: {shell_msg['content']}"


@ui_case
def test_df_view_widget_traits(monkeypatch):
    try:
        import ipywidgets
    except ImportError:
        pytest.skip("DFViewWidget not available (no ipywidgets)")

    # odps.df.ui snapshots the widgets at import time and another test may have
    # imported it first, so patch the accessor and reload.
    monkeypatch.setattr("odps.console.get_widgets", lambda: ipywidgets)

    import odps.df.ui as df_ui

    DFViewWidget = importlib.reload(df_ui).DFViewWidget
    assert DFViewWidget is not None, "DFViewWidget is None despite ipywidgets"
    traits = _get_trait_names(DFViewWidget)
    assert "_view_module" in traits
    assert "_view_module_version" in traits
    assert "_model_module" in traits
    assert "_model_module_version" in traits
    # The request contract the front end depends on: one record trait per
    # chart plus the error slot the view renders.
    for name in (
        "table_records",
        "bar_chart_records",
        "pie_chart_records",
        "line_chart_records",
        "scatter_chart_records",
        "last_error",
        "closed",
    ):
        assert name in dir(DFViewWidget), f"Missing trait {name}"


@contextlib.contextmanager
def _notebook_widget(value):
    """Temporarily set display.notebook_widget (patch.object cannot restore it)."""
    from odps.config import options

    old = options.display.notebook_widget
    options.display.notebook_widget = value
    try:
        yield
    finally:
        options.display.notebook_widget = old


def _patch_backend(monkeypatch, backend):
    monkeypatch.setattr("odps.console.get_notebook_backend", lambda b=backend: b)


def test_is_widgets_available(monkeypatch):
    """Widgets need ipywidgets >= 7.0 and follow display.notebook_widget."""
    from odps.console import is_widgets_available

    try:
        import ipywidgets
    except ImportError:
        pytest.skip("ipywidgets is not installed")

    # `get_widgets()` is None unless IPython was already imported; patch it so
    # the option / backend / version logic below is exercised.
    monkeypatch.setattr("odps.console.get_widgets", lambda: ipywidgets)
    monkeypatch.setattr("ipywidgets.__version__", "8.1.9")

    # None (the default) auto-detects: enabled except on DSW / VSCode.
    with _notebook_widget(None):
        for backend in ("JupyterLab", "JupyterNotebook"):
            _patch_backend(monkeypatch, backend)
            assert is_widgets_available() is True
        for backend in ("DSW", "VSCode"):
            _patch_backend(monkeypatch, backend)
            assert is_widgets_available() is False

    # The user option wins over the detected backend and the version gate.
    with _notebook_widget(True):
        _patch_backend(monkeypatch, "VSCode")
        assert is_widgets_available() is True
        monkeypatch.setattr("ipywidgets.__version__", "6.0.0")
        assert is_widgets_available() is False
    with _notebook_widget(False):
        monkeypatch.setattr("ipywidgets.__version__", "8.1.9")
        assert is_widgets_available() is False


def test_get_notebook_backend(monkeypatch):
    """Env-based backend detection, including VS Code remote (GH#258)."""
    from odps import console

    for env_name in list(console.os.environ):
        if env_name.lower().startswith(("vscode_", "dsw_")):
            monkeypatch.delenv(env_name)

    def detect(**env):
        monkeypatch.setattr(console, "_backend_name", None)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return console.get_notebook_backend()

    assert detect() == "JupyterNotebook"
    # Local VS Code sets VSCODE_PID; VS Code remote only exports others.
    assert detect(VSCODE_PID="12345") == "VSCode"
    assert detect(VSCODE_IPC_HOOK_CLI="/tmp/vscode-remote-ipc.sock") == "VSCode"
    # DSW wins over the VSCODE_* variables left in the environment.
    assert detect(DSW_SERVICE_VERSION="1.0") == "DSW"


def test_get_notebook_backend_cache_scope(monkeypatch):
    """The backend is only cached once an IPython front end exists."""
    from odps import console

    for env_name in list(console.os.environ):
        if env_name.lower().startswith(("vscode_", "dsw_")):
            monkeypatch.delenv(env_name)
    monkeypatch.setattr(console, "_backend_name", None)

    # Outside a front end the process may start a kernel later on, so the
    # detection must not freeze the current answer.
    monkeypatch.setattr(console, "in_ipython_frontend", lambda: False)
    assert console.get_notebook_backend() == "JupyterNotebook"
    assert console._backend_name is None

    monkeypatch.setattr(console, "in_ipython_frontend", lambda: True)
    assert console.get_notebook_backend() == "JupyterNotebook"
    assert console._backend_name == "JupyterNotebook"
