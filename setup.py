#!/usr/bin/env python
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

# Parts of this file were taken from the pandas project
# (https://github.com/pandas-dev/pandas), which is permitted for use under
# the BSD 3-Clause License

import json
import os
import platform
import shutil
import subprocess
import sys

from setuptools import Command, Extension, setup
from setuptools.command.install import install

try:
    from setuptools.command.build import build as build_command
except ImportError:  # setuptools < 62
    from distutils.command.build import build as build_command

try:
    from setuptools.command.develop import develop as develop_command
except ImportError:  # setuptools without develop (very old)
    develop_command = None

try:
    from setuptools.command.editable_wheel import editable_wheel
except ImportError:  # setuptools < 64 has no PEP 660 support
    editable_wheel = None

try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version

try:
    from sysconfig import get_config_var
except ImportError:
    from distutils.sysconfig import get_config_var

# From https://github.com/pandas-dev/pandas/pull/24274:
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distutils behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if sys.platform == "darwin":
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        current_system = Version(platform.mac_ver()[0])
        python_target = Version(get_config_var("MACOSX_DEPLOYMENT_TARGET"))
        if python_target < Version("10.9") and current_system >= Version("10.9"):
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"

repo_root = os.path.dirname(os.path.abspath(__file__))

version_ns = {}
with open(os.path.join(repo_root, "odps", "_version.py")) as f:
    exec(f.read(), version_ns)

extra_install_cmds = []


# http://stackoverflow.com/questions/12683834/how-to-copy-directory-recursively-in-python-and-overwrite-all
def recursive_overwrite(src, dest, filter_func=None):
    destinations = []
    filter_func = filter_func or (lambda s: True)
    if os.path.isdir(src):
        if not os.path.isdir(dest):
            os.makedirs(dest)
        files = os.listdir(src)
        for f in files:
            if not filter_func(f):
                continue
            destinations.extend(
                recursive_overwrite(os.path.join(src, f), os.path.join(dest, f))
            )
    else:
        shutil.copyfile(src, dest)
        destinations.append(dest)
    return destinations


class CustomInstall(install):
    def run(self):
        global extra_install_cmds
        install.run(self)
        [self.run_command(cmd) for cmd in extra_install_cmds]


if sys.version_info[:2] < (3, 7):
    raise Exception(
        "Since PyODPS 0.13.0, supports for Python 3.6 or below is dropped, "
        "please install an older version."
    )
try:
    try:
        from importlib.metadata import distributions
    except ImportError:
        from importlib_metadata import distributions

    for pk in distributions():
        pk_name = pk.metadata["Name"]
        if pk_name.lower() in ("odps", "distribute"):
            raise Exception(
                f"Package `{pk_name}` collides with PyODPS. Please uninstall it before installing PyODPS."
            )
except (ImportError, KeyError):
    pass

try:
    from jupyter_core.paths import jupyter_data_dir

    has_jupyter = True
except ImportError:
    has_jupyter = False

if len(sys.argv) > 1 and sys.argv[1] == "clean":
    build_cmd = sys.argv[1]
else:
    build_cmd = None

setup_options = dict(
    cmdclass={"install": CustomInstall},
    ext_modules=[],
    version=version_ns["__version__"],
)

PYPY = platform.python_implementation().lower() == "pypy"

if build_cmd != "clean" and not PYPY:  # skip cython in pypy
    try:
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext

        # detect if cython works
        if sys.platform == "win32":
            import cython

            cython.inline("return a + b", a=1, b=1)

        cythonize_kw = dict(language_level=sys.version_info[0])
        extension_kw = dict(language="c++", include_dirs=[])

        # Detect free-threaded Python and set appropriate flags
        is_free_threaded = False
        if sys.version_info >= (3, 13):
            # Check for free-threaded Python 3.13+
            is_free_threaded = not getattr(sys, "_is_gil_enabled", lambda: True)()

        # Set up define macros
        define_macros = list(extension_kw.get("define_macros", []))
        if is_free_threaded:
            define_macros.append(("Py_GIL_DISABLED", "1"))
        extension_kw["define_macros"] = define_macros

        # Set up Cython compiler directives
        compiler_directives = cythonize_kw.get("compiler_directives", {})
        if is_free_threaded:
            compiler_directives["freethreading_compatible"] = True
        cythonize_kw["compiler_directives"] = compiler_directives

        if "MSC" in sys.version:
            extra_compile_args = ["/Ot", "/I" + os.path.join(repo_root, "misc")]
            extension_kw["extra_compile_args"] = extra_compile_args
        else:
            extra_compile_args = ["-O3"]
            extension_kw["extra_compile_args"] = extra_compile_args

        if os.environ.get("CYTHON_TRACE"):
            extension_kw["define_macros"].extend(
                [
                    ("CYTHON_TRACE_NOGIL", "1"),
                    ("CYTHON_TRACE", "1"),
                ]
            )
            cythonize_kw["compiler_directives"]["linetrace"] = True

        extensions = [
            Extension("odps.src.types_c", ["odps/src/types_c.pyx"], **extension_kw),
            Extension("odps.src.crc32c_c", ["odps/src/crc32c_c.pyx"], **extension_kw),
            Extension("odps.src.utils_c", ["odps/src/utils_c.pyx"], **extension_kw),
            Extension(
                "odps.tunnel.pb.encoder_c",
                ["odps/tunnel/pb/encoder_c.pyx"],
                **extension_kw,
            ),
            Extension(
                "odps.tunnel.pb.decoder_c",
                ["odps/tunnel/pb/decoder_c.pyx"],
                **extension_kw,
            ),
            Extension(
                "odps.tunnel.io.writer_c",
                ["odps/tunnel/io/writer_c.pyx"],
                **extension_kw,
            ),
            Extension(
                "odps.tunnel.io.reader_c",
                ["odps/tunnel/io/reader_c.pyx"],
                **extension_kw,
            ),
            Extension(
                "odps.tunnel.checksum_c", ["odps/tunnel/checksum_c.pyx"], **extension_kw
            ),
            Extension(
                "odps.tunnel.hasher_c", ["odps/tunnel/hasher_c.pyx"], **extension_kw
            ),
        ]

        setup_options["cmdclass"].update({"build_ext": build_ext})
        force_recompile = bool(int(os.getenv("CYTHON_FORCE_RECOMPILE", "0")))
        setup_options["ext_modules"] = cythonize(
            extensions, force=force_recompile, **cythonize_kw
        )
    except Exception:
        pass

# --- Jupyter widget extension data files ---
# The webpack bundles are built from odps/static/ui/src and shipped as Jupyter
# data files: pip installs them into
# sys.prefix/share/jupyter/{labextensions,nbextensions}/pyodps/.
# JupyterLab discovers them via _jupyter_labextension_paths() in odps/__init__.py;
# classic Notebook discovers them via _jupyter_nbextension_paths().
#
# They are deliberately absent from git and from the sdist (see .gitignore,
# MANIFEST.in and pyproject.toml). The `build` command below produces them when
# npm is available and attaches them to the distribution, so the wheel carries
# the bundles and an install from the sdist builds them on the fly.


def _js_ui_dir():
    return os.path.join(repo_root, "odps", "static", "ui")


def _labextension_built():
    """True when the JupyterLab bundle is complete, not merely present.

    The generated manifest names the remote entry the front end loads, so that
    exact file has to exist: a glob alone would also accept a stale entry left
    behind by an earlier, different build.
    """
    lab_dir = os.path.join(_js_ui_dir(), "labextension")
    if not os.path.isfile(os.path.join(lab_dir, "install.json")):
        return False

    try:
        with open(os.path.join(lab_dir, "package.json"), encoding="utf-8") as fp:
            manifest = json.load(fp)
    except (OSError, ValueError):
        return False

    load = manifest.get("jupyterlab", {}).get("_build", {}).get("load")
    if not load:
        return False
    return os.path.isfile(os.path.join(lab_dir, *load.split("/")))


def _nbextension_built():
    """True when the classic Notebook bundle is complete, not merely present."""
    return os.path.isfile(os.path.join(_js_ui_dir(), "nbextension", "main.js"))


def _js_bundles_built():
    """True when both webpack bundles are complete in the tree.

    Testing the directories alone is not enough: an interrupted `npm run build`
    leaves them behind, possibly empty, which would suppress the rebuild and
    ship a wheel whose widget extensions cannot load.
    """
    return _labextension_built() and _nbextension_built()


def _npm_argv(*args):
    """Return an npm argv that both POSIX and Windows can launch.

    Windows ships npm as ``npm.cmd`` plus a POSIX shell script named ``npm``.
    CreateProcess only appends ``.exe`` when searching PATH, so invoking the
    bare name fails with WinError 2 even when npm is installed; ``shutil.which``
    honours PATHEXT and yields the launcher's full path.
    """
    return [shutil.which("npm") or "npm", *args]


def _build_js_bundles(registry=None):
    """Build the webpack bundles with npm. Returns False when npm is missing."""
    if not shutil.which("npm"):
        return False

    ui_dir = _js_ui_dir()
    if not os.path.isdir(ui_dir):
        return False

    cwd = os.getcwd()
    os.chdir(ui_dir)
    try:
        cmd = _npm_argv("install")
        if registry:
            cmd += ["--registry=" + registry]
        print("executing " + " ".join(cmd))
        subprocess.check_call(cmd)

        print("building JS bundles (nbextension + labextension)")
        subprocess.check_call(_npm_argv("run", "build"))
    finally:
        os.chdir(cwd)
    return True


def _collect_js_data_files():
    """Collect prebuilt labextension/nbextension files as Jupyter data_files.

    Returns (dest_dir, [relative_src_paths]) pairs. setuptools requires
    /-separated paths relative to the setup.py directory.
    """
    data_files = []
    ui_dir = os.path.join("odps", "static", "ui")

    # Labextension: share/jupyter/labextensions/pyodps/
    # Only complete bundles are collected, so that an interrupted build is
    # reported as missing bundles instead of being shipped half-written.
    lab_src = os.path.join(ui_dir, "labextension")
    if _labextension_built():
        for root, _dirs, files in os.walk(lab_src):
            for fname in files:
                rel_src = os.path.relpath(os.path.join(root, fname), ".")
                rel_to_lab = os.path.relpath(rel_src, lab_src)
                dest = os.path.join(
                    "share", "jupyter", "labextensions", "pyodps", rel_to_lab
                )
                data_files.append((os.path.dirname(dest), [rel_src]))

    # Nbextension: share/jupyter/nbextensions/pyodps/
    nb_src = os.path.join(ui_dir, "nbextension")
    if _nbextension_built():
        for root, _dirs, files in os.walk(nb_src):
            for fname in files:
                rel_src = os.path.relpath(os.path.join(root, fname), ".")
                rel_to_nb = os.path.relpath(rel_src, nb_src)
                dest = os.path.join(
                    "share", "jupyter", "nbextensions", "pyodps", rel_to_nb
                )
                data_files.append((os.path.dirname(dest), [rel_src]))

        # Copying the nbextension is not enough: classic Notebook only loads
        # it when the entry point is enabled. Ship the nbconfig fragment so a
        # plain pip install activates `pyodps/main` without a manual
        # `jupyter nbextension enable --py pyodps`.
        nbconfig_src = os.path.join(ui_dir, "nbconfig", "notebook.d", "pyodps.json")
        if os.path.isfile(nbconfig_src):
            data_files.append(
                (
                    os.path.join("etc", "jupyter", "nbconfig", "notebook.d"),
                    [nbconfig_src],
                )
            )

    return data_files


def _report_js_missing(reason, required):
    """Fail when the bundles are required, warn otherwise.

    The bundles are a user-experience enhancement, not a requirement, so a
    distribution without them is still installable and usable unless
    PYODPS_REQUIRE_JS=1 says otherwise.
    """
    if required:
        raise RuntimeError(
            f"{reason}. PYODPS_REQUIRE_JS=1 requires the Jupyter widget JS "
            "bundles; install npm and run `python setup.py build_js`, or "
            "drop PYODPS_REQUIRE_JS to build them best-effort."
        )
    print(
        f"WARNING: {reason}. The PyODPS Jupyter widgets will not be available;"
        " they are an optional user-experience enhancement and the package "
        "installs and works without them. Run `python setup.py build_js` (with"
        " npm) to bundle them."
    )


def _attach_js_data_files(distribution):
    """Attach the built bundles to ``distribution.data_files`` once.

    More than one command can ask for the bundles in a single invocation, so
    skip entries that are already present instead of duplicating them.
    """
    existing = {
        (dest, src)
        for dest, sources in (distribution.data_files or [])
        for src in sources
    }
    new_entries = []
    for dest, sources in _collect_js_data_files():
        sources = [src for src in sources if (dest, src) not in existing]
        if sources:
            new_entries.append((dest, sources))
    if new_entries:
        distribution.data_files = list(distribution.data_files or []) + new_entries


def _ensure_js_bundles(distribution, attach_data_files=True):
    """Build missing bundles (best effort) and optionally attach them.

    Shared by the regular build, ``pip install -e .`` and the legacy
    ``setup.py develop`` path. Editable installs are why this cannot live only
    in ``BuildWithJS.run``: setuptools' PEP 660 command does not run the
    ``build`` command, so an editable install would otherwise leave JupyterLab
    without the widget view bundle. ``develop`` installs no data files, so it
    passes ``attach_data_files=False``.
    """
    skip = os.environ.get("PYODPS_SKIP_JS") == "1"
    rebuild = os.environ.get("PYODPS_REBUILD_JS") == "1"
    require = os.environ.get("PYODPS_REQUIRE_JS", "0") == "1"

    if not skip and (rebuild or not _js_bundles_built()):
        if not shutil.which("npm"):
            _report_js_missing(
                "npm was not found, so the Jupyter widget extensions were not built",
                require,
            )
        else:
            # Bundling is best-effort unless required: a failed npm build
            # must not make the distribution uninstallable, so warn and
            # continue instead of aborting.
            try:
                _build_js_bundles(os.environ.get("PYODPS_NPM_REGISTRY"))
            except Exception as exc:
                _report_js_missing(
                    f"building the Jupyter widget extensions failed: {exc}",
                    require,
                )

    if attach_data_files:
        _attach_js_data_files(distribution)
    # Warn about whichever bundle is missing, so a partially built tree is
    # never packaged silently.
    if not _labextension_built():
        _report_js_missing("the JupyterLab widget extension bundle is missing", require)
    if not _nbextension_built():
        _report_js_missing(
            "the classic Notebook widget extension bundle is missing", require
        )


class BuildJS(Command):
    """Build the Jupyter widget JS bundles (labextension + nbextension).

    Run with: python setup.py build_js
    Requires npm and network access to the npm registry.
    """

    description = "build JavaScript widget extension bundles"
    user_options = [("registry=", None, "npm registry")]

    def initialize_options(self):
        self.registry = None

    def finalize_options(self):
        pass

    def run(self):
        if not _build_js_bundles(getattr(self, "registry", None)):
            raise Exception("You need to install npm before building the scripts.")


class BuildWithJS(build_command):
    """Produce the JS bundles and attach them to the distribution.

    The data_files are collected here rather than at import time because the
    bundles only exist once this command has run. That is what lets an install
    from the sdist -- which ships sources only -- still end up with a wheel
    that registers the widgets with Jupyter.

    Environment variables:
      PYODPS_REBUILD_JS=1    rebuild even when bundles are already present
      PYODPS_SKIP_JS=1       never invoke npm
      PYODPS_REQUIRE_JS=1    fail the build when npm is missing or the JS
                             bundles cannot be built
      PYODPS_NPM_REGISTRY    registry passed to ``npm install``
    """

    def run(self):
        _ensure_js_bundles(self.distribution)
        super().run()


if editable_wheel is not None:

    class EditableWithJS(editable_wheel):
        """Install the widget bundles together with an editable install.

        PEP 660 says non-Python files are not live-editable, but setuptools
        also skips ``install_data`` for editable wheels: ``_run_install`` looks
        for a ``has_data`` hook that does not exist (the real method is
        ``has_data_files``), so the data_files never reach the wheel. Run the
        command ourselves after the build commands have configured the install
        command; its output directory is the unpacked wheel's ``.data/data``
        tree, which pip copies into ``sys.prefix``.
        """

        def _run_build_commands(self, *args, **kwargs):
            _ensure_js_bundles(self.distribution)
            result = super()._run_build_commands(*args, **kwargs)
            if self.distribution.data_files:
                self.run_command("install_data")
            return result


if develop_command is not None:

    class DevelopWithJS(develop_command):
        """Pre-build the bundles for the legacy ``setup.py develop`` path.

        ``develop`` installs no ``data_files``, but the extension paths prefer
        the source tree, so building the bundles there is still needed.
        """

        def run(self):
            _ensure_js_bundles(self.distribution, attach_data_files=False)
            super().run()


setup_options["cmdclass"].update({"build_js": BuildJS, "build": BuildWithJS})
if editable_wheel is not None:
    setup_options["cmdclass"]["editable_wheel"] = EditableWithJS
if develop_command is not None:
    setup_options["cmdclass"]["develop"] = DevelopWithJS

setup(**setup_options)

if build_cmd == "clean":
    for root, dirs, files in os.walk(os.path.normpath("odps/")):
        pyx_files = set()
        c_file_pairs = []
        if "__pycache__" in dirs:
            full_path = os.path.join(root, "__pycache__")
            print(f"removing '{full_path}'")
            shutil.rmtree(full_path)
        for f in files:
            fn, ext = os.path.splitext(f)
            # delete compiled binaries
            if ext.lower() in (".pyd", ".so", ".pyc"):
                full_path = os.path.join(root, f)
                print(f"removing '{full_path}'")
                os.unlink(full_path)
            elif ext.lower() == ".pyx":
                pyx_files.add(fn)
            elif ext.lower() in (".c", ".cpp", ".cc"):
                c_file_pairs.append((fn, f))

        # remove cython-generated files
        for cfn, cf in c_file_pairs:
            if cfn in pyx_files:
                full_path = os.path.join(root, cf)
                print(f"removing '{full_path}'")
                os.unlink(full_path)
