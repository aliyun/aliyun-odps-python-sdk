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

import os
import platform
import shutil
import sys

from setuptools import Command, Extension, setup
from setuptools.command.install import install

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
    except:
        pass

if build_cmd != "clean" and has_jupyter:

    class InstallJS(Command):
        description = "install JavaScript extensions"
        user_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            src_dir = os.path.join(repo_root, "odps", "static", "ui", "target")
            if not os.path.exists(src_dir):
                return

            dest_dir = os.path.join(jupyter_data_dir(), "nbextensions", "pyodps")
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            recursive_overwrite(src_dir, dest_dir)

            try:
                from notebook.nbextensions import enable_nbextension
            except ImportError:
                return
            enable_nbextension("notebook", "pyodps/main")

    class BuildJS(Command):
        description = "build JavaScript files"
        user_options = [("registry=", None, "npm registry")]

        def initialize_options(self):
            self.registry = None

        def finalize_options(self):
            pass

        def run(self):
            if not shutil.which("npm"):
                raise Exception("You need to install npm before building the scripts.")

            cwd = os.getcwd()

            ui_path = os.path.join(os.path.abspath(os.getcwd()), "odps", "static", "ui")
            if not os.path.exists(ui_path):
                return

            os.chdir(ui_path)
            cmd = "npm install"
            if getattr(self, "registry", None):
                cmd += " --registry=" + self.registry
            print("executing " + cmd)
            ret = os.system(cmd)
            ret >>= 8
            if ret != 0:
                print(f"{cmd} exited with error: {ret}")

            print("executing grunt")
            ret = os.system("npm run grunt")
            ret >>= 8
            if ret != 0:
                print(f"grunt exited with error: {ret}")

            os.chdir(cwd)

    setup_options["cmdclass"].update({"install_js": InstallJS, "build_js": BuildJS})
    extra_install_cmds.append("install_js")

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
