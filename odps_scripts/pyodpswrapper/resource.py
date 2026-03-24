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

import atexit
import functools
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

RESOURCE_SIZE_LIMIT = 100 * 1024**2

_tmp_resource_dir = None
_total_resource_size = 0

logger = logging.getLogger(__name__)


def load_packages_in_subprocess(res_str):
    global _tmp_resource_dir

    res_str = res_str.split("#", 1)[0].strip() if res_str else res_str
    # skip when already in a resource-load process or resource already loaded
    if (
        not res_str
        or "--load-resource" in sys.argv
        or "PYODPS_RESOURCE_LOADED" in os.environ
    ):
        return

    # remove code-related args ro replace with resource args
    res_args = list(sys.argv)
    code_idx = None
    if "--code-file" in res_args:
        code_idx = res_args.index("--code-file")
    elif "--code" in res_args:
        code_idx = res_args.index("--code")
    if code_idx is not None:
        res_args = res_args[:code_idx] + res_args[code_idx + 2 :]

    main_module = __name__.rsplit(".", 1)[0] + ".__main__"
    res_args = (
        [sys.executable, "-m", main_module]
        + res_args[1:]
        + ["--code", "", "--load-resource", res_str]
    )
    proc = subprocess.Popen(res_args, stdout=subprocess.PIPE)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Failed to load resource " + res_str)

    ret_data = json.loads(proc.stdout.read().strip().splitlines()[-1])
    if ret_data.get("tempdir"):
        _tmp_resource_dir = ret_data["tempdir"]
        atexit.register(
            functools.partial(shutil.rmtree, _tmp_resource_dir, ignore_errors=True)
        )
    sys.path = ret_data["paths"] + sys.path
    os.environ["PYTHONPATH"] = ":".join(sys.path)
    os.environ["PYODPS_RESOURCE_LOADED"] = "true"


def run_load_resource_command(args, odps_entry):
    ext_paths = []
    for res_name in args.load_resource.split(","):
        ext_paths.extend(load_resource_package(res_name.strip(), odps_entry))
    # need to export extracted dirs via stdout.
    print(json.dumps({"tempdir": _tmp_resource_dir, "paths": ext_paths}))
    return 0


def _check_resource_limit(increment_size):
    global _total_resource_size
    if _total_resource_size + increment_size > RESOURCE_SIZE_LIMIT:
        raise SystemError("Not allowed to load resource packages larger than 100MB")
    _total_resource_size += increment_size


def load_resource_package(name, odps_entry, project=None, supersede=True):
    import tarfile
    import zipfile

    from odps.lib.importer import CompressImporter

    old_paths = set(sys.path)

    env = os.getenv("SKYNET_SYSTEM_ENV", "local")
    if env == "local":
        global _tmp_resource_dir
        if _tmp_resource_dir is None:
            _tmp_resource_dir = tempfile.mkdtemp(prefix="tmp-pyodps-pkg-")
            if "--load-resource" not in sys.argv:
                # only recycle temp dir when not spawned
                atexit.register(
                    functools.partial(
                        shutil.rmtree, _tmp_resource_dir, ignore_errors=True
                    )
                )
        pkg_dir = _tmp_resource_dir
    else:
        pkg_dir = os.getenv("TASK_EXEC_PATH")

    pkg_dir = pkg_dir.rstrip(os.path.sep)
    pkg_extract_path = os.path.join(pkg_dir, "pkg_extract")

    if not os.path.isfile(name):
        pkg_archive_fn = os.path.join(pkg_dir, name)
        res = odps_entry.get_resource(name, project=project)
        _check_resource_limit(res.size)
        with res.open("rb"):
            pkg_content = res.read()
        with open(pkg_archive_fn, "wb") as pkg_file:
            pkg_file.write(pkg_content)
    else:
        _check_resource_limit(os.path.getsize(name))
        pkg_archive_fn = name

    _, file_ext = os.path.splitext(name)
    CompressImporter._extract_path = pkg_extract_path

    is_wheel = False
    if file_ext == ".py":
        shutil.copyfile(pkg_archive_fn, pkg_extract_path)
        if pkg_extract_path not in sys.path:
            sys.path.append(pkg_extract_path)
    elif file_ext in {".zip", ".whl", ".egg"}:
        is_wheel = file_ext == ".whl"
        sys.meta_path.append(
            CompressImporter(
                zipfile.ZipFile(pkg_archive_fn, mode="r"),
                extract_all=True,
                supersede=supersede,
            )
        )
    elif name.endswith(".tar.gz"):
        sys.meta_path.append(
            CompressImporter(
                tarfile.open(pkg_archive_fn, mode="r:gz"),
                extract_all=True,
                supersede=supersede,
            )
        )
        is_wheel = True
        for path in glob.glob(os.path.join(pkg_extract_path, "*")):
            if not os.path.exists(os.path.join(path, "packages", ".pyodps-pack-meta")):
                is_wheel = False
                break
    else:
        raise NotImplementedError("Only supports wheels and pure source package.")
    shutil.rmtree(pkg_archive_fn, ignore_errors=True)

    if not is_wheel:
        logger.warning(
            "Package %s loaded from MaxCompute resource. "
            "Note that non-wheel packages might not function as expected." % name
        )
    return [p for p in sys.path if p not in old_paths]
