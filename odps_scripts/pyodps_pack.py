#!/usr/bin/env python
from __future__ import print_function
import argparse
import contextlib
import glob
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings
from collections import defaultdict

_DEFAULT_PYABI = "cp37-cp37m"
_MCPY27_PYABI = "cp27-cp27m"
_DWPY27_PYABI = "cp27-cp27mu"
_DEFAULT_DOCKER_IMAGE_X64 = "quay.io/pypa/manylinux2014_x86_64"
_DEFAULT_DOCKER_IMAGE_ARM64 = "quay.io/pypa/manylinux2014_aarch64"
_LEGACY_DOCKER_IMAGE_X64 = "quay.io/pypa/manylinux1_x86_64"
_DEFAULT_PACKAGE_SITE = "packages"
_DEFAULT_OUTPUT_FILE = "packages.tar.gz"
_REQUIREMENT_FILE_NAME = "requirements-extra.txt"
_INSTALL_REQ_FILE_NAME = "install-requires.txt"
_EXCLUDE_FILE_NAME = "excludes.txt"
_BEFORE_SCRIPT_FILE_NAME = "before-script.sh"
_VCS_FILE_NAME = "requirements-vcs.txt"
_PACK_SCRIPT_FILE_NAME = "pack.sh"

_SEGFAULT_ERR_CODE = 139

python_abi_env = os.getenv("PYABI")
cmd_before_build = os.getenv("BEFORE_BUILD") or ""
cmd_after_build = os.getenv("AFTER_BUILD") or ""
docker_image_env = os.getenv("DOCKER_IMAGE")
docker_path = os.getenv("DOCKER_PATH")
package_site = os.getenv("PACKAGE_SITE") or _DEFAULT_PACKAGE_SITE

_is_linux = sys.platform.lower().startswith("linux")
_is_windows = sys.platform.lower().startswith("win")
_vcs_prefixes = [prefix + "+" for prefix in "git hg svn bzr".split()]

logger = logging.getLogger(__name__)


script_template = r"""
#!/bin/bash
if [[ -n "$NON_DOCKER_MODE" ]]; then
  PYBIN="$(dirname "$PYEXECUTABLE")"
  PIP_PLATFORM_ARGS="--platform=$PYPLATFORM --abi=$PYABI --python-version=$PYVERSION --only-binary=:all:"
  MACHINE_TAG=$($PYBIN/python -c "import platform; print(platform.machine())")
  if [[ $(uname) != "Linux" || "$MACHINE_TAG" != "$TARGET_ARCH" ]]; then
    # does not allow compiling under non-linux or different arch
    echo "WARNING: target ($TARGET_ARCH-Linux) not matching host ($MACHINE_TAG-$(uname)), may encounter errors when compiling binary packages."
    export CC=/dev/null
  fi
else
  PYBIN="/opt/python/{python_abi_version}/bin"
  MACHINE_TAG=$($PYBIN/python -c "import platform; print(platform.machine())")
fi
export PATH="$PYBIN:$PATH"

SCRIPT_PATH="$PACK_ROOT/scripts"
BUILD_PKG_PATH="$PACK_ROOT/build"
WHEELS_PATH="$PACK_ROOT/wheels"
INSTALL_PATH="$PACK_ROOT/install"
TEMP_SCRIPT_PATH="$PACK_ROOT/tmp/scripts"
INSTALL_REQUIRE_PATH="$PACK_ROOT/install-req"
WHEELHOUSE_PATH="$PACK_ROOT/wheelhouse"

function handle_sigint() {{
  touch "$SCRIPT_PATH/.cancelled"
  exit 1
}}
trap handle_sigint SIGINT

if [[ "{debug}" == "true" ]]; then
  echo "Files under $SCRIPT_PATH:"
  ls "$SCRIPT_PATH"
  set -e -x
else
  set -e
fi
export PIP_ROOT_USER_ACTION=ignore
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ -z "{pypi_index}" ]]; then
  PYPI_EXTRA_ARG=""
else
  PYPI_EXTRA_ARG="-i {pypi_index}"
fi
if [[ "{prefer_binary}" == "true" ]]; then
  PYPI_EXTRA_ARG="$PYPI_EXTRA_ARG --prefer-binary"
fi
if [[ "{no_deps}" == "true" ]]; then
  PYPI_NO_DEPS_ARG="--no-deps"
fi
if [[ "{without_merge}" == "true" ]]; then
  WITHOUT_MERGE="true"
fi
if [[ -n "{pypi_trusted_hosts}" ]]; then
  for trusted_host in `echo "{pypi_trusted_hosts}"`; do
    PYPI_EXTRA_ARG="$PYPI_EXTRA_ARG --trusted-host $trusted_host"
  done
fi

if [[ -n "$NON_DOCKER_MODE" && "$PYEXECUTABLE" == *"venv"* ]]; then
  VENV_REQS="pip wheel setuptools"
  "$PYBIN/python" -m pip install -U --quiet $PYPI_EXTRA_ARG $VENV_REQS
fi

mkdir -p "$BUILD_PKG_PATH" "$INSTALL_PATH" "$WHEELS_PATH" "$INSTALL_REQUIRE_PATH" "$TEMP_SCRIPT_PATH"

if [[ -f "$SCRIPT_PATH/{_BEFORE_SCRIPT_FILE_NAME}" ]]; then
  echo "Running before build command..."
  /bin/bash "$SCRIPT_PATH/{_BEFORE_SCRIPT_FILE_NAME}"
  echo ""
fi

if [[ -f "$SCRIPT_PATH/{_INSTALL_REQ_FILE_NAME}" ]]; then
  echo "Installing build prerequisites..."
  "$PYBIN/python" -m pip install --target "$INSTALL_REQUIRE_PATH" \
    -r "$SCRIPT_PATH/{_INSTALL_REQ_FILE_NAME}" $PYPI_EXTRA_ARG
  export OLDPYTHONPATH="$PYTHONPATH"
  export PYTHONPATH="$INSTALL_REQUIRE_PATH:$PYTHONPATH"
  echo ""
fi

# build user-defined package
cd "$BUILD_PKG_PATH"
if [[ -f "$SCRIPT_PATH/requirements-user.txt" ]]; then
  cp "$SCRIPT_PATH/requirements-user.txt" "$TEMP_SCRIPT_PATH/requirements-user.txt"
fi
if [[ -f "$SCRIPT_PATH/{_REQUIREMENT_FILE_NAME}" ]]; then
  cp "$SCRIPT_PATH/{_REQUIREMENT_FILE_NAME}" "$TEMP_SCRIPT_PATH/requirements-extra.txt"
fi

function build_package_at_staging () {{
  mkdir -p "$WHEELS_PATH/staging"
  rm -rf "$WHEELS_PATH/staging/*"

  if [[ -n "$2" ]]; then
    "$PYBIN/python" -m pip wheel --no-deps --wheel-dir "$WHEELS_PATH/staging" $PYPI_EXTRA_ARG "$1" "$2"
  else
    "$PYBIN/python" -m pip wheel --no-deps --wheel-dir "$WHEELS_PATH/staging" $PYPI_EXTRA_ARG "$1"
  fi

  cd "$WHEELS_PATH/staging"
  for dep_wheel in $(ls *.whl); do
    WHEEL_NAMES="$WHEEL_NAMES $dep_wheel"
    dep_name="$(echo $dep_wheel | sed -E 's/-[0-9]/ /' | awk '{{print $1}}')"
    USER_PACK_NAMES="$USER_PACK_NAMES $dep_name "
    echo "$dep_name" >> "$TEMP_SCRIPT_PATH/requirements-user.txt"
    if [[ -z "$NON_DOCKER_MODE" || "$dep_wheel" == *"-none-"* ]]; then
      mv "$dep_wheel" ../
    fi
  done

  if [[ -z "$PYPI_NO_DEPS_ARG" ]]; then
    cd "$WHEELS_PATH"
    if [[ -z "$NON_DOCKER_MODE" ]]; then
      "$PYBIN/python" -m pip wheel --wheel-dir "$WHEELS_PATH" $PYPI_EXTRA_ARG $WHEEL_NAMES
    else
      "$PYBIN/python" -m pip wheel --wheel-dir "$WHEELS_PATH/staging" $PYPI_EXTRA_ARG $WHEEL_NAMES
      cd "$WHEELS_PATH/staging"
      for dep_wheel in $(ls *.whl); do
        dep_name="$(echo $dep_wheel | sed -E 's/-[0-9]/ /' | awk '{{print $1}}')"
        if [[ "$USER_PACK_NAMES" != *"$dep_name"* ]]; then
          echo "$dep_name" >> "$TEMP_SCRIPT_PATH/requirements-dep-wheels.txt"
          if [[ "$dep_wheel" == *"-none-"* ]]; then
            mv "$dep_wheel" ../
          fi
        fi
      done
    fi
  fi
}}

echo "Building user-defined packages..."
for path in `ls 2>/dev/null`; do
  if [[ -d "$path" ]]; then
    path="$path/"
  fi
  build_package_at_staging "$path"
done

if [[ -f "$SCRIPT_PATH/{_VCS_FILE_NAME}" ]]; then
  echo ""
  echo "Building VCS packages..."

  if [[ -z "$NON_DOCKER_MODE" ]]; then
    # enable saving password when cloning with git to avoid repeat typing
    git config --global credential.helper store
  fi

  cat "$SCRIPT_PATH/{_VCS_FILE_NAME}" | while read vcs_url ; do
    build_package_at_staging "$vcs_url"
  done
fi

echo ""
echo "Building and installing requirements..."
cd "$WHEELS_PATH"

# download and build all requirements as wheels
if [[ -f "$TEMP_SCRIPT_PATH/requirements-extra.txt" ]]; then
  if [[ -n "$NON_DOCKER_MODE" ]]; then
    "$PYBIN/python" -m pip download -r "$TEMP_SCRIPT_PATH/requirements-extra.txt" \
      $PIP_PLATFORM_ARGS $PYPI_EXTRA_ARG $PYPI_NO_DEPS_ARG || export BINARY_FAILED=1
    if [[ -n "$BINARY_FAILED" ]]; then
      build_package_at_staging -r "$TEMP_SCRIPT_PATH/requirements-extra.txt"
    fi
  else
    "$PYBIN/python" -m pip wheel -r "$TEMP_SCRIPT_PATH/requirements-extra.txt" $PYPI_EXTRA_ARG $PYPI_NO_DEPS_ARG
  fi
fi

if [[ -f "$TEMP_SCRIPT_PATH/requirements-dep-wheels.txt" ]]; then
  cd "$WHEELS_PATH"
  "$PYBIN/python" -m pip download -r "$TEMP_SCRIPT_PATH/requirements-dep-wheels.txt" \
    $PIP_PLATFORM_ARGS $PYPI_EXTRA_ARG $PYPI_NO_DEPS_ARG
fi

if [[ -n "$(which auditwheel)" ]]; then
  # make sure newly-built binary wheels fixed by auditwheel utility
  for fn in `ls *-linux_$MACHINE_TAG.whl 2>/dev/null`; do
    auditwheel repair "$fn" && rm -f "$fn"
  done
  if [[ -d wheelhouse ]]; then
    mv wheelhouse/*.whl ./
  fi
fi

if [[ -f "$TEMP_SCRIPT_PATH/requirements-user.txt" ]]; then
  cat "$TEMP_SCRIPT_PATH/requirements-user.txt" >> "$TEMP_SCRIPT_PATH/requirements-extra.txt"
fi

export PYTHONPATH="$OLDPYTHONPATH"

if [[ -n "$WITHOUT_MERGE" ]]; then
  # move all wheels into wheelhouse
  if [[ -n "$NON_DOCKER_MODE" ]]; then
    mv "$WHEELS_PATH"/*.whl "$WHEELHOUSE_PATH"
  else
    cp --no-preserve=mode,ownership "$WHEELS_PATH"/*.whl "$WHEELHOUSE_PATH"
  fi
  if [[ -f "$SCRIPT_PATH/{_EXCLUDE_FILE_NAME}" ]]; then
    echo ""
    echo "Removing exclusions..."
    for dep in `cat "$SCRIPT_PATH/{_EXCLUDE_FILE_NAME}"`; do
      rm -f "$WHEELHOUSE_PATH/$dep-"*.whl
    done
  fi
else
  # install with recently-built wheels
  "$PYBIN/python" -m pip install --target "$INSTALL_PATH/{package_site}" -r "$TEMP_SCRIPT_PATH/requirements-extra.txt" \
    $PYPI_NO_DEPS_ARG $PIP_PLATFORM_ARGS --no-index --find-links "file://$WHEELS_PATH"
  rm -rf "$WHEELS_PATH/*"

  if [[ -f "$SCRIPT_PATH/{_EXCLUDE_FILE_NAME}" ]]; then
    echo ""
    echo "Removing exclusions..."

    cd "$INSTALL_PATH/packages"
    for dep in `cat "$SCRIPT_PATH/{_EXCLUDE_FILE_NAME}"`; do
      dist_dir=`ls -d "$dep-"*".dist-info"`
      cat "$dist_dir/RECORD" | while read rec_line ; do
        fn="$(cut -d ',' -f 1 <<< "$rec_line" )"
        cur_root="$(cut -d '/' -f 1 <<< "$fn" )"
        echo "$cur_root" >> /tmp/.rmv_roots
        if [[ -f "$fn" ]]; then
          rm "$fn"
        fi
      done
    done

    if [[ -f "/tmp/.rmv_roots" ]]; then
      for root_dir in `cat /tmp/.rmv_roots | sort | uniq`; do
        find "$root_dir" -type d -empty -delete
      done
    fi
  fi

  # make sure the package is handled as a binary
  touch "$INSTALL_PATH/{package_site}/.pyodps-force-bin.so"

  echo ""
  echo "Running after build command..."
  {cmd_after_build}

  echo ""
  echo "Packages will be included in your archive:"
  PACKAGE_LIST_FILE="$INSTALL_PATH/{package_site}/.pyodps-pack-meta"
  "$PYBIN/python" -m pip list --path "$INSTALL_PATH/{package_site}" > "$PACKAGE_LIST_FILE"
  cat "$PACKAGE_LIST_FILE"

  echo ""
  echo "Creating archive..."
  mkdir -p "$WHEELHOUSE_PATH"
  cd "$INSTALL_PATH"
  if [[ -n "$MSYSTEM" ]]; then
    pack_path="$(cygpath -u "$WHEELHOUSE_PATH/{_DEFAULT_OUTPUT_FILE}")"
  else
    pack_path="$WHEELHOUSE_PATH/{_DEFAULT_OUTPUT_FILE}"
  fi
  tar --exclude="*.pyc" --exclude="__pycache__" -czf "$pack_path" "{package_site}"
fi
"""


class PackException(Exception):
    pass


def _indent(text, prefix, predicate=None):
    """Adds 'prefix' to the beginning of selected lines in 'text'.

    If 'predicate' is provided, 'prefix' will only be added to the lines
    where 'predicate(line)' is True. If 'predicate' is not provided,
    it will default to adding 'prefix' to all non-empty lines that do not
    consist solely of whitespace characters.
    """
    """Copied from textwrap.indent method of Python 3"""
    if predicate is None:
        def predicate(line):
            return line.strip()

    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if predicate(line) else line)

    return ''.join(prefixed_lines())


def _makedirs(name, mode=0o777, exist_ok=False):
    """makedirs(name [, mode=0o777][, exist_ok=False])

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.
    """
    """Copied from os.makedirs method of Python 3"""
    head, tail = os.path.split(name)
    if not tail:
        head, tail = os.path.split(head)
    if head and tail and not os.path.exists(head):
        try:
            _makedirs(head, exist_ok=exist_ok)
        except FileExistsError:
            # Defeats race condition when another thread created the path
            pass
        cdir = os.curdir
        if isinstance(tail, bytes):
            cdir = bytes(os.curdir, 'ASCII')
        if tail == cdir:  # xxx/newdir/. exists if xxx/newdir exists
            return
    try:
        os.mkdir(name, mode)
    except OSError:
        # Cannot rely on checking for EEXIST, since the operating system
        # could give priority to other errors like EACCES or EROFS
        if not exist_ok or not os.path.isdir(name):
            raise


def _to_unix(s):
    return s.replace(b"\r\n", b"\n")


def _create_build_script(**kwargs):
    template_params = defaultdict(lambda: "")
    template_params.update({k: v for k, v in globals().items() if v is not None})
    template_params.update(kwargs)
    return script_template.lstrip().format(**template_params)


def _copy_to_workdir(src_path, work_dir):
    _makedirs(os.path.join(work_dir, "build"), exist_ok=True)
    path_base_name = os.path.basename(src_path.rstrip("/").rstrip("\\"))
    dest_dir = os.path.join(work_dir, "build", path_base_name)

    shutil.copytree(src_path, dest_dir)


def _copy_package_paths(package_paths=None, work_dir=None, skip_user_path=True):
    remained = []
    for package_path in (package_paths or ()):
        base_name = os.path.basename(package_path.rstrip("/").rstrip("\\"))
        abs_path = os.path.abspath(package_path)
        if not skip_user_path or not abs_path.startswith(os.path.expanduser("~")):
            # not on user path, copy it into build path
            _copy_to_workdir(base_name, work_dir)
        else:
            remained.append(abs_path)
    return remained


def _build_docker_run_command(container_name, docker_image, work_dir, package_paths, docker_args):
    docker_executable = "docker" if not docker_path else os.path.join(docker_path, "docker")
    script_path_mapping = work_dir + "/scripts:/scripts"
    wheelhouse_path_mapping = work_dir + "/wheelhouse:/wheelhouse"
    build_path_mapping = work_dir + "/build:/build"

    cmdline = [docker_executable, "run"]
    if sys.stdin.isatty():
        cmdline.append("-it")

    if docker_args:
        cmdline.extend(shlex.split(docker_args))

    cmdline.extend(["--rm", "--name", container_name])
    cmdline.extend(["-v", script_path_mapping, "-v", wheelhouse_path_mapping])

    if package_paths:
        # need to create build path first for mount
        _makedirs(os.path.join(work_dir, "build"), exist_ok=True)
        cmdline.extend(["-v", build_path_mapping])

    remained = _copy_package_paths(package_paths, work_dir)
    for abs_path in remained:
        base_name = os.path.basename(abs_path.rstrip("/").rstrip("\\"))
        cmdline.extend(["-v", "%s:/build/%s" % (abs_path, base_name)])
    cmdline.extend(
        [docker_image, "/bin/bash", "/scripts/%s" % _PACK_SCRIPT_FILE_NAME]
    )
    return cmdline


def _build_docker_rm_command(container_name):
    docker_executable = "docker" if not docker_path else os.path.join(docker_path, "docker")
    return [docker_executable, "rm", "-f", container_name]


def _log_indent(title, text, indent=2):
    if logger.getEffectiveLevel() <= logging.DEBUG:
        logger.debug(title + "\n%s", _indent(text, " " * indent))


@contextlib.contextmanager
def _create_temp_work_dir(
    requirement_list,
    vcs_list,
    install_requires,
    exclude_list,
    before_script,
    **script_kwargs
):
    tmp_path = None
    try:
        tmp_path = os.path.expanduser("~/.tmp-pyodps-pack-%d" % int(time.time()))
        _makedirs(tmp_path, exist_ok=True)
        script_path = os.path.join(tmp_path, "scripts")
        _makedirs(script_path, exist_ok=True)
        _makedirs(os.path.join(tmp_path, "wheelhouse"), exist_ok=True)

        if requirement_list:
            req_text = "\n".join(requirement_list) + "\n"
            _log_indent("Content of requirements.txt:", req_text)
            with open(os.path.join(script_path, _REQUIREMENT_FILE_NAME), "w") as res_file:
                res_file.write(req_text)

        if vcs_list:
            vcs_text = "\n".join(vcs_list) + "\n"
            _log_indent("Content of requirements-vcs.txt:", vcs_text)
            with open(os.path.join(script_path, _VCS_FILE_NAME), "w") as res_file:
                res_file.write(vcs_text)

        if install_requires:
            install_req_text = "\n".join(install_requires) + "\n"
            _log_indent("Content of install-requires.txt:", install_req_text)
            with open(os.path.join(script_path, _INSTALL_REQ_FILE_NAME), "w") as install_req_file:
                install_req_file.write(install_req_text)

        if exclude_list:
            exclude_text = "\n".join(exclude_list) + "\n"
            _log_indent("Content of excludes.txt:", exclude_text)
            with open(os.path.join(script_path, _EXCLUDE_FILE_NAME), "w") as exclude_file:
                exclude_file.write(exclude_text)

        if before_script or cmd_before_build:
            with open(os.path.join(script_path, _BEFORE_SCRIPT_FILE_NAME), "wb") as before_script_file:
                if before_script:
                    with open(before_script, "rb") as src_before_file:
                        before_script_file.write(_to_unix(src_before_file.read()))
                        before_script_file.write(b"\n\n")
                if cmd_before_build:
                    before_script_file.write(_to_unix(cmd_before_build.encode()))
            if logger.getEffectiveLevel() <= logging.DEBUG:
                with open(os.path.join(script_path, _BEFORE_SCRIPT_FILE_NAME), "r") as before_script_file:
                    _log_indent("Content of before-script.sh:", before_script_file.read())

        with open(os.path.join(script_path, _PACK_SCRIPT_FILE_NAME), "wb") as pack_file:
            build_script = _create_build_script(**script_kwargs)
            _log_indent("Pack script:", build_script)
            # make sure script work well under windows
            pack_file.write(_to_unix(build_script.encode()))
        yield tmp_path
    finally:
        if tmp_path and os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)


def _get_default_pypi_index():
    proc = subprocess.Popen([sys.executable, "-m", "pip", "config", "list"], stdout=subprocess.PIPE)
    proc.wait()
    if proc.returncode != 0:
        warnings.warn(
            'Failed to call `pip config list`, return code is %s. '
            'Will use default index instead. Specify "-i <index-url>" '
            'if you want to use another package index.' % proc.returncode
        )
        return None
    for line in proc.stdout.read().decode().splitlines():
        var, value = line.split("=", 1)
        if var == "global.index-url":
            return value.strip("'")
    return None


def _filter_local_package_paths(parsed_args):
    filtered_req = []
    package_path = []
    vcs_urls = []

    parsed_args.specifiers = list(parsed_args.specifiers)
    for req_file in parsed_args.requirement:
        with open(req_file, "r") as input_req_file:
            for req in input_req_file:
                parsed_args.specifiers.append(req)

    for req in parsed_args.specifiers:
        if re.findall(r"[^=\>\<]=[^=\>\<]", req):
            req = req.replace("=", "==")
        if os.path.exists(req):
            package_path.append(req)
        elif any(req.startswith(prefix) for prefix in _vcs_prefixes):
            vcs_urls.append(req)
        else:
            filtered_req.append(req)
    parsed_args.specifiers = filtered_req
    parsed_args.package_path = package_path
    parsed_args.vcs_urls = vcs_urls


def _collect_install_requires(parsed_args):
    install_requires = parsed_args.install_requires or []
    for req_file_name in parsed_args.install_requires_file:
        with open(req_file_name, "r") as req_file:
            install_requires.extend(req_file.read().splitlines())
    parsed_args.install_requires = install_requires


def _collect_env_packages(exclude_editable=False, exclude=None, index_url=None):
    print("Extracting packages from local environment...")
    exclude = set(exclude or [])
    pip_cmd = [sys.executable, "-m", "pip", "list", "--format", "json"]
    if exclude_editable:
        pip_cmd += ["--exclude-editable"]
    if index_url:
        pip_cmd += ["--index-url", index_url]

    pack_descriptions = []
    proc = subprocess.Popen(pip_cmd, stdout=subprocess.PIPE)
    proc.wait()
    if proc.returncode != 0:
        raise PackException(
            "ERROR: Failed to call `pip list`. Return code is %s" % proc.returncode
        )
    pack_descriptions.extend(json.loads(proc.stdout.read()))

    specifiers = []
    missing_packs = []
    for desc in pack_descriptions:
        pack_name = desc["name"]
        if pack_name in exclude:
            continue
        if "editable_project_location" in desc:
            specifiers.append(desc["editable_project_location"])
        else:
            specifiers.append("%s==%s" % (desc["name"], desc["version"]))
    if missing_packs:
        warnings.warn(
            "Cannot find packages %s in package index. These packages cannot be included."
            % ",".join(missing_packs)
        )
    return specifiers


def _get_arch(arch=None):
    arch = (arch or "x86_64").lower()
    if arch in ("arm64", "aarch64"):
        return "aarch64"
    elif arch == "x86_64":
        return arch
    raise PackException("Arch %s not supported" % arch)


def _get_default_image(use_legacy_image=False, arch=None):
    arch = _get_arch(arch)
    if arch != "x86_64" and use_legacy_image:
        raise PackException("Cannot use legacy image when building on other arches")
    if use_legacy_image:
        return _LEGACY_DOCKER_IMAGE_X64
    elif arch == "x86_64":
        return _DEFAULT_DOCKER_IMAGE_X64
    elif arch == "aarch64":
        return _DEFAULT_DOCKER_IMAGE_ARM64
    else:
        raise PackException("Arch %s not supported" % arch)


def _get_python_abi_version(python_version=None, mcpy27=None, dwpy27=None):
    if python_abi_env and python_version is not None:
        raise PackException(
            "You should not specify environment variable 'PYABI' and '--python-version' at the same time."
        )
    if python_version is None:
        python_abi_version = python_abi_env or _DEFAULT_PYABI
    else:
        if "." not in python_version:
            version_parts = (int(python_version[0]), int(python_version[1:]))
        else:
            version_parts = tuple(int(pt) for pt in python_version.split("."))[:2]
        cp_tag = "cp%d%d" % version_parts
        python_abi_version = cp_tag + "-" + cp_tag
        if version_parts < (3, 8):
            python_abi_version += "m"
    if dwpy27:
        if mcpy27:
            raise PackException("You should not specify '--dwpy27' and '--mcpy27' at the same time.")
        python_abi_version = _DWPY27_PYABI
    elif mcpy27:
        python_abi_version = _MCPY27_PYABI
    return python_abi_version


def _get_bash_path():
    """Get bash executable. When under Windows, retrieves path of Git bash. Otherwise returns /bin/bash."""
    if not _is_windows:
        return "/bin/bash"

    import winreg

    key = None
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\GitForWindows")
        git_path = winreg.QueryValueEx(key, "InstallPath")[0]
        bash_path = os.path.join(git_path, "bin", "bash.exe")
        if not os.path.exists(bash_path):
            raise OSError("bash.exe not found")
        return bash_path
    except OSError:
        err_msg = (
            "Failed to locate Git Bash. Please check your installation of Git for Windows "
            "which can be obtained at https://gitforwindows.org/."
        )
        if int(platform.win32_ver()[1].rsplit(".")[-1]) > 19041:
            err_msg += " You may also try packing under WSL or with Docker."
        else:
            err_msg += " You may also try packing with Docker."
        raise PackException(err_msg)
    finally:
        if key:
            key.Close()


def _get_local_pack_executable(work_dir):
    """Create a virtualenv for local packing if possible"""
    try:
        import venv
    except ImportError:
        return sys.executable

    env_dir = os.path.join(work_dir, "venv")
    print("Creating virtual environment for local build...")
    venv.create(env_dir, symlinks=not _is_windows, with_pip=True)
    if _is_windows:
        return os.path.join(env_dir, "Scripts", "python.exe")
    else:
        return os.path.join(env_dir, "bin/python")


def _main(parsed_args):
    if parsed_args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if parsed_args.pack_env:
        if parsed_args.specifiers:
            raise PackException(
                "ERROR: Cannot supply --pack-env with other package specifiers."
            )
        parsed_args.specifiers = _collect_env_packages(
            parsed_args.exclude_editable, parsed_args.exclude, parsed_args.index_url
        )

    _filter_local_package_paths(parsed_args)
    _collect_install_requires(parsed_args)

    if not parsed_args.specifiers and not parsed_args.package_path and not parsed_args.vcs_urls:
        raise PackException("ERROR: You must give at least one requirement to install.")

    index_url = parsed_args.index_url or _get_default_pypi_index() or ""
    if index_url:
        logger.debug("Using PyPI index %s", index_url)
    else:
        logger.debug("Using default PyPI index")

    prefer_binary_str = "true" if parsed_args.prefer_binary else ""
    no_deps_str = "true" if parsed_args.no_deps else ""
    debug_str = "true" if parsed_args.debug else ""
    without_merge_str = "true" if parsed_args.without_merge else ""

    python_abi_version = _get_python_abi_version(
        parsed_args.python_version, parsed_args.mcpy27, parsed_args.dwpy27
    )
    pypi_trusted_hosts = " ".join(parsed_args.trusted_host or [])

    with _create_temp_work_dir(
        parsed_args.specifiers,
        parsed_args.vcs_urls,
        parsed_args.install_requires,
        parsed_args.exclude,
        parsed_args.run_before,
        pypi_index=index_url,
        prefer_binary=prefer_binary_str,
        no_deps=no_deps_str,
        without_merge=without_merge_str,
        python_abi_version=python_abi_version,
        pypi_trusted_hosts=pypi_trusted_hosts,
        debug=debug_str,
    ) as work_dir:
        container_name = "pack-cnt-%d" % int(time.time())

        use_legacy_image = parsed_args.legacy_image or parsed_args.mcpy27 or parsed_args.dwpy27
        default_image = _get_default_image(use_legacy_image, parsed_args.arch)
        docker_image = docker_image_env or default_image

        if parsed_args.without_docker:
            _copy_package_paths(parsed_args.package_path, work_dir, skip_user_path=False)

            pyversion, pyabi = python_abi_version.split("-", 1)
            pyversion = pyversion[2:]
            build_cmd = [_get_bash_path(), os.path.join(work_dir, "scripts", _PACK_SCRIPT_FILE_NAME)]
            build_env = {
                "NON_DOCKER_MODE": "true",
                "PACK_ROOT": work_dir,
                "PYEXECUTABLE": _get_local_pack_executable(work_dir),
                "PYPLATFORM": default_image.replace("quay.io/pypa/", ""),
                "PYVERSION": pyversion,
                "PYABI": pyabi,
                "TARGET_ARCH": _get_arch(parsed_args.arch),
            }
            logger.debug("Command: %r", build_cmd)
            logger.debug("Environment variables: %r", build_cmd)
        else:
            build_cmd = _build_docker_run_command(
                container_name,
                docker_image,
                work_dir,
                parsed_args.package_path,
                parsed_args.docker_args,
            )
            build_env = None
            logger.debug("Docker command: %r", build_cmd)

        proc = subprocess.Popen(build_cmd, env=build_env)
        cancelled = False
        try:
            proc.wait()
        except KeyboardInterrupt:
            cancelled = True
            if not parsed_args.without_docker:
                docker_rm_cmd = _build_docker_rm_command(container_name)
                logger.debug("Docker rm command: %r", docker_rm_cmd)
                subprocess.Popen(docker_rm_cmd, stdout=subprocess.PIPE)
                proc.wait()

        if proc.returncode != 0:
            cancelled = cancelled or os.path.exists(os.path.join(work_dir, "scripts", ".cancelled"))
            if cancelled:
                print("Cancelled by user.")
            else:
                if parsed_args.without_docker:
                    print(
                        "Errors occurred when creating your package. This is often caused "
                        "by mismatching Python version, platform or architecture when "
                        "encountering binary packages. Please check outputs for details. "
                        "You may try building your packages inside Docker by removing "
                        "--without-docker option, which often resolves the issue."
                    )
                else:
                    print(
                        "Errors occurred when creating your package. Please check outputs "
                        "for details. You may add a `--debug` option to obtain more "
                        "information."
                    )

                if proc.returncode == _SEGFAULT_ERR_CODE and use_legacy_image:
                    print(
                        "Image manylinux1 might crash silently under some Docker environments. "
                        "You may try under a native Linux environment. Details can be seen at "
                        "https://mail.python.org/pipermail/wheel-builders/2016-December/000239.html."
                    )
                elif _is_linux and "SUDO_USER" not in os.environ:
                    print(
                        "You need to run pyodps-pack with sudo to make sure docker is "
                        "executed properly."
                    )
        else:
            if parsed_args.without_merge:
                src_path = os.path.join(work_dir, "wheelhouse", "*.whl")
                for wheel_name in glob.glob(src_path):
                    shutil.move(wheel_name, os.path.basename(wheel_name))
            else:
                src_path = os.path.join(work_dir, "wheelhouse", _DEFAULT_OUTPUT_FILE)
                shutil.move(src_path, parsed_args.output)

            if _is_linux and "SUDO_UID" in os.environ and "SUDO_GID" in os.environ:
                own_desc = "%s:%s" % (os.environ["SUDO_UID"], os.environ["SUDO_GID"])
                target_path = "*.whl" if parsed_args.without_merge else parsed_args.output
                chown_proc = subprocess.Popen(["chown", own_desc, target_path])
                chown_proc.wait()

            if parsed_args.without_merge:
                print("Result wheels stored at current dir")
            else:
                print("Result package stored as %s" % parsed_args.output)
        return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "specifiers", metavar="REQ", nargs="*",
        help="a requirement item compatible with pip command",
    )
    parser.add_argument(
        "--requirement", "-r", action="append", default=[],
        metavar="PATH", help="Path of requirements.txt including file name",
    )
    parser.add_argument(
        "--install-requires", action="append", default=[], help="Requirement for install time"
    )
    parser.add_argument(
        "--install-requires-file", action="append", default=[],
        help="Requirement file for install time",
    )
    parser.add_argument(
        "--run-before", help="Prepare script before package build."
    )
    parser.add_argument(
        "--no-deps", action="store_true", default=False,
        help="Don't put package dependencies into archives",
    )
    parser.add_argument(
        "--exclude", "-X", action="append", default=[],
        metavar="DEPEND", help="Requirements to exclude from the package",
    )
    parser.add_argument(
        "--index-url", "-i", default="",
        help="Base URL of PyPI package. If absent, will use "
             "`global.index-url` in `pip config list` command by default.",
    )
    parser.add_argument(
        "--trusted-host", metavar="HOST:PATH", action="append", default=[],
        help="Mark this host or host:port pair as trusted, "
             "even though it does not have valid or any HTTPS.",
    )
    parser.add_argument(
        "--legacy-image", "-l", action="store_true", default=False,
        help="Use legacy image to make packages",
    )
    parser.add_argument(
        "--mcpy27", action="store_true", default=False,
        help="Build package for Python 2.7 on MaxCompute. "
             "If enabled, will assume `legacy-image` to be true.",
    )
    parser.add_argument(
        "--dwpy27", action="store_true", default=False,
        help="Build package for Python 2.7 on DataWorks. "
             "If enabled, will assume `legacy-image` to be true.",
    )
    parser.add_argument(
        "--prefer-binary", action="store_true", default=False,
        help="Prefer older binary packages over newer source packages",
    )
    parser.add_argument(
        "--output", "-o", default="packages.tar.gz", help="Target archive file name to store"
    )
    parser.add_argument(
        "--pack-env", action="store_true", default=False, help="Pack full environment"
    )
    parser.add_argument(
        "--exclude-editable", action="store_true", default=False,
        help="Exclude editable packages when packing",
    )
    parser.add_argument(
        "--arch", default="x86_64",
        help="Architecture of target package, x86_64 by default. Currently only x86_64 "
             "and aarch64 supported. Do not use this argument if you are not running "
             "your code in a proprietary cloud."
    )
    parser.add_argument(
        "--python-version",
        help="Version of Python your environment is on, for instance 3.6. "
             "You may also use 36 instead. Do not use this argument if you "
             "are not running your code in a proprietary cloud."
    )
    parser.add_argument(
        "--docker-args", help="Extra arguments for Docker."
    )
    parser.add_argument(
        "--without-docker", action="store_true", default=False,
        help="Create packages without Docker. May cause errors if incompatible "
             "binaries involved.",
    )
    parser.add_argument(
        "--without-merge", action="store_true", default=False,
        help="Create or download wheels without merging them.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Dump debug messages for diagnose purpose",
    )

    args = parser.parse_args()

    try:
        sys.exit(_main(args) or 0)
    except PackException as ex:
        print(ex.args[0], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
