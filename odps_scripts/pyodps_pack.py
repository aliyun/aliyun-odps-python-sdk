#!/usr/bin/env python
from __future__ import print_function
import argparse
import contextlib
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections import defaultdict

_DEFAULT_PYABI = "cp37-cp37m"
_MCPY27_PYABI = "cp27-cp27m"
_DWPY27_PYABI = "cp27-cp27mu"
_DEFAULT_DOCKER_IMAGE = "quay.io/pypa/manylinux2014_x86_64"
_LEGACY_DOCKER_IMAGE = "quay.io/pypa/manylinux1_x86_64"
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
_vcs_prefixes = [prefix + "+" for prefix in "git hg svn bzr".split()]

logger = logging.getLogger(__name__)


script_template = r"""
#!/bin/bash
if [[ "{debug}" == "true" ]]; then
  echo "Files under /scripts:"
  ls /scripts
  set -e -x
else
  set -e
fi
export PIP_ROOT_USER_ACTION=ignore
export PIP_DISABLE_PIP_VERSION_CHECK=1

PYBIN="/opt/python/{python_abi}/bin"
export PATH="$PYBIN:$PATH"

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
if [[ -n "{pypi_trusted_hosts}" ]]; then
  for trusted_host in `echo "{pypi_trusted_hosts}"`; do
    PYPI_EXTRA_ARG="$PYPI_EXTRA_ARG --trusted-host $trusted_host"
  done
fi

mkdir -p /build /dist /wheels /install-req /tmp/scripts

if [[ -f "/scripts/{_BEFORE_SCRIPT_FILE_NAME}" ]]; then
  echo "Running before build command..."
  /bin/bash "/scripts/{_BEFORE_SCRIPT_FILE_NAME}"
  echo ""
fi

if [[ -f "/scripts/{_INSTALL_REQ_FILE_NAME}" ]]; then
  echo "Installing build prerequisites..."
  "$PYBIN/pip" install --target /install-req -r "/scripts/{_INSTALL_REQ_FILE_NAME}"
  export OLDPYTHONPATH="$PYTHONPATH"
  export PYTHONPATH="/install-req:$PYTHONPATH"
  echo ""
fi

# build user-defined package
cd /build
if [[ -f /scripts/requirements-user.txt ]]; then
  cp /scripts/requirements-user.txt /tmp/scripts/requirements-user.txt
fi
if [[ -f "/scripts/{_REQUIREMENT_FILE_NAME}" ]]; then
  cp "/scripts/{_REQUIREMENT_FILE_NAME}" /tmp/scripts/requirements-extra.txt
fi

echo "Building user-defined packages..."
for path in `ls 2>/dev/null`; do
  if [[ -d "$path" ]]; then
    path="$path/"
  fi
  "$PYBIN/pip" wheel --no-deps --wheel-dir /wheels/staging $PYPI_EXTRA_ARG "$path"

  cd /wheels/staging
  WHEEL_NAME="$(ls *.whl)"
  USER_PACKAGE_NAME="$(echo $WHEEL_NAME | sed -E 's/-[0-9]/ /' | awk '{{print $1}}')"
  echo "$USER_PACKAGE_NAME" >> /tmp/scripts/requirements-user.txt
  mv *.whl ../

  if [[ -z "$PYPI_NO_DEPS_ARG" ]]; then
    cd /wheels
    "$PYBIN/pip" wheel --wheel-dir /wheels $PYPI_EXTRA_ARG "$WHEEL_NAME"
  fi
done

if [[ -f "/scripts/{_VCS_FILE_NAME}" ]]; then
  echo ""
  echo "Building VCS packages..."

  # enable saving password when cloning with git to avoid repeative typing
  git config --global credential.helper store

  cat "/scripts/{_VCS_FILE_NAME}" | while read vcs_url ; do
    "$PYBIN/pip" wheel --no-deps --wheel-dir /wheels/staging $PYPI_EXTRA_ARG "$vcs_url"

    cd /wheels/staging
    WHEEL_NAME="$(ls *.whl)"
    USER_PACKAGE_NAME="$(echo $WHEEL_NAME | sed -E 's/-[0-9]/ /' | awk '{{print $1}}')"
    echo "$USER_PACKAGE_NAME" >> /tmp/scripts/requirements-user.txt
    mv *.whl ../

    if [[ -z "$PYPI_NO_DEPS_ARG" ]]; then
      cd /wheels
      "$PYBIN/pip" wheel --wheel-dir /wheels $PYPI_EXTRA_ARG "$WHEEL_NAME"
    fi
  done
fi

echo ""
echo "Building and installing requirements..."
cd /wheels

# download and build all requirements as wheels
if [[ -f /tmp/scripts/requirements-extra.txt ]]; then
  "$PYBIN/pip" wheel -r /tmp/scripts/requirements-extra.txt $PYPI_EXTRA_ARG $PYPI_NO_DEPS_ARG
fi

# make sure newly-built binary wheels fixed by auditwheel utility
for fn in `ls *-linux_x86_64.whl 2>/dev/null`; do
  auditwheel repair "$fn" && rm -f "$fn"
done
if [[ -d wheelhouse ]]; then
  mv wheelhouse/*.whl ./
fi

if [[ -f /tmp/scripts/requirements-user.txt ]]; then
  cat /tmp/scripts/requirements-user.txt >> /tmp/scripts/requirements-extra.txt
fi

export PYTHONPATH="$OLDPYTHONPATH"

# install with recently-built wheels
"$PYBIN/pip" install --target /dist/{package_site} -r /tmp/scripts/requirements-extra.txt \
  $PYPI_NO_DEPS_ARG --no-index --find-links file:///wheels
rm -rf /wheels/*

if [[ -f "/scripts/{_EXCLUDE_FILE_NAME}" ]]; then
  echo ""
  echo "Removing exclusions..."

  cd /dist/packages
  for dep in `cat "/scripts/{_EXCLUDE_FILE_NAME}"`; do
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
touch "/dist/{package_site}/.pyodps-force-bin.so"

echo ""
echo "Running after build command..."
{cmd_after_build}

echo ""
echo "Packages will be included in your archive:"
PACKAGE_LIST_FILE="/dist/{package_site}/.pyodps-pack-meta"
"$PYBIN/pip" list --path "/dist/{package_site}" > "$PACKAGE_LIST_FILE"
cat "$PACKAGE_LIST_FILE"

echo ""
echo "Creating archive..."
cd /dist
tar --exclude="*.pyc" --exclude="__pycache__" -czf "/wheelhouse/{_DEFAULT_OUTPUT_FILE}" "{package_site}"
"""


class PackException(Exception):
    pass


def _to_unix(s):
    return s.replace(b"\r\n", b"\n")


def _create_build_script(**kwargs):
    template_params = defaultdict(lambda: "")
    template_params.update({k: v for k, v in globals().items() if v is not None})
    template_params.update(kwargs)
    return script_template.lstrip().format(**template_params)


def _copy_to_workdir(src_path, work_dir):
    os.makedirs(os.path.join(work_dir, "build"))
    path_base_name = os.path.basename(src_path.rstrip("/").rstrip("\\"))
    dest_dir = os.path.join(work_dir, "build", path_base_name)

    shutil.copytree(src_path, dest_dir)


def _build_docker_run_command(container_name, docker_image, work_dir, package_paths):
    docker_executable = "docker" if not docker_path else os.path.join(docker_path, "docker")
    script_path_mapping = work_dir + "/scripts:/scripts"
    wheelhouse_path_mapping = work_dir + "/wheelhouse:/wheelhouse"
    build_path_mapping = work_dir + "/build:/build"

    cmdline = [docker_executable, "run"]
    if sys.stdin.isatty():
        cmdline.append("-it")

    cmdline.extend(["--rm", "--name", container_name])
    cmdline.extend(["-v", script_path_mapping, "-v", wheelhouse_path_mapping])

    if package_paths:
        cmdline.extend(["-v", build_path_mapping])

    package_paths = package_paths or []
    for package_path in package_paths:
        base_name = os.path.basename(package_path.rstrip("/").rstrip("\\"))
        abs_path = os.path.abspath(package_path)
        if not abs_path.startswith(os.path.expanduser("~")):
            _copy_to_workdir(base_name, work_dir)
        else:
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
        logger.debug(title + "\n%s", textwrap.indent(text, " " * indent))


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
        os.makedirs(tmp_path, exist_ok=True)
        script_path = os.path.join(tmp_path, "scripts")
        os.makedirs(script_path, exist_ok=True)
        os.makedirs(os.path.join(tmp_path, "wheelhouse"), exist_ok=True)

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
        if tmp_path:
            shutil.rmtree(tmp_path)


def _get_default_pypi_index():
    proc = subprocess.Popen(["pip", "config", "list"], stdout=subprocess.PIPE)
    proc.wait()
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


def _main(parsed_args):
    if parsed_args.debug:
        logging.basicConfig(level=logging.DEBUG)

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

    python_abi = python_abi_env or _DEFAULT_PYABI
    if parsed_args.dwpy27:
        if parsed_args.mcpy27:
            raise PackException("You should not specify '--dwpy27' and '--mcpy27' at the same time.")
        python_abi = _DWPY27_PYABI
    elif parsed_args.mcpy27:
        python_abi = _MCPY27_PYABI

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
        python_abi=python_abi,
        pypi_trusted_hosts=pypi_trusted_hosts,
        debug=debug_str,
    ) as work_dir:
        container_name = "pack-cnt-%d" % int(time.time())

        use_legacy_image = parsed_args.legacy_image or parsed_args.mcpy27 or parsed_args.dwpy27
        default_image = _DEFAULT_DOCKER_IMAGE if not use_legacy_image else _LEGACY_DOCKER_IMAGE
        docker_image = docker_image_env or default_image

        docker_cmd = _build_docker_run_command(
            container_name, docker_image, work_dir, parsed_args.package_path
        )
        logger.debug("Docker command: %r", docker_cmd)

        proc = subprocess.Popen(docker_cmd)
        cancelled = False
        try:
            proc.wait()
        except KeyboardInterrupt:
            docker_rm_cmd = _build_docker_rm_command(container_name)
            logger.debug("Docker rm command: %r", docker_cmd)
            subprocess.Popen(docker_rm_cmd, stdout=subprocess.PIPE)
            proc.wait()
            cancelled = True

        if proc.returncode != 0:
            if cancelled:
                print("Cancelled by user.")
            else:
                print("Errors occurred when creating your package. Please check outputs for details. "
                      "You may add a `--debug` option to obtain more information.")
                if proc.returncode == _SEGFAULT_ERR_CODE and use_legacy_image:
                    print("Image manylinux1 might crash silently under some Docker environments. "
                          "You may try under a native Linux environment. Details can be seen at "
                          "https://mail.python.org/pipermail/wheel-builders/2016-December/000239.html.")
                elif _is_linux and "SUDO_USER" not in os.environ:
                    print("You need to run pyodps-pack with sudo to make sure docker is executed properly.")
        else:
            src_path = os.path.join(work_dir, "wheelhouse", _DEFAULT_OUTPUT_FILE)
            shutil.move(src_path, parsed_args.output)

            if _is_linux and "SUDO_UID" in os.environ and "SUDO_GID" in os.environ:
                own_desc = "%s:%s" % (os.environ["SUDO_UID"], os.environ["SUDO_GID"])
                chown_proc = subprocess.Popen(["chown", own_desc, parsed_args.output])
                chown_proc.wait()
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
