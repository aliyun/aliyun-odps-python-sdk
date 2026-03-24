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

import argparse
import binascii
import json
import logging
import multiprocessing
import os
import pickle
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import types
import uuid
import warnings

from .compat import patch_compat
from .diagnose import diagnose_exit_code
from .envs import *
from .headers import config_with_headers, use_spawn_method
from .mars_support import create_mars_cluster, create_sign_server
from .resource import run_load_resource_command
from .runner import run_code
from .utils import import_all_sub_modules, is_limit_exceeded, kill_process_tree

logger = logging.getLogger("odps.pyodpswrapper")

DEFAULT_SHARED_MEM_LIMIT = "8g"

_ak_encrypt_key = None


def _decrypt_ak(key):
    from pyDes import ECB, PAD_PKCS5, des

    global _ak_encrypt_key

    if _ak_encrypt_key is None:
        _ak_encrypt_key = os.environ.pop(ACCESS_KEY_ENCRYPT_KEY, None)
    k = des(_ak_encrypt_key, ECB, "\0\0\0\0\0\0\0\0", pad=None, padmode=PAD_PKCS5)
    return k.decrypt(binascii.a2b_hex(key))


def config_with_code_header(args):
    if args.code_file is not None:
        with open(args.code_file, "rb") as src_file:
            code = src_file.read()
    elif args.code is not None:
        code = args.code.encode()
    else:
        return
    config_with_headers(code)


def call_in_process(args, odps):
    from odps import tempobj

    tempobj.TEMP_ROOT = tempfile.mkdtemp()

    if args.user:
        uid = pwd.getpwnam(args.user).pw_uid
    else:
        uid = None

    mem_limit = None
    if args.mem_limit:
        mem_limit = args.mem_limit
    elif os.getenv(REGION_ENV, "").lower() == "d2":
        mem_limit = DEFAULT_SHARED_MEM_LIMIT

    if mem_limit:
        print("Execution memory is limited to %s" % mem_limit)

    try:
        os.environ[PICKLE_ACCOUNT] = "true"
        if use_spawn_method():
            # use spawn to reduce possibility of bugs introduced by fork if possible
            mp_ctx = multiprocessing.get_context("spawn")
        else:
            mp_ctx = multiprocessing
        p_conn, conn = mp_ctx.Pipe()

        p = mp_ctx.Process(target=run_code, args=(odps, args, uid, conn))
        p.start()

        # paths = cgroup_limit_resource(memory_size=args.mem, cpu_percent=args.cpu,
        #                               label=args.label, pid=p.pid)
        paths = None

        def handle_kill(signum, frame):
            print("Try to terminate process")
            if p.is_alive():
                kill_process_tree(p.pid, signum)
            p.join(10)  # wait 10 secs
            if p.is_alive():
                print("Try to force to terminate process")
                kill_process_tree(p.pid, 9)  # kill -9
            print("Kill done")

        # set signals
        signal.signal(signal.SIGTERM, handle_kill)
        signal.signal(signal.SIGINT, handle_kill)

        try:
            p_conn.send(1)  # send to make the process run
            while p.is_alive():
                p.join(1)
                if mem_limit and is_limit_exceeded(mem_limit):
                    kill_process_tree(p.pid, signal.SIGKILL)

            res = p_conn.poll()
            if not res:
                diagnose_exit_code(p.exitcode, logger)
                return -1
            return p_conn.recv()
        finally:
            try:
                # use tricky way to remove the cgroup dir
                if paths:
                    for path in paths:
                        subprocess.call(
                            r"find %s -depth -type d -print -exec rmdir {} \;" % path,
                            shell=True,
                            stdout=subprocess.PIPE,
                        )
            except Exception as e:
                logger.error(e)
    finally:
        try:
            # clean temporary objects
            try:
                t = threading.Thread(
                    target=tempobj.clean_objects, args=(odps, [args.biz_id])
                )
                t.daemon = True
                t.start()
                t.join(10)
            finally:
                try:
                    shutil.rmtree(tempobj.TEMP_ROOT)
                except OSError:
                    pass
            # upload remaining surveys
            odps.get_project()._client.upload_survey_log()
        except Exception as e:
            logger.error(e)


def wrapper_main(args, args_loader=None):
    from odps import ODPS
    from odps import __version__ as pyodps_version
    from odps import options
    from odps.accounts import SignServerAccount
    from odps.core import DEFAULT_ENDPOINT

    try:
        from odps.accounts import StsAccount
    except ImportError:
        StsAccount = None

    if args_loader is not None:
        args = args_loader(args)

    if args.access_key is None and CONFIG_FILE not in os.environ:
        raise RuntimeError("Either access id and key or config file should be provided")

    if os.getenv(REGION_ENV) and os.getenv(REGION_ENV, "").lower() != "d2":
        os.environ["ODPS_REGION_NAME"] = os.getenv(REGION_ENV)

    if args.access_key is not None:
        access_id, access_key, project = (
            args.access_id[0],
            args.access_key[0],
            args.project[0],
        )
        endpoint, tunnel_endpoint = args.endpoint, args.tunnel_endpoint
        kw = dict(tunnel_endpoint=tunnel_endpoint)
        if args.security_token:
            kw["sts_token"] = args.security_token[0]
        obj = (
            access_id,
            access_key,
            project,
            endpoint or DEFAULT_ENDPOINT,
            kw,
        )

        if args.load_resource:
            encrypted = os.environ.get(ACCESS_KEY_ENCRYPT, "true") == "true"
            if encrypted:
                obj = obj[:1] + (_decrypt_ak(obj[1]),) + obj[2:]
            return run_load_resource_command(args, ODPS(*obj[:-1], **obj[-1]))

        code = None
        if args.code_file is None:
            code = sys.stdin.read() if args.code is None else args.code

        f, path = tempfile.mkstemp()
        with os.fdopen(f, "wb") as f:
            f.write(pickle.dumps(obj))

        exec_args = []
        if args.mem is not None:
            exec_args.extend(["--mem", str(args.mem)])
        if args.mem_limit is not None:
            exec_args.extend(["--mem-limit", str(args.mem_limit)])
        if args.cpu is not None:
            exec_args.extend(["--cpu", str(args.cpu)])
        if args.cpu_timeout is not None:
            exec_args.extend(["--cpu-timeout", str(args.cpu_timeout)])
        if args.code_file is not None:
            exec_args.extend(["--code-file", args.code_file])
        if code:
            exec_args.extend(["--code", code])
        if args.user is not None:
            exec_args.extend(["--user", str(args.user)])
        if args.label is not None:
            exec_args.extend(["--label", str(args.label)])
        if args.chroot is not None:
            exec_args.extend(["--chroot", args.chroot])
        if args.args:
            exec_args.extend(["--args"] + args.args)

        os.environ[CONFIG_FILE] = path
        os.environ["biz_id"] = args.biz_id or get_biz_id(project.rstrip("_dev"))
        os.execve(
            sys.executable, [sys.executable] + sys.argv[:1] + exec_args, os.environ
        )
    else:
        config_file = os.environ[CONFIG_FILE]
        sign_server = None
        server_token = str(uuid.uuid4())
        try:
            with open(config_file, "rb") as f:
                # set d2 label to user agent, this has to be performed before creation of ODPS entrance
                ua_parts = [
                    options.user_agent_pattern,
                    "Python%d-DW" % sys.version_info[0],
                ]
                if "SKYNET_ID" in os.environ:
                    ua_parts.append("SKYNET-ID-" + os.environ["SKYNET_ID"])
                if "SKYNET_TASKID" in os.environ:
                    ua_parts.append("SKYNET-TASKID-" + os.environ["SKYNET_TASKID"])
                options.user_agent_pattern = " ".join(ua_parts)

                obj = pickle.load(f)
                encrypted = os.environ.get(ACCESS_KEY_ENCRYPT, "true") == "true"
                odps_args = obj[:-1]
                odps_kw = obj[-1]
                if encrypted:
                    odps_args = (
                        odps_args[:1] + (_decrypt_ak(odps_args[1]),) + odps_args[2:]
                    )

                if not args.load_resource and is_internal():
                    access_id, access_key = odps_args[:2]
                    set_skynet_to_odps_options()
                    sign_server = create_sign_server(token=server_token)
                    sign_server.accounts[access_id] = access_key
                    sign_server.start(("127.0.0.1", 0))

                    account = SignServerAccount(
                        access_id, sign_server.server.server_address, token=server_token
                    )
                    odps = ODPS(None, None, *odps_args[2:], account=account, **odps_kw)
                    odps.create_mars_cluster = types.MethodType(
                        create_mars_cluster, odps
                    )
                else:
                    if "sts_token" not in odps_kw:
                        odps = ODPS(*odps_args, **odps_kw)
                    else:
                        access_id, access_key, project, endpoint = odps_args
                        sts_account = StsAccount(
                            access_id, access_key, odps_kw["sts_token"]
                        )
                        odps = ODPS(
                            account=sts_account,
                            project=project,
                            endpoint=endpoint,
                            tunnel_endpoint=odps_kw["tunnel_endpoint"],
                        )
        finally:
            if not args.load_resource:
                os.remove(config_file)

        if args.chroot is not None:
            tmpdir = args.chroot or tempfile.mkdtemp()
            try:
                os.chroot(tmpdir)
            except Exception as e:
                logger.warning("Fail to chroot\n" + str(e))

        if args.load_resource:
            return_code = run_load_resource_command(args, odps)
        else:
            args.biz_id = os.environ.pop("biz_id", None)
            pyver_str = "%s.%s" % tuple(sys.version_info[:2])
            print(
                "Executing user script with PyODPS %s(W) under Python %s"
                % (pyodps_version, pyver_str)
            )
            return_code = call_in_process(args, odps)
        if sign_server:
            sign_server.stop()
        sys.exit(return_code)


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--access-id", dest="access_id", nargs=1, help="Access ID")
    parser.add_argument("--access-key", dest="access_key", nargs=1, help="Access key")
    parser.add_argument(
        "--security-token", dest="security_token", nargs=1, help="Security token"
    )
    parser.add_argument("--project", dest="project", nargs=1, help="Default project")
    parser.add_argument("--endpoint", dest="endpoint", nargs="?", help="Endpoint")
    parser.add_argument(
        "--tunnel-endpoint", dest="tunnel_endpoint", nargs="?", help="Tunnel endpoint"
    )
    parser.add_argument("--biz-id", dest="biz_id", nargs="?", help="biz id")

    parser.add_argument("--mem", dest="mem", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--mem-limit", dest="mem_limit", nargs="?", help="Memory limit")
    parser.add_argument(
        "--cpu", dest="cpu", nargs="?", type=float, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--cpu-timeout", dest="cpu_timeout", nargs="?", type=int, help=argparse.SUPPRESS
    )
    parser.add_argument("--code", dest="code", const="", nargs="?", help="code")
    parser.add_argument(
        "--code-file", dest="code_file", nargs="?", help="code file path"
    )
    parser.add_argument("--args", dest="args", nargs="*", help="args for user code")

    parser.add_argument("--user", dest="user", nargs="?", help="User to run code")
    parser.add_argument("--label", dest="label", nargs="?", help="Label to set cgroup")
    parser.add_argument("--chroot", dest="chroot", nargs="?", const="", help="chroot")

    parser.add_argument(
        "--self-check", default=False, action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--load-resource", help=argparse.SUPPRESS)
    return parser


def main(args_loader=None):
    try:
        import faulthandler

        faulthandler.enable(all_threads=True)
    except ImportError:
        pass

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    # disable charset_normalizer logs
    logging.getLogger("charset_normalizer").disabled = True
    # show all deprecation warnings of odps module
    warnings.filterwarnings(action="always", category=DeprecationWarning, module="odps")

    parser = get_arg_parser()
    args, _ = parser.parse_known_args()

    config_with_code_header(args)

    if not args.self_check:
        patch_compat()
        wrapper_main(args, args_loader=args_loader)
    else:
        import_all_sub_modules("odps")
        print(
            "Self-check of pyodpswrapper under py%s%s passed"
            % tuple(sys.version_info[:2])
        )


if __name__ == "__main__":
    main()
