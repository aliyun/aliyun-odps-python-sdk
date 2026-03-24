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

import contextlib
import logging
import os
import sys

logger = logging.getLogger(__name__)
CODE_FILE_NAME = "<pyodps_user_code>"


def import_all_sub_modules(mod_name):
    """
    Make sure all submodules imported to avoid conflicts caused by installation
    """
    try:
        mod = __import__(mod_name)
    except:
        return
    mod_path = os.path.dirname(mod.__file__)
    mod_parent = os.path.dirname(mod_path)
    for root, _dirs, files in os.walk(mod_path):
        skip_part = ("tests", "mars_extension", "lib")
        if any((os.path.sep + d) in root for d in skip_part):
            continue
        for filename in files:
            try:
                if not filename.endswith(".py") and not filename.endswith(".pyx"):
                    continue
                file_dir = os.path.relpath(os.path.join(root, filename), mod_parent)
                if filename == "__init__.py":
                    file_dir = os.path.dirname(file_dir).rstrip(os.path.sep)
                else:
                    file_dir = file_dir.rsplit(".", 1)[0]
                sub_mod_name = file_dir.replace(os.path.sep, ".")
                if sub_mod_name in sys.modules:
                    continue
                __import__(sub_mod_name)
            except:
                pass


def truncate_traceback(tb, filename):
    cur_tb = tb
    while cur_tb:
        try:
            frame_file = cur_tb.tb_frame.f_code.co_filename
            if frame_file == filename:
                break
        except:
            cur_tb = None
            break
        cur_tb = cur_tb.tb_next
    if cur_tb is None:
        return tb
    else:
        return cur_tb


@contextlib.contextmanager
def suspend_option_errors():
    from odps.config import OptionError

    try:
        yield
    except (AttributeError, OptionError):
        pass


def is_limit_exceeded(mem_limit):
    import psutil

    mem_units = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    mem_limit = mem_limit.strip().lower()
    if mem_limit[-1] not in mem_units:
        mem_limit_bytes = int(mem_limit)
    else:
        mem_limit_bytes = int(mem_limit[:-1].strip()) * mem_units[mem_limit[-1]]

    cur_proc = psutil.Process()
    total_rss = cur_proc.memory_info().rss
    for sub_proc in cur_proc.children(recursive=True):
        try:
            total_rss += sub_proc.memory_info().rss
        except psutil.NoSuchProcess:
            pass

    if total_rss >= mem_limit_bytes:
        logger.fatal(
            "Actual memory usage (%d) exceeds memory limit (%s), "
            "will kill node. This limitation cannot be lifted.",
            total_rss,
            mem_limit,
        )
        return True
    return False


def kill_process_tree(pid, signum):
    import psutil

    from odps.compat import futures

    def _kill(process, signum):
        try:
            process.send_signal(signum)
        except psutil.NoSuchProcess:
            logger.error("No such process, ignore it")
        except:
            logger.exception("Fail to kill process")

    try:
        p = psutil.Process(pid)

        tpe = futures.ThreadPoolExecutor(max_workers=1)
        fut = tpe.submit(lambda: list(p.children(recursive=True)))

        try:
            for child in fut.result(10):
                _kill(child, signum)
        except futures.TimeoutError:
            logger.error(
                "Timeout when fetching subprocess of %s, stop killing children", pid
            )
        _kill(p, signum)
    except psutil.NoSuchProcess:
        logger.error("No such process, skip kill")
