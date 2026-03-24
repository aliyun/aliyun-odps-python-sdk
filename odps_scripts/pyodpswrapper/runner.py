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
import cProfile
import datetime
import io
import linecache
import logging
import multiprocessing
import os
import pickle
import pstats
import re
import signal
import sys
import threading
import time
import traceback
import types
import warnings

from .diagnose import (
    config_warning_messages,
    diagnose_pyodps_errors,
    log_dev_project_warning,
    log_python2_deprecation_warning,
    log_reload_usage_warning,
    log_replaced_vars,
)
from .envs import *
from .headers import get_run_flags, use_spawn_method
from .resource import load_resource_package
from .utils import (
    CODE_FILE_NAME,
    import_all_sub_modules,
    suspend_option_errors,
    truncate_traceback,
)

logger = logging.getLogger("odps.pyodpswrapper")


class GlobalCopyDict(dict):
    """
    Automatically copy user globals to __main__ to make them available
    to pickle when called in exec method.
    """

    def __init__(self, wrapped, **kw):
        from odps.compat import builtins

        dict.__init__(self, **kw)
        self.update(wrapped)
        self._globals = globals()
        self._global_vars = set(globals().keys()) | set(dir(builtins))
        if not use_spawn_method():
            self._mp_main_mod = __import__("__main__")
        else:
            self._mp_main_mod = __import__("__mp_main__")
            self._global_vars |= set(dir(self._mp_main_mod))
        self._new_keys = set()

    @staticmethod
    def _patch_module(obj):
        """
        Reduce side effects of bpo-42406 when module is not specified
        by manually setting __module__ before it is injected into __mp_main__
        """
        try:
            if not hasattr(pickle, "whichmodule"):
                return
            if (
                not hasattr(obj, "__name__")
                or not hasattr(obj, "__module__")
                or obj.__module__ is not None
            ):
                return

            obj.__module__ = pickle.whichmodule(obj, obj.__name__)
        except:
            # all errors should be ignored
            pass

    def __setitem__(self, key, value):
        self._patch_module(value)
        dict.__setitem__(self, key, value)
        if key not in self._global_vars:
            self._globals[key] = value
            if self._mp_main_mod is not None:
                try:
                    setattr(self._mp_main_mod, key, value)
                except AttributeError:
                    pass
            self._new_keys.add(key)
        elif key in self._globals and self._globals[key] is not value:
            warnings.warn(
                "Global variable %s you are about to set conflicts "
                "with pyodpswrapper or builtin variables. "
                "It might not be runnable with multiprocessing." % key
            )

    def __delitem__(self, key):
        # remove in globals also to avoid refcount issue
        dict.__delitem__(self, key)
        if key in self._new_keys:
            self._globals.pop(key, None)
            if self._mp_main_mod is not None:
                try:
                    delattr(self._mp_main_mod, key)
                except AttributeError:
                    pass

    def check_changes(self, raw_locals):
        var_names = []
        raw_locals.pop("odps", None)
        raw_locals.pop("__doc__", None)
        for var_name in raw_locals:
            if var_name in self:
                raw_local_val = raw_locals[var_name]
                actual_local_val = self[var_name]
                if raw_local_val is not actual_local_val:
                    if (
                        isinstance(raw_local_val, types.ModuleType)
                        and isinstance(actual_local_val, types.ModuleType)
                        and raw_local_val.__file__ == actual_local_val.__file__
                    ):
                        continue
                    var_names.append(var_name)
        log_replaced_vars(var_names, logger)

    def erase_copied(self):
        for k in self._new_keys:
            self._globals.pop(k, None)
            if self._mp_main_mod is not None:
                try:
                    delattr(self._mp_main_mod, k)
                except AttributeError:
                    pass


@contextlib.contextmanager
def build_user_locals(odps, args):
    from odps import DataFrame, NullScalar, Scalar, options
    from odps.compat import builtins

    if args.args is not None:
        args_dict = dict([a.split("=", 1) for a in args.args if "=" in a])
    else:
        args_dict = dict()

    # customize reload
    if sys.version_info[0] < 3:
        _old_reload = builtins.reload
    else:
        try:
            import importlib

            _old_reload = importlib.reload
        except ImportError:
            import imp

            _old_reload = imp.reload

    def _custom_reload(mod):
        _stdout = sys.stdout
        _stderr = sys.stderr
        _old_reload(mod)
        sys.stdout = _stdout
        sys.stderr = _stderr

    if sys.version_info[0] >= 3:
        try:
            import importlib

            importlib.reload = _custom_reload
        except ImportError:
            pass

    def user_load_resource_package(name, odps_entry=None, project=None, supersede=None):
        """
        Load a Python package from MaxCompute resource. Note that you need to
        have right to read the resource, and the resource should be a wheel.

        :param name: Name of the package
        :param odps_entry: ODPS entry object
        :param project: ODPS project of the resource
        :param supersede: Supersede default libraries
        """
        odps_entry = odps_entry or odps
        supersede = (
            supersede if supersede is not None else options.df.supersede_libraries
        )
        return load_resource_package(
            name, odps_entry, project=project, supersede=supersede
        )

    odps.to_global()
    local = {
        "odps": odps,
        "o": odps,
        "DataFrame": DataFrame,
        "Scalar": Scalar,
        "NullScalar": NullScalar,
        "options": options,
        "args": args_dict,
        "load_resource_package": user_load_resource_package,
        "__name__": "__main__",
    }
    if sys.version_info[0] <= 2:
        local["reload"] = _custom_reload
    local = GlobalCopyDict(local)

    local_and_builtin_copy = builtins.__dict__.copy()
    local_and_builtin_copy.update(local)

    try:
        yield local
    except:
        local.check_changes(local_and_builtin_copy)
        raise
    finally:
        local.erase_copied()


@contextlib.contextmanager
def enable_dumper_thread(run_flags):
    try:
        import faulthandler
    except ImportError:
        faulthandler = None

    dump_thread_running = [True]

    def _dumper_thread():
        while dump_thread_running[0]:
            time.sleep(30)
            logger.warning(
                "Periodical stack dump at %s",
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            faulthandler.dump_traceback(all_threads=True)

    if faulthandler and run_flags.get("dump_traceback"):
        dumper_thread = threading.Thread(target=_dumper_thread, daemon=True)
        dumper_thread.start()

    try:
        yield
    finally:
        dump_thread_running[0] = False


@contextlib.contextmanager
def enable_profiling(run_flags):
    if run_flags.get("profile"):
        profiler = cProfile.Profile()
        profiler.enable()
    else:
        profiler = None

    try:
        yield

        if profiler is not None:
            p = pstats.Stats(profiler)
            p.strip_dirs()
            p.sort_stats("cumtime")
            p.print_stats(40)
    except:
        raise
    finally:
        if profiler is not None:
            profiler.disable()


def patch_linecache():
    _old_checkcache = linecache.checkcache

    def _pyodps_checkcache(filename):
        if filename == CODE_FILE_NAME:
            return
        else:
            _old_checkcache(filename)

    _pyodps_checkcache._pyodps_wrapped = True
    if not getattr(linecache.checkcache, "_pyodps_wrapped", False):
        linecache.checkcache = _pyodps_checkcache


def _set_maxframe_options():
    try:
        from maxframe.config import options as mf_options

        # set maxframe logger level to show logview address automatically
        mf_client_logger = logging.getLogger("maxframe_client")
        mf_client_logger.setLevel(logging.INFO)

        if is_production:
            mf_options.is_production = is_production
            mf_options.schedule_id = os.environ.get("SKYNET_ID")
    except:
        pass


def set_running_options(odps, args):
    from odps import options
    from odps.config import OptionError

    try:
        from odps.config import reset_global_options
    except ImportError:
        reset_global_options = None

    cur_pid = os.getpid()
    instances = []

    def append_instance(instance_id):
        instances.append(instance_id)

    def kill_instances(*_):
        for i in instances:
            try:
                odps.stop_instance(i)
            except:
                continue
        # check to avoid unexpected stack
        if cur_pid == os.getpid():
            sys.exit(-1)

    # set signals
    signal.signal(signal.SIGTERM, kill_instances)
    signal.signal(signal.SIGINT, kill_instances)

    if reset_global_options is not None:
        reset_global_options()

    options.verbose = True
    options.instance_create_callback = append_instance
    options.tunnel.use_instance_tunnel = False
    options.tunnel.limit_instance_tunnel = True
    options.biz_id = args.biz_id
    try:
        options.logview_hours = 7 * 24
    except OptionError:
        options.log_view_hours = 7 * 24

    if PRIORITY in os.environ:
        options.priority = get_priority()

    config_warning_messages(logger)

    def _use_legacy_parsedate_validator(value):
        if not value:
            return True
        re_legacy_date = re.compile(r"odps\.[^:]+:legacy_parsedate")
        if re_legacy_date not in options.skipped_survey_regexes:
            options.skipped_survey_regexes.append(re_legacy_date)
        return True

    def _struct_as_dict_validator(value):
        if not value:
            return True
        re_struct = re.compile(r"options\.struct_as_dict")
        if re_struct not in options.skipped_survey_regexes:
            options.skipped_survey_regexes.append(re_struct)
        return True

    with suspend_option_errors():
        options.verify_ssl = False
    with suspend_option_errors():
        options.upload_resource_in_chunks = False
    with suspend_option_errors():
        options.sqlalchemy.project_as_schema = True
    with suspend_option_errors():
        options.sql.ignore_fields_not_null = True
    with suspend_option_errors():
        options.sql.parse_set_as_hints = False
    # legacy compatibility options
    with suspend_option_errors():
        options.use_legacy_parsedate = True
        options.add_validator("use_legacy_parsedate", _use_legacy_parsedate_validator)
    with suspend_option_errors():
        options.struct_as_dict = True
        options.add_validator("struct_as_dict", _struct_as_dict_validator)
    with suspend_option_errors():
        options.struct_as_ordered_dict = True
    with suspend_option_errors():
        options.map_as_ordered_dict = True

    with suspend_option_errors():
        if not is_internal() and RUNNING_QUOTA_ENV in os.environ:
            options.quota_name = os.environ[RUNNING_QUOTA_ENV]
    with suspend_option_errors():
        if is_internal():
            options.use_legacy_logview = True

    set_skynet_to_odps_options()
    _set_maxframe_options()


def run_code(odps, args, uid=None, conn=None):
    from odps.compat import StringIO, six

    # limit_resource(args.cpu_timeout, args.mem or args.mem_limit)
    import_all_sub_modules("odps")
    run_flags = get_run_flags()

    patch_linecache()

    # https://stackoverflow.com/questions/107705/disable-output-buffering
    # set STDOUT to unbuffer mode
    try:
        # Python 3, open as binary, then wrap in a TextIOWrapper, and write through
        # everything. Alternatively, use line_buffering=True to flush on newlines.
        sys.stdout = io.TextIOWrapper(
            open(sys.stdout.fileno(), "wb", 0), write_through=True
        )
    except TypeError:
        # Python 2
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 0)

    if uid is not None:
        os.setuid(uid)
    if args.code_file is not None:
        with open(args.code_file) as f:
            code = f.read()
        sys.argv = [args.code_file] + (args.args if args.args is not None else [])
    else:
        code = args.code
    code = code or ""

    if conn:
        conn.recv()  # block until the signal to run received

    # replace sensitive environs
    tailored_env = os.environ.copy()
    tailored_env.pop(ACCESS_KEY_ENV, None)
    tailored_env.pop(ACCESS_KEY_ENCRYPT, None)
    tailored_env.pop(PICKLE_ACCOUNT, None)
    os.environ = tailored_env

    class Empty(object):
        pass

    sys.modules["resource"] = Empty

    set_running_options(odps, args)

    log_python2_deprecation_warning(logger)
    log_dev_project_warning(odps.project, logger)
    log_reload_usage_warning(code, logger)

    try:
        lines = StringIO(code).readlines()
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        linecache.cache[CODE_FILE_NAME] = (
            len(code),
            datetime.datetime.now(),
            lines,
            CODE_FILE_NAME,
        )

        compiled = compile(code, CODE_FILE_NAME, "exec")

        if use_spawn_method():
            old_start_method = multiprocessing.get_start_method(True)
        else:
            old_start_method = None

        try:
            if use_spawn_method():
                # set start method as fork to make sure mp works for user code
                multiprocessing.set_start_method("fork", force=True)

            with enable_profiling(run_flags), enable_dumper_thread(
                run_flags
            ), build_user_locals(odps, args) as local:
                six.exec_(compiled, local)
        finally:
            if use_spawn_method() and old_start_method:
                multiprocessing.set_start_method(old_start_method, force=True)

        if conn:
            conn.send(0)
    except SystemExit as ex:
        ret_code = 1 if ex.code else 0
        if ret_code:
            exc_type, exc, tb = sys.exc_info()
            traceback.print_exception(
                exc_type, exc, truncate_traceback(tb, CODE_FILE_NAME)
            )
        if conn:
            conn.send(ret_code)
    except Exception:
        exc_type, exc, tb = sys.exc_info()
        diagnose_pyodps_errors(code, exc, tb, logger)
        if conn:
            conn.send(1)
    finally:
        if conn:
            conn.close()
