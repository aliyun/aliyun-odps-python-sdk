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

import datetime
import os
import sys
import traceback

from ..envs import is_production
from ..utils import CODE_FILE_NAME, suspend_option_errors, truncate_traceback
from .messages import *

_tunnel_session_create_timeout_displayed = [False]
_change_time_threshold_days = 35


def _with_pyodps_team_code_path(path):
    team_paths = ["/odps/", "/mars/", "/maxframe", "/pyodpswrapper.py"]
    return any(p in path for p in team_paths)


def _is_pure_sql_execution_error_traceback(tb):
    from odps import core

    try:
        while tb.tb_next:
            next_tb = tb.tb_next
            tb_frame_code = next_tb.tb_frame.f_code
            if tb_frame_code.co_filename == core.__file__.replace(
                ".pyc", ".py"
            ) and tb_frame_code.co_name in ("run_sql", "execute_sql"):
                # need the stack prior to `run_sql` or `execute_sql` be user code
                if not _with_pyodps_team_code_path(tb.tb_frame.f_code.co_filename):
                    return True
                break
            tb = next_tb
    except:
        return False
    return False


def _judge_change_time(logger):
    try:
        import odps
        from odps import _version as version_mod

        if not is_production():
            return

        file_list = [
            odps.__file__,
            version_mod.__file__,
            __file__,
            os.path.join(os.path.dirname(__file__), "pyodps_commit_hash"),
            os.path.join(os.path.dirname(__file__), "wrapper_version"),
        ]
        change_ts = 0
        for file_path in file_list:
            if os.path.exists(file_path):
                change_ts = max(
                    change_ts, os.path.getctime(file_path), os.path.getmtime(file_path)
                )
        if not change_ts:
            return

        change_time = datetime.datetime.fromtimestamp(change_ts)
        if change_time < datetime.datetime.now() - datetime.timedelta(
            days=_change_time_threshold_days
        ):
            # in case some jobs are scheduled monthly
            msg = INSTALLATION_NOT_CHANGED_MSG.safe_replace(
                "<last_change_time>", change_time.isoformat()
            )
            logger.warning(msg)
    except:
        pass


def diagnose_pyodps_errors(code, exc, tb, logger):
    from odps.errors import InternalServerError, ODPSError, ScriptError

    try:
        from odps.errors import ParseError
    except ImportError:
        ParseError = None
    try:
        from odps.tunnel.errors import TunnelWriteTimeout
    except ImportError:
        TunnelWriteTimeout = None
    try:
        from odps.lib import requests
    except (AttributeError, ImportError, TypeError):
        import requests

    exc_str = str(exc)
    extra_help_msg = None
    truncated_tb = truncate_traceback(tb, CODE_FILE_NAME)

    if isinstance(exc, ODPSError):
        if (
            "is protected" in exc_str
            and "You have NO privilege 'odps:Select'" in exc_str
        ):
            extra_help_msg = PROJECT_PROTECT_PRIVILEGE_MSG
        elif "the page you are looking for is currentlv unavailable." in exc_str:
            extra_help_msg = INTERNAL_SERVER_ERROR_MSG
        elif "bizdate" in exc_str:
            extra_help_msg = SCHEDULE_ARGS_MSG
        elif isinstance(exc, ScriptError):
            if "No module named 'odps." in exc_str:
                extra_help_msg = ODPS_OBJ_CLOSURE_ERROR_MSG
            elif "ImportError" in exc_str or "ModuleNotFoundError" in exc_str:
                extra_help_msg = SCRIPT_MODULE_NOT_FOUND_ERROR_MSG
                if "@resource_reference" in code:
                    extra_help_msg += RESOURCE_REFERENCE_USED_MSG
            elif (
                "TypeError" in exc_str
                and "'generator' object is not callable" in exc_str
            ):
                extra_help_msg = SCRIPT_GENERATOR_NOT_CALLABLE_ERROR_MSG
            else:
                extra_help_msg = SCRIPT_ERROR_MSG
        elif isinstance(exc, InternalServerError) or "System internal error" in exc_str:
            extra_help_msg = INTERNAL_SERVER_ERROR_MSG
        elif TunnelWriteTimeout is not None and isinstance(exc, TunnelWriteTimeout):
            extra_help_msg = REQUESTS_TIMEOUT_MSG
        elif (
            ParseError is not None
            and isinstance(exc, ParseError)
            and _is_pure_sql_execution_error_traceback(truncated_tb)
        ):
            extra_help_msg = PURE_SQL_PARSE_ERROR
    elif isinstance(exc, ImportError):
        extra_help_msg = NO_LOCAL_IMPORT_MSG
        if "@resource_reference" in code:
            extra_help_msg += RESOURCE_REFERENCE_USED_MSG
    elif isinstance(exc, AttributeError) and exc_str.endswith(
        "'DataFrame' object has no attribute 'persist'"
    ):
        extra_help_msg = DATAFRAME_NO_PERSIST_MSG
    elif isinstance(exc, (requests.ReadTimeout, requests.ConnectTimeout)) or (
        isinstance(exc, requests.ConnectionError) and "timed out" in exc_str
    ):
        if not _tunnel_session_create_timeout_displayed[0]:
            extra_help_msg = REQUESTS_TIMEOUT_MSG
    elif "bizdate" in exc_str and isinstance(exc, ValueError):
        extra_help_msg = SCHEDULE_ARGS_MSG
    elif isinstance(exc, UnicodeError):
        extra_help_msg = ENCODING_ERROR_MSG
    elif isinstance(exc, OSError) and "No such file or directory" in exc_str:
        extra_help_msg = NO_LOCAL_IMPORT_MSG
    elif isinstance(exc, TypeError) and "Unknown dtype: object" in exc_str:
        extra_help_msg = DATAFRAME_UNKNOWN_DTYPE_MSG
    elif isinstance(exc, SyntaxError):
        extra_help_msg = SYNTAX_ERROR_MSG
    elif isinstance(exc, NameError):
        extra_help_msg = NAME_ERROR_MSG
    elif not _with_pyodps_team_code_path("".join(traceback.format_tb(truncated_tb))):
        extra_help_msg = TRACEBACK_NO_PYODPS_MSG

    traceback.print_exception(type(exc), exc, truncated_tb)
    if extra_help_msg:
        logger.warning(str(extra_help_msg))
    _judge_change_time(logger)


def diagnose_exit_code(exitcode, logger):
    if exitcode == -9:  # SIGKILL
        logger.error("Got killed (%s)" % exitcode)
        logger.error(OOM_KILL_MSG)
    else:
        logger.error("Exited accidentally with exit code %s" % exitcode)
        logger.error(OTHER_KILL_MSG)


def config_warning_messages(logger):
    from odps import options

    def _show_read_timeout_message(*_):
        if not _tunnel_session_create_timeout_displayed[0]:
            logger.error(TUNNEL_TIMEOUT_MSG)

    def _show_tunnel_session_create_timeout_message(*_):
        _tunnel_session_create_timeout_displayed[0] = True
        logger.error(TUNNEL_SESSION_CREATE_TIMEOUT_MSG)

    def _on_create_tunnel_session(tunnel_obj):
        from odps.tunnel.instancetunnel import InstanceDownloadSession

        if (
            not is_production()
            and isinstance(tunnel_obj, InstanceDownloadSession)
            and getattr(tunnel_obj, "_limit_enabled", False)
        ):
            logger.warning(INSTANCE_TUNNEL_LIMIT_MSG)

    def _on_create_result_reader(_reader_obj):
        if not is_production():
            logger.warning(RESULT_READER_CREATE_MSG)

    with suspend_option_errors():
        options.tunnel_session_create_callback = _on_create_tunnel_session
        options.tunnel_session_create_timeout_callback = (
            _show_tunnel_session_create_timeout_message
        )
        options.tunnel_read_timeout_callback = _show_read_timeout_message
        options.result_reader_create_callback = _on_create_result_reader


def log_replaced_vars(var_names, logger):
    if not var_names:
        return
    msg = BUILTIN_OBJECT_REPLACED_MSG.safe_replace(
        "{REPLACED_VARS}", ", ".join(var_names)
    )
    logger.warning(msg)


def log_python2_deprecation_warning(logger):
    if sys.version_info[0] == 2:
        logger.warning(str(PYTHON2_DEPRECATE_MSG))


def log_dev_project_warning(project_name, logger):
    if not is_production() and project_name.endswith("_dev"):
        logger.warning(
            DEV_PROJECT_MSG.safe_replace("<dw_project_name>", project_name[:-4])
        )


def log_reload_usage_warning(code, logger):
    if "reload(sys)" in code:
        logger.warning(str(RELOAD_SYS_WARN_MSG))
