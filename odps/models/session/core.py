#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from ... import utils
from ...compat import six
from .v1 import McqaV1Methods
from .v2 import McqaV2Methods


def _redirect_v1(func):
    @six.wraps(func)
    def wrapped(*args, **kwargs):
        meth = getattr(McqaV1Methods, func.__name__)
        if args and isinstance(args[0], type):
            args = args[1:]
        return meth(*args, **kwargs)

    return wrapped


class SessionMethods(object):
    @classmethod
    @utils.deprecated(
        "You no longer have to manipulate session instances to use MaxCompute QueryAcceleration. "
        "Try `run_sql_interactive`."
    )
    @_redirect_v1
    def attach_session(cls, odps, session_name, taskname=None, hints=None):
        """
        Attach to an existing session.

        :param session_name: The session name.
        :param taskname: The created sqlrt task name. If not provided, the default value is used.
            Mostly doesn't matter, default works.
        :return: A SessionInstance you may execute select tasks within.
        """
        pass

    @classmethod
    @_redirect_v1
    def _attach_mcqa_session(cls, odps, session_name=None, task_name=None, hints=None):
        pass

    @classmethod
    @utils.deprecated(
        "You no longer have to manipulate session instances to use MaxCompute QueryAcceleration. "
        "Try `run_sql_interactive`."
    )
    @_redirect_v1
    def default_session(cls, odps):
        """
        Attach to the default session of your project.

        :return: A SessionInstance you may execute select tasks within.
        """
        pass

    @classmethod
    @_redirect_v1
    def _get_default_mcqa_session(
        cls, odps, session_name=None, hints=None, wait=True, service_startup_timeout=60
    ):
        pass

    @classmethod
    @utils.deprecated(
        "You no longer have to manipulate session instances to use MaxCompute QueryAcceleration. "
        "Try `run_sql_interactive`."
    )
    @_redirect_v1
    def create_session(
        cls,
        odps,
        session_worker_count,
        session_worker_memory,
        session_name=None,
        worker_spare_span=None,
        taskname=None,
        hints=None,
    ):
        """
        Create session.

        :param session_worker_count: How much workers assigned to the session.
        :param session_worker_memory: How much memory each worker consumes.
        :param session_name: The session name. Not specifying to use its ID as name.
        :param worker_spare_span: format "00-24", allocated workers will be reduced during this time.
            Not specifying to disable this.
        :param taskname: The created sqlrt task name. If not provided, the default value is used.
            Mostly doesn't matter, default works.
        :param hints: Extra hints provided to the session. Parameters of this method will override
            certain hints.
        :return: A SessionInstance you may execute select tasks within.
        """
        pass

    @classmethod
    @_redirect_v1
    def _create_mcqa_session(
        cls,
        odps,
        session_worker_count,
        session_worker_memory,
        session_name=None,
        worker_spare_span=None,
        task_name=None,
        hints=None,
    ):
        pass

    @classmethod
    def _get_mcqa_v2_quota_name(cls, odps, hints=None, quota_name=None):
        hints = hints or {}
        return quota_name or hints.get("odps.task.wlm.quota") or odps.quota_name

    @classmethod
    def run_sql_interactive(cls, odps, sql, hints=None, use_mcqa_v2=False, **kwargs):
        """
        Run SQL query in interactive mode (a.k.a MaxCompute QueryAcceleration).
        Won't fallback to offline mode automatically if query not supported or fails

        :param sql: the sql query.
        :param hints: settings for sql query.
        :return: instance.
        """
        if use_mcqa_v2:
            quota_name = cls._get_mcqa_v2_quota_name(
                odps, hints=hints, quota_name=kwargs.pop("quota_name", None)
            )
            return McqaV2Methods.run_sql_interactive(
                odps, sql, hints=hints, quota_name=quota_name, **kwargs
            )
        return McqaV1Methods.run_sql_interactive(odps, sql, hints=hints, **kwargs)

    @classmethod
    @utils.deprecated(
        "The method `run_sql_interactive_with_fallback` is deprecated. "
        "Try `execute_sql_interactive` with fallback=True argument instead."
    )
    def run_sql_interactive_with_fallback(cls, odps, sql, hints=None, **kwargs):
        return cls.execute_sql_interactive(
            odps, sql, hints=hints, fallback="all", wait_fallback=False, **kwargs
        )

    @classmethod
    def execute_sql_interactive(
        cls,
        odps,
        sql,
        hints=None,
        fallback=True,
        wait_fallback=True,
        offline_quota_name=None,
        use_mcqa_v2=False,
        **kwargs
    ):
        """
        Run SQL query in interactive mode (a.k.a MaxCompute QueryAcceleration).
        If query is not supported or fails, and fallback is True,
        will fallback to offline mode automatically

        :param sql: the sql query.
        :param hints: settings for sql query.
        :param fallback: fallback query to non-interactive mode, True by default.
            Both boolean type and policy names separated by commas are acceptable.
        :param bool wait_fallback: wait fallback instance to finish, True by default.
        :return: instance.
        """
        if use_mcqa_v2:
            quota_name = cls._get_mcqa_v2_quota_name(
                odps, hints=hints, quota_name=kwargs.pop("quota_name", None)
            )
            return McqaV2Methods.execute_sql_interactive(
                odps, sql, hints=hints, quota_name=quota_name, **kwargs
            )
        return McqaV1Methods.execute_sql_interactive(
            odps,
            sql,
            hints=hints,
            fallback=fallback,
            wait_fallback=wait_fallback,
            offline_quota_name=offline_quota_name,
            **kwargs
        )
