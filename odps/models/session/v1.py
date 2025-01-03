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

import copy
import glob
import hashlib
import itertools
import json
import os
import re
import sys
import time
import warnings

from ... import errors, readers, utils
from ...compat import enum, six
from ...lib.monotonic import monotonic
from .. import tasks
from ..instance import Instance, InstanceArrowReader, InstanceRecordReader

DEFAULT_TASK_NAME = "AnonymousSQLRTTask"
PUBLIC_SESSION_NAME = "public.default"
_SUBQUERY_ID_PATTERN = re.compile(r"[\d\D]*_query_(\d+)_[\d\D]*")
_SESSION_FILE_PREFIX = "mcqa-session-"
_SESSION_FILE_EXPIRE_TIME = 3600 * 24


class FallbackMode(enum.Enum):
    OFFLINE = 0
    INTERACTIVE = 1


class FallbackPolicy:
    def __init__(
        self,
        policy=None,
        always=False,
        noresource=False,
        unsupported=False,
        timeout=False,
        upgrading=False,
        generic=False,
    ):
        policies = set()
        if policy:
            if isinstance(policy, (set, list, tuple)):
                policies = set(policy)
            else:
                policy = policy.lower().strip()
                if policy == "default":
                    policy = "unsupported,upgrading,noresource,timeout"
                elif policy == "all":
                    always = True
                policies = set(s.strip() for s in policy.split(","))
        self.always = always
        self.noresource = noresource or always or "noresource" in policies
        self.unsupported = unsupported or always or "unsupported" in policies
        self.timeout = timeout or always or "timeout" in policies
        self.upgrading = upgrading or always or "upgrading" in policies
        self.generic = generic or always or "generic" in policies

    def get_mode_from_exception(self, exc_value):
        err_msg = str(exc_value)
        if isinstance(exc_value, errors.SQARetryError):
            return FallbackMode.INTERACTIVE
        elif "OdpsJobCancelledException" in err_msg or "Job is cancelled" in err_msg:
            return None
        elif self.always:
            return FallbackMode.OFFLINE
        elif self.unsupported and isinstance(exc_value, errors.SQAUnsupportedFeature):
            return FallbackMode.OFFLINE
        elif self.upgrading and isinstance(
            exc_value, (errors.SQAServiceUnavailable, errors.SQAAccessDenied)
        ):
            return FallbackMode.OFFLINE
        elif self.noresource and isinstance(exc_value, errors.SQAResourceNotEnough):
            return FallbackMode.OFFLINE
        elif self.timeout and (
            isinstance(exc_value, errors.SQAQueryTimedout)
            or "Wait for cache data timeout" in err_msg
            or "Get select desc from SQLRTTask timeout" in err_msg
        ):
            return FallbackMode.OFFLINE
        elif self.generic and isinstance(exc_value, errors.SQAGenericError):
            return FallbackMode.OFFLINE
        return None

    def __repr__(self):
        policies = [
            s
            for s in ["generic", "unsupported", "upgrading", "noresource", "timeout"]
            if getattr(self, s, None)
        ]
        return "<FallbackPolicy %s>" % ",".join(policies)


@enum.unique
class SessionTaskStatus(enum.Enum):
    """
    Possible statuses of tasks executing inside a session.
    """

    Running = 2
    Failed = 4
    Terminated = 5
    Cancelled = 6
    Unknown = -1


TASK_STATUS_VALUES = {
    2: SessionTaskStatus.Running,
    4: SessionTaskStatus.Failed,
    5: SessionTaskStatus.Terminated,
    6: SessionTaskStatus.Cancelled,
}


def _task_status_value_to_enum(task_status):
    return TASK_STATUS_VALUES.get(task_status, SessionTaskStatus.Unknown)


def _get_session_failure_info(task_results):
    try:
        taskname, result_txt = list(task_results.items())[0]
    except BaseException:
        return ""
    return result_txt


class SessionInstance(Instance):
    """
    This represents the instance
    created right after you call 'attach_session' or 'create_session'.
    Further SQL tasks has to be created using this instance.
    """

    __slots__ = ("_project", "_task_name", "_session_name")

    def __init__(self, **kw):
        if "session_task_name" not in kw or "session_project" not in kw:
            raise errors.InvalidArgument(
                "Creating InSessionInstance without enough information."
            )
        self._task_name = kw.pop("session_task_name", "")
        self._project = kw.pop("session_project", None)
        self._session_name = kw.pop("session_name", "")
        super(SessionInstance, self).__init__(**kw)

    @classmethod
    def from_instance(cls, instance, **kw):
        return SessionInstance(
            name=instance.id, parent=instance.parent, client=instance._client, **kw
        )

    def _extract_json_info(self):
        return {
            "id": self.id,
            "session_project_name": self._project.name,
            "session_task_name": self._task_name,
            "session_name": self._session_name,
        }

    def wait_for_startup(self, interval=1, timeout=-1, retry=True, max_interval=None):
        """
        Wait for the session to startup(status changed to RUNNING).

        :param interval: time interval to check (unit seconds)
        :param timeout: wait timeout (unit seconds), < 0 means no timeout
        :param retry: if failed to query session status, should we retry silently
        :raise: :class:`odps.errors.WaitTimeoutError` if wait timeout and session is not started.
        :return: None
        """
        start_time = monotonic()
        end_time = start_time + timeout
        while not self.is_running(retry):
            if timeout > 0:
                if monotonic() > end_time:
                    raise errors.WaitTimeoutError(
                        "Waited %.1f seconds, but session is not started."
                        % (monotonic() - start_time),
                        instance_id=self.id,
                    )
            try:
                time.sleep(interval)
                if max_interval:
                    interval = min(interval * 2, max_interval)
            except KeyboardInterrupt:
                return

    def _parse_result_session_name(self, result_str):
        if not self._session_name:
            session_name_search = re.search("Session name: (.*)$", result_str)
            if session_name_search:
                self._session_name = session_name_search.group(1)

    def run_sql(self, sql, hints=None, **kwargs):
        task = tasks.SQLTask(query=utils.to_text(sql), **kwargs)
        task.update_sql_settings(hints)
        return self._create_internal_instance(task=task)

    @staticmethod
    def _check_is_select(sql_statement):
        try:
            splited = utils.split_sql_by_semicolon(sql_statement)
        except Exception as ex:
            warnings.warn(
                "Cannot split sql statement %s: %s" % (sql_statement, str(ex)),
                RuntimeWarning,
            )
            return False
        return splited[-1].lower().strip(" \t\r\n(").startswith("select")

    def _create_internal_instance(self, task=None):
        project_name = self._project.name
        is_select = self._check_is_select(task.query.strip())

        self.parent._fill_task_properties(task)
        rquery = task.query
        if not rquery.endswith(";"):
            rquery = rquery + ";"
        query_object = {
            "query": rquery,
            "settings": json.loads(task.properties["settings"]),
        }
        query_json = json.dumps(query_object)

        resp_content = self.put_task_info(
            self._task_name, "query", query_json, check_location=True
        )

        created_subquery_id = -1
        try:
            query_result = json.loads(resp_content)
            query_status = query_result["status"]
            if query_status != "ok":
                raise errors.ODPSError(
                    "Failed to run subquery: [%s]: %s"
                    % (query_status, query_result["result"])
                )
            query_subresult = json.loads(query_result["result"])
            created_subquery_id = query_subresult["queryId"]
            if created_subquery_id == -1:
                raise errors.parse_instance_error(query_subresult)
        except KeyError as ex:
            six.raise_from(
                errors.ODPSError(
                    "Invalid Response Format: %s\n Response JSON:%s\n"
                    % (str(ex), resp_content.decode())
                ),
                None,
            )
        instance = InSessionInstance(
            session_project_name=project_name,
            session_task_name=self._task_name,
            name=self.id,
            session_subquery_id=created_subquery_id,
            session_instance=self,
            parent=self.parent,
            session_is_select=is_select,
            client=self._client,
        )
        return instance

    def reload(self, blocking=False):
        resp_text = self.get_task_info(self._task_name, "status")
        session_status = SessionTaskStatus.Unknown
        try:
            poll_result = json.loads(resp_text)
            self._parse_result_session_name(poll_result["result"])
            session_status = _task_status_value_to_enum(poll_result["status"])
        except BaseException:
            error_string = _get_session_failure_info(self.get_task_results())
            if error_string:
                self._status = Instance.Status.TERMINATED
                six.raise_from(errors.parse_instance_error(error_string), None)
            else:
                # this is a task meta info update problem. Just retry.
                self._status = Instance.Status.SUSPENDED
                return
        if session_status == SessionTaskStatus.Running:
            self._status = Instance.Status.RUNNING
        elif session_status == SessionTaskStatus.Cancelled:
            error_string = _get_session_failure_info(self.get_task_results())
            self._status = Instance.Status.TERMINATED
            raise errors.parse_instance_error(error_string)
        elif session_status == SessionTaskStatus.Failed:
            error_string = _get_session_failure_info(self.get_task_results())
            self._status = Instance.Status.TERMINATED
            raise errors.parse_instance_error(error_string)
        elif poll_result["status"] == SessionTaskStatus.Terminated:
            self._status = Instance.Status.TERMINATED
            raise errors.ODPSError("Session terminated.")
        else:
            self._status = Instance.Status.SUSPENDED


class InSessionTunnelReaderMixin(object):
    @property
    def schema(self):
        # is not available before open_reader().
        if self._download_session.schema is None:
            # open reader once to enforce schema fetched.
            tmprd = self._download_session.open_record_reader(0, 1)
            tmprd.close()
        return self._download_session.schema

    @property
    def count(self):
        # we can't count session results before it's
        # fully retrieved.
        return -1

    @property
    def status(self):
        # force reload to update download session status
        # this is for supporting the stream download of instance tunnel
        # without the following line will not trigger reload
        self._download_session.reload()
        return self._download_session.status

    def read(
        self, start=None, count=None, step=None, compress=False, columns=None, **kw
    ):
        start = start or 0
        step = step or 1
        stop = None if count is None else start + step * count

        with self._download_session.open_record_reader(
            0, 1, compress=compress, columns=columns
        ) as reader:
            for record in itertools.islice(reader, start, stop, step):
                yield record


class InSessionInstanceArrowReader(InSessionTunnelReaderMixin, InstanceArrowReader):
    pass


class InSessionInstanceRecordReader(InSessionTunnelReaderMixin, InstanceRecordReader):
    pass


class InSessionInstance(Instance):
    """
    This represents the instance created
    for SQL tasks that run inside a session. This instance is useful
    when fetching results.
    """

    __slots__ = (
        "_project_name",
        "_session_task_name",
        "_session",
        "_session_instance",
        "_is_select",
        "_subquery_id",
        "_report_result",
        "_report_warning",
        "_session_task_status",
        "_task_data",
    )

    def __init__(self, **kw):
        if (
            "session_task_name" not in kw
            or "session_project_name" not in kw
            or "session_instance" not in kw
            or "session_subquery_id" not in kw
        ):
            raise errors.InvalidArgument(
                "Creating InSessionInstance without enough information."
            )
        self._session_task_name = kw.pop("session_task_name", "")
        self._project_name = kw.pop("session_project_name", "")
        self._session_instance = kw.pop("session_instance", None)
        self._is_select = kw.pop("session_is_select", False)
        self._subquery_id = kw.pop("session_subquery_id", -1)
        self._report_result = ""
        self._report_warning = ""
        self._session_task_status = -1
        self._task_data = None
        if self._subquery_id < 0:
            raise errors.InternalServerError(
                "Subquery id not legal: %s" % self._subquery_id
            )
        super(InSessionInstance, self).__init__(**kw)

    @property
    def subquery_id(self):
        return self._subquery_id

    @utils.survey
    def _open_result_reader(self, schema=None, task_name=None, **kwargs):
        """
        Fetch result directly from odps. This way does not support limiting, and will cache
        all result in local memory. To achieve better performance and efficiency, use tunnel instead.
        """
        if not self._is_select:
            raise errors.InstanceTypeNotSupported("No results for non-select SQL.")
        self.reload()
        if not self.is_successful(retry=True):
            raise errors.ODPSError(
                "Cannot open reader, instance(%s) may fail or has not finished yet"
                % self.id
            )
        return readers.CsvRecordReader(schema, self._report_result)

    def _wait_subquery_id_ready(self):
        while self._subquery_id == -1 and self._status != Instance.Status.TERMINATED:
            self.reload()
        if self._subquery_id == -1:
            raise errors.InternalServerError("SubQueryId not returned by the server.")

    def _open_tunnel_reader(self, **kw):
        if not self._is_select:
            raise errors.InstanceTypeNotSupported(
                "InstanceTunnel cannot be opened at a non-select SQL Task."
            )

        self._wait_subquery_id_ready()

        kw.pop("reopen", False)
        arrow = kw.pop("arrow", False)
        endpoint = kw.pop("endpoint", None)
        quota_name = kw.pop("quota_name", None)
        kw["sessional"] = True
        kw["session_subquery_id"] = self._subquery_id
        if "session_task_name" not in kw:
            kw["session_task_name"] = self._session_task_name

        tunnel = self._create_instance_tunnel(endpoint=endpoint, quota_name=quota_name)

        try:
            download_session = tunnel.create_download_session(instance=self, **kw)
        except errors.InternalServerError:
            e, tb = sys.exc_info()[1:]
            e.__class__ = Instance.DownloadSessionCreationError
            six.reraise(Instance.DownloadSessionCreationError, e, tb)

        self._download_id = download_session.id

        if arrow:
            return InSessionInstanceArrowReader(self, download_session)
        else:
            return InSessionInstanceRecordReader(self, download_session)

    def reload(self, blocking=False):
        resp_text = self.get_task_info(
            self._session_task_name, "result_%s" % self._subquery_id
        )
        try:
            query_result = json.loads(resp_text)
            query_status = query_result["status"]
            self._report_result = query_result["result"]
            self._report_warning = query_result["warnings"]
            self._session_task_status = _task_status_value_to_enum(query_status)
            if self._session_task_status in (
                SessionTaskStatus.Terminated,
                SessionTaskStatus.Failed,
                SessionTaskStatus.Cancelled,
            ):
                self._status = Instance.Status.TERMINATED
            elif self._session_task_status == SessionTaskStatus.Running:
                self._status = Instance.Status.RUNNING
            else:
                self._status = Instance.Status.SUSPENDED
            self._subquery_id = int(query_result.get("subQueryId", -1))
        except BaseException as ex:
            raise errors.ODPSError(
                "Invalid Response Format: %s\n Response JSON:%s\n"
                % (str(ex), resp_text)
            )

    def is_successful(self, retry=False, retry_timeout=None):
        """
        If the instance runs successfully.

        :return: True if successful else False
        :rtype: bool
        """

        if not self.is_terminated(retry=retry, retry_timeout=retry_timeout):
            return False
        if self._session_task_status in (
            SessionTaskStatus.Failed,
            SessionTaskStatus.Cancelled,
        ):
            return False
        return True

    def wait_for_success(
        self, interval=1, timeout=None, max_interval=None, blocking=True
    ):
        """
        Wait for instance to complete, and check if the instance is successful.

        :param interval: time interval to check
        :param max_interval: if specified, next check interval will be
            multiplied by 2 till max_interval is reached.
        :param blocking: whether to block waiting at server side. Note that this option does
            not affect client behavior.
        :param timeout: time
        :return: None
        :raise: :class:`odps.errors.ODPSError` if the instance failed
        """

        self.wait_for_completion(
            interval=interval,
            max_interval=max_interval,
            timeout=timeout,
            blocking=blocking,
        )

        if not self.is_successful(retry=True):
            raise errors.parse_instance_error(self._report_result)

    def get_warnings(self):
        """
        Get the warnings reported by ODPS.

        :return: warning string if ever reported, or empty string for no warning.
        """

        self.reload()
        return self._report_warning

    def get_printable_result(self):
        """
        Get the result string that can be directly printed to screen.
        This should only be used for interactive display. The returning format is not guaranteed.

        :return: The printable result. On not completed or no result returned, will return empty string.
        :raise: :class:`odps.errors.ODPSError` if the instance failed.
        """

        self.reload()
        if self.is_terminated() and not self.is_successful():
            raise errors.parse_instance_error(self._report_result)
        return self._report_result

    def _get_sql_task(self):
        resp_text_list = [None]

        def _load_task_data():
            resp_text_list[0] = self.get_task_info(
                self._session_task_name, "sourcexml_%s" % self._subquery_id
            )
            xml_data = json.loads(resp_text_list[0])["result"]
            return tasks.SQLTask.parse(None, xml_data)

        if not self._task_data:
            self._wait_subquery_id_ready()
            try:
                self._task_data = utils.call_with_retry(_load_task_data)
            except BaseException as ex:
                raise errors.ODPSError(
                    "Invalid Response Format: %s\n Response JSON:%s\n"
                    % (ex, resp_text_list[0])
                )
        return self._task_data

    def get_sql_query(self):
        try:
            return self._get_sql_task().query
        except errors.ODPSError:
            return None

    def _parse_subquery_id(self, job_name):
        if not job_name:
            return ""
        match = _SUBQUERY_ID_PATTERN.match(job_name)
        if match:
            return match.group(1)
        elif self.id in job_name:
            return job_name.split(self.id, 1)[1].replace("_", "")
        else:
            return job_name

    def get_task_detail2(self, task_name=None, **kw):
        assert task_name is None or task_name == self._session_task_name
        self._wait_subquery_id_ready()
        kw["subquery_id"] = "session_query_%d" % self._subquery_id
        return super(InSessionInstance, self).get_task_detail2(
            task_name=task_name, **kw
        )

    def _get_queueing_info(self, **kw):
        self._wait_subquery_id_ready()
        kw["subquery_id"] = "session_query_%d" % self._subquery_id
        return super(InSessionInstance, self)._get_queueing_info(**kw)

    def get_logview_address(self, hours=None, use_legacy=None):
        self._wait_subquery_id_ready()
        subquery_suffix = "&subQuery=%s" % self.subquery_id
        return (
            super(InSessionInstance, self).get_logview_address(
                hours=hours, use_legacy=use_legacy
            )
            + subquery_suffix
        )


class McqaV1Methods(object):
    @classmethod
    @utils.deprecated(
        "You no longer have to manipulate session instances to use MaxCompute QueryAcceleration. "
        "Try `run_sql_interactive`."
    )
    def attach_session(cls, odps, session_name, taskname=None, hints=None):
        """
        Attach to an existing session.

        :param session_name: The session name.
        :param taskname: The created sqlrt task name. If not provided, the default value is used.
            Mostly doesn't matter, default works.
        :return: A SessionInstance you may execute select tasks within.
        """
        return cls._attach_mcqa_session(
            odps, session_name, task_name=taskname, hints=hints
        )

    @classmethod
    def _attach_mcqa_session(cls, odps, session_name=None, task_name=None, hints=None):
        session_name = session_name or PUBLIC_SESSION_NAME
        task_name = task_name or DEFAULT_TASK_NAME

        task = tasks.SQLRTTask(name=task_name)
        task.update_sql_rt_settings(hints)
        task.update_sql_rt_settings(
            {
                "odps.sql.session.share.id": session_name,
                "odps.sql.submit.mode": "script",
            }
        )
        project = odps.get_project()
        return project.instances.create(
            task=task, session_project=project, session_name=session_name
        )

    @classmethod
    @utils.deprecated(
        "You no longer have to manipulate session instances to use MaxCompute QueryAcceleration. "
        "Try `run_sql_interactive`."
    )
    def default_session(cls, odps):
        """
        Attach to the default session of your project.

        :return: A SessionInstance you may execute select tasks within.
        """
        return cls._get_default_mcqa_session(odps, wait=False)

    @classmethod
    def _get_default_mcqa_session(
        cls, odps, session_name=None, hints=None, wait=True, service_startup_timeout=60
    ):
        session_name = session_name or PUBLIC_SESSION_NAME
        if odps._default_session is None:
            odps._default_session = cls._attach_mcqa_session(
                odps, session_name, hints=hints
            )
            odps._default_session_name = session_name
            if wait:
                odps._default_session.wait_for_startup(
                    0.1, service_startup_timeout, max_interval=1
                )
        return odps._default_session

    @classmethod
    @utils.deprecated(
        "You no longer have to manipulate session instances to use MaxCompute QueryAcceleration. "
        "Try `run_sql_interactive`."
    )
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
        return cls._create_mcqa_session(
            odps,
            session_worker_count,
            session_worker_memory,
            session_name,
            worker_spare_span,
            taskname,
            hints,
        )

    @classmethod
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
        if not task_name:
            task_name = DEFAULT_TASK_NAME

        session_hints = {
            "odps.sql.session.worker.count": str(session_worker_count),
            "odps.sql.session.worker.memory": str(session_worker_memory),
            "odps.sql.submit.mode": "script",
        }
        if session_name:
            session_hints["odps.sql.session.name"] = session_name
        if worker_spare_span:
            session_hints["odps.sql.session.worker.sparespan"] = worker_spare_span
        task = tasks.SQLRTTask(name=task_name)
        task.update_sql_rt_settings(hints)
        task.update_sql_rt_settings(session_hints)
        project = odps.get_project()
        return project.instances.create(
            task=task, session_project=project, session_name=session_name
        )

    @classmethod
    def _get_mcqa_session_file(cls, odps):
        try:
            dir_name = utils.build_pyodps_dir()
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            expire_time = time.time() - _SESSION_FILE_EXPIRE_TIME
            for session_file in glob.glob(
                os.path.join(dir_name, _SESSION_FILE_PREFIX + "*")
            ):
                if os.path.getctime(session_file) < expire_time:
                    try:
                        os.unlink(session_file)
                    except OSError:
                        pass
            access_id_digest = hashlib.md5(
                utils.to_binary(odps.account.access_id)
            ).hexdigest()
            return os.path.join(dir_name, _SESSION_FILE_PREFIX + access_id_digest)
        except:
            return None

    @classmethod
    def run_sql_interactive(cls, odps, sql, hints=None, **kwargs):
        """
        Run SQL query in interactive mode (a.k.a MaxCompute QueryAcceleration).
        Won't fallback to offline mode automatically if query not supported or fails

        :param sql: the sql query.
        :param hints: settings for sql query.
        :return: instance.
        """
        cached_is_running = False
        service_name = kwargs.pop("service_name", PUBLIC_SESSION_NAME)
        task_name = kwargs.pop("task_name", None)
        service_startup_timeout = kwargs.pop("service_startup_timeout", 60)
        force_reattach = kwargs.pop("force_reattach", False)

        session_file_name = cls._get_mcqa_session_file(odps)
        if (
            odps._default_session is None
            and session_file_name
            and os.path.exists(session_file_name)
        ):
            try:
                with open(session_file_name, "r") as session_file:
                    session_info = json.loads(session_file.read())
                instance_obj = odps.get_instance(session_info.pop("id"))
                session_project = odps.get_project(
                    session_info.pop("session_project_name")
                )
                odps._default_session_name = session_info["session_name"]
                odps._default_session = SessionInstance.from_instance(
                    instance_obj, session_project=session_project, **session_info
                )
            except:
                pass

        if odps._default_session is not None:
            try:
                cached_is_running = odps._default_session.is_running()
            except:
                pass
        if (
            force_reattach
            or not cached_is_running
            or odps._default_session_name != service_name
        ):
            # should reattach, for whatever reason (timed out, terminated, never created,
            # forced using another session)
            odps._default_session = cls._attach_mcqa_session(
                odps, service_name, task_name=task_name
            )
            odps._default_session.wait_for_startup(
                0.1, service_startup_timeout, max_interval=1
            )
            odps._default_session_name = service_name

            if session_file_name:
                try:
                    with open(session_file_name, "w") as session_file:
                        session_file.write(
                            json.dumps(odps._default_session._extract_json_info())
                        )
                except:
                    pass
        return odps._default_session.run_sql(sql, hints, **kwargs)

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
        if isinstance(fallback, (six.string_types, set, list, tuple)):
            fallback_policy = FallbackPolicy(fallback)
        elif fallback is False:
            fallback_policy = None
        elif fallback is True:
            fallback_policy = FallbackPolicy("all")
        else:
            assert isinstance(fallback, FallbackPolicy)
            fallback_policy = fallback

        inst = None
        use_tunnel = kwargs.pop("tunnel", True)
        fallback_callback = kwargs.pop("fallback_callback", None)
        offline_hints = kwargs.pop("offline_hints", None) or {}
        try:
            inst = cls.run_sql_interactive(odps, sql, hints=hints, **kwargs)
            inst.wait_for_success(interval=0.1, max_interval=1)
            try:
                rd = inst.open_reader(tunnel=use_tunnel, limit=True)
                if not rd:
                    raise errors.ODPSError("Get sql result fail")
            except errors.InstanceTypeNotSupported:
                # sql is not a select, just skip creating reader
                pass
            return inst
        except BaseException as ex:
            if fallback_policy is None:
                raise
            fallback_mode = fallback_policy.get_mode_from_exception(ex)
            if fallback_mode is None:
                raise
            elif fallback_mode == FallbackMode.INTERACTIVE:
                kwargs["force_reattach"] = True
                return cls.execute_sql_interactive(
                    odps,
                    sql,
                    hints=hints,
                    fallback=fallback,
                    wait_fallback=wait_fallback,
                    **kwargs
                )
            else:
                kwargs.pop("service_name", None)
                kwargs.pop("force_reattach", None)
                kwargs.pop("service_startup_timeout", None)
                hints = copy.copy(offline_hints or hints or {})
                hints["odps.task.sql.sqa.enable"] = "false"

                if fallback_callback is not None:
                    fallback_callback(inst, ex)

                if inst is not None:
                    hints["odps.sql.session.fallback.instance"] = "%s_%s" % (
                        inst.id,
                        inst.subquery_id,
                    )
                else:
                    hints[
                        "odps.sql.session.fallback.instance"
                    ] = "fallback4AttachFailed"
                inst = odps.execute_sql(
                    sql, hints=hints, quota_name=offline_quota_name, **kwargs
                )
                if wait_fallback:
                    inst.wait_for_success()
                return inst
