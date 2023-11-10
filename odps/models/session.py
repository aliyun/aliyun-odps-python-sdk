#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import itertools
import json
import re
import sys
import time
import warnings

from .. import errors, readers, utils
from ..compat import six, enum
from ..models import tasks
from ..serializers import XMLSerializableModel, XMLNodeField
from .instance import Instance, InstanceArrowReader, InstanceRecordReader


DEFAULT_TASK_NAME = "AnonymousSQLRTTask"
PUBLIC_SESSION_NAME = "public.default"


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
        elif self.upgrading and isinstance(exc_value, (errors.SQAServiceUnavailable, errors.SQAAccessDenied)):
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
            s for s in ["generic", "unsupported", "upgrading", "noresource", "timeout"]
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
    6: SessionTaskStatus.Cancelled
}


def _task_status_value_to_enum(task_status):
    return TASK_STATUS_VALUES.get(task_status, SessionTaskStatus.Unknown)


def _get_session_failure_info(task_results):
    try:
        taskname, result_txt = list(task_results.items())[0]
    except BaseException:
        return ''
    return result_txt


class SessionInstance(Instance):
    """
    This represents the instance
    created right after you call 'attach_session' or 'create_session'.
    Further SQL tasks has to be created using this instance.
    """

    __slots__ = ('_project', '_task_name', '_session_name')

    def __init__(self, **kw):
        if 'session_task_name' not in kw or 'session_project' not in kw:
            raise errors.InvalidArgument("Creating InSessionInstance without enough information.")
        self._task_name = kw.pop("session_task_name", "")
        self._project = kw.pop("session_project", None)
        self._session_name = kw.pop("session_name", "")
        super(SessionInstance, self).__init__(**kw)

    def wait_for_startup(self, interval=1, timeout=-1, retry=True, max_interval=None):
        """
        Wait for the session to startup(status changed to RUNNING).

        :param interval: time interval to check (unit seconds)
        :param timeout: wait timeout (unit seconds), < 0 means no timeout
        :param retry: if failed to query session status, should we retry silently
        :raise: :class:`odps.errors.WaitTimeoutError` if wait timeout and session is not started.
        :return: None
        """
        waited = 0
        while not self.is_running(retry):
            if timeout > 0:
                if waited > timeout:
                    raise errors.WaitTimeoutError(
                        "Waited %s seconds, but session is not started." % waited
                    )
            try:
                time.sleep(interval)
                if max_interval:
                    interval = min(interval * 2, max_interval)
                waited += interval
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
                "Cannot split sql statement %s: %s" % (sql_statement, str(ex)), RuntimeWarning
            )
            return False
        return splited[-1].lower().strip(' \t\r\n(').startswith("select")

    def _create_internal_instance(self, task=None, headers=None, encoding=None):
        proj_insts = self._project.instances
        session_inst = self
        project_name = self._project.name
        is_select = self._check_is_select(task.query.strip())

        _job = proj_insts._create_job(task=task)  # noqa: F841
        rquery = task.query
        if not rquery.endswith(";"):
            rquery = rquery + ";"
        query_object = {
            "query": rquery,
            "settings": json.loads(task.properties['settings'])
        }
        query_json = json.dumps(query_object)

        class CreateInstanceQuery(XMLSerializableModel):
            __slots__ = 'key', 'value'
            key = XMLNodeField('Key')
            value = XMLNodeField('Value')
            _root = 'Instance'

            def __init__(self, k, v):
                super(CreateInstanceQuery, self).__init__()
                self.key = k
                self.value = v

        xml = CreateInstanceQuery('query', query_json).serialize()

        headers = headers or dict()
        headers['Content-Type'] = 'application/xml'
        url = session_inst.resource() + "?info"
        params = {
            "curr_project": project_name,
            "taskname": self._task_name,
        }
        resp = proj_insts._client.put(url, xml, headers=headers, params=params)

        location = resp.headers.get('Location')
        if location is None or len(location) == 0:
            raise errors.ODPSError('Invalid response, Location header required.')

        created_subquery_id = -1
        try:
            query_result = json.loads(resp.text)
            query_status = query_result["status"]
            if query_status != "ok":
                raise errors.ODPSError(
                    'Failed to run subquery: [' + query_status + "]: " + query_result["result"]
                )
            query_subresult = json.loads(query_result["result"])
            created_subquery_id = query_subresult["queryId"]
            if created_subquery_id == -1:
                raise errors.parse_instance_error(query_subresult)
        except KeyError as ex:
            six.raise_from(
                errors.ODPSError(
                    "Invalid Response Format: %s\n Response JSON:%s\n" % (str(ex), resp.text)
                ), None
            )
        instance_id = location.rsplit('/', 1)[1]

        instance = InSessionInstance(
            session_project_name=project_name, session_task_name=self._task_name,
            name=instance_id, session_subquery_id=created_subquery_id,
            session_instance=self, parent=proj_insts, session_is_select=is_select,
            client=self._client,
        )
        return instance

    def reload(self):
        url = self.resource() + "?info"
        st_resp = self._client.get(url, params={
            "curr_project": self._project.name,
            "taskname": self._task_name,
            "key": "status"})
        if not st_resp.ok:
            raise errors.ODPSError("HTTP " + str(st_resp.status_code))
        session_status = SessionTaskStatus.Unknown
        try:
            poll_result = json.loads(st_resp.text)
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

    def read(self, start=None, count=None, step=None,
             compress=False, columns=None):
        start = start or 0
        step = step or 1
        stop = None if count is None else start + step * count

        with self._download_session.open_record_reader(
                0, 1, compress=compress, columns=columns) as reader:
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
        '_project_name', '_session_task_name', '_session', '_session_instance',
        '_is_select', '_subquery_id', '_report_result', '_report_warning',
        '_session_task_status',
    )

    def __init__(self, **kw):
        if (
            'session_task_name' not in kw
            or 'session_project_name' not in kw
            or 'session_instance' not in kw
            or 'session_subquery_id' not in kw
        ):
            raise errors.InvalidArgument("Creating InSessionInstance without enough information.")
        self._session_task_name = kw.pop("session_task_name", "")
        self._project_name = kw.pop("session_project_name", "")
        self._session_instance = kw.pop("session_instance", None)
        self._is_select = kw.pop("session_is_select", False)
        self._subquery_id = kw.pop("session_subquery_id", -1)
        self._report_result = ''
        self._report_warning = ''
        self._session_task_status = -1
        if self._subquery_id < 0:
            raise errors.InternalServerError("Subquery id not legal: " + str(self._subquery_id))
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
                'Cannot open reader, instance(%s) may fail or has not finished yet' % self.id)
        return readers.CsvRecordReader(schema, self._report_result)

    def _open_tunnel_reader(self, **kw):
        if not self._is_select:
            raise errors.InstanceTypeNotSupported(
                "InstanceTunnel cannot be opened at a non-select SQL Task."
            )

        while (self._subquery_id == -1) and (self._status != Instance.Status.TERMINATED):
            self.reload()

        if self._subquery_id == -1:
            raise errors.InternalServerError("SubQueryId not returned by the server.")

        kw.pop('reopen', False)
        arrow = kw.pop("arrow", False)
        endpoint = kw.pop('endpoint', None)
        kw['sessional'] = True
        kw['session_subquery_id'] = self._subquery_id
        if 'session_task_name' not in kw:
            kw['session_task_name'] = self._session_task_name

        tunnel = self._create_instance_tunnel(endpoint=endpoint)

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

    def reload(self):
        url = self._session_instance.resource() + "?info"
        params = {
            "curr_project": self._project_name,
            "taskname": self._session_task_name,
            "key": "result_" + str(self._subquery_id)
        }
        st_resp = self._client.get(url, params=params)
        if not st_resp.ok:
            raise errors.ODPSError("HTTP " + str(st_resp.status_code))
        try:
            query_result = json.loads(st_resp.text)
            query_status = query_result["status"]
            self._report_result = query_result["result"]
            self._report_warning = query_result["warnings"]
            self._session_task_status = _task_status_value_to_enum(query_status)
            if self._session_task_status in (
                SessionTaskStatus.Terminated, SessionTaskStatus.Failed, SessionTaskStatus.Cancelled
            ):
                self._status = Instance.Status.TERMINATED
            elif self._session_task_status == SessionTaskStatus.Running:
                self._status = Instance.Status.RUNNING
            else:
                self._status = Instance.Status.SUSPENDED
            self._subquery_id = int(query_result.get("subQueryId", -1))
        except BaseException as ex:
            raise errors.ODPSError("Invalid Response Format: %s\n Response JSON:%s\n" % (str(ex), st_resp.text))

    def is_successful(self, retry=False):
        """
        If the instance runs successfully.

        :return: True if successful else False
        :rtype: bool
        """

        if not self.is_terminated(retry=retry):
            return False
        if self._session_task_status in (SessionTaskStatus.Failed, SessionTaskStatus.Cancelled):
            return False
        return True

    def wait_for_success(self, interval=1, timeout=None, max_interval=None):
        """
        Wait for instance to complete, and check if the instance is successful.

        :param interval: time interval to check
        :param max_interval: if specified, next check interval will be
            multiplied by 2 till max_interval is reached.
        :param timeout: time
        :return: None
        :raise: :class:`odps.errors.ODPSError` if the instance failed
        """

        self.wait_for_completion(interval=interval, max_interval=max_interval, timeout=timeout)

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
