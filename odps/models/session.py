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

import sys
import json
import re
import time

from .. import errors, readers, utils
from ..compat import six, enum
from ..models import tasks
from ..serializers import XMLSerializableModel, XMLNodeField
from .instance import Instance


DEFAULT_TASK_NAME = "PyOdpsSQLRTTask"

PUBLIC_SESSION_NAME = "public.default"

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
        if ('session_task_name' not in kw) or ('session_project' not in kw):
            raise errors.InvalidArgument("Creating InSessionInstance without enough information.")
        self._task_name = kw.pop("session_task_name", "")
        self._project = kw.pop("session_project", None)
        self._session_name = kw.pop("session_name", "")
        super(SessionInstance, self).__init__(**kw)

    def wait_for_startup(self, interval=1, timeout=-1, retry=True):
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
                    raise errors.WaitTimeoutError("Waited " + waited + " seconds, but session is not started.")
            try:
                time.sleep(interval)
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

    def _create_internal_instance(self, task=None, headers=None, encoding=None):
        proj_insts = self._project.instances
        session_inst = self
        project_name = self._project.name
        is_select = True
        if not task.query.strip().lower().startswith("select"):
            is_select = False

        _job = proj_insts._create_job(task=task)  # noqa: F841
        rquery = task.query
        if not rquery.endswith(";"):
            rquery = rquery + ";"
        query_object = {"query": rquery,
                        "settings": json.loads(task.properties['settings'])}
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
        resp = proj_insts._client.put(url, xml, headers=headers,
                params={"curr_project": project_name,
                        "taskname": self._task_name})

        location = resp.headers.get('Location')
        if location is None or len(location) == 0:
            raise errors.ODPSError('Invalid response, Location header required.')

        created_subquery_id = -1
        try:
            query_reslt = json.loads(resp.text)
            query_status = query_reslt["status"]
            if query_status != "ok":
                raise errors.ODPSError('Failed to run subquery: [' + query_status + "]: " + query_reslt["result"])
            query_subresult = json.loads(query_reslt["result"])
            created_subquery_id = query_subresult["queryId"]
            if created_subquery_id == -1:
                raise errors.parse_instance_error(query_subresult)
        except KeyError as ex:
            raise errors.ODPSError("Invalid Response Format: %s\n Response JSON:%s\n" % (str(ex), resp.text))
        instance_id = location.rsplit('/', 1)[1]

        instance = InSessionInstance(session_project_name=project_name, session_task_name=self._task_name,
                                     name=instance_id, session_subquery_id=created_subquery_id,
                                     session_instance=self, parent=proj_insts,
                                     session_is_select = is_select, client=self._client)
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
                raise errors.parse_instance_error(error_string)
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


class InSessionInstance(Instance):
    """
        This represents the instance created
        for SQL tasks that run inside a session. This instance is useful
        when fetching results.
    """

    __slots__ = '_project_name', '_session_task_name', '_session', \
            '_session_instance', '_is_select', '_subquery_id', \
            '_report_result', '_report_warning', '_session_task_status'

    def __init__(self, **kw):
        if ('session_task_name' not in kw) \
                or ('session_project_name' not in kw) \
                or ('session_instance' not in kw) \
                or ('session_subquery_id' not in kw):
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
        return readers.RecordReader(schema, self._report_result)

    def _open_tunnel_reader(self, **kw):
        if not self._is_select:
            raise errors.InstanceTypeNotSupported("InstanceTunnel cannot be opened at a non-select SQL Task.")

        while (self._subquery_id == -1) and (self._status != Instance.Status.TERMINATED):
            self.reload()

        if self._subquery_id == -1:
            raise errors.InternalServerError("SubQueryId not returned by the server.")

        kw.pop('reopen', False)
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

        class RecordReader(readers.AbstractRecordReader):
            def __init__(self):
                self._it = iter(self)

            @property
            def schema(self):
                # is not available before open_reader().
                if download_session.schema is None:
                    # open reader once to enforce schema fetched.
                    tmprd = download_session.open_record_reader(0, 1)
                    tmprd.close()
                return download_session.schema

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
                download_session.reload()
                return download_session.status

            def __iter__(self):
                for record in self.read():
                    yield record

            def __next__(self):
                return next(self._it)

            next = __next__

            # also, session results won't slice. but you may skip.
            def _iter(self, step=None):
                return self.read(step=step)

            def read(self, step=None,
                     compress=False, columns=None):
                step = step or 1

                with download_session.open_record_reader(
                        0, 1, compress=compress, columns=columns) as reader:
                    for record in reader[::step]:
                        yield record

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return RecordReader()

    def reload(self):
        url = self._session_instance.resource() + "?info"
        st_resp = self._client.get(url, params={"curr_project": self._project_name,
                                                "taskname": self._session_task_name,
                                                "key": "result_" + str(self._subquery_id)})
        if not st_resp.ok:
            raise errors.ODPSError("HTTP " + str(st_resp.status_code))
        try:
            query_reslt = json.loads(st_resp.text)
            query_status = query_reslt["status"]
            self._report_result = query_reslt["result"]
            self._report_warning = query_reslt["warnings"]
            self._session_task_status = _task_status_value_to_enum(query_status)
            if self._session_task_status == SessionTaskStatus.Terminated \
                    or self._session_task_status == SessionTaskStatus.Failed \
                    or self._session_task_status == SessionTaskStatus.Cancelled:
                self._status = Instance.Status.TERMINATED
            elif self._session_task_status == SessionTaskStatus.Running:
                self._status = Instance.Status.RUNNING
            else:
                self._status = Instance.Status.SUSPENDED
            self._subquery_id = int(query_reslt.get("subQueryId", -1))
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
        if self._session_task_status == SessionTaskStatus.Failed or \
            self._session_task_status == SessionTaskStatus.Cancelled:
            return False
        return True

    def wait_for_success(self, interval=1):
        """
        Wait for instance to complete, and check if the instance is successful.

        :param interval: time interval to check
        :return: None
        :raise: :class:`odps.errors.ODPSError` if the instance failed
        """

        self.wait_for_completion(interval=interval)

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