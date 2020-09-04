#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2019 Alibaba Group Holding Ltd.
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
import xml.dom.minidom as minidom
import re
import platform


from .instance import Instance
from .. import errors, readers, utils
from ..compat import six, enum
from ..models import tasks
from ..serializers import XMLSerializableModel, XMLNodeField


DEFAULT_TASK_NAME = "PyOdpsSQLRTTask"

@enum.unique
class SessionTaskStatus(enum.Enum):
    """
    Possible statuses of tasks executing inside a session.
    """
    Running = 2
    Failed = 4
    Terminated = 5
    Cancelled = 6


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

        job = proj_insts._create_job(task=task)
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

        instance_id = location.rsplit('/', 1)[1]

        instance = InSessionInstance(session_project_name=project_name, session_task_name=self._task_name, name=instance_id,
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
        try:
            poll_result = json.loads(st_resp.text)
            self._parse_result_session_name(poll_result["result"])
            if poll_result["status"] == SessionTaskStatus.Running.value:
                self._status = Instance.Status.RUNNING
            elif poll_result["status"] == SessionTaskStatus.Cancelled.value:
                raise errors.NoSuchObject("Session attach cancelled")
            else:
                self._status = Instance.Status.SUSPENDED
        except BaseException:
            error_string = _get_session_failure_info(self.get_task_results())
            if error_string:
                raise errors.ODPSError("Session operation failed: %s" % error_string)
            else:
                # this is a task meta info update problem. Just retry.
                self._status = Instance.Status.SUSPENDED


class InSessionInstance(Instance):
    """
        This represents the instance created
        for SQL tasks that run inside a session. This instance is useful
        when fetching results.
    """

    __slots__ = ('_project_name', '_session_task_name', '_session_instance', '_is_select', '_subquery_id')

    def __init__(self, **kw):
        if ('session_task_name' not in kw) or ('session_project_name' not in kw) or ('session_instance' not in kw):
            raise errors.InvalidArgument("Creating InSessionInstance without enough information.")
        self._session_task_name = kw.pop("session_task_name", "")
        self._project_name = kw.pop("session_project_name", "")
        self._session_instance = kw.pop("session_instance", None)
        self._is_select = kw.pop("session_is_select", False)
        self._subquery_id = -1
        super(InSessionInstance, self).__init__(**kw)

    @utils.survey
    def _open_result_reader(self, schema=None, task_name=None, **_):
        # you just can't.
        # for non-select SQL, no result you may fetch.
        # for select SQL, use instance tunnel. This is deprecated.
        if self._is_select:
            raise errors.InstanceTypeNotSupported("Use tunnel to fetch select result.")
        else:
            raise errors.InstanceTypeNotSupported("No results for non-select SQL.")

    def _open_tunnel_reader(self, **kw):
        if not self._is_select:
            raise errors.InstanceTypeNotSupported("InstanceTunnel cannot be opened at a non-select SQL Task.")
        
        while (self._subquery_id == -1) and (self._status != Instance.Status.TERMINATED):
            self.reload()
        
        if self._subquery_id == -1:
            raise errors.InternalServerError("SubQueryId not returned by the server.")

        from ..tunnel.instancetunnel import InstanceDownloadSession

        reopen = kw.pop('reopen', False)
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
                self._schema = download_session.schema

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
                                                "key": "result"})
        if not st_resp.ok:
            raise errors.ODPSError("HTTP " + str(st_resp.status_code))
        try:
            query_reslt = json.loads(st_resp.text)
            if query_reslt["status"] == SessionTaskStatus.Terminated.value:
                self._status = Instance.Status.TERMINATED
            elif query_reslt["status"] == SessionTaskStatus.Running.value:
                self._status = Instance.Status.RUNNING
            else:
                self._status = Instance.Status.SUSPENDED
            self._subquery_id = int(query_reslt.get("subQueryId", -1))
        except BaseException as ex:
            raise errors.ODPSError("Invalid Response Format: %s\n Response JSON:%s\n" % (str(ex), st_resp.text))
