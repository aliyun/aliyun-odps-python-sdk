#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import base64
import json
import time
from datetime import datetime
import contextlib

from .core import LazyLoad, XMLRemoteModel
from .job import Job
from .. import serializers, utils, errors, compat, readers, options
from ..compat import ElementTree, Enum, six


class Instance(LazyLoad):
    """
    Instance means that a ODPS task will sometimes run as an instance.

    ``status`` can reflect the current situation of a instance.
    ``is_terminated`` method indicates if the instance has finished.
    ``is_successful`` method indicates if the instance runs successfully.
    ``wait_for_success`` method will block the main process until the instance has finished.

    For a SQL instance, we can use open_reader to read the results.

    :Example:

    >>> instance = odps.execute_sql('select * from dual')  # this sql return the structured data
    >>> with instance.open_reader() as reader:
    >>>     # handle the record
    >>>
    >>> instance = odps.execute_sql('desc dual')  # this sql do not return structured data
    >>> with instance.open_reader() as reader:
    >>>    print(reader.raw)  # just return the raw result
    """

    __slots__ = '_task_results', '_is_sync'

    def __init__(self, **kwargs):
        if 'task_results' in kwargs:
            kwargs['_task_results'] = kwargs.pop('task_results')
        super(Instance, self).__init__(**kwargs)

        if self._task_results is not None and len(self._task_results) > 0:
            self._is_sync = True
            self._status = Instance.Status.TERMINATED
        else:
            self._is_sync = False

    @property
    def id(self):
        return self.name

    class Status(Enum):
        RUNNING = 'Running'
        SUSPENDED = 'Suspended'
        TERMINATED = 'Terminated'

    class InstanceStatus(XMLRemoteModel):
        _root = 'Instance'

        status = serializers.XMLNodeField('Status')

    class InstanceResult(XMLRemoteModel):

        class TaskResult(XMLRemoteModel):

            class Result(XMLRemoteModel):

                transform = serializers.XMLNodeAttributeField(attr='Transform')
                format = serializers.XMLNodeAttributeField(attr='Format')
                text = serializers.XMLNodeField('.', default='')

                def __str__(self):
                    if six.PY2:
                        text = utils.to_binary(self.text)
                    else:
                        text = self.text
                    if self.transform is not None and self.transform == 'Base64':
                        try:
                            return base64.b64decode(text)
                        except TypeError:
                            return text
                    return text

            type = serializers.XMLNodeAttributeField(attr='Type')
            name = serializers.XMLNodeField('Name')
            result = serializers.XMLNodeReferenceField(Result, 'Result')

        task_results = serializers.XMLNodesReferencesField(TaskResult, 'Tasks', 'Task')

    class Task(XMLRemoteModel):
        """
        Task stands for each task inside an instance.

        It has a name, a task type, the start to end time, and a running status.
        """

        name = serializers.XMLNodeField('Name')
        type = serializers.XMLNodeAttributeField(attr='Type')
        start_time = serializers.XMLNodeField('StartTime', parse_callback=utils.parse_rfc822)
        end_time = serializers.XMLNodeField('EndTime', parse_callback=utils.parse_rfc822)
        status = serializers.XMLNodeField(
            'Status', parse_callback=lambda s: Instance.Task.TaskStatus(s.upper()))
        histories = serializers.XMLNodesReferencesField('Instance.Task', 'Histories', 'History')

        class TaskStatus(Enum):
            WAITING = 'WAITING'
            RUNNING = 'RUNNING'
            SUCCESS = 'SUCCESS'
            FAILED = 'FAILED'
            SUSPENDED = 'SUSPENDED'
            CANCELLED = 'CANCELLED'

        class TaskProgress(XMLRemoteModel):
            """
            TaskProgress reprents for the progress of a task.

            A single TaskProgress may consist of several stages.

            :Example:

            >>> progress = instance.get_task_progress('task_name')
            >>> progress.get_stage_progress_formatted_string()
            2015-11-19 16:39:07 M1_Stg1_job0:0/0/1[0%]	R2_1_Stg1_job0:0/0/1[0%]
            """

            class StageProgress(XMLRemoteModel):

                name = serializers.XMLNodeAttributeField(attr='ID')
                backup_workers = serializers.XMLNodeField('BackupWorkers', parse_callback=int)
                terminated_workers = serializers.XMLNodeField('TerminatedWorkers', parse_callback=int)
                running_workers = serializers.XMLNodeField('RunningWorkers', parse_callback=int)
                total_workers = serializers.XMLNodeField('TotalWorkers', parse_callback=int)
                input_records = serializers.XMLNodeField('InputRecords', parse_callback=int)
                output_records = serializers.XMLNodeField('OutputRecords', parse_callback=int)
                finished_percentage = serializers.XMLNodeField('FinishedPercentage', parse_callback=int)

            stages = serializers.XMLNodesReferencesField(StageProgress, 'Stage')

            def get_stage_progress_formatted_string(self):
                buf = six.StringIO()

                buf.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                buf.write(' ')

                for stage in self.stages:
                    buf.write('{0}:{1}/{2}/{3}{4}[{5}%]\t'.format(
                        stage.name,
                        stage.running_workers,
                        stage.terminated_workers,
                        stage.total_workers,
                        '(+%s backups)' % stage.backup_workers if stage.backup_workers > 0 else '',
                        stage.finished_percentage
                    ))

                return buf.getvalue()

    name = serializers.XMLNodeField('Name')
    owner = serializers.XMLNodeField('Owner')
    start_time = serializers.XMLNodeField('StartTime', parse_callback=utils.parse_rfc822)
    end_time = serializers.XMLNodeField('EndTime', parse_callback=utils.parse_rfc822)
    _status = serializers.XMLNodeField('Status', parse_callback=lambda s: Instance.Status(s))
    _tasks = serializers.XMLNodesReferencesField(Task, 'Tasks', 'Task')

    class TaskSummary(dict):
        def __init__(self, *args, **kwargs):
            super(Instance.TaskSummary, self).__init__(*args, **kwargs)
            self.summary_text, self.json_summary = None, None

    class AnonymousSubmitInstance(XMLRemoteModel):
        _root = 'Instance'
        job = serializers.XMLNodeReferenceField(Job, 'Job')

    def reload(self):
        resp = self._client.get(self.resource())

        self.owner = resp.headers.get('x-odps-owner')
        self.start_time = utils.parse_rfc822(resp.headers.get('x-odps-start-time'))
        end_time_header = 'x-odps-end-time'
        if end_time_header in resp.headers and \
                len(resp.headers[end_time_header].strip()) > 0:
            self.end_time = utils.parse_rfc822(resp.headers.get(end_time_header))

        self.parse(self._client, resp, obj=self)
        # remember not to set `_loaded = True`

    def stop(self):
        """
        Stop this instance.

        :return: None
        """

        instance_status = Instance.InstanceStatus(status='Terminated')
        xml_content = instance_status.serialize()

        headers = {'Content-Type': 'application/xml'}
        self._client.put(self.resource(), xml_content, headers=headers)

    @property
    def project(self):
        return self.parent.parent

    def get_task_results_without_format(self):
        if self._is_sync:
            return self._task_results

        params = {'result': ''}
        resp = self._client.get(self.resource(), params=params)

        instance_result = Instance.InstanceResult.parse(self._client, resp)
        return compat.OrderedDict([(r.name, r.result) for r in instance_result.task_results])

    def get_task_results(self):
        """
        Get all the task results.

        :return: a dict which key is task name, and value is the task result as string
        :rtype: dict
        """

        results = self.get_task_results_without_format()
        return compat.OrderedDict([(k, str(result)) for k, result in six.iteritems(results)])

    def get_task_result(self, task_name):
        """
        Get a single task result.

        :param task_name: task name
        :return: task result
        :rtype: str
        """
        return self.get_task_results().get(task_name)

    def get_task_summary(self, task_name):
        """
        Get a task's summary, mostly used for MapReduce.

        :param task_name: task name
        :return: summary as a dict parsed from JSON
        :rtype: dict
        """

        params = {'instancesummary': '', 'taskname': task_name}
        resp = self._client.get(self.resource(), params=params)

        map_reduce = json.loads(resp.content).get('Instance')
        if map_reduce:
            json_summary = map_reduce.get('JsonSummary')
            if json_summary:
                summary = Instance.TaskSummary(json.loads(json_summary))
                summary.summary_text = map_reduce.get('Summary')
                summary.json_summary = json_summary

                return summary

    def get_task_statuses(self):
        """
        Get all tasks' statuses

        :return: a dict which key is the task name and value is the :class:`odps.models.Instance.Task` object
        :rtype: dict
        """

        params = {'taskstatus': ''}

        resp = self._client.get(self.resource(), params=params)
        self.parse(self._client, resp, obj=self)

        return dict([(task.name, task) for task in self._tasks])

    def get_task_names(self):
        """
        Get names of all tasks

        :return: task names
        :rtype: list
        """

        return compat.lkeys(self.get_task_statuses())

    @property
    def status(self):
        if self._status != Instance.Status.TERMINATED:
            self.reload()

        return self._status

    def is_terminated(self):
        """
        If this instance has finished or not.

        :return: True if finished else False
        :rtype: bool
        """

        return self.status == Instance.Status.TERMINATED

    def is_successful(self):
        """
        If the instance runs successfully.

        :return: True if successful else False
        :rtype: bool
        """

        if not self.is_terminated():
            return False
        return all(task.status == Instance.Task.TaskStatus.SUCCESS
                   for task in self.get_task_statuses().values())

    @property
    def is_sync(self):
        return self._is_sync

    def wait_for_completion(self, interval=1):
        """
        Wait for the instance to complete, and neglect the consequence.

        :param interval: time interval to check
        :return: None
        """

        while not self.is_terminated():
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                break

    def wait_for_success(self, interval=1):
        """
        Wait for instance to complete, and check if the instance is successful.

        :param interval: time interval to check
        :return: None
        :raise: :class:`odps.errors.ODPSError` if the instance failed
        """

        self.wait_for_completion(interval=interval)

        if not self.is_successful():
            for task_name, task in six.iteritems(self.get_task_statuses()):
                if task.status == Instance.Task.TaskStatus.FAILED:
                    raise errors.ODPSError(self.get_task_result(task_name))
                elif task.status != Instance.Task.TaskStatus.SUCCESS:
                    raise errors.ODPSError('%s, status=%s' % (task_name, task.status.value))

    def get_task_progress(self, task_name):
        """
        Get task's current progress

        :param task_name: task_name
        :return: the task's progress
        :rtype: :class:`odps.models.Instance.Task.TaskProgress`
        """

        params = {'instanceprogress': task_name, 'taskname': task_name}

        resp = self._client.get(self.resource(), params=params)
        return Instance.Task.TaskProgress.parse(self._client, resp)

    def get_logview_address(self, hours=None):
        """
        Get logview address of the instance object by hours.

        :param hours:
        :return: logview address
        :rtype: str
        """
        hours = hours or options.log_view_hours

        project = self.project
        url = '%s/authorization' % project.resource()

        policy = {
            'expires_in_hours': hours,
            'policy': {
                'Statement': [{
                    'Action': ['odps:Read'],
                    'Effect': 'Allow',
                    'Resource': 'acs:odps:*:projects/%s/instances/%s' % \
                                (project.name, self.id)
                }],
                'Version': '1',
            }
        }
        headers = {'Content-Type': 'application/json'}
        params = {'sign_bearer_token': ''}
        data = json.dumps(policy)
        res = self._client.post(url, data, headers=headers, params=params)

        content = res.text if six.PY3 else res.content
        root = ElementTree.fromstring(content)
        token = root.find('Result').text

        link = options.log_view_host + "/logview/?h=" + self._client.endpoint + "&p=" \
               + project.name + "&i=" + self.id + "&token=" + token
        return link

    def __str__(self):
        return self.id

    def _get_job(self):
        url = self.resource()
        params = {'source': ''}
        resp = self._client.get(url, params=params)

        job = Job.parse(self._client, resp)
        return job

    def get_tasks(self):
        job = self._get_job()
        return job.tasks

    @property
    def priority(self):
        job = self._get_job()
        return job.priority

    @contextlib.contextmanager
    def open_reader(self, schema=None, task_name=None):
        if not self.is_successful():
            raise errors.ODPSError(
                'Cannot open reader, instance(%s) may fail or has not finished yet' % self.id)

        sql_tasks = dict([(name, task) for name, task in six.iteritems(self.get_task_statuses())
                          if task.type.lower() == 'sql'])
        if len(sql_tasks) > 1:
            if task_name is None:
                raise errors.ODPSError(
                    'Cannot open reader, job has more than one sql tasks, please specify one')
            elif task_name not in sql_tasks:
                raise errors.ODPSError(
                    'Cannot open reader, unknown task name: %s' % task_name)
        elif len(sql_tasks) == 1:
            task_name = list(sql_tasks)[0]
        else:
            raise errors.ODPSError(
                'Cannot open reader, job has no sql task')

        result = self.get_task_result(task_name)
        with readers.RecordReader(schema, result) as reader:
            yield reader
