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

import base64
import json
import sys
import threading
import time
import warnings
from datetime import datetime

import requests

from .. import serializers, utils, errors, compat, readers, options
from ..accounts import BearerTokenAccount
from ..compat import ElementTree, Enum, six, OrderedDict
from ..utils import to_str
from .core import LazyLoad, XMLRemoteModel, JSONRemoteModel
from .job import Job
from .readers import TunnelRecordReader, TunnelArrowReader
from .worker import WorkerDetail2, LOG_TYPES_MAPPING

try:
    from functools import wraps
except ImportError:
    def wraps(_f):
        def wrapper(fun):
            return fun

        return wrapper


_RESULT_LIMIT_HELPER_MSG = (
    'See https://pyodps.readthedocs.io/zh_CN/latest/base-sql.html#read-sql-exec-result '
    'for more information.'
)


def _with_status_api_lock(func):
    @wraps(func)
    def wrapped(self, *args, **kw):
        with self._status_api_lock:
            return func(self, *args, **kw)

    wrapped.__name__ = func.__name__
    wrapped.__doc__ = func.__doc__
    return wrapped


class InstanceRecordReader(TunnelRecordReader):
    def __init__(self, instance, download_session):
        super(InstanceRecordReader, self).__init__(
            instance, download_session
        )
        self._schema = download_session.schema

    @property
    def schema(self):
        return self._schema

    def _get_process_split_reader(self):
        rest_client = self._parent._client
        project = self._parent.project.name
        tunnel_endpoint = self._parent.project._tunnel_endpoint
        instance_id = self._parent.id

        def read_instance_split(conn, download_id, start, count, idx):
            # read part data
            from ..tunnel import InstanceTunnel

            instance_tunnel = InstanceTunnel(
                client=rest_client, project=project, endpoint=tunnel_endpoint
            )
            session = instance_tunnel.create_download_session(
                instance=instance_id, download_id=download_id
            )
            with session.open_record_reader(start, count) as reader:
                conn.send((idx, reader.to_pandas()))

        return read_instance_split


class InstanceArrowReader(TunnelArrowReader):
    def __init__(self, instance, download_session):
        super(InstanceArrowReader, self).__init__(
            instance, download_session
        )
        self._schema = download_session.schema

    @property
    def schema(self):
        return self._schema


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

    __slots__ = (
        "_task_results",
        "_is_sync",
        "_instance_tunnel",
        "_id_thread_local",
        "_status_api_lock",
        "_logview_address",
        "_logview_address_time",
    )

    _download_id = utils.thread_local_attribute('_id_thread_local', lambda: None)

    def __init__(self, **kwargs):
        if 'task_results' in kwargs:
            kwargs['_task_results'] = kwargs.pop('task_results')
        super(Instance, self).__init__(**kwargs)

        try:
            del self._id_thread_local
        except AttributeError:
            pass

        if self._task_results is not None and len(self._task_results) > 0:
            self._is_sync = True
            self._status = Instance.Status.TERMINATED
        else:
            self._is_sync = False

        self._status_api_lock = threading.RLock()

        self._logview_address = None
        self._logview_address_time = None

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
                            return utils.to_str(base64.b64decode(text))
                        except TypeError:
                            return text
                    return text

                def __bytes__(self):
                    text = utils.to_binary(self.text)
                    if self.transform is not None and self.transform == 'Base64':
                        try:
                            return utils.to_binary(base64.b64decode(text))
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

    class TaskInfo(serializers.XMLSerializableModel):
        _root = 'Instance'
        __slots__ = 'key', 'value'

        key = serializers.XMLNodeField('Key')
        value = serializers.XMLNodeField('Value')

    class TaskCost(object):
        __slots__ = 'cpu_cost', 'memory_cost', 'input_size'

        def __init__(self, cpu_cost=None, memory_cost=None, input_size=None):
            self.cpu_cost = cpu_cost
            self.memory_cost = memory_cost
            self.input_size = input_size

    class SQLCost(object):
        __slots__ = 'udf_num', 'complexity', 'input_size'

        def __init__(self, udf_num=None, complexity=None, input_size=None):
            self.udf_num = udf_num
            self.complexity = complexity
            self.input_size = input_size

    class DownloadSessionCreationError(errors.InternalServerError):
        pass

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

    class InstanceQueueingInfo(JSONRemoteModel):
        __slots__ = '_instance',

        class Status(Enum):
            RUNNING = 'Running'
            SUSPENDED = 'Suspended'
            TERMINATED = 'Terminated'
            UNKNOWN = 'Unknown'

        _properties = serializers.JSONRawField()  # hold the raw dict
        instance_id = serializers.JSONNodeField('instanceId')
        priority = serializers.JSONNodeField('instancePriority')
        progress = serializers.JSONNodeField('instanceProcess')
        job_name = serializers.JSONNodeField('jobName')
        project = serializers.JSONNodeField('projectName')
        skynet_id = serializers.JSONNodeField('skynetId')
        start_time = serializers.JSONNodeField('startTime', parse_callback=utils.strptime_with_tz)
        task_type = serializers.JSONNodeField('taskType')
        task_name = serializers.JSONNodeField('taskName')
        user_account = serializers.JSONNodeField('userAccount')
        status = serializers.JSONNodeField('status', parse_callback=Status)

        @property
        def instance(self):
            if hasattr(self, '_instance') and self._instance:
                return self._instance

            from .projects import Projects
            self._instance = Projects(client=self._client)[self.project].instances[self.instance_id]
            return self._instance

        def __getattr__(self, item):
            item = utils.underline_to_camel(item)

            if item in self._properties:
                return self._properties[item]

            return super(Instance.InstanceQueueingInfo, self).__getattr__(item)

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

    @_with_status_api_lock
    def get_task_results_without_format(self):
        if self._is_sync:
            return self._task_results

        params = {'result': ''}
        resp = self._client.get(self.resource(), params=params)

        instance_result = Instance.InstanceResult.parse(self._client, resp)
        return compat.OrderedDict([(r.name, r.result) for r in instance_result.task_results])

    @_with_status_api_lock
    def get_task_results(self):
        """
        Get all the task results.

        :return: a dict which key is task name, and value is the task result as string
        :rtype: dict
        """

        results = self.get_task_results_without_format()
        if options.tunnel.string_as_binary:
            return compat.OrderedDict([(k, bytes(result)) for k, result in six.iteritems(results)])
        else:
            return compat.OrderedDict([(k, str(result)) for k, result in six.iteritems(results)])

    @_with_status_api_lock
    def get_task_result(self, task_name):
        """
        Get a single task result.

        :param task_name: task name
        :return: task result
        :rtype: str
        """
        return self.get_task_results().get(task_name)

    @_with_status_api_lock
    def get_task_summary(self, task_name):
        """
        Get a task's summary, mostly used for MapReduce.

        :param task_name: task name
        :return: summary as a dict parsed from JSON
        :rtype: dict
        """

        params = {'instancesummary': '', 'taskname': task_name}
        resp = self._client.get(self.resource(), params=params)

        map_reduce = resp.json().get('Instance')
        if map_reduce:
            json_summary = map_reduce.get('JsonSummary')
            if json_summary:
                summary = Instance.TaskSummary(json.loads(json_summary))
                summary.summary_text = map_reduce.get('Summary')
                summary.json_summary = json_summary

                return summary

    @_with_status_api_lock
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

    @_with_status_api_lock
    def get_task_names(self):
        """
        Get names of all tasks

        :return: task names
        :rtype: list
        """

        return compat.lkeys(self.get_task_statuses())

    @_with_status_api_lock
    def get_task_cost(self, task_name):
        """
        Get task cost

        :param task_name: name of the task
        :return: task cost
        :rtype: Instance.TaskCost

        :Example:

        >>> cost = instance.get_task_cost(instance.get_task_names()[0])
        >>> cost.cpu_cost
        200
        >>> cost.memory_cost
        4096
        >>> cost.input_size
        0
        """
        summary = self.get_task_summary(task_name)
        if summary is None:
            return None

        if 'Cost' in summary:
            task_cost = summary['Cost']

            cpu_cost = task_cost.get('CPU')
            memory = task_cost.get('Memory')
            input_size = task_cost.get('Input')

            return Instance.TaskCost(cpu_cost, memory, input_size)

    @_with_status_api_lock
    def get_task_info(self, task_name, key):
        """
        Get task related information.

        :param task_name: name of the task
        :param key: key of the information item
        :return: a string of the task information
        """
        params = OrderedDict([('info', ''), ('taskname', task_name), ('key', key)])

        resp = self._client.get(self.resource(), params=params)
        return resp.text

    @_with_status_api_lock
    def put_task_info(self, task_name, key, value):
        """
        Put information into a task.

        :param task_name: name of the task
        :param key: key of the information item
        :param value: value of the information item
        """
        params = OrderedDict([('info', ''), ('taskname', task_name)])
        headers = {'Content-Type': 'application/xml'}
        body = self.TaskInfo(key=key, value=value).serialize()

        self._client.put(self.resource(), params=params, headers=headers, data=body)

    @_with_status_api_lock
    def get_task_quota(self, task_name):
        """
        Get queueing info of the task.
        Note that time between two calls should larger than 30 seconds, otherwise empty dict is returned.

        :param task_name: name of the task
        :return: quota info in dict format
        """
        params = OrderedDict([('instancequota', ''), ('taskname', task_name)])
        resp = self._client.get(self.resource(), params=params)
        return json.loads(resp.text)

    @_with_status_api_lock
    def get_sql_task_cost(self):
        """
        Get cost information of the sql task.
        Including input data size, number of UDF, Complexity of the sql task

        :return: cost info in dict format
        """
        resp = self.get_task_result(self.get_task_names()[0])
        cost = json.loads(resp)
        sql_cost = cost['Cost']['SQL']

        udf_num = sql_cost.get('UDF')
        complexity = sql_cost.get('Complexity')
        input_size = sql_cost.get('Input')
        return Instance.SQLCost(udf_num, complexity, input_size)

    @property
    @_with_status_api_lock
    def status(self):
        if self._status != Instance.Status.TERMINATED:
            self.reload()

        return self._status

    def is_terminated(self, retry=False):
        """
        If this instance has finished or not.

        :return: True if finished else False
        :rtype: bool
        """
        retry_num = options.retry_times
        while retry_num > 0:
            try:
                return self.status == Instance.Status.TERMINATED
            except (errors.InternalServerError, errors.RequestTimeTooSkewed):
                retry_num -= 1
                if not retry or retry_num <= 0:
                    raise

    def is_running(self, retry=False):
        """
        If this instance is still running.

        :return: True if still running else False
        :rtype: bool
        """
        retry_num = options.retry_times
        while retry_num > 0:
            try:
                return self.status == Instance.Status.RUNNING
            except (errors.InternalServerError, errors.RequestTimeTooSkewed):
                retry_num -= 1
                if not retry or retry_num <= 0:
                    raise

    def is_successful(self, retry=False):
        """
        If the instance runs successfully.

        :return: True if successful else False
        :rtype: bool
        """

        if not self.is_terminated(retry=retry):
            return False
        retry_num = options.retry_times
        while retry_num > 0:
            try:
                statuses = self.get_task_statuses()
                return all(task.status == Instance.Task.TaskStatus.SUCCESS
                           for task in statuses.values())
            except (errors.InternalServerError, errors.RequestTimeTooSkewed):
                retry_num -= 1
                if not retry or retry_num <= 0:
                    raise

    @property
    def is_sync(self):
        return self._is_sync

    def wait_for_completion(self, interval=1, timeout=None):
        """
        Wait for the instance to complete, and neglect the consequence.

        :param interval: time interval to check
        :param timeout: time
        :return: None
        """

        start_time = time.time()
        while not self.is_terminated(retry=True):
            try:
                time.sleep(interval)
                if timeout is not None and time.time() - start_time > timeout:
                    raise errors.WaitTimeoutError(instance_id=self.id)
            except KeyboardInterrupt:
                break

    def wait_for_success(self, interval=1, timeout=None):
        """
        Wait for instance to complete, and check if the instance is successful.

        :param interval: time interval to check
        :return: None
        :raise: :class:`odps.errors.ODPSError` if the instance failed
        """

        self.wait_for_completion(interval=interval, timeout=timeout)

        if not self.is_successful(retry=True):
            for task_name, task in six.iteritems(self.get_task_statuses()):
                exc = None
                if task.status == Instance.Task.TaskStatus.FAILED:
                    exc = errors.parse_instance_error(self.get_task_result(task_name))
                elif task.status != Instance.Task.TaskStatus.SUCCESS:
                    exc = errors.ODPSError('%s, status=%s' % (task_name, task.status.value))
                if exc:
                    exc.instance_id = self.id
                    raise exc

    @_with_status_api_lock
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

    @_with_status_api_lock
    def get_task_detail(self, task_name):
        """
        Get task's detail

        :param task_name: task name
        :return: the task's detail
        :rtype: list or dict according to the JSON
        """
        def _get_detail():
            from ..compat import json  # fix object_pairs_hook parameter for Py2.6

            params = {'instancedetail': '',
                      'taskname': task_name}

            resp = self._client.get(self.resource(), params=params)
            return json.loads(
                resp.content.decode() if six.PY3 else resp.content,
                object_pairs_hook=OrderedDict
            )

        result = _get_detail()
        if not result:
            # todo: this is a workaround for the bug that get_task_detail returns nothing.
            return self.get_task_detail2(task_name)
        else:
            return result

    @_with_status_api_lock
    def get_task_detail2(self, task_name):
        """
        Get task's detail v2

        :param task_name: task name
        :return: the task's detail
        :rtype: list or dict according to the JSON
        """

        from ..compat import json  # fix object_pairs_hook parameter for Py2.6

        params = {'detail': '',
                  'taskname': task_name}

        resp = self._client.get(self.resource(), params=params)
        res = resp.content.decode() if six.PY3 else resp.content
        try:
            return json.loads(res, object_pairs_hook=OrderedDict)
        except ValueError:
            return res

    @_with_status_api_lock
    def get_task_workers(self, task_name=None, json_obj=None):
        """
        Get workers from task
        :param task_name: task name
        :param json_obj: json object parsed from get_task_detail2
        :return: list of workers

        .. seealso:: :class:`odps.models.Worker`
        """
        if task_name is None and json_obj is None:
            raise ValueError('Either task_name or json_obj should be provided')

        if json_obj is None:
            json_obj = self.get_task_detail2(task_name)
        return WorkerDetail2.extract_from_json(json_obj, client=self._client, parent=self)

    @_with_status_api_lock
    def get_worker_log(self, log_id, log_type, size=0):
        """
        Get logs from worker.

        :param log_id: id of log, can be retrieved from details.
        :param log_type: type of logs. Possible log types contains {log_types}
        :param size: length of the log to retrieve
        :return: log content
        """
        params = OrderedDict([('log', ''), ('id', log_id)])
        if log_type is not None:
            log_type = log_type.lower()
            if log_type not in LOG_TYPES_MAPPING:
                raise ValueError('log_type should choose a value in ' +
                                 ' '.join(six.iterkeys(LOG_TYPES_MAPPING)))
            params['logtype'] = LOG_TYPES_MAPPING[log_type]
        if size > 0:
            params['size'] = str(size)
        resp = self._client.get(self.resource(), params=params)
        return resp.text
    get_worker_log.__doc__ = get_worker_log.__doc__.format(log_types=', '.join(sorted(six.iterkeys(LOG_TYPES_MAPPING))))

    @_with_status_api_lock
    def get_logview_address(self, hours=None):
        """
        Get logview address of the instance object by hours.

        :param hours:
        :return: logview address
        :rtype: str
        """
        if (
            self._logview_address is not None
            and time.time() - self._logview_address_time < 600
        ):
            return self._logview_address

        project = self.project
        if isinstance(project.odps.account, BearerTokenAccount):
            token = to_str(project.odps.account.token)
        else:
            hours = hours or options.logview_hours

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

            content = res.content.decode() if six.PY3 else res.content
            root = ElementTree.fromstring(content)
            token = root.find('Result').text

        link = (
            self.project.odps.logview_host
            + "/logview/?h="
            + self._client.endpoint
            + "&p="
            + project.name
            + "&i="
            + self.id
            + "&token="
            + token
        )
        self._logview_address = link
        self._logview_address_time = time.time()
        return link

    def __str__(self):
        return self.id

    def _get_job(self):
        url = self.resource()
        params = {'source': ''}
        resp = self._client.get(url, params=params)

        job = Job.parse(self._client, resp, parent=self)
        return job

    def get_tasks(self):
        return self.tasks

    @property
    def tasks(self):
        job = self._get_job()
        return job.tasks

    @property
    def priority(self):
        job = self._get_job()
        return job.priority

    def _get_queueing_info(self):
        url = self.resource()
        params = {'cached': ''}
        resp = self._client.get(url, params=params)
        return Instance.InstanceQueueingInfo.parse(
            self._client, resp, parent=self.project.instance_queueing_infos), resp

    def get_queueing_info(self):
        info, _ = self._get_queueing_info()
        return info

    def _create_instance_tunnel(self, endpoint=None):
        if self._instance_tunnel is not None:
            return self._instance_tunnel

        from ..tunnel import InstanceTunnel

        self._instance_tunnel = InstanceTunnel(client=self._client, project=self.project,
                                               endpoint=endpoint or self.project._tunnel_endpoint)
        return self._instance_tunnel

    @utils.survey
    def _open_result_reader(self, schema=None, task_name=None, **_):
        if not self.is_successful(retry=True):
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
        reader = readers.RecordReader(schema, result)
        if options.result_reader_create_callback:
            options.result_reader_create_callback(reader)
        return reader

    def _open_tunnel_reader(self, **kw):
        from ..tunnel.instancetunnel import InstanceDownloadSession

        reopen = kw.pop('reopen', False)
        endpoint = kw.pop('endpoint', None)
        arrow = kw.pop("arrow", False)
        tunnel = self._create_instance_tunnel(endpoint=endpoint)
        download_id = self._download_id if not reopen else None

        try:
            download_session = tunnel.create_download_session(
                instance=self, download_id=download_id, **kw
            )
            if download_id and download_session.status != InstanceDownloadSession.Status.Normal:
                download_session = tunnel.create_download_session(instance=self, **kw)
        except errors.InternalServerError:
            e, tb = sys.exc_info()[1:]
            e.__class__ = Instance.DownloadSessionCreationError
            six.reraise(Instance.DownloadSessionCreationError, e, tb)

        self._download_id = download_session.id

        if arrow:
            return InstanceArrowReader(self, download_session)
        else:
            return InstanceRecordReader(self, download_session)

    def open_reader(self, *args, **kwargs):
        """
        Open the reader to read records from the result of the instance. If `tunnel` is `True`,
        instance tunnel will be used. Otherwise conventional routine will be used. If instance tunnel
        is not available and `tunnel` is not specified,, the method will fall back to the
        conventional routine.
        Note that the number of records returned is limited unless `options.limited_instance_tunnel`
        is set to `True` or `limit=True` is configured under instance tunnel mode. Otherwise
        the number of records returned is always limited.

        :param tunnel: if true, use instance tunnel to read from the instance.
                       if false, use conventional routine.
                       if absent, `options.tunnel.use_instance_tunnel` will be used and automatic fallback
                       is enabled.
        :param limit: if True, enable the limitation
        :type limit: bool
        :param reopen: the reader will reuse last one, reopen is true means open a new reader.
        :type reopen: bool
        :param endpoint: the tunnel service URL
        :param compress_option: compression algorithm, level and strategy
        :type compress_option: :class:`odps.tunnel.CompressOption`
        :param compress_algo: compression algorithm, work when ``compress_option`` is not provided,
                              can be ``zlib``, ``snappy``
        :param compress_level: used for ``zlib``, work when ``compress_option`` is not provided
        :param compress_strategy: used for ``zlib``, work when ``compress_option`` is not provided
        :return: reader, ``count`` means the full size, ``status`` means the tunnel status

        :Example:

        >>> with instance.open_reader() as reader:
        >>>     count = reader.count  # How many records of a table or its partition
        >>>     for record in reader[0: count]:
        >>>         # read all data, actually better to split into reading for many times
        """
        use_tunnel = kwargs.get('use_tunnel', kwargs.get('tunnel'))
        auto_fallback_result = use_tunnel is None
        if use_tunnel is None:
            use_tunnel = options.tunnel.use_instance_tunnel
        result_fallback_errors = (errors.InvalidProjectTable, errors.InvalidArgument)
        if use_tunnel:
            # for compatibility
            if 'limit_enabled' in kwargs:
                kwargs['limit'] = kwargs['limit_enabled']
                del kwargs['limit_enabled']

            if 'limit' not in kwargs:
                kwargs['limit'] = options.tunnel.limit_instance_tunnel

            auto_fallback_protection = False
            if kwargs['limit'] is None:
                kwargs['limit'] = False
                auto_fallback_protection = True

            try:
                return self._open_tunnel_reader(**kwargs)
            except result_fallback_errors:
                # service version too low to support instance tunnel.
                if not auto_fallback_result:
                    raise
                if not kwargs.get('limit'):
                    warnings.warn('Instance tunnel not supported, will fallback to '
                                  'conventional ways. 10000 records will be limited. '
                                  + _RESULT_LIMIT_HELPER_MSG)
            except requests.Timeout:
                # tunnel creation timed out, which might be caused by too many files
                # on the service.
                if not auto_fallback_result:
                    raise
                if not kwargs.get('limit'):
                    warnings.warn('Instance tunnel timed out, will fallback to '
                                  'conventional ways. 10000 records will be limited.'
                                  + _RESULT_LIMIT_HELPER_MSG)
            except (Instance.DownloadSessionCreationError, errors.InstanceTypeNotSupported):
                # this is for DDL sql instances such as `show partitions` which raises
                # InternalServerError when creating download sessions.
                if not auto_fallback_result:
                    raise
            except errors.NoPermission:
                # project is protected
                if not auto_fallback_protection:
                    raise
                if not kwargs.get('limit'):
                    warnings.warn('Project under protection, 10000 records will be limited.'
                                  + _RESULT_LIMIT_HELPER_MSG)
                    kwargs['limit'] = True
                    return self._open_tunnel_reader(**kwargs)

        return self._open_result_reader(*args, **kwargs)
