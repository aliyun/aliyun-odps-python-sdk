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

from __future__ import print_function

import base64
import functools
import json
import logging
import sys
import threading
import time
import warnings
from collections import OrderedDict
from datetime import datetime

import requests

from .. import compat, errors, options, readers, serializers, utils
from ..accounts import BearerTokenAccount
from ..compat import Enum, six
from ..lib.monotonic import monotonic
from ..lib.tblib import pickling_support
from ..utils import to_str
from .core import JSONRemoteModel, LazyLoad, XMLRemoteModel
from .job import Job
from .readers import TunnelArrowReader, TunnelRecordReader
from .tasks import SQLTask
from .worker import LOG_TYPES_MAPPING, WorkerDetail2

logger = logging.getLogger(__name__)
pickling_support.install()


_RESULT_LIMIT_HELPER_MSG = (
    "See https://pyodps.readthedocs.io/zh_CN/latest/base-sql.html#read-sql-exec-result "
    "for more information about limits on instance results."
)
_STATUS_QUERY_TIMEOUT = 5 * 60  # timeout when getting status


def _with_status_api_lock(func):
    @six.wraps(func)
    def wrapped(self, *args, **kw):
        with self._status_api_lock:
            return func(self, *args, **kw)

    return wrapped


class SpawnedInstanceReaderMixin(object):
    @property
    def schema(self):
        return self._schema

    @staticmethod
    def _read_instance_split(
        conn,
        download_id,
        start,
        count,
        idx,
        rest_client=None,
        project=None,
        instance_id=None,
        tunnel_endpoint=None,
        columns=None,
        arrow=False,
    ):
        # read part data
        from ..tunnel import InstanceTunnel

        try:
            instance_tunnel = InstanceTunnel(
                client=rest_client, project=project, endpoint=tunnel_endpoint
            )
            session = utils.call_with_retry(
                instance_tunnel.create_download_session,
                instance=instance_id,
                download_id=download_id,
            )

            def _data_to_pandas():
                if not arrow:
                    with session.open_record_reader(
                        start, count, columns=columns
                    ) as reader:
                        return reader.to_pandas()
                else:
                    with session.open_arrow_reader(
                        start, count, columns=columns
                    ) as reader:
                        return reader.to_pandas()

            data = utils.call_with_retry(_data_to_pandas)
            conn.send((idx, data, True))
        except:
            try:
                conn.send((idx, sys.exc_info(), False))
            except:
                logger.exception("Failed to write in process %d", idx)
                raise

    def _get_process_split_reader(self, columns=None, append_partitions=None):  # noqa
        rest_client = self._parent._client
        project = self._parent.project.name
        tunnel_endpoint = self._parent.project._tunnel_endpoint
        instance_id = self._parent.id

        return functools.partial(
            self._read_instance_split,
            rest_client=rest_client,
            project=project,
            instance_id=instance_id,
            arrow=isinstance(self, TunnelArrowReader),
            tunnel_endpoint=tunnel_endpoint,
            columns=columns or self._column_names,
        )


class InstanceRecordReader(SpawnedInstanceReaderMixin, TunnelRecordReader):
    def __init__(self, instance, download_session, columns=None):
        super(InstanceRecordReader, self).__init__(
            instance, download_session, columns=columns
        )
        self._schema = download_session.schema


class InstanceArrowReader(SpawnedInstanceReaderMixin, TunnelArrowReader):
    def __init__(self, instance, download_session, columns=None):
        super(InstanceArrowReader, self).__init__(
            instance, download_session, columns=columns
        )
        self._schema = download_session.schema


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
        "_id_thread_local",
        "_status_api_lock",
        "_logview_address",
        "_logview_address_time",
        "_last_progress_value",
        "_last_progress_time",
        "_logview_logged",
        "_job_source",
    )

    _download_id = utils.thread_local_attribute("_id_thread_local", lambda: None)

    def __init__(self, **kwargs):
        if "task_results" in kwargs:
            kwargs["_task_results"] = kwargs.pop("task_results")
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
        self._last_progress_value = None
        self._last_progress_time = None
        self._logview_logged = False
        self._job_source = None

    @property
    def id(self):
        return self.name

    class Status(Enum):
        RUNNING = "Running"
        SUSPENDED = "Suspended"
        TERMINATED = "Terminated"

    class InstanceStatus(XMLRemoteModel):
        _root = "Instance"

        status = serializers.XMLNodeField("Status")

    class InstanceResult(XMLRemoteModel):
        class TaskResult(XMLRemoteModel):
            class Result(XMLRemoteModel):
                transform = serializers.XMLNodeAttributeField(attr="Transform")
                format = serializers.XMLNodeAttributeField(attr="Format")
                text = serializers.XMLNodeField(".", default="")

                def __str__(self):
                    if six.PY2:
                        text = utils.to_binary(self.text)
                    else:
                        text = self.text
                    if self.transform is not None and self.transform == "Base64":
                        try:
                            return utils.to_str(base64.b64decode(text))
                        except TypeError:
                            return text
                    return text

                def __bytes__(self):
                    text = utils.to_binary(self.text)
                    if self.transform is not None and self.transform == "Base64":
                        try:
                            return utils.to_binary(base64.b64decode(text))
                        except TypeError:
                            return text
                    return text

            type = serializers.XMLNodeAttributeField(attr="Type")
            name = serializers.XMLNodeField("Name")
            result = serializers.XMLNodeReferenceField(Result, "Result")

        task_results = serializers.XMLNodesReferencesField(TaskResult, "Tasks", "Task")

    class Task(XMLRemoteModel):
        """
        Task stands for each task inside an instance.

        It has a name, a task type, the start to end time, and a running status.
        """

        name = serializers.XMLNodeField("Name")
        type = serializers.XMLNodeAttributeField(attr="Type")
        start_time = serializers.XMLNodeField(
            "StartTime", parse_callback=utils.parse_rfc822
        )
        end_time = serializers.XMLNodeField(
            "EndTime", parse_callback=utils.parse_rfc822
        )
        status = serializers.XMLNodeField(
            "Status", parse_callback=lambda s: Instance.Task.TaskStatus(s.upper())
        )
        histories = serializers.XMLNodesReferencesField(
            "Instance.Task", "Histories", "History"
        )

        class TaskStatus(Enum):
            WAITING = "WAITING"
            RUNNING = "RUNNING"
            SUCCESS = "SUCCESS"
            FAILED = "FAILED"
            SUSPENDED = "SUSPENDED"
            CANCELLED = "CANCELLED"

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
                name = serializers.XMLNodeAttributeField(attr="ID")
                backup_workers = serializers.XMLNodeField(
                    "BackupWorkers", parse_callback=int
                )
                terminated_workers = serializers.XMLNodeField(
                    "TerminatedWorkers", parse_callback=int
                )
                running_workers = serializers.XMLNodeField(
                    "RunningWorkers", parse_callback=int
                )
                total_workers = serializers.XMLNodeField(
                    "TotalWorkers", parse_callback=int
                )
                input_records = serializers.XMLNodeField(
                    "InputRecords", parse_callback=int
                )
                output_records = serializers.XMLNodeField(
                    "OutputRecords", parse_callback=int
                )
                finished_percentage = serializers.XMLNodeField(
                    "FinishedPercentage", parse_callback=int
                )

            stages = serializers.XMLNodesReferencesField(StageProgress, "Stage")

            def get_stage_progress_formatted_string(self):
                buf = six.StringIO()

                buf.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                buf.write(" ")

                for stage in self.stages:
                    buf.write(
                        "{0}:{1}/{2}/{3}{4}[{5}%]\t".format(
                            stage.name,
                            stage.running_workers,
                            stage.terminated_workers,
                            stage.total_workers,
                            "(+%s backups)" % stage.backup_workers
                            if stage.backup_workers > 0
                            else "",
                            stage.finished_percentage,
                        )
                    )

                return buf.getvalue()

    class TaskInfo(serializers.XMLSerializableModel):
        _root = "Instance"
        __slots__ = "key", "value"

        key = serializers.XMLNodeField("Key")
        value = serializers.XMLNodeField("Value")

    class TaskCost(object):
        __slots__ = "cpu_cost", "memory_cost", "input_size"

        def __init__(self, cpu_cost=None, memory_cost=None, input_size=None):
            self.cpu_cost = cpu_cost
            self.memory_cost = memory_cost
            self.input_size = input_size

    class SQLCost(object):
        __slots__ = "udf_num", "complexity", "input_size"

        def __init__(self, udf_num=None, complexity=None, input_size=None):
            self.udf_num = udf_num
            self.complexity = complexity
            self.input_size = input_size

    class DownloadSessionCreationError(errors.InternalServerError):
        pass

    class TaskSummary(dict):
        def __init__(self, *args, **kwargs):
            super(Instance.TaskSummary, self).__init__(*args, **kwargs)
            self.summary_text, self.json_summary = None, None

    class AnonymousSubmitInstance(XMLRemoteModel):
        _root = "Instance"
        job = serializers.XMLNodeReferenceField(Job, "Job")

    class InstanceQueueingInfo(JSONRemoteModel):
        __slots__ = ("_instance",)

        class Status(Enum):
            RUNNING = "Running"
            SUSPENDED = "Suspended"
            TERMINATED = "Terminated"
            UNKNOWN = "Unknown"

        _properties = serializers.JSONRawField()  # hold the raw dict
        instance_id = serializers.JSONNodeField("instanceId")
        priority = serializers.JSONNodeField("instancePriority")
        progress = serializers.JSONNodeField("instanceProcess")
        job_name = serializers.JSONNodeField("jobName")
        project = serializers.JSONNodeField("projectName")
        skynet_id = serializers.JSONNodeField("skynetId")
        start_time = serializers.JSONNodeField(
            "startTime", parse_callback=utils.strptime_with_tz
        )
        task_type = serializers.JSONNodeField("taskType")
        task_name = serializers.JSONNodeField("taskName")
        user_account = serializers.JSONNodeField("userAccount")
        status = serializers.JSONNodeField("status", parse_callback=Status)

        @property
        def instance(self):
            if hasattr(self, "_instance") and self._instance:
                return self._instance

            from .projects import Projects

            self._instance = Projects(client=self._client)[self.project].instances[
                self.instance_id
            ]
            return self._instance

        def __getattr__(self, item):
            item = utils.underline_to_camel(item)

            if item in self._properties:
                return self._properties[item]

            return super(Instance.InstanceQueueingInfo, self).__getattr__(item)

    name = serializers.XMLNodeField("Name")
    owner = serializers.XMLNodeField("Owner")
    start_time = serializers.XMLNodeField(
        "StartTime", parse_callback=utils.parse_rfc822
    )
    end_time = serializers.XMLNodeField("EndTime", parse_callback=utils.parse_rfc822)
    _status = serializers.XMLNodeField(
        "Status", parse_callback=lambda s: Instance.Status(s)
    )
    _tasks = serializers.XMLNodesReferencesField(Task, "Tasks", "Task")

    def reload(self, blocking=False):
        actions = []
        if blocking:
            actions.append("instancestatus")
        resp = self._client.get(self.resource(), actions=actions)

        self.owner = resp.headers.get("x-odps-owner")
        self.start_time = utils.parse_rfc822(resp.headers.get("x-odps-start-time"))
        end_time_header = "x-odps-end-time"
        if (
            end_time_header in resp.headers
            and len(resp.headers[end_time_header].strip()) > 0
        ):
            self.end_time = utils.parse_rfc822(resp.headers.get(end_time_header))

        self.parse(self._client, resp, obj=self)
        # remember not to set `_loaded = True`

    def stop(self):
        """
        Stop this instance.

        :return: None
        """

        instance_status = Instance.InstanceStatus(status="Terminated")
        xml_content = instance_status.serialize()

        headers = {"Content-Type": "application/xml"}
        self._client.put(self.resource(), xml_content, headers=headers)

    @staticmethod
    def _call_with_retry(func, retry=False, retry_timeout=None):
        retry_kw = {
            "retry_times": options.retry_times if retry else 0,
            "exc_type": (errors.InternalServerError, errors.RequestTimeTooSkewed),
        }
        if retry and retry_timeout is not None:
            # use retry timeout instead of retry count
            retry_kw.update({"retry_times": None, "retry_timeout": retry_timeout})
        return utils.call_with_retry(func, **retry_kw)

    @_with_status_api_lock
    def get_task_results_without_format(self, timeout=None, retry=True):
        if self._is_sync:
            return self._task_results

        def _get_resp():
            return self._client.get(self.resource(), action="result", timeout=timeout)

        resp = self._call_with_retry(_get_resp, retry=retry, retry_timeout=timeout)
        instance_result = Instance.InstanceResult.parse(self._client, resp)
        return OrderedDict([(r.name, r.result) for r in instance_result.task_results])

    @_with_status_api_lock
    def get_task_results(self, timeout=None, retry=True):
        """
        Get all the task results.

        :return: a dict which key is task name, and value is the task result as string
        :rtype: dict
        """

        results = self.get_task_results_without_format(timeout=timeout, retry=retry)
        if options.tunnel.string_as_binary:
            return OrderedDict(
                [(k, bytes(result)) for k, result in six.iteritems(results)]
            )
        else:
            return OrderedDict(
                [(k, str(result)) for k, result in six.iteritems(results)]
            )

    def _get_default_task_name(self):
        job = self._get_job()
        if len(job.tasks) != 1:
            msg = "No tasks" if len(job.tasks) == 0 else "Multiple tasks"
            raise errors.ODPSError("%s in instance." % msg)
        return job.tasks[0].name

    @_with_status_api_lock
    def get_task_result(self, task_name=None, timeout=None, retry=True):
        """
        Get a single task result.

        :param task_name: task name
        :return: task result
        :rtype: str
        """
        task_name = task_name or self._get_default_task_name()
        return self.get_task_results(timeout=timeout, retry=retry).get(task_name)

    @_with_status_api_lock
    def get_task_summary(self, task_name=None):
        """
        Get a task's summary, mostly used for MapReduce.

        :param task_name: task name
        :return: summary as a dict parsed from JSON
        :rtype: dict
        """
        task_name = task_name or self._get_default_task_name()
        params = {"taskname": task_name}
        resp = self._client.get(
            self.resource(), action="instancesummary", params=params
        )

        map_reduce = resp.json().get("Instance")
        if map_reduce:
            json_summary = map_reduce.get("JsonSummary")
            if json_summary:
                summary = Instance.TaskSummary(json.loads(json_summary))
                summary.summary_text = map_reduce.get("Summary")
                summary.json_summary = json_summary

                return summary

    @_with_status_api_lock
    def get_task_statuses(self, retry=True, timeout=None):
        """
        Get all tasks' statuses

        :return: a dict which key is the task name and value is the :class:`odps.models.Instance.Task` object
        :rtype: dict
        """

        def _get_resp():
            return self._client.get(self.resource(), action="taskstatus")

        resp = self._call_with_retry(_get_resp, retry=retry, retry_timeout=timeout)
        self.parse(self._client, resp, obj=self)
        return dict([(task.name, task) for task in self._tasks])

    @_with_status_api_lock
    def get_task_names(self, retry=True, timeout=None):
        """
        Get names of all tasks

        :return: task names
        :rtype: list
        """

        return compat.lkeys(self.get_task_statuses(retry=retry, timeout=timeout))

    def get_task_cost(self, task_name=None):
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
        task_name = task_name or self._get_default_task_name()
        summary = self.get_task_summary(task_name)
        if summary is None:
            return None

        if "Cost" in summary:
            task_cost = summary["Cost"]

            cpu_cost = task_cost.get("CPU")
            memory = task_cost.get("Memory")
            input_size = task_cost.get("Input")

            return Instance.TaskCost(cpu_cost, memory, input_size)

    def _raise_empty_task_info(self, resp):
        raise errors.EmptyTaskInfoError(
            "Empty response. Task server maybe dead.",
            code=resp.status_code,
            instance_id=self.id,
            endpoint=self._client.endpoint,
            request_id=resp.headers.get("x-odps-request-id"),
            tag="ODPS",
        )

    def get_task_info(self, task_name, key, raise_empty=False):
        """
        Get task related information.

        :param task_name: name of the task
        :param key: key of the information item
        :param raise_empty: if True, will raise error when response is empty
        :return: a string of the task information
        """
        actions = ["info"]
        params = OrderedDict([("taskname", task_name), ("key", key)])

        resp = self._client.get(self.resource(), actions=actions, params=params)
        resp_data = resp.content.decode()
        if raise_empty and not resp_data:
            self._raise_empty_task_info(resp)
        return resp_data

    def put_task_info(
        self, task_name, key, value, check_location=False, raise_empty=False
    ):
        """
        Put information into a task.

        :param task_name: name of the task
        :param key: key of the information item
        :param value: value of the information item
        :param check_location: raises if Location header is missing
        :param raise_empty: if True, will raise error when response is empty
        """
        actions = ["info"]
        params = {"taskname": task_name}
        headers = {"Content-Type": "application/xml"}
        body = self.TaskInfo(key=key, value=value).serialize()

        resp = self._client.put(
            self.resource(), actions=actions, params=params, headers=headers, data=body
        )

        location = resp.headers.get("Location")
        if check_location and (location is None or len(location) == 0):
            raise errors.ODPSError("Invalid response, Location header required.")
        resp_data = resp.content.decode()
        if raise_empty and not resp_data:
            self._raise_empty_task_info(resp)
        return resp_data

    def get_task_quota(self, task_name=None):
        """
        Get queueing info of the task.
        Note that time between two calls should larger than 30 seconds, otherwise empty dict is returned.

        :param task_name: name of the task
        :return: quota info in dict format
        """
        task_name = task_name or self._get_default_task_name()
        actions = ["instancequota"]
        params = {"taskname": task_name}
        resp = self._client.get(self.resource(), actions=actions, params=params)
        return json.loads(resp.text)

    def get_sql_task_cost(self):
        """
        Get cost information of the sql cost task, including input data size,
        number of UDF, Complexity of the sql task.

        NOTE that DO NOT use this function directly as it cannot be applied to
        instances returned from SQL. Use ``o.execute_sql_cost`` instead.

        :return: cost info in dict format
        """
        resp = self.get_task_result(self.get_task_names()[0])
        cost = json.loads(resp)
        sql_cost = cost["Cost"]["SQL"]

        udf_num = sql_cost.get("UDF")
        complexity = sql_cost.get("Complexity")
        input_size = sql_cost.get("Input")
        return Instance.SQLCost(udf_num, complexity, input_size)

    def _get_status(self, blocking=False):
        if self._status != Instance.Status.TERMINATED:
            self.reload(blocking)
        return self._status

    @property
    @_with_status_api_lock
    def status(self):
        return self._get_status()

    def is_terminated(self, retry=True, blocking=False, retry_timeout=None):
        """
        If this instance has finished or not.

        :return: True if finished else False
        :rtype: bool
        """
        return self._call_with_retry(
            lambda: self._get_status(blocking) == Instance.Status.TERMINATED,
            retry=retry,
            retry_timeout=retry_timeout,
        )

    def is_running(self, retry=True, blocking=False, retry_timeout=None):
        """
        If this instance is still running.

        :return: True if still running else False
        :rtype: bool
        """
        return self._call_with_retry(
            lambda: self._get_status(blocking) == Instance.Status.RUNNING,
            retry=retry,
            retry_timeout=retry_timeout,
        )

    def is_successful(self, retry=True, retry_timeout=None):
        """
        If the instance runs successfully.

        :return: True if successful else False
        :rtype: bool
        """

        if not self.is_terminated(retry=retry):
            return False

        def _get_successful():
            statuses = self.get_task_statuses()
            return all(
                task.status == Instance.Task.TaskStatus.SUCCESS
                for task in statuses.values()
            )

        return self._call_with_retry(
            _get_successful, retry=retry, retry_timeout=retry_timeout
        )

    @property
    def is_sync(self):
        return self._is_sync

    def get_all_task_progresses(self):
        return {
            task_name: self.get_task_progress(task_name)
            for task_name in self.get_task_names()
        }

    def _dump_instance_progress(self, start_time, check_time, final=False):
        if logger.getEffectiveLevel() > logging.INFO:
            return

        prog_time_interval = options.progress_time_interval
        prog_percentage_gap = options.progress_percentage_gap
        logview_latency = min(options.logview_latency, prog_time_interval)
        try:
            task_progresses = self.get_all_task_progresses()
            total_progress = sum(
                stage.finished_percentage
                for progress in task_progresses.values()
                for stage in progress.stages
            )

            if not self._logview_logged and check_time - start_time >= logview_latency:
                self._logview_logged = True
                logger.info(
                    "Instance ID: %s\n  Log view: %s",
                    self.id,
                    self.get_logview_address(),
                )

            # final log need to be outputed once the progress is updated and logview
            #  address is printed
            need_final_log = (
                final
                and self._logview_logged
                and self._last_progress_value < total_progress
            )
            # intermediate log need to be outputed once current progress exceeds certain
            #  gap or certain time elapsed
            need_intermediate_log = check_time - start_time >= prog_time_interval and (
                total_progress - self._last_progress_value >= prog_percentage_gap
                or check_time - self._last_progress_time >= prog_time_interval
            )
            if need_final_log or need_intermediate_log:
                output_parts = [str(self.id)] + [
                    progress.get_stage_progress_formatted_string()
                    for progress in task_progresses.values()
                ]
                if len(output_parts) > 1:
                    logger.info(" ".join(output_parts))
                self._last_progress_value = total_progress
                self._last_progress_time = check_time
        except:  # pragma: no cover
            # make sure progress display does not affect execution
            pass

    def wait_for_completion(
        self, interval=1, timeout=None, max_interval=None, blocking=True
    ):
        """
        Wait for the instance to complete, and neglect the consequence.

        :param interval: time interval to check
        :param max_interval: if specified, next check interval will be
            multiplied by 2 till max_interval is reached.
        :param timeout: time
        :param blocking: whether to block waiting at server side. Note that this option does
            not affect client behavior.
        :return: None
        """

        start_time = check_time = self._last_progress_time = monotonic()
        self._last_progress_value = 0
        while not self.is_terminated(
            retry=True, blocking=blocking, retry_timeout=_STATUS_QUERY_TIMEOUT
        ):
            try:
                sleep_interval_left = interval - (monotonic() - check_time)
                if sleep_interval_left > 0:
                    time.sleep(sleep_interval_left)
                check_time = monotonic()
                if max_interval is not None:
                    interval = min(interval * 2, max_interval)
                if timeout is not None and check_time - start_time > timeout:
                    raise errors.WaitTimeoutError(
                        "Wait completion of instance %s timed out" % self.id,
                        instance_id=self.id,
                    )

                self._dump_instance_progress(start_time, check_time)
            except KeyboardInterrupt:
                break
        # dump final progress
        self._dump_instance_progress(start_time, check_time, final=True)

    def wait_for_success(
        self, interval=1, timeout=None, max_interval=None, blocking=True
    ):
        """
        Wait for instance to complete, and check if the instance is successful.

        :param interval: time interval to check
        :param max_interval: if specified, next check interval will be
            multiplied by 2 till max_interval is reached.
        :param timeout: time
        :param blocking: whether to block waiting at server side. Note that this option does
            not affect client behavior.
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
            for task_name, task in six.iteritems(self.get_task_statuses()):
                exc = None
                if task.status == Instance.Task.TaskStatus.FAILED:
                    exc = errors.parse_instance_error(self.get_task_result(task_name))
                elif task.status != Instance.Task.TaskStatus.SUCCESS:
                    exc = errors.ODPSError(
                        "%s, status=%s" % (task_name, task.status.value)
                    )
                if exc:
                    exc.instance_id = self.id
                    raise exc

    @_with_status_api_lock
    def get_task_progress(self, task_name=None):
        """
        Get task's current progress

        :param task_name: task_name
        :return: the task's progress
        :rtype: :class:`odps.models.Instance.Task.TaskProgress`
        """
        task_name = task_name or self._get_default_task_name()
        params = {"instanceprogress": task_name, "taskname": task_name}

        resp = self._client.get(self.resource(), params=params)
        return Instance.Task.TaskProgress.parse(self._client, resp)

    @_with_status_api_lock
    def get_task_detail(self, task_name=None):
        """
        Get task's detail

        :param task_name: task name
        :return: the task's detail
        :rtype: list or dict according to the JSON
        """

        def _get_detail():
            from ..compat import json  # fix object_pairs_hook parameter for Py2.6

            params = {"taskname": task_name}
            resp = self._client.get(
                self.resource(), action="instancedetail", params=params
            )
            res = resp.content.decode() if six.PY3 else resp.content
            try:
                return json.loads(res, object_pairs_hook=OrderedDict)
            except ValueError:
                return res

        task_name = task_name or self._get_default_task_name()
        result = _get_detail()
        if not result:
            # todo: this is a workaround for the bug that get_task_detail returns nothing.
            return self.get_task_detail2(task_name)
        else:
            return result

    @_with_status_api_lock
    def get_task_detail2(self, task_name=None, **kw):
        """
        Get task's detail v2

        :param task_name: task name
        :return: the task's detail
        :rtype: list or dict according to the JSON
        """
        task_name = task_name or self._get_default_task_name()
        params = {"taskname": task_name}
        if "subquery_id" in kw:
            params["subquery_id"] = str(kw.pop("subquery_id"))

        resp = self._client.get(self.resource(), action="detail", params=params)
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
        if json_obj is None:
            task_name = task_name or self._get_default_task_name()

        if json_obj is None:
            json_obj = self.get_task_detail2(task_name)
        return WorkerDetail2.extract_from_json(
            json_obj, client=self._client, parent=self
        )

    @_with_status_api_lock
    def get_worker_log(self, log_id, log_type, size=0):
        """
        Get logs from worker.

        :param log_id: id of log, can be retrieved from details.
        :param log_type: type of logs. Possible log types contains {log_types}
        :param size: length of the log to retrieve
        :return: log content
        """
        params = OrderedDict([("log", ""), ("id", log_id)])
        if log_type is not None:
            log_type = log_type.lower()
            if log_type not in LOG_TYPES_MAPPING:
                raise ValueError(
                    "log_type should choose a value in "
                    + " ".join(six.iterkeys(LOG_TYPES_MAPPING))
                )
            params["logtype"] = LOG_TYPES_MAPPING[log_type]
        if size > 0:
            params["size"] = str(size)
        resp = self._client.get(self.resource(), params=params)
        return resp.text

    get_worker_log.__doc__ = get_worker_log.__doc__.format(
        log_types=", ".join(sorted(six.iterkeys(LOG_TYPES_MAPPING)))
    )

    @_with_status_api_lock
    def get_logview_address(self, hours=None, use_legacy=None):
        """
        Get logview address of the instance object by hours.

        :param hours:
        :return: logview address
        :rtype: str
        """
        if use_legacy is None:
            use_legacy = options.use_legacy_logview
        if use_legacy is None and self.project.odps.job_insight_host is not None:
            use_legacy = False
        if (
            self.project.odps.job_insight_host is None
            or self.project.odps.region_name is None
        ):
            use_legacy = True
        if use_legacy is False:
            return self._get_job_insight_address()
        return self._get_legacy_logview_address(hours=hours)

    def _get_job_insight_address(self):
        return (
            "%(job_insight_host)s/%(region_id)s/job-insights?h=%(endpoint)s"
            "&p=%(project_name)s&i=%(instance_id)s"
        ) % dict(
            job_insight_host=self.project.odps.job_insight_host,
            region_id=self.project.odps.region_name,
            endpoint=self._client.endpoint,
            project_name=self.project.name,
            instance_id=self.id,
        )

    def _get_legacy_logview_address(self, hours=None):
        if (
            self._logview_address is not None
            and monotonic() - self._logview_address_time < 600
        ):
            return self._logview_address

        project = self.project
        if isinstance(project.odps.account, BearerTokenAccount):
            token = to_str(project.odps.account.token)
        else:
            hours = hours or options.logview_hours
            policy = {
                "Statement": [
                    {
                        "Action": ["odps:Read"],
                        "Effect": "Allow",
                        "Resource": "acs:odps:*:projects/%s/instances/%s"
                        % (project.name, self.id),
                    }
                ],
                "Version": "1",
            }
            token = self.project.generate_auth_token(policy, "bearer", hours)

        link = (
            "%(logview_host)s/logview/?h=%(endpoint)s&p=%(project_name)s"
            "&i=%(instance_id)s&token=%(token)s"
        ) % dict(
            logview_host=self.project.odps.logview_host,
            endpoint=self._client.endpoint,
            project_name=project.name,
            instance_id=self.id,
            token=token,
        )
        self._logview_address = link
        self._logview_address_time = monotonic()
        return link

    def __str__(self):
        return self.id

    def _get_job(self):
        if not self._job_source:
            url = self.resource()
            resp = self._client.get(url, action="source")

            self._job_source = Job.parse(self._client, resp, parent=self)
        return self._job_source

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

    def _get_queueing_info(self, **kw):
        params = {}
        if "subquery_id" in kw:
            params["subquery_id"] = str(kw.pop("subquery_id"))

        url = self.resource()
        resp = self._client.get(url, action="cached", params=params)
        return (
            Instance.InstanceQueueingInfo.parse(
                self._client, resp, parent=self.project.instance_queueing_infos
            ),
            resp,
        )

    def get_queueing_info(self):
        info, _ = self._get_queueing_info()
        return info

    def get_sql_query(self):
        task = [t for t in self.tasks if isinstance(t, SQLTask)]
        if not task:
            raise errors.ODPSError("Instance %s does not contain a SQLTask.", self.id)
        if len(task) > 1:  # pragma: no cover
            raise errors.ODPSError("Multiple SQLTasks exist in instance %s.", self.id)
        return task[0].query

    def get_unique_identifier_id(self):
        job = self._get_job()
        return job.unique_identifier_id

    def _check_get_task_name(self, task_type, task_name=None, err_head=None):
        if not self.is_successful(retry=True):
            raise errors.ODPSError(
                "%s, instance(%s) may fail or has not finished yet"
                % (err_head, self.id)
            )

        task_type = task_type.lower()
        filtered_tasks = {
            name: task
            for name, task in six.iteritems(self.get_task_statuses())
            if task.type.lower() == task_type
        }
        if len(filtered_tasks) > 1:
            if task_name is None:
                raise errors.ODPSError(
                    "%s, job has more than one %s tasks, please specify one"
                    % (err_head, task_type)
                )
            elif task_name not in filtered_tasks:
                raise errors.ODPSError(
                    "%s, unknown task name: %s" % (err_head, task_name)
                )
            return task_name
        elif len(filtered_tasks) == 1:
            return list(filtered_tasks)[0]
        else:
            raise errors.ODPSError("%s, job has no %s task" % (err_head, task_type))

    def _create_instance_tunnel(self, endpoint=None, quota_name=None):
        from ..tunnel import InstanceTunnel

        return InstanceTunnel(
            client=self._client,
            project=self.project,
            quota_name=quota_name,
            endpoint=endpoint or self.project._tunnel_endpoint,
        )

    @utils.survey
    def _open_result_reader(self, schema=None, task_name=None, timeout=None, **kw):
        task_name = self._check_get_task_name(
            "sql", task_name=task_name, err_head="Cannot open reader"
        )
        result = self.get_task_result(task_name, timeout=timeout)
        reader = readers.CsvRecordReader(schema, result, **kw)
        if options.result_reader_create_callback:
            options.result_reader_create_callback(reader)
        return reader

    def _open_tunnel_reader(self, **kw):
        from ..tunnel.instancetunnel import InstanceDownloadSession

        reopen = kw.pop("reopen", False)
        endpoint = kw.pop("endpoint", None)
        quota_name = kw.pop("quota_name", None)
        arrow = kw.pop("arrow", False)
        columns = kw.pop("columns", None)

        tunnel = self._create_instance_tunnel(endpoint=endpoint, quota_name=quota_name)
        download_id = self._download_id if not reopen else None

        try:
            download_session = tunnel.create_download_session(
                instance=self, download_id=download_id, **kw
            )
            if (
                download_id
                and download_session.status != InstanceDownloadSession.Status.Normal
            ):
                download_session = tunnel.create_download_session(instance=self, **kw)
        except errors.InternalServerError:
            e, tb = sys.exc_info()[1:]
            e.__class__ = Instance.DownloadSessionCreationError
            six.reraise(Instance.DownloadSessionCreationError, e, tb)

        self._download_id = download_session.id

        if arrow:
            return InstanceArrowReader(self, download_session, columns=columns)
        else:
            return InstanceRecordReader(self, download_session, columns=columns)

    def open_reader(self, *args, **kwargs):
        """
        Open the reader to read records from the result of the instance. If `tunnel` is `True`,
        instance tunnel will be used. Otherwise conventional routine will be used. If instance tunnel
        is not available and `tunnel` is not specified, the method will fall back to the
        conventional routine.
        Note that the number of records returned is limited unless `options.limited_instance_tunnel`
        is set to `True` or `limit=True` is configured under instance tunnel mode. Otherwise
        the number of records returned is always limited.

        :param tunnel: if true, use instance tunnel to read from the instance.
                       if false, use conventional routine.
                       if absent, `options.tunnel.use_instance_tunnel` will be used and automatic fallback
                       is enabled.
        :param bool limit: if True, enable the limitation
        :param bool reopen: the reader will reuse last one, reopen is true means open a new reader.
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
        use_tunnel = kwargs.get("use_tunnel", kwargs.get("tunnel"))
        auto_fallback_result = use_tunnel is None

        timeout = kwargs.pop("timeout", None)
        if use_tunnel is None:
            use_tunnel = options.tunnel.use_instance_tunnel
            if use_tunnel:
                timeout = (
                    timeout
                    if timeout is not None
                    else options.tunnel.legacy_fallback_timeout
                )
        kwargs["timeout"] = timeout

        result_fallback_errors = (
            errors.InvalidProjectTable,
            errors.InvalidArgument,
            errors.NoSuchProject,
        )
        if use_tunnel:
            # for compatibility
            if "limit_enabled" in kwargs:
                kwargs["limit"] = kwargs["limit_enabled"]
                del kwargs["limit_enabled"]

            if "limit" not in kwargs:
                kwargs["limit"] = options.tunnel.limit_instance_tunnel

            auto_fallback_protection = False
            if kwargs["limit"] is None:
                kwargs["limit"] = False
                auto_fallback_protection = True

            try:
                return self._open_tunnel_reader(**kwargs)
            except result_fallback_errors:
                # service version too low to support instance tunnel.
                if not auto_fallback_result:
                    raise
                if not kwargs.get("limit"):
                    warnings.warn(
                        "Instance tunnel not supported, will fallback to "
                        "restricted approach. 10000 records will be limited. "
                        + _RESULT_LIMIT_HELPER_MSG
                    )
            except requests.Timeout:
                # tunnel creation timed out, which might be caused by too many files
                # on the service.
                if not auto_fallback_result:
                    raise
                if not kwargs.get("limit"):
                    warnings.warn(
                        "Instance tunnel timed out, will fallback to restricted approach. "
                        "10000 records will be limited. You may try merging small files "
                        "on your source table. " + _RESULT_LIMIT_HELPER_MSG
                    )
            except (
                Instance.DownloadSessionCreationError,
                errors.InstanceTypeNotSupported,
            ):
                # this is for DDL sql instances such as `show partitions` which raises
                # InternalServerError when creating download sessions.
                if not auto_fallback_result:
                    raise
            except errors.NoPermission as exc:
                # project is protected or data permission is configured
                if not auto_fallback_protection:
                    raise
                if not kwargs.get("limit"):
                    warnings.warn(
                        "Project or data under protection, 10000 records will be limited. "
                        "Raw error message:\n"
                        + str(exc)
                        + "\n"
                        + _RESULT_LIMIT_HELPER_MSG
                    )
                    kwargs["limit"] = True
                    return self.open_reader(*args, **kwargs)

        return self._open_result_reader(*args, **kwargs)

    def _iter_reader_with_pandas(self, iter_func, **kw):
        try:
            with self.open_reader(**kw) as reader:
                for batch in iter_func(reader):
                    yield batch
        except (errors.ChecksumError, errors.MethodNotAllowed):
            # arrow tunnel not implemented or not supported
            kw.pop("arrow", None)
            with self.open_reader(**kw) as reader:
                for batch in iter_func(reader):
                    yield batch

    def to_pandas(
        self,
        columns=None,
        limit=None,
        start=None,
        count=None,
        n_process=1,
        quota_name=None,
        tags=None,
        **kwargs
    ):
        """
        Read instance data into pandas DataFrame. The limit argument follows definition
        of `open_reader` API.

        :param list columns: columns to read
        :param bool limit: if True, enable the limitation
        :param int start: start row index from 0
        :param int count: data count to read
        :param int n_process: number of processes to accelerate reading
        :param str quota_name: name of tunnel quota to use
        """
        try:
            import pyarrow as pa
        except ImportError:
            pa = None

        arrow = (pa is not None) and kwargs.pop("arrow", True)
        kw = dict(
            limit=limit,
            columns=columns,
            arrow=arrow,
            quota_name=quota_name,
            tags=tags,
            **kwargs
        )
        if limit is None:
            kw.pop("limit")

        def _it(reader):
            yield reader.to_pandas(start=start, count=count, n_process=n_process)

        return next(self._iter_reader_with_pandas(_it, **kw))

    def iter_pandas(
        self,
        columns=None,
        limit=None,
        batch_size=None,
        start=None,
        count=None,
        quota_name=None,
        tags=None,
        **kwargs
    ):
        """
        Iterate table data in blocks as pandas DataFrame. The limit argument
        follows definition of `open_reader` API.

        :param list columns: columns to read
        :param bool limit: if True, enable the limitation
        :param int batch_size: size of DataFrame batch to read
        :param int start: start row index from 0
        :param int count: data count to read
        :param str quota_name: name of tunnel quota to use
        """
        try:
            import pyarrow as pa
        except ImportError:
            pa = None

        batch_size = batch_size or options.tunnel.read_row_batch_size
        kw = dict(
            limit=limit,
            columns=columns,
            arrow=pa is not None,
            quota_name=quota_name,
            tags=tags,
            **kwargs
        )
        if limit is None:
            kw.pop("limit")

        def _it(reader):
            for batch in reader.iter_pandas(
                batch_size, start=start, count=count, columns=columns
            ):
                yield batch

        for batch in self._iter_reader_with_pandas(_it, **kw):
            yield batch
