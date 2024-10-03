#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2024 Alibaba Group Holding Ltd.
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
import random
import textwrap
import time
from datetime import datetime, timedelta

import mock
import pytest
import requests

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import pyarrow as pa
except ImportError:
    pa = None

from ... import compat, errors, options
from ... import types as odps_types
from ... import utils
from ...compat import six
from ...errors import ODPSError
from ...tests.core import (
    flaky,
    odps2_typed_case,
    pandas_case,
    pyarrow_case,
    tn,
    wait_filled,
)
from .. import Instance, SQLTask, TableSchema

expected_xml_template = """<?xml version="1.0" encoding="utf-8"?>
<Instance>
  <Job>
    <Priority>%(priority)s</Priority>
    <Tasks>
      <SQL>
        <Name>AnonymousSQLTask</Name>
        <Config>
          <Property>
            <Name>biz_id</Name>
            <Value>%(biz_id)s</Value>
          </Property>
          <Property>
            <Name>uuid</Name>
            <Value>%(uuid)s</Value>
          </Property>
          <Property>
            <Name>settings</Name>
            <Value>{"odps.sql.udf.strict.mode": "true"}</Value>
          </Property>
        </Config>
        <Query><![CDATA[%(query)s]]></Query>
      </SQL>
    </Tasks>
    <DAG>
      <RunMode>Sequence</RunMode>
    </DAG>
  </Job>
</Instance>
"""


class TunnelLimitedInstance(Instance):
    _exc = None

    def _open_tunnel_reader(self, **kw):
        self.wait_for_success()
        cls = type(self)
        if cls._exc is not None:
            if not isinstance(cls._exc, errors.NoPermission) or not kw.get("limit"):
                raise cls._exc
        return super(TunnelLimitedInstance, self)._open_tunnel_reader(**kw)


@pytest.fixture(autouse=True)
def wrap_options():
    try:
        options.connect_timeout = 10
        yield
    finally:
        options.connect_timeout = 120


def test_instances(odps):
    assert odps.get_project().instances is odps.get_project().instances

    size = len(list(itertools.islice(odps.list_instances(), 0, 5)))
    assert size >= 0

    instances = list(
        itertools.islice(odps.list_instances(status="running", only_owner=True), 0, 5)
    )
    assert len(instances) >= 0
    if len(instances) > 0:
        # fix: use _status instead of status to prevent from fetching the instance which is just terminated
        assert (
            all(instance._status == Instance.Status.RUNNING for instance in instances)
            is True
        )
        assert len(set(instance.owner for instance in instances)) == 1

    start_time = time.time() - 10 * 24 * 3600
    end_time = time.time() - 24 * 3600
    instances = list(
        itertools.islice(
            odps.list_instances(start_time=start_time, end_time=end_time), 0, 5
        )
    )
    assert len(instances) >= 0


def test_list_instances_in_page(odps):
    test_table = tn("pyodps_t_tmp_list_instances_in_page")

    delay_udf = textwrap.dedent(
        """
    from odps.udf import annotate
    import sys
    import time

    @annotate("bigint->bigint")
    class Delayer(object):
       def evaluate(self, arg0):
           print('Start Logging')
           sys.stdout.flush()
           time.sleep(45)
           print('End Logging')
           sys.stdout.flush()
           return arg0
    """
    )
    resource_name = tn("test_delayer_function_resource")
    function_name = tn("test_delayer_function")

    if odps.exist_resource(resource_name + ".py"):
        odps.delete_resource(resource_name + ".py")
    res = odps.create_resource(resource_name + ".py", "py", file_obj=delay_udf)

    if odps.exist_function(function_name):
        odps.delete_function(function_name)
    fun = odps.create_function(
        function_name, class_type=resource_name + ".Delayer", resources=[res]
    )

    data = [[random.randint(0, 1000)] for _ in compat.irange(100)]
    odps.delete_table(test_table, if_exists=True)
    t = odps.create_table(test_table, TableSchema.from_lists(["num"], ["bigint"]))
    odps.write_table(t, data)

    instance = odps.run_sql(
        "select sum({0}(num)), 1 + '1' as warn_col from {1} group by num".format(
            function_name, test_table
        )
    )

    try:
        assert instance.status == Instance.Status.RUNNING
        assert instance.id in [
            it.id
            for it in odps.get_project().instances.iterate(
                status=Instance.Status.RUNNING,
                start_time=datetime.now() - timedelta(days=2),
                end_time=datetime.now() + timedelta(days=1),
                max_items=20,
            )
        ]

        wait_filled(lambda: instance.tasks)
        task = instance.tasks[0]
        task.put_info("testInfo", "TestInfo")
        with pytest.raises(errors.EmptyTaskInfoError):
            task.put_info("testInfo", "TestInfo", raise_empty=True)
        assert task.warnings is not None

        wait_filled(lambda: task.workers, 30)
        wait_filled(lambda: [w.log_id for w in task.workers if w.log_id], 30)
        assert task.workers[0].get_log("stdout") is not None
    finally:
        try:
            instance.stop()
        except:
            pass
        res.drop()
        fun.drop()
        t.drop()


def test_instance_exists(odps):
    non_exists_instance = "a_non_exists_instance"
    assert odps.exist_instance(non_exists_instance) is False


def test_instance(odps):
    instances = odps.list_instances(status=Instance.Status.TERMINATED)
    instance = next(inst for inst in instances if inst.is_successful())

    assert instance is odps.get_instance(instance.name)

    assert instance._getattr("name") is not None
    assert instance._getattr("owner") is not None
    assert instance._getattr("start_time") is not None
    assert instance._getattr("end_time") is not None
    assert instance._getattr("_status") is not None
    assert instance._status == Instance.Status.TERMINATED

    instance.reload()
    assert instance.status == Instance.Status.TERMINATED
    assert not instance.is_running()
    assert instance.is_terminated()

    task_names = instance.get_task_names()

    task_statuses = instance.get_task_statuses()
    for task_status in task_statuses.values():
        assert task_status.status in (
            Instance.Task.TaskStatus.CANCELLED,
            Instance.Task.TaskStatus.FAILED,
            Instance.Task.TaskStatus.SUCCESS,
        )
    for task_status in instance._tasks:
        assert task_status.name in task_names
        assert len(task_status.type) >= 0
        assert task_status.start_time >= instance.start_time
        assert task_status.end_time <= instance.end_time

    results = instance.get_task_results()
    for name, result in results.items():
        assert name in task_names
        assert isinstance(result, str)

    assert instance.priority >= 0


def test_create_instance_xml(odps):
    instances = odps._project.instances

    uuid = "359696d4-ac73-4e6c-86d1-6649b01f1a22"
    query = "select * from dual if fake < 1;"
    priority = 5

    try:
        options.biz_id = "012345"

        task = SQLTask(query=query)
        job = instances._create_job(task=task, priority=priority, uuid_=uuid)
        xml = instances._get_submit_instance_content(job)
        expected_xml = expected_xml_template % {
            "query": query,
            "uuid": uuid,
            "priority": priority,
            "biz_id": options.biz_id,
        }
        assert utils.to_str(xml) == utils.to_str(expected_xml)
    finally:
        options.biz_id = None


def test_create_instance(odps):
    test_table = tn("pyodps_t_tmp_create_instance")

    task = SQLTask(query="drop table if exists %s" % test_table)
    instance = odps._project.instances.create(task=task)
    assert instance.get_sql_query().rstrip(";") == task.query.rstrip(";")
    instance.wait_for_completion()
    assert instance.is_successful() is True
    assert odps.exist_table(test_table) is False
    assert instance.start_time < datetime.now()
    assert instance.start_time > datetime.now() - timedelta(hours=1)

    task = SQLTask(query="create table %s(id string);" % test_table)
    instance = odps._project.instances.create(task=task)
    instance.wait_for_completion()
    assert instance.is_successful() is True
    assert odps.exist_table(test_table) is True

    instance = odps.execute_sql("select id `中文标题` from %s" % test_table)
    assert instance.is_successful() is True

    instance = odps.execute_sql("drop table %s" % test_table)
    assert instance.is_successful() is True
    assert odps.exist_table(test_table) is False

    tasks = instance.get_tasks()
    assert any(map(lambda task: isinstance(task, SQLTask), tasks)) is True

    for name in instance.get_task_names():
        assert instance.get_task_detail(name) is not None
        assert instance.get_task_detail2(name) is not None

    # test stop
    pytest.raises(errors.InvalidStateSetting, instance.stop)


def test_read_sql_instance(odps):
    test_table = tn("pyodps_t_tmp_read_sql_instance")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table, TableSchema.from_lists(["size"], ["bigint"]), if_not_exists=True
    )
    odps.write_table(table, 0, [table.new_record([1]), table.new_record([2])])
    odps.write_table(table, [table.new_record([3])])

    instance = odps.execute_sql("select * from %s" % test_table)
    with instance.open_reader(table.table_schema) as reader:
        assert len(list(reader[::2])) == 2
    with instance.open_reader(table.table_schema) as reader:
        assert len(list(reader[1::2])) == 1

    hints = {"odps.sql.mapper.split.size": "16"}
    instance = odps.run_sql(
        "select sum(size) as count from %s" % test_table, hints=hints
    )

    while (
        len(instance.get_task_names()) == 0
        or compat.lvalues(instance.get_task_statuses())[0].status
        == Instance.Task.TaskStatus.WAITING
    ):
        continue

    while True:
        progress = instance.get_task_progress(instance.get_task_names()[0])
        if len(progress.stages) == 0:
            continue
        assert len(progress.get_stage_progress_formatted_string().split()) > 2
        break

    with mock.patch(
        "odps.models.instance.Instance.is_terminated", new=lambda *_, **__: False
    ):
        with pytest.raises(errors.WaitTimeoutError):
            instance.wait_for_completion(timeout=3, max_interval=3)

    instance.wait_for_success()
    assert (
        json.loads(instance.tasks[0].properties["settings"])[
            "odps.sql.mapper.split.size"
        ]
        == hints["odps.sql.mapper.split.size"]
    )
    assert instance.tasks[0].summary is not None

    with instance.open_reader(
        TableSchema.from_lists(["count"], ["bigint"]), tunnel=False
    ) as reader:
        records = list(reader)
        assert len(records) == 1
        assert records[0]["count"] == 6

    with instance.open_reader(tunnel=True) as reader:
        records = list(reader)
        assert len(records) == 1
        assert records[0]["count"] == 6

    with instance.open_reader(tunnel=False) as reader:
        records = list(reader)
        assert len(records) == 1
        assert records[0]["count"] == "6"

    if pd is not None:
        with instance.open_reader(tunnel=True) as reader:
            pd_data = reader.to_pandas()
            assert len(pd_data) == 1

        with instance.open_reader(tunnel=True) as reader:
            pd_data = reader.to_pandas(n_process=2)
            assert len(pd_data) == 1

        with instance.open_reader(tunnel=False) as reader:
            pd_data = reader.to_pandas()
            assert len(pd_data) == 1

    if pa is not None:
        with instance.open_reader(tunnel=True, arrow=True) as reader:
            pd_data = reader.to_pandas()
            assert len(pd_data) == 1

    table.drop()


@pandas_case
@pyarrow_case
def test_instance_to_pandas(odps):
    test_table = tn("pyodps_t_tmp_inst_to_pandas")
    odps.delete_table(test_table, if_exists=True)
    data = pd.DataFrame(
        [[0, 134, "a", "a"], [1, 24, "a", "b"], [2, 131, "a", "a"], [3, 141, "a", "b"]],
        columns=["a", "b", "c", "d"],
    )
    odps.write_table(test_table, data, create_table=True, lifecycle=1)

    instance = odps.execute_sql("select * from %s" % test_table)

    result = instance.to_pandas(columns=["a", "b"])
    pd.testing.assert_frame_equal(result, data[["a", "b"]])

    # test fallback when arrow format not supported
    raised_list = [False]

    def _new_to_pandas(self, *_, **__):
        raised_list[0] = True
        raise errors.ChecksumError("Checksum invalid")

    with mock.patch(
        "odps.models.readers.TunnelArrowReader.to_pandas", new=_new_to_pandas
    ):
        result = instance.to_pandas(columns=["a", "b"])
        assert raised_list[0]
        pd.testing.assert_frame_equal(result, data[["a", "b"]])

    # test fallback when instance tunnel not supported
    raised_list = [False]

    def _new_open_tunnel_reader(self, *_, **__):
        raised_list[0] = True
        raise errors.InvalidProjectTable("InvalidProjectTable")

    with mock.patch(
        "odps.models.instance.Instance._open_tunnel_reader", new=_new_open_tunnel_reader
    ):
        result = instance.to_pandas(columns=["a", "b"])
        assert raised_list[0]
        pd.testing.assert_frame_equal(result, data[["a", "b"]])

    batches = []
    for batch in instance.iter_pandas(columns=["a", "b"], batch_size=2):
        assert len(batch) == 2
        batches.append(batch)
    assert len(batches) == 2

    odps.delete_table(test_table, if_exists=True)


def test_limited_instance_tunnel(odps):
    test_table = tn("pyodps_t_tmp_limit_instance_tunnel")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table, TableSchema.from_lists(["size"], ["bigint"]), if_not_exists=True
    )
    odps.write_table(table, 0, [table.new_record([1]), table.new_record([2])])
    odps.write_table(table, [table.new_record([3])])

    instance = odps.execute_sql("select * from %s" % test_table)
    instance = TunnelLimitedInstance(
        client=instance._client, parent=instance.parent, name=instance.id
    )

    TunnelLimitedInstance._exc = errors.InvalidArgument("Mock fallback error")
    pytest.raises(errors.InvalidArgument, instance.open_reader, tunnel=True)
    with instance.open_reader() as reader:
        assert hasattr(reader, "raw") is True

    TunnelLimitedInstance._exc = requests.Timeout("Mock timeout")
    pytest.raises(requests.Timeout, instance.open_reader, tunnel=True)
    with instance.open_reader() as reader:
        assert hasattr(reader, "raw") is True

    TunnelLimitedInstance._exc = errors.InstanceTypeNotSupported(
        "Mock instance not supported"
    )
    pytest.raises(errors.InstanceTypeNotSupported, instance.open_reader, tunnel=True)
    with instance.open_reader() as reader:
        assert hasattr(reader, "raw") is True

    TunnelLimitedInstance._exc = errors.NoPermission("Mock permission error")
    pytest.raises(errors.NoPermission, instance.open_reader, limit=False)
    with instance.open_reader() as reader:
        assert hasattr(reader, "raw") is False


def test_read_sql_write(odps):
    test_table = tn("pyodps_t_tmp_read_sql_instance_write")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table, TableSchema.from_lists(["size"], ["bigint"]), if_not_exists=True
    )
    odps.write_table(table, 0, [table.new_record([1]), table.new_record([2])])
    odps.write_table(table, [table.new_record([3])])

    test_table2 = tn("pyodps_t_tmp_read_sql_instance_write2")
    odps.delete_table(test_table2, if_exists=True)
    table2 = odps.create_table(test_table2, table.table_schema)

    try:
        with odps.execute_sql("select * from %s" % test_table).open_reader() as reader:
            with table2.open_writer() as writer:
                for record in reader:
                    writer.write(table2.new_record(record.values))
    finally:
        table.drop()
        table2.drop()


def test_read_binary_sql_instance(odps):
    try:
        options.tunnel.string_as_binary = True
        test_table = tn("pyodps_t_tmp_read_binary_sql_instance")
        odps.delete_table(test_table, if_exists=True)
        table = odps.create_table(
            test_table,
            TableSchema.from_lists(["size", "name"], ["bigint", "string"]),
            if_not_exists=True,
        )

        data = [
            [
                1,
                u"中".encode("utf-8")
                + b"\\\\n\\\n"
                + u"文".encode("utf-8")
                + b" ,\r\xe9",
            ],
            [
                2,
                u"测试".encode("utf-8")
                + b"\x00\x01\x02"
                + u"数据".encode("utf-8")
                + b"\xe9",
            ],
        ]
        odps.write_table(table, 0, [table.new_record(it) for it in data])

        with odps.execute_sql("select name from %s" % test_table).open_reader(
            tunnel=False
        ) as reader:
            read_data = sorted([r[0] for r in reader])
            expected_data = sorted([r[1] for r in data])

            assert list(read_data) == list(expected_data)

        table.drop()
    finally:
        options.tunnel.string_as_binary = False


def test_read_non_ascii_sql_instance(odps):
    test_table = tn("pyodps_t_tmp_read_non_ascii_sql_instance")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table,
        TableSchema.from_lists(["size", "name"], ["bigint", "string"]),
        if_not_exists=True,
    )

    data = [[1, "中\\\\n\\\n文 ,\r "], [2, "测试\x00\x01\x02数据"]]
    odps.write_table(table, 0, [table.new_record(it) for it in data])

    with odps.execute_sql("select name from %s" % test_table).open_reader(
        tunnel=False
    ) as reader:
        read_data = sorted([utils.to_str(r[0]) for r in reader])
        expected_data = sorted([utils.to_str(r[1]) for r in data])

        assert list(read_data) == list(expected_data)

    table.drop()


def test_read_map_array_sql_instance(odps):
    test_table = tn("pyodps_t_tmp_read_map_array_sql_instance")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table,
        TableSchema.from_lists(
            ["idx", "map_col", "array_col"],
            [
                "bigint",
                odps_types.Map(odps_types.string, odps_types.string),
                odps_types.Array(odps_types.string),
            ],
        ),
    )

    data = [
        [0, {"key1": "value1", "key2": "value2"}, ["item1", "item2", "item3"]],
        [1, {"key3": "value3", "key4": "value4"}, ["item4", "item5"]],
    ]
    odps.write_table(test_table, data)

    inst = odps.execute_sql("select * from %s" % test_table)

    with inst.open_reader(table.table_schema, tunnel=False) as reader:
        read_data = [list(r.values) for r in reader]
        read_data = sorted(read_data, key=lambda r: r[0])
        expected_data = sorted(data, key=lambda r: r[0])

        assert list(read_data) == list(expected_data)

    with inst.open_reader(table.table_schema, tunnel=True) as reader:
        read_data = [list(r.values) for r in reader]
        read_data = sorted(read_data, key=lambda r: r[0])
        expected_data = sorted(data, key=lambda r: r[0])

        assert list(read_data) == list(expected_data)

    table.drop()


def test_sql_alias_instance(odps):
    test_table = tn("pyodps_t_tmp_sql_aliases_instance")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table, TableSchema.from_lists(["size"], ["bigint"]), if_not_exists=True
    )

    data = [[1]]
    odps.write_table(table, 0, data)

    res_name1 = tn("pyodps_t_tmp_resource_1")
    res_name2 = tn("pyodps_t_tmp_resource_2")
    try:
        odps.delete_resource(res_name1)
    except ODPSError:
        pass
    try:
        odps.delete_resource(res_name2)
    except ODPSError:
        pass
    res1 = odps.create_resource(res_name1, "file", file_obj="1")
    res2 = odps.create_resource(res_name2, "file", file_obj="2")

    test_func_content = (
        """
    from odps.udf import annotate
    from odps.distcache import get_cache_file

    @annotate('bigint->bigint')
    class Example(object):
        def __init__(self):
            self.n = int(get_cache_file('%s').read())

        def evaluate(self, arg):
            return arg + self.n
    """
        % res_name1
    )
    test_func_content = textwrap.dedent(test_func_content)

    py_res_name = tn("pyodps_t_tmp_func_res")
    try:
        odps.delete_resource(py_res_name + ".py")
    except ODPSError:
        pass

    py_res = odps.create_resource(py_res_name + ".py", "py", file_obj=test_func_content)

    test_func_name = tn("pyodps_t_tmp_func_1")
    try:
        odps.delete_function(test_func_name)
    except ODPSError:
        pass
    func = odps.create_function(
        test_func_name,
        class_type="{0}.Example".format(py_res_name),
        resources=[py_res_name + ".py", res_name1],
    )

    for i in range(1, 3):
        aliases = None
        if i == 2:
            aliases = {res_name1: res_name2}
        with odps.execute_sql(
            "select %s(size) from %s" % (test_func_name, test_table), aliases=aliases
        ).open_reader() as reader:
            data = reader[0]
            assert int(data[0]) == i + 1

    for obj in (func, py_res, res1, res2, table):
        obj.drop()


def test_read_non_select_sql_instance(odps):
    test_table = tn("pyodps_t_tmp_read_non_select_sql_instance")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table,
        TableSchema.from_lists(["size"], ["bigint"], ["pt"], ["string"]),
        if_not_exists=True,
    )
    pt_spec = "pt=20170410"
    table.create_partition(pt_spec)

    inst = odps.execute_sql("desc %s" % test_table)

    with pytest.raises(
        (Instance.DownloadSessionCreationError, errors.InstanceTypeNotSupported)
    ):
        inst.open_reader(tunnel=True)

    reader = inst.open_reader()
    assert hasattr(reader, "raw") is True

    inst = odps.execute_sql("show partitions %s" % test_table)
    reader = inst.open_reader()
    assert hasattr(reader, "raw") is True
    assert utils.to_text(pt_spec) in utils.to_text(reader.raw)


@pandas_case
def test_instance_result_to_result_frame(odps):
    test_table = tn("pyodps_t_tmp_instance_result_to_pd")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table, TableSchema.from_lists(["size"], ["bigint"]), if_not_exists=True
    )
    odps.write_table(table, [[1], [2], [3]])

    inst = odps.execute_sql("select * from %s" % test_table)
    tunnel_pd = inst.open_reader(tunnel=True).to_pandas()
    result_pd = inst.open_reader(tunnel=False).to_pandas()
    assert tunnel_pd.values.tolist() == result_pd.values.tolist()


def test_instance_logview(odps):
    instance = odps.run_sql("drop table if exists non_exist_table_name")
    assert isinstance(odps.get_logview_address(instance.id, 12), six.string_types)


@flaky(max_runs=3)
def test_instance_queueing_info(odps):
    instance = odps.run_sql("select * from dual")
    queue_info, resp = instance._get_queueing_info()
    if json.loads(resp.content if six.PY2 else resp.text):
        assert queue_info.instance is instance
        assert queue_info.instance_id is not None
        assert queue_info.priority is not None
        assert queue_info.project is not None
        assert queue_info.start_time is not None
        assert queue_info.user_account is not None
        assert queue_info.status in (
            Instance.InstanceQueueingInfo.Status.RUNNING,
            Instance.InstanceQueueingInfo.Status.SUSPENDED,
            Instance.InstanceQueueingInfo.Status.TERMINATED,
            Instance.InstanceQueueingInfo.Status.UNKNOWN,
        )
        assert queue_info.sub_status_history is not None


@flaky(max_runs=3)
def test_instance_queueing_infos(odps):
    odps.run_sql("select * from dual")

    infos = [
        info
        for i, info in compat.izip(
            itertools.count(0), odps.list_instance_queueing_infos()
        )
        if i < 5
    ]
    if len(infos) > 0:
        assert isinstance(infos[0], Instance.InstanceQueueingInfo)
        assert infos[0].instance_id is not None
        assert isinstance(infos[0].instance, Instance)
        assert infos[0].instance.id == infos[0].instance_id


@odps2_typed_case
def test_sql_cost_instance(odps):
    test_table = tn("pyodps_t_tmp_sql_cost_instance")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table, TableSchema.from_lists(["size"], ["bigint"]), if_not_exists=True
    )
    odps.write_table(table, [[1], [2], [3]])

    sql_cost = odps.execute_sql_cost("select * from %s" % test_table)
    assert isinstance(sql_cost, Instance.SQLCost)
    assert sql_cost.udf_num == 0
    assert sql_cost.complexity == 1.0
    assert sql_cost.input_size >= 100

    test_table = tn("pyodps_t_tmp_sql_cost_odps2_instance")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table, TableSchema.from_lists(["size"], ["tinyint"]), if_not_exists=True
    )
    odps.write_table(table, [[1], [2], [3]])

    sql_cost = odps.execute_sql_cost("select * from %s" % test_table)
    assert isinstance(sql_cost, Instance.SQLCost)
    assert sql_cost.udf_num == 0
    assert sql_cost.complexity == 1.0
    assert sql_cost.input_size >= 100


def test_instance_progress_log(odps):
    test_table = tn("pyodps_t_tmp_sql_cost_instance")
    odps.delete_table(test_table, if_exists=True)
    table = odps.create_table(
        test_table, TableSchema.from_lists(["size"], ["bigint"]), if_not_exists=True
    )
    odps.write_table(table, [[1], [2], [3]])

    logs = []

    try:
        options.verbose = True
        options.verbose_log = logs.append
        options.progress_time_interval = 0.1

        inst = odps.run_sql("select * from %s where size > 0" % test_table)
        inst.wait_for_success(interval=0.1, blocking=False)
        assert any("instance" in log.lower() for log in logs)
        assert any("_job_" in log.lower() for log in logs)
    finally:
        options.verbose = False
        options.verbose_log = None
        options.progress_time_interval = 5 * 60


def test_sql_statement_error(odps):
    statement = "WRONG_SQL"
    try:
        odps.run_sql(statement)
    except errors.ParseError as ex:
        assert ex.statement == statement
        assert statement in str(ex)
