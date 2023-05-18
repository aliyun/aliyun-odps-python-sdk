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

from ...errors import ODPSError
from ...tests.core import tn
from .. import Instance, TableSchema, Record

TEST_SESSION_WORKERS = 4
TEST_SESSION_WORKER_MEMORY = 512

TEST_TABLE_NAME = tn("_pyodps__session_test_table")
TEST_CREATE_SCHEMA = TableSchema.from_lists(['id'], ['string'])
TEST_DATA = [['1'], ['2'], ['3'], ['4'], ['5']]
TEST_SELECT_STRING = "select * from %s" % TEST_TABLE_NAME


def test_create_session(odps):
    sess_instance = odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
    assert sess_instance
    # wait to running
    try:
        sess_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + sess_instance.get_logview_address())
        print("Task results: " + str(sess_instance.get_task_results()))
        sess_instance.stop()
        raise ex
    # the status should keep consistent
    assert sess_instance.status == Instance.Status.RUNNING
    # finally stop it.
    sess_instance.stop()


def test_attach_session(odps):
    sess_instance = odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
    assert sess_instance
    # wait to running
    try:
        sess_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + sess_instance.get_logview_address())
        print("Task results: " + str(sess_instance.get_task_results()))
        sess_instance.stop()
        raise ex
    # the status should keep consistent
    assert sess_instance.status == Instance.Status.RUNNING
    att_instance = odps.attach_session(sess_instance._session_name)
    assert att_instance
    # wait to running
    try:
        att_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + att_instance.get_logview_address())
        print("Task results: " + str(att_instance.get_task_results()))
        sess_instance.stop()
        raise ex
    # the status should keep consistent
    assert att_instance.status == Instance.Status.RUNNING
    # finally stop it.
    sess_instance.stop()


def test_attach_default_session(odps):
    sess_instance = odps.default_session()
    assert sess_instance
    # wait to running
    try:
        sess_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + sess_instance.get_logview_address())
        print("Task results: " + str(sess_instance.get_task_results()))
        sess_instance.stop()
        raise ex
    # the status should keep consistent
    assert sess_instance.status == Instance.Status.RUNNING


def test_session_failing_sql(odps):
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    sess_instance = odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
    assert sess_instance
    # wait to running
    try:
        sess_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + sess_instance.get_logview_address())
        print("Task results: " + str(sess_instance.get_task_results()))
        sess_instance.stop()
        raise ex
    # the status should keep consistent
    assert sess_instance.status == Instance.Status.RUNNING
    select_inst = sess_instance.run_sql(TEST_SELECT_STRING)
    select_inst.wait_for_completion() # should return normally even the task is failed
    try:
        select_inst.wait_for_success()
        # should not reach here: wait_for_success should throw exception on failed instance
        assert False
    except ODPSError:
        pass  # good
    sess_instance.stop()


def test_direct_execute_failing_sql(odps):
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    # the default public session may not exist, so we create one beforehand
    sess_instance = odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
    assert sess_instance
    # wait to running
    try:
        sess_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + sess_instance.get_logview_address())
        print("Task results: " + str(sess_instance.get_task_results()))
        raise ex
    # the status should keep consistent
    assert sess_instance.status == Instance.Status.RUNNING
    select_inst = odps.run_sql_interactive(TEST_SELECT_STRING, service_name=sess_instance.name)
    select_inst.wait_for_completion() # should return normally even the task is failed
    try:
        select_inst.wait_for_success()
        # should not reach here: wait_for_success should throw exception on failed instance
        assert False
    except ODPSError:
        pass  # good
    finally:
        sess_instance.stop()


def test_session_sql(odps):
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    table = odps.create_table(TEST_TABLE_NAME, TEST_CREATE_SCHEMA)
    assert table
    sess_instance = odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
    assert sess_instance
    # wait to running
    try:
        sess_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + sess_instance.get_logview_address())
        print("Task results: " + str(sess_instance.get_task_results()))
        sess_instance.stop()
        raise ex
    # the status should keep consistent
    assert sess_instance.status == Instance.Status.RUNNING
    records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
    odps.write_table(table, 0, records)
    select_inst = sess_instance.run_sql(TEST_SELECT_STRING)
    select_inst.wait_for_success()
    rows = []
    try:
        with select_inst.open_reader() as rd:
            for each_row in rd:
                rows.append(each_row.values)
    except BaseException as ex:
        print("LOGVIEW: " + select_inst.get_logview_address())
        print("Task Result:" + str(select_inst.get_task_results()))
        sess_instance.stop()
        raise ex
    assert len(rows) == len(TEST_DATA)
    assert len(rows[0]) == len(TEST_DATA[0])
    for index in range(5):
        assert int(rows[index][0]) == int(TEST_DATA[index][0])
    # OK, clear up
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    sess_instance.stop()


def test_direct_execute_sql(odps):
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    table = odps.create_table(TEST_TABLE_NAME, TEST_CREATE_SCHEMA)
    assert table
    # the default public session may not exist, so we create one beforehand
    sess_instance = odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
    assert sess_instance
    # wait to running
    try:
        sess_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + sess_instance.get_logview_address())
        print("Task results: " + str(sess_instance.get_task_results()))
        sess_instance.stop()
        raise ex
    # the status should keep consistent
    assert sess_instance.status == Instance.Status.RUNNING
    records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
    odps.write_table(table, 0, records)
    select_inst = odps.run_sql_interactive(TEST_SELECT_STRING, service_name=sess_instance.name)
    select_inst.wait_for_success()
    rows = []
    try:
        with select_inst.open_reader() as rd:
            for each_row in rd:
                rows.append(each_row.values)
    except BaseException as ex:
        print("LOGVIEW: " + select_inst.get_logview_address())
        print("Task Result:" + str(select_inst.get_task_results()))
        sess_instance.stop()
        raise ex
    assert len(rows) == len(TEST_DATA)
    assert len(rows[0]) == len(TEST_DATA[0])
    for index in range(5):
        assert int(rows[index][0]) == int(TEST_DATA[index][0])
    # OK, clear up
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    sess_instance.stop()


def test_direct_execute_sql_fallback(odps):
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    table = odps.create_table(TEST_TABLE_NAME, TEST_CREATE_SCHEMA)
    assert table
    # the default public session may not exist, so we create one beforehand
    sess_instance = odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
    assert sess_instance
    # wait to running
    try:
        sess_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + sess_instance.get_logview_address())
        print("Task results: " + str(sess_instance.get_task_results()))
        sess_instance.stop()
        raise ex
    # the status should keep consistent
    assert sess_instance.status == Instance.Status.RUNNING
    records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
    odps.write_table(table, 0, records)
    hints = {"odps.mcqa.disable":"true"}
    select_inst = odps.run_sql_interactive_with_fallback(TEST_SELECT_STRING, service_name=sess_instance.name, hints=hints)
    select_inst.wait_for_success()
    rows = []
    try:
        with select_inst.open_reader() as rd:
            for each_row in rd:
                rows.append(each_row.values)
    except BaseException as ex:
        print("LOGVIEW: " + select_inst.get_logview_address())
        print("Task Result:" + str(select_inst.get_task_results()))
        sess_instance.stop()
        raise ex
    assert len(rows) == len(TEST_DATA)
    assert len(rows[0]) == len(TEST_DATA[0])
    for index in range(5):
        assert int(rows[index][0]) == int(TEST_DATA[index][0])
    # OK, clear up
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    sess_instance.stop()


def test_session_sql_with_instance_tunnel(odps):
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    table = odps.create_table(TEST_TABLE_NAME, TEST_CREATE_SCHEMA)
    assert table
    sess_instance = odps.create_session(TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY)
    assert sess_instance
    # wait to running
    try:
        sess_instance.wait_for_startup()
    except ODPSError as ex:
        print("LOGVIEW: " + sess_instance.get_logview_address())
        print("Task results: " + str(sess_instance.get_task_results()))
        sess_instance.stop()
        raise ex
    # the status should keep consistent
    assert sess_instance.status == Instance.Status.RUNNING
    records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
    odps.write_table(table, 0, records)
    select_inst = sess_instance.run_sql(TEST_SELECT_STRING)
    select_inst.wait_for_success()
    rows = []
    try:
        with select_inst.open_reader(tunnel=True) as rd:
            for each_row in rd:
                rows.append(each_row.values)
    except BaseException as ex:
        print("LOGVIEW: " + select_inst.get_logview_address())
        print("Task Result:" + str(select_inst.get_task_results()))
        sess_instance.stop()
        raise ex
    assert len(rows) == len(TEST_DATA)
    assert len(rows[0]) == len(TEST_DATA[0])
    for index in range(5):
        assert int(rows[index][0]) == int(TEST_DATA[index][0])
    # OK, clear up
    odps.delete_table(TEST_TABLE_NAME, if_exists=True)
    sess_instance.stop()
