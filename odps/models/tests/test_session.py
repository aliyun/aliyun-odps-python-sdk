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

import contextlib
import logging
import os
import sys
import tempfile
import time

import mock
import pytest

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    from filelock import FileLock
    from filelock import Timeout as FLTimeout
except ImportError:
    FileLock = None

from ... import ODPS, errors
from ...errors import InvalidStateSetting, ODPSError
from ...tests.core import get_test_unique_name, tn
from .. import Instance, Quota, Record, TableSchema
from ..session import FallbackMode, FallbackPolicy

logger = logging.getLogger(__name__)
is_windows = sys.platform.lower().startswith("win")

TEST_SESSION_WORKERS = 4
TEST_SESSION_WORKER_MEMORY = 512

TEST_TABLE_NAME = tn("_pyodps__session_test_table")
TEST_CREATE_SCHEMA = TableSchema.from_lists(["id"], ["string"])
TEST_DATA = [["1"], ["2"], ["3"], ["4"], ["5"]]
TEST_SELECT_STRING = "select * from %(table_name)s"


@pytest.fixture(autouse=True)
def auto_stop():
    session_insts = []
    old_create_mcqa_session = ODPS._create_mcqa_session

    def _create_session_patch(self, *args, **kwargs):
        result = old_create_mcqa_session(self, *args, **kwargs)
        if result:
            session_insts.append(result)
        return result

    with mock.patch("odps.core.ODPS._create_mcqa_session", new=_create_session_patch):
        lock = None
        if FileLock and not is_windows:
            lock_file = os.path.join(tempfile.gettempdir(), "pyodps_test_session.lock")
            if not os.path.exists(lock_file):
                try:
                    open(lock_file, "wb").close()
                except OSError:
                    pass
                try:
                    os.chmod(lock_file, 0o777)
                except OSError:
                    pass
            lock = FileLock(lock_file)
        try:
            if lock:
                for trial in range(5):
                    try:
                        lock.acquire(30)
                        break
                    except (OSError, FLTimeout):
                        if trial == 4:
                            raise
                        time.sleep(1)
            yield
        finally:
            if lock:
                lock.release()
    for inst in session_insts:
        try:
            inst.stop()
        except InvalidStateSetting:
            pass


@contextlib.contextmanager
def _dump_instance_results(instance):
    try:
        yield
    except:
        logger.error("LOGVIEW: " + instance.get_logview_address())
        logger.error("Task results: %s", instance.get_task_results())
        try:
            instance.stop()
        except InvalidStateSetting:
            pass
        raise


def _wait_session_startup(session_instance):
    with _dump_instance_results(session_instance):
        session_instance.wait_for_startup(timeout=60)
    assert session_instance.status == Instance.Status.RUNNING


def test_create_mcqa_session(odps):
    sess_instance = odps._create_mcqa_session(
        TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY
    )
    assert sess_instance
    with mock.patch(
        "odps.models.instance.Instance.is_running", new=lambda *_, **__: False
    ):
        with pytest.raises(errors.WaitTimeoutError):
            sess_instance.wait_for_startup(timeout=2, max_interval=1)
    # wait to running
    _wait_session_startup(sess_instance)


def test_attach_mcqa_session(odps):
    sess_instance = odps._create_mcqa_session(
        TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY
    )
    assert sess_instance
    # wait to running
    _wait_session_startup(sess_instance)

    att_instance = odps._attach_mcqa_session(sess_instance._session_name)
    assert att_instance
    # wait to running
    _wait_session_startup(sess_instance)


def test_attach_default_session(odps):
    sess_instance = odps._get_default_mcqa_session()
    assert sess_instance
    # wait to running
    _wait_session_startup(sess_instance)


def test_session_failing_sql(odps):
    table_name = TEST_TABLE_NAME + get_test_unique_name(5)
    odps.delete_table(table_name, if_exists=True)
    sess_instance = odps._create_mcqa_session(
        TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY
    )
    assert sess_instance
    # wait to running
    _wait_session_startup(sess_instance)

    select_inst = sess_instance.run_sql(
        TEST_SELECT_STRING % dict(table_name=table_name)
    )
    select_inst.wait_for_completion()  # should return normally even the task is failed
    with pytest.raises(ODPSError):
        # wait_for_success should throw exception on failed instance
        select_inst.wait_for_success()


def test_direct_execute_failing_sql(odps):
    table_name = TEST_TABLE_NAME + get_test_unique_name(5)
    odps.delete_table(table_name, if_exists=True)
    # the default public session may not exist, so we create one beforehand
    sess_instance = odps._create_mcqa_session(
        TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY
    )
    assert sess_instance
    # wait to running
    _wait_session_startup(sess_instance)

    select_inst = odps.run_sql_interactive(
        TEST_SELECT_STRING % dict(table_name=table_name),
        service_name=sess_instance.name,
    )
    select_inst.wait_for_completion()  # should return normally even the task is failed

    with pytest.raises(ODPSError):
        # wait_for_success should throw exception on failed instance
        select_inst.wait_for_success()


def test_session_sql(odps):
    table_name = TEST_TABLE_NAME + get_test_unique_name(5)
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(table_name, TEST_CREATE_SCHEMA)
    assert table
    sess_instance = odps._create_mcqa_session(
        TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY
    )
    assert sess_instance
    # wait to running
    _wait_session_startup(sess_instance)

    records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
    odps.write_table(table, 0, records)
    select_inst = sess_instance.run_sql(
        TEST_SELECT_STRING % dict(table_name=table_name)
    )
    select_inst.wait_for_success()
    rows = []

    with _dump_instance_results(select_inst), select_inst.open_reader() as rd:
        for each_row in rd:
            rows.append(each_row.values)

    if pd is not None:
        with _dump_instance_results(select_inst), select_inst.open_reader(
            tunnel=True
        ) as rd:
            pd_result = rd.to_pandas()
        pd.testing.assert_frame_equal(
            pd_result, pd.DataFrame(TEST_DATA, columns=["id"])
        )

        with _dump_instance_results(select_inst):
            pd_result = select_inst.to_pandas()
        pd.testing.assert_frame_equal(
            pd_result, pd.DataFrame(TEST_DATA, columns=["id"])
        )

    assert len(rows) == len(TEST_DATA)
    assert len(rows[0]) == len(TEST_DATA[0])
    for index in range(5):
        assert int(rows[index][0]) == int(TEST_DATA[index][0])
    # OK, clear up
    odps.delete_table(table_name, if_exists=True)


def test_direct_execute_sql(odps):
    table_name = TEST_TABLE_NAME + get_test_unique_name(5)
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(table_name, TEST_CREATE_SCHEMA)
    assert table
    # the default public session may not exist, so we create one beforehand
    sess_instance = odps._create_mcqa_session(
        TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY
    )
    assert sess_instance
    # wait to running
    _wait_session_startup(sess_instance)

    records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
    odps.write_table(table, 0, records)
    test_sql = TEST_SELECT_STRING % dict(table_name=table_name)
    select_inst = odps.run_sql_interactive(test_sql, service_name=sess_instance.name)

    assert "subQuery" in select_inst.get_logview_address()
    assert select_inst._get_queueing_info()
    time.sleep(3)
    assert select_inst.get_sql_query().rstrip(";") == test_sql.rstrip(";")
    assert select_inst.get_task_detail2(select_inst._session_task_name)

    rows = []

    with _dump_instance_results(select_inst), select_inst.open_reader() as rd:
        for each_row in rd:
            rows.append(each_row.values)

    assert len(rows) == len(TEST_DATA)
    assert len(rows[0]) == len(TEST_DATA[0])
    for index in range(5):
        assert int(rows[index][0]) == int(TEST_DATA[index][0])
    # OK, clear up
    odps.delete_table(table_name, if_exists=True)


def test_direct_execute_sql_fallback(odps):
    table_name = TEST_TABLE_NAME + get_test_unique_name(5)
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(table_name, TEST_CREATE_SCHEMA)
    assert table
    # the default public session may not exist, so we create one beforehand
    sess_instance = odps._create_mcqa_session(
        TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY
    )
    assert sess_instance
    # wait to running
    _wait_session_startup(sess_instance)

    records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
    odps.write_table(table, 0, records)
    hints = {"odps.mcqa.disable": "true"}

    test_sql = TEST_SELECT_STRING % dict(table_name=table_name)

    with pytest.raises(ODPSError):
        odps.execute_sql_interactive(
            test_sql,
            service_name=sess_instance.name,
            hints=hints,
            fallback=False,
        )
    with pytest.raises(ODPSError):
        odps.execute_sql_interactive(
            test_sql,
            service_name=sess_instance.name,
            hints=hints,
            fallback="noresource",
        )
    with pytest.raises(ODPSError):
        odps.execute_sql_interactive(
            test_sql,
            service_name=sess_instance.name,
            hints=hints,
            fallback={"generic", "noresource"},
        )

    select_inst = odps.execute_sql_interactive(
        test_sql, service_name=sess_instance.name, hints=hints, fallback=True
    )
    select_inst.wait_for_success()
    rows = []

    with _dump_instance_results(select_inst), select_inst.open_reader() as rd:
        for each_row in rd:
            rows.append(each_row.values)

    assert len(rows) == len(TEST_DATA)
    assert len(rows[0]) == len(TEST_DATA[0])
    for index in range(5):
        assert int(rows[index][0]) == int(TEST_DATA[index][0])
    # OK, clear up
    odps.delete_table(table_name, if_exists=True)


def test_session_sql_with_instance_tunnel(odps):
    table_name = TEST_TABLE_NAME + get_test_unique_name(5)
    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(table_name, TEST_CREATE_SCHEMA)
    assert table
    sess_instance = odps._create_mcqa_session(
        TEST_SESSION_WORKERS, TEST_SESSION_WORKER_MEMORY
    )
    assert sess_instance
    # wait to running
    _wait_session_startup(sess_instance)

    records = [Record(schema=TEST_CREATE_SCHEMA, values=values) for values in TEST_DATA]
    odps.write_table(table, 0, records)
    select_inst = sess_instance.run_sql(
        TEST_SELECT_STRING % dict(table_name=table_name)
    )
    select_inst.wait_for_success()
    rows = []

    with _dump_instance_results(select_inst), select_inst.open_reader(
        tunnel=True
    ) as rd:
        for each_row in rd:
            rows.append(each_row.values)

    assert len(rows) == len(TEST_DATA)
    assert len(rows[0]) == len(TEST_DATA[0])
    for index in range(5):
        assert int(rows[index][0]) == int(TEST_DATA[index][0])
    # OK, clear up
    odps.delete_table(table_name, if_exists=True)


def test_fallback_policy():
    enabled_set = ["unsupported", "generic"]

    all_policy = FallbackPolicy("all")
    default_policy = FallbackPolicy("default")
    set_policy = FallbackPolicy(enabled_set)
    set_str_policy = FallbackPolicy(",".join(enabled_set))

    for policy_name in "unsupported,upgrading,noresource,timeout,generic".split(","):
        policy = FallbackPolicy(policy_name)
        assert getattr(policy, policy_name)
        assert getattr(all_policy, policy_name)
        if policy_name != "generic":
            assert getattr(default_policy, policy_name)
        if policy_name in enabled_set:
            assert getattr(set_policy, policy_name)
            assert getattr(set_str_policy, policy_name)
            assert policy_name in repr(set_policy)

        assert (
            policy.get_mode_from_exception(errors.SQARetryError("Retry"))
            == FallbackMode.INTERACTIVE
        )
        assert (
            policy.get_mode_from_exception(errors.ODPSError("Job is cancelled")) is None
        )
    assert (
        all_policy.get_mode_from_exception(errors.ODPSError("MiscError"))
        == FallbackMode.OFFLINE
    )
    assert set_policy.get_mode_from_exception(errors.ODPSError("MiscError")) is None
    assert (
        set_policy.get_mode_from_exception(errors.SQAGenericError("MiscError"))
        is FallbackMode.OFFLINE
    )
    assert (
        default_policy.get_mode_from_exception(errors.SQAGenericError("MiscError"))
        is None
    )
    assert (
        default_policy.get_mode_from_exception(
            errors.SQAUnsupportedFeature("UnsupportedFeature")
        )
        is FallbackMode.OFFLINE
    )
    assert (
        default_policy.get_mode_from_exception(
            errors.SQAServiceUnavailable("ServiceUnavailable")
        )
        is FallbackMode.OFFLINE
    )
    assert (
        default_policy.get_mode_from_exception(
            errors.SQAResourceNotEnough("ResourceNotEnough")
        )
        is FallbackMode.OFFLINE
    )
    assert (
        default_policy.get_mode_from_exception(errors.SQAQueryTimedout("QueryTimedout"))
        is FallbackMode.OFFLINE
    )


def test_reuse_session(odps):
    inst = odps.run_sql_interactive("select * from dual", force_reattach=True)
    inst.wait_for_completion()
    new_odps = ODPS(account=odps.account, project=odps.project, endpoint=odps.endpoint)
    new_inst = new_odps.run_sql_interactive("select * from dual")
    assert inst.id == new_inst.id


def test_mcqa_v2_session(odps_with_mcqa2):
    odps = odps_with_mcqa2
    table_name = TEST_TABLE_NAME + get_test_unique_name(5)

    odps.delete_table(table_name, if_exists=True)
    table = odps.create_table(table_name, TEST_CREATE_SCHEMA)

    test_data = [["abcde"], ["fghijkl"]]

    with table.open_writer() as writer:
        writer.write(test_data)

    raw_reload = Quota.reload

    def _mock_quota_reload(self):
        res = raw_reload(self)
        self._mcqa_conn_header = None
        # make sure fields can be loaded again
        self._loaded = False
        return res

    with mock.patch("odps.models.quota.Quota.reload", new=_mock_quota_reload):
        with pytest.raises(ODPSError) as exc_info:
            odps.execute_sql_interactive(
                "select * from %s where id is not null" % table_name, use_mcqa_v2=True
            )
        assert odps_with_mcqa2.quota_name in str(exc_info.value)
        assert exc_info.value.request_id is not None

    inst = odps.execute_sql_interactive(
        "select * from %s where id is not null" % table_name, use_mcqa_v2=True
    )
    with inst.open_reader(tunnel=True) as reader:
        result = [res.values for res in reader]
    assert result == test_data

    odps_inst = odps.get_instance(inst.id, quota_name=odps.quota_name)
    assert odps_inst.status == Instance.Status.TERMINATED
    assert odps_inst.get_task_detail() is not None
    with odps_inst.open_reader(tunnel=False) as reader:
        result = [res.values for res in reader]
    assert result == test_data

    # query again will reuse current conn
    inst = odps.execute_sql_interactive(
        "select * from %s" % table_name, use_mcqa_v2=True
    )

    with inst.open_reader(tunnel=False) as reader:
        result = [res.values for res in reader]
    assert result == test_data

    table.drop(if_exists=True)
