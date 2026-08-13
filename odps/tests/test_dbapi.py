# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import mock
import pytest

from ..dbapi import Cursor, connect
from ..errors import InstanceNotTerminate, InstanceTypeNotSupported, ODPSError
from .core import tn


def test_replace_sql_parameters_tuple():
    with pytest.raises(TypeError):
        Cursor._replace_sql_parameters("select * from dummy where col1=?", "test")
    with pytest.raises(ValueError):
        Cursor._replace_sql_parameters("select * from dummy where col1=?", ())
    with pytest.raises(KeyError):
        Cursor._replace_sql_parameters(
            "select * from dummy where col1=:name", {"abc": "def"}
        )

    stmt = "select * from dummy where col1='?\"\\'' and col2=?"
    expected = "select * from dummy where col1='?\"\\'' and col2='repl\\''"
    assert expected == Cursor._replace_sql_parameters(stmt, ("repl'",))

    stmt = (
        "select * from dummy where col1 = '?\"' and col2=':name' "
        "and col3={'abc':1} and col4=:name"
    )
    expected = (
        "select * from dummy where col1 = '?\"' and col2=':name' "
        "and col3={'abc':1} and col4='repl\\''"
    )
    assert expected == Cursor._replace_sql_parameters(stmt, {"name": "repl'"})


def test_dbapi_execute_sql(odps):
    table_name = tn("test_dbapi_execute_sql")
    odps.delete_table(table_name, if_exists=True)
    odps.create_table(table_name, "col1 string, col2 bigint", lifecycle=1)
    odps.write_table(table_name, [["str1", 1234], ["str2", 5678]])

    with pytest.raises(ValueError):
        connect("access_id", odps=odps)

    conn = connect(odps)
    cursor = conn.cursor()

    cursor.execute(f"desc {table_name}")
    assert cursor.description[0][:2] == ("_c0", "string")
    recs = list(cursor)
    assert odps.project in recs[0][0]

    cursor.execute(f"select * from {table_name}")
    assert [("col1", "string"), ("col2", "bigint")] == [
        tp[:2] for tp in cursor.description
    ]
    assert cursor.fetchall() == [["str1", 1234], ["str2", 5678]]

    cursor.execute(f"select * from {table_name}")
    assert cursor.fetchmany(1) == [["str1", 1234]]

    cursor = conn.cursor()
    cursor.execute(f"select * from {table_name} where col2=?", (5678,))
    assert cursor.fetchone() == ["str2", 5678]

    odps.delete_table(table_name, if_exists=True)


def test_dbapi_execute_sql_with_sqa(odps):
    table_name = tn("test_dbapi_execute_sql_with_sqa")
    odps.delete_table(table_name, if_exists=True)
    odps.create_table(table_name, "col1 string, col2 bigint", lifecycle=1)
    odps.write_table(table_name, [["str1", 1234], ["str2", 5678]])

    conn = connect(
        account=odps.account,
        project=odps.project,
        endpoint=odps.endpoint,
        use_sqa="v1",
        fallback_policy="all",
    )
    cursor = conn.cursor()
    cursor.execute(f"select * from {table_name}")
    assert list(cursor) == [["str1", 1234], ["str2", 5678]]

    def new_run_sql_interactive(self, *args, **kwargs):
        raise ODPSError(code="ODPS-182", msg="ODPS-182: Mock error")

    cursor = conn.cursor()
    with mock.patch("odps.core.ODPS.run_sql_interactive", new=new_run_sql_interactive):
        cursor.execute(f"select * from {table_name}")
    assert list(cursor) == [["str1", 1234], ["str2", 5678]]

    odps.delete_table(table_name, if_exists=True)


def _make_sqa_run(quota_name="q1", hints=None):
    odps = mock.MagicMock()
    odps.quota_name = quota_name
    conn = connect(
        odps=odps,
        use_sqa="v2",
        fallback_policy="all",
        quota_name=quota_name,
        hints=hints,
    )
    inst = mock.MagicMock()
    odps.run_sql_interactive.return_value = inst
    return conn.cursor(), odps, inst


def test_sqa_uses_try_result_first_and_forwards_quota():
    cursor, odps, inst = _make_sqa_run(quota_name="my_quota")
    inline_reader = mock.MagicMock()
    inst.open_reader.return_value = inline_reader

    cursor.execute("select * from t")

    inst.open_reader.assert_called_once_with(
        _try_result_first=True, tunnel=True, limit=False
    )
    assert cursor._download_session is inline_reader
    _, kw = odps.run_sql_interactive.call_args
    assert kw["quota_name"] == "my_quota"
    assert kw["use_mcqa_v2"] is True


def test_sqa_non_select_returns_instance_without_session():
    # EXPLAIN and other non-SELECT statements: open_reader raises
    # InstanceTypeNotSupported; the cursor leaves _download_session unset
    # so _fetch_non_select() returns the raw text via reader.raw.
    cursor, odps, inst = _make_sqa_run()
    inst.open_reader.side_effect = InstanceTypeNotSupported(
        "InstanceTunnel cannot be opened at a non-select SQL Task."
    )

    cursor.execute("explain select 1")

    assert cursor._download_session is None
    assert cursor._instance is inst


def test_sqa_async_waits_before_tunnel():
    # queries that do not finish synchronously must terminate before a tunnel
    # download session can be created, otherwise the server rejects it with
    # "InstanceNotTerminate".
    cursor, odps, inst = _make_sqa_run()
    tunnel_reader = mock.MagicMock()

    inst.open_reader.return_value = tunnel_reader

    cursor.execute("select * from t")

    inst.wait_for_success.assert_called()
    _, kw = inst.open_reader.call_args
    assert kw.get("_try_result_first") is True
    assert kw.get("tunnel") is True
    assert cursor._download_session is tunnel_reader


def test_sqa_not_terminated_retries_after_wait():
    # an InstanceNotTerminate error from the tunnel should be retried
    # after waiting, instead of being raised to the caller.
    cursor, odps, inst = _make_sqa_run()
    tunnel_reader = mock.MagicMock()

    state = {"calls": 0}

    def _open_reader(**kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise InstanceNotTerminate("The instance does not terminate.")
        return tunnel_reader

    inst.open_reader.side_effect = _open_reader

    cursor.execute("select * from t")

    assert state["calls"] >= 2
    inst.wait_for_success.assert_called()
    assert cursor._download_session is tunnel_reader


def test_sqa_fallback_on_unsupported():
    cursor, odps, inst = _make_sqa_run()
    inst.open_reader.side_effect = ODPSError(
        code="ODPS-185", msg="ODPS-185: unsupported"
    )
    fallback_inst = mock.MagicMock()
    odps.execute_sql.return_value = fallback_inst

    cursor.execute("select * from t")

    odps.execute_sql.assert_called_once()
    assert cursor._instance is fallback_inst


def test_sqa_unrelated_error_is_raised():
    # errors that are neither fallback-eligible nor InstanceNotTerminate
    # must propagate to the caller instead of being silently retried.
    cursor, odps, inst = _make_sqa_run()

    inst.open_reader.side_effect = ODPSError(
        "some other tunnel error", code="SomeOtherCode"
    )

    with pytest.raises(ODPSError, match="some other tunnel error"):
        cursor.execute("select * from t")


def test_reuse_odps_cache_keeps_quota_name():
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.engine import make_url

    from .. import sqlalchemy_odps

    sqlalchemy_odps._sqlalchemy_global_reusable_odps.clear()
    url = make_url(
        "odps://access_id:secret_key@my_project/"
        "?endpoint=http://example.com&reuse_odps=true&quota_name=my_quota"
    )
    try:
        with mock.patch.object(sqlalchemy_odps, "ODPS", autospec=True) as m_odps:
            dialect = sqlalchemy_odps.ODPSDialect()
            # first call builds and caches the ODPS object
            dialect.create_connect_args(url)
            _, build_kw = m_odps.call_args
            # the cached ODPS object must be built carrying quota_name
            assert build_kw.get("quota_name") == "my_quota"

            # a second call reuses the cache and does not rebuild the object
            m_odps.reset_mock()
            _, kwargs2 = dialect.create_connect_args(url)
            m_odps.assert_not_called()
            assert kwargs2["odps"] is m_odps.return_value
    finally:
        sqlalchemy_odps._sqlalchemy_global_reusable_odps.clear()
