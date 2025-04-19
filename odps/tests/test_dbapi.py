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

import mock
import pytest

from ..dbapi import Cursor, connect
from ..errors import ODPSError
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

    cursor.execute("desc %s" % table_name)
    assert cursor.description[0][:2] == ("_c0", "string")
    recs = list(cursor)
    assert odps.project in recs[0][0]

    cursor.execute("select * from %s" % table_name)
    assert [("col1", "string"), ("col2", "bigint")] == [
        tp[:2] for tp in cursor.description
    ]
    assert cursor.fetchall() == [["str1", 1234], ["str2", 5678]]

    cursor.execute("select * from %s" % table_name)
    assert cursor.fetchmany(1) == [["str1", 1234]]

    cursor = conn.cursor()
    cursor.execute("select * from %s where col2=?" % table_name, (5678,))
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
    cursor.execute("select * from %s" % table_name)
    assert list(cursor) == [["str1", 1234], ["str2", 5678]]

    def new_run_sql_interactive(self, *args, **kwargs):
        raise ODPSError(code="ODPS-182", msg="ODPS-182: Mock error")

    cursor = conn.cursor()
    with mock.patch("odps.core.ODPS.run_sql_interactive", new=new_run_sql_interactive):
        cursor.execute("select * from %s" % table_name)
    assert list(cursor) == [["str1", 1234], ["str2", 5678]]

    odps.delete_table(table_name, if_exists=True)
