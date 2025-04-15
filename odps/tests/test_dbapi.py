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

import pytest

from ..dbapi import Cursor


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
