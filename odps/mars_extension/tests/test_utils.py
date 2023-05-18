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

import pytest

from ..utils import rewrite_partition_predicate, filter_partitions


def test_rewrite_predicate():
    assert rewrite_partition_predicate("pt>20210125 & hh=08", ["pt", "hh"]).replace(
            " ", ""
        ) == 'pt>"20210125"&hh=="08"'
    assert rewrite_partition_predicate("pt>20210125 && hh=08", ["pt", "hh"]).replace(
            " ", ""
        ) == 'pt>"20210125"&hh=="08"'
    assert rewrite_partition_predicate("pt>20210125, w=SUN", ["pt", "w"]).replace(
            " ", ""
        ) == 'pt>"20210125"&w=="SUN"'
    assert rewrite_partition_predicate('pt=MAX_PT("table")', ["pt"]) == 'pt ==@max_pt ("table")'
    with pytest.raises(SyntaxError):
        rewrite_partition_predicate("pt>20210125 &&& w=SUN", ["pt", "w"])


def test_filter_partitions(odps):
    table_names = ["test_pyodps_table_nopart", "test_pyodps_table_multi_part"]
    try:
        odps.create_table(
            "test_pyodps_table_nopart", "col1 string", lifecycle=1
        )
        test_multi_part = odps.create_table(
            "test_pyodps_table_multi_part",
            ("col1 string", "pt1 string, pt2 string"),
            lifecycle=1,
        )
        test_multi_part.create_partition("pt1=20210101,pt2=01")
        test_multi_part.create_partition("pt1=20210101,pt2=02")
        test_multi_part.create_partition("pt1=20210102,pt2=01")
        test_multi_part.create_partition("pt1=20210102,pt2=02")

        parts = list(test_multi_part.partitions)

        filtered = filter_partitions(odps, parts, "pt1>20210101")
        assert len(filtered) == 2

        # not partitioned
        with pytest.raises(ValueError):
            filter_partitions(
                odps, parts, "pt1=max_pt(test_pyodps_table_nopart)"
            )

        # parts without data
        with pytest.raises(ValueError):
            filter_partitions(odps, parts, "pt1=max_pt()")

        odps.execute_sql(
            "INSERT INTO test_pyodps_table_multi_part "
            'PARTITION (pt1="20210101", pt2="01") '
            'VALUES ("ABCDEFG")'
        )

        filtered = filter_partitions(odps, parts, "pt1=max_pt()")
        assert filtered[0].partition_spec.kv["pt1"] == "20210101"
    finally:
        for tb in table_names:
            odps.delete_table(tb, if_exists=True, async_=True)
