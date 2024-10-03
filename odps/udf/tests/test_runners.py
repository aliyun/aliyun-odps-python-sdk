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

from ...compat import StringIO
from ...config import options
from ...tests.core import tn
from ..tools.runners import DirectCollector, get_csv_runner, get_table_runner
from .udf_examples import CatStrings, Plus, ZipArray

csv_data = """
0,1
3,4
2,1
""".strip()


def test_csv_runner():
    text_io = StringIO(csv_data)
    runner = get_csv_runner(Plus, stdin=text_io, collector_cls=DirectCollector)
    runner.run()
    assert runner.collector.results == [1, 7, 3]


def test_table_runner(odps):
    table_name = tn("pyodps_t_tmp_test_udf_table_runner")
    try:
        options.sql.use_odps2_extension = True
        odps.delete_table(table_name, if_exists=True)
        table = odps.create_table(
            table_name,
            "col1 array<string>, col2 array<bigint>, col3 string",
            lifecycle=1,
        )

        with table.open_writer() as writer:
            writer.write([[["abcd", "efgh"], [1342, 5412], "uvfw"]])
            writer.write([[["alkf"], [1261], "asfd"]])
            writer.write([[["uvews", "asdfsaf"], [3245, 2345], "poes"]])
            writer.write([[["kslazd", "fdsal"], [342, 244], "poes"]])

        runner = get_table_runner(
            ZipArray, odps, table_name + ".c(col1, col2)", collector_cls=DirectCollector
        )
        runner.run()
        assert [
            {"abcd": 1342, "efgh": 5412},
            {"alkf": 1261},
            {"uvews": 3245, "asdfsaf": 2345},
            {"kslazd": 342, "fdsal": 244},
        ] == runner.collector.results

        runner = get_table_runner(
            ZipArray,
            odps,
            table_name + ".c(col1, col2)",
            record_limit=2,
            collector_cls=DirectCollector,
        )
        runner.run()
        assert [
            {"abcd": 1342, "efgh": 5412},
            {"alkf": 1261},
        ] == runner.collector.results
    finally:
        odps.delete_table(table_name, if_exists=True)
        options.sql.use_odps2_extension = False


def test_table_runner_with_parts(odps):
    table_name = tn("pyodps_t_tmp_test_udf_table_runner_with_part")
    try:
        odps.delete_table(table_name, if_exists=True)
        table = odps.create_table(
            table_name,
            ("col1 bigint, col2 bigint", "pt string"),
            lifecycle=1,
        )

        with table.open_writer("pt=abcd", create_partition=True) as writer:
            writer.write([[123, 541], [11, 92]])

        table_spec = table_name + ".p(pt=abcd).c(col1,col2)"
        runner = get_table_runner(Plus, odps, table_spec, collector_cls=DirectCollector)
        runner.run()
        assert [664, 103] == runner.collector.results

        table_spec = table_name + ".p(pt=abcd).c(col1,pt)"
        runner = get_table_runner(
            CatStrings, odps, table_spec, collector_cls=DirectCollector
        )
        runner.run()
        assert ["123abcd", "11abcd"] == runner.collector.results
    finally:
        odps.delete_table(table_name, if_exists=True)
