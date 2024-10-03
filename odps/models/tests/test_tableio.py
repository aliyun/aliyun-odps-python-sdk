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

import logging
import multiprocessing
import sys
from collections import OrderedDict

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
except (AttributeError, ImportError):
    np = pd = pa = None

import mock
import pytest

from ... import types as odps_types
from ...compat import futures, six
from ...config import options
from ...errors import NoSuchObject
from ...tests.core import get_test_unique_name, pandas_case, py_and_c, pyarrow_case, tn
from ...tunnel import TableTunnel
from ...utils import to_text
from .. import Record, TableSchema
from ..tableio import MPBlockClient, MPBlockServer


def _reloader():
    from ...conftest import get_config

    cfg = get_config()
    cfg.tunnel = TableTunnel(cfg.odps, endpoint=cfg.odps._tunnel_endpoint)


py_and_c_deco = py_and_c(
    [
        "odps.models.record",
        "odps.models",
        "odps.tunnel.io.reader",
        "odps.tunnel.io.writer",
        "odps.tunnel.tabletunnel",
        "odps.tunnel.instancetunnel",
    ],
    _reloader,
)


@pytest.mark.parametrize("use_legacy", [False, True])
def test_record_read_write_table(odps, use_legacy):
    test_table_name = tn("pyodps_t_tmp_read_write_table_" + get_test_unique_name(5))
    schema = TableSchema.from_lists(
        ["id", "name", "right"], ["bigint", "string", "boolean"]
    )

    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(test_table_name, schema)
    data = [
        [111, "aaa", True],
        [222, "bbb", False],
        [333, "ccc", True],
        [5940813139082772990, "中文", False],
    ]
    length = len(data)
    records = [Record(schema=schema, values=values) for values in data]

    texted_data = [[it[0], to_text(it[1]), it[2]] for it in data]

    if use_legacy:
        records = (rec for rec in records)

    odps.write_table(table, records)
    assert texted_data == [record.values for record in odps.read_table(table, length)]
    assert texted_data[::2] == [
        record.values for record in odps.read_table(table, length, step=2)
    ]

    assert texted_data == [
        record.values for record in table.head(length, use_legacy=use_legacy)
    ]

    table.truncate()
    assert [] == list(odps.read_table(table))

    odps.delete_table(test_table_name)
    assert odps.exist_table(test_table_name) is False


def test_array_iter_read_write_table(odps):
    test_table_name = tn("pyodps_t_tmp_array_iter_read_write_table")
    schema = TableSchema.from_lists(
        ["id", "name", "right"], ["bigint", "string", "boolean"]
    )

    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(test_table_name, schema)
    data = [
        [111, "aaa", True],
        [222, "bbb", False],
        [333, "ccc", True],
        [444, "中文", False],
    ]
    length = len(data)

    texted_data = [[it[0], to_text(it[1]), it[2]] for it in data]

    odps.write_table(table, 0, data)
    assert texted_data == [record.values for record in odps.read_table(table, length)]
    assert texted_data[::2] == [
        record.values for record in odps.read_table(table, length, step=2)
    ]

    assert texted_data == [record.values for record in table.head(length)]

    table.truncate()
    with table.open_writer() as writer:
        writer.write(0, (rec for rec in []))
        writer.write(0, (rec for rec in data))
    assert texted_data == [record.values for record in odps.read_table(table, length)]

    odps.delete_table(test_table_name)
    assert odps.exist_table(test_table_name) is False


def test_read_write_partition_table(odps):
    test_table_name = tn("pyodps_t_tmp_read_write_partition_table")
    schema = TableSchema.from_lists(
        ["id", "name"], ["bigint", "string"], ["pt"], ["string"]
    )

    odps.delete_table(test_table_name, if_exists=True)
    assert odps.exist_table(test_table_name) is False

    table = odps.create_table(test_table_name, schema)
    table._upload_ids = dict()

    pt1 = "pt=20151122"
    pt2 = "pt=20151123"
    table.create_partition(pt1)
    table.create_partition(pt2)

    with table.open_reader(pt1) as reader:
        assert len(list(reader)) == 0

    with table.open_writer(pt1, commit=False) as writer:
        record = table.new_record([1, "name1"])
        writer.write(record)

        record = table.new_record()
        record[0] = 3
        record[1] = "name3"
        writer.write(record)

    assert len(table._upload_ids) == 1
    upload_id = list(table._upload_ids.values())[0]
    with table.open_writer(pt1):
        assert len(table._upload_ids) == 1
        assert upload_id == list(table._upload_ids.values())[0]

    with table.open_writer(pt2) as writer:
        writer.write([2, "name2"])

    with table.open_reader(pt1, reopen=True) as reader:
        records = list(reader)
        assert len(records) == 2
        assert sum(r[0] for r in records) == 4

    with table.open_reader(pt2, append_partitions=False, reopen=True) as reader:
        records = list(reader)
        assert len(records[0]) == 2
        assert len(records) == 1
        assert sum(r[0] for r in records) == 2

    with table.open_reader(pt2, reopen=True) as reader:
        records = list(reader)
        assert len(records[0]) == 3
        assert len(records) == 1
        assert sum(r[0] for r in records) == 2

    # need to guarantee generators of
    odps.write_table(table, (rec for rec in records), partition=pt2)
    ret_records = list(odps.read_table(table, partition=pt2))
    assert len(ret_records) == 2

    if pa is not None and pd is not None:
        with table.open_reader(pt2, arrow=True, reopen=True) as reader:
            result = reader.to_pandas()
            assert len(result.dtypes) == 2

        with table.open_reader(
            pt2, arrow=True, append_partitions=True, reopen=True
        ) as reader:
            result = reader.to_pandas()
            assert len(result.dtypes) == 3

    table.drop()


def test_simple_record_read_write_table(odps):
    test_table_name = tn("pyodps_t_tmp_simple_record_read_write_table")
    schema = TableSchema.from_lists(["num"], ["string"], ["pt"], ["string"])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    partition = "pt=20151122"
    table.create_partition(partition)

    with table.open_writer(partition) as writer:
        record = table.new_record()
        record[0] = "1"
        writer.write(record)

    with table.open_reader(partition) as reader:
        assert reader.count == 1
        record = next(reader)
        assert record[0] == "1"
        assert record.num == "1"

    if pd is not None:
        with table.open_reader(partition, reopen=True) as reader:
            pd_data = reader.to_pandas()
            assert len(pd_data) == 1

    partition = "pt=20151123"
    with pytest.raises(NoSuchObject):
        table.open_writer(partition, create_partition=False)

    with table.open_writer(partition, create_partition=True) as writer:
        record = table.new_record()
        record[0] = "1"
        writer.write(record)

    with table.open_reader(partition) as reader:
        assert reader.count == 1
        record = next(reader)
        assert record[0] == "1"
        assert record.num == "1"

    table.drop()


def test_simple_array_read_write_table(odps):
    test_table_name = tn("pyodps_t_tmp_simple_array_read_write_table")
    schema = TableSchema.from_lists(["num"], ["string"], ["pt"], ["string"])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    partition = "pt=20151122"
    table.create_partition(partition)

    with table.open_writer(partition) as writer:
        writer.write(["1"])

    with table.open_reader(partition) as reader:
        assert reader.count == 1
        record = next(reader)
        assert record[0] == "1"
        assert record.num == "1"

    with table.open_reader(partition, async_mode=True, reopen=True) as reader:
        assert reader.count == 1
        record = next(reader)
        assert record[0] == "1"
        assert record.num == "1"

    table.drop()


def test_table_write_error(odps):
    test_table_name = tn("pyodps_t_tmp_test_table_write_error")
    schema = TableSchema.from_lists(["name"], ["string"])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    try:
        with table.open_writer() as writer:
            writer.write([["Content"]])
            raise ValueError("Mock error")
    except ValueError as ex:
        assert str(ex) == "Mock error"


@pandas_case
@pyarrow_case
def test_table_to_pandas(odps):
    test_table_name = tn("pyodps_t_tmp_to_pandas")
    schema = TableSchema.from_lists(["num"], ["bigint"])
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema, lifecycle=1)
    with table.open_writer(arrow=True) as writer:
        writer.write(pd.DataFrame({"num": np.random.randint(0, 1000, 1000)}))

    pd_data = table.to_pandas(columns=["num"], start=10, count=20)
    assert len(pd_data) == 20

    pd_data = table.to_pandas(columns=["num"], start=10)
    assert len(pd_data) == 990

    batches = []
    for batch in table.iter_pandas(columns=["num"], start=10, count=30, batch_size=10):
        assert len(batch) == 10
        batches.append(batch)
    assert len(batches) == 3

    table.drop()


@pandas_case
def test_multi_process_to_pandas(odps):
    from ...tunnel.tabletunnel import TableDownloadSession

    if pa is None:
        pytest.skip("Need pyarrow to run the test.")

    test_table_name = tn("pyodps_t_tmp_mproc_read_table")
    schema = TableSchema.from_lists(["num"], ["bigint"])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema, lifecycle=1)
    with table.open_writer(arrow=True) as writer:
        writer.write(pd.DataFrame({"num": np.random.randint(0, 1000, 1000)}))

    with table.open_reader() as reader:
        pd_data = reader.to_pandas(n_process=2)
        assert len(pd_data) == 1000

    orginal_meth = TableDownloadSession.open_record_reader

    def patched(self, start, *args, **kwargs):
        if start != 0:
            raise ValueError("Intentional error")
        return orginal_meth(self, start, *args, **kwargs)

    with pytest.raises(ValueError):
        with mock.patch(
            "odps.tunnel.tabletunnel.TableDownloadSession.open_record_reader",
            new=patched,
        ):
            with table.open_reader() as reader:
                reader.to_pandas(n_process=2)

    with table.open_reader(arrow=True) as reader:
        pd_data = reader.to_pandas(n_process=2)
        assert len(pd_data) == 1000

    table.drop()


@pandas_case
def test_column_select_to_pandas(odps):
    if pa is None:
        pytest.skip("Need pyarrow to run the test.")

    test_table_name = tn("pyodps_t_tmp_col_select_table")
    schema = TableSchema.from_lists(["num1", "num2"], ["bigint", "bigint"])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    with table.open_writer(arrow=True) as writer:
        writer.write(
            pd.DataFrame(
                {
                    "num1": np.random.randint(0, 1000, 1000),
                    "num2": np.random.randint(0, 1000, 1000),
                }
            )
        )

    with table.open_reader(columns=["num1"]) as reader:
        pd_data = reader.to_pandas()
        assert len(pd_data) == 1000
        assert len(pd_data.columns) == 1

    with table.open_reader(columns=["num1"], arrow=True) as reader:
        pd_data = reader.to_pandas()
        assert len(pd_data) == 1000
        assert len(pd_data.columns) == 1


@pandas_case
def test_complex_type_to_pandas(odps):
    test_table_name = tn("pyodps_t_tmp_complex_type_to_pd")
    schema = TableSchema.from_lists(
        ["cp1", "cp2", "cp3"],
        ["array<string>", "map<string,bigint>", "struct<a: string, b: bigint>"],
    )

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    row = [
        ["abc", "def"],
        OrderedDict([("uvw", 1), ("xyz", 2)]),
        OrderedDict([("a", "data"), ("b", 1)]),
    ]
    with table.open_writer() as writer:
        writer.write([row])

    with table.open_reader() as reader:
        pd_data = reader.to_pandas()
        assert pd_data.iloc[0].to_list() == row

    if pa is not None:
        with table.open_reader(arrow=True) as reader:
            pd_data = reader.to_pandas()
            assert [
                pd_data.iloc[0, 0].tolist(),
                OrderedDict(pd_data.iloc[0, 1]),
                OrderedDict(pd_data.iloc[0, 2]),
            ] == row


@pandas_case
def test_record_to_pandas_batches(odps):
    test_table_name = tn("pyodps_t_read_in_batches")
    odps.delete_table(test_table_name, if_exists=True)
    rec_count = 37

    data = [[idx, "str_" + str(idx)] for idx in range(rec_count)]

    table = odps.create_table(test_table_name, "col1 bigint, col2 string")
    with table.open_writer() as writer:
        writer.write(data)

    try:
        options.tunnel.read_row_batch_size = 5
        options.tunnel.batch_merge_threshold = 5
        with table.open_reader() as reader:
            pd_result = reader.to_pandas()
        assert len(pd_result) == rec_count

        with table.open_reader() as reader:
            pd_result = reader[:10].to_pandas()
        assert len(pd_result) == 10
    finally:
        options.tunnel.read_row_batch_size = 1024
        options.tunnel.batch_merge_threshold = 128


@pytest.mark.skipif(pa is None, reason="Need pyarrow to run this test")
def test_simple_arrow_read_write_table(odps):
    test_table_name = tn("pyodps_t_tmp_simple_arrow_read_write_table")
    schema = TableSchema.from_lists(["num"], ["string"], ["pt"], ["string"])

    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema)
    partition = "pt=20151122"
    table.create_partition(partition)

    with table.open_writer(partition, arrow=True) as writer:
        writer.write(pd.DataFrame({"num": list("ABCDE")}))

    with table.open_reader(partition, arrow=True) as reader:
        assert reader.count == 5
        batches = list(reader)
        assert len(batches) == 1
        assert batches[0].num_rows == 5

    table.truncate(partition_spec=partition)

    with table.open_writer(partition, arrow=True) as writer:
        writer.write(0, pd.DataFrame({"num": list("ABCDE")}))

    with table.open_reader(partition, reopen=True, arrow=True) as reader:
        pd_data = reader.to_pandas()
        assert len(pd_data) == 5

    # now test corner case of empty table
    table.truncate(partition)

    with table.open_reader(partition, arrow=True) as reader:
        batches = list(reader)
        assert len(batches) == 0

    with table.open_reader(partition, reopen=True, arrow=True) as reader:
        pd_data = reader.to_pandas()
        assert len(pd_data.columns) == 1
        assert len(pd_data) == 0

    table.drop()


def test_mp_block_server():
    class MockWriter(object):
        def __init__(self):
            self.idx = 0
            self._used_block_id_queue = six.moves.queue.Queue()

        def _gen_next_block_id(self):
            idx = self.idx
            self.idx += 1
            return idx

    mock_writer = MockWriter()
    block_server = MPBlockServer(mock_writer)
    block_server.start()
    block_client = MPBlockClient(block_server.address, block_server.authkey)

    try:
        # test block_id request
        assert 0 == block_client.get_next_block_id()
        assert 1 == block_client.get_next_block_id()

        # test blocks request
        block_count = int(MPBlockClient._MAX_BLOCK_COUNT * 1.5)
        block_client.put_written_blocks(list(range(block_count)))

        written_blocks = []
        while not mock_writer._used_block_id_queue.empty():
            written_blocks.extend(mock_writer._used_block_id_queue.get())
        assert list(range(block_count)) == written_blocks

        errored_block_client = MPBlockClient(block_server.address, b"e" * 32)
        with pytest.raises(AssertionError):
            errored_block_client.get_next_block_id()
    finally:
        block_client.close()
        block_server.stop()


def _spawned_write(idx, writer, close=True):
    try:
        writer.write([idx, 0, "row1"])
        writer.write([idx, 1, "row2"])
        if close:
            writer.close()
    except:
        logging.exception("Unexpected inprocess error occurred!")
        raise


def test_multi_thread_write(odps):
    test_table_name = tn("pyodps_t_tmp_multi_thread_write")
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, "col1 bigint, col2 bigint, col3 string")

    try:
        pool = futures.ThreadPoolExecutor(2)
        with table.open_writer() as writer:
            futs = []
            for idx in range(2):
                fut = pool.submit(_spawned_write, idx, writer, close=False)
                futs.append(fut)

            for fut in futs:
                fut.result()
        with table.open_reader() as reader:
            results = sorted([rec.values for rec in reader])
        assert results == [
            [0, 0, "row1"],
            [0, 1, "row2"],
            [1, 0, "row1"],
            [1, 1, "row2"],
        ]
    finally:
        table.drop()


@pytest.mark.parametrize(
    "ctx_name",
    ["fork", "forkserver", "spawn"]
    if sys.version_info[0] > 2 and sys.platform != "win32"
    else ["spawn"],
)
def test_multi_process_write(odps, ctx_name):
    test_table_name = tn("pyodps_t_tmp_multi_process_write")
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, "col1 bigint, col2 bigint, col3 string")

    if sys.version_info[0] > 2:
        orig_ctx = multiprocessing.get_start_method()
    try:
        if sys.version_info[0] > 2:
            multiprocessing.set_start_method(ctx_name, force=True)
        with table.open_writer() as writer:
            procs = []
            for idx in range(2):
                proc = multiprocessing.Process(
                    target=_spawned_write, args=(idx, writer)
                )
                proc.start()
                procs.append(proc)

            for proc in procs:
                proc.join()
                assert proc.exitcode == 0
        with table.open_reader() as reader:
            results = sorted([rec.values for rec in reader])
        assert results == [
            [0, 0, "row1"],
            [0, 1, "row2"],
            [1, 0, "row1"],
            [1, 1, "row2"],
        ]
    finally:
        if sys.version_info[0] > 2:
            multiprocessing.set_start_method(orig_ctx, force=True)
        table.drop()


def test_multi_process_pool_write(odps):
    test_table_name = tn("pyodps_t_tmp_multi_process_pool_write")
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, "col1 bigint, col2 bigint, col3 string")

    try:
        with table.open_writer() as writer:
            futures = []
            pool = multiprocessing.Pool(processes=2)
            for idx in range(2):
                futures.append(pool.apply_async(_spawned_write, (idx, writer)))
            for fut in futures:
                fut.get()
        with table.open_reader() as reader:
            results = sorted([rec.values for rec in reader])
        assert results == [
            [0, 0, "row1"],
            [0, 1, "row2"],
            [1, 0, "row1"],
            [1, 1, "row2"],
        ]
    finally:
        table.drop()


@pandas_case
@pyarrow_case
@pytest.mark.parametrize("use_arrow", [False, True])
def test_write_table_with_pandas_or_arrow(odps, use_arrow):
    suffix = "arrow" if use_arrow else "pd"
    test_table_name = tn("pyodps_t_tmp_write_table_pandas_arrow_" + suffix)
    odps.delete_table(test_table_name, if_exists=True)

    data = pd.DataFrame(
        pd.DataFrame(
            [["falcon", 2, 2], ["dog", 4, 0], ["cat", 4, 0], ["ant", 6, 0]],
            columns=["names", "num_legs", "num_wings"],
        )
    )

    # test write pandas dataframe
    with pytest.raises(NoSuchObject):
        odps.write_table(test_table_name, data, lifecycle=1)

    try:
        if use_arrow:
            data_to_write = pa.Table.from_pandas(data)
        else:
            data_to_write = data

        odps.write_table(test_table_name, data_to_write, create_table=True, lifecycle=1)
        fetched = odps.get_table(test_table_name).to_pandas()
        pd.testing.assert_frame_equal(data, fetched)
    finally:
        odps.delete_table(test_table_name, if_exists=True)


@pandas_case
@pyarrow_case
def test_write_table_with_pandas_or_arrow_parted(odps):
    test_table_name = tn("pyodps_t_tmp_write_table_pandas_arrow_parted")
    odps.delete_table(test_table_name, if_exists=True)

    data = pd.DataFrame(
        pd.DataFrame(
            [["falcon", 2, 2], ["dog", 4, 0], ["cat", 4, 0], ["ant", 6, 0]],
            columns=["names", "num_legs", "num_wings"],
        )
    )
    try:
        odps.write_table(
            test_table_name,
            data,
            partition=odps_types.PartitionSpec("pt=test"),
            create_table=True,
            create_partition=True,
            lifecycle=1,
        )
        fetched = odps.get_table(test_table_name).to_pandas(partition="pt=test")
        pd.testing.assert_frame_equal(data, fetched)

        schema = odps.get_table(test_table_name).table_schema
        assert len(schema.simple_columns) == len(data.columns)

        odps.write_table(
            test_table_name,
            data,
            partition="pt=test2",
            create_partition=True,
            lifecycle=1,
        )
        fetched = odps.get_table(test_table_name).to_pandas(partition="pt=test2")
        pd.testing.assert_frame_equal(data, fetched)
    finally:
        odps.delete_table(test_table_name, if_exists=True)


@pandas_case
@pyarrow_case
@pytest.mark.parametrize("use_arrow", [False, True])
def test_write_pandas_with_dynamic_parts(odps, use_arrow):
    suffix = "arrow" if use_arrow else "pd"
    test_table_name = tn("pyodps_t_tmp_write_pandas_dyn_parts_" + suffix)
    odps.delete_table(test_table_name, if_exists=True)

    data = pd.DataFrame(
        [[0, 134, "a", "a"], [1, 24, "a", "b"], [2, 131, "a", "a"], [3, 141, "a", "b"]],
        columns=["a", "b", "p1", "p2"],
    )

    try:
        if use_arrow:
            data_to_write = pa.Table.from_pandas(data)
        else:
            data_to_write = data

        odps.write_table(
            test_table_name,
            data_to_write,
            create_table=True,
            partitions=["p1", "p2"],
            create_partition=True,
            lifecycle=1,
        )
        fetched = odps.get_table(test_table_name).to_pandas(partition="p1=a,p2=a")
        expected = data[data.p1 == "a"][data.p2 == "a"][["a", "b"]].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(fetched, expected)

        fetched = odps.get_table(test_table_name).to_pandas(partition="p1=a,p2=b")
        expected = data[data.p1 == "a"][data.p2 == "b"][["a", "b"]].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(fetched, expected)
    finally:
        odps.delete_table(test_table_name, if_exists=True)


@py_and_c_deco
def test_write_record_with_dynamic_parts(odps):
    test_table_name = tn("pyodps_t_tmp_write_rec_dyn_parts")
    odps.delete_table(test_table_name, if_exists=True)

    data = [[0, 134, "a"], [1, 24, "b"], [2, 131, "a"], [3, 141, "b"]]
    schema = odps_types.OdpsSchema(
        [
            odps_types.Column("a", odps_types.bigint),
            odps_types.Column("b", odps_types.bigint),
        ],
        [
            odps_types.Column("p1", odps_types.string),
            odps_types.Column("pt", odps_types.string),
        ],
    )

    try:
        odps.create_table(test_table_name, schema, lifecycle=1)

        with pytest.raises(ValueError):
            odps.write_table(
                test_table_name,
                odps,
                partitions="p1",
                partition="pt=test",
                create_partition=True,
            )

        odps.write_table(
            test_table_name,
            data,
            partitions="p1",
            partition="pt=test",
            create_partition=True,
        )
        fetched = [
            r.values[:2]
            for r in odps.read_table(test_table_name, partition="p1=a,pt=test")
        ]
        expected = [d[:2] for d in data if d[2:] == ["a"]]
        assert fetched == expected

        fetched = [
            r.values[:2]
            for r in odps.read_table(test_table_name, partition="p1=b,pt=test")
        ]
        expected = [d[:2] for d in data if d[2:] == ["b"]]
        assert fetched == expected
    finally:
        odps.delete_table(test_table_name, if_exists=True)
