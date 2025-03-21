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

import csv
import datetime
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
from ...compat import datetime_utcnow, futures, six
from ...config import options
from ...errors import NoSuchObject
from ...tests.core import (
    get_test_unique_name,
    odps2_typed_case,
    pandas_case,
    py_and_c,
    pyarrow_case,
    tn,
)
from ...tunnel import TableTunnel
from ...utils import to_text
from .. import TableSchema
from ..tableio import MPBlockClient, MPBlockServer


def _reloader():
    from ...conftest import get_config
    from .. import table
    from ..record import Record

    cfg = get_config()
    cfg.tunnel = TableTunnel(cfg.odps, endpoint=cfg.odps._tunnel_endpoint)
    table.Record = Record


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
    from .. import Record

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

    if use_legacy:
        assert csv.field_size_limit() > 131072  # check csv limit is shifted


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
    test_table_name = tn("pyodps_t_tmp_table_to_pandas")
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
@pyarrow_case
def test_partition_to_pandas(odps):
    test_table_name = tn("pyodps_t_tmp_part_table_to_pandas")
    schema = TableSchema.from_lists(["num"], ["bigint"], ["pt"], ["string"])
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(test_table_name, schema, lifecycle=1)
    with table.open_writer(
        partition="pt=00", create_partition=True, arrow=True
    ) as writer:
        writer.write(pd.DataFrame({"num": np.random.randint(0, 1000, 1000)}))

    pd_data = table.to_pandas(partition="pt=00", columns=["num"], start=10, count=20)
    assert len(pd_data) == 20

    pd_data = table.partitions["pt=00"].to_pandas(columns=["num"], start=10)
    assert len(pd_data) == 990

    batches = []
    for batch in table.iter_pandas(
        partition="pt=00", columns=["num"], start=10, count=30, batch_size=10
    ):
        assert len(batch) == 10
        batches.append(batch)
    assert len(batches) == 3

    batches = []
    for batch in table.partitions["pt=00"].iter_pandas(
        columns=["num"], start=10, count=30, batch_size=10
    ):
        assert len(batch) == 10
        batches.append(batch)
    assert len(batches) == 3


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
    test_table_name = tn("pyodps_t_tmp_multi_process_write_" + get_test_unique_name(5))
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


@pyarrow_case
@pandas_case
@odps2_typed_case
def test_write_pandas_with_decimal(odps):
    test_table_name = tn("pyodps_t_tmp_write_pandas_decimal")
    odps.delete_table(test_table_name, if_exists=True)

    table = odps.create_table(
        test_table_name,
        "idx string, dec_number decimal(8, 6)",
        lifecycle=1,
    )

    data = pd.DataFrame(
        [
            ["05ac09c4-f947-45a8-8c14-88f430f8b294", "62.388819"],
            ["cfae9054-940b-42a1-84d4-052daae6194f", "81.251166"],
            ["6029501d-c274-49f2-a69d-4c75a3d9931d", "23.395968"],
            ["c653e520-df81-4a5f-b44b-bb1b4c1b7846", "72.210084"],
            ["59caed0d-53d6-473c-a88c-3726c7693f05", "68.602943"],
        ],
        columns=["idx", "dec_number"],
    )

    try:
        odps.write_table(test_table_name, data)
        fetched = odps.get_table(test_table_name).to_pandas()
        pd.testing.assert_frame_equal(fetched.applymap(str), data)
    finally:
        table.drop()


@py_and_c_deco
def test_write_record_with_dynamic_parts(odps):
    test_table_name = tn("pyodps_t_tmp_write_rec_dyn_parts")
    odps.delete_table(test_table_name, if_exists=True)

    data = [[0, 134, "a"], [1, 24, "b"], [2, 131, "a"], [3, 141, "b"]]
    try:
        odps.create_table(
            test_table_name, ("a bigint, b bigint", "p1 string, pt string"), lifecycle=1
        )

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


def test_read_write_transactional_table(odps):
    test_table_name = tn("pyodps_t_tmp_read_write_transactional_table")
    odps.delete_table(test_table_name, if_exists=True)

    data = [["abcd", 12345], ["efgh", 94512], ["eragf", 434]]
    try:
        table = odps.create_table(
            test_table_name,
            ("a string not null, b bigint", "pt string"),
            transactional=True,
            primary_key="a",
            lifecycle=1,
        )
        with table.open_writer(partition="pt=test", create_partition=True) as writer:
            writer.write(data)
            writer.delete(data[0])
        with table.open_reader(partition="pt=test") as reader:
            result = sorted([rec.values[:2] for rec in reader])
        assert result == sorted(data[1:])
    finally:
        odps.delete_table(test_table_name, if_exists=True)


@pandas_case
def test_write_table_with_schema_evolution(odps):
    test_table_name = tn("pyodps_t_tmp_write_table_with_evol")
    odps.delete_table(test_table_name, if_exists=True)
    dest_table = odps.create_table(
        test_table_name,
        ("a string, b bigint", "pt string, pt2 string"),
        lifecycle=1,
    )
    df = pd.DataFrame(
        [["abcd", 12345, 3.456, "part1", "part2"]], columns=["a", "b", "c", "pt", "pt2"]
    )
    odps.write_table(
        test_table_name,
        df,
        partition_cols=["pt", "pt2"],
        append_missing_cols=True,
        create_partition=True,
    )
    assert "c" in odps.get_table(test_table_name).table_schema
    with dest_table.open_reader("pt=part1,pt2=part2", reopen=True) as reader:
        assert reader.count == 1


@pandas_case
def test_write_table_with_error_check(odps):
    test_table_name = tn("pyodps_t_tmp_write_table_with_error_check_pt")
    test_table_name2 = tn("pyodps_t_tmp_write_table_with_error_check_no_pt")
    odps.delete_table(test_table_name, if_exists=True)
    odps.delete_table(test_table_name2, if_exists=True)
    odps.create_table(
        test_table_name,
        ("a string, b bigint", "pt string, pt2 string"),
        lifecycle=1,
    )
    odps.create_table(test_table_name2, "a string, b bigint", lifecycle=1)

    df = pd.DataFrame(
        [["abcd", 12345, "part1", "part2"]], columns=["a", "b", "pt", "pt2"]
    )
    # no parts specified
    with pytest.raises(ValueError):
        odps.write_table(test_table_name, df[["a", "b"]])
    # write into a non-partition table
    with pytest.raises(ValueError):
        odps.write_table(
            test_table_name2,
            df[["a", "b", "pt"]],
            partition_cols="pt",
            partition="pt2=test",
        )
    # pt3 not in table
    with pytest.raises(ValueError):
        odps.write_table(
            test_table_name, df, partition_cols=["pt", "pt2"], partition="pt3=test"
        )
    # pt2 missing
    with pytest.raises(ValueError):
        odps.write_table(test_table_name, df[["a", "b", "pt"]], partition_cols="pt")


@pandas_case
@odps2_typed_case
def test_write_table_with_generate_parts(odps_daily):
    odps = odps_daily

    test_table_name = tn("pyodps_t_tmp_write_table_with_gen_pts")
    odps.delete_table(test_table_name, if_exists=True)
    tb = odps.create_table(
        test_table_name,
        ("dt datetime, a string", "trunc_time(dt, 'day') as pt"),
        lifecycle=1,
    )

    df = pd.DataFrame(
        [
            [pd.Timestamp("2025-02-25 11:23:41"), "r1"],
            [pd.Timestamp("2025-02-26 12:23:41"), "r2"],
            [pd.Timestamp("2025-02-26 13:23:41"), "r3"],
            [pd.Timestamp("2025-02-26 14:23:41"), "r4"],
            [pd.Timestamp("2025-02-27 15:23:41"), "r5"],
        ],
        columns=["dt", "a"],
    )
    odps.write_table(tb, df, create_partition=True)
    assert tb.exist_partition("pt=2025-02-25")
    assert tb.exist_partition("pt=2025-02-26")
    assert tb.exist_partition("pt=2025-02-27")
    pd.testing.assert_frame_equal(
        tb.get_partition("pt=2025-02-26").to_pandas(),
        df[df.dt.dt.date == datetime.date(2025, 2, 26)].reset_index(drop=True),
    )

    dt_col = pa.array(
        [
            pd.Timestamp("2025-02-23 11:23:41"),
            pd.Timestamp("2025-02-24 12:23:41"),
            pd.Timestamp("2025-02-24 13:23:41"),
            pd.Timestamp("2025-02-24 14:23:41"),
            pd.Timestamp("2025-02-25 15:23:41"),
        ]
    )
    s_col = pa.array(["r%s" % (1 + idx) for idx in range(5)])
    pa_batch = pa.RecordBatch.from_arrays([dt_col, s_col], ["dt", "a"])
    odps.write_table(tb, pa_batch, create_partition=True)
    assert tb.exist_partition("pt=2025-02-23")
    assert tb.exist_partition("pt=2025-02-24")

    recs = [
        [pd.Timestamp("2025-02-21 11:23:41"), "s1"],
        [pd.Timestamp("2025-02-22 11:23:41"), "s2"],
        [pd.Timestamp("2025-02-22 11:23:41"), "s3"],
        [pd.Timestamp("2025-02-22 11:23:41"), "s4"],
        [pd.Timestamp("2025-02-23 11:23:41"), "s5"],
    ]
    odps.write_table(tb, recs, create_partition=True)
    assert tb.exist_partition("pt=2025-02-21")
    assert tb.exist_partition("pt=2025-02-22")


@pandas_case
@odps2_typed_case
def test_write_table_with_generate_cols_and_parts(odps_daily):
    odps = odps_daily

    test_table_name = tn("pyodps_t_tmp_write_table_with_gen_col_pts")
    odps.delete_table(test_table_name, if_exists=True)
    tb = odps.create_table(
        test_table_name,
        (
            "_partitiontime timestamp, a string",
            "trunc_time(_partitiontime, 'day') as pt",
        ),
        table_properties={"ingestion_time_partition": "true"},
        lifecycle=1,
    )

    cur_dt = datetime_utcnow()
    pt_spec = "pt=%s" % cur_dt.strftime("%Y-%m-%d")
    # fixme remove this when table_properties is ready on server response
    tb.table_properties = {"ingestion_time_partition": "true"}

    df = pd.DataFrame([["r1"], ["r2"], ["r3"], ["r4"], ["r5"]], columns=["a"])
    odps.write_table(tb, df, create_partition=True)
    assert tb.exist_partition(pt_spec)

    tb.delete_partition(pt_spec)

    s_col = pa.array(["r%s" % (1 + idx) for idx in range(5)])
    pa_batch = pa.RecordBatch.from_arrays([s_col], ["a"])
    odps.write_table(tb, pa_batch, create_partition=True)
    assert tb.exist_partition(pt_spec)

    tb.delete_partition(pt_spec)

    recs = [["s1"], ["s2"], ["s3"], ["s4"], ["s5"]]
    odps.write_table(tb, recs, create_partition=True)
    assert tb.exist_partition(pt_spec)


def test_write_sql_to_simple_table(odps):
    test_src_table_name = tn("pyodps_t_tmp_write_sql_to_simple_tbl_src")
    test_dest_table_name = tn("pyodps_t_tmp_write_sql_to_simple_tbl_dest")
    odps.delete_table(test_src_table_name, if_exists=True)
    odps.delete_table(test_dest_table_name, if_exists=True)

    data = [["abcd", 12345], ["efgh", 94512], ["eragf", 434]]
    src_table = odps.create_table(
        test_src_table_name, "a string, b bigint", lifecycle=1
    )
    with src_table.open_writer() as writer:
        writer.write(data)

    # test table absense and create_table=False
    with pytest.raises(ValueError):
        odps.write_sql_result_to_table(
            test_dest_table_name, "select * from %s" % test_src_table_name
        )

    # test table absense and create_table=True
    odps.write_sql_result_to_table(
        test_dest_table_name,
        "select * from %s" % test_src_table_name,
        create_table=True,
    )
    dest_table = odps.get_table(test_dest_table_name)
    assert dest_table.lifecycle == -1
    with dest_table.open_reader() as reader:
        assert reader.count == 3

    # test write with insert
    odps.write_sql_result_to_table(
        test_dest_table_name, "select * from %s" % test_src_table_name
    )
    with dest_table.open_reader(reopen=True) as reader:
        assert reader.count == 6

    # test write with overwrite
    odps.write_sql_result_to_table(
        test_dest_table_name, "select * from %s" % test_src_table_name, overwrite=True
    )
    with dest_table.open_reader(reopen=True) as reader:
        assert reader.count == 3

    # test table creation with lifecycle
    odps.delete_table(test_dest_table_name, if_exists=True)
    odps.write_sql_result_to_table(
        test_dest_table_name,
        "select * from %s" % test_src_table_name,
        create_table=True,
        lifecycle=1,
    )
    dest_table.reload()
    assert dest_table.lifecycle == 1

    # test table creation with other properties
    odps.delete_table(test_dest_table_name, if_exists=True)
    odps.write_sql_result_to_table(
        test_dest_table_name,
        "select * from %s" % test_src_table_name,
        create_table=True,
        lifecycle=1,
        transactional=True,
    )
    dest_table.reload()
    assert dest_table.lifecycle == 1
    assert dest_table.is_transactional


def test_write_sql_to_partitioned_table(odps):
    test_src_table_name = tn("pyodps_t_tmp_write_sql_to_part_tbl_src")
    test_dest_table_name = tn("pyodps_t_tmp_write_sql_to_part_tbl_dest")
    odps.delete_table(test_src_table_name, if_exists=True)
    odps.delete_table(test_dest_table_name, if_exists=True)

    data = [
        ["abcd", 12345, "a", "a"],
        ["efgh", 94512, "a", "b"],
        ["eragf", 434, "a", "b"],
    ]
    src_table = odps.create_table(
        test_src_table_name, "a string, b bigint, pt1 string, pt2 string", lifecycle=1
    )
    with src_table.open_writer() as writer:
        writer.write(data)

    # test partition absense and create_partition=False
    with pytest.raises(ValueError):
        odps.write_sql_result_to_table(
            test_dest_table_name,
            "select * from %s" % test_src_table_name,
            partition="pt1=a,pt2=a",
            create_table=True,
        )

    # test partition absense and create_partition=True
    odps.write_sql_result_to_table(
        test_dest_table_name,
        "@var1 := select a, b from %s; select * from @var1" % test_src_table_name,
        partition="pt1=a,pt2=a",
        create_table=True,
        create_partition=True,
    )
    dest_table = odps.get_table(test_dest_table_name)
    with dest_table.open_reader("pt1=a,pt2=a") as reader:
        assert reader.count == 3

    # test writing dynamic partition
    odps.write_sql_result_to_table(
        test_dest_table_name,
        "select a, b, pt2 from %s" % test_src_table_name,
        partition="pt1=a",
        partition_cols="pt2",
    )
    with dest_table.open_reader("pt1=a,pt2=a", reopen=True) as reader:
        assert reader.count == 4

    # test writing with dynamic partition and overwrite
    odps.write_sql_result_to_table(
        test_dest_table_name,
        "select a, b, pt2 from %s" % test_src_table_name,
        partition="pt1=a",
        partition_cols="pt2",
        overwrite=True,
    )
    with dest_table.open_reader("pt1=a,pt2=a", reopen=True) as reader:
        assert reader.count == 1


def test_write_sql_to_partitioned_table_with_schema_evolution(odps):
    test_src_table_name = tn("pyodps_t_tmp_write_sql_to_part_evl_tbl_src")
    test_dest_table_name = tn("pyodps_t_tmp_write_sql_to_part_evl_tbl_dest")
    odps.delete_table(test_src_table_name, if_exists=True)
    odps.delete_table(test_dest_table_name, if_exists=True)

    data = [
        ["abcd", 12345, 2.141, "a", "a"],
        ["efgh", 94512, 5.671, "a", "b"],
        ["eragf", 434, 9.358, "a", "b"],
    ]
    src_table = odps.create_table(
        test_src_table_name,
        "a string, b bigint, c double, pt1 string, pt2 string",
        lifecycle=1,
    )
    with src_table.open_writer() as writer:
        writer.write(data)

    dest_table = odps.create_table(
        test_dest_table_name,
        ("a string, b bigint", "pt1 string, pt2 string"),
        lifecycle=1,
    )
    dest_table.create_partition("pt1=a,pt2=a")

    odps.write_sql_result_to_table(
        test_dest_table_name,
        "select a, b, c, pt2 from %s" % test_src_table_name,
        partition="pt1=a",
        partition_cols="pt2",
        append_missing_cols=True,
        overwrite=True,
    )
    assert "c" in odps.get_table(test_dest_table_name).table_schema
    with dest_table.open_reader("pt1=a,pt2=b", reopen=True) as reader:
        assert reader.count == 2


@pandas_case
@odps2_typed_case
def test_write_sql_to_generate_parted_table(odps_daily):
    odps = odps_daily

    test_src_table_name = tn("pyodps_t_tmp_write_sql_to_gen_pt_tbl_src")
    test_dest_table_name = tn("pyodps_t_tmp_write_sql_to_auto_pt_tbl_dest")
    odps.delete_table(test_src_table_name, if_exists=True)
    odps.delete_table(test_dest_table_name, if_exists=True)

    data = [
        [pd.Timestamp("2025-02-21 11:23:41"), "abcd", 12345],
        [pd.Timestamp("2025-02-21 12:23:41"), "efgh", 94512],
        [pd.Timestamp("2025-02-21 13:23:41"), "eragf", 434],
    ]
    src_table = odps.create_table(
        test_src_table_name, "ts timestamp, a string, b bigint", lifecycle=1
    )
    with src_table.open_writer() as writer:
        writer.write(data)

    dest_table = odps.create_table(
        test_dest_table_name,
        (
            "ts timestamp_ntz, a string, b bigint",
            "trunc_time(ts, 'day') as pt",
        ),
        lifecycle=1,
    )
    odps.write_sql_result_to_table(
        test_dest_table_name, "select * from %s" % test_src_table_name
    )
    pt_spec = "pt='2025-02-21'"
    assert dest_table.exist_partition(pt_spec)
    with dest_table.open_reader(pt_spec) as reader:
        assert reader.count == 3

    odps.write_sql_result_to_table(
        test_dest_table_name,
        "select * from %s" % test_src_table_name,
        overwrite=True,
    )
    pt_spec = "pt='2025-02-21'"
    assert dest_table.exist_partition(pt_spec)
    with dest_table.open_reader(pt_spec) as reader:
        assert reader.count == 3


@odps2_typed_case
def test_write_sql_to_auto_parted_table(odps_daily):
    odps = odps_daily

    test_src_table_name = tn("pyodps_t_tmp_write_sql_to_auto_pt_tbl_src")
    test_dest_table_name = tn("pyodps_t_tmp_write_sql_to_auto_pt_tbl_dest")
    odps.delete_table(test_src_table_name, if_exists=True)
    odps.delete_table(test_dest_table_name, if_exists=True)

    data = [["abcd", 12345], ["efgh", 94512], ["eragf", 434]]
    src_table = odps.create_table(
        test_src_table_name, "a string, b bigint", lifecycle=1
    )
    with src_table.open_writer() as writer:
        writer.write(data)

    dest_table = odps.create_table(
        test_dest_table_name,
        (
            "_partitiontime timestamp_ntz, a string, b bigint",
            "trunc_time(_partitiontime, 'day') as pt",
        ),
        table_properties={"ingestion_time_partition": "true"},
        lifecycle=1,
    )
    # fixme remove this when table_properties is ready on server response
    dest_table.table_properties = {"ingestion_time_partition": "true"}
    odps.write_sql_result_to_table(
        test_dest_table_name, "select * from %s" % test_src_table_name
    )
    pt_spec = "pt='%s'" % datetime_utcnow().strftime("%Y-%m-%d")
    assert dest_table.exist_partition(pt_spec)
    with dest_table.open_reader(pt_spec) as reader:
        assert reader.count == 3

    odps.write_sql_result_to_table(
        test_dest_table_name,
        "select * from %s" % test_src_table_name,
        overwrite=True,
    )
    pt_spec = "pt='%s'" % datetime_utcnow().strftime("%Y-%m-%d")
    assert dest_table.exist_partition(pt_spec)
    with dest_table.open_reader(pt_spec) as reader:
        assert reader.count == 3
