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

import datetime
import math
from collections import OrderedDict, namedtuple

try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters  # noqa: F401

import mock
import pytest

try:
    import zoneinfo
except ImportError:
    zoneinfo = None
try:
    import pytz
except ImportError:
    pytz = None
try:
    import numpy as np
    import pandas as pd
except ImportError:
    pd = None
    np = None
try:
    import pyarrow as pa
except (ImportError, AttributeError):
    pa = None
    pytestmark = pytest.mark.skip("Need pyarrow to run this test")

from ...config import options
from ...models import TableSchema
from ...tests.core import get_test_unique_name, tn


def _get_tz_str():
    from ...lib import tzlocal

    if options.local_timezone is False:
        return "UTC"
    elif options.local_timezone is True or options.local_timezone is None:
        return tzlocal.get_localzone()
    else:
        return str(options.local_timezone)


@pytest.fixture
def setup(odps, tunnel):
    def upload_data(test_table, data, compress=False, **kw):
        upload_ss = tunnel.create_upload_session(test_table, **kw)
        writer = upload_ss.open_arrow_writer(0, compress=compress)

        writer.write(data)
        writer.close()
        upload_ss.commit([0])

    def buffered_upload_data(test_table, data, buffer_size=None, compress=False, **kw):
        upload_ss = tunnel.create_upload_session(test_table, **kw)
        writer = upload_ss.open_arrow_writer(compress=compress, buffer_size=buffer_size)

        pd_data = data.to_pandas()
        part1 = pd_data.iloc[: len(pd_data) // 2]
        writer.write(part1)
        part2 = pd_data.iloc[len(pd_data) // 2 :]
        writer.write(part2)
        writer.close()

        if buffer_size is None:
            assert len(writer.get_blocks_written()) == 1
        else:
            assert len(writer.get_blocks_written()) > 1
        upload_ss.commit(writer.get_blocks_written())

    def download_data(
        test_table, columns=None, compress=False, append_partitions=None, **kw
    ):
        count = kw.pop("count", None)
        download_ss = tunnel.create_download_session(test_table, **kw)
        count = count or download_ss.count or 1
        down_kw = (
            {"append_partitions": append_partitions}
            if append_partitions is not None
            else {}
        )
        with download_ss.open_arrow_reader(
            0, count, compress=compress, columns=columns, **down_kw
        ) as reader:
            pd_data = reader.to_pandas()
        for col_name, dtype in pd_data.dtypes.items():
            if isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.datetime64):
                pd_data[col_name] = pd_data[col_name].dt.tz_localize(_get_tz_str())
        return pd_data

    def gen_data(repeat=1):
        data = OrderedDict()
        data["id"] = ["hello \x00\x00 world", "goodbye", "c" * 2, "c" * 20] * repeat
        data["int_num"] = [2**63 - 1, 222222, -(2**63) + 1, -(2**11) + 1] * repeat
        data["float_num"] = [math.pi, math.e, -2.222, 2.222] * repeat
        data["bool"] = [True, False, True, True] * repeat
        data["date"] = [
            datetime.date.today() + datetime.timedelta(days=idx) for idx in range(4)
        ] * repeat
        data["dt"] = [
            datetime.datetime.now() + datetime.timedelta(days=idx) for idx in range(4)
        ] * repeat
        data["dt"] = [
            dt.replace(microsecond=dt.microsecond // 1000 * 1000) for dt in data["dt"]
        ]
        pd_data = pd.DataFrame(data)

        return pa.RecordBatch.from_pandas(pd_data)

    def create_table(table_name):
        fields = ["id", "int_num", "float_num", "bool", "date", "dt"]
        types = ["string", "bigint", "double", "boolean", "date", "datetime"]

        odps.delete_table(table_name, if_exists=True)
        return odps.create_table(
            table_name, TableSchema.from_lists(fields, types), lifecycle=1
        )

    def create_partitioned_table(table_name):
        fields = ["id", "int_num", "float_num", "bool", "date", "dt"]
        types = ["string", "bigint", "double", "boolean", "date", "datetime"]

        odps.delete_table(table_name, if_exists=True)
        return odps.create_table(
            table_name, TableSchema.from_lists(fields, types, ["ds"], ["string"])
        )

    def delete_table(table_name):
        odps.delete_table(table_name)

    nt = namedtuple(
        "NT",
        "upload_data, buffered_upload_data download_data, gen_data, "
        "create_table, create_partitioned_table, delete_table",
    )
    raw_chunk_size = options.chunk_size
    raw_buffer_size = options.tunnel.block_buffer_size
    try:
        options.sql.use_odps2_extension = True
        yield nt(
            upload_data,
            buffered_upload_data,
            download_data,
            gen_data,
            create_table,
            create_partitioned_table,
            delete_table,
        )
    finally:
        options.sql.use_odps2_extension = None
        options.chunk_size = raw_chunk_size
        options.tunnel.block_buffer_size = raw_buffer_size


def _assert_frame_equal(left, right):
    if isinstance(left, pa.RecordBatch):
        left = left.to_pandas()
    if "dt" in left and not getattr(left["dt"].dtype, "tz", None):
        left["dt"] = left["dt"].dt.tz_localize(_get_tz_str())
    if isinstance(right, pa.RecordBatch):
        right = right.to_pandas()
    if "dt" in right and not getattr(right["dt"].dtype, "tz", None):
        right["dt"] = right["dt"].dt.tz_localize(_get_tz_str())
    pd.testing.assert_frame_equal(left, right)


def test_upload_and_download_by_raw_tunnel(odps, setup):
    test_table_name = tn("pyodps_test_arrow_tunnel")
    setup.create_table(test_table_name)
    pd_df = setup.download_data(test_table_name)
    assert len(pd_df) == 0

    data = setup.gen_data(repeat=1024)
    setup.upload_data(test_table_name, data)

    pd_df = setup.download_data(test_table_name)
    _assert_frame_equal(data, pd_df)

    data_dict = OrderedDict(zip(data.schema.names, data.columns))
    data_dict["float_num"] = data_dict["float_num"].cast("float32")
    new_data = pa.RecordBatch.from_arrays(
        list(data_dict.values()), names=list(data_dict.keys())
    )
    setup.upload_data(test_table_name, new_data)

    data_size = new_data.num_rows
    new_data = pa.Table.from_batches(
        [new_data.slice(length=data_size // 2), new_data.slice(offset=data_size // 2)]
    )
    setup.upload_data(test_table_name, new_data)

    data_dict["int_num"] = data.columns[data.schema.get_field_index("id")]
    new_data = pa.RecordBatch.from_arrays(
        list(data_dict.values()), names=list(data_dict.keys())
    )
    with pytest.raises(ValueError) as err_info:
        setup.upload_data(test_table_name, new_data)
    assert "Failed to cast" in str(err_info.value)

    data_dict.pop("int_num")
    new_data = pa.RecordBatch.from_arrays(
        list(data_dict.values()), names=list(data_dict.keys())
    )
    with pytest.raises(ValueError) as err_info:
        setup.upload_data(test_table_name, new_data)
    assert "not contain" in str(err_info.value)


def test_buffered_upload_and_download_by_raw_tunnel(odps, setup):
    from ..tabletunnel import TableUploadSession

    test_table_name = tn("pyodps_test_buffered_arrow_tunnel")
    table = setup.create_table(test_table_name)
    pd_df = setup.download_data(test_table_name)
    assert len(pd_df) == 0

    # test upload and download without errors
    data = setup.gen_data(1024)
    setup.buffered_upload_data(test_table_name, data, buffer_size=4096)

    pd_df = setup.download_data(test_table_name)
    _assert_frame_equal(data, pd_df)

    # test upload and download with retry
    table.truncate()
    raw_iter_data_in_batches = TableUploadSession._iter_data_in_batches
    raises = [True]

    def _gen_with_error(cls, data):
        gen = raw_iter_data_in_batches(data)
        yield next(gen)
        if raises[0]:
            raises[0] = False
            raise ValueError
        for chunk in gen:
            yield chunk

    with mock.patch(
        "odps.tunnel.tabletunnel.TableUploadSession._iter_data_in_batches",
        new=_gen_with_error,
    ):
        setup.buffered_upload_data(test_table_name, data)
        assert not raises[0], "error not raised"

    pd_df = setup.download_data(test_table_name)
    _assert_frame_equal(data, pd_df)


def test_download_with_retry(odps, setup, tunnel):
    from ..tabletunnel import TableDownloadSession

    test_table_name = tn("pyodps_test_buffered_arrow_tunnel_retry")
    setup.create_table(test_table_name)

    data = setup.gen_data()
    setup.buffered_upload_data(test_table_name, data)
    setup.buffered_upload_data(test_table_name, data)

    ranges = []
    original = TableDownloadSession._build_input_stream

    def new_build_input_stream(self, start, count, *args, **kw):
        ranges.append((start, count))
        assert start in (0, 4)
        assert start == 0 or count == session.count - 4
        return original(self, start, count, *args, **kw)

    with mock.patch(
        "odps.tunnel.tabletunnel.TableDownloadSession._build_input_stream",
        new=new_build_input_stream,
    ):
        session = tunnel.create_download_session(test_table_name)
        reader = session.open_arrow_reader(0, session.count)

        reader._inject_error(4, ValueError)
        pd_df = reader.to_pandas()

    assert ranges == [(0, 8), (4, 4)]
    pd_data = data.to_pandas()
    _assert_frame_equal(pd.concat([pd_data, pd_data], ignore_index=True), pd_df)


def test_download_with_specified_columns(odps, setup):
    test_table_name = tn("pyodps_test_arrow_tunnel_columns")
    setup.create_table(test_table_name)

    data = setup.gen_data()
    setup.upload_data(test_table_name, data)

    result = setup.download_data(
        test_table_name, columns=["id"], append_partitions=True
    )
    _assert_frame_equal(data.to_pandas()[["id"]], result)
    setup.delete_table(test_table_name)


def test_partition_upload_and_download_by_raw_tunnel(odps, setup):
    test_table_name = tn("pyodps_test_arrow_partition_tunnel")
    test_table_partition = "ds=test"
    odps.delete_table(test_table_name, if_exists=True)

    table = setup.create_partitioned_table(test_table_name)
    table.create_partition(test_table_partition)
    data = setup.gen_data()

    setup.upload_data(test_table_name, data, partition_spec=test_table_partition)
    result = setup.download_data(
        test_table_name, partition_spec=test_table_partition, append_partitions=True
    )
    pd_data = data.to_pandas()
    pd_data["ds"] = "test"
    _assert_frame_equal(pd_data, result)

    result = setup.download_data(test_table_name, partition_spec=test_table_partition)
    _assert_frame_equal(data, result)


def test_partition_download_with_specified_columns(odps, setup):
    test_table_name = tn("pyodps_test_arrow_tunnel_partition_columns")
    test_table_partition = "ds=test"
    odps.delete_table(test_table_name, if_exists=True)

    table = setup.create_partitioned_table(test_table_name)
    table.create_partition(test_table_partition)
    data = setup.gen_data()

    setup.upload_data(test_table_name, data, partition_spec=test_table_partition)
    result = setup.download_data(
        test_table_name, partition_spec=test_table_partition, columns=["int_num"]
    )
    _assert_frame_equal(data.to_pandas()[["int_num"]], result)

    result = setup.download_data(
        test_table_name, partition_spec=test_table_partition, columns=["int_num", "ds"]
    )
    pd_data = data.to_pandas()[["int_num"]]
    pd_data["ds"] = "test"
    _assert_frame_equal(pd_data, result)


@pytest.mark.parametrize(
    "compress_algo, module", [("zlib", None), ("lz4", "lz4.frame")]
)
def test_upload_and_download_with_compress(odps, setup, compress_algo, module):
    options.chunk_size = 16
    if module:
        pytest.importorskip(module)

    test_table_name = tn("pyodps_test_arrow_zlib_tunnel_" + get_test_unique_name(5))
    odps.delete_table(test_table_name, if_exists=True)

    setup.create_table(test_table_name)
    data = setup.gen_data()

    setup.upload_data(test_table_name, data, compress=True, compress_algo=compress_algo)
    result = setup.download_data(test_table_name, compress=True)
    _assert_frame_equal(data, result)

    setup.delete_table(test_table_name)


@pytest.mark.skipif(pytz is None and zoneinfo is None, reason="pytz not installed")
@pytest.mark.parametrize("zone", [False, True, "Asia/Shanghai", "America/Los_Angeles"])
def test_buffered_upload_and_download_with_timezone(odps, setup, zone):
    test_table_name = tn("pyodps_test_arrow_tunnel_with_tz_" + get_test_unique_name(5))
    odps.delete_table(test_table_name, if_exists=True)
    try:
        options.local_timezone = zone

        setup.create_table(test_table_name)
        data = setup.gen_data()

        setup.buffered_upload_data(test_table_name, data)
        result = setup.download_data(test_table_name, compress=True)

        _assert_frame_equal(data, result)
    finally:
        odps.delete_table(test_table_name, if_exists=True)
        options.local_timezone = None
