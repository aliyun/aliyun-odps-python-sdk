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

import datetime
import math
from collections import namedtuple
try:
    from string import letters
except ImportError:
    from string import ascii_letters as letters  # noqa: F401

import pytest
try:
    import pytz
except ImportError:
    pytz = None
try:
    import pandas as pd
    import numpy as np
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
from ...tests.core import tn


@pytest.fixture
def setup(odps, tunnel):
    def upload_data(test_table, data, compress=False, **kw):
        upload_ss = tunnel.create_upload_session(test_table, **kw)
        writer = upload_ss.open_arrow_writer(0, compress=compress)

        writer.write(data)
        writer.close()
        upload_ss.commit([0, ])

    def download_data(test_table, columns=None, compress=False, **kw):
        from ...lib import tzlocal

        count = kw.pop('count', 4)
        download_ss = tunnel.create_download_session(test_table, **kw)
        with download_ss.open_arrow_reader(0, count, compress=compress, columns=columns) as reader:
            pd_data = reader.to_pandas()
        for col_name, dtype in pd_data.dtypes.items():
            if isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.datetime64):
                pd_data[col_name] = pd_data[col_name].dt.tz_localize(tzlocal.get_localzone())
        return pd_data

    def gen_data(repeat=1):
        from ...lib import tzlocal

        data = dict()
        data['id'] = ['hello \x00\x00 world', 'goodbye', 'c' * 2, 'c' * 20] * repeat
        data['int_num'] = [2 ** 63 - 1, 222222, -2 ** 63 + 1, -2 ** 11 + 1] * repeat
        data['float_num'] = [math.pi, math.e, -2.222, 2.222] * repeat
        data['bool'] = [True, False, True, True] * repeat
        data['date'] = [
            datetime.date.today() + datetime.timedelta(days=idx) for idx in range(4)
        ]
        data['dt'] = [
                         datetime.datetime.now() + datetime.timedelta(days=idx)
                         for idx in range(4)
                     ] * repeat
        data['dt'] = [
            dt.replace(microsecond=dt.microsecond // 1000 * 1000) for dt in data['dt']
        ]
        pd_data = pd.DataFrame(data)
        pd_data["dt"] = pd_data["dt"].dt.tz_localize(tzlocal.get_localzone())

        return pa.RecordBatch.from_pandas(pd_data)

    def create_table(table_name):
        fields = ['id', 'int_num', 'float_num', 'bool', 'date', 'dt']
        types = ['string', 'bigint', 'double', 'boolean', 'date', 'datetime']

        odps.delete_table(table_name, if_exists=True)
        return odps.create_table(
            table_name, TableSchema.from_lists(fields, types), lifecycle=1
        )

    def create_partitioned_table(table_name):
        fields = ['id', 'int_num', 'float_num', 'bool', 'date', 'dt']
        types = ['string', 'bigint', 'double', 'boolean', 'date', 'datetime']

        odps.delete_table(table_name, if_exists=True)
        return odps.create_table(
            table_name, TableSchema.from_lists(fields, types, ['ds'], ['string'])
        )

    def delete_table(table_name):
        odps.delete_table(table_name)

    nt = namedtuple("NT", "upload_data, download_data, gen_data, create_table, create_partitioned_table, delete_table")
    raw_chunk_size = options.chunk_size
    try:
        options.sql.use_odps2_extension = True
        yield nt(upload_data, download_data, gen_data, create_table, create_partitioned_table, delete_table)
    finally:
        options.sql.use_odps2_extension = None
        options.chunk_size = raw_chunk_size


def test_upload_and_download_by_raw_tunnel(odps, setup):
    test_table_name = tn('pyodps_test_arrow_tunnel')
    setup.create_table(test_table_name)
    pd_df = setup.download_data(test_table_name)
    assert len(pd_df) == 0

    data = setup.gen_data()
    setup.upload_data(test_table_name, data)

    pd_df = setup.download_data(test_table_name)
    pd.testing.assert_frame_equal(data.to_pandas(), pd_df)

    data_dict = dict(zip(data.schema.names, data.columns))
    data_dict["float_num"] = data_dict["float_num"].cast("float32")
    new_data = pa.RecordBatch.from_arrays(
        list(data_dict.values()), names=list(data_dict.keys())
    )
    setup.upload_data(test_table_name, new_data)

    new_data = pa.Table.from_batches([new_data])
    setup.upload_data(test_table_name, new_data)

    data_dict['int_num'] = data['id']
    new_data = pa.RecordBatch.from_arrays(
        list(data_dict.values()), names=list(data_dict.keys())
    )
    with pytest.raises(ValueError) as err_info:
        setup.upload_data(test_table_name, new_data)
    assert "Failed to cast" in str(err_info.value)

    data_dict.pop('int_num')
    new_data = pa.RecordBatch.from_arrays(
        list(data_dict.values()), names=list(data_dict.keys())
    )
    with pytest.raises(ValueError) as err_info:
        setup.upload_data(test_table_name, new_data)
    assert "not contain" in str(err_info.value)


def test_download_with_specified_columns(odps, setup):
    test_table_name = tn('pyodps_test_arrow_tunnel_columns')
    setup.create_table(test_table_name)

    data = setup.gen_data()
    setup.upload_data(test_table_name, data)

    records = setup.download_data(test_table_name, columns=['id'])
    pd.testing.assert_frame_equal(data.to_pandas()[['id']], records)
    setup.delete_table(test_table_name)


def test_partition_upload_and_download_by_raw_tunnel(odps, setup):
    test_table_name = tn('pyodps_test_arrow_partition_tunnel')
    test_table_partition = 'ds=test'
    odps.delete_table(test_table_name, if_exists=True)

    table = setup.create_partitioned_table(test_table_name)
    table.create_partition(test_table_partition)
    data = setup.gen_data()

    setup.upload_data(test_table_name, data, partition_spec=test_table_partition)
    records = setup.download_data(test_table_name, partition_spec=test_table_partition)
    pd.testing.assert_frame_equal(data.to_pandas(), records)


def test_partition_download_with_specified_columns(odps, setup):
    test_table_name = tn('pyodps_test_arrow_tunnel_partition_columns')
    test_table_partition = 'ds=test'
    odps.delete_table(test_table_name, if_exists=True)

    table = setup.create_partitioned_table(test_table_name)
    table.create_partition(test_table_partition)
    data = setup.gen_data()

    setup.upload_data(test_table_name, data, partition_spec=test_table_partition)
    records = setup.download_data(test_table_name, partition_spec=test_table_partition,
                                  columns=['int_num'])
    pd.testing.assert_frame_equal(data.to_pandas()[['int_num']], records)


@pytest.mark.parametrize("compress_algo, module", [("zlib", None), ("lz4", "lz4.frame")])
def test_upload_and_download_with_compress(odps, setup, compress_algo, module):
    options.chunk_size = 16
    if module:
        pytest.importorskip(module)

    test_table_name = tn('pyodps_test_arrow_zlib_tunnel')
    odps.delete_table(test_table_name, if_exists=True)

    setup.create_table(test_table_name)
    data = setup.gen_data()

    setup.upload_data(test_table_name, data, compress=True, compress_algo=compress_algo)
    records = setup.download_data(test_table_name, compress=True)
    pd.testing.assert_frame_equal(data.to_pandas(), records)

    setup.delete_table(test_table_name)
