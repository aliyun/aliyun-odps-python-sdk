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

import random

import pytest

from ...tests.core import tn
from ...config import options
from .. import TableTunnel

try:
    from ..pdio import TunnelPandasReader, TunnelPandasWriter
except ImportError:
    TunnelPandasReader, TunnelPandasWriter = None, None

try:
    import numpy as np
    from numpy.testing import assert_array_equal
    import pandas as pd
except ImportError:
    pd = None


@pytest.fixture(autouse=True)
def wrap_options():
    old_pd_mem_cache_size = options.tunnel.pd_mem_cache_size
    try:
        yield
    finally:
        options.tunnel.pd_mem_cache_size = old_pd_mem_cache_size


@pytest.mark.skipif(not TunnelPandasReader, reason='Accelerated pandas IO not available')
def test_read_into(odps):
    options.tunnel.pd_mem_cache_size = 200

    test_table_name = tn('test_pdio_read_into')
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, 'col1 bigint, col2 double, col3 boolean')

    data = [table.new_record([random.randint(0, 1048576), random.random(), random.random() > 0.5])
            for _ in range(10000)]
    odps.write_table(test_table_name, data)

    tunnel = TableTunnel(odps)
    download_session = tunnel.create_download_session(table.name)
    reader = download_session.open_pandas_reader(0, download_session.count)

    read_buffer = [np.empty(5000, dtype=np.int64), np.empty(5000, dtype=np.float64),
                   np.empty(5000, dtype=np.bool_)]
    count = reader.readinto(read_buffer)

    assert count == 5000
    assert_array_equal(read_buffer[0], np.array([r[0] for r in data[:5000]], dtype=np.int64))
    assert_array_equal(read_buffer[1], np.array([r[1] for r in data[:5000]], dtype=np.float64))
    assert_array_equal(read_buffer[2], np.array([r[2] for r in data[:5000]], dtype=np.bool_))

    count = reader.readinto(read_buffer)

    assert count == 5000
    assert_array_equal(read_buffer[0], np.array([r[0] for r in data[5000:]], dtype=np.int64))
    assert_array_equal(read_buffer[1], np.array([r[1] for r in data[5000:]], dtype=np.float64))
    assert_array_equal(read_buffer[2], np.array([r[2] for r in data[5000:]], dtype=np.bool_))

    assert reader.readinto(read_buffer) == 0

    tunnel = TableTunnel(odps)
    download_session = tunnel.create_download_session(table.name)
    reader = download_session.open_pandas_reader(0, download_session.count, columns=['col2', 'col3', 'col1'])

    read_buffer = [np.empty(10000, dtype=np.float64), np.empty(10000, dtype=np.bool_),
                   np.empty(10000, dtype=np.int64)]
    count = reader.readinto(read_buffer)

    assert count == 10000
    assert_array_equal(read_buffer[0], np.array([r[1] for r in data], dtype=np.float64))
    assert_array_equal(read_buffer[1], np.array([r[2] for r in data], dtype=np.bool_))
    assert_array_equal(read_buffer[2], np.array([r[0] for r in data], dtype=np.int64))

    tunnel = TableTunnel(odps)
    download_session = tunnel.create_download_session(table.name)
    reader = download_session.open_pandas_reader(0, download_session.count, columns=['col2', 'col3', 'col1'])

    read_buffer = [np.empty(10000, dtype=np.int64), np.empty(10000, dtype=np.float64),
                   np.empty(10000, dtype=np.bool_)]
    count = reader.readinto(read_buffer, columns=['col1', 'col2', 'col3'])

    assert count == 10000
    assert_array_equal(read_buffer[0], np.array([r[0] for r in data], dtype=np.int64))
    assert_array_equal(read_buffer[1], np.array([r[1] for r in data], dtype=np.float64))
    assert_array_equal(read_buffer[2], np.array([r[2] for r in data], dtype=np.bool_))

    try:
        import pandas as pd
        tunnel = TableTunnel(odps)
        download_session = tunnel.create_download_session(table.name)
        reader = download_session.open_pandas_reader(0, download_session.count)

        read_buffer = pd.DataFrame(dict(col1=np.empty(10000, dtype=np.int64),
                                        col2=np.empty(10000, dtype=np.float64),
                                        col3=np.empty(10000, dtype=np.bool_)), columns='col1 col2 col3'.split())

        count = reader.readinto(read_buffer)
        assert count == 10000

        assert_array_equal(read_buffer.col1.values, np.array([r[0] for r in data], dtype=np.int64))
        assert_array_equal(read_buffer.col2.values, np.array([r[1] for r in data], dtype=np.float64))
        assert_array_equal(read_buffer.col3.values, np.array([r[2] for r in data], dtype=np.bool_))
    except ImportError:
        pass


@pytest.mark.skipif(not pd, reason='pandas not available')
def test_read(odps):
    options.tunnel.pd_mem_cache_size = 200
    options.tunnel.pd_row_cache_size = 200

    test_table_name = tn('test_pdio_read_into')
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, 'col1 bigint, col2 double, col3 boolean')

    data = [table.new_record([random.randint(0, 1048576), random.random(), random.random() > 0.5])
            for _ in range(10000)]
    odps.write_table(test_table_name, data)

    tunnel = TableTunnel(odps)
    download_session = tunnel.create_download_session(table.name)
    reader = download_session.open_pandas_reader(0, download_session.count)

    result = reader.read()
    assert_array_equal(result.col1.values, np.array([r[0] for r in data], dtype=np.int64))
    assert_array_equal(result.col2.values, np.array([r[1] for r in data], dtype=np.float64))
    assert_array_equal(result.col3.values, np.array([r[2] for r in data], dtype=np.bool_))


@pytest.mark.skipif(not TunnelPandasWriter, reason='Accelerated pandas IO not available')
def test_write_array(odps):
    options.tunnel.pd_mem_cache_size = 200

    test_table_name = tn('test_pdio_write_array')
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, 'col1 bigint, col2 bigint, col3 double')

    data = np.random.rand(100, 300) * 1000

    tunnel = TableTunnel(odps)
    upload_session = tunnel.create_upload_session(table.name)
    writer = upload_session.open_pandas_writer(0)

    writer.write(data)

    writer.close()
    upload_session.commit([0])

    recv_data = np.empty((100, 300), dtype=np.double)
    for rec in odps.read_table(test_table_name):
        recv_data[rec[0], rec[1]] = rec[2]

    assert_array_equal(data, recv_data)

    table.truncate()

    tunnel = TableTunnel(odps)
    upload_session = tunnel.create_upload_session(table.name)
    writer = upload_session.open_pandas_writer(0)

    writer.write(data, dim_offsets=(500, 100))

    writer.close()
    upload_session.commit([0])

    recv_data = np.empty((100, 300), dtype=np.double)
    for rec in odps.read_table(test_table_name):
        recv_data[rec[0] - 500, rec[1] - 100] = rec[2]

    assert_array_equal(data, recv_data)


@pytest.mark.skipif(not TunnelPandasWriter, reason='Accelerated pandas IO not available')
def test_write_arrays(odps):
    def assert_results():
        recv_data = [np.empty((10000, ), dtype=np.int64), np.empty((10000, ), dtype=np.double),
                     np.empty((10000, ), dtype=np.bool_)]

        for idx, rec in enumerate(odps.read_table(test_table_name)):
            recv_data[0][idx] = rec[0]
            recv_data[1][idx] = rec[1]
            recv_data[2][idx] = rec[2]

        assert_array_equal(raw_data[0], recv_data[0])
        assert_array_equal(raw_data[1], recv_data[1])
        assert_array_equal(raw_data[2], recv_data[2])

    options.tunnel.pd_mem_cache_size = 200

    test_table_name = tn('test_pdio_write_arrays')
    odps.delete_table(test_table_name, if_exists=True)
    table = odps.create_table(test_table_name, 'col1 bigint, col2 double, col3 boolean')

    raw_data = [np.random.randint(1048576, size=(10000,)), np.random.rand(10000),
                np.random.rand(10000) > 0.5]
    data = raw_data

    tunnel = TableTunnel(odps)
    upload_session = tunnel.create_upload_session(table.name)
    writer = upload_session.open_pandas_writer(0)

    writer.write(data)

    writer.close()
    upload_session.commit([0])
    assert_results()

    table.truncate()

    data = dict(col1=raw_data[0], col2=raw_data[1], col3=raw_data[2])

    tunnel = TableTunnel(odps)
    upload_session = tunnel.create_upload_session(table.name)
    writer = upload_session.open_pandas_writer(0)

    writer.write(data)

    writer.close()
    upload_session.commit([0])
    assert_results()

    table.truncate()

    try:
        import pandas as pd
        data = pd.DataFrame(dict(col1=raw_data[0], col2=raw_data[1], col3=raw_data[2]),
                            columns='col1 col2 col3'.split())

        tunnel = TableTunnel(odps)
        upload_session = tunnel.create_upload_session(table.name)
        writer = upload_session.open_pandas_writer(0)

        writer.write(data)

        writer.close()
        upload_session.commit([0])
        assert_results()
    except ImportError:
        pass
