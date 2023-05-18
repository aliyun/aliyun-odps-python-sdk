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

import logging
import sys
import threading

import pytest
try:
    import pyarrow as pa
except ImportError:
    pa = None

if sys.version_info[0] >= 3 and pa is not None:
    from ..storage_api import *
    from .util import *
else:
    pytestmark = pytest.mark.skip("Need python3.5+ to run this test")

logger = logging.getLogger(__name__)


def test_split_limit(storage_api_client):
    req = TableBatchScanRequest()
    req.split_options.split_number = 1

    resp = None
    try:
        resp = storage_api_client.create_read_session(req)
    except Exception as e:
        logger.info(e)

    assert resp is None

    req.split_options = SplitOptions.get_default_options(SplitOptions.SplitMode.ROW_OFFSET)
    resp = storage_api_client.create_read_session(req)
    if resp.status != Status.OK and resp.status != Status.WAIT:
        logger.info("Create read session by row offset split option failed")


def test_create_read_session_neg1(storage_api_client):
    req = TableBatchScanRequest()
    req.split_options = SplitOptions.get_default_options(SplitOptions.SplitMode.BUCKET)

    resp = None
    try:
        resp = storage_api_client.create_read_session(req)
    except Exception as e:
        logger.info(e)

    assert resp is None


def test_write_rows_with_partition_by_diff_schema_neg1(storage_api_client):
    item = TestDataItem()
    item.load_conf("with_partition")

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True

    req = WriteRowsRequest(session_id=item.write_session_id)

    if item.test_data_format:
        req.data_format = item.data_format

    bigint_list = list(range(item.batch_size))
    string_list = ["test_write_1"] * item.batch_size

    record_batch = pa.RecordBatch.from_arrays([pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(string_list)], names=["a", "b", "c", "d"])
    writer = storage_api_client.write_rows_arrow(req)

    write_rows_exception = None
    try:
        for i in range(0, item.batch_count):
            suc = writer.write(record_batch)
            if not suc:
                break

        _, suc = writer.finish()
        assert not suc
    except Exception as e:
        write_rows_exception = e

    assert write_rows_exception is not None
    if write_rows_exception:
        logger.info("RecordBatch's schema is not right")
        logger.info(write_rows_exception)

    record_batch = pa.RecordBatch.from_arrays([pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list)], names=["a", "b", "c"])
    writer = storage_api_client.write_rows_arrow(req)

    write_rows_exception = None
    try:
        for i in range(0, item.batch_count):
            suc = writer.write(record_batch)
            if not suc:
                break

        _, suc = writer.finish()
        assert not suc
    except Exception as e:
        write_rows_exception = e

    assert write_rows_exception is not None
    if write_rows_exception:
        logger.info("RecordBatch's column number doesn't match")
        logger.info(write_rows_exception)

    record_batch = pa.RecordBatch.from_arrays(
        [pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list)],
        names=["a", "b", "c", "d", "e"]
    )
    writer = storage_api_client.write_rows_arrow(req)

    write_rows_exception = None
    try:
        for i in range(0, item.batch_count):
            suc = writer.write(record_batch)
            if not suc:
                break

        _, suc = writer.finish()
        assert not suc
    except Exception as e:
        write_rows_exception = e

    assert write_rows_exception is not None
    if write_rows_exception:
        logger.info("Write rows with duplicated partition failed")
        logger.info(write_rows_exception)

    record_batch = pa.RecordBatch.from_arrays(
        [pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(string_list)],
        names=["a", "b", "c", "d", "e"]
    )
    writer = storage_api_client.write_rows_arrow(req)

    write_rows_exception = None
    try:
        for i in range(0, item.batch_count):
            suc = writer.write(record_batch)
            if not suc:
                break

        _, suc = writer.finish()
        assert not suc
    except Exception as e:
        write_rows_exception = e

    assert write_rows_exception is not None
    if write_rows_exception:
        logger.info("Write rows with duplicated partition failed")
        logger.info(write_rows_exception)


def test_write_rows_without_partition_by_diff_schema_neg1(storage_api_client):
    item = TestDataItem()
    item.load_conf("without_partition")

    req = TableBatchWriteRequest()

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True

    req = WriteRowsRequest(session_id=item.write_session_id)

    if item.test_data_format:
        req.data_format = item.data_format

    bigint_list = list(range(item.batch_size))
    string_list = ["test_write_1"] * item.batch_size

    record_batch = pa.RecordBatch.from_arrays(
        [pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list)],
        names=["a", "b", "c", "d"],
    )
    writer = storage_api_client.write_rows_arrow(req)

    write_rows_exception = None
    try:
        for i in range(0, item.batch_count):
            suc = writer.write(record_batch)
            if not suc:
                break

        _, suc = writer.finish()
        assert not suc
    except Exception as e:
        write_rows_exception = e

    assert write_rows_exception is not None
    if write_rows_exception:
        logger.info("Write rows failed without partition specified")
        logger.info(write_rows_exception)

    record_batch = generate_base_table(item)
    start = time.time()
    writer = storage_api_client.write_rows_arrow(req)

    for i in range(0, item.batch_count):
        suc = writer.write(record_batch)
        if not suc:
            break

    _, suc = writer.finish()

    assert suc is True
    if not suc:
        logger.info("Write rows failed with generate record batch from base table")
        return
    else:
        end = time.time()
        logger.info("Write rows cost: " + str(end - start) + "s")

    record_batch = pa.RecordBatch.from_arrays([pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(string_list)],
                                                names=["a", "b", "c", "d", "e"])
    start = time.time()
    writer = storage_api_client.write_rows_arrow(req)

    for i in range(0, item.batch_count):
        suc = writer.write(record_batch)
        if not suc:
            break

    _, suc = writer.finish()

    assert suc is True
    if not suc:
        logger.info("Write rows with string partition should be success")
        return
    else:
        end = time.time()
        logger.info("Write rows cost: " + str(end - start) + "s")

    unique_bigint_list = [10] * item.batch_size

    record_batch = pa.RecordBatch.from_arrays([pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(unique_bigint_list)],
                                                names=["a", "b", "c", "d", "e"])
    start = time.time()
    writer = storage_api_client.write_rows_arrow(req)

    for i in range(0, item.batch_count):
        suc = writer.write(record_batch)
        if not suc:
            break

    _, suc = writer.finish()

    assert suc is True
    if not suc:
        logger.info("Write rows with bigint partition failed")
        return
    else:
        end = time.time()
        logger.info("Write rows cost: " + str(end - start) + "s")


def test_write_read_with_zstd_compression(storage_api_client):
    item = TestDataItem()
    item.load_conf("with_partition")

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True
    assert write_rows(item, storage_api_client, Compression.ZSTD) is True
    assert commit_write_session(item, storage_api_client) is True

    assert create_read_session(item, storage_api_client) is True
    assert get_read_session(item, storage_api_client) is True
    assert read_rows(item, storage_api_client, Compression.ZSTD) is True


def test_write_read_with_lz4_frame_compression(storage_api_client):
    item = TestDataItem()
    item.load_conf("with_partition")

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True
    assert write_rows(item, storage_api_client, Compression.LZ4_FRAME) is True
    assert commit_write_session(item, storage_api_client) is True

    assert create_read_session(item, storage_api_client) is True
    assert get_read_session(item, storage_api_client) is True
    assert read_rows(item, storage_api_client, Compression.LZ4_FRAME) is True


def test_write_with_invalid_data_format(storage_api_client):
    item = TestDataItem()
    item.load_conf("with_partition")

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True

    item.test_data_format = True
    item.data_format.type = "arrow"
    item.data_format.version = "v6"
    assert not write_rows(item, storage_api_client)


def write_read(storage_api_client):
    item = TestDataItem()
    item.load_conf("with_partition")

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True
    assert write_rows(item, storage_api_client) is True
    assert commit_write_session(item, storage_api_client) is True

    assert create_read_session(item, storage_api_client) is True
    assert get_read_session(item, storage_api_client) is True
    assert read_rows(item, storage_api_client) is True


def write_read_thread(num, storage_api_client):
    threads = []
    for i in range(0, num):
        write_read_thread = threading.Thread(
            target=write_read, args=[storage_api_client]
        )
        threads.append(write_read_thread)

    for i in range(0, num):
        threads[i].start()

    for i in range(0, num):
        threads[i].join()


def test_write_read_thread(storage_api_client):
    write_read_thread(1, storage_api_client)


def test_read_with_invalid_data_format(storage_api_client):
    item = TestDataItem()
    item.load_conf("with_partition")

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True
    assert write_rows(item, storage_api_client) is True
    assert commit_write_session(item, storage_api_client) is True

    assert create_read_session(item, storage_api_client) is True
    assert get_read_session(item, storage_api_client) is True

    item.test_data_format = True
    item.data_format.type = ""
    item.data_format.version = "v6"
    assert not read_rows(item, storage_api_client)

    item.data_format.type = "arrow1"
    item.data_format.version = ""
    assert not read_rows(item, storage_api_client)


def test_split_row(storage_api_client):
    item = TestDataItem()
    item.load_conf("with_partition")

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True
    assert write_rows(item, storage_api_client) is True
    assert commit_write_session(item, storage_api_client) is True

    item.split_options = SplitOptions.get_default_options(SplitOptions.SplitMode.ROW_OFFSET)

    assert create_read_session(item, storage_api_client) is True
    assert get_read_session(item, storage_api_client) is True
    assert read_rows(item, storage_api_client) is True


def test_get_info_before_read_rows_finish(storage_api_client):
    item = TestDataItem()
    item.load_conf("with_partition")

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True
    assert write_rows(item, storage_api_client) is True
    assert commit_write_session(item, storage_api_client) is True

    assert create_read_session(item, storage_api_client) is True
    assert get_read_session(item, storage_api_client) is True

    req = ReadRowsRequest(session_id=item.read_session_id, max_batch_rows=4096)

    if item.test_data_format:
        req.data_format = item.data_format

    req.split_index = 0
    reader = storage_api_client.read_rows_arrow(req)

    record_batch = reader.read()

    assert record_batch is not None
    assert reader.get_status() == Status.RUNNING
    assert reader.get_request_id() is None


def test_get_info_before_write_rows_finish(storage_api_client):
    item = TestDataItem()
    item.load_conf("with_partition")

    assert create_write_session(item, storage_api_client) is True
    assert get_write_session(item, storage_api_client) is True

    req = WriteRowsRequest(session_id=item.write_session_id)

    if item.test_data_format:
        req.data_format = item.data_format

    record_batch = generate_base_table(item)

    writer = storage_api_client.write_rows_arrow(req)

    suc = writer.write(record_batch)

    assert suc is True
    assert writer.get_status() == Status.RUNNING
    assert writer.get_request_id() is None

    _, _ = writer.finish()
