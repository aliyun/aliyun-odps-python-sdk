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
import time

import pytest

from odps.apis.storage_api.conftest import storage_api_client  # noqa: F401

if sys.version_info[0] == 3:
    from odps.apis.storage_api import *
else:
    pytestmark = pytest.mark.skip("Need python3.5+ to run this test")

logger = logging.getLogger(__name__)

lock = threading.Lock()
global_total_record = 0
thread_num = 1
compress_type = None


def read_rows(storage_api_client, compress_type, read_session_id, split_count):
    global global_total_record

    if compress_type == 0:
        compression = Compression.UNCOMPRESSED
    elif compress_type == 1:
        compression = Compression.ZSTD
    else:
        compression = Compression.LZ4_FRAME

    req = ReadRowsRequest(session_id=read_session_id, compression=compression)

    total_line = 0
    for i in range(0, split_count):
        req.split_index = i
        start = time.time()
        reader = storage_api_client.read_rows_arrow(req)

        while True:
            record_batch = reader.read()
            if record_batch is None:
                break

            total_line += record_batch.num_rows
            with lock:
                global_total_record += record_batch.num_rows

        if reader.get_status() != Status.OK:
            logger.info("Read rows failed")
            return False
        end = time.time()
        logger.info("Read rows cost (index " + str(i) + "): " + str(end - start) + "s")


def read_performance(storage_api_client):
    read_session_id, split_count = create_read_session(storage_api_client)
    assert read_session_id is not None
    assert get_read_session(storage_api_client, read_session_id) is True

    read_rows(storage_api_client, compress_type, read_session_id, split_count)


def test_read_thread(storage_api_client):
    global table

    table = storage_api_client.table

    if table == "None":
        logger.info("No table name specified")
        raise ValueError("Please input table name")

    global global_total_record
    read_performance_threads = []
    for i in range(0, thread_num):
        read_performance_thread = threading.Thread(
            target=read_performance,
            args=[storage_api_client],
        )
        read_performance_threads.append(read_performance_thread)

    start = time.time()
    for i in range(0, thread_num):
        read_performance_threads[i].start()

    count = 0
    start_count = 0
    now_count = 0
    cal_total_count = 0
    cal_count = 0
    judge = False
    while count < 20:
        time.sleep(1)
        now = time.time()
        now_count = global_total_record
        logger.info(
            "index: %d, read, %f records per second"
            % (count, (now_count - start_count) / (now - start))
        )

        if judge and cal_count < 5:
            cal_total_count += (now_count - start_count) / (now - start)
            cal_count += 1

        if now_count - start_count > 0:
            judge = True

        start_count = now_count
        start = now
        count += 1

    if cal_count == 5:
        logger.info("average count: %f" % (cal_total_count / 5.0))
    else:
        logger.info("less than 5 valid result generated.")

    for i in range(0, thread_num):
        read_performance_threads[i].join()


def create_read_session(storage_api_client):
    req = TableBatchScanRequest()

    req.required_partitions = ["pt=test_write_1"]

    try:
        resp = storage_api_client.create_read_session(req)
    except Exception as e:
        logger.info(e)
        return None, None

    if resp.status != Status.OK and resp.status != Status.WAIT:
        logger.info("create read session failed")
        return None, None
    return resp.session_id, resp.split_count


def get_read_session(storage_api_client, read_session_id):
    req = SessionRequest(session_id=read_session_id)

    while True:
        try:
            resp = storage_api_client.get_read_session(req)
        except Exception as e:
            logger.info(e)
            return False

        if resp.status != Status.OK:
            logger.info("get read session failed")
            return False

        if resp.session_status == SessionStatus.INIT:
            logger.info("Wait...")
            time.sleep(1)
            continue

        break
    return True
