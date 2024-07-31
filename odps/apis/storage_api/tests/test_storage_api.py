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

import sys
import time
import logging

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


def test_storage_api(storage_api_client):
    req = TableBatchWriteRequest(partition_spec="pt=test_write_1")

    resp = storage_api_client.create_write_session(req)

    assert resp.status == Status.OK
    if resp.status != Status.OK:
        raise IOError("Create write session failed")

    req = SessionRequest(session_id=resp.session_id)

    while True:
        resp = storage_api_client.get_write_session(req)

        assert resp.status == Status.OK

        if resp.status != Status.OK:
            raise IOError("Get write session failed")
            return

        if resp.session_status != SessionStatus.NORMAL and resp.session_status != SessionStatus.COMMITTED:
            logger.info("Wait...")
            time.sleep(1)
            continue

        break

    req = WriteRowsRequest(session_id=resp.session_id)

    bigint_list = list(range(4096))

    record_batch = pa.RecordBatch.from_arrays(
        [pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list), pa.array(bigint_list)],
        names=["a", "b", "c", "d"]
    )
    try:
        writer = storage_api_client.write_rows_stream(req)
    except Exception as e:
        logger.info(e)
        return

    start = time.time()
    for i in range(0, 300):
        if i == 0:
            suc = writer.write(record_batch.schema.serialize().to_pybytes())
            if not suc:
                raise IOError("write arrow schema failed")

        suc = writer.write(record_batch.serialize().to_pybytes())
        if not suc:
            raise IOError("write arrow record batch failed")

    # write EOS given https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format
    suc = writer.write(b'\xff\xff\xff\xff\x00\x00\x00\x00')
    if not suc:
        raise IOError("write EOS failed")
    commit_message, suc = writer.finish()

    assert suc is True
    if not suc:
        raise IOError("Write rows failed")
    else:
        end = time.time()
        logger.info("Write rows cost: " + str(end - start) + "s")

    req = SessionRequest(session_id=resp.session_id)

    commit_messages = []
    commit_messages.append(commit_message)
    resp = storage_api_client.commit_write_session(req, commit_messages)

    if resp.status != Status.OK and resp.status != Status.WAIT:
        raise IOError("Fail to commit write session")

    if resp.status == Status.WAIT:
        req = SessionRequest(session_id=resp.session_id)
        while True:
            resp = storage_api_client.get_write_session(req)

            assert resp.status == Status.OK

            if resp.status != Status.OK:
                raise IOError("Get write session failed")

            if resp.session_status != SessionStatus.NORMAL and resp.session_status != SessionStatus.COMMITTED:
                logger.info("Wait...")
                time.sleep(1)
                continue

            break

    req = TableBatchScanRequest()

    req.required_partitions = ["pt=test_write_1"]

    resp = storage_api_client.create_read_session(req)

    if resp.status != Status.OK and resp.status != Status.WAIT:
        raise IOError("create read session failed")

    req = SessionRequest(session_id=resp.session_id)

    while True:
        resp = storage_api_client.get_read_session(req)

        if resp.status != Status.OK:
            raise IOError("get read session failed")

        if resp.session_status == SessionStatus.INIT:
            logger.info("Wait...")
            time.sleep(1)
            continue

        split_count = resp.split_count
        break

    req = ReadRowsRequest(session_id=resp.session_id, max_batch_rows=4096)

    read_size = 65536
    buf = b''
    for i in range(0, split_count):
        req.split_index = i
        start = time.time()
        reader = storage_api_client.read_rows_stream(req)

        while True:
            data = reader.read(read_size)
            if len(data) == 0:
                break
            buf += data

        reader.close()
        if reader.get_status() != Status.OK:
            raise IOError("Read rows failed")

        end = time.time()
        logger.info("Read rows cost (index " + str(i) + "): " + str(end - start) + "s")

    with pa.ipc.open_stream(buf) as reader:
        schema = reader.schema
        batches = [b for b in reader]
    logger.info(schema)
    logger.info(batches[0])
